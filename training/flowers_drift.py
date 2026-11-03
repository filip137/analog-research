"""Stateful Gaussian-copula Flowers retention, in declared affine coordinates.

The Gaussian Markov transition implements the reference covariance exactly.
The NIG transform applies to the *total* standardized residual, as in the
reference predictor; its components are not separately measured physical noise.
"""
import hashlib
import math
from functools import lru_cache
import numpy as np
from scipy import stats
import torch


def named_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], 'little') % (2**63-1)


def interp(x, xp, fp):
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp)-1)
    w = ((x-xp[i-1])/(xp[i]-xp[i-1])).clamp(0, 1)
    return fp[i-1] + w*(fp[i]-fp[i-1])


@lru_cache(maxsize=8)
def nig_table(parameters):
    x = np.linspace(-40, 40, 8001)
    u = np.clip(stats.norminvgauss(*parameters).cdf(x), 1e-15, 1-1e-15)
    z, keep = np.unique(stats.norm.ppf(u), return_index=True)
    return z, x[keep]


class FlowersLaw:
    def __init__(self, parameters, device='cpu', dtype=torch.float32):
        self.P = parameters
        self.device, self.dtype = torch.device(device), dtype
        self.scale = 79.2/103.2
        edges = np.asarray(parameters['bin_edges'], float)
        if edges.ndim != 1 or not np.all(np.diff(edges) > 0):
            raise ValueError('Invalid Flowers bins.')
        self.centers_np = (edges[1:]+edges[:-1])/2
        self.centers = self.tensor(self.centers_np)
        self.edges = self.tensor(edges)
        self.coefficient_tables = {}
        for key in ('m_1h', 's_1h', 'c0_sd', 'read_sd'):
            p = np.asarray(parameters[key], float)
            if p.shape != self.centers_np.shape or not np.isfinite(p).all() or (key != 'm_1h' and (p < 0).any()):
                raise ValueError('Invalid Flowers coefficient table.')
            self.coefficient_tables[key] = self.tensor(p*self.scale)
        self.z_np, self.nig_np = nig_table(tuple(parameters['nig']))
        z = self.z_np
        # FP32 rounding may collapse very close upper-tail abscissae.
        self.z, indices = np.unique(np.asarray(z, dtype=np.float64 if dtype == torch.float64 else np.float32), return_index=True)
        self.nig = self.tensor(self.nig_np[indices])
        self.z = self.tensor(self.z)

    def tensor(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def to_source(self, anchor):
        return 51.6+(anchor-9)/self.scale

    def coefficients(self, anchor):
        source = self.to_source(anchor)
        return tuple(interp(source, self.centers, self.coefficient_tables[k])
                     for k in ('m_1h', 's_1h', 'c0_sd', 'read_sd'))

    def phi(self, age):
        return torch.log1p(age/self.P['tc'])/math.log1p(3600/self.P['tc'])

    def mean_factor(self, age):
        if self.P['mean_form'] == 'pow':
            return (age/3600)**self.P['alpha']
        tc = self.P['tau_mean']
        return torch.log1p(age/tc)/math.log1p(3600/tc)

    def transform(self, z):
        return interp(z, self.z, self.nig)

    def expected_clipped(self, anchors, ages, quadrature=4096):
        """Deterministic marginal quadrature, never policy PPL fitting."""
        a = np.asarray(anchors, float)
        source = 51.6+(a-9)/self.scale
        m, s, c, r = [np.interp(source, self.centers_np, self.P[k])*self.scale
                      for k in ('m_1h', 's_1h', 'c0_sd', 'read_sd')]
        z = np.interp(stats.norm.ppf((np.arange(quadrature)+.5)/quadrature), self.z_np, self.nig_np)
        result = np.empty((len(a), len(ages)))
        for j, age in enumerate(ages):
            phi = math.log1p(age/self.P['tc'])/math.log1p(3600/self.P['tc'])
            f = ((age/3600)**self.P['alpha'] if self.P['mean_form'] == 'pow' else
                 math.log1p(age/self.P['tau_mean'])/math.log1p(3600/self.P['tau_mean']))
            sd = np.sqrt(s*s*phi+c*c+r*r)
            for start in range(0, len(a), 256):
                sl = slice(start, start+256)
                values = a[sl, None]+m[sl, None]*f+sd[sl, None]*z
                result[sl, j] = np.clip(values, 9, 88.2).mean(1)
        return result


class FlowersState:
    """One coherent episode per cell; read() returns a held apparent snapshot."""
    def __init__(self, law, anchor, clock, namespace):
        if not torch.isfinite(anchor).all() or (anchor < 9).any() or (anchor > 88.2).any():
            raise ValueError('Programming targets must be finite and feasible.')
        self.law, self.namespace = law, tuple(namespace)
        self.anchor = anchor.clone()
        self.written = torch.full_like(anchor, clock)
        self.clock = float(clock)
        self.d = torch.zeros_like(anchor)
        self.e0 = self.normal('anchor', float(clock))
        self.coefficients = law.coefficients(self.anchor)
        self.cached_clock, self.cached = None, None

    def normal(self, kind, clock):
        generator = torch.Generator(device=self.anchor.device).manual_seed(named_seed(*self.namespace, kind, float(clock)))
        return torch.randn(self.anchor.shape, dtype=self.anchor.dtype, device=self.anchor.device, generator=generator)

    def advance(self, clock):
        clock = float(clock)
        if clock < self.clock:
            raise ValueError('Cannot move a drift trajectory backward.')
        if clock == self.clock:
            return
        old = (self.clock-self.written).clamp_min(0)
        age = clock-self.written
        a = torch.where(old > 0, (old/age)**self.law.P['lam'], torch.zeros_like(age))
        variance = (self.law.phi(age)-a*a*self.law.phi(old)).clamp_min(0)
        self.d = a*self.d+variance.sqrt()*self.normal('drift', clock)
        self.clock = clock
        self.cached_clock, self.cached = None, None

    def rewrite(self, target, mask, clock):
        self.advance(clock)
        if target.shape != self.anchor.shape or mask.shape != target.shape or mask.dtype != torch.bool:
            raise ValueError('Rewrite requires an explicit matching target and boolean mask.')
        if not torch.isfinite(target).all() or (target < 9).any() or (target > 88.2).any():
            raise ValueError('Infeasible rewrite target.')
        self.anchor = torch.where(mask, target, self.anchor)
        self.written = torch.where(mask, float(clock), self.written)
        self.d = torch.where(mask, 0., self.d)
        self.e0 = torch.where(mask, self.normal('anchor', float(clock)), self.e0)
        self.coefficients = self.law.coefficients(self.anchor)
        self.cached_clock, self.cached = None, None

    def read(self, clock, *, gaussian=False):
        self.advance(clock)
        if self.cached_clock == (clock, gaussian):
            return self.cached
        age = clock-self.written
        if (age < 3).any() or (age > 6242).any():
            raise ValueError('Read age outside the supported interval; use G0 directly at zero.')
        m, s, c, r = self.coefficients
        sd = (s*s*self.law.phi(age)+c*c+r*r).sqrt()
        z = (s*self.d-c*self.e0+r*self.normal('read', clock))/sd
        raw = self.anchor+m*self.law.mean_factor(age)+sd*(z if gaussian else self.law.transform(z))
        if not torch.isfinite(raw).all():
            raise FloatingPointError('Nonfinite Flowers observation.')
        self.cached = (raw.clamp(9, 88.2), raw)
        self.cached_clock = (clock, gaussian)
        return self.cached
