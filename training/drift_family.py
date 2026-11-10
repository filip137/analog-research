"""Drift-regimes device family: writable ReRAM-like and PCM-like drift laws.

Mirrored verbatim in the DRN worktree; change both copies together. Times are
seconds and conductances microSiemens. The law and its knobs are specified in
on-chip-learning/campaigns/drift-regimes/drift-family.md.

ReRAM-like (one tensor of physical devices, write episode k of device i):

    G_p   = clamp(target + c_p sigma_CMO(target) xi, program_min, program_max)
    G(tau) = clip(G_p + m u^alpha + s u^beta a_i Y(tau), drift_min, drift_max),  u = tau / 1 h
    (m, s) = (mean_us, spread_us), or the step bin of G_p when coefficient_bins is set
    Y_0    = C_0                              (initial deployment episode)
    Y_k    = sqrt(f) D + sqrt(1 - f) C_k      (k >= 1)
    D      = sqrt(f) C_0 + sqrt(1 - f) I      (persistent path, replayed from tau = 0 by every episode)
    Corr(X(tau1), X(tau2)) = (tau1 / tau2)^kappa for every path X in {C_k, I, D}

This construction has the same joint law as Y_k = sqrt(f) D + sqrt(1-f) C_k for
every k, and pairs the initial deployment across f. Every write is a device
reset: a fresh CMO endpoint, tau restarted and a fresh C_k.

With kappa = 0 the paths are constant within an episode and the arithmetic is
that of ``training.writable_drift.WritableDrift`` (f == rho). With kappa > 0
each path is an exact AR(1) chain on a fixed log-age grid (64 points per decade
from 1 s), keyed by device, episode and grid index, so a value never depends on
which ages were queried. Between grid points the path is interpolated linearly
in log-age with its variance restored to 1. Below 1 s it is held at its grid-0
value.

Controllers see only ``read``; Y, the amplitudes and nu stay hidden except
through the named ``oracle_*`` methods.
"""
from dataclasses import asdict, dataclass
import math

import torch

from training.writable_drift import cycle_normal, stream_seed

# AIHWKit 1.1.0 ReRamCMONoiseModel: sigma(G) = c0 + c1 G (uS), keyed by the
# P&V acceptance range in percent (same table as hwa training/reram_cmo.py).
CMO_SIGMA = {0.2: (0.00081107, 0.00106879), 2.0: (0.0112185391, 0.01129027418)}
# AIHWKit 1.1.0 PCM drift-exponent law (same as hwa training/inference_retention.py).
PCM_T0 = 20.0
GRID_PER_DECADE = 64
GRID_POINTS = 640                    # 1 s .. 1e10 s
_STEP_KEY = 1024                     # > GRID_POINTS: (episode, grid index) -> one counter
CORRELATION_AGES = (300.0, 90000.0)  # rho* = Corr(Y(5 min), Y(25 h))
_INDEPENDENT_SALT = 0x85ebca6b       # WritableDrift's persistent-draw salt


def kappa_from_rho_star(rho_star):
    """kappa such that (5 min / 25 h)^kappa == rho*."""
    return math.log(rho_star) / math.log(CORRELATION_AGES[0] / CORRELATION_AGES[1])


def _finite(values, names):
    for name in names:
        v = values[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f'Expected finite {name}.')


@dataclass(frozen=True)
class ReramFamily:
    """ReRAM-like law. Defaults are the campaign anchor R0.

    ``coefficient_bins`` (optional) replaces the scalar ``mean_us`` and ``spread_us``,
    which must then be None, by step bins of the programmed conductance:
    rows ``(lower_us, mean_us, spread_us)`` with strictly increasing lower edges, the
    first at or below ``program_min``. A device's bin is that of its programmed
    conductance G_p, looked up again at every write. Without it the arithmetic is the
    scalar law's, bit for bit.
    """
    rho_star: float = 0.335
    persistent_share: float = 0.0
    cmo_scale: float = 1.0
    cmo_acceptance: float = 2.0
    amplitude_sigma: float = 0.0
    mean_us: float | None = -2.0
    spread_us: float | None = 3.0
    mean_exponent: float = .23
    spread_exponent: float = .11
    reference_seconds: float = 3600.0
    program_min: float = 9.0
    program_max: float = 88.2
    drift_min: float = 5.0
    drift_max: float = 150.0
    time_dtype: str = 'float64'
    coefficient_bins: tuple = ()

    def __post_init__(self):
        values = asdict(self)
        scalar = ('mean_us', 'spread_us')
        _finite(values, [k for k in values if k not in ('time_dtype', 'coefficient_bins', *scalar)])
        if not isinstance(self.coefficient_bins, (list, tuple)):
            raise ValueError('Expected a sequence of coefficient bins.')
        if self.coefficient_bins:
            if self.mean_us is not None or self.spread_us is not None:
                raise ValueError('coefficient_bins replace mean_us and spread_us; set both to None.')
            rows = []
            for row in self.coefficient_bins:
                if not isinstance(row, (list, tuple)) or len(row) != 3:
                    raise ValueError('Expected coefficient bins (lower_us, mean_us, spread_us).')
                _finite(dict(zip(('lower_us', 'bin mean_us', 'bin spread_us'), row)),
                        ('lower_us', 'bin mean_us', 'bin spread_us'))
                rows.append(tuple(float(v) for v in row))
            lower = [r[0] for r in rows]
            if any(b <= a for a, b in zip(lower, lower[1:])) or lower[0] > self.program_min:
                raise ValueError('Expected strictly increasing bin edges, the first at or below program_min.')
            if any(r[2] < 0 for r in rows):
                raise ValueError('Expected nonnegative bin spreads.')
            object.__setattr__(self, 'coefficient_bins', tuple(rows))
        else:
            object.__setattr__(self, 'coefficient_bins', ())
            _finite(values, scalar)
            if self.spread_us < 0:
                raise ValueError('Expected nonnegative spread_us.')
        if not 0 < self.rho_star <= 1 or not 0 <= self.persistent_share <= 1:
            raise ValueError('Expected rho_star in (0, 1] and persistent_share in [0, 1].')
        if self.cmo_scale < 0 or self.amplitude_sigma < 0:
            raise ValueError('Expected nonnegative cmo_scale and amplitude_sigma.')
        if self.cmo_acceptance not in CMO_SIGMA:
            raise ValueError(f'CMO acceptance must be one of {sorted(CMO_SIGMA)} percent.')
        if min(self.mean_exponent, self.spread_exponent, self.reference_seconds) <= 0:
            raise ValueError('Expected positive exponents and reference time.')
        if not (self.drift_min <= self.program_min < self.program_max <= self.drift_max):
            raise ValueError('The drift window must contain the programming window.')
        if self.time_dtype not in ('float32', 'float64'):
            raise ValueError('time_dtype must be float32 or float64.')

    @property
    def kappa(self):
        return 0.0 if self.rho_star == 1 else kappa_from_rho_star(self.rho_star)

    def cmo_sigma(self, g):
        c0, c1 = CMO_SIGMA[self.cmo_acceptance]
        return c0 + c1 * g

    def _bin_values(self, g, column):
        bins = self.coefficient_bins
        edges = torch.tensor([r[0] for r in bins[1:]], dtype=g.dtype, device=g.device)
        table = torch.tensor([r[column] for r in bins], dtype=torch.float64, device=g.device)
        return table[torch.bucketize(g, edges, right=True)]

    def mean_at(self, g):
        """Mean coefficient (uS at the reference age) for conductances ``g``: the scalar
        ``mean_us`` (a float), or each value's bin mean as a float64 tensor."""
        return self.mean_us if not self.coefficient_bins else self._bin_values(g, 1)

    def spread_at(self, g):
        """Spread coefficient (uS at the reference age) for conductances ``g``; see ``mean_at``."""
        return self.spread_us if not self.coefficient_bins else self._bin_values(g, 2)


@dataclass(frozen=True)
class PcmFamily:
    """PCM-like law: G = G_p ((tau + t0) / t0)^(-nu), nu drawn at every write.

    nu = |mu(G_p / g_reference) + nu_std_scale * sigma(G_p / g_reference) xi|,
    or |nu_mean + ...| when nu_mean is set (PCM-common uses nu_std_scale = 0).
    Programming is exact.
    """
    nu_std_scale: float = 1.0
    nu_mean: float | None = None
    g_reference: float = 88.2
    program_min: float = 9.0
    program_max: float = 88.2
    drift_min: float = 5.0
    drift_max: float = 150.0
    t0_seconds: float = PCM_T0
    time_dtype: str = 'float64'

    def __post_init__(self):
        values = asdict(self)
        _finite(values, ['nu_std_scale', 'g_reference', 'program_min', 'program_max',
                         'drift_min', 'drift_max', 't0_seconds'])
        if self.nu_mean is not None:
            _finite(values, ['nu_mean'])
            if self.nu_mean < 0:
                raise ValueError('Expected nonnegative nu_mean.')
        if self.nu_std_scale < 0 or self.g_reference <= 0 or self.t0_seconds <= 0:
            raise ValueError('Expected nonnegative nu_std_scale and positive g_reference, t0.')
        if not (0 <= self.drift_min <= self.program_min < self.program_max <= self.drift_max):
            raise ValueError('The drift window must contain the programming window.')
        if self.time_dtype not in ('float32', 'float64'):
            raise ValueError('time_dtype must be float32 or float64.')

    def nu_moments(self, g):
        relative = (g / self.g_reference).abs().clamp_min(1e-7)
        mean = (-0.0155 * relative.log() + 0.0244).clamp(0.049, 0.1)
        if self.nu_mean is not None:
            mean = torch.full_like(mean, self.nu_mean)
        std = (-0.0125 * relative.log() - 0.0059).clamp(0.008, 0.045) * self.nu_std_scale
        return mean, std


class _Cells:
    """Shared bookkeeping: one physical tensor, a global clock, counted reads and resets."""

    def _init_common(self, target, law, seed, name, start_seconds):
        if target.dtype != torch.float32 or not bool(torch.isfinite(target).all()):
            raise ValueError('Expected finite FP32 physical targets.')
        if not math.isfinite(start_seconds):
            raise ValueError('Expected a finite start time.')
        self.law, self.name = law, name
        self.seed = stream_seed(seed, name)
        self.shape = target.shape
        self.target = target.detach().clamp(law.program_min, law.program_max).clone()
        self.requested = self.target.clone()
        self.last_write = torch.full(target.shape, float(start_seconds), device=target.device,
                                     dtype=getattr(torch, law.time_dtype))
        self.cycles = torch.zeros(target.shape, device=target.device, dtype=torch.int32)
        self.clock = float(start_seconds)
        self.write_commands = 0
        self.selection_reads = 0
        self.verify_reads = 0
        return torch.arange(target.numel(), device=target.device, dtype=torch.int64)

    def _seconds(self, seconds):
        seconds = self.clock if seconds is None else float(seconds)
        if not math.isfinite(seconds) or seconds < self.clock:
            raise ValueError('A writable trajectory cannot move backwards in time.')
        return seconds

    def _ids(self, indices):
        n = self.target.numel()
        if indices is None:
            return None
        ids = indices.to(device=self.target.device, dtype=torch.long).flatten()
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= n
                            or ids.unique().numel() != ids.numel()):
            raise ValueError('Device indices must be unique and in range.')
        return ids

    def _flat(self, tensor, ids):
        return tensor.reshape(-1) if ids is None else tensor.reshape(-1)[ids]

    def _write_checks(self, indices, values):
        ids = self._ids(indices)
        values = values.detach().to(self.target).flatten()
        if ids.numel() != values.numel() or not bool(torch.isfinite(values).all()):
            raise ValueError('Invalid programming command.')
        return ids, values

    def advance(self, seconds):
        value = self.value(seconds)
        self.clock = float(seconds)
        return value

    def read(self, indices=None):
        """Controller read at the current clock (exact: there is no read noise)."""
        ids = self._ids(indices)
        value = self.value(indices=ids)
        self.selection_reads += value.numel()
        return value

    def bound_statistics(self, values):
        return {'devices': values.numel(),
                'at_drift_min': int((values <= self.law.drift_min).sum()),
                'at_drift_max': int((values >= self.law.drift_max).sum())}

    def _common_state(self, schema):
        return {'schema': schema, 'law': asdict(self.law), 'name': self.name, 'seed': self.seed,
                'clock': self.clock, 'write_commands': self.write_commands,
                'selection_reads': self.selection_reads, 'verify_reads': self.verify_reads}

    @classmethod
    def _restore_common(cls, state, schema, law_type, tensors, device):
        if state.get('schema') != schema:
            raise ValueError('Unknown drift-family state.')
        obj = cls.__new__(cls)
        obj.law = law_type(**state['law'])
        for k in ('name', 'seed', 'clock', 'write_commands', 'selection_reads', 'verify_reads'):
            setattr(obj, k, state[k])
        for k in tensors:
            setattr(obj, k, None if state[k] is None else state[k].to(device).clone())
        obj.shape = obj.target.shape
        if any(getattr(obj, k) is not None and getattr(obj, k).shape != obj.shape for k in tensors):
            raise ValueError('Inconsistent serialized cell shapes.')
        if bool((obj.last_write > obj.clock).any()):
            raise ValueError('Invalid serialized cell clock.')
        return obj


class ReramDriftCells(_Cells):
    """ReRAM-like family on one tensor of physical devices."""

    _TENSORS = ('target', 'requested', 'anchor', 'last_write', 'cycles', 'persistent',
                'amplitude', 'epsilon', 'grid', 'c_lo', 'c_hi', 'd_lo', 'd_hi')

    def __init__(self, target, *, law, seed, name, start_seconds=0.0):
        ids = self._init_common(target, law, seed, name, start_seconds)
        self._init_streams()
        cycles = self.cycles.reshape(-1)
        self.anchor = self._program(ids, self.target.reshape(-1), cycles).reshape(self.shape)
        f = law.persistent_share
        c0 = cycle_normal(ids, cycles, self.seed).reshape(self.shape)
        independent = cycle_normal(ids, cycles, self.seed ^ _INDEPENDENT_SALT).reshape(self.shape)
        # D at tau = 0; with kappa = 0 the whole persistent path (WritableDrift construction).
        self.persistent = math.sqrt(f) * c0 + math.sqrt(1-f) * independent
        if law.amplitude_sigma:
            z = cycle_normal(ids, cycles, self._amplitude_seed).reshape(self.shape)
            s = law.amplitude_sigma
            self.amplitude = torch.exp(s * z - s * s)        # E[a^2] = 1
        else:
            self.amplitude = None
        self.epsilon = self.grid = self.c_lo = self.c_hi = self.d_lo = self.d_hi = None
        if law.kappa == 0:
            self.epsilon = c0
        else:
            self.grid = torch.zeros(self.shape, device=target.device, dtype=torch.int32)
            self.c_lo, self.c_hi, self.d_lo, self.d_hi = (
                v.reshape(self.shape) if v is not None else None
                for v in self._episode_start(ids, cycles, c0.reshape(-1),
                                             self.persistent.reshape(-1)))

    def _init_streams(self):
        self._amplitude_seed = stream_seed(self.seed, 'amplitude')
        self._program_seed = stream_seed(self.seed, 'cmo-endpoint')
        self._c_step_seed = stream_seed(self.seed, 'C-steps')
        self._i_step_seed = stream_seed(self.seed, 'I-steps')
        c = 10.0 ** (-self.law.kappa / GRID_PER_DECADE)
        self._c, self._q = c, math.sqrt(max(0.0, 1 - c * c))

    # -- programming and paths -------------------------------------------------
    def _program(self, ids, values, cycles):
        law = self.law
        v = values.clamp(law.program_min, law.program_max)
        if not law.cmo_scale:
            return v
        xi = cycle_normal(ids, cycles, self._program_seed)
        return (v + law.cmo_scale * law.cmo_sigma(v) * xi).clamp(law.program_min, law.program_max)

    def _xi_c(self, ids, cycles, j):
        return cycle_normal(ids, cycles.to(torch.int64) * _STEP_KEY + j, self._c_step_seed)

    def _xi_d(self, ids, j):
        f = self.law.persistent_share
        zero = torch.zeros_like(ids)
        return (math.sqrt(f) * self._xi_c(ids, zero, j)
                + math.sqrt(1-f) * cycle_normal(ids, j, self._i_step_seed))

    def _episode_start(self, ids, cycles, c0, d0):
        one = torch.ones_like(ids)
        c_hi = self._c * c0 + self._q * self._xi_c(ids, cycles, one)
        if not self.law.persistent_share:
            return c0, c_hi, None, None
        return c0, c_hi, d0.clone(), self._c * d0 + self._q * self._xi_d(ids, one)

    def _step_paths(self, ids, cycles, grid, c_lo, c_hi, d_lo, d_hi, target_grid):
        """Advance (lo, hi) pairs forward to target_grid; deterministic per device."""
        if bool((target_grid >= GRID_POINTS - 1).any()):
            raise ValueError('Age beyond the drift-family grid (1e10 s).')
        grid, c_lo, c_hi = grid.clone(), c_lo.clone(), c_hi.clone()
        if d_lo is not None:
            d_lo, d_hi = d_lo.clone(), d_hi.clone()
        while True:
            need = torch.nonzero(grid < target_grid, as_tuple=False).flatten()
            if not need.numel():
                return grid, c_lo, c_hi, d_lo, d_hi
            if need.numel() == grid.numel():
                need = slice(None)          # common case: every device steps; avoid gathers
            k = ids[need]
            nxt = grid[need].to(torch.int64) + 2
            new_c = self._c * c_hi[need] + self._q * self._xi_c(k, cycles[need], nxt)
            c_lo[need] = c_hi[need]
            c_hi[need] = new_c
            if d_lo is not None:
                new_d = self._c * d_hi[need] + self._q * self._xi_d(k, nxt)
                d_lo[need] = d_hi[need]
                d_hi[need] = new_d
            grid[need] += 1

    @staticmethod
    def _grid_position(age_seconds):
        g = age_seconds.double().clamp_min(1.0).log10() * GRID_PER_DECADE
        j = g.floor()
        return j.to(torch.int32), (g - j)

    def _interpolate(self, lo, hi, w):
        c = self._c
        norm = ((1 - w).square() + w.square() + 2 * c * w * (1 - w)).sqrt()
        return (((1 - w) * lo.double() + w * hi.double()) / norm).float()

    def _path_state(self, seconds, ids):
        """(grid, c_lo, c_hi, d_lo, d_hi, w) for devices ``ids`` at ``seconds``, without mutation."""
        flat = lambda t: None if t is None else self._flat(t, ids)
        age = seconds - flat(self.last_write)
        target_grid, w = self._grid_position(age)
        all_ids = torch.arange(self.target.numel(), device=self.target.device) if ids is None else ids
        grid, c_lo, c_hi, d_lo, d_hi = self._step_paths(
            all_ids, flat(self.cycles), flat(self.grid), flat(self.c_lo), flat(self.c_hi),
            flat(self.d_lo), flat(self.d_hi), target_grid)
        return grid, c_lo, c_hi, d_lo, d_hi, w

    def _y(self, seconds, ids, state=None):
        if self.law.kappa == 0:
            return self._flat(self.epsilon, ids)
        _, c_lo, c_hi, d_lo, d_hi, w = state or self._path_state(seconds, ids)
        y = self._interpolate(c_lo, c_hi, w)
        if d_lo is not None:
            f = self.law.persistent_share
            later = self._flat(self.cycles, ids) > 0
            mixed = math.sqrt(f) * self._interpolate(d_lo, d_hi, w) + math.sqrt(1-f) * y
            y = torch.where(later, mixed, y)
        return y

    # -- public API ------------------------------------------------------------
    def value(self, seconds=None, indices=None):
        """Conductances at ``seconds`` (default: the clock), without moving the clock."""
        seconds = self._seconds(seconds)
        ids = self._ids(indices)
        law = self.law
        age = (seconds - self._flat(self.last_write, ids)) / law.reference_seconds
        anchor = self._flat(self.anchor, ids)
        mean = (law.mean_at(anchor) * age.pow(law.mean_exponent)).to(torch.float32)
        spread = (law.spread_at(anchor) * age.pow(law.spread_exponent)).to(torch.float32)
        if self.amplitude is not None:
            spread = spread * self._flat(self.amplitude, ids)
        raw = anchor + mean + spread * self._y(seconds, ids)
        held = raw.clamp(law.drift_min, law.drift_max)
        return held.reshape(self.shape) if ids is None else held

    def advance(self, seconds):
        seconds = self._seconds(seconds)
        if self.law.kappa != 0:
            grid, c_lo, c_hi, d_lo, d_hi, _ = self._path_state(seconds, None)
            self.grid = grid.reshape(self.shape)
            self.c_lo, self.c_hi = c_lo.reshape(self.shape), c_hi.reshape(self.shape)
            if d_lo is not None:
                self.d_lo, self.d_hi = d_lo.reshape(self.shape), d_hi.reshape(self.shape)
        value = self.value(seconds)
        self.clock = seconds
        return value

    @torch.no_grad()
    def write(self, indices, values):
        """Device reset at the clock: fresh CMO endpoint, tau restarted, fresh C_k."""
        ids, values = self._write_checks(indices, values)
        if not ids.numel():
            return
        cycles = self.cycles.reshape(-1)[ids] + 1
        self.anchor.reshape(-1)[ids] = self._program(ids, values, cycles)
        self.requested.reshape(-1)[ids] = values.clamp(self.law.program_min, self.law.program_max)
        self.last_write.reshape(-1)[ids] = self.clock
        self.cycles.reshape(-1)[ids] = cycles
        f = self.law.persistent_share
        c0 = cycle_normal(ids, cycles, self.seed)
        if self.law.kappa == 0:
            self.epsilon.reshape(-1)[ids] = (math.sqrt(f) * self.persistent.reshape(-1)[ids]
                                             + math.sqrt(1-f) * c0)
        else:
            c_lo, c_hi, d_lo, d_hi = self._episode_start(ids, cycles, c0,
                                                         self.persistent.reshape(-1)[ids])
            self.grid.reshape(-1)[ids] = 0
            self.c_lo.reshape(-1)[ids], self.c_hi.reshape(-1)[ids] = c_lo, c_hi
            if d_lo is not None:
                self.d_lo.reshape(-1)[ids], self.d_hi.reshape(-1)[ids] = d_lo, d_hi
        self.write_commands += ids.numel()

    def verify_score(self, read, requested, seconds_since_write):
        """Standardized verify deviation for P&V; uses only the nominal law, never hidden draws.
        With coefficient bins the controller uses the bin of the requested target (G_p is hidden)."""
        law = self.law
        u = seconds_since_write / law.reference_seconds
        if not law.coefficient_bins:
            expected = requested + law.mean_us * u**law.mean_exponent
            sd = ((law.cmo_scale * law.cmo_sigma(requested)).square()
                  + (law.spread_us * u**law.spread_exponent) ** 2).sqrt()
        else:
            # Same rounding as the scalar branch: coefficient terms in float64, sums in the read dtype.
            expected = requested + (law.mean_at(requested) * u**law.mean_exponent).to(requested.dtype)
            sd = ((law.cmo_scale * law.cmo_sigma(requested)).square()
                  + (law.spread_at(requested) * u**law.spread_exponent).square().to(requested.dtype)).sqrt()
        return torch.where(sd > 0, (read - expected) / sd, torch.zeros_like(sd))

    def oracle_y(self):
        """Hidden normalized drift Y at the clock. Labelled oracle arms only."""
        return self._y(self.clock, None).reshape(self.shape)

    def state_dict(self):
        return {**self._common_state('drift_family_reram.v1'),
                **{k: None if getattr(self, k) is None else getattr(self, k).detach().cpu().clone()
                   for k in self._TENSORS}}

    @classmethod
    def restore(cls, state, device='cpu'):
        obj = cls._restore_common(state, 'drift_family_reram.v1', ReramFamily, cls._TENSORS, device)
        obj._init_streams()
        if not bool(torch.isfinite(obj.value()).all()):
            raise ValueError('Invalid serialized drift state.')
        return obj


class PcmDriftCells(_Cells):
    """PCM-like family on one tensor of physical devices (R1 writes: tau restarted, nu redrawn)."""

    _TENSORS = ('target', 'requested', 'anchor', 'last_write', 'cycles', 'nu')

    def __init__(self, target, *, law, seed, name, start_seconds=0.0):
        ids = self._init_common(target, law, seed, name, start_seconds)
        self._nu_seed = stream_seed(self.seed, 'pcm-nu')
        self.anchor = self.target.clone()
        self.nu = self._draw_nu(ids, self.anchor.reshape(-1), self.cycles.reshape(-1)).reshape(self.shape)

    def _draw_nu(self, ids, programmed, cycles):
        mean, std = self.law.nu_moments(programmed)
        return (mean + std * cycle_normal(ids, cycles, self._nu_seed)).abs()

    def value(self, seconds=None, indices=None):
        seconds = self._seconds(seconds)
        ids = self._ids(indices)
        law = self.law
        age = seconds - self._flat(self.last_write, ids)
        ratio = ((age + law.t0_seconds) / law.t0_seconds).to(torch.float32)
        anchor = self._flat(self.anchor, ids)
        drifted = torch.where(age > 0, anchor * ratio ** (-self._flat(self.nu, ids)), anchor)
        held = drifted.clamp(law.drift_min, law.drift_max)
        return held.reshape(self.shape) if ids is None else held

    @torch.no_grad()
    def write(self, indices, values):
        ids, values = self._write_checks(indices, values)
        if not ids.numel():
            return
        cycles = self.cycles.reshape(-1)[ids] + 1
        programmed = values.clamp(self.law.program_min, self.law.program_max)
        self.anchor.reshape(-1)[ids] = programmed
        self.requested.reshape(-1)[ids] = programmed
        self.last_write.reshape(-1)[ids] = self.clock
        self.cycles.reshape(-1)[ids] = cycles
        self.nu.reshape(-1)[ids] = self._draw_nu(ids, programmed, cycles)
        self.write_commands += ids.numel()

    def verify_score(self, read, requested, seconds_since_write):
        """Standardized log-conductance deviation from the nominal exponent law."""
        mean, std = self.law.nu_moments(requested)
        log_time = math.log((seconds_since_write + self.law.t0_seconds) / self.law.t0_seconds)
        y = (read / requested).log() + mean * log_time
        sd = std * log_time
        return torch.where(sd > 0, y / sd, torch.zeros_like(sd))

    def oracle_nu(self):
        """Hidden per-episode exponents. Labelled oracle arms only."""
        return self.nu.clone()

    def state_dict(self):
        return {**self._common_state('drift_family_pcm.v1'),
                **{k: getattr(self, k).detach().cpu().clone() for k in self._TENSORS}}

    @classmethod
    def restore(cls, state, device='cpu'):
        obj = cls._restore_common(state, 'drift_family_pcm.v1', PcmFamily, cls._TENSORS, device)
        obj._nu_seed = stream_seed(obj.seed, 'pcm-nu')
        return obj


@torch.no_grad()
def program_verify(cells, indices, targets, *, attempts, threshold, verify_seconds, first_written=False):
    """Reset ``indices`` to ``targets`` with program-and-verify; returns cost counts.

    Attempt k resets the pending devices at the clock. Unless k == attempts, the
    clock then moves on by ``verify_seconds``, the pending devices are read and
    those whose standardized deviation exceeds ``threshold`` are reset again.
    The last attempt is kept without a verify read, so attempts == 1 is a single
    unverified reset. Devices outside ``indices`` keep drifting meanwhile.
    With ``first_written`` the devices were just programmed to ``targets`` at the
    clock (for example by construction), and that write is attempt 1.
    """
    if type(attempts) is not int or attempts < 1:
        raise ValueError('Expected a positive integer attempt cap.')
    if (attempts == 1) != (threshold is None):
        raise ValueError('A threshold is required exactly when attempts > 1.')
    if threshold is not None and (not math.isfinite(threshold) or threshold <= 0):
        raise ValueError('Expected a positive finite threshold.')
    if not math.isfinite(verify_seconds) or verify_seconds <= 0:
        raise ValueError('Expected a positive verify delay.')
    pending = cells._ids(indices)
    goal = targets.detach().to(cells.target).flatten()
    if pending.numel() != goal.numel():
        raise ValueError('Expected one target per device.')
    goal = goal.clamp(cells.law.program_min, cells.law.program_max)
    if first_written:
        flat = lambda t: t.reshape(-1)[pending]
        if (bool((flat(cells.last_write) != cells.clock).any())
                or not torch.equal(flat(cells.requested), goal)):
            raise ValueError('first_written requires the devices to be programmed to targets at the clock.')
    histogram = [0] * attempts
    writes = reads = 0
    start = cells.clock
    for k in range(1, attempts + 1):
        if k > 1 or not first_written:
            cells.write(pending, goal)
        writes += pending.numel()
        if k == attempts:
            histogram[k - 1] += pending.numel()
            break
        cells.advance(cells.clock + verify_seconds)
        read = cells.value(indices=pending)
        cells.verify_reads += pending.numel()
        reads += pending.numel()
        reject = cells.verify_score(read, goal, verify_seconds).abs() > threshold
        histogram[k - 1] += int((~reject).sum())
        pending, goal = pending[reject], goal[reject]
        if not pending.numel():
            break
    return {'devices': sum(histogram), 'reset_attempts': writes, 'verify_reads': reads,
            'accepted_at_attempt': histogram, 'capped': histogram[-1] if attempts > 1 else 0,
            'start_seconds': start, 'end_seconds': cells.clock}
