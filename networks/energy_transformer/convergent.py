"""Circuit-oriented convergent ET with persistent inputs and quasistatic ports.

This is an independently derived ET variant, not a reproduction of the full
Høier et al. CET configuration. Analog macro assumptions are part of the model.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp, softmax

PARAMETERS = ("query", "key", "memory", "encoder", "decoder", "position")


@dataclass(frozen=True)
class Parameters:
    query: np.ndarray       # H, head width, token width
    key: np.ndarray
    memory: np.ndarray      # memory width, token width
    encoder: np.ndarray     # token width, patch width
    decoder: np.ndarray     # patch width, token width
    position: np.ndarray    # tokens, token width
    epsilon: float = .5
    kappa: float = 1 / np.sqrt(2)

    def __post_init__(self):
        for name in PARAMETERS:
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {name}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.query.ndim != 3 or self.position.ndim != 2:
            raise ValueError("Expected three-dimensional projections and token biases")
        h, a, d = self.query.shape
        n, dd = self.position.shape
        if min(h, a, n) < 1 or d < 3 or d != dd or self.key.shape != self.query.shape:
            raise ValueError("Inconsistent token/attention dimensions; width must be >=3")
        if self.memory.ndim != 2 or self.memory.shape[1] != d or not self.memory.shape[0]:
            raise ValueError("Expected a nonempty memory matrix")
        if self.encoder.ndim != 2 or self.encoder.shape[0] != d or not self.encoder.shape[1]:
            raise ValueError("Invalid encoder dimensions")
        if self.decoder.shape != self.encoder.T.shape:
            raise ValueError("Invalid decoder dimensions")
        if any(not np.isfinite(v) or v <= 0 for v in (self.epsilon, self.kappa)):
            raise ValueError("epsilon and kappa must be positive and finite")

    @property
    def shape(self):
        return self.position.shape

    @property
    def output_shape(self):
        return (self.shape[0], self.decoder.shape[0])

    @property
    def size(self):
        return self.position.size + np.prod(self.output_shape)

    def split(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (self.size,) or not np.isfinite(state).all():
            raise ValueError("Invalid state")
        return state[:self.position.size].reshape(self.shape), state[self.position.size:].reshape(self.output_shape)


@dataclass(frozen=True)
class Boundary:
    inputs: np.ndarray
    target: np.ndarray
    mask: np.ndarray

    def checked(self, parameters, nudge=0.):
        if not np.isfinite(nudge):
            raise ValueError("Nonfinite nudge")
        for name in ("inputs", "target", "mask"):
            value = np.asarray(getattr(self, name))
            if value.shape != parameters.output_shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid boundary {name}")
        if np.any(self.mask < 0) or np.any(self.mask > 1):
            raise ValueError("Mask must be in [0,1]")
        if np.any(1+nudge*self.mask <= 0):
            raise ValueError("Nudge must preserve positive output leak")
        return self


def ports(parameters, state):
    z, y = parameters.split(state)
    u = z-z.mean(axis=-1, keepdims=True)
    s = np.sqrt(np.mean(u*u, axis=-1, keepdims=True)+parameters.epsilon)
    g = u/s
    q = np.einsum("had,nd->hna", parameters.query, g)
    k = np.einsum("had,nd->hna", parameters.key, g)
    logits = np.einsum("hia,hja->hij", q, k)
    p = softmax(parameters.kappa*logits, axis=-1)
    r = np.maximum(g@parameters.memory.T, 0.)
    return {"z": z, "y": y, "g": g, "q": q, "k": k, "p": p, "r": r, "s": s}


def force(parameters, state, boundary, nudge=0., *, omit_key_return=False):
    b = boundary.checked(parameters, nudge)
    v = ports(parameters, state)
    q_return = np.einsum("hij,hja->hia", v["p"], v["k"])
    k_return = np.einsum("hij,hia->hja", v["p"], v["q"])
    drive = b.inputs@parameters.encoder.T+parameters.position
    drive += np.einsum("hna,had->nd", q_return, parameters.query)
    if not omit_key_return:
        drive += np.einsum("hna,had->nd", k_return, parameters.key)
    drive += v["r"]@parameters.memory+v["y"]@parameters.decoder
    dz = drive-v["z"]
    dy = v["g"]@parameters.decoder.T-v["y"]-nudge*b.mask*(v["y"]-b.target)
    return np.concatenate((dz.ravel(), dy.ravel()))


def resistor_projection(matrix, signal, floor=.02):
    """Differential currents into virtual ground, divided by the unit conductance."""
    plus = floor+np.maximum(matrix, 0.)
    minus = floor+np.maximum(-matrix, 0.)
    return np.sum(plus*signal[None, :]-minus*signal[None, :], axis=1)


def branch_force(parameters, state, boundary, nudge=0.):
    """Independent port wiring: signed resistor currents and two-input products."""
    b = boundary.checked(parameters, nudge)
    v = ports(parameters, state)
    n, d = parameters.shape
    h, a, _ = parameters.query.shape
    q = np.array([[resistor_projection(parameters.query[t], g) for g in v["g"]] for t in range(h)])
    k = np.array([[resistor_projection(parameters.key[t], g) for g in v["g"]] for t in range(h)])
    prob = np.array([[softmax(parameters.kappa*np.array([sum(q[t,i,c]*k[t,j,c] for c in range(a)) for j in range(n)])) for i in range(n)] for t in range(h)])
    dz, dy = np.empty_like(v["z"]), np.empty_like(v["y"])
    for i in range(n):
        current = -v["z"][i]+parameters.position[i]
        current += resistor_projection(parameters.encoder, b.inputs[i])
        current += resistor_projection(parameters.decoder.T, v["y"][i])
        relu = np.maximum(resistor_projection(parameters.memory, v["g"][i]), 0.)
        current += resistor_projection(parameters.memory.T, relu)
        for t in range(h):
            qr = sum(prob[t,i,j]*k[t,j] for j in range(n))
            kr = sum(prob[t,j,i]*q[t,j] for j in range(n))
            current += resistor_projection(parameters.query[t].T, qr)
            current += resistor_projection(parameters.key[t].T, kr)
        dz[i] = current
        dy[i] = resistor_projection(parameters.decoder, v["g"][i])-v["y"][i]-nudge*b.mask[i]*(v["y"][i]-b.target[i])
    return np.concatenate((dz.ravel(), dy.ravel()))


def energy(parameters, state, boundary, nudge=0.):
    b = boundary.checked(parameters, nudge)
    v = ports(parameters, state)
    logits = np.einsum("hia,hja->hij", v["q"], v["k"])
    e = np.sum(v["z"]*v["g"])-parameters.shape[1]*np.sum(v["s"])
    e -= np.sum((b.inputs@parameters.encoder.T+parameters.position)*v["g"])
    e -= np.sum(logsumexp(parameters.kappa*logits, axis=-1))/parameters.kappa
    e -= .5*np.sum(v["r"]**2)
    e += .5*np.sum(v["y"]**2)-np.sum(v["y"]*(v["g"]@parameters.decoder.T))
    return float(e+nudge*loss(parameters, state, b))


def loss(parameters, state, boundary):
    y = parameters.split(state)[1]
    return float(.5*np.sum(boundary.mask*(y-boundary.target)**2))


def dissipation(parameters, state, boundary, nudge=0., taus=(1., 1.3)):
    v = ports(parameters, state)
    fz, fy = parameters.split(force(parameters, state, boundary, nudge))
    centered = fz-fz.mean(axis=-1, keepdims=True)
    u = v["z"]-v["z"].mean(axis=-1, keepdims=True)
    hg = centered/v["s"]-u*np.sum(u*fz, axis=-1, keepdims=True)/(parameters.shape[1]*v["s"]**3)
    return float(-np.sum(fz*hg)/taus[0]-np.sum(fy*fy)/taus[1])


@dataclass(frozen=True)
class Trace:
    time: np.ndarray
    states: np.ndarray
    residual: float


def simulate(parameters, boundary, nudge=0., initial=None, times=None, taus=(1., 1.3), residual_tolerance=1e-9):
    boundary.checked(parameters, nudge)
    if len(taus) != 2 or any(not np.isfinite(t) or t <= 0 for t in taus):
        raise ValueError("Expected positive scalar token/output time constants")
    initial = np.zeros(parameters.size) if initial is None else np.asarray(initial, dtype=float)
    parameters.split(initial)
    times = np.linspace(0., 80., 401) if times is None else np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times)<2 or times[0]<0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("Expected strictly increasing nonnegative sample times")
    scale = np.r_[np.full(parameters.position.size, taus[0]), np.full(np.prod(parameters.output_shape), taus[1])]
    sol = solve_ivp(lambda t,s: force(parameters,s,boundary,nudge)/scale, (0., times[-1]), initial,
                    t_eval=times, rtol=1e-10, atol=1e-12, method="DOP853")
    if not sol.success or sol.y.shape != (parameters.size,len(times)) or not np.isfinite(sol.y).all():
        raise RuntimeError("Failed or nonfinite equilibration")
    residual = float(np.max(np.abs(force(parameters, sol.y[:,-1], boundary, nudge))))
    if residual > residual_tolerance:
        raise RuntimeError(f"Circuit did not settle: residual={residual:g}")
    return Trace(sol.t, sol.y.T, residual)


def torch_energy(parameters, raw, boundary, nudge=0.):
    """Independent scalar expression, used only by verification."""
    import torch
    t = {name:torch.as_tensor(np.array(getattr(parameters,name)), dtype=raw.dtype) for name in PARAMETERS}
    z = raw[:parameters.position.size].reshape(parameters.shape)
    y = raw[parameters.position.size:].reshape(parameters.output_shape)
    u = z-z.mean(-1,keepdim=True)
    s = (u.square().mean(-1,keepdim=True)+parameters.epsilon).sqrt()
    g = u/s
    x = torch.as_tensor(boundary.inputs,dtype=raw.dtype)
    target = torch.as_tensor(boundary.target,dtype=raw.dtype)
    mask = torch.as_tensor(boundary.mask,dtype=raw.dtype)
    q = torch.einsum("had,nd->hna",t["query"],g)
    k = torch.einsum("had,nd->hna",t["key"],g)
    logits = q@k.transpose(-1,-2)
    return ((z*g).sum()-parameters.shape[1]*s.sum()
            -(g*(x@t["encoder"].T+t["position"])).sum()
            -torch.logsumexp(parameters.kappa*logits,-1).sum()/parameters.kappa
            -.5*torch.relu(g@t["memory"].T).square().sum()
            +.5*y.square().sum()-(y*(g@t["decoder"].T)).sum()
            +.5*nudge*(mask*(y-target).square()).sum())


def pilot_case():
    """Frozen declaration, not searched for a passing sample."""
    rng = np.random.default_rng(20261113)
    shapes = ((2,2,4),(2,2,4),(6,4),(4,1),(1,4),(4,4))
    arrays = [rng.normal(0., .2 if i==5 else .08, shape) for i,shape in enumerate(shapes)]
    p = Parameters(*arrays)
    b = Boundary(np.array([[.7],[0.],[-.4],[0.]]), np.array([[.7],[-.2],[-.4],[.5]]), np.array([[0.],[1.],[0.],[1.]]))
    return p,b


def perturb(parameters, name, index, delta):
    value = np.array(getattr(parameters,name),copy=True)
    value[index] += delta
    return replace(parameters, **{name:value})
