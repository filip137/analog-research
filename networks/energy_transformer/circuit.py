"""Explicit dual-rail resistor currents and RC energy-attention dynamics.

The mathematical energy is a Lyapunov function, not electrical dissipation.
Buffers, current conveyors, compensation and rectifiers are ideal active parts.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import logsumexp, softmax, xlogy


@dataclass(frozen=True)
class Parameters:
    K: np.ndarray
    W: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    kappa: float = 1.7

    def __post_init__(self):
        for name in ("K", "W", "a", "b", "c"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {name}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.a.ndim != 1 or self.b.ndim != 1 or self.c.ndim != 1:
            raise ValueError("Drives must be vectors")
        if self.a.size < 1 or self.b.size < 2:
            raise ValueError("Require visible states and at least two logits")
        if self.K.shape != (self.b.size, self.a.size) or self.W.shape != (self.c.size, self.a.size):
            raise ValueError("Coupling shape must match the state pools")
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be positive and finite")

    @property
    def sizes(self):
        return self.a.size, self.b.size, self.c.size

    @property
    def size(self):
        return sum(self.sizes)

    def split(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (self.size,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite state with the declared shape")
        d, l, _ = self.sizes
        return state[:d], state[d:d+l], state[d+l:]

    def activations(self, state):
        v, h, u = self.split(state)
        return v, softmax(self.kappa*h), np.maximum(u, 0)

    def stability_margin(self, nudge=0.0):
        wnorm = np.linalg.norm(self.W, 2) if self.W.size else 0.0
        return float(1+nudge-0.5*self.kappa*np.linalg.norm(self.K, 2)**2-wnorm**2)


@dataclass(frozen=True)
class Components:
    conductance: float = 1e-4
    visible_capacitance: float = 1e-7
    logit_capacitance: float = 1.5e-7
    relu_capacitance: float = 8e-8
    floor: float = 0.02

    def __post_init__(self):
        if any(not np.isfinite(x) or x <= 0 for x in vars(self).values()):
            raise ValueError("All circuit components must be positive and finite")

    def capacitances(self, parameters):
        return np.repeat([self.visible_capacitance, self.logit_capacitance,
                          self.relu_capacitance], parameters.sizes)


class Circuit:
    """Resistor adjacency is stored in siemens; compensation is dimensionless."""

    def __init__(self, parameters, components=None):
        self.parameters = parameters
        self.components = components or Components()
        p, c = self.parameters, self.components
        d, l, _ = p.sizes
        self.edges = []
        self.conductances = np.zeros((2*p.size, 2*p.size))
        for matrix, offset in ((p.K, d), (p.W, d+l)):
            for (j, i), weight in np.ndenumerate(matrix):
                gp, gn = c.floor+max(weight, 0), c.floor+max(-weight, 0)
                for left, right, g in ((2*i, 2*(offset+j), gp),
                                       (2*i+1, 2*(offset+j)+1, gp),
                                       (2*i, 2*(offset+j)+1, gn),
                                       (2*i+1, 2*(offset+j), gn)):
                    self.edges.append((left, right, c.conductance*g))
                    self.conductances[left, right] += c.conductance*g
                    self.conductances[right, left] += c.conductance*g
        self.compensation = self.conductances.sum(axis=1)[::2]/c.conductance
        self.capacitances = c.capacitances(p)
        self.taus = self.capacitances/c.conductance

    def rail_currents(self, state):
        y = np.concatenate(self.parameters.activations(state))
        rails = np.column_stack((y, -y)).ravel()
        # Ohm's law independently on every exposed physical rail.
        return np.sum(self.conductances*(rails[:, None]-rails[None, :]), axis=1)

    def normalized_force(self, state, target, nudge=0.0, compensation=None):
        p, c = self.parameters, self.components
        v, _, _ = p.split(state)
        target = checked_target(p, target)
        if not np.isfinite(nudge):
            raise ValueError("Nonfinite nudge")
        y = np.concatenate(p.activations(state))
        outward = self.rail_currents(state)
        differential = (outward[::2]-outward[1::2])/2
        gains = self.compensation if compensation is None else np.asarray(compensation)
        if gains.shape != (p.size,) or not np.isfinite(gains).all():
            raise ValueError("Invalid compensation gains")
        force = np.concatenate((p.a, p.b, p.c))-state+gains*y-differential/c.conductance
        force[:v.size] -= nudge*(v-target)
        return force

    def rhs(self, time, state, target, nudge=0.0):
        return self.normalized_force(state, target, nudge)/self.taus


def checked_target(parameters, target):
    target = np.asarray(target, dtype=float)
    if target.shape != parameters.a.shape or not np.isfinite(target).all():
        raise ValueError("Target must be a finite visible vector")
    return target


def matrix_force(p, state, target, nudge=0.0):
    """Independent compact equation; not used to construct physical currents."""
    v, h, u = p.split(state)
    _, prob, r = p.activations(state)
    target = checked_target(p, target)
    return np.concatenate((p.a-v+p.K.T@prob+p.W.T@r-nudge*(v-target),
                           p.b-h+p.K@v, p.c-u+p.W@v))


def energy(p, state, target, nudge=0.0):
    v, prob, r = p.activations(state)
    target = checked_target(p, target)
    return float(0.5*np.sum((v-p.a)**2)+np.sum(xlogy(prob, prob))/p.kappa
                 -p.b@prob+0.5*r@r-p.c@r-prob@p.K@v-r@p.W@v
                 +0.5*nudge*np.sum((v-target)**2))


def dissipation(circuit, state, target, nudge=0.0):
    """Proven dE/dt for matched logit RC and tracking compensation."""
    p = circuit.parameters
    force = matrix_force(p, state, target, nudge)
    v, prob, _ = p.activations(state)
    _, _, u = p.split(state)
    d, l, _ = p.sizes
    fv, fh, fu = force[:d], force[d:d+l], force[d+l:]
    variance = np.sum(prob*fh**2)-(prob@fh)**2
    return float(-np.sum(fv**2)/circuit.taus[0]
                 -p.kappa*variance/circuit.taus[d]
                 -np.sum(fu[u > 0]**2)/(circuit.components.relu_capacitance/circuit.components.conductance))


def torch_energy(p, state, target, nudge=0.0):
    """Autograd reference of the scalar energy, independent of KCL algebra."""
    import torch
    d, l, _ = p.sizes
    v, h, u = state[:d], state[d:d+l], state[d+l:]
    tensor = lambda x: torch.tensor(x, dtype=state.dtype, device=state.device)
    logp = torch.log_softmax(p.kappa*h, dim=0)
    prob, r = logp.exp(), torch.relu(u)
    return (0.5*((v-tensor(p.a))**2).sum()+(prob*logp).sum()/p.kappa
            -tensor(p.b)@prob+0.5*r@r-tensor(p.c)@r-prob@tensor(p.K)@v
            -r@tensor(p.W)@v+0.5*nudge*((v-tensor(target))**2).sum())


@dataclass(frozen=True)
class Trajectory:
    time: np.ndarray
    states: np.ndarray
    residual: float


def simulate(circuit, target, nudge=0.0, initial=None, times=None):
    """Settle physical port-current KCL; never polish an unconverged state."""
    p = circuit.parameters
    checked_target(p, target)
    if p.stability_margin(nudge) <= 0:
        raise ValueError("Outside the declared sufficient stability bound")
    initial = np.zeros(p.size) if initial is None else np.asarray(initial, dtype=float)
    p.split(initial)
    stop = 80*max(circuit.taus)
    if times is None:
        times = np.linspace(0, stop, 1001)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0) or times[0] < 0 or not np.isfinite(times).all():
        raise ValueError("Expected increasing finite nonnegative times")
    solution = solve_ivp(lambda t, s: circuit.rhs(t, s, target, nudge),
                         (0, times[-1]), initial, t_eval=times,
                         rtol=2e-11, atol=2e-12, max_step=min(circuit.taus)/2)
    if not solution.success or solution.y.shape != (p.size, len(times)) or not np.isfinite(solution.y).all():
        raise RuntimeError(f"RC integration failed: {solution.message}")
    residual = float(np.max(np.abs(circuit.normalized_force(solution.y[:, -1], target, nudge))))
    if residual > 1e-9:
        raise RuntimeError(f"RC equilibrium did not settle: residual={residual}")
    return Trajectory(solution.t, solution.y.T, residual)


def reduced_energy(p, visible, target, nudge=0.0):
    r = np.maximum(p.W@visible+p.c, 0)
    return float(0.5*np.sum((visible-p.a)**2)
                 -logsumexp(p.kappa*(p.K@visible+p.b))/p.kappa
                 -0.5*r@r+0.5*nudge*np.sum((visible-target)**2))


def equilibrium_reference(p, target, nudge=0.0):
    """Independent root of the reduced energy gradient, used only as reference."""
    checked_target(p, target)
    if p.stability_margin(nudge) <= 0:
        raise ValueError("Outside the declared sufficient stability bound")
    def gradient(v):
        return v-p.a-p.K.T@softmax(p.kappa*(p.K@v+p.b))-p.W.T@np.maximum(p.W@v+p.c, 0)+nudge*(v-target)
    result = root(gradient, p.a, tol=1e-11)
    if not np.isfinite(result.x).all() or np.max(np.abs(gradient(result.x))) > 1e-10:
        raise RuntimeError(f"Independent equilibrium root failed: {result.message}")
    return np.concatenate((result.x, p.K@result.x+p.b, p.W@result.x+p.c))


def perturb(p, name, index, amount):
    value = getattr(p, name).copy()
    value[index] += amount
    return replace(p, **{name: value})
