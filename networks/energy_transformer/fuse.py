"""Passive fuse-attention network: every coupling is one two-terminal element.

Tokens are differential node pairs. Signed weights are four positive resistors,
and each query/key pair is joined by a similarity-gated two-terminal element
(a resistive fuse): I = gf * w(|dV|^2) * dV. The same element carries the
current in both directions, so the network has a co-content energy

    E = sum_res g dV^2/2 + sum_fuse gf psi(|dV|^2) + beta * sum mask (y-t)^2/2,

whatever the individual element values are. Node capacitors give C dv/dt = -dE/dv.
The equilibrium-propagation derivative for every resistor is its own dV^2/2.

This is an ideal-element model. The explicit element list defined here is also
what `fuse_spice` exports. `cross_error`/`fuse_error` deliberately break
reciprocity to model separately built forward and return elements.
"""
from dataclasses import dataclass, field

import torch

ARRAYS = ("encoder", "position", "query", "key", "decoder")
KERNELS = ("gaussian", "cauchy", "linear")
DTYPE = torch.float64


@dataclass(frozen=True)
class Shape:
    tokens: int
    width: int
    heads: int = 1
    head_width: int = 1
    inputs: int = 1
    outputs: int = 1

    def array_shapes(self):
        n, d, h, a, p, o = (self.tokens, self.width, self.heads, self.head_width,
                            self.inputs, self.outputs)
        return {"encoder": (d, p), "position": (n, d), "query": (h, a, d),
                "key": (h, a, d), "decoder": (o, d)}


def initial_parameters(shape, seed, scales, floor=.02):
    """Signed draws w -> (gp, gn) = floor + (max(w,0), max(-w,0)); one flat vector."""
    generator = torch.Generator().manual_seed(seed)
    parts = []
    for name in ARRAYS:
        size = shape.array_shapes()[name]
        w = scales[name]*torch.randn(size, generator=generator, dtype=DTYPE)
        parts += [floor+w.clamp(min=0), floor+(-w).clamp(min=0)]
    return torch.cat([p.reshape(-1) for p in parts])


def split_parameters(shape, theta):
    out, start = {}, 0
    for name in ARRAYS:
        size = shape.array_shapes()[name]
        count = int(torch.tensor(size).prod())
        out[name] = (theta[start:start+count].reshape(size),
                     theta[start+count:start+2*count].reshape(size))
        start += 2*count
    if start != len(theta):
        raise ValueError("Parameter vector does not match the declared shape")
    return out


def kernel(r2, sigma, kind):
    u = r2/(2*sigma**2)
    if kind == "gaussian":
        return torch.exp(-u)
    if kind == "cauchy":
        return 1/(1+u)
    if kind == "linear":
        return torch.ones_like(u)
    raise ValueError(f"Unknown fuse kernel {kind}")


def potential(r2, sigma, kind):
    """psi with d psi(|d|^2)/d d = kernel(|d|^2) d."""
    u = r2/(2*sigma**2)
    if kind == "gaussian":
        return sigma**2*(1-torch.exp(-u))
    if kind == "cauchy":
        return sigma**2*torch.log1p(u)
    if kind == "linear":
        return r2/2
    raise ValueError(f"Unknown fuse kernel {kind}")


@dataclass
class Network:
    """Explicit element list of the full differential circuit."""
    shape: Shape
    fuse_conductance: float = 1.
    fuse_width: float = .25
    fuse_kernel: str = "gaussian"
    readout_gain: float = 4.
    self_pairs: bool = False
    nodes: dict = field(init=False)
    sources: dict = field(init=False)

    def __post_init__(self):
        s = self.shape
        if self.fuse_kernel not in KERNELS:
            raise ValueError(f"Unknown fuse kernel {self.fuse_kernel}")
        names = [("z", i, d, r) for i in range(s.tokens) for d in range(s.width) for r in (0, 1)]
        names += [(kind, i, h, c, r) for kind in ("q", "k") for i in range(s.tokens)
                  for h in range(s.heads) for c in range(s.head_width) for r in (0, 1)]
        names += [("o", i, p, r) for i in range(s.tokens) for p in range(s.outputs) for r in (0, 1)]
        self.nodes = {name: index for index, name in enumerate(names)}
        sources = [("x", i, p, r) for i in range(s.tokens) for p in range(s.inputs) for r in (0, 1)]
        sources += [("b", r) for r in (0, 1)]
        self.sources = {name: len(self.nodes)+index for index, name in enumerate(sources)}
        terminal = {**self.nodes, **self.sources}
        # Resistors: terminals a, b; logical parameter id; input-port gate (-1 = always on).
        ra, rb, pid, gate = [], [], [], []
        offsets, start = {}, 0
        for name in ARRAYS:
            count = int(torch.tensor(s.array_shapes()[name]).prod())
            offsets[name] = (start, start+count)
            start += 2*count
        self.parameter_count = start

        def synapse(y, z, name, flat, port=-1):
            """Signal pair y (rails y0,y1) to pair z: gp straight, gn crossed."""
            for rail in (0, 1):
                ra.append(terminal[y+(rail,)]); rb.append(terminal[z+(rail,)])
                pid.append(offsets[name][0]+flat); gate.append(port)
                ra.append(terminal[y+(rail,)]); rb.append(terminal[z+(1-rail,)])
                pid.append(offsets[name][1]+flat); gate.append(port)

        for i in range(s.tokens):
            for d in range(s.width):
                for p in range(s.inputs):
                    synapse(("x", i, p), ("z", i, d), "encoder", d*s.inputs+p, port=i*s.inputs+p)
                self._bias_synapse(terminal, ra, rb, pid, gate, offsets, i, d)
                for h in range(s.heads):
                    for c in range(s.head_width):
                        flat = (h*s.head_width+c)*s.width+d
                        synapse(("z", i, d), ("q", i, h, c), "query", flat)
                        synapse(("z", i, d), ("k", i, h, c), "key", flat)
                for p in range(s.outputs):
                    synapse(("z", i, d), ("o", i, p), "decoder", p*s.width+d)
        self.res_a, self.res_b = torch.tensor(ra), torch.tensor(rb)
        self.res_parameter, self.res_gate = torch.tensor(pid), torch.tensor(gate)
        # Fuses: one vector element per ordered pair, head and rail.
        fq, fk, pair = [], [], []
        for i in range(s.tokens):
            for j in range(s.tokens):
                if i == j and not self.self_pairs:
                    continue
                for h in range(s.heads):
                    for rail in (0, 1):
                        fq.append([terminal[("q", i, h, c, rail)] for c in range(s.head_width)])
                        fk.append([terminal[("k", j, h, c, rail)] for c in range(s.head_width)])
                        pair.append((i, j, h, rail))
        self.fuse_q, self.fuse_k, self.fuse_pair = torch.tensor(fq), torch.tensor(fk), pair
        self.out_plus = torch.tensor([terminal[("o", i, p, 0)] for i in range(s.tokens) for p in range(s.outputs)])
        self.out_minus = torch.tensor([terminal[("o", i, p, 1)] for i in range(s.tokens) for p in range(s.outputs)])
        self.node_count, self.terminal_count = len(self.nodes), len(terminal)
        self.offsets = offsets

    def _bias_synapse(self, terminal, ra, rb, pid, gate, offsets, i, d):
        flat = i*self.shape.width+d
        for rail in (0, 1):
            ra.append(terminal[("b", rail)]); rb.append(terminal[("z", i, d, rail)])
            pid.append(offsets["position"][0]+flat); gate.append(-1)
            ra.append(terminal[("b", rail)]); rb.append(terminal[("z", i, d, 1-rail)])
            pid.append(offsets["position"][1]+flat); gate.append(-1)

    # -- per-sample boundary -------------------------------------------------
    def boundary(self, inputs, connected, target, mask):
        """inputs/connected: (B, tokens, inputs); target/mask: (B, tokens, outputs)."""
        inputs = torch.as_tensor(inputs, dtype=DTYPE)
        batch = len(inputs)
        flat = inputs.reshape(batch, -1)
        rails = torch.stack((flat, -flat), -1).reshape(batch, -1)
        bias = torch.tensor([1., -1.], dtype=DTYPE).expand(batch, 2)
        ports = torch.as_tensor(connected, dtype=DTYPE).reshape(batch, -1)
        gates = torch.where(self.res_gate[None] < 0, torch.ones((), dtype=DTYPE),
                            ports[:, self.res_gate.clamp(min=0)])
        return {"source": torch.cat((rails, bias), 1), "gate": gates,
                "target": torch.as_tensor(target, dtype=DTYPE).reshape(batch, -1),
                "mask": torch.as_tensor(mask, dtype=DTYPE).reshape(batch, -1)}

    # -- physical realization ------------------------------------------------
    def realize(self, theta, *, mismatch=0., cross_error=0., fuse_error=0., seed=0):
        """Element values. `mismatch` is per-element and reciprocal (one device,
        both directions). `cross_error`/`fuse_error` give the two directions of
        each coupling independent gain errors, as separately built forward and
        return elements would have."""
        generator = torch.Generator().manual_seed(seed)

        def draw(count, scale):
            return torch.exp(scale*torch.randn(count, generator=generator, dtype=DTYPE))

        n_res, n_fuse = len(self.res_a), len(self.fuse_q)
        return {"res_scale": draw(n_res, mismatch), "conductance": None,
                "theta": theta,
                "cross_a": draw(n_res, cross_error), "cross_b": draw(n_res, cross_error),
                "fuse_g": self.fuse_conductance*draw(n_fuse, mismatch),
                "fuse_sigma": self.fuse_width*draw(n_fuse, mismatch),
                "fuse_q": draw(n_fuse, fuse_error), "fuse_k": draw(n_fuse, fuse_error)}

    def conductances(self, physical):
        if physical["conductance"] is not None:
            return physical["conductance"]
        return physical["theta"][self.res_parameter]*physical["res_scale"]

    # -- circuit equations (single sample; batch with vmap) --------------------
    def readout(self, state):
        return self.readout_gain*(state[..., self.out_plus]-state[..., self.out_minus])/2

    def current(self, physical, state, source, gate, target, mask, beta):
        """Net current into each free node (units of G0 * volt)."""
        t = torch.cat((state, source))
        g = self.conductances(physical)*gate
        ta, tb = t[self.res_a], t[self.res_b]
        f = torch.zeros(self.terminal_count, dtype=state.dtype)
        f = f.index_add(0, self.res_a, g*(physical["cross_a"]*tb-ta))
        f = f.index_add(0, self.res_b, g*(physical["cross_b"]*ta-tb))
        d = t[self.fuse_q]-t[self.fuse_k]
        w = physical["fuse_g"]*kernel((d*d).sum(-1), physical["fuse_sigma"], self.fuse_kernel)
        through = w[:, None]*d
        f = f.index_add(0, self.fuse_q.reshape(-1), (-physical["fuse_q"][:, None]*through).reshape(-1))
        f = f.index_add(0, self.fuse_k.reshape(-1), (physical["fuse_k"][:, None]*through).reshape(-1))
        inject = beta*mask*self.readout_gain*(target-self.readout(state))/2
        f = f.index_add(0, self.out_plus, inject).index_add(0, self.out_minus, -inject)
        return f[:self.node_count]

    def energy(self, physical, state, source, gate, target, mask, beta):
        """Co-content; defined only for reciprocal elements."""
        t = torch.cat((state, source))
        g = self.conductances(physical)*gate
        e = (g*(t[self.res_a]-t[self.res_b])**2).sum()/2
        d = t[self.fuse_q]-t[self.fuse_k]
        e = e+(physical["fuse_g"]*potential((d*d).sum(-1), physical["fuse_sigma"], self.fuse_kernel)).sum()
        return e+beta*(mask*(self.readout(state)-target)**2).sum()/2

    def loss(self, state, target, mask):
        return (mask*(self.readout(state)-target)**2).sum(-1)/2

    def drops(self, state, source, gate):
        """Gated squared voltage drop / 2 across every resistor: dE/dg."""
        t = torch.cat((state, source), -1)
        return gate*(t[..., self.res_a]-t[..., self.res_b])**2/2

    def attention(self, physical, state):
        """Effective conductance of every fuse at this state: (B, fuses)."""
        d = state[..., self.fuse_q]-state[..., self.fuse_k]
        return physical["fuse_g"]*kernel((d*d).sum(-1), physical["fuse_sigma"], self.fuse_kernel)

    def reduce(self, per_element):
        """Sum element quantities onto their logical parameters."""
        out = torch.zeros(per_element.shape[:-1]+(self.parameter_count,), dtype=per_element.dtype)
        return out.index_add(-1, self.res_parameter, per_element)


class Unsettled(RuntimeError):
    pass


def _batched(network, physical, beta):
    def f(state, source, gate, target, mask):
        return network.current(physical, state, source, gate, target, mask, beta)
    args = lambda b: (b["source"], b["gate"], b["target"], b["mask"])
    return (lambda s, b: torch.func.vmap(f)(s, *args(b)),
            lambda s, b: torch.func.vmap(torch.func.jacrev(f))(s, *args(b)))


@torch.no_grad()
def settle(network, physical, boundary, beta=0., initial=None, *, tolerance=1e-12,
           max_steps=400, first_step=.05, growth=1.5):
    """Implicit-Euler transient of C dv/dt = I(v) with unit capacitors and a
    growing step (what a circuit simulator does); ends as Newton iterations."""
    batch = len(boundary["source"])
    state = (torch.zeros(batch, network.node_count, dtype=DTYPE) if initial is None
             else initial.clone())
    current, jacobian = _batched(network, physical, beta)
    eye = torch.eye(network.node_count, dtype=DTYPE)
    step = first_step
    for iteration in range(max_steps+1):
        f = current(state, boundary)
        residual = f.abs().amax(1)
        if not torch.isfinite(residual).all():
            raise Unsettled("Nonfinite circuit state")
        if residual.max() <= tolerance:
            return state, {"residual": residual.max().item(), "iterations": iteration}
        if iteration == max_steps:
            break
        state = state+torch.linalg.solve(eye/step-jacobian(state, boundary), f)
        step = min(step*growth, 1e12)
    raise Unsettled(f"Unsettled phase beta={beta:g}: residual={residual.max().item():g}")


def stability(network, physical, boundary, state, beta=0.):
    """Largest real part of the node Jacobian's eigenvalues (<0 is stable)."""
    _, jacobian = _batched(network, physical, beta)
    return torch.linalg.eigvals(jacobian(state, boundary)).real.amax(1)


def eqprop(network, physical, boundary, beta, free=None):
    """Centered EP from the free state: per-resistor d loss / d g and phases."""
    if free is None:
        free, _ = settle(network, physical, boundary)
    plus, _ = settle(network, physical, boundary, beta, free)
    minus, _ = settle(network, physical, boundary, -beta, free)
    drop = lambda s: network.drops(s, boundary["source"], boundary["gate"])
    return (drop(plus)-drop(minus))/(2*beta), {"free": free, "plus": plus, "minus": minus}


def implicit_gradient(network, physical, boundary, free):
    """Exact per-resistor loss gradient at the fixed point; also valid when the
    elements are not reciprocal."""
    g = network.conductances(physical).clone().requires_grad_(True)
    local = {**physical, "conductance": g}
    current, jacobian = _batched(network, local, 0.)
    state = free.clone().requires_grad_(True)
    loss = network.loss(state, boundary["target"], boundary["mask"])
    dloss = torch.autograd.grad(loss.sum(), state)[0]
    with torch.no_grad():
        adjoint = torch.linalg.solve(jacobian(free, boundary).transpose(1, 2), dloss)
    f = current(free, boundary)
    rows = []
    for b in range(len(free)):
        rows.append(torch.autograd.grad(-(adjoint[b]*f[b]).sum(), g, retain_graph=True)[0])
    # d f / d g carries the gate; rows are per-sample gradients w.r.t. shared g.
    return torch.stack(rows)


def finite_difference(network, theta, boundary, free, index, step=1e-6, **realization):
    """Central difference of the settled loss w.r.t. one logical parameter."""
    out = []
    for sign in (1, -1):
        shifted = theta.clone()
        shifted[index] += sign*step
        state, _ = settle(network, network.realize(shifted, **realization), boundary, initial=free)
        out.append(network.loss(state, boundary["target"], boundary["mask"]))
    return (out[0]-out[1])/(2*step)
