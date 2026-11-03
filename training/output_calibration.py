"""Closed-form digital calibration of each analog MLP output, fitted layer by layer.

Mirrored in the DRN worktree. An architecture supplies ``branch(k, hidden, view)``
returning the residual stream after attention, the analog MLP output ``z`` before
its digital bias, and that bias. The correction ``z' = gain * z + shift`` acts
only on ``z``; conductances and model parameters never change. References are the
first-read statistics of ``z`` on the same calibration tokens, as they could be
recorded at deployment. Layer k is fitted after layers < k are corrected.
"""

from dataclasses import dataclass

import torch

METHODS = ("mlp_gain", "mlp_shift", "mlp_moments")


@dataclass
class OutputCorrection:
    gain: torch.Tensor
    shift: torch.Tensor

    def __call__(self, z):
        return z * self.gain + self.shift

    def state_dict(self):
        return {"gain": self.gain.detach().cpu(), "shift": self.shift.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state, device):
        return cls(state["gain"].to(device), state["shift"].to(device))


@dataclass(frozen=True)
class Statistics:
    count: int
    mean: torch.Tensor
    variance: torch.Tensor
    mean_abs: float

    def to_dict(self):
        return {"count": self.count, "mean": self.mean.cpu(), "variance": self.variance.cpu(), "mean_abs": self.mean_abs}

    @classmethod
    def from_dict(cls, state, device):
        return cls(state["count"], state["mean"].to(device), state["variance"].to(device), state["mean_abs"])


def statistics(outputs):
    """Per-channel moments of z over every token, accumulated in float64."""
    n, total, squares, absolute = 0, None, None, 0.
    for z in outputs:
        v = z.detach().reshape(-1, z.shape[-1]).double()
        s, ss = v.sum(0), v.square().sum(0)
        total, squares = (s, ss) if total is None else (total + s, squares + ss)
        absolute += float(v.abs().sum())
        n += v.shape[0]
    if not n:
        raise ValueError("Expected calibration outputs.")
    mean = total / n
    variance = (squares / n - mean.square()).clamp_min(0)
    return Statistics(n, mean, variance, absolute / (n * mean.numel()))


def solve(method, reference, current, bounds):
    """Map current statistics of z onto the first-read reference."""
    if method not in METHODS:
        raise ValueError(f"Expected one of {METHODS}. Provided value: {method!r}.")
    if not 0 < bounds[0] <= 1 <= bounds[1]:
        raise ValueError("Expected gain bounds containing one.")
    channels = reference.mean.numel()
    if method == "mlp_gain":
        ratio = reference.mean_abs / max(current.mean_abs, 1e-12)
        gain = torch.full((channels,), min(max(ratio, bounds[0]), bounds[1]), dtype=torch.float64,
                          device=reference.mean.device)
        shift = torch.zeros_like(gain)
    elif method == "mlp_shift":
        gain = torch.ones_like(reference.mean)
        shift = reference.mean - current.mean
    else:
        gain = (reference.variance / current.variance.clamp_min(1e-12)).sqrt().clamp(*bounds)
        # Constant channels stay constant, without a large gain.
        gain = torch.where((reference.variance <= 1e-12) | (current.variance <= 1e-12), 1., gain)
        shift = reference.mean - gain * current.mean
    return OutputCorrection(gain.float(), shift.float())


@torch.no_grad()
def propagate(prefix, branch, layers, view, corrections=None, method=None, reference=None, bounds=None):
    """Run the analog suffix over cached prefix states.

    With ``method``, fit each layer's correction from its observed statistics
    after the earlier layers are corrected; otherwise apply ``corrections`` (or
    none) and return the per-layer statistics.
    """
    fitting = method is not None
    if fitting and (corrections is not None or reference is None or bounds is None):
        raise ValueError("Expected either a method with reference statistics or fixed corrections.")
    hidden, fitted, observed = list(prefix), [], []
    for k in range(layers):
        parts = [branch(k, h, view) for h in hidden]
        stats = statistics(z for _, z, _ in parts)
        observed.append(stats)
        if fitting:
            correction = solve(method, reference[k], stats, bounds)
            fitted.append(correction)
        else:
            correction = None if corrections is None else corrections[k]
        hidden = [r + (z + b if correction is None else correction(z) + b) for r, z, b in parts]
        del parts
    return fitted if fitting else observed


def forward(hidden, branch, layers, view, corrections=None):
    """Suffix forward from one prefix state; identity corrections reproduce the native order."""
    for k in range(layers):
        residual, z, bias = branch(k, hidden, view)
        hidden = residual + ((z if corrections is None else corrections[k](z)) + bias)
    return hidden


def parameters(method, channels, layers):
    return layers * {"mlp_gain": 1, "mlp_shift": channels, "mlp_moments": 2 * channels}[method]


def costs(method, count, length, layers, channels):
    """Model-based acquisition counts: one calibration pass per age and once at deployment."""
    tokens = count * length
    return {"device_writes": 0, "verify_reads": 0, "digital_parameters": parameters(method, channels, layers),
            "calibration_sequences": count, "calibration_tokens": tokens,
            "age_analog_mlp_token_passes": tokens * layers,
            "deployment_reference_analog_mlp_token_passes": tokens * layers,
            "reference_storage_bytes": 4 * layers * (2 * channels + 1)}
