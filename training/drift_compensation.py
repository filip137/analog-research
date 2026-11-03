"""Digital compensation fitted to observable, held crossbar readouts.

Matrices are one-hot MVM responses transposed to [outputs, inputs]. Physical
rows are inputs and physical columns are outputs. No conductance is changed.
Global/column mean-absolute readouts follow AIHWKit 1.1.0; the other methods
are explicit extensions. Calibration references are deployment readouts, not
ideal weights or hidden device parameters.
"""

from dataclasses import dataclass

import torch

METHODS = ("global", "tile_global", "row", "column", "row_column", "global_shift", "global_moments")


@dataclass
class MatrixCorrection:
    input_gain: torch.Tensor
    output_gain: torch.Tensor
    shift: torch.Tensor
    affine_gain: torch.Tensor
    tile_gain: torch.Tensor | None = None
    tile_size: int = 256

    def mvm(self, x, matrix):
        x = x * self.input_gain
        if self.tile_gain is None:
            result = x @ matrix.T
        else:
            # Digital gain on every partial tile output, before summation.
            result = None
            for i in range(0, matrix.shape[1], self.tile_size):
                part = x[:, i:i+self.tile_size] @ matrix[:, i:i+self.tile_size].T
                gain = self.tile_gain[:, i // self.tile_size].repeat_interleave(self.tile_size)[:matrix.shape[0]]
                part = part * gain
                result = part if result is None else result + part
        return result * self.output_gain

    def after_bias(self, value):
        return value * self.affine_gain + self.shift

    def state_dict(self):
        return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in vars(self).items()}

    @classmethod
    def from_state_dict(cls, state, device):
        return cls(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in state.items()})


@torch.no_grad()
def fit_readouts(reference, current, method, *, tile_size, iterations=10, bounds=(0.1, 10.0)):
    """Fit using complete one-hot readouts; reuse them across methods."""
    if method not in METHODS or reference.shape != current.shape or current.ndim != 2:
        raise ValueError("Expected a known method and matching 2D one-hot readouts.")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(current).all()):
        raise ValueError("Expected finite readouts.")
    if not (0 < bounds[0] <= 1 <= bounds[1]) or iterations < 1 or tile_size < 1:
        raise ValueError("Invalid compensation bounds, iterations or tile size.")
    outputs, inputs = current.shape
    result = MatrixCorrection(torch.ones(inputs, device=current.device), torch.ones(outputs, device=current.device),
                              torch.zeros(outputs, device=current.device), torch.ones(outputs, device=current.device),
                              tile_size=tile_size)

    def ratio(a, b, dim=None):
        return (a.abs().mean(dim=dim).clamp_min(1e-4) /
                b.abs().mean(dim=dim).clamp_min(1e-4)).clamp(*bounds)

    if method in ("global", "global_shift", "global_moments"):
        result.output_gain.fill_(ratio(reference, current))
    elif method == "column":
        result.output_gain = ratio(reference, current, 1)
    elif method == "row":
        result.input_gain = ratio(reference, current, 0)
    elif method == "tile_global":
        result.tile_gain = torch.stack([torch.stack([
            ratio(reference[o:o+tile_size, i:i+tile_size], current[o:o+tile_size, i:i+tile_size])
            for i in range(0, inputs, tile_size)]) for o in range(0, outputs, tile_size)])
    else:
        # Alternating diagonal least squares on observable responses. Fixed
        # iteration count; no held-out selection or gradients through devices.
        result.output_gain = ratio(reference, current, 1)
        for _ in range(iterations):
            scaled = current * result.output_gain[:, None]
            denominator = scaled.square().sum(0)
            gain = (scaled * reference).sum(0) / denominator.clamp_min(1e-20)
            result.input_gain = torch.where(denominator > 1e-20, gain, 1.).clamp(*bounds)
            scaled = current * result.input_gain[None, :]
            denominator = scaled.square().sum(1)
            gain = (scaled * reference).sum(1) / denominator.clamp_min(1e-20)
            result.output_gain = torch.where(denominator > 1e-20, gain, 1.).clamp(*bounds)
    return result


class Compensation:
    """Digital operators and optional streamed output statistics, outside model state."""

    def __init__(self, matrices, *, observe=False):
        self.matrices, self.observe = matrices, observe
        self.statistics = [None] * len(matrices)

    def mvm(self, flat, index, matrix):
        return self.matrices[index].mvm(flat, matrix)

    def after_bias(self, value, index):
        result = self.matrices[index].after_bias(value)
        if self.observe:
            v = result.detach().double()
            n, total, squares = v.shape[0], v.sum(0), v.square().sum(0)
            previous = self.statistics[index]
            self.statistics[index] = (n, total, squares) if previous is None else (
                previous[0] + n, previous[1] + total, previous[2] + squares)
        return result

    def moments(self):
        if any(s is None for s in self.statistics):
            raise ValueError("Expected observed outputs for every matrix.")
        return [(s / n, (ss / n - (s / n).square()).clamp_min(0)) for n, s, ss in self.statistics]

    def state_dict(self):
        return [m.state_dict() for m in self.matrices]

    @classmethod
    def from_state_dict(cls, states, device):
        return cls([MatrixCorrection.from_state_dict(s, device) for s in states])


@torch.no_grad()
def apply_moments(compensation, reference, current, *, shift_only, bounds=(0.1, 10.0)):
    """Match baseline mean, and optionally standard deviation, after global gain."""
    for matrix, (mean0, var0), (mean, var) in zip(compensation.matrices, reference, current, strict=True):
        gain = torch.ones_like(mean) if shift_only else (var0 / var.clamp_min(1e-12)).sqrt().clamp(*bounds)
        # Constant reference channels should remain constant, without a large gain.
        if not shift_only:
            gain = torch.where((var0 <= 1e-12) | (var <= 1e-12), 1., gain)
        matrix.affine_gain = gain.to(matrix.affine_gain)
        matrix.shift = (mean0 - gain * mean).to(matrix.shift)
