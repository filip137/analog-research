"""hardware.readout — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import dataclass
import torch


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
