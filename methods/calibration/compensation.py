"""methods.calibration.compensation — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from hardware.readout import MatrixCorrection


METHODS = ("global", "tile_global", "row", "column", "row_column", "global_shift", "global_moments")


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
