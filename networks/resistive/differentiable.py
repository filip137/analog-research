"""Coupled, composable perfect-diode DRNs with truncated BPTT.

Conductances have (source, destination) orientation, hidden rails are halves,
and linear output rails are adjacent pairs. The equations are the dense
``ResistiveInteraction`` quadratic updates, including reciprocal loading.
T settles without a graph; K more sweeps propagate gradients through the
external input and the supplied, held hardware view. Evaluation uses T sweeps.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class DrnSettings:
    settling_iterations: int = 4
    gradient_iterations: int = 4
    voltage_amp: float = 5.0
    current_amp: float = 0.1
    input_gain: float = 100.0
    token_chunk_size: int = 256
    activation_checkpointing: bool = True


def diode(value):
    positive, negative = value.chunk(2, dim=-1)
    return torch.cat((positive.clamp_min(0), negative.clamp_max(0)), dim=-1)


def coefficients(g1, g2, settings):
    # Cancel the output layer's common energy metric B/A. The hidden layer's
    # outgoing load is AB * sum(G2), and its feedback current is B * v2 G2^T.
    dh = g1.sum(0) + settings.voltage_amp * settings.current_amp * g2.sum(1)
    do = g2.sum(0)
    return dh, do


def rail_product(voltage, conductance):
    # In the FP16 timing arm only conductance matmuls use half inputs.
    return (voltage.to(conductance.dtype) @ conductance).to(voltage.dtype)


def sweep(drive, output, g2, dh, do, settings):
    hidden = diode((drive + settings.current_amp * rail_product(output, g2.T)) / dh)
    output = settings.voltage_amp * rail_product(hidden, g2) / do
    return hidden, output


class DifferentiableDrn(nn.Module):
    """No stored voltages or noise draws; the caller owns the hardware view."""

    def __init__(self, settings=DrnSettings()):
        super().__init__()
        self.settings = settings
        self.fp16_update_products = False
        self.update_execution = None

    def _inputs(self, x, g1, g2, hidden_current, input_multiplier):
        s = self.settings
        dh, do = coefficients(g1, g2, s)
        rails = torch.cat((x, -x), dim=-1) * s.input_gain * input_multiplier
        # Signed digital bias currents bypass the input gain calibration.
        if self.fp16_update_products and self.training:
            g1, g2 = g1.half(), g2.half()
        drive = rail_product(rails, g1) + hidden_current
        return drive, g2, dh, do

    def _solve(self, x, g1, g2, hidden_current, input_multiplier, output_multiplier):
        s = self.settings
        args = (x, g1, g2, hidden_current, input_multiplier)
        execution = self.update_execution if self.training else None
        split_checkpoint = (execution and execution.checkpoint_gradient_only and
                            s.activation_checkpointing and torch.is_grad_enabled())
        with torch.set_grad_enabled(torch.is_grad_enabled() and not split_checkpoint):
            drive, product_g2, dh, do = self._inputs(*args)
        output = x.new_zeros((x.shape[0], g2.shape[1]))
        skip_first = execution and execution.skip_zero_feedback and s.settling_iterations > 0
        with torch.no_grad():
            if skip_first:
                output = s.voltage_amp * rail_product(diode(drive / dh), product_g2) / do
            for _ in range(int(bool(skip_first)), s.settling_iterations):
                _, output = sweep(drive, output, product_g2, dh, do, s)

        def continuation(output, drive, product_g2, dh, do):
            for _ in range(s.gradient_iterations):
                _, output = sweep(drive, output, product_g2, dh, do, s)
            return output

        if split_checkpoint:
            def from_settled(output, *args):
                return continuation(output, *self._inputs(*args))
            output = checkpoint(from_settled, output, *args, use_reentrant=False)
        elif self.training and torch.is_grad_enabled():
            output = continuation(output, drive, product_g2, dh, do)
        decoded = output[:, 0::2] - output[:, 1::2]
        return decoded * output_multiplier

    def forward(self, x, g1, g2, hidden_current, input_multiplier, output_multiplier):
        s = self.settings
        flat = x.reshape(-1, x.shape[-1])
        args = (g1, g2, hidden_current, input_multiplier, output_multiplier)
        pieces = []
        execution = self.update_execution if self.training else None
        chunk_size = execution.token_chunk_size if execution else s.token_chunk_size
        for part in flat.split(chunk_size):
            if self.training and torch.is_grad_enabled() and s.activation_checkpointing and not (execution and execution.checkpoint_gradient_only):
                pieces.append(checkpoint(self._solve, part, *args, use_reentrant=False))
            else:
                pieces.append(self._solve(part, *args))
        return torch.cat(pieces).reshape(*x.shape[:-1], g2.shape[1] // 2)

    @torch.no_grad()
    def diagnostics(self, x, g1, g2, hidden_current, input_multiplier):
        s = self.settings
        dh, do = coefficients(g1, g2, s)
        if not all(bool(torch.isfinite(g).all() and (g >= 0).all()) for g in (g1, g2)):
            raise FloatingPointError("Physical DRN conductances must be finite and nonnegative.")
        if not bool((dh > 0).all() and (do > 0).all()):
            raise FloatingPointError("DRN denominators must be positive.")
        flat = x.reshape(-1, x.shape[-1])
        drive = torch.cat((flat, -flat), -1) * s.input_gain * input_multiplier @ g1 + hidden_current
        out = flat.new_zeros((len(flat), g2.shape[1]))
        for _ in range(s.settling_iterations):
            hidden, out = sweep(drive, out, g2, dh, do, s)
        next_hidden, next_out = sweep(drive, out, g2, dh, do, s)
        defect = max(float((next_hidden-hidden).abs().max()), float((next_out-out).abs().max()))
        return {"projected_voltage_defect_max": defect, "minimum_denominator": min(float(dh.min()), float(do.min())),
                "maximum_voltage": max(float(hidden.abs().max()), float(out.abs().max()))}
