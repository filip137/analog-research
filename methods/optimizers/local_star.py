"""methods.optimizers.local_star — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import torch
from hardware.om_crossbar import CrossbarTileSpec, StandardCrossbarForwardStates, flatten_logical_crossbar_matrices, standard_crossbar_forward_states


@dataclass(frozen=True)
class LocalStarCrossbarStep:
    """One label-addressed STAR step made only from layer-local errors."""

    gradient_q: torch.Tensor
    hidden_error: torch.Tensor
    output_error: torch.Tensor
    hidden_loss: torch.Tensor
    output_loss: torch.Tensor
    states: StandardCrossbarForwardStates


@torch.no_grad()
def local_star_crossbar_step(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    effective_state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    digital_scales: Sequence[float],
    hidden_targets: torch.Tensor,
    output_targets: torch.Tensor,
    hidden_gain: float,
    output_gain: float,
) -> LocalStarCrossbarStep:
    """Compute stopped, layer-local STAR errors and physical-``q`` gradients.

    The class label only addresses the stored target row.  There is no
    softmax/cross-entropy term, teacher query, transpose MVM, or cross-layer
    error propagation.  The hidden and output outer products are gradients of
    two separate local state-matching objectives.
    """

    if inputs.ndim != 2 or labels.ndim != 1 or labels.shape[0] != inputs.shape[0]:
        raise ValueError("Expected rank-2 inputs and one label per example.")
    gains = (float(hidden_gain), float(output_gain))
    if any(not math.isfinite(value) or value < 0.0 for value in gains):
        raise ValueError("Expected finite non-negative STAR gains.")
    scales = tuple(float(value) for value in digital_scales)
    if len(scales) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in scales
    ):
        raise ValueError("Expected two finite positive digital scales.")
    labels_device = labels.to(device=inputs.device, dtype=torch.int64)
    hidden_table = hidden_targets.to(device=inputs.device, dtype=inputs.dtype)
    output_table = output_targets.to(device=inputs.device, dtype=inputs.dtype)
    if (
        hidden_table.ndim != 2
        or output_table.ndim != 2
        or hidden_table.shape[0] != output_table.shape[0]
        or hidden_table.shape[0] < 1
        or bool(torch.any(labels_device < 0))
        or bool(torch.any(labels_device >= hidden_table.shape[0]))
        or not bool(torch.all(torch.isfinite(hidden_table)))
        or not bool(torch.all(torch.isfinite(output_table)))
    ):
        raise ValueError("Expected finite class-indexed STAR target tables.")

    states = standard_crossbar_forward_states(
        inputs,
        effective_state,
        layout,
        digital_scales=scales,
    )
    if (
        hidden_table.shape[1] != states.hidden_post_relu.shape[1]
        or output_table.shape[1] != states.output_logits.shape[1]
    ):
        raise ValueError("Expected STAR target widths to match local network states.")
    batch_size = inputs.shape[0]
    hidden_delta = states.hidden_post_relu - hidden_table[labels_device]
    output_delta = states.output_logits - output_table[labels_device]
    hidden_error = (
        gains[0]
        * hidden_delta
        * (states.hidden_preactivation > 0).to(inputs.dtype)
        / batch_size
    )
    output_error = gains[1] * output_delta / batch_size

    # Stop the inter-layer path: W1 sees only its hidden target error, while W2
    # sees the current hidden state solely as its locally available row signal.
    gradient_w0 = inputs.transpose(0, 1) @ hidden_error
    gradient_w1 = states.hidden_post_relu.detach().transpose(0, 1) @ output_error
    gradient_q = flatten_logical_crossbar_matrices(
        (scales[0] * gradient_w0, scales[1] * gradient_w1),
        layout,
    )
    return LocalStarCrossbarStep(
        gradient_q=gradient_q,
        hidden_error=hidden_error,
        output_error=output_error,
        hidden_loss=0.5 * gains[0] * hidden_delta.square().sum(dim=1).mean(),
        output_loss=0.5 * gains[1] * output_delta.square().sum(dim=1).mean(),
        states=states,
    )
