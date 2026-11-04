"""Small, deterministic reference equations for AIHWKit 1.1.0 TTv2 buffers.

This module models the native C++ buffer update shared by
``BufferedTransferCompound`` and ``ChoppedTransferCompound`` after the caller
has computed the effective transfer scale.  It is an executable oracle for
the digital buffer, learning-rate placement, scan axis, and sequential unit
soft-bounds response; it is not a replacement for an AIHWKit pulsed fast
array.  In particular, it does not turn an averaged full-matrix DRN gradient
into the per-vector outer-product events used by the published TTv2 presets.

The buffer edge semantics are pinned to AIHWKit 1.1.0 native C++ rather than
the package's illustrative Python transfer tile.  The deployed-recovery
engine separately computes the corrected ``ChoppedTransferCompound`` scale;
``labs/tools/verify_ttv2_aihwkit_native_parity.py`` checks both pieces against
the installed native tile.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BufferedTransferResult:
    """One native-equation buffer update before physical slow-device noise."""

    omega: torch.Tensor
    hidden: torch.Tensor
    pulse_count: torch.Tensor


def _finite_nonnegative(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"Expected {name} to be finite and non-negative.")
    return numeric


def effective_fast_learning_rate(*, optimizer_lr: float, fast_lr: float) -> float:
    """Return AIHWKit's native fast-array LR choice.

    A strictly positive ``fast_lr`` is independent of the optimizer schedule;
    zero delegates to the current optimizer learning rate.
    """

    optimizer = _finite_nonnegative("optimizer_lr", optimizer_lr)
    fast = _finite_nonnegative("fast_lr", fast_lr)
    return fast if fast > 0.0 else optimizer


def effective_transfer_learning_rate(
    *,
    optimizer_lr: float,
    transfer_lr: float,
    scale_transfer_lr: bool,
) -> float:
    """Return the non-negative native fast-to-slow transfer LR."""

    optimizer = _finite_nonnegative("optimizer_lr", optimizer_lr)
    transfer = _finite_nonnegative("transfer_lr", transfer_lr)
    if not isinstance(scale_transfer_lr, bool):
        raise ValueError("Expected scale_transfer_lr to be a boolean.")
    return transfer * optimizer if scale_transfer_lr else transfer


def buffered_transfer_update(
    hidden: torch.Tensor,
    read_value: torch.Tensor,
    *,
    transfer_learning_rate: float,
    threshold: float,
    desired_bl: int = 1,
    momentum: float = 0.0,
    forget_buffer: bool = True,
) -> BufferedTransferResult:
    """Apply the native AIHWKit 1.1.0 transfer-buffer update equation.

    ``pulse_count`` is signed in the direction of the accumulated analog read.
    AIHWKit supplies its negation to the pulsed update primitive, whose update
    convention restores the same physical direction on the slow array.
    """

    if hidden.shape != read_value.shape:
        raise ValueError("Expected hidden and read_value to have the same shape.")
    if not hidden.is_floating_point() or not read_value.is_floating_point():
        raise ValueError("Expected floating-point hidden and read tensors.")
    if not bool(torch.all(torch.isfinite(hidden))) or not bool(
        torch.all(torch.isfinite(read_value))
    ):
        raise ValueError("Expected finite hidden and read tensors.")
    rate = _finite_nonnegative(
        "transfer_learning_rate",
        transfer_learning_rate,
    )
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value <= 0.0:
        raise ValueError("Expected threshold to be finite and positive.")
    if isinstance(desired_bl, bool) or not isinstance(desired_bl, int) or desired_bl < 1:
        raise ValueError("Expected desired_bl to be a positive integer.")
    momentum_value = float(momentum)
    if not math.isfinite(momentum_value):
        raise ValueError("Expected momentum to be finite.")
    if not isinstance(forget_buffer, bool):
        raise ValueError("Expected forget_buffer to be a boolean.")

    omega = hidden + read_value * rate
    pulse_count = torch.trunc(omega / threshold_value).clamp(
        min=-desired_bl,
        max=desired_bl,
    )
    crossing = pulse_count != 0.0
    updated = omega.clone()
    if forget_buffer:
        updated[crossing] = omega[crossing] * momentum_value
    else:
        retained_fraction = 1.0 - min(max(momentum_value, 0.0), 1.0)
        updated[crossing] = (
            omega[crossing]
            - retained_fraction * pulse_count[crossing] * threshold_value
        )
    return BufferedTransferResult(
        omega=omega,
        hidden=updated,
        pulse_count=pulse_count.to(torch.int64),
    )


def read_sequential_slice(
    fast_weight: torch.Tensor,
    *,
    cursor: int,
    transfer_columns: bool = True,
) -> tuple[torch.Tensor, int]:
    """Read one native logical ``[out, in]`` slice and advance its cursor."""

    if fast_weight.ndim != 2:
        raise ValueError("Expected a two-dimensional logical [out, in] tensor.")
    if not isinstance(transfer_columns, bool):
        raise ValueError("Expected transfer_columns to be a boolean.")
    axis_size = fast_weight.shape[1 if transfer_columns else 0]
    if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor < axis_size:
        raise ValueError("Expected cursor to index the selected transfer axis.")
    value = (
        fast_weight[:, cursor]
        if transfer_columns
        else fast_weight[cursor, :]
    ).clone()
    return value, (cursor + 1) % axis_size


def sequential_unit_soft_bounds_pulses(
    value: torch.Tensor,
    pulse_count: torch.Tensor,
    *,
    dw_min: float,
    limit: float = 1.0,
) -> torch.Tensor:
    """Apply noiseless unit soft-bounds pulses sequentially.

    This helper makes the non-commutation between a native pulse train and one
    bulk Euler update explicit.  It supports integer signed counts and the
    symmetric ``[-limit, limit]`` device used by the deterministic oracle.
    """

    if value.shape != pulse_count.shape or not value.is_floating_point():
        raise ValueError("Expected matching floating value and pulse-count tensors.")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("Expected finite initial soft-bounds state.")
    counts = pulse_count.to(torch.int64)
    if not torch.equal(counts.to(pulse_count.dtype), pulse_count):
        raise ValueError("Expected integer-valued pulse counts.")
    step = float(dw_min)
    bound = float(limit)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Expected dw_min to be finite and positive.")
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("Expected limit to be finite and positive.")

    result = value.clone()
    maximum = int(counts.abs().max().item()) if counts.numel() else 0
    for index in range(maximum):
        positive = counts > index
        negative = counts < -index
        result[positive] += step * (bound - result[positive]) / bound
        result[negative] -= step * (bound + result[negative]) / bound
    return result.clamp(-bound, bound)


__all__ = [
    "BufferedTransferResult",
    "buffered_transfer_update",
    "effective_fast_learning_rate",
    "effective_transfer_learning_rate",
    "read_sequential_slice",
    "sequential_unit_soft_bounds_pulses",
]
