"""Readback layer: DRN conductances <-> per-cell [0, 1] fractions.

The modifier reads the clean master conductances as fractions of the model
range, writes programmed endpoints back through the same affine map for the
duration of one context, and restores the clean values afterwards.  The
mapping is not undone on readback: the DRN sees the programmed (mapped)
conductances.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from model.resistive.builders import ParameterBinding


def fractions_from_bindings(
    bindings: Sequence[ParameterBinding],
    *,
    conductance_min: float,
    conductance_max: float,
) -> tuple[torch.Tensor, list[tuple[int, tuple[int, ...]]]]:
    """Return the clean conductances as one flat fraction vector plus layout."""

    flat = []
    layout = []
    offset = 0
    span = conductance_max - conductance_min
    for binding in bindings:
        state = binding.state
        target = (state.detach() - conductance_min) / span
        if not bool(torch.all(torch.isfinite(target))):
            raise ValueError(
                f"Expected finite clean target for {binding.key!r}."
            )
        flat.append(target.reshape(-1))
        layout.append((offset, tuple(state.shape)))
        offset += target.numel()
    return torch.cat(flat), layout


def write_endpoints(
    bindings: Sequence[ParameterBinding],
    endpoint: torch.Tensor,
    layout: Sequence[tuple[int, tuple[int, ...]]],
    *,
    conductance_min: float,
    conductance_max: float,
    clamp_to_parameter_bounds: bool,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Write programmed endpoints into the bindings; return restore snapshots."""

    snapshots = []
    span = conductance_max - conductance_min
    for binding, (offset, shape) in zip(bindings, layout):
        state = binding.state
        count = math.prod(shape)
        clean = state.detach().clone()
        snapshots.append((state, clean))
        programmed = endpoint[offset : offset + count].reshape(shape)
        state.copy_(conductance_min + span * programmed.to(state))
        if clamp_to_parameter_bounds:
            binding.parameter.clamp_()
    return snapshots


def restore(snapshots: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Copy the clean master conductances back into place."""

    with torch.no_grad():
        for state, clean in snapshots:
            state.copy_(clean)
