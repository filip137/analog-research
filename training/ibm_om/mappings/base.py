"""Shared inputs and helpers of the IBM OM target mappings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from training.ibm_om.characterization import IbmReramResetCommissioning
from training.ibm_om.coordinates import LOGICAL, RAW_ACTIVE, DeviceCoordinate
from training.ibm_om.population import IbmReramArrayPopulation


@dataclass(frozen=True)
class MappingArguments:
    """Validated public arguments of one target-mapping call."""

    global_targets: torch.Tensor
    population: IbmReramArrayPopulation
    target_mapping: str
    layouts: tuple[tuple[str, str], ...] | None
    margin: float
    reset_commissioning: IbmReramResetCommissioning | None
    reset_relative_mode: str | None
    reset_relative_contrast_step: float | None
    raw_active_mode: str | None
    raw_active_unsupported_quad_policy: str | None
    differential_pairs: tuple[tuple[int, str, str, tuple[int, int]], ...]


@dataclass(frozen=True)
class MappingInputs(MappingArguments):
    """Arguments plus the resolved device, bounds, masks and base report."""

    device: torch.device
    dtype: torch.dtype
    coordinate: DeviceCoordinate
    lower: torch.Tensor
    upper: torch.Tensor
    corrupt: torch.Tensor
    published_corrupt: torch.Tensor
    noncorrupt: torch.Tensor
    support_counts: Callable[..., dict[str, int]]
    global_support: dict[str, int]
    report: dict[str, Any]


def _round_half_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)


def validate_quad_layout(arguments: MappingArguments) -> None:
    """Shared argument checks of the three four-cell quad mappings."""

    global_targets = arguments.global_targets
    population = arguments.population
    layouts = arguments.layouts
    expected_keys = set(population.binding_keys)
    provided_keys = set(dict(layouts or ()))
    if provided_keys != expected_keys:
        raise ValueError(
            "Expected dual_rail_layout_by_parameter keys to equal the "
            "fixed IBM OM population binding keys. Provided value: "
            f"expected={sorted(expected_keys)!r}, "
            f"provided={sorted(provided_keys)!r}."
        )
    outside_global = (global_targets < 0.0) | (global_targets > 1.0)
    if bool(torch.any(outside_global)):
        offending = global_targets[outside_global]
        raise ValueError(
            "Expected four-cell-mapped clean conductance fractions inside "
            "[0, 1]. Provided extrema: "
            f"minimum={float(offending.min().item())}, "
            f"maximum={float(offending.max().item())}."
        )


def prepare_inputs(arguments: MappingArguments) -> MappingInputs:
    """Resolve device, per-cell bounds, corruption masks and the base report."""

    global_targets = arguments.global_targets
    population = arguments.population
    target_mapping = arguments.target_mapping
    margin = arguments.margin
    device = global_targets.device
    dtype = global_targets.dtype
    coordinate = (
        RAW_ACTIVE if target_mapping == "raw_active_p90_quad" else LOGICAL
    )
    lower, upper = coordinate.cell_bounds(population, device=device, dtype=dtype)
    corrupt = population.corrupt.to(device=device)
    published_corrupt = population.published_corrupt.to(device=device)
    noncorrupt = ~corrupt

    def support_counts(
        targets: torch.Tensor,
        selection: torch.Tensor | None = None,
    ) -> dict[str, int]:
        selected = noncorrupt if selection is None else noncorrupt & selection
        return {
            "below_lower_bound": int((selected & (targets < lower)).sum().item()),
            "above_upper_bound": int((selected & (targets > upper)).sum().item()),
            "inside_bounds": int(
                (selected & (targets >= lower) & (targets <= upper)).sum().item()
            ),
        }

    global_support = support_counts(global_targets)
    report: dict[str, Any] = {
        "target_mapping": target_mapping,
        "common_window_margin_fraction": margin,
        "population_fingerprint": population.fingerprint,
        "corruption_policy": population.corruption_policy,
        "devices": population.size,
        "global_target_support": global_support,
        "global_target_outside_0_1": int(
            ((global_targets < 0.0) | (global_targets > 1.0)).sum().item()
        ),
        "parameters": {},
    }
    return MappingInputs(
        global_targets=arguments.global_targets,
        population=arguments.population,
        target_mapping=arguments.target_mapping,
        layouts=arguments.layouts,
        margin=arguments.margin,
        reset_commissioning=arguments.reset_commissioning,
        reset_relative_mode=arguments.reset_relative_mode,
        reset_relative_contrast_step=arguments.reset_relative_contrast_step,
        raw_active_mode=arguments.raw_active_mode,
        raw_active_unsupported_quad_policy=arguments.raw_active_unsupported_quad_policy,
        differential_pairs=arguments.differential_pairs,
        device=device,
        dtype=dtype,
        coordinate=coordinate,
        lower=lower,
        upper=upper,
        corrupt=corrupt,
        published_corrupt=published_corrupt,
        noncorrupt=noncorrupt,
        support_counts=support_counts,
        global_support=global_support,
        report=report,
    )
