"""Shared inputs and helpers of the IBM OM target mappings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from training.ibm_om.characterization import IbmReramResetCommissioning
from training.ibm_om.coordinates import DeviceCoordinate
from training.ibm_om.population import IbmReramArrayPopulation
from training.ibm_om.programming.base import OutOfSupportContract


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
    coordinate: DeviceCoordinate


@dataclass(frozen=True)
class MappingInputs(MappingArguments):
    """Arguments plus the resolved device, bounds, masks and base report."""

    device: torch.device
    dtype: torch.dtype
    lower: torch.Tensor
    upper: torch.Tensor
    corrupt: torch.Tensor
    published_corrupt: torch.Tensor
    noncorrupt: torch.Tensor
    support_counts: Callable[..., dict[str, int]]
    global_support: dict[str, int]
    report: dict[str, Any]


@dataclass(frozen=True)
class MappingResult:
    """Per-cell targets plus what programming needs from the mapping."""

    targets: torch.Tensor
    report: dict[str, Any]
    program_mask: torch.Tensor | None
    out_of_support: OutOfSupportContract


# Historical compact-endpoint labels for mappings without an exact fallback.
# They name the quad fallback because that branch was the default; the
# fallback itself never runs for these mappings.
DEFAULT_FALLBACK_POLICY = "pulse_resolved_noncorrupt_out_of_bound_empty_quad_only"
DEFAULT_FALLBACK_EXECUTION_DETAIL = "compact_endpoint_with_exact_empty_quad_fallback"
DEFAULT_ENDPOINT_GENERATION_POLICY = (
    "compact_covered_exact_out_of_bound_empty_quad_fallback"
)


def reject_out_of_support() -> OutOfSupportContract:
    """No exact fallback: out-of-support targets follow the sampler policy."""

    return OutOfSupportContract(
        mode="reject",
        expected_below=None,
        expected_above=None,
        expected_nonempty_below=None,
        expected_nonempty_above=None,
        require_nonempty_zero=False,
        missing_audit_message=None,
        mismatch_message="",
        fallback_policy=DEFAULT_FALLBACK_POLICY,
        execution_detail=DEFAULT_FALLBACK_EXECUTION_DETAIL,
        endpoint_generation_policy=DEFAULT_ENDPOINT_GENERATION_POLICY,
    )


def empty_group_fallback(
    report: dict[str, Any],
    group_suffix: str,
    *,
    fallback_policy: str,
    execution_detail: str,
    endpoint_generation_policy: str,
) -> OutOfSupportContract:
    """Exact fallback for out-of-support cells of empty common windows."""

    return OutOfSupportContract(
        mode="pulse_fallback",
        expected_below=report.get(
            f"mapped_target_below_lower_bound_empty_{group_suffix}"
        ),
        expected_above=report.get(
            f"mapped_target_above_upper_bound_empty_{group_suffix}"
        ),
        expected_nonempty_below=report.get(
            "mapped_target_below_lower_bound_nonempty_"
            f"{group_suffix}"
        ),
        expected_nonempty_above=report.get(
            "mapped_target_above_upper_bound_nonempty_"
            f"{group_suffix}"
        ),
        require_nonempty_zero=True,
        missing_audit_message=None,
        mismatch_message=(
            "Expected compact exact-fallback cells to equal the "
            "preflighted out-of-bound cells from empty "
            f"{group_suffix} windows."
        ),
        fallback_policy=fallback_policy,
        execution_detail=execution_detail,
        endpoint_generation_policy=endpoint_generation_policy,
    )


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
    coordinate = arguments.coordinate
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
        coordinate=arguments.coordinate,
        lower=lower,
        upper=upper,
        corrupt=corrupt,
        published_corrupt=published_corrupt,
        noncorrupt=noncorrupt,
        support_counts=support_counts,
        global_support=global_support,
        report=report,
    )


def validate_quad_population(
    population: IbmReramArrayPopulation,
    layouts: tuple[tuple[str, str], ...] | None,
) -> None:
    """Modifier-time checks shared by the three four-cell quad mappings."""

    expected_layout_keys = set(population.binding_keys)
    provided_layout_keys = set(
        dict(layouts or ())
    )
    if provided_layout_keys != expected_layout_keys:
        raise ValueError(
            "Expected dual_rail_layout_by_parameter keys to equal the "
            "fixed IBM OM population binding keys. Provided value: "
            f"expected={sorted(expected_layout_keys)!r}, "
            f"provided={sorted(provided_layout_keys)!r}."
        )
    invalid_shapes = {
        key: shape
        for key, shape in zip(
            population.binding_keys,
            population.binding_shapes,
        )
        if len(shape) != 2 or shape[0] % 2 or shape[1] % 2
    }
    if invalid_shapes:
        raise ValueError(
            "Expected four-cell dual-rail bindings to be even-by-even "
            f"rank-2 tensors. Provided value: {invalid_shapes!r}."
        )
