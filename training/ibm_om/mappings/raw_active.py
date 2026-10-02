"""Raw-active p90 quad mapping in the frozen array-wide g coordinate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from training.ibm_om.constants import (
    _RAW_ACTIVE_MODES,
    IBM_OM_RAW_ACTIVE_A_MAX,
    IBM_OM_RAW_ACTIVE_A_MIN,
    IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
    IBM_OM_RAW_ACTIVE_D90,
    IBM_OM_RAW_ACTIVE_QUANTIZED_LEVELS,
    IBM_OM_RAW_ACTIVE_SCALE,
)
from training.ibm_om.coordinates import RAW_ACTIVE
from training.ibm_om.mappings.base import (
    MappingArguments,
    MappingInputs,
    _round_half_away_from_zero,
    validate_quad_layout,
)
from training.ibm_om.population import (
    IbmReramArrayPopulation,
    _tensor_sha256,
    _tensor_summary,
)
from training.ibm_om.topology import _normalize_dual_rail_layouts, _quad_axes


NAME = "raw_active_p90_quad"


def _raw_active_structural_cell_mask(
    population: IbmReramArrayPopulation,
    layouts: Mapping[str, str] | Sequence[Sequence[str]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact raw-coordinate bounds and the frozen-D90 cell mask."""

    normalized = _normalize_dual_rail_layouts(layouts)
    if normalized is None or set(dict(normalized)) != set(
        population.binding_keys
    ):
        raise ValueError(
            "Expected raw-active layouts to match the population bindings."
        )
    lower, upper = RAW_ACTIVE.cell_bounds(population, device=device)
    corrupt = population.corrupt.to(device=device)
    eligible_cells = torch.zeros(
        population.size, dtype=torch.bool, device=device
    )
    layout_by_key = dict(normalized)
    offset = 0
    for key, shape in zip(
        population.binding_keys, population.binding_shapes
    ):
        count = math.prod(shape)
        parameter_lower = lower[offset : offset + count].reshape(shape)
        parameter_upper = upper[offset : offset + count].reshape(shape)
        parameter_corrupt = corrupt[offset : offset + count].reshape(shape)
        parameter_eligible = eligible_cells[
            offset : offset + count
        ].reshape(shape)
        row_groups, column_groups = _quad_axes(
            shape,
            layout_by_key[key],
            device=device,
        )
        indices = tuple(
            (rows[:, None], columns)
            for rows in row_groups
            for columns in column_groups
        )
        group_lower = torch.stack(
            tuple(parameter_lower[index] for index in indices)
        )
        group_upper = torch.stack(
            tuple(parameter_upper[index] for index in indices)
        )
        capacity = group_upper.amin(dim=0) - group_lower.amax(dim=0)
        group_corrupt = torch.stack(
            tuple(parameter_corrupt[index] for index in indices)
        ).any(dim=0)
        eligible = (
            (capacity >= IBM_OM_RAW_ACTIVE_D90)
            & (group_lower >= 0.0).all(dim=0)
            & ~group_corrupt
        )
        for index in indices:
            parameter_eligible[index] = eligible
        offset += count
    if offset != population.size:  # pragma: no cover - population validates
        raise RuntimeError(
            "Expected raw-active structural mask to cover every cell."
        )
    return eligible_cells, lower, upper

def validate_arguments(arguments: MappingArguments) -> MappingArguments:
    margin = arguments.margin
    reset_commissioning = arguments.reset_commissioning
    reset_relative_mode = arguments.reset_relative_mode
    reset_relative_contrast_step = arguments.reset_relative_contrast_step
    raw_active_mode = arguments.raw_active_mode
    raw_active_unsupported_quad_policy = arguments.raw_active_unsupported_quad_policy
    validate_quad_layout(arguments)
    if margin != 0.0:
        raise ValueError(
            "Expected raw_active_p90_quad to use zero common-window "
            "margin around the frozen D90 budget."
        )
    if raw_active_mode not in _RAW_ACTIVE_MODES:
        raise ValueError(
            "Expected raw_active_mode to be 'continuous' or "
            "'quantized_7_level'."
        )
    if raw_active_unsupported_quad_policy != "structural_failure":
        raise ValueError(
            "Expected the versioned raw-active unsupported-quad "
            "policy to equal 'structural_failure'."
        )
    if (
        reset_commissioning is not None
        or reset_relative_mode is not None
        or reset_relative_contrast_step is not None
    ):
        raise ValueError(
            "Expected raw_active_p90_quad not to consume RESET-relative "
            "commissioning inputs."
        )
    return arguments


def map_targets(inputs: MappingInputs) -> tuple[torch.Tensor, dict[str, Any]]:
    global_targets = inputs.global_targets
    population = inputs.population
    layouts = inputs.layouts
    raw_active_mode = inputs.raw_active_mode
    raw_active_unsupported_quad_policy = inputs.raw_active_unsupported_quad_policy
    device = inputs.device
    lower = inputs.lower
    upper = inputs.upper
    corrupt = inputs.corrupt
    support_counts = inputs.support_counts
    report = inputs.report
    report["coordinate"] = {
        "version": IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
        "a_min": IBM_OM_RAW_ACTIVE_A_MIN,
        "a_max": IBM_OM_RAW_ACTIVE_A_MAX,
        "scale": IBM_OM_RAW_ACTIVE_SCALE,
        "reference_consumed": False,
    }
    assert raw_active_mode is not None
    assert raw_active_unsupported_quad_policy is not None
    layout_by_key = dict(layouts or ())
    mapped = torch.empty_like(global_targets)
    expanded_eligible = torch.zeros(
        population.size, dtype=torch.bool, device=device
    )
    expanded_capacity = torch.empty_like(global_targets)
    expanded_baseline = torch.empty_like(global_targets)
    all_capacity: list[torch.Tensor] = []
    all_baseline: list[torch.Tensor] = []
    all_baseline_low: list[torch.Tensor] = []
    all_baseline_high: list[torch.Tensor] = []
    all_eligible: list[torch.Tensor] = []
    all_exact_support: list[torch.Tensor] = []
    all_corrupt_quad: list[torch.Tensor] = []
    all_passive_compatible: list[torch.Tensor] = []
    all_logical_u: list[torch.Tensor] = []
    all_code: list[torch.Tensor] = []
    all_differential: list[torch.Tensor] = []
    offset = 0
    for key, shape in zip(
        population.binding_keys, population.binding_shapes
    ):
        count = math.prod(shape)
        target = global_targets[offset : offset + count].reshape(shape)
        target_mapped = mapped[offset : offset + count].reshape(shape)
        cell_lower = lower[offset : offset + count].reshape(shape)
        cell_upper = upper[offset : offset + count].reshape(shape)
        cell_corrupt = corrupt[offset : offset + count].reshape(shape)
        cell_eligible = expanded_eligible[
            offset : offset + count
        ].reshape(shape)
        cell_capacity = expanded_capacity[
            offset : offset + count
        ].reshape(shape)
        cell_baseline = expanded_baseline[
            offset : offset + count
        ].reshape(shape)
        row_groups, column_groups = _quad_axes(
            shape,
            layout_by_key[key],
            device=device,
        )
        plus_rows, minus_rows = row_groups
        plus_columns, minus_columns = column_groups
        cell_indices = tuple(
            (rows[:, None], columns)
            for rows in row_groups
            for columns in column_groups
        )
        group_lower = torch.stack(
            tuple(cell_lower[index] for index in cell_indices)
        )
        group_upper = torch.stack(
            tuple(cell_upper[index] for index in cell_indices)
        )
        group_corrupt = torch.stack(
            tuple(cell_corrupt[index] for index in cell_indices)
        ).any(dim=0)
        common_lower = group_lower.amax(dim=0)
        common_upper = group_upper.amin(dim=0)
        capacity = common_upper - common_lower
        passive_compatible = (group_lower >= 0.0).all(dim=0)
        eligible = (
            (capacity >= IBM_OM_RAW_ACTIVE_D90)
            & passive_compatible
            & ~group_corrupt
        )
        baseline_low = common_lower + IBM_OM_RAW_ACTIVE_D90 / 2.0
        baseline_high = common_upper - IBM_OM_RAW_ACTIVE_D90 / 2.0
        # The midpoint is also well-defined for an unsupported quad.  It
        # is retained only to make the structural failure auditable; that
        # quad receives no target-programming pulses.
        baseline = 0.5 * (common_lower + common_upper)

        q_pp = target[plus_rows[:, None], plus_columns]
        q_pm = target[plus_rows[:, None], minus_columns]
        q_mp = target[minus_rows[:, None], plus_columns]
        q_mm = target[minus_rows[:, None], minus_columns]
        logical_u = 0.5 * (q_pp - q_pm - q_mp + q_mm)
        bounded_u = logical_u.clamp(-1.0, 1.0)
        if raw_active_mode == "quantized_7_level":
            code = _round_half_away_from_zero(
                3.0 * bounded_u
            ).clamp(-3.0, 3.0)
            differential = (
                code * (IBM_OM_RAW_ACTIVE_D90 / 3.0)
            )
        else:
            code = bounded_u
            differential = bounded_u * IBM_OM_RAW_ACTIVE_D90
        positive = baseline + differential / 2.0
        negative = baseline - differential / 2.0
        target_mapped[plus_rows[:, None], plus_columns] = positive
        target_mapped[plus_rows[:, None], minus_columns] = negative
        target_mapped[minus_rows[:, None], plus_columns] = negative
        target_mapped[minus_rows[:, None], minus_columns] = positive

        for index in cell_indices:
            cell_eligible[index] = eligible
            cell_capacity[index] = capacity
            cell_baseline[index] = baseline
        exact_cell_support = torch.stack(
            tuple(
                (target_mapped[index] >= cell_lower[index])
                & (target_mapped[index] <= cell_upper[index])
                for index in cell_indices
            )
        )
        exact_quad_support = exact_cell_support.all(dim=0)
        parameter_mapped = target_mapped.reshape(-1)
        parameter_noncorrupt = ~cell_corrupt.reshape(-1)
        parameter_below = parameter_noncorrupt & (
            parameter_mapped < cell_lower.reshape(-1)
        )
        parameter_above = parameter_noncorrupt & (
            parameter_mapped > cell_upper.reshape(-1)
        )
        report["parameters"][key] = {
            "dual_rail_layout": layout_by_key[key],
            "devices": count,
            "quad_count": int(capacity.numel()),
            "p90_eligible_quad_count": int(eligible.sum().item()),
            "p90_eligible_quad_fraction": float(
                eligible.to(torch.float64).mean().item()
            ),
            "structural_failure_quad_count": int(
                (~eligible).sum().item()
            ),
            "capacity": _tensor_summary(capacity),
            "eligible_baseline": (
                _tensor_summary(baseline[eligible])
                if bool(torch.any(eligible))
                else None
            ),
            "requested_logical_u": _tensor_summary(logical_u),
            "requested_differential": _tensor_summary(differential),
            "exact_target_supported_quad_count": int(
                exact_quad_support.sum().item()
            ),
            "mapped_target_below_lower_bound": int(
                parameter_below.sum().item()
            ),
            "mapped_target_above_upper_bound": int(
                parameter_above.sum().item()
            ),
        }
        all_capacity.append(capacity.reshape(-1))
        all_baseline.append(baseline.reshape(-1))
        all_baseline_low.append(baseline_low.reshape(-1))
        all_baseline_high.append(baseline_high.reshape(-1))
        all_eligible.append(eligible.reshape(-1))
        all_exact_support.append(exact_quad_support.reshape(-1))
        all_corrupt_quad.append(group_corrupt.reshape(-1))
        all_passive_compatible.append(
            passive_compatible.reshape(-1)
        )
        all_logical_u.append(logical_u.reshape(-1))
        all_code.append(code.reshape(-1))
        all_differential.append(differential.reshape(-1))
        offset += count
    if offset != population.size:  # pragma: no cover - validated layout
        raise RuntimeError(
            "Expected raw-active target mapping to cover every cell."
        )
    capacity = torch.cat(all_capacity)
    baseline = torch.cat(all_baseline)
    baseline_low = torch.cat(all_baseline_low)
    baseline_high = torch.cat(all_baseline_high)
    eligible = torch.cat(all_eligible)
    exact_support = torch.cat(all_exact_support)
    corrupt_quad = torch.cat(all_corrupt_quad)
    passive_compatible = torch.cat(all_passive_compatible)
    logical_u = torch.cat(all_logical_u)
    code = torch.cat(all_code)
    differential = torch.cat(all_differential)
    mapped_support = support_counts(mapped)
    code_histogram = None
    if raw_active_mode == "quantized_7_level":
        code_histogram = {
            str(index): int((code == float(index)).sum().item())
            for index in range(-3, 4)
        }
    eligible_cells = expanded_eligible
    report.update(
        {
            "common_window_grouping": "raw_active_p90_quad",
            "common_window_group_size": 4,
            "common_window_group_count": int(capacity.numel()),
            "common_window_empty_group_count": int(
                (~eligible).sum().item()
            ),
            "quad_count": int(capacity.numel()),
            "candidate_quad_count": int(capacity.numel()),
            "required_p90_quad_count": int(
                math.ceil(0.9 * capacity.numel())
            ),
            "p90_eligible_quad_count": int(eligible.sum().item()),
            "p90_eligible_quad_fraction": float(
                eligible.to(torch.float64).mean().item()
            ),
            "structural_failure_quad_count": int(
                (~eligible).sum().item()
            ),
            "structural_failure_cell_count": int(
                (~eligible_cells).sum().item()
            ),
            "corrupt_quad_count": int(corrupt_quad.sum().item()),
            "passive_compatible_quad_count": int(
                passive_compatible.sum().item()
            ),
            "exact_target_supported_quad_count": int(
                exact_support.sum().item()
            ),
            "raw_active_mode": raw_active_mode,
            "raw_active_unsupported_quad_policy": (
                raw_active_unsupported_quad_policy
            ),
            "frozen_differential_budget": IBM_OM_RAW_ACTIVE_D90,
            "one_cell_full_scale_displacement": (
                IBM_OM_RAW_ACTIVE_D90 / 2.0
            ),
            "quantized_level_count": (
                IBM_OM_RAW_ACTIVE_QUANTIZED_LEVELS
                if raw_active_mode == "quantized_7_level"
                else None
            ),
            "quantized_differential_spacing": (
                IBM_OM_RAW_ACTIVE_D90 / 3.0
                if raw_active_mode == "quantized_7_level"
                else None
            ),
            "logical_code_histogram": code_histogram,
            "requested_logical_u": _tensor_summary(logical_u),
            "requested_differential": _tensor_summary(differential),
            "quad_capacity": _tensor_summary(capacity),
            "eligible_quad_baseline": (
                _tensor_summary(baseline[eligible])
                if bool(torch.any(eligible))
                else None
            ),
            "eligible_quad_baseline_low": (
                _tensor_summary(baseline_low[eligible])
                if bool(torch.any(eligible))
                else None
            ),
            "eligible_quad_baseline_high": (
                _tensor_summary(baseline_high[eligible])
                if bool(torch.any(eligible))
                else None
            ),
            "baseline_shared_within_every_quad": True,
            "differential_scale_shared_across_eligible_quads": True,
            "reference_consumed_by_target_mapper": False,
            "mapped_target_below_lower_bound_nonempty_quad": 0,
            "mapped_target_above_upper_bound_nonempty_quad": 0,
            "mapped_target_below_lower_bound_empty_quad": (
                mapped_support["below_lower_bound"]
            ),
            "mapped_target_above_upper_bound_empty_quad": (
                mapped_support["above_upper_bound"]
            ),
            "mapped_target_below_lower_bound_nonempty_group": 0,
            "mapped_target_above_upper_bound_nonempty_group": 0,
            "mapped_target_below_lower_bound_empty_group": (
                mapped_support["below_lower_bound"]
            ),
            "mapped_target_above_upper_bound_empty_group": (
                mapped_support["above_upper_bound"]
            ),
            "mapped_target_support": mapped_support,
            "mapped_target_outside_0_1": int(
                ((mapped < 0.0) | (mapped > 1.0)).sum().item()
            ),
            "expanded_quad_capacity_sha256": _tensor_sha256(
                expanded_capacity
            ),
            "expanded_quad_baseline_sha256": _tensor_sha256(
                expanded_baseline
            ),
            "expanded_quad_eligibility_sha256": _tensor_sha256(
                expanded_eligible
            ),
        }
    )
    return mapped, report



def check_preflight(
    report: Mapping[str, Any],
    *,
    maximum_empty_quads: int,
    maximum_empty_pairs: int,
) -> None:
    devices = report.get("devices")
    quad_count = report.get("quad_count")
    required = report.get("required_p90_quad_count")
    eligible = report.get("p90_eligible_quad_count")
    structural = report.get("structural_failure_quad_count")
    mapped_support = report.get("mapped_target_support")
    coordinate = report.get("coordinate")
    if (
        not isinstance(devices, int)
        or devices < 1
        or not isinstance(quad_count, int)
        or quad_count < 1
        or devices != 4 * quad_count
        or required != math.ceil(0.9 * quad_count)
        or not isinstance(eligible, int)
        or not 0 <= eligible <= quad_count
        or structural != quad_count - eligible
        or report.get("structural_failure_cell_count")
        != 4 * structural
        or report.get("common_window_grouping")
        != "raw_active_p90_quad"
        or report.get("common_window_group_size") != 4
        or report.get("common_window_group_count") != quad_count
        or report.get("frozen_differential_budget")
        != IBM_OM_RAW_ACTIVE_D90
        or report.get("baseline_shared_within_every_quad") is not True
        or report.get(
            "differential_scale_shared_across_eligible_quads"
        )
        is not True
        or report.get("reference_consumed_by_target_mapper") is not False
        or report.get("raw_active_unsupported_quad_policy")
        != "structural_failure"
        or report.get("raw_active_mode") not in _RAW_ACTIVE_MODES
        or not isinstance(mapped_support, Mapping)
        or set(mapped_support)
        != {"below_lower_bound", "above_upper_bound", "inside_bounds"}
        or sum(mapped_support.values()) != devices
        or not isinstance(coordinate, Mapping)
        or coordinate.get("version")
        != IBM_OM_RAW_ACTIVE_COORDINATE_VERSION
        or coordinate.get("reference_consumed") is not False
    ):
        raise ValueError(
            "Expected a strict raw-active p90 quad target-mapping "
            "preflight report."
        )
    if report.get("global_target_outside_0_1") != 0:
        raise ValueError(
            "Expected raw-active source targets inside [0, 1]."
        )
    if (
        report.get("mapped_target_below_lower_bound_nonempty_quad") != 0
        or report.get("mapped_target_above_upper_bound_nonempty_quad")
        != 0
    ):
        raise ValueError(
            "Expected every p90-eligible raw-active target to remain "
            "inside its exact per-cell support."
        )
    return
