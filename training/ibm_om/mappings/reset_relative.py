"""Shared RESET-relative quad mapping on controller-observed baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

import torch

from training.ibm_om.characterization import IbmReramResetCommissioning
from training.ibm_om.constants import _RESET_RELATIVE_MODES
from training.ibm_om.mappings.base import (
    MappingArguments,
    MappingInputs,
    _round_half_away_from_zero,
    validate_quad_layout,
)
from training.ibm_om.programming.base import OutOfSupportContract
from training.ibm_om.population import _tensor_summary
from training.ibm_om.topology import _quad_axes


NAME = "shared_reset_relative_quad"


def validate_arguments(arguments: MappingArguments) -> MappingArguments:
    population = arguments.population
    layouts = arguments.layouts
    margin = arguments.margin
    reset_commissioning = arguments.reset_commissioning
    reset_relative_mode = arguments.reset_relative_mode
    reset_relative_contrast_step = arguments.reset_relative_contrast_step
    validate_quad_layout(arguments)
    if margin != 0.0:
        raise ValueError(
            "Expected shared_reset_relative_quad to use zero "
            "common-window margin."
        )
    if (
        not isinstance(reset_commissioning, IbmReramResetCommissioning)
        or reset_commissioning.population_fingerprint
        != population.fingerprint
        or reset_commissioning.binding_keys != population.binding_keys
        or reset_commissioning.binding_shapes
        != population.binding_shapes
        or reset_commissioning.dual_rail_layout_by_parameter
        != layouts
    ):
        raise ValueError(
            "Expected shared_reset_relative_quad to receive the "
            "controller-observed commissioning artifact for this "
            "exact population and layout."
        )
    if reset_relative_mode not in _RESET_RELATIVE_MODES:
        raise ValueError(
            "Expected reset_relative_mode to be 'continuous' or "
            "'quantized_9_level'."
        )
    step = reset_relative_contrast_step
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float))
        or not math.isfinite(float(step))
        or not 0.0 < float(step) <= 0.5
    ):
        raise ValueError(
            "Expected a finite RESET-relative contrast step in "
            f"(0, 0.5]. Provided value: {step!r}."
        )
    reset_relative_contrast_step = float(step)
    return replace(
        arguments,
        reset_relative_contrast_step=reset_relative_contrast_step,
    )


def map_targets(inputs: MappingInputs) -> tuple[torch.Tensor, dict[str, Any]]:
    global_targets = inputs.global_targets
    population = inputs.population
    layouts = inputs.layouts
    reset_commissioning = inputs.reset_commissioning
    reset_relative_mode = inputs.reset_relative_mode
    reset_relative_contrast_step = inputs.reset_relative_contrast_step
    device = inputs.device
    dtype = inputs.dtype
    lower = inputs.lower
    upper = inputs.upper
    corrupt = inputs.corrupt
    published_corrupt = inputs.published_corrupt
    support_counts = inputs.support_counts
    report = inputs.report
    assert reset_commissioning is not None
    assert reset_relative_mode is not None
    assert reset_relative_contrast_step is not None
    baseline_flat = reset_commissioning.baseline.to(
        device=device, dtype=dtype
    )
    layout_by_key = dict(layouts or ())
    mapped = torch.empty_like(global_targets)
    all_quad_supported: list[torch.Tensor] = []
    all_codes: list[torch.Tensor] = []
    all_nonzero_contrast: list[torch.Tensor] = []
    all_corrupt_quad: list[torch.Tensor] = []
    all_published_corrupt_quad: list[torch.Tensor] = []
    offset = 0
    for key, shape in zip(
        population.binding_keys, population.binding_shapes
    ):
        count = math.prod(shape)
        target = global_targets[offset : offset + count].reshape(shape)
        target_mapped = mapped[offset : offset + count].reshape(shape)
        baseline = baseline_flat[offset : offset + count].reshape(shape)
        cell_lower = lower[offset : offset + count].reshape(shape)
        cell_upper = upper[offset : offset + count].reshape(shape)
        cell_corrupt = corrupt[offset : offset + count].reshape(shape)
        cell_published_corrupt = published_corrupt[
            offset : offset + count
        ].reshape(shape)
        row_groups, column_groups = _quad_axes(
            shape,
            layout_by_key[key],
            device=device,
        )
        plus_rows, minus_rows = row_groups
        plus_columns, minus_columns = column_groups
        q_pp = target[plus_rows[:, None], plus_columns]
        q_pm = target[plus_rows[:, None], minus_columns]
        q_mp = target[minus_rows[:, None], plus_columns]
        q_mm = target[minus_rows[:, None], minus_columns]
        logical_u = 0.5 * (q_pp - q_pm - q_mp + q_mm)
        raw_code = (4.0 * logical_u).clamp(-4.0, 4.0)
        code = (
            _round_half_away_from_zero(raw_code).clamp(-4.0, 4.0)
            if reset_relative_mode == "quantized_9_level"
            else raw_code
        )
        contrast = code * reset_relative_contrast_step
        positive_offset = contrast.clamp_min(0.0) / 2.0
        negative_offset = (-contrast).clamp_min(0.0) / 2.0
        # Every cell receives one of five shared RESET-relative offsets.
        # Only the observed per-quad RESET baseline varies by placement.
        target_mapped[plus_rows[:, None], plus_columns] = (
            baseline[plus_rows[:, None], plus_columns] + positive_offset
        )
        target_mapped[plus_rows[:, None], minus_columns] = (
            baseline[plus_rows[:, None], minus_columns] + negative_offset
        )
        target_mapped[minus_rows[:, None], plus_columns] = (
            baseline[minus_rows[:, None], plus_columns] + negative_offset
        )
        target_mapped[minus_rows[:, None], minus_columns] = (
            baseline[minus_rows[:, None], minus_columns] + positive_offset
        )
        group_inside = torch.stack(
            tuple(
                (~cell_corrupt[rows[:, None], columns])
                & (
                    target_mapped[rows[:, None], columns]
                    >= cell_lower[rows[:, None], columns]
                )
                & (
                    target_mapped[rows[:, None], columns]
                    <= cell_upper[rows[:, None], columns]
                )
                for rows in row_groups
                for columns in column_groups
            )
        ).all(dim=0)
        group_corrupt = torch.stack(
            tuple(
                cell_corrupt[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        ).any(dim=0)
        group_published_corrupt = torch.stack(
            tuple(
                cell_published_corrupt[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        ).any(dim=0)
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
            "quad_count": int(code.numel()),
            "reset_relative_mode": reset_relative_mode,
            "reset_relative_contrast_step": (
                reset_relative_contrast_step
            ),
            "quad_baseline": _tensor_summary(
                baseline[plus_rows[:, None], plus_columns]
            ),
            "logical_u": _tensor_summary(logical_u),
            "logical_code": _tensor_summary(code),
            "logical_contrast": _tensor_summary(contrast),
            "fully_supported_quad_count": int(group_inside.sum().item()),
            "fully_supported_quad_fraction": float(
                group_inside.to(torch.float64).mean().item()
            ),
            "mapped_target_below_lower_bound": int(
                parameter_below.sum().item()
            ),
            "mapped_target_above_upper_bound": int(
                parameter_above.sum().item()
            ),
        }
        all_quad_supported.append(group_inside.reshape(-1))
        all_codes.append(code.reshape(-1))
        all_nonzero_contrast.append(contrast.reshape(-1))
        all_corrupt_quad.append(group_corrupt.reshape(-1))
        all_published_corrupt_quad.append(
            group_published_corrupt.reshape(-1)
        )
        offset += count
    if offset != population.size:  # pragma: no cover - validated layout
        raise RuntimeError("Expected RESET-relative mapping to cover all cells.")
    quad_supported = torch.cat(all_quad_supported)
    codes = torch.cat(all_codes)
    contrasts = torch.cat(all_nonzero_contrast)
    corrupt_quad = torch.cat(all_corrupt_quad)
    published_corrupt_quad = torch.cat(all_published_corrupt_quad)
    mapped_support = support_counts(mapped)
    code_histogram = None
    if reset_relative_mode == "quantized_9_level":
        code_histogram = {
            str(index): int((codes == float(index)).sum().item())
            for index in range(-4, 5)
        }
    report.update(
        {
            "common_window_grouping": "shared_reset_relative_quad",
            "common_window_group_size": 4,
            "common_window_group_count": int(quad_supported.numel()),
            "common_window_empty_group_count": 0,
            "corrupt_group_count": int(corrupt_quad.sum().item()),
            "published_corrupt_group_count": int(
                published_corrupt_quad.sum().item()
            ),
            "differential_pair_binding": None,
            "quad_count": int(quad_supported.numel()),
            "common_window_empty_quad_count": 0,
            "common_window_empty_fraction": 0.0,
            "corrupt_quad_count": int(corrupt_quad.sum().item()),
            "published_corrupt_quad_count": int(
                published_corrupt_quad.sum().item()
            ),
            "raw_common_span": None,
            "inner_common_span": None,
            "nonempty_inner_common_span": None,
            "reset_relative_mode": reset_relative_mode,
            "reset_relative_contrast_step": reset_relative_contrast_step,
            "reset_relative_signed_levels": (
                9
                if reset_relative_mode == "quantized_9_level"
                else None
            ),
            "reset_relative_cell_offsets": (
                [
                    0.0,
                    reset_relative_contrast_step / 2.0,
                    reset_relative_contrast_step,
                    1.5 * reset_relative_contrast_step,
                    2.0 * reset_relative_contrast_step,
                ]
                if reset_relative_mode == "quantized_9_level"
                else None
            ),
            "reset_relative_maximum_absolute_contrast": (
                4.0 * reset_relative_contrast_step
            ),
            "reset_relative_maximum_cell_offset": (
                2.0 * reset_relative_contrast_step
            ),
            "logical_code": _tensor_summary(codes),
            "logical_code_histogram": code_histogram,
            "logical_contrast": _tensor_summary(contrasts),
            "fully_supported_quad_count": int(
                quad_supported.sum().item()
            ),
            "fully_supported_quad_fraction": float(
                quad_supported.to(torch.float64).mean().item()
            ),
            "hidden_support_is_audit_only": True,
            "hidden_device_bounds_consumed_by_target_mapper": False,
            "commissioning": dict(reset_commissioning.report),
            "mapped_target_below_lower_bound_nonempty_quad": (
                mapped_support["below_lower_bound"]
            ),
            "mapped_target_above_upper_bound_nonempty_quad": (
                mapped_support["above_upper_bound"]
            ),
            "mapped_target_below_lower_bound_empty_quad": 0,
            "mapped_target_above_upper_bound_empty_quad": 0,
            "mapped_target_below_lower_bound_nonempty_group": (
                mapped_support["below_lower_bound"]
            ),
            "mapped_target_above_upper_bound_nonempty_group": (
                mapped_support["above_upper_bound"]
            ),
            "mapped_target_below_lower_bound_empty_group": 0,
            "mapped_target_above_upper_bound_empty_group": 0,
            "mapped_target_support": mapped_support,
            "mapped_target_outside_0_1": int(
                ((mapped < 0.0) | (mapped > 1.0)).sum().item()
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
    supported = report.get("fully_supported_quad_count")
    supported_fraction = report.get("fully_supported_quad_fraction")
    mapped_support = report.get("mapped_target_support")
    commissioning = report.get("commissioning")
    if (
        not isinstance(devices, int)
        or devices < 1
        or not isinstance(quad_count, int)
        or quad_count < 1
        or devices != 4 * quad_count
        or not isinstance(supported, int)
        or not 0 <= supported <= quad_count
        or not isinstance(supported_fraction, (int, float))
        or not math.isfinite(float(supported_fraction))
        or not math.isclose(
            float(supported_fraction),
            supported / quad_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not isinstance(mapped_support, Mapping)
        or set(mapped_support)
        != {"below_lower_bound", "above_upper_bound", "inside_bounds"}
        or sum(mapped_support.values()) != devices
        or not isinstance(commissioning, Mapping)
        or commissioning.get("controller_observations_only") is not True
        or commissioning.get(
            "hidden_device_bounds_consumed_by_target_mapper"
        )
        is not False
        or report.get("hidden_support_is_audit_only") is not True
        or report.get("hidden_device_bounds_consumed_by_target_mapper")
        is not False
        or report.get("common_window_grouping")
        != "shared_reset_relative_quad"
        or report.get("common_window_group_size") != 4
        or report.get("common_window_group_count") != quad_count
        or report.get("common_window_empty_group_count") != 0
        or report.get("common_window_empty_quad_count") != 0
    ):
        raise ValueError(
            "Expected a strict shared RESET-relative target-mapping "
            "preflight report."
        )
    if float(supported_fraction) < 0.95:
        raise ValueError(
            "Expected at least 95 percent of RESET-relative quads to be "
            "fully inside their hidden per-cell support in the audit. "
            f"Provided value: {float(supported_fraction):.9f}."
        )
    if report.get("global_target_outside_0_1") != 0:
        raise ValueError(
            "Expected RESET-relative source targets inside [0, 1]."
        )
    if report.get("mapped_target_outside_0_1") != 0:
        raise ValueError(
            "Expected RESET-relative commissioned targets inside the "
            "shared characterized coordinate [0, 1] without clipping. "
            f"Provided outside count: "
            f"{report.get('mapped_target_outside_0_1')!r}."
        )
    return


def programming_contract(
    arguments: MappingArguments,
    report: dict[str, Any],
) -> tuple[torch.Tensor | None, OutOfSupportContract]:
    """The audit-only out-of-support set is programmed pulse-resolved."""

    expected_support = report.get("mapped_target_support")
    missing_audit_message = None
    expected_below = None
    expected_above = None
    if not isinstance(expected_support, Mapping):
        missing_audit_message = (
            "Expected RESET-relative compact execution to receive "
            "its authoritative hidden-support audit."
        )
    else:
        expected_below = expected_support.get("below_lower_bound")
        expected_above = expected_support.get("above_upper_bound")
    return None, OutOfSupportContract(
        mode="pulse_fallback",
        expected_below=expected_below,
        expected_above=expected_above,
        expected_nonempty_below=None,
        expected_nonempty_above=None,
        require_nonempty_zero=False,
        missing_audit_message=missing_audit_message,
        mismatch_message=(
            "Expected RESET-relative exact-fallback cells to equal "
            "the audit-only out-of-bound target set."
        ),
        fallback_policy=(
            "pulse_resolved_noncorrupt_out_of_bound_shared_reset_relative_only"
        ),
        execution_detail=(
            "compact_endpoint_with_exact_reset_relative_support_fallback"
        ),
        endpoint_generation_policy=(
            "compact_covered_exact_out_of_bound_reset_relative_fallback"
        ),
    )
