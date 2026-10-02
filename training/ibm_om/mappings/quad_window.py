"""Dual-rail quad common window: four cells share one intersected window."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch

from training.ibm_om.mappings.base import (
    empty_group_fallback,
    MappingArguments,
    MappingInputs,
    validate_quad_layout,
)
from training.ibm_om.programming.base import OutOfSupportContract
from training.ibm_om.population import _tensor_summary
from training.ibm_om.topology import _quad_axes


NAME = "dual_rail_quad_common_window"


def validate_arguments(arguments: MappingArguments) -> MappingArguments:
    reset_commissioning = arguments.reset_commissioning
    reset_relative_mode = arguments.reset_relative_mode
    reset_relative_contrast_step = arguments.reset_relative_contrast_step
    validate_quad_layout(arguments)
    if (
        reset_commissioning is not None
        or reset_relative_mode is not None
        or reset_relative_contrast_step is not None
    ):
        raise ValueError(
            "Expected RESET-relative mapper inputs only for "
            "shared_reset_relative_quad."
        )
    return arguments


def map_targets(inputs: MappingInputs) -> tuple[torch.Tensor, dict[str, Any]]:
    global_targets = inputs.global_targets
    population = inputs.population
    layouts = inputs.layouts
    margin = inputs.margin
    device = inputs.device
    lower = inputs.lower
    upper = inputs.upper
    corrupt = inputs.corrupt
    published_corrupt = inputs.published_corrupt
    support_counts = inputs.support_counts
    report = inputs.report
    layout_by_key = dict(layouts or ())
    mapped = torch.empty_like(global_targets)
    all_overlap: list[torch.Tensor] = []
    all_corrupt_quad: list[torch.Tensor] = []
    all_published_corrupt_quad: list[torch.Tensor] = []
    all_raw_span: list[torch.Tensor] = []
    all_inner_span: list[torch.Tensor] = []
    all_nonempty_cell: list[torch.Tensor] = []
    offset = 0
    for key, shape in zip(population.binding_keys, population.binding_shapes):
        count = math.prod(shape)
        if len(shape) != 2 or shape[0] % 2 or shape[1] % 2:
            raise ValueError(
                "Expected quad common-window population bindings to be "
                "even-by-even rank-2 tensors. Provided value: "
                f"key={key!r}, shape={shape!r}."
            )
        target = global_targets[offset : offset + count].reshape(shape)
        target_mapped = mapped[offset : offset + count].reshape(shape)
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

        group_lower = torch.stack(
            tuple(
                cell_lower[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        )
        group_upper = torch.stack(
            tuple(
                cell_upper[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        )
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

        # Intersect physical support with the characterized global coordinate.
        common_low = group_lower.amax(dim=0).clamp_min(0.0)
        common_high = group_upper.amin(dim=0).clamp_max(1.0)
        # A point intersection cannot carry a logical fraction and is therefore
        # budgeted as an empty IBM quad, even though that one point is reachable.
        overlap = common_high > common_low
        raw_span = (common_high - common_low).clamp_min(0.0)
        inner_span = raw_span * (1.0 - 2.0 * margin)
        inner_low = common_low + margin * raw_span
        baseline = torch.where(
            overlap,
            inner_low,
            (0.5 * (common_low + common_high)).clamp(0.0, 1.0),
        )
        nonempty_cell = torch.empty(shape, dtype=torch.bool, device=device)
        for rows in row_groups:
            for columns in column_groups:
                target_mapped[rows[:, None], columns] = (
                    baseline
                    + target[rows[:, None], columns] * inner_span
                )
                nonempty_cell[rows[:, None], columns] = overlap

        parameter_noncorrupt = ~cell_corrupt.reshape(-1)
        parameter_mapped = target_mapped.reshape(-1)
        parameter_lower = cell_lower.reshape(-1)
        parameter_upper = cell_upper.reshape(-1)
        parameter_nonempty = nonempty_cell.reshape(-1)
        parameter_below = parameter_noncorrupt & (
            parameter_mapped < parameter_lower
        )
        parameter_above = parameter_noncorrupt & (
            parameter_mapped > parameter_upper
        )

        parameter_report = {
            "dual_rail_layout": layout_by_key[key],
            "devices": count,
            "quad_count": int(overlap.numel()),
            "common_window_empty_quad_count": int((~overlap).sum().item()),
            "common_window_empty_fraction": float(
                (~overlap).to(torch.float64).mean().item()
            ),
            "corrupt_quad_count": int(group_corrupt.sum().item()),
            "published_corrupt_quad_count": int(
                group_published_corrupt.sum().item()
            ),
            "raw_common_span": _tensor_summary(raw_span),
            "inner_common_span": _tensor_summary(inner_span),
            "mapped_target_below_lower_bound_nonempty_quad": int(
                (parameter_below & parameter_nonempty).sum().item()
            ),
            "mapped_target_above_upper_bound_nonempty_quad": int(
                (parameter_above & parameter_nonempty).sum().item()
            ),
            "mapped_target_below_lower_bound_empty_quad": int(
                (parameter_below & ~parameter_nonempty).sum().item()
            ),
            "mapped_target_above_upper_bound_empty_quad": int(
                (parameter_above & ~parameter_nonempty).sum().item()
            ),
        }
        report["parameters"][key] = parameter_report
        all_overlap.append(overlap.reshape(-1))
        all_corrupt_quad.append(group_corrupt.reshape(-1))
        all_published_corrupt_quad.append(group_published_corrupt.reshape(-1))
        all_raw_span.append(raw_span.reshape(-1))
        all_inner_span.append(inner_span.reshape(-1))
        all_nonempty_cell.append(nonempty_cell.reshape(-1))
        offset += count
    assert offset == population.size

    overlap = torch.cat(all_overlap)
    corrupt_quad = torch.cat(all_corrupt_quad)
    published_corrupt_quad = torch.cat(all_published_corrupt_quad)
    raw_span = torch.cat(all_raw_span)
    inner_span = torch.cat(all_inner_span)
    nonempty_cell = torch.cat(all_nonempty_cell)
    nonempty_support = support_counts(mapped, nonempty_cell)
    empty_support = support_counts(mapped, ~nonempty_cell)
    report.update(
        {
            "common_window_grouping": "dual_rail_quad",
            "common_window_group_size": 4,
            "common_window_group_count": int(overlap.numel()),
            "common_window_empty_group_count": int((~overlap).sum().item()),
            "corrupt_group_count": int(corrupt_quad.sum().item()),
            "published_corrupt_group_count": int(
                published_corrupt_quad.sum().item()
            ),
            "differential_pair_binding": None,
            "quad_count": int(overlap.numel()),
            "common_window_empty_quad_count": int((~overlap).sum().item()),
            "common_window_empty_fraction": float(
                (~overlap).to(torch.float64).mean().item()
            ),
            "corrupt_quad_count": int(corrupt_quad.sum().item()),
            "published_corrupt_quad_count": int(
                published_corrupt_quad.sum().item()
            ),
            "raw_common_span": _tensor_summary(raw_span),
            "inner_common_span": _tensor_summary(inner_span),
            "nonempty_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "nonempty_quad_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "mapped_target_below_lower_bound_nonempty_quad": nonempty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_nonempty_quad": nonempty_support[
                "above_upper_bound"
            ],
            "mapped_target_below_lower_bound_empty_quad": empty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_empty_quad": empty_support[
                "above_upper_bound"
            ],
            "mapped_target_below_lower_bound_nonempty_group": nonempty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_nonempty_group": nonempty_support[
                "above_upper_bound"
            ],
            "mapped_target_below_lower_bound_empty_group": empty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_empty_group": empty_support[
                "above_upper_bound"
            ],
            "mapped_target_support": support_counts(mapped),
        }
    )
    return mapped, report




def check_preflight(
    report: Mapping[str, Any],
    *,
    maximum_empty_quads: int,
    maximum_empty_pairs: int,
) -> None:
    if (
        isinstance(maximum_empty_quads, bool)
        or not isinstance(maximum_empty_quads, int)
        or maximum_empty_quads < 0
    ):
        raise ValueError(
            "Expected maximum_empty_quads to be a non-negative integer."
        )
    below = report.get("mapped_target_below_lower_bound_nonempty_quad")
    above = report.get("mapped_target_above_upper_bound_nonempty_quad")
    empty = report.get("common_window_empty_quad_count")
    if not all(
        isinstance(value, int) and value >= 0
        for value in (below, above, empty)
    ):
        raise ValueError(
            "Expected a strict IBM OM quad target-mapping preflight report."
        )
    if below or above:
        raise ValueError(
            "Expected zero mapped targets outside physical support on "
            "nonempty quad windows. Provided value: "
            f"below={below}, above={above}."
        )
    if empty > maximum_empty_quads:
        raise ValueError(
            "Expected no more than "
            f"{maximum_empty_quads} empty quad common windows. Provided "
            f"value: {empty}."
        )
    return


def programming_contract(
    arguments: MappingArguments,
    report: dict[str, Any],
) -> tuple[torch.Tensor | None, OutOfSupportContract]:
    """Empty quad windows are the only budgeted out-of-support cells."""

    return None, empty_group_fallback(
        report,
        "quad",
        fallback_policy="pulse_resolved_noncorrupt_out_of_bound_empty_quad_only",
        execution_detail="compact_endpoint_with_exact_empty_quad_fallback",
        endpoint_generation_policy=(
            "compact_covered_exact_out_of_bound_empty_quad_fallback"
        ),
    )
