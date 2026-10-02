"""Differential-pair common window over adjacent G+/G- tensors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

import torch

from training.ibm_om.mappings.base import (
    empty_group_fallback,
    MappingArguments,
    MappingInputs,
)
from training.ibm_om.programming.base import OutOfSupportContract
from training.ibm_om.population import _tensor_summary
from training.ibm_om.topology import _canonical_differential_pair_layout


NAME = "differential_pair_common_window"


def validate_arguments(arguments: MappingArguments) -> MappingArguments:
    global_targets = arguments.global_targets
    population = arguments.population
    layouts = arguments.layouts
    differential_pairs = arguments.differential_pairs
    if layouts is not None:
        raise ValueError(
            "Expected differential_pair_common_window to have null "
            "layouts; pairing is fixed by the canonical adjacent "
            "plus/minus binding catalog."
        )
    differential_pairs = _canonical_differential_pair_layout(
        population.binding_keys,
        population.binding_shapes,
    )
    outside_global = (global_targets < 0.0) | (global_targets > 1.0)
    if bool(torch.any(outside_global)):
        offending = global_targets[outside_global]
        raise ValueError(
            "Expected differential-pair-mapped clean conductance "
            "fractions inside [0, 1]. Provided extrema: "
            f"minimum={float(offending.min().item())}, "
            f"maximum={float(offending.max().item())}."
        )
    return replace(arguments, differential_pairs=differential_pairs)


def map_targets(inputs: MappingInputs) -> tuple[torch.Tensor, dict[str, Any]]:
    global_targets = inputs.global_targets
    population = inputs.population
    margin = inputs.margin
    differential_pairs = inputs.differential_pairs
    lower = inputs.lower
    upper = inputs.upper
    corrupt = inputs.corrupt
    published_corrupt = inputs.published_corrupt
    support_counts = inputs.support_counts
    report = inputs.report
    mapped = torch.empty_like(global_targets)
    report["parameter_pairs"] = {}
    provenance_pairs = []
    all_overlap: list[torch.Tensor] = []
    all_corrupt_pair: list[torch.Tensor] = []
    all_published_corrupt_pair: list[torch.Tensor] = []
    all_raw_span: list[torch.Tensor] = []
    all_inner_span: list[torch.Tensor] = []
    all_nonempty_cell: list[torch.Tensor] = []
    offset = 0
    for pair_index, plus_key, minus_key, shape in differential_pairs:
        count = math.prod(shape)
        plus_slice = slice(offset, offset + count)
        minus_slice = slice(offset + count, offset + 2 * count)
        plus_target = global_targets[plus_slice].reshape(shape)
        minus_target = global_targets[minus_slice].reshape(shape)
        plus_lower = lower[plus_slice].reshape(shape)
        minus_lower = lower[minus_slice].reshape(shape)
        plus_upper = upper[plus_slice].reshape(shape)
        minus_upper = upper[minus_slice].reshape(shape)
        plus_corrupt = corrupt[plus_slice].reshape(shape)
        minus_corrupt = corrupt[minus_slice].reshape(shape)
        plus_published_corrupt = published_corrupt[plus_slice].reshape(shape)
        minus_published_corrupt = published_corrupt[minus_slice].reshape(shape)

        # Each logical branch coordinate uses exactly the two identities
        # at that coordinate. Physical support is intersected with the
        # characterized global coordinate before applying the margin.
        common_low = torch.maximum(plus_lower, minus_lower).clamp(0.0, 1.0)
        common_high = torch.minimum(plus_upper, minus_upper).clamp(0.0, 1.0)
        # A point intersection cannot carry a logical fraction.
        overlap = common_high > common_low
        raw_span = (common_high - common_low).clamp_min(0.0)
        inner_span = raw_span * (1.0 - 2.0 * margin)
        inner_low = common_low + margin * raw_span
        baseline = torch.where(
            overlap,
            inner_low,
            (0.5 * (common_low + common_high)).clamp(0.0, 1.0),
        )
        plus_mapped = baseline + plus_target * inner_span
        minus_mapped = baseline + minus_target * inner_span
        mapped[plus_slice] = plus_mapped.reshape(-1)
        mapped[minus_slice] = minus_mapped.reshape(-1)

        group_corrupt = plus_corrupt | minus_corrupt
        group_published_corrupt = (
            plus_published_corrupt | minus_published_corrupt
        )
        nonempty_cell = torch.cat(
            (overlap.reshape(-1), overlap.reshape(-1))
        )
        pair_mapped = torch.cat(
            (plus_mapped.reshape(-1), minus_mapped.reshape(-1))
        )
        pair_lower = torch.cat(
            (plus_lower.reshape(-1), minus_lower.reshape(-1))
        )
        pair_upper = torch.cat(
            (plus_upper.reshape(-1), minus_upper.reshape(-1))
        )
        pair_noncorrupt = torch.cat(
            ((~plus_corrupt).reshape(-1), (~minus_corrupt).reshape(-1))
        )
        pair_below = pair_noncorrupt & (pair_mapped < pair_lower)
        pair_above = pair_noncorrupt & (pair_mapped > pair_upper)
        nonempty_below = int((pair_below & nonempty_cell).sum().item())
        nonempty_above = int((pair_above & nonempty_cell).sum().item())
        empty_below = int((pair_below & ~nonempty_cell).sum().item())
        empty_above = int((pair_above & ~nonempty_cell).sum().item())
        pair_key = f"base.differential_pair.{pair_index}"
        pair_report = {
            "pair_index": pair_index,
            "conductance_plus_key": plus_key,
            "conductance_minus_key": minus_key,
            "shape": list(shape),
            "devices": 2 * count,
            "pair_count": count,
            "common_window_empty_pair_count": int((~overlap).sum().item()),
            "common_window_empty_fraction": float(
                (~overlap).to(torch.float64).mean().item()
            ),
            "corrupt_pair_count": int(group_corrupt.sum().item()),
            "published_corrupt_pair_count": int(
                group_published_corrupt.sum().item()
            ),
            "raw_common_span": _tensor_summary(raw_span),
            "inner_common_span": _tensor_summary(inner_span),
            "nonempty_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "nonempty_pair_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "mapped_target_below_lower_bound_nonempty_pair": nonempty_below,
            "mapped_target_above_upper_bound_nonempty_pair": nonempty_above,
            "mapped_target_below_lower_bound_empty_pair": empty_below,
            "mapped_target_above_upper_bound_empty_pair": empty_above,
        }
        report["parameter_pairs"][pair_key] = pair_report
        branch_values = (
            (
                "conductance_plus",
                plus_key,
                minus_key,
                plus_mapped,
                plus_lower,
                plus_upper,
                plus_corrupt,
            ),
            (
                "conductance_minus",
                minus_key,
                plus_key,
                minus_mapped,
                minus_lower,
                minus_upper,
                minus_corrupt,
            ),
        )
        for (
            role,
            key,
            peer_key,
            branch_mapped,
            branch_lower,
            branch_upper,
            branch_corrupt,
        ) in branch_values:
            branch_noncorrupt = ~branch_corrupt
            branch_below = branch_noncorrupt & (branch_mapped < branch_lower)
            branch_above = branch_noncorrupt & (branch_mapped > branch_upper)
            report["parameters"][key] = {
                "differential_role": role,
                "paired_parameter": peer_key,
                "pair_index": pair_index,
                "shape": list(shape),
                "devices": count,
                "pair_count": count,
                "common_window_empty_pair_count": int(
                    (~overlap).sum().item()
                ),
                "raw_common_span": _tensor_summary(raw_span),
                "inner_common_span": _tensor_summary(inner_span),
                "nonempty_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "nonempty_pair_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "mapped_target_below_lower_bound_nonempty_pair": int(
                    (branch_below & overlap).sum().item()
                ),
                "mapped_target_above_upper_bound_nonempty_pair": int(
                    (branch_above & overlap).sum().item()
                ),
                "mapped_target_below_lower_bound_empty_pair": int(
                    (branch_below & ~overlap).sum().item()
                ),
                "mapped_target_above_upper_bound_empty_pair": int(
                    (branch_above & ~overlap).sum().item()
                ),
            }
        provenance_pairs.append(
            {
                "pair_index": pair_index,
                "conductance_plus_key": plus_key,
                "conductance_minus_key": minus_key,
                "shape": list(shape),
            }
        )
        all_overlap.append(overlap.reshape(-1))
        all_corrupt_pair.append(group_corrupt.reshape(-1))
        all_published_corrupt_pair.append(
            group_published_corrupt.reshape(-1)
        )
        all_raw_span.append(raw_span.reshape(-1))
        all_inner_span.append(inner_span.reshape(-1))
        all_nonempty_cell.extend(
            (overlap.reshape(-1), overlap.reshape(-1))
        )
        offset += 2 * count
    assert offset == population.size

    overlap = torch.cat(all_overlap)
    corrupt_pair = torch.cat(all_corrupt_pair)
    published_corrupt_pair = torch.cat(all_published_corrupt_pair)
    raw_span = torch.cat(all_raw_span)
    inner_span = torch.cat(all_inner_span)
    nonempty_cell = torch.cat(all_nonempty_cell)
    nonempty_support = support_counts(mapped, nonempty_cell)
    empty_support = support_counts(mapped, ~nonempty_cell)
    below_nonempty = nonempty_support["below_lower_bound"]
    above_nonempty = nonempty_support["above_upper_bound"]
    below_empty = empty_support["below_lower_bound"]
    above_empty = empty_support["above_upper_bound"]
    pair_count = int(overlap.numel())
    empty_pair_count = int((~overlap).sum().item())
    corrupt_pair_count = int(corrupt_pair.sum().item())
    published_corrupt_pair_count = int(
        published_corrupt_pair.sum().item()
    )
    report.update(
        {
            "common_window_grouping": "differential_pair",
            "common_window_group_size": 2,
            "common_window_group_count": pair_count,
            "common_window_empty_group_count": empty_pair_count,
            "corrupt_group_count": corrupt_pair_count,
            "published_corrupt_group_count": published_corrupt_pair_count,
            "differential_pair_binding": {
                "policy": "canonical_adjacent_plus_minus_same_coordinate",
                "catalog_order": list(population.binding_keys),
                "pairs": provenance_pairs,
            },
            "pair_count": pair_count,
            "common_window_empty_pair_count": empty_pair_count,
            "common_window_empty_fraction": float(
                (~overlap).to(torch.float64).mean().item()
            ),
            "corrupt_pair_count": corrupt_pair_count,
            "published_corrupt_pair_count": published_corrupt_pair_count,
            "raw_common_span": _tensor_summary(raw_span),
            "inner_common_span": _tensor_summary(inner_span),
            "nonempty_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "nonempty_pair_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "mapped_target_below_lower_bound_nonempty_group": below_nonempty,
            "mapped_target_above_upper_bound_nonempty_group": above_nonempty,
            "mapped_target_below_lower_bound_empty_group": below_empty,
            "mapped_target_above_upper_bound_empty_group": above_empty,
            "mapped_target_below_lower_bound_nonempty_pair": below_nonempty,
            "mapped_target_above_upper_bound_nonempty_pair": above_nonempty,
            "mapped_target_below_lower_bound_empty_pair": below_empty,
            "mapped_target_above_upper_bound_empty_pair": above_empty,
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
        isinstance(maximum_empty_pairs, bool)
        or not isinstance(maximum_empty_pairs, int)
        or maximum_empty_pairs < 0
    ):
        raise ValueError(
            "Expected maximum_empty_pairs to be a non-negative integer."
        )
    below = report.get("mapped_target_below_lower_bound_nonempty_pair")
    above = report.get("mapped_target_above_upper_bound_nonempty_pair")
    empty = report.get("common_window_empty_pair_count")
    pair_count = report.get("pair_count")
    if not all(
        isinstance(value, int) and value >= 0
        for value in (below, above, empty, pair_count)
    ):
        raise ValueError(
            "Expected a strict IBM OM differential-pair target-mapping "
            "preflight report."
        )
    expected_generic = {
        "common_window_grouping": "differential_pair",
        "common_window_group_size": 2,
        "common_window_group_count": pair_count,
        "common_window_empty_group_count": empty,
        "mapped_target_below_lower_bound_nonempty_group": below,
        "mapped_target_above_upper_bound_nonempty_group": above,
        "mapped_target_below_lower_bound_empty_group": report.get(
            "mapped_target_below_lower_bound_empty_pair"
        ),
        "mapped_target_above_upper_bound_empty_group": report.get(
            "mapped_target_above_upper_bound_empty_pair"
        ),
    }
    if any(report.get(key) != value for key, value in expected_generic.items()):
        raise ValueError(
            "Expected differential-pair-specific and generic common-window "
            "preflight fields to agree exactly."
        )
    if empty > pair_count:
        raise ValueError(
            "Expected common_window_empty_pair_count not to exceed pair_count."
        )
    if below or above:
        raise ValueError(
            "Expected zero mapped targets outside physical support on "
            "nonempty differential-pair windows. Provided value: "
            f"below={below}, above={above}."
        )
    if empty > maximum_empty_pairs:
        raise ValueError(
            "Expected no more than "
            f"{maximum_empty_pairs} empty differential-pair common windows. "
            f"Provided value: {empty}."
        )


def programming_contract(
    arguments: MappingArguments,
    report: dict[str, Any],
) -> tuple[torch.Tensor | None, OutOfSupportContract]:
    """Empty differential-pair windows are the only budgeted failures."""

    return None, empty_group_fallback(
        report,
        "pair",
        fallback_policy="pulse_resolved_noncorrupt_out_of_bound_empty_pair_only",
        execution_detail="compact_endpoint_with_exact_empty_pair_fallback",
        endpoint_generation_policy=(
            "compact_covered_exact_out_of_bound_empty_pair_fallback"
        ),
    )
