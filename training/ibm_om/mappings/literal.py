"""Literal global mapping: each cell is programmed to its own fraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from training.ibm_om.mappings.base import (
    MappingArguments,
    MappingInputs,
    reject_out_of_support,
)
from training.ibm_om.programming.base import OutOfSupportContract


NAME = "literal_global"


def validate_arguments(arguments: MappingArguments) -> MappingArguments:
    layouts = arguments.layouts
    margin = arguments.margin
    if layouts is not None or margin != 0.0:
        raise ValueError(
            "Expected literal_global target mapping to have null layouts and "
            "zero common-window margin."
        )

    return arguments


def map_targets(inputs: MappingInputs) -> tuple[torch.Tensor, dict[str, Any]]:
    global_targets = inputs.global_targets
    support_counts = inputs.support_counts
    report = inputs.report
    mapped = global_targets.detach().clone()
    report.update(
        {
            "common_window_grouping": None,
            "common_window_group_size": None,
            "common_window_group_count": 0,
            "common_window_empty_group_count": 0,
            "corrupt_group_count": 0,
            "published_corrupt_group_count": 0,
            "differential_pair_binding": None,
            "nonempty_inner_common_span": None,
            "quad_count": 0,
            "common_window_empty_quad_count": 0,
            "corrupt_quad_count": 0,
            "published_corrupt_quad_count": 0,
            "raw_common_span": None,
            "inner_common_span": None,
            "mapped_target_below_lower_bound_nonempty_quad": 0,
            "mapped_target_above_upper_bound_nonempty_quad": 0,
            "mapped_target_below_lower_bound_empty_quad": 0,
            "mapped_target_above_upper_bound_empty_quad": 0,
            "mapped_target_below_lower_bound_nonempty_group": 0,
            "mapped_target_above_upper_bound_nonempty_group": 0,
            "mapped_target_below_lower_bound_empty_group": 0,
            "mapped_target_above_upper_bound_empty_group": 0,
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
    """Literal mapping has no preflight contract."""


def programming_contract(
    arguments: MappingArguments,
    report: dict[str, Any],
) -> tuple[torch.Tensor | None, OutOfSupportContract]:
    """Literal targets are programmed as-is; compact sampling fails closed."""

    return None, reject_out_of_support()
