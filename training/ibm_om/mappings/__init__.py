"""Target mappings: intended DRN fractions -> per-cell IBM OM targets.

Each mapping module owns its argument checks, its mapping and its
preflight contract.  ``MAPPINGS`` selects one by name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from training.ibm_om.characterization import IbmReramResetCommissioning
from training.ibm_om.constants import (
    _MAX_EMPTY_COMMON_WINDOW_PAIRS,
    _MAX_EMPTY_COMMON_WINDOW_QUADS,
    _TARGET_MAPPINGS,
)
from training.ibm_om.mappings import (
    literal,
    pair_window,
    quad_window,
    raw_active,
    reset_relative,
)
from training.ibm_om.mappings.base import MappingArguments, prepare_inputs
from training.ibm_om.population import IbmReramArrayPopulation
from training.ibm_om.topology import _normalize_dual_rail_layouts


MAPPINGS = {
    strategy.NAME: strategy
    for strategy in (literal, quad_window, pair_window, reset_relative, raw_active)
}


def map_ibm_reram_array_targets(
    global_targets: torch.Tensor,
    population: IbmReramArrayPopulation,
    *,
    target_mapping: str,
    dual_rail_layout_by_parameter: (
        Mapping[str, str] | Sequence[Sequence[str]] | None
    ),
    common_window_margin_fraction: float,
    reset_commissioning: IbmReramResetCommissioning | None = None,
    reset_relative_mode: str | None = None,
    reset_relative_contrast_step: float | None = None,
    raw_active_mode: str | None = None,
    raw_active_unsupported_quad_policy: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Map clean global fractions onto one fixed IBM OM array assignment.

    The quad mapping uses the exact four physical identities assigned to the
    Cartesian product of a logical synapse's two source and two destination
    rails.  Corrupt identities are deliberately retained in the intersection;
    the published arm therefore keeps its collapsed/stuck-cell semantics,
    while the repaired arm naturally uses its sampled donor bounds.
    """

    if (
        not isinstance(global_targets, torch.Tensor)
        or not global_targets.is_floating_point()
        or global_targets.ndim != 1
        or global_targets.numel() != population.size
        or not bool(torch.all(torch.isfinite(global_targets)))
    ):
        raise ValueError(
            "Expected finite floating global_targets with shape "
            f"({population.size},). Provided value: "
            f"shape={getattr(global_targets, 'shape', None)!r}."
        )
    if target_mapping not in _TARGET_MAPPINGS:
        raise ValueError(
            f"Expected target_mapping to be one of {_TARGET_MAPPINGS!r}. "
            f"Provided value: {target_mapping!r}."
        )
    layouts = _normalize_dual_rail_layouts(dual_rail_layout_by_parameter)
    margin = common_window_margin_fraction
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or not 0.0 <= float(margin) < 0.5
    ):
        raise ValueError(
            "Expected common_window_margin_fraction to be finite in "
            f"[0, 0.5). Provided value: {margin!r}."
        )
    margin = float(margin)
    differential_pairs: tuple[
        tuple[int, str, str, tuple[int, int]], ...
    ] = ()
    strategy = MAPPINGS[target_mapping]
    arguments = strategy.validate_arguments(
        MappingArguments(
            global_targets=global_targets,
            population=population,
            target_mapping=target_mapping,
            layouts=layouts,
            margin=margin,
            reset_commissioning=reset_commissioning,
            reset_relative_mode=reset_relative_mode,
            reset_relative_contrast_step=reset_relative_contrast_step,
            raw_active_mode=raw_active_mode,
            raw_active_unsupported_quad_policy=raw_active_unsupported_quad_policy,
            differential_pairs=differential_pairs,
        )
    )
    return strategy.map_targets(prepare_inputs(arguments))


def validate_ibm_reram_target_mapping_preflight(
    report: Mapping[str, Any],
    *,
    maximum_empty_quads: int = _MAX_EMPTY_COMMON_WINDOW_QUADS,
    maximum_empty_pairs: int = _MAX_EMPTY_COMMON_WINDOW_PAIRS,
) -> None:
    """Fail closed on unsupported nonempty common-window groups."""

    target_mapping = report.get("target_mapping")
    if target_mapping not in {
        "dual_rail_quad_common_window",
        "differential_pair_common_window",
        "shared_reset_relative_quad",
        "raw_active_p90_quad",
    }:
        return
    MAPPINGS[target_mapping].check_preflight(
        report,
        maximum_empty_quads=maximum_empty_quads,
        maximum_empty_pairs=maximum_empty_pairs,
    )
