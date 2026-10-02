"""DRN topology seen by the IBM OM layers: dual-rail quads and G+/G- pairs.

A logical signed synapse of a dual-rail DRN tensor occupies the four physical
cells at rows ``(r, r + rows/2)`` and the plus/minus columns given by the
tensor's layout (``halves`` or ``paired``).  A differential DRN instead uses
adjacent ``base.conductance_plus.i`` / ``base.conductance_minus.i`` tensors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch

from model.resistive.builders import ParameterBinding
from model.variable.parameter import DenseWeight


def _normalize_dual_rail_layouts(
    value: Mapping[str, str] | Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    items = value.items() if isinstance(value, Mapping) else value
    try:
        normalized = tuple(sorted((str(item[0]), str(item[1])) for item in items))
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(
            "Expected dual_rail_layout_by_parameter to be null or a mapping "
            "from stable parameter keys to 'halves' or 'paired'."
        ) from error
    if (
        not normalized
        or len({key for key, _layout in normalized}) != len(normalized)
        or any(not key or layout not in {"halves", "paired"} for key, layout in normalized)
    ):
        raise ValueError(
            "Expected dual_rail_layout_by_parameter to map unique non-empty "
            "parameter keys to 'halves' or 'paired'. Provided value: "
            f"{normalized!r}."
        )
    return normalized



def _quad_axes(
    shape: tuple[int, ...],
    layout: str,
    *,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    if len(shape) != 2 or shape[0] % 2 or shape[1] % 2:
        raise ValueError(
            "Expected four-cell dual-rail bindings to be even-by-even "
            f"rank-2 tensors. Provided shape: {shape!r}."
        )
    if layout not in {"halves", "paired"}:
        raise ValueError(
            "Expected a four-cell dual-rail layout of 'halves' or 'paired'. "
            f"Provided value: {layout!r}."
        )
    input_count = shape[0] // 2
    output_count = shape[1] // 2
    plus_rows = torch.arange(input_count, device=device)
    minus_rows = plus_rows + input_count
    if layout == "halves":
        plus_columns = torch.arange(output_count, device=device)
        minus_columns = plus_columns + output_count
    else:
        plus_columns = torch.arange(output_count, device=device) * 2
        minus_columns = plus_columns + 1
    return (plus_rows, minus_rows), (plus_columns, minus_columns)



def _canonical_differential_pair_layout(
    binding_keys: Sequence[str],
    binding_shapes: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, str, str, tuple[int, int]], ...]:
    """Validate the canonical adjacent G+/G- catalog and return its pairs."""

    keys = tuple(binding_keys)
    shapes = tuple(tuple(shape) for shape in binding_shapes)
    if not keys or len(keys) != len(shapes) or len(keys) % 2:
        raise ValueError(
            "Expected differential-pair common-window bindings to be a "
            "non-empty even-length canonical adjacent plus/minus catalog. "
            f"Provided value: keys={keys!r}, shapes={shapes!r}."
        )
    pairs = []
    for pair_index in range(len(keys) // 2):
        plus_position = 2 * pair_index
        minus_position = plus_position + 1
        plus_key = keys[plus_position]
        minus_key = keys[minus_position]
        expected_plus = f"base.conductance_plus.{pair_index}"
        expected_minus = f"base.conductance_minus.{pair_index}"
        plus_shape = shapes[plus_position]
        minus_shape = shapes[minus_position]
        if plus_key != expected_plus or minus_key != expected_minus:
            raise ValueError(
                "Expected differential-pair common-window bindings in "
                "canonical adjacent plus/minus key and suffix order. "
                f"Provided pair {pair_index}: keys="
                f"({plus_key!r}, {minus_key!r}); expected="
                f"({expected_plus!r}, {expected_minus!r})."
            )
        if (
            len(plus_shape) != 2
            or any(value < 1 for value in plus_shape)
            or minus_shape != plus_shape
        ):
            raise ValueError(
                "Expected each canonical differential plus/minus pair to "
                "have identical non-empty rank-2 shapes. Provided pair "
                f"{pair_index}: plus_shape={plus_shape!r}, "
                f"minus_shape={minus_shape!r}."
            )
        pairs.append(
            (
                pair_index,
                plus_key,
                minus_key,
                (int(plus_shape[0]), int(plus_shape[1])),
            )
        )
    return tuple(pairs)



def _validate_differential_pair_bindings(
    bindings: Sequence[ParameterBinding],
) -> tuple[tuple[int, str, str, tuple[int, int]], ...]:
    """Validate roles and tensors in addition to the stable catalog layout."""

    selected = tuple(bindings)
    pairs = _canonical_differential_pair_layout(
        tuple(binding.key for binding in selected),
        tuple(tuple(binding.state.shape) for binding in selected),
    )
    for pair_index, _plus_key, _minus_key, _shape in pairs:
        plus = selected[2 * pair_index]
        minus = selected[2 * pair_index + 1]
        if (
            plus.role != "conductance_plus"
            or minus.role != "conductance_minus"
            or not isinstance(plus.parameter, DenseWeight)
            or not isinstance(minus.parameter, DenseWeight)
        ):
            raise ValueError(
                "Expected canonical differential-pair bindings to expose "
                "DenseWeight conductance_plus/conductance_minus roles. "
                f"Provided pair {pair_index}: plus_role={plus.role!r}, "
                f"minus_role={minus.role!r}, "
                f"plus_type={type(plus.parameter).__name__!r}, "
                f"minus_type={type(minus.parameter).__name__!r}."
            )
        raw_bounds = (
            plus.parameter.min_cond,
            plus.parameter.max_cond,
            minus.parameter.min_cond,
            minus.parameter.max_cond,
        )
        try:
            plus_min, plus_max, minus_min, minus_max = tuple(
                float(value) for value in raw_bounds
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Expected each canonical differential plus/minus pair to "
                "have finite, increasing, exactly matching conductance "
                f"bounds. Provided pair {pair_index}: bounds={raw_bounds!r}."
            ) from error
        if (
            not all(
                math.isfinite(value)
                for value in (plus_min, plus_max, minus_min, minus_max)
            )
            or not plus_min < plus_max
            or not minus_min < minus_max
            or plus_min != minus_min
            or plus_max != minus_max
        ):
            raise ValueError(
                "Expected each canonical differential plus/minus pair to "
                "have finite, increasing, exactly matching conductance "
                f"bounds. Provided pair {pair_index}: "
                f"plus_bounds=({plus_min!r}, {plus_max!r}), "
                f"minus_bounds=({minus_min!r}, {minus_max!r})."
            )
    return pairs
