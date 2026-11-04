"""Strict physical layout declarations, independent of numerical imports."""
from collections.abc import Mapping, Sequence

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
