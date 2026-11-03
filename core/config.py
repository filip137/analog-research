"""core.config — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import math
import re
from typing import Any, Mapping, Optional
from core.contracts import config_error


MAX_SEED = 2**31 - 1


_LIFECYCLE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


_ARM_ID = re.compile(r"[a-z0-9][a-z0-9_]*\Z")


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


# ---------------------------------------------------------------------------
# Strict primitive parsers. Messages state the expected format first.


def _object(value: Any, path: str, keys) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise config_error(path, "to be an object", value)
    if set(value) != set(keys):
        raise config_error(
            f"{path} fields", f"to be exactly {sorted(keys)}", sorted(value)
        )
    return value


def _choice(value: Any, path: str, choices) -> str:
    if not isinstance(value, str) or value not in choices:
        raise config_error(path, f"to be one of {list(choices)}", value)
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise config_error(path, "to be non-empty text without outer whitespace", value)
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise config_error(path, "to be a boolean", value)
    return value


def _integer(value: Any, path: str, *, minimum: int = 0, maximum: Optional[int] = None) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = f"at least {minimum}" + ("" if maximum is None else f" and at most {maximum}")
        raise config_error(path, f"to be an integer {bound}", value)
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (positive and value <= 0)
        or (maximum is not None and value > maximum)
    ):
        bound = ("positive" if positive else f"at least {minimum}") + (
            "" if maximum is None else f" and at most {maximum}"
        )
        raise config_error(path, f"to be a finite number {bound}", value)
    return float(value)


def _list(value: Any, path: str, *, nonempty: bool = True) -> list:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise config_error(path, "to be a " + ("non-empty " if nonempty else "") + "list", value)
    return list(value)


def _seeds(value: Any, path: str, *, nonempty: bool = True) -> tuple[int, ...]:
    seeds = tuple(
        _integer(item, f"{path}[{index}]", minimum=1, maximum=MAX_SEED)
        for index, item in enumerate(_list(value, path, nonempty=nonempty))
    )
    if len(set(seeds)) != len(seeds):
        raise config_error(path, "to contain unique seeds", list(seeds))
    return seeds


def _identifier(value: Any, path: str, pattern=_ARM_ID) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise config_error(path, f"to match {pattern.pattern!r}", value)
    return value
