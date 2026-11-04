"""Torch-free configuration contract for passive low-rank adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from numbers import Integral, Real
from typing import Any


_DEFAULT_PATH = "passive_low_rank_adapter"
_OUTPUT_FACTOR_INITS = ("zero", "off_conductance")


def _is_finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and isfinite(float(value))
    )


@dataclass(frozen=True)
class PassiveLowRankAdapterConfig:
    """Configuration for the passive ``input -> rank -> output`` branch."""

    rank: int
    input_factor_gain: float
    input_factor_min: float
    conductance_max: float
    output_factor_init: str
    output_off_conductance: float | None = None

    def __post_init__(self) -> None:
        _validate_config(self, path=_DEFAULT_PATH)


def _validate_config(
    config: PassiveLowRankAdapterConfig,
    *,
    path: str,
) -> None:
    if (
        isinstance(config.rank, bool)
        or not isinstance(config.rank, Integral)
        or config.rank <= 0
    ):
        raise ValueError(
            f"Expected {path}.rank to be a positive integer. "
            f"Provided value: {config.rank!r}."
        )

    for name in (
        "input_factor_gain",
        "input_factor_min",
        "conductance_max",
    ):
        value = getattr(config, name)
        if not _is_finite_real(value) or value <= 0.0:
            raise ValueError(
                f"Expected {path}.{name} to be a positive finite number. "
                f"Provided value: {value!r}."
            )

    if config.input_factor_min >= config.conductance_max:
        raise ValueError(
            f"Expected {path}.input_factor_min to be strictly less than "
            f"{path}.conductance_max. Provided value: "
            f"input_factor_min={config.input_factor_min!r}, "
            f"conductance_max={config.conductance_max!r}."
        )

    if config.output_factor_init not in _OUTPUT_FACTOR_INITS:
        raise ValueError(
            f"Expected {path}.output_factor_init to be 'zero' or "
            f"'off_conductance'. Provided value: "
            f"{config.output_factor_init!r}."
        )

    if config.output_factor_init == "zero":
        if config.output_off_conductance is not None:
            raise ValueError(
                f"Expected {path}.output_off_conductance to be null or "
                "omitted when output_factor_init='zero'. "
                f"Provided value: {config.output_off_conductance!r}."
            )
        return

    value = config.output_off_conductance
    if not _is_finite_real(value) or value <= 0.0:
        raise ValueError(
            f"Expected {path}.output_off_conductance to be a positive finite "
            "number when output_factor_init='off_conductance'. "
            f"Provided value: {value!r}."
        )
    if value >= config.conductance_max:
        raise ValueError(
            f"Expected {path}.output_off_conductance to be strictly less "
            f"than {path}.conductance_max. Provided value: "
            f"output_off_conductance={value!r}, "
            f"conductance_max={config.conductance_max!r}."
        )


def parse_passive_low_rank_adapter(
    value: PassiveLowRankAdapterConfig | Mapping[str, Any] | None,
    *,
    path: str = _DEFAULT_PATH,
) -> PassiveLowRankAdapterConfig | None:
    """Normalize one strict adapter object to its typed representation."""

    if value is None:
        return None
    if isinstance(value, PassiveLowRankAdapterConfig):
        _validate_config(value, path=path)
        return _normalized_config(value)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Expected {path} to be null, a PassiveLowRankAdapterConfig, or "
            "an object containing the adapter configuration. "
            f"Provided value: {value!r}."
        )

    allowed = {field.name for field in fields(PassiveLowRankAdapterConfig)}
    required = allowed - {"output_off_conductance"}
    unknown = sorted(
        (key for key in value if key not in allowed),
        key=repr,
    )
    if unknown:
        raise ValueError(
            f"Expected {path} keys to be drawn from: "
            f"{', '.join(sorted(allowed))}. Provided value: unknown keys "
            f"{unknown!r} in {value!r}."
        )
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            f"Expected {path} to contain keys: "
            f"{', '.join(sorted(required))}. Provided value: missing keys "
            f"{missing!r} in {value!r}."
        )

    try:
        parsed = PassiveLowRankAdapterConfig(
            **{name: value[name] for name in allowed if name in value}
        )
    except ValueError as error:
        if path == _DEFAULT_PATH:
            raise
        raise ValueError(
            str(error).replace(_DEFAULT_PATH, path)
        ) from error
    return _normalized_config(parsed)


def _normalized_config(
    config: PassiveLowRankAdapterConfig,
) -> PassiveLowRankAdapterConfig:
    return PassiveLowRankAdapterConfig(
        rank=int(config.rank),
        input_factor_gain=float(config.input_factor_gain),
        input_factor_min=float(config.input_factor_min),
        conductance_max=float(config.conductance_max),
        output_factor_init=config.output_factor_init,
        output_off_conductance=(
            None
            if config.output_off_conductance is None
            else float(config.output_off_conductance)
        ),
    )


__all__ = [
    "PassiveLowRankAdapterConfig",
    "parse_passive_low_rank_adapter",
]
