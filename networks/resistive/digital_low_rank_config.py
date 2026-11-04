"""Torch-free configuration for an ideal digital low-rank DRN readout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from numbers import Integral, Real
from typing import Any

from hardware.config import (
    DeviceProgrammingConfig,
    Wan2022ProgrammingConfig,
    parse_device_programming_config,
)


_DEFAULT_PATH = "digital_low_rank_adapter"


def _finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and isfinite(float(value))
    )


@dataclass(frozen=True)
class DigitalLowRankAdapterConfig:
    """Configuration for a signed FP32 residual on settled DRN logits."""

    rank: int
    alpha: float
    input_factor_gain: float
    device_noise: DeviceProgrammingConfig


def parse_digital_low_rank_adapter(
    value: DigitalLowRankAdapterConfig | Mapping[str, Any] | None,
    *,
    path: str = _DEFAULT_PATH,
) -> DigitalLowRankAdapterConfig | None:
    """Normalize one strict digital adapter object."""

    if value is None:
        return None
    if isinstance(value, DigitalLowRankAdapterConfig):
        _validate(value, path=path)
        return _normalized(value)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Expected {path} to be null, a DigitalLowRankAdapterConfig, or "
            "an object containing the adapter configuration. "
            f"Provided value: {value!r}."
        )

    allowed = {field.name for field in fields(DigitalLowRankAdapterConfig)}
    _check_exact_keys(value, allowed=allowed, path=path)
    noise_path = f"{path}.device_noise"
    noise_value = value["device_noise"]
    if not isinstance(noise_value, Mapping):
        raise ValueError(
            f"Expected {noise_path} to be an object. "
            f"Provided value: {noise_value!r}."
        )
    parsed_noise = parse_device_programming_config(
        noise_value,
        path=noise_path,
    )
    try:
        parsed = DigitalLowRankAdapterConfig(
            rank=value["rank"],
            alpha=value["alpha"],
            input_factor_gain=value["input_factor_gain"],
            device_noise=parsed_noise,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Expected {path} to contain a valid digital low-rank adapter. "
            f"Provided value: {value!r}."
        ) from error
    _validate(parsed, path=path)
    return _normalized(parsed)


def _check_exact_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    path: str,
) -> None:
    unknown = sorted((key for key in value if key not in allowed), key=repr)
    missing = sorted(allowed - set(value))
    if unknown or missing:
        raise ValueError(
            f"Expected {path} to contain exactly the keys: "
            f"{', '.join(sorted(allowed))}. Provided value: "
            f"missing={missing!r}, unknown={unknown!r} in {value!r}."
        )


def _validate(config: DigitalLowRankAdapterConfig, *, path: str) -> None:
    if (
        isinstance(config.rank, bool)
        or not isinstance(config.rank, Integral)
        or config.rank <= 0
    ):
        raise ValueError(
            f"Expected {path}.rank to be a positive integer. "
            f"Provided value: {config.rank!r}."
        )
    for name in ("alpha", "input_factor_gain"):
        value = getattr(config, name)
        if not _finite_real(value) or value <= 0.0:
            raise ValueError(
                f"Expected {path}.{name} to be a positive finite number. "
                f"Provided value: {value!r}."
            )

    noise_path = f"{path}.device_noise"
    parse_device_programming_config(config.device_noise, path=noise_path)


def _normalized(
    config: DigitalLowRankAdapterConfig,
) -> DigitalLowRankAdapterConfig:
    return DigitalLowRankAdapterConfig(
        rank=int(config.rank),
        alpha=float(config.alpha),
        input_factor_gain=float(config.input_factor_gain),
        device_noise=parse_device_programming_config(
            config.device_noise,
            path=f"{_DEFAULT_PATH}.device_noise",
        ),
    )


__all__ = [
    "DigitalLowRankAdapterConfig",
    "DeviceProgrammingConfig",
    "Wan2022ProgrammingConfig",
    "parse_digital_low_rank_adapter",
]
