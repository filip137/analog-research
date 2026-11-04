"""Torch-free configuration for measured program-and-verify device models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite
from numbers import Integral, Real
from typing import Any, TypeAlias


WAN2022 = "aihwkit_reram_wan2022"
WAN2022_PHYSICAL = "aihwkit_reram_wan2022_physical"
IBM_AFM2025_PCM = "ibm_afm2025_pcm"
AIHWKIT_RERAM_CMO = "aihwkit_reram_cmo"
DEVICE_TYPES = (
    WAN2022,
    WAN2022_PHYSICAL,
    IBM_AFM2025_PCM,
    AIHWKIT_RERAM_CMO,
)
_WAN_TIMES_SECONDS = (1.0, 86400.0, 172800.0)


@dataclass(frozen=True)
class Wan2022ProgrammingConfig:
    """Physical mapping and one fixed Wan-2022 ReRAM realization."""

    type: str
    programming_seed: int
    g_max_us: float
    drn_conductance_at_g_max: float
    noise_scale: float
    t_inference_seconds: float


@dataclass(frozen=True)
class Wan2022PhysicalProgrammingConfig:
    """Wan-2022 endpoint model with an explicit physical floor mapping."""

    type: str
    programming_seed: int
    g_min_us: float
    g_max_us: float
    drn_conductance_at_g_max: float
    noise_scale: float
    t_inference_seconds: float
    mapping: str


@dataclass(frozen=True)
class IbmAfm2025PcmProgrammingConfig:
    """Programming-noise fit released with IBM Analog Foundation Models."""

    type: str
    programming_seed: int
    noise_scale: float
    fit_max: float
    zero_threshold: float
    max_input_size: int


@dataclass(frozen=True)
class CmoHfOxProgrammingConfig:
    """AIHWKit CMO/HfOx program, relaxation, and read-noise model."""

    type: str
    programming_seed: int
    g_min_us: float
    g_max_us: float
    drn_conductance_at_g_max: float
    noise_scale: float
    acceptance_range_percent: float
    t_inference_seconds: float
    read_noise_scale: float
    drift_scale: float
    mapping: str


DeviceProgrammingConfig: TypeAlias = (
    Wan2022ProgrammingConfig
    | Wan2022PhysicalProgrammingConfig
    | IbmAfm2025PcmProgrammingConfig
    | CmoHfOxProgrammingConfig
)


def parse_device_programming_config(
    value: DeviceProgrammingConfig | Mapping[str, Any],
    *,
    path: str = "device_programming",
) -> DeviceProgrammingConfig:
    """Parse one strict tagged device-programming configuration."""

    if isinstance(
        value,
        (
            Wan2022ProgrammingConfig,
            Wan2022PhysicalProgrammingConfig,
            IbmAfm2025PcmProgrammingConfig,
            CmoHfOxProgrammingConfig,
        ),
    ):
        config = value
    else:
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Expected {path} to be a device-programming object. "
                f"Provided value: {value!r}."
            )
        device_type = value.get("type")
        classes = {
            WAN2022: Wan2022ProgrammingConfig,
            WAN2022_PHYSICAL: Wan2022PhysicalProgrammingConfig,
            IBM_AFM2025_PCM: IbmAfm2025PcmProgrammingConfig,
            AIHWKIT_RERAM_CMO: CmoHfOxProgrammingConfig,
        }
        config_type = classes.get(device_type)
        if config_type is None:
            raise ValueError(
                f"Expected {path}.type to be one of {DEVICE_TYPES!r}. "
                f"Provided value: {device_type!r}."
            )
        allowed = {field.name for field in fields(config_type)}
        _check_exact_keys(value, allowed=allowed, path=path)
        try:
            config = config_type(
                **{name: value[name] for name in allowed}
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Expected {path} to contain valid device settings. "
                f"Provided value: {value!r}."
            ) from error
    _validate_device_config(config, path=path)
    return _normalize_device_config(config)


def device_programming_to_mapping(
    config: DeviceProgrammingConfig,
) -> dict[str, Any]:
    return asdict(parse_device_programming_config(config))


def _validate_device_config(
    config: DeviceProgrammingConfig,
    *,
    path: str,
) -> None:
    _non_negative_integer(
        config.programming_seed,
        name=f"{path}.programming_seed",
    )
    if config.type == WAN2022:
        if not isinstance(config, Wan2022ProgrammingConfig):
            _wrong_dataclass(config, path)
        _positive(config.g_max_us, name=f"{path}.g_max_us")
        _positive(
            config.drn_conductance_at_g_max,
            name=f"{path}.drn_conductance_at_g_max",
        )
        _non_negative(config.noise_scale, name=f"{path}.noise_scale")
        if (
            not _finite_real(config.t_inference_seconds)
            or float(config.t_inference_seconds) not in _WAN_TIMES_SECONDS
        ):
            raise ValueError(
                f"Expected {path}.t_inference_seconds to be one of "
                f"{_WAN_TIMES_SECONDS!r}. Provided value: "
                f"{config.t_inference_seconds!r}."
            )
        return
    if config.type == WAN2022_PHYSICAL:
        if not isinstance(config, Wan2022PhysicalProgrammingConfig):
            _wrong_dataclass(config, path)
        for name in (
            "g_min_us",
            "g_max_us",
            "drn_conductance_at_g_max",
        ):
            _positive(getattr(config, name), name=f"{path}.{name}")
        if float(config.g_min_us) >= float(config.g_max_us):
            raise ValueError(
                f"Expected {path}.g_min_us < {path}.g_max_us. "
                f"Provided value: g_min_us={config.g_min_us!r}, "
                f"g_max_us={config.g_max_us!r}."
            )
        _non_negative(config.noise_scale, name=f"{path}.noise_scale")
        if (
            not _finite_real(config.t_inference_seconds)
            or float(config.t_inference_seconds) not in _WAN_TIMES_SECONDS
        ):
            raise ValueError(
                f"Expected {path}.t_inference_seconds to be one of "
                f"{_WAN_TIMES_SECONDS!r}. Provided value: "
                f"{config.t_inference_seconds!r}."
            )
        if config.mapping not in (
            "literal_conductance",
            "affine_floor",
            "normalized_offset",
        ):
            raise ValueError(
                f"Expected {path}.mapping to be 'literal_conductance', "
                "'affine_floor', or 'normalized_offset'. Provided value: "
                f"{config.mapping!r}."
            )
        return
    if config.type == IBM_AFM2025_PCM:
        if not isinstance(config, IbmAfm2025PcmProgrammingConfig):
            _wrong_dataclass(config, path)
        _non_negative(config.noise_scale, name=f"{path}.noise_scale")
        if float(config.fit_max) != 180.0:
            raise ValueError(
                f"Expected {path}.fit_max to equal the released unitless fit "
                f"value 180.0. Provided value: {config.fit_max!r}."
            )
        if float(config.zero_threshold) != 0.6:
            raise ValueError(
                f"Expected {path}.zero_threshold to equal the released "
                f"unitless fit value 0.6. Provided value: "
                f"{config.zero_threshold!r}."
            )
        if (
            isinstance(config.max_input_size, bool)
            or not isinstance(config.max_input_size, Integral)
            or int(config.max_input_size) == 0
            or int(config.max_input_size) < -1
        ):
            raise ValueError(
                f"Expected {path}.max_input_size to be -1 or a positive "
                f"integer. Provided value: {config.max_input_size!r}."
            )
        return
    if config.type == AIHWKIT_RERAM_CMO:
        if not isinstance(config, CmoHfOxProgrammingConfig):
            _wrong_dataclass(config, path)
        for name in (
            "g_min_us",
            "g_max_us",
            "drn_conductance_at_g_max",
        ):
            _positive(getattr(config, name), name=f"{path}.{name}")
        if float(config.g_min_us) >= float(config.g_max_us):
            raise ValueError(
                f"Expected {path}.g_min_us < {path}.g_max_us. "
                f"Provided value: g_min_us={config.g_min_us!r}, "
                f"g_max_us={config.g_max_us!r}."
            )
        for name in ("noise_scale", "read_noise_scale", "drift_scale"):
            _non_negative(getattr(config, name), name=f"{path}.{name}")
        if float(config.acceptance_range_percent) not in (0.2, 2.0):
            raise ValueError(
                f"Expected {path}.acceptance_range_percent to be 0.2 or "
                f"2.0. Provided value: "
                f"{config.acceptance_range_percent!r}."
            )
        _non_negative(
            config.t_inference_seconds,
            name=f"{path}.t_inference_seconds",
        )
        if config.mapping not in (
            "literal_conductance",
            "affine_floor",
            "normalized_offset",
        ):
            raise ValueError(
                f"Expected {path}.mapping to be 'literal_conductance', "
                "'affine_floor', or 'normalized_offset'. Provided value: "
                f"{config.mapping!r}."
            )
        return
    raise ValueError(
        f"Expected {path}.type to be one of {DEVICE_TYPES!r}. "
        f"Provided value: {config.type!r}."
    )


def _normalize_device_config(
    config: DeviceProgrammingConfig,
) -> DeviceProgrammingConfig:
    values = asdict(config)
    values["programming_seed"] = int(config.programming_seed)
    for key, value in tuple(values.items()):
        if key not in {"type", "programming_seed", "max_input_size", "mapping"}:
            values[key] = float(value)
    if "max_input_size" in values:
        values["max_input_size"] = int(values["max_input_size"])
    return type(config)(**values)


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


def _wrong_dataclass(config: Any, path: str) -> None:
    raise ValueError(
        f"Expected {path} type tag to match its configuration class. "
        f"Provided value: {config!r}."
    )


def _finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and isfinite(float(value))
    )


def _positive(value: Any, *, name: str) -> None:
    if not _finite_real(value) or float(value) <= 0.0:
        raise ValueError(
            f"Expected {name} to be a positive finite number. "
            f"Provided value: {value!r}."
        )


def _non_negative(value: Any, *, name: str) -> None:
    if not _finite_real(value) or float(value) < 0.0:
        raise ValueError(
            f"Expected {name} to be a non-negative finite number. "
            f"Provided value: {value!r}."
        )


def _non_negative_integer(value: Any, *, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 0
    ):
        raise ValueError(
            f"Expected {name} to be a non-negative integer. "
            f"Provided value: {value!r}."
        )


__all__ = [
    "AIHWKIT_RERAM_CMO",
    "DEVICE_TYPES",
    "IBM_AFM2025_PCM",
    "WAN2022",
    "WAN2022_PHYSICAL",
    "CmoHfOxProgrammingConfig",
    "DeviceProgrammingConfig",
    "IbmAfm2025PcmProgrammingConfig",
    "Wan2022ProgrammingConfig",
    "Wan2022PhysicalProgrammingConfig",
    "device_programming_to_mapping",
    "parse_device_programming_config",
]
