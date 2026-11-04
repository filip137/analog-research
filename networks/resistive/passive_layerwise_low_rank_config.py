"""Configuration for passive low-rank branches on both edges of a DRN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from networks.resistive.digital_low_rank_config import (
    DigitalLowRankAdapterConfig,
    parse_digital_low_rank_adapter,
)
from hardware.config import (
    DeviceProgrammingConfig,
    device_programming_to_mapping,
)
from networks.resistive.low_rank_config import (
    PassiveLowRankAdapterConfig,
    parse_passive_low_rank_adapter,
)


_DEFAULT_PATH = "passive_layerwise_low_rank_adapter"
_EXPECTED_PARAMETER_KEYS = (
    "base.dense_weight.0",
    "base.dense_weight.1",
)
_FACTOR_KEYS = {
    "rank",
    "input_factor_gain",
    "input_factor_min",
    "conductance_max",
    "output_factor_init",
    "output_off_conductance",
}


@dataclass(frozen=True)
class PassiveLayerLowRankConfig:
    """One physical rank branch and its independently programmed base edge."""

    parameter_key: str
    rank: int
    input_factor_gain: float
    input_factor_min: float
    conductance_max: float
    output_factor_init: str
    output_off_conductance: float | None
    device_noise: DeviceProgrammingConfig | None

    @property
    def factor_config(self) -> PassiveLowRankAdapterConfig:
        return PassiveLowRankAdapterConfig(
            rank=self.rank,
            input_factor_gain=self.input_factor_gain,
            input_factor_min=self.input_factor_min,
            conductance_max=self.conductance_max,
            output_factor_init=self.output_factor_init,
            output_off_conductance=self.output_off_conductance,
        )


@dataclass(frozen=True)
class PassiveLayerwiseLowRankAdapterConfig:
    """Two physical branches attached across the two base dense edges."""

    layers: tuple[PassiveLayerLowRankConfig, ...]

    @property
    def by_parameter_key(self) -> dict[str, PassiveLayerLowRankConfig]:
        return {layer.parameter_key: layer for layer in self.layers}


def parse_passive_layerwise_low_rank_adapter(
    value: PassiveLayerwiseLowRankAdapterConfig | Mapping[str, Any] | None,
    *,
    path: str = _DEFAULT_PATH,
) -> PassiveLayerwiseLowRankAdapterConfig | None:
    """Normalize a strict stable-keyed two-edge passive adapter."""

    if value is None:
        return None
    if isinstance(value, PassiveLayerwiseLowRankAdapterConfig):
        _validate_config(value, path=path)
        return _normalized(value)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Expected {path} to be null, a "
            "PassiveLayerwiseLowRankAdapterConfig, or an object containing "
            f"exactly a 'layers' object. Provided value: {value!r}."
        )
    if set(value) != {"layers"}:
        _raise_exact_keys(value, expected=("layers",), path=path)
    layers_value = value["layers"]
    layers_path = f"{path}.layers"
    if not isinstance(layers_value, Mapping):
        raise ValueError(
            f"Expected {layers_path} to be an object keyed by base dense "
            f"parameter name. Provided value: {layers_value!r}."
        )
    if set(layers_value) != set(_EXPECTED_PARAMETER_KEYS):
        _raise_exact_keys(
            layers_value,
            expected=_EXPECTED_PARAMETER_KEYS,
            path=layers_path,
        )

    layers = []
    for parameter_key in _EXPECTED_PARAMETER_KEYS:
        layer_path = f"{layers_path}.{parameter_key}"
        layer_value = layers_value[parameter_key]
        if not isinstance(layer_value, Mapping):
            raise ValueError(
                f"Expected {layer_path} to be an object containing factor "
                f"and device-noise settings. Provided value: {layer_value!r}."
            )
        allowed = _FACTOR_KEYS | {"device_noise"}
        required = (allowed - {"output_off_conductance"})
        unknown = sorted(
            (key for key in layer_value if key not in allowed),
            key=repr,
        )
        missing = sorted(required - set(layer_value))
        if unknown or missing:
            raise ValueError(
                f"Expected {layer_path} keys to be drawn from "
                f"{', '.join(sorted(allowed))} and to include "
                f"{', '.join(sorted(required))}. Provided value: "
                f"missing={missing!r}, unknown={unknown!r} in "
                f"{layer_value!r}."
            )
        factor = parse_passive_low_rank_adapter(
            {
                key: layer_value[key]
                for key in _FACTOR_KEYS
                if key in layer_value
            },
            path=layer_path,
        )
        if factor is None:  # pragma: no cover - mapping is non-null
            raise AssertionError("passive layerwise factor resolved to null")
        noise = _parse_noise(
            layer_value["device_noise"],
            path=layer_path,
        )
        layers.append(_layer_config(parameter_key, factor, noise))

    config = PassiveLayerwiseLowRankAdapterConfig(layers=tuple(layers))
    _validate_config(config, path=path)
    return _normalized(config)


def passive_layerwise_low_rank_to_mapping(
    config: PassiveLayerwiseLowRankAdapterConfig,
) -> dict[str, Any]:
    """Return the canonical strict-schema representation."""

    normalized = _normalized(config)
    return {
        "layers": {
            layer.parameter_key: {
                "rank": layer.rank,
                "input_factor_gain": layer.input_factor_gain,
                "input_factor_min": layer.input_factor_min,
                "conductance_max": layer.conductance_max,
                "output_factor_init": layer.output_factor_init,
                "output_off_conductance": layer.output_off_conductance,
                "device_noise": (
                    None
                    if layer.device_noise is None
                    else device_programming_to_mapping(layer.device_noise)
                ),
            }
            for layer in normalized.layers
        }
    }


def _parse_noise(
    value: Any,
    *,
    path: str,
) -> DeviceProgrammingConfig | None:
    if value is None:
        return None
    wrapper = (
        DigitalLowRankAdapterConfig(
            rank=1,
            alpha=1.0,
            input_factor_gain=1.0,
            device_noise=value,
        )
        if not isinstance(value, Mapping)
        else {
            "rank": 1,
            "alpha": 1.0,
            "input_factor_gain": 1.0,
            "device_noise": value,
        }
    )
    parsed = parse_digital_low_rank_adapter(wrapper, path=path)
    if parsed is None:  # pragma: no cover - mapping is non-null
        raise AssertionError("device configuration resolved to null")
    return parsed.device_noise


def _layer_config(
    parameter_key: str,
    factor: PassiveLowRankAdapterConfig,
    noise: DeviceProgrammingConfig | None,
) -> PassiveLayerLowRankConfig:
    return PassiveLayerLowRankConfig(
        parameter_key=parameter_key,
        rank=factor.rank,
        input_factor_gain=factor.input_factor_gain,
        input_factor_min=factor.input_factor_min,
        conductance_max=factor.conductance_max,
        output_factor_init=factor.output_factor_init,
        output_off_conductance=factor.output_off_conductance,
        device_noise=noise,
    )


def _raise_exact_keys(
    value: Mapping[str, Any],
    *,
    expected: tuple[str, ...],
    path: str,
) -> None:
    missing = sorted(set(expected) - set(value))
    unknown = sorted((key for key in value if key not in expected), key=repr)
    raise ValueError(
        f"Expected {path} to contain exactly the keys: "
        f"{', '.join(expected)}. Provided value: missing={missing!r}, "
        f"unknown={unknown!r} in {value!r}."
    )


def _validate_config(
    config: PassiveLayerwiseLowRankAdapterConfig,
    *,
    path: str,
) -> None:
    if not isinstance(config.layers, tuple):
        raise ValueError(
            f"Expected {path}.layers to be a tuple in normalized "
            f"configuration. Provided value: {config.layers!r}."
        )
    keys = tuple(layer.parameter_key for layer in config.layers)
    if keys != _EXPECTED_PARAMETER_KEYS:
        raise ValueError(
            f"Expected {path}.layers to configure {_EXPECTED_PARAMETER_KEYS!r} "
            f"in stable order. Provided value: {keys!r}."
        )
    for layer in config.layers:
        parse_passive_low_rank_adapter(
            layer.factor_config,
            path=f"{path}.layers.{layer.parameter_key}",
        )
        _parse_noise(
            layer.device_noise,
            path=f"{path}.layers.{layer.parameter_key}",
        )


def _normalized(
    config: PassiveLayerwiseLowRankAdapterConfig,
) -> PassiveLayerwiseLowRankAdapterConfig:
    layers = []
    for layer in config.layers:
        factor = parse_passive_low_rank_adapter(layer.factor_config)
        if factor is None:  # pragma: no cover - dataclass is non-null
            raise AssertionError("passive layerwise factor resolved to null")
        noise = _parse_noise(layer.device_noise, path=_DEFAULT_PATH)
        layers.append(_layer_config(layer.parameter_key, factor, noise))
    return PassiveLayerwiseLowRankAdapterConfig(layers=tuple(layers))


__all__ = [
    "PassiveLayerLowRankConfig",
    "PassiveLayerwiseLowRankAdapterConfig",
    "parse_passive_layerwise_low_rank_adapter",
    "passive_layerwise_low_rank_to_mapping",
]
