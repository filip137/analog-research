"""Temporary additive-Gaussian perturbations for trainable DRN weights.

The modifier keeps every parameter tensor as a clean master weight. Entering
the training context samples one realization for a complete minibatch and
restores the clean value before the optimizer step.

``output_channel_abs_max`` follows AIHWKit-Lightning's
``ADD_NORMAL_PER_CHANNEL`` rule. Dense DRN tensors use the layout
``(pre, post)``, so the maximum is taken over dimension 0, leaving one scale
per post-synaptic/output channel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any, Iterator

import torch

from networks.variable.parameter import ConvWeight, DenseWeight


_STATE_DICT_VERSION = 1
_MAX_TORCH_SEED = 2**64 - 1
_EVALUATION_STREAM_XOR = 0x9E3779B97F4A7C15
_STREAMS = ("train", "evaluation")
_SCALE_MODES = ("tensor_abs_max", "output_channel_abs_max")


@dataclass(frozen=True)
class AddNormalConfig:
    """Configuration for temporary additive Gaussian weight noise."""

    std_dev: float
    seed: int | None = None
    noisy_evaluation: bool = False
    scale_mode: str = "tensor_abs_max"

    def __post_init__(self) -> None:
        std_dev = _validated_non_negative_real(
            "add_normal.std_dev",
            self.std_dev,
        )
        seed = _validated_optional_seed("add_normal.seed", self.seed)
        if not isinstance(self.noisy_evaluation, bool):
            raise ValueError(
                "Expected add_normal.noisy_evaluation to be a boolean. "
                f"Provided value: {self.noisy_evaluation!r}."
            )
        if self.scale_mode not in _SCALE_MODES:
            raise ValueError(
                "Expected add_normal.scale_mode to be 'tensor_abs_max' or "
                "'output_channel_abs_max'. "
                f"Provided value: {self.scale_mode!r}."
            )
        object.__setattr__(self, "std_dev", std_dev)
        object.__setattr__(self, "seed", seed)


class AddNormalParameterModifier:
    """Apply additive-normal noise through parameter-modifier contexts."""

    def __init__(
        self,
        parameters: Iterable[Any],
        config: AddNormalConfig,
        *,
        run_seed: int | None = None,
    ) -> None:
        if not isinstance(config, AddNormalConfig):
            raise ValueError(
                "Expected config to be an AddNormalConfig. "
                f"Provided value: {config!r}."
            )
        normalized_run_seed = _validated_optional_seed("run_seed", run_seed)
        eligible = []
        seen: set[int] = set()
        for parameter in parameters:
            if (
                isinstance(parameter, (DenseWeight, ConvWeight))
                and id(parameter) not in seen
            ):
                eligible.append(parameter)
                seen.add(id(parameter))

        if config.seed is not None:
            resolved_seed = config.seed
        elif normalized_run_seed is not None:
            resolved_seed = normalized_run_seed
        else:
            resolved_seed = int(torch.initial_seed())

        self._parameters = tuple(eligible)
        self._config = config
        self._resolved_seed = resolved_seed
        self._generators: dict[str, dict[str, torch.Generator]] = {
            stream: {} for stream in _STREAMS
        }
        self._pending_generator_states: dict[
            str, dict[str, torch.Tensor]
        ] = {stream: {} for stream in _STREAMS}
        self._active_stream: str | None = None

    @property
    def config(self) -> AddNormalConfig:
        return self._config

    @property
    def resolved_seed(self) -> int:
        return self._resolved_seed

    def training_context(self):
        return self._noise_context(
            "train",
            enabled=self._config.std_dev > 0.0,
        )

    def evaluation_context(self):
        return self._noise_context(
            "evaluation",
            enabled=(
                self._config.noisy_evaluation
                and self._config.std_dev > 0.0
            ),
        )

    def _stream_seed(self, stream: str) -> int:
        if stream == "train":
            return self._resolved_seed
        return self._resolved_seed ^ _EVALUATION_STREAM_XOR

    def _generator(
        self,
        stream: str,
        device: torch.device,
    ) -> torch.Generator:
        key = _device_key(device)
        generator = self._generators[stream].get(key)
        if generator is not None:
            return generator
        generator = torch.Generator(device=torch.device(key))
        generator.manual_seed(self._stream_seed(stream))
        pending = self._pending_generator_states[stream].pop(key, None)
        if pending is not None:
            generator.set_state(pending)
        self._generators[stream][key] = generator
        return generator

    @contextmanager
    def _noise_context(
        self,
        stream: str,
        *,
        enabled: bool,
    ) -> Iterator["AddNormalParameterModifier"]:
        if self._active_stream is not None:
            raise RuntimeError(
                "Expected add-normal modifier contexts not to overlap. "
                f"Provided value: active={self._active_stream!r}, "
                f"requested={stream!r}."
            )
        self._active_stream = stream
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            if enabled:
                with torch.no_grad():
                    for parameter in self._parameters:
                        state = parameter.state
                        clean = state.detach().clone()
                        snapshots.append((state, clean))
                        noise = torch.randn(
                            state.shape,
                            dtype=state.dtype,
                            device=state.device,
                            generator=self._generator(stream, state.device),
                        )
                        state.add_(
                            noise
                            * self._config.std_dev
                            * _noise_scale(clean, parameter, self._config)
                        )
                        parameter.clamp_()
            yield self
        finally:
            with torch.no_grad():
                for state, clean in snapshots:
                    state.copy_(clean)
            self._active_stream = None

    def state_dict(self) -> dict[str, Any]:
        if self._active_stream is not None:
            raise RuntimeError(
                "Expected state_dict outside an active add-normal context. "
                f"Provided value: active={self._active_stream!r}."
            )
        streams: dict[str, dict[str, torch.Tensor]] = {}
        for stream in _STREAMS:
            states = {
                key: value.detach().cpu().clone()
                for key, value in self._pending_generator_states[stream].items()
            }
            states.update(
                {
                    key: generator.get_state().detach().cpu().clone()
                    for key, generator in self._generators[stream].items()
                }
            )
            streams[stream] = dict(sorted(states.items()))
        return {
            "version": _STATE_DICT_VERSION,
            "config": asdict(self._config),
            "resolved_seed": self._resolved_seed,
            "generator_states": streams,
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        if self._active_stream is not None:
            raise RuntimeError(
                "Expected load_state_dict outside an active add-normal "
                f"context. Provided value: active={self._active_stream!r}."
            )
        if not isinstance(state_dict, Mapping):
            raise ValueError(
                "Expected add-normal state_dict to be an object. "
                f"Provided value: {state_dict!r}."
            )
        expected = {
            "version",
            "config",
            "resolved_seed",
            "generator_states",
        }
        if set(state_dict) != expected:
            raise ValueError(
                "Expected add-normal state_dict keys to be exactly config, "
                "generator_states, resolved_seed, and version. "
                f"Provided value: {sorted(state_dict, key=repr)!r}."
            )
        if state_dict["version"] != _STATE_DICT_VERSION:
            raise ValueError(
                "Expected add-normal state_dict.version to equal "
                f"{_STATE_DICT_VERSION}. "
                f"Provided value: {state_dict['version']!r}."
            )
        try:
            saved_config = asdict(AddNormalConfig(**state_dict["config"]))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Expected add-normal state_dict.config to be valid. "
                f"Provided value: {state_dict['config']!r}."
            ) from error
        if saved_config != asdict(self._config):
            raise ValueError(
                "Expected add-normal state_dict.config to match the current "
                "modifier configuration. "
                f"Provided value: saved={saved_config!r}, "
                f"current={asdict(self._config)!r}."
            )
        resolved_seed = _validated_required_seed(
            "add-normal state_dict.resolved_seed",
            state_dict["resolved_seed"],
        )
        raw_streams = state_dict["generator_states"]
        if not isinstance(raw_streams, Mapping) or set(raw_streams) != set(
            _STREAMS
        ):
            raise ValueError(
                "Expected add-normal state_dict.generator_states to contain "
                f"train and evaluation. Provided value: {raw_streams!r}."
            )
        staged: dict[str, dict[str, torch.Tensor]] = {
            stream: {} for stream in _STREAMS
        }
        for stream in _STREAMS:
            raw_states = raw_streams[stream]
            if not isinstance(raw_states, Mapping):
                raise ValueError(
                    "Expected each add-normal generator stream to be an "
                    f"object. Provided value: {raw_states!r}."
                )
            for key, value in raw_states.items():
                device = _validated_device_key(key)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype != torch.uint8
                    or value.ndim != 1
                    or value.device.type != "cpu"
                ):
                    raise ValueError(
                        "Expected each add-normal generator state to be a "
                        "one-dimensional CPU uint8 tensor. "
                        f"Provided value: stream={stream!r}, key={key!r}."
                    )
                probe = torch.Generator(device=device)
                probe.set_state(value)
                staged[stream][key] = value.detach().clone()
        self._resolved_seed = resolved_seed
        self._generators = {stream: {} for stream in _STREAMS}
        self._pending_generator_states = staged


def _noise_scale(
    clean: torch.Tensor,
    parameter: Any,
    config: AddNormalConfig,
) -> torch.Tensor:
    if config.scale_mode == "tensor_abs_max":
        return clean.abs().amax()
    if isinstance(parameter, DenseWeight):
        return clean.abs().amax(dim=0, keepdim=True)
    if isinstance(parameter, ConvWeight):
        reduce_dims = tuple(range(1, clean.ndim))
        return clean.abs().amax(dim=reduce_dims, keepdim=True)
    raise TypeError(
        "Expected output-channel scaling only for dense or convolutional "
        f"weights. Provided value: {parameter!r}."
    )


def build_add_normal_modifier(
    parameters: Iterable[Any],
    config: AddNormalConfig,
    *,
    run_seed: int | None = None,
) -> AddNormalParameterModifier | None:
    if not isinstance(config, AddNormalConfig):
        raise ValueError(
            "Expected config to be an AddNormalConfig. "
            f"Provided value: {config!r}."
        )
    if config.std_dev == 0.0:
        return None
    return AddNormalParameterModifier(parameters, config, run_seed=run_seed)


def _validated_non_negative_real(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"Expected {name} to be a finite number >= 0.0. "
            f"Provided value: {value!r}."
        )
    return float(value)


def _validated_optional_seed(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _validated_required_seed(name, value)


def _validated_required_seed(name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) <= _MAX_TORCH_SEED
    ):
        raise ValueError(
            f"Expected {name} to be an integer in [0, {_MAX_TORCH_SEED}]. "
            f"Provided value: {value!r}."
        )
    return int(value)


def _device_key(device: torch.device) -> str:
    concrete = torch.device(device)
    if concrete.type == "cuda" and concrete.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return str(concrete)


def _validated_device_key(value: Any) -> torch.device:
    if not isinstance(value, str):
        raise ValueError(
            "Expected a canonical device string. "
            f"Provided value: {value!r}."
        )
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "Expected a canonical device string. "
            f"Provided value: {value!r}."
        ) from error
    if str(device) != value or (
        device.type == "cuda" and device.index is None
    ):
        raise ValueError(
            "Expected a canonical concrete device string. "
            f"Provided value: {value!r}."
        )
    return device


__all__ = [
    "AddNormalConfig",
    "AddNormalParameterModifier",
    "build_add_normal_modifier",
]
