"""Persistent noisy writes around a digital program-and-verify target.

This backend is intentionally distinct from pulsed in-situ training. BPTT
computes gradients at the currently realized device state. SGD applies those
gradients to a clean digital target, and every optimizer step then reprograms
the trainable dense arrays through the configured endpoint noise model.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from networks.resistive.builders import ParameterCatalog
from hardware.config import (
    DeviceProgrammingConfig,
    device_programming_to_mapping,
    parse_device_programming_config,
)
from networks.variable.parameter import ConvWeight, DenseWeight
from methods.device_programming import program_device_binding


_STATE_VERSION = 1


class ProgramVerifyOptimizer:
    """Wrap a digital optimizer with a persistent noisy-write boundary."""

    def __init__(
        self,
        optimizer: Any,
        catalog: ParameterCatalog,
        config: DeviceProgrammingConfig,
    ) -> None:
        self._optimizer = optimizer
        self._config = parse_device_programming_config(config)
        self._bindings = tuple(
            binding
            for binding in catalog.trainable
            if isinstance(binding.parameter, (DenseWeight, ConvWeight))
        )
        if not self._bindings:
            raise ValueError(
                "Expected program_verify to target at least one trainable "
                "dense or convolutional weight. Provided value: none."
            )
        self._shadows: dict[str, torch.Tensor] = {}
        self._generators: dict[str, torch.Generator] = {}
        self._pending_generator_states: dict[str, torch.Tensor] = {}
        self._initialized = False
        self._step_count = 0
        self._initial_report: dict[str, Any] | None = None
        self._last_write_report: dict[str, Any] | None = None

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def configuration(self) -> dict[str, Any]:
        return device_programming_to_mapping(self._config)

    @property
    def programming_report(self) -> dict[str, Any]:
        return {
            "semantics": "digital_shadow_target_reprogrammed_after_each_step",
            "pulse_model": False,
            "step_count": self._step_count,
            "initial": deepcopy(self._initial_report),
            "last_write": deepcopy(self._last_write_report),
        }

    def initialize_from_loaded_targets(self) -> dict[str, Any]:
        """Capture clean targets and perform the first physical write."""

        if self._initialized:
            raise RuntimeError(
                "Expected program_verify initialization exactly once. "
                "Provided value: optimizer is already initialized."
            )
        self._shadows = {
            binding.key: binding.state.detach().clone()
            for binding in self._bindings
        }
        reports = self._program_all()
        self._initial_report = _aggregate_reports(reports)
        self._last_write_report = deepcopy(self._initial_report)
        self._initialized = True
        return deepcopy(self._initial_report)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError(
                "Expected program_verify BPTT updates without an optimizer "
                f"closure. Provided value: {closure!r}."
            )
        if not self._initialized:
            raise RuntimeError(
                "Expected program_verify.initialize_from_loaded_targets() "
                "after checkpoint loading and before the first optimizer "
                "step. Provided value: uninitialized."
            )
        with torch.no_grad():
            for binding in self._bindings:
                binding.state.copy_(self._shadows[binding.key])
        result = self._optimizer.step()
        with torch.no_grad():
            for binding in self._bindings:
                binding.parameter.clamp_()
                self._shadows[binding.key].copy_(binding.state)
        reports = self._program_all()
        self._step_count += 1
        self._last_write_report = _aggregate_reports(reports)
        return result

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "configuration": self.configuration,
            "optimizer": self._optimizer.state_dict(),
            "initialized": self._initialized,
            "step_count": self._step_count,
            "shadows": {
                key: value.detach().cpu().clone()
                for key, value in self._shadows.items()
            },
            "generator_states": self._generator_states(),
            "initial_report": deepcopy(self._initial_report),
            "last_write_report": deepcopy(self._last_write_report),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise ValueError(
                "Expected program_verify optimizer state to be an object. "
                f"Provided value: {state_dict!r}."
            )
        expected = {
            "version",
            "configuration",
            "optimizer",
            "initialized",
            "step_count",
            "shadows",
            "generator_states",
            "initial_report",
            "last_write_report",
        }
        if set(state_dict) != expected:
            raise ValueError(
                "Expected program_verify optimizer state to contain exactly "
                f"{sorted(expected)!r}. Provided value: "
                f"{sorted(state_dict, key=repr)!r}."
            )
        if state_dict["version"] != _STATE_VERSION:
            raise ValueError(
                "Expected program_verify optimizer state version to equal "
                f"{_STATE_VERSION}. Provided value: "
                f"{state_dict['version']!r}."
            )
        saved_config = parse_device_programming_config(
            state_dict["configuration"],
            path="program_verify state.configuration",
        )
        if device_programming_to_mapping(saved_config) != self.configuration:
            raise ValueError(
                "Expected program_verify checkpoint configuration to match "
                "the current configuration. Provided value: "
                f"saved={device_programming_to_mapping(saved_config)!r}, "
                f"current={self.configuration!r}."
            )
        initialized = state_dict["initialized"]
        step_count = state_dict["step_count"]
        if not isinstance(initialized, bool):
            raise ValueError(
                "Expected program_verify state.initialized to be a boolean. "
                f"Provided value: {initialized!r}."
            )
        if (
            isinstance(step_count, bool)
            or not isinstance(step_count, int)
            or step_count < 0
        ):
            raise ValueError(
                "Expected program_verify state.step_count to be a "
                f"non-negative integer. Provided value: {step_count!r}."
            )
        raw_shadows = state_dict["shadows"]
        expected_keys = tuple(binding.key for binding in self._bindings)
        if not isinstance(raw_shadows, Mapping) or set(raw_shadows) != (
            set(expected_keys) if initialized else set()
        ):
            raise ValueError(
                "Expected program_verify shadow keys to match trainable "
                f"device arrays {expected_keys!r}. Provided value: "
                f"{raw_shadows!r}."
            )
        staged_shadows: dict[str, torch.Tensor] = {}
        for binding in self._bindings:
            if not initialized:
                break
            value = raw_shadows[binding.key]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(binding.state.shape)
                or value.dtype != binding.state.dtype
            ):
                raise ValueError(
                    "Expected each program_verify shadow to match its live "
                    "tensor shape and dtype. Provided value: "
                    f"key={binding.key!r}, value={value!r}."
                )
            staged_shadows[binding.key] = value.to(
                device=binding.state.device
            ).clone()
        staged_generator_states = _validated_generator_states(
            state_dict["generator_states"]
        )

        self._optimizer.load_state_dict(deepcopy(state_dict["optimizer"]))
        self._initialized = initialized
        self._step_count = step_count
        self._shadows = staged_shadows
        self._generators = {}
        self._pending_generator_states = staged_generator_states
        self._initial_report = deepcopy(state_dict["initial_report"])
        self._last_write_report = deepcopy(
            state_dict["last_write_report"]
        )

    def _program_all(self) -> list[dict[str, Any]]:
        reports = []
        for binding in self._bindings:
            reports.append(
                program_device_binding(
                    binding,
                    self._config,
                    generator=self._generator(binding.state.device),
                )
            )
        return reports

    def _generator(self, device: torch.device) -> torch.Generator:
        key = _device_key(device)
        generator = self._generators.get(key)
        if generator is not None:
            return generator
        generator = torch.Generator(device=torch.device(key))
        generator.manual_seed(int(self._config.programming_seed))
        pending = self._pending_generator_states.pop(key, None)
        if pending is not None:
            generator.set_state(pending)
        self._generators[key] = generator
        return generator

    def _generator_states(self) -> dict[str, torch.Tensor]:
        states = {
            key: value.detach().cpu().clone()
            for key, value in self._pending_generator_states.items()
        }
        states.update(
            {
                key: generator.get_state().detach().cpu().clone()
                for key, generator in self._generators.items()
            }
        )
        return dict(sorted(states.items()))


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(report["count"]) for report in reports)

    def aggregate_rmse(name: str) -> float | None:
        if not count or any(report.get(name) is None for report in reports):
            return None
        return (
            sum(
                float(report[name]) ** 2 * int(report["count"])
                for report in reports
            )
            / count
        ) ** 0.5

    def maximum(name: str) -> float | None:
        values = [report.get(name) for report in reports]
        if any(value is None for value in values):
            return None
        return max(float(value) for value in values)

    return {
        "model": reports[0]["model"],
        "parameter_keys": [report["parameter_key"] for report in reports],
        "count": count,
        "error_rmse": aggregate_rmse("error_rmse"),
        "error_abs_max": max(
            float(report["error_abs_max"]) for report in reports
        ),
        "error_reference": "realized_effective_drn_minus_clean_digital_target",
        "mapping_error_rmse": aggregate_rmse("mapping_error_rmse"),
        "mapping_error_abs_max": maximum("mapping_error_abs_max"),
        "programming_error_rmse": aggregate_rmse(
            "programming_error_rmse"
        ),
        "programming_error_abs_max": maximum(
            "programming_error_abs_max"
        ),
        "programming_error_reference": (
            "realized_effective_drn_minus_ideal_mapped_target"
        ),
        "parameters": reports,
    }


def _device_key(device: torch.device) -> str:
    concrete = torch.device(device)
    if concrete.type == "cuda" and concrete.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return str(concrete)


def _validated_generator_states(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "Expected program_verify generator_states to be an object. "
            f"Provided value: {value!r}."
        )
    staged = {}
    for key, state in value.items():
        try:
            device = torch.device(key)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                "Expected program_verify generator state keys to be "
                f"canonical device strings. Provided value: {key!r}."
            ) from error
        if str(device) != key or (
            device.type == "cuda" and device.index is None
        ):
            raise ValueError(
                "Expected program_verify generator state keys to be "
                f"canonical concrete device strings. Provided value: {key!r}."
            )
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.device.type != "cpu"
        ):
            raise ValueError(
                "Expected program_verify generator states to be "
                "one-dimensional CPU uint8 tensors. "
                f"Provided value: key={key!r}, state={state!r}."
            )
        probe = torch.Generator(device=device)
        probe.set_state(state)
        staged[key] = state.detach().clone()
    return staged


__all__ = ["ProgramVerifyOptimizer"]
