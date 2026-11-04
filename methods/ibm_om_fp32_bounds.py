"""Ideal FP32 training inside one sampled IBM OM array's cell bounds.

This module intentionally uses hidden per-cell OM bounds.  It is an
array-specific oracle reference, not a deployable controller: weights remain
ordinary FP32 tensors, updates are digital, and no pulse, read-noise, or
program-and-verify model is applied.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
from typing import Any

import torch

from networks.resistive.builders import ParameterBinding, ParameterCatalog
from networks.variable.parameter import DenseWeight
from methods.ibm_om.api import (
    IbmReramArrayPopulation,
    load_om_array_population,
)


_STATE_SCHEMA = "ibm-om.fp32-cell-bounds-optimizer"
_STATE_VERSION = 1
_REPORT_SCHEMA = "ibm-om.fp32-cell-bounds-report"
_REPORT_VERSION = 1
_SHA256_LENGTH = 64


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    staged = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(staged.dtype).encode("utf-8"))
    digest.update(str(tuple(staged.shape)).encode("utf-8"))
    digest.update(staged.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _summary(value: torch.Tensor) -> dict[str, float | int]:
    staged = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if staged.numel() < 1 or not bool(torch.all(torch.isfinite(staged))):
        raise ValueError("Expected a non-empty finite tensor for OM bounds summary.")
    quantiles = torch.quantile(
        staged,
        torch.tensor(
            [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
            dtype=torch.float64,
        ),
    )
    names = ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99")
    return {
        "count": int(staged.numel()),
        "minimum": float(staged.min().item()),
        "maximum": float(staged.max().item()),
        "mean": float(staged.mean().item()),
        "std_population": float(staged.std(unbiased=False).item()),
        **{
            name: float(item)
            for name, item in zip(names, quantiles.tolist(), strict=True)
        },
    }


def effective_controller_bounds(
    population: IbmReramArrayPopulation,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Intersect each sampled OM cell range with the forward [0, 1] range."""

    raw_lower = (population.logical_min + 1.0) / 2.0
    raw_upper = (population.logical_max + 1.0) / 2.0
    lower = raw_lower.clamp(0.0, 1.0)
    upper = raw_upper.clamp(0.0, 1.0)
    if bool(torch.any(lower > upper)):
        raise ValueError(
            "Expected every clipped OM lower bound not to exceed its upper bound."
        )
    clipping = {
        "raw_lower_below_zero": int((raw_lower < 0.0).sum().item()),
        "raw_lower_above_one": int((raw_lower > 1.0).sum().item()),
        "raw_upper_below_zero": int((raw_upper < 0.0).sum().item()),
        "raw_upper_above_one": int((raw_upper > 1.0).sum().item()),
        "zero_span_after_clipping": int((lower == upper).sum().item()),
    }
    return lower, upper, clipping


def _dense_bindings(catalog: ParameterCatalog) -> tuple[ParameterBinding, ...]:
    bindings = tuple(
        binding
        for binding in catalog.trainable
        if isinstance(binding.parameter, DenseWeight)
    )
    if not bindings:
        raise ValueError(
            "Expected IBM OM FP32 bounds training to have trainable DenseWeight "
            "bindings. Provided value: none."
        )
    return bindings


def _validate_digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Expected {name} to be a lowercase SHA-256 digest.")
    return value


class IbmOmFp32BoundsOptimizer:
    """Wrap SGD/Adam with deterministic per-cell initialization/projection."""

    def __init__(
        self,
        optimizer: Any,
        catalog: ParameterCatalog,
        parameters: Mapping[str, Any],
        population_path: Path,
    ) -> None:
        self._optimizer = optimizer
        self._configuration = _plain(parameters)
        self._bindings = _dense_bindings(catalog)
        self._population_path = Path(population_path).expanduser().resolve()
        if not self._population_path.is_file():
            raise FileNotFoundError(
                "Expected --device-data to name an IBM OM population artifact. "
                f"Provided value: {str(self._population_path)!r}."
            )
        self._population_sha256 = _sha256_file(self._population_path)
        expected_sha256 = _validate_digest(
            parameters.get("expected_population_sha256"),
            name="expected_population_sha256",
        )
        if self._population_sha256 != expected_sha256:
            raise ValueError(
                "Expected IBM OM population SHA-256 to match the frozen "
                "configuration. Provided value: "
                f"expected={expected_sha256!r}, actual={self._population_sha256!r}."
            )

        self._population = load_om_array_population(self._population_path)
        self._validate_population(parameters)
        flat_lower, flat_upper, self._clipping = effective_controller_bounds(
            self._population
        )
        self._flat_lower = flat_lower.detach().cpu().clone()
        self._flat_upper = flat_upper.detach().cpu().clone()
        self._lower: dict[str, torch.Tensor] = {}
        self._upper: dict[str, torch.Tensor] = {}
        self._unique_lower: dict[str, torch.Tensor] = {}
        self._unique_upper: dict[str, torch.Tensor] = {}
        self._lower_projection_count = {binding.key: 0 for binding in self._bindings}
        self._upper_projection_count = {binding.key: 0 for binding in self._bindings}
        self._step_count = 0

        offset = 0
        for binding, shape in zip(
            self._bindings,
            self._population.binding_shapes,
            strict=True,
        ):
            count = math.prod(shape)
            lower = flat_lower[offset : offset + count].reshape(shape)
            upper = flat_upper[offset : offset + count].reshape(shape)
            self._lower[binding.key] = lower.to(
                device=binding.state.device,
                dtype=binding.state.dtype,
            )
            self._upper[binding.key] = upper.to(
                device=binding.state.device,
                dtype=binding.state.dtype,
            )
            self._unique_lower[binding.key] = torch.zeros(
                shape, dtype=torch.bool, device=binding.state.device
            )
            self._unique_upper[binding.key] = torch.zeros(
                shape, dtype=torch.bool, device=binding.state.device
            )
            offset += count
        if offset != self._population.size:  # pragma: no cover - validated layout
            raise AssertionError("OM population split did not consume every cell.")

        self._initialize_weights(int(parameters["initialization_seed"]))
        self._initial_report = self._weight_report()

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def configuration(self) -> dict[str, Any]:
        return deepcopy(self._configuration)

    def _validate_population(self, parameters: Mapping[str, Any]) -> None:
        exact = {
            "preset": "reram_array_om",
            "bounds_coordinate": "controller_clipped_0_1",
            "initialization_distribution": "uniform_per_cell_effective_bounds",
        }
        for name, expected in exact.items():
            if parameters.get(name) != expected:
                raise ValueError(
                    f"Expected IBM OM FP32 bounds {name} to equal {expected!r}. "
                    f"Provided value: {parameters.get(name)!r}."
                )
        expected_fingerprint = _validate_digest(
            parameters.get("expected_population_fingerprint"),
            name="expected_population_fingerprint",
        )
        if self._population.fingerprint != expected_fingerprint:
            raise ValueError(
                "Expected IBM OM population fingerprint to match the frozen "
                "configuration. Provided value: "
                f"expected={expected_fingerprint!r}, "
                f"actual={self._population.fingerprint!r}."
            )
        expected_assignment = int(parameters["assignment_seed"])
        if self._population.assignment_seed != expected_assignment:
            raise ValueError(
                "Expected IBM OM population assignment seed to match the "
                "configuration. Provided value: "
                f"expected={expected_assignment}, "
                f"actual={self._population.assignment_seed}."
            )
        expected_policy = str(parameters["corruption_policy"])
        if self._population.corruption_policy != expected_policy:
            raise ValueError(
                "Expected IBM OM corruption policy to match the configuration. "
                f"Provided value: expected={expected_policy!r}, "
                f"actual={self._population.corruption_policy!r}."
            )
        keys = tuple(binding.key for binding in self._bindings)
        shapes = tuple(tuple(binding.state.shape) for binding in self._bindings)
        if (
            keys != self._population.binding_keys
            or shapes != self._population.binding_shapes
        ):
            raise ValueError(
                "Expected the trainable dense binding layout to match the IBM "
                "OM population exactly. Provided value: "
                f"bindings={tuple(zip(keys, shapes, strict=True))!r}, "
                "population="
                f"{tuple(zip(self._population.binding_keys, self._population.binding_shapes, strict=True))!r}."
            )

    def _initialize_weights(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        with torch.no_grad():
            for binding in self._bindings:
                lower = self._lower[binding.key]
                upper = self._upper[binding.key]
                random = torch.rand(
                    tuple(binding.state.shape),
                    generator=generator,
                    device="cpu",
                    dtype=torch.float32,
                ).to(device=binding.state.device, dtype=binding.state.dtype)
                binding.state.copy_(lower + (upper - lower) * random)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError(
                "Expected IBM OM FP32 bounds training without an optimizer "
                f"closure. Provided value: {closure!r}."
            )
        result = self._optimizer.step()
        for binding in self._bindings:
            state = binding.state
            lower = self._lower[binding.key]
            upper = self._upper[binding.key]
            below = state < lower
            above = state > upper
            self._lower_projection_count[binding.key] += int(below.sum().item())
            self._upper_projection_count[binding.key] += int(above.sum().item())
            self._unique_lower[binding.key].logical_or_(below)
            self._unique_upper[binding.key].logical_or_(above)
            state.copy_(torch.maximum(torch.minimum(state, upper), lower))
        self._step_count += 1
        return result

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def _weight_report(self) -> dict[str, Any]:
        flat = torch.cat(
            [binding.state.detach().cpu().reshape(-1) for binding in self._bindings]
        )
        parameters = {}
        for binding in self._bindings:
            state = binding.state.detach().cpu()
            lower = self._lower[binding.key].detach().cpu()
            upper = self._upper[binding.key].detach().cpu()
            span = upper - lower
            relative = torch.where(span > 0.0, (state - lower) / span, torch.zeros_like(span))
            parameters[binding.key] = {
                "shape": list(state.shape),
                "sha256": _tensor_sha256(state),
                "values": _summary(state),
                "relative_position": _summary(relative),
                "at_lower_bound": int((state == lower).sum().item()),
                "at_upper_bound": int((state == upper).sum().item()),
            }
        return {
            "sha256": _tensor_sha256(flat),
            "values": _summary(flat),
            "parameters": parameters,
        }

    @property
    def bounds_report(self) -> dict[str, Any]:
        span = self._flat_upper - self._flat_lower
        projection_parameters = {}
        for binding in self._bindings:
            projection_parameters[binding.key] = {
                "lower_events": self._lower_projection_count[binding.key],
                "upper_events": self._upper_projection_count[binding.key],
                "unique_lower_cells": int(
                    self._unique_lower[binding.key].sum().item()
                ),
                "unique_upper_cells": int(
                    self._unique_upper[binding.key].sum().item()
                ),
            }
        return {
            "schema": _REPORT_SCHEMA,
            "schema_version": _REPORT_VERSION,
            "semantics": "array_specific_hidden_bounds_fp32_oracle",
            "pulse_model": False,
            "read_noise": False,
            "program_verify": False,
            "quantization": False,
            "hidden_bounds_used": True,
            "configuration": self.configuration,
            "population": {
                "path": str(self._population_path),
                "sha256": self._population_sha256,
                "fingerprint": self._population.fingerprint,
                "assignment_seed": self._population.assignment_seed,
                "corruption_policy": self._population.corruption_policy,
                "aihwkit_version": self._population.aihwkit_version,
                "cells": self._population.size,
            },
            "effective_bounds": {
                "coordinate": "controller_clipped_0_1",
                "lower_sha256": _tensor_sha256(self._flat_lower),
                "upper_sha256": _tensor_sha256(self._flat_upper),
                "lower": _summary(self._flat_lower),
                "upper": _summary(self._flat_upper),
                "span": _summary(span),
                "clipping": dict(self._clipping),
            },
            "initial_weights": deepcopy(self._initial_report),
            "terminal_weights": self._weight_report(),
            "projection": {
                "step_count": self._step_count,
                "parameters": projection_parameters,
                "lower_events": sum(self._lower_projection_count.values()),
                "upper_events": sum(self._upper_projection_count.values()),
                "unique_lower_cells": sum(
                    int(mask.sum().item()) for mask in self._unique_lower.values()
                ),
                "unique_upper_cells": sum(
                    int(mask.sum().item()) for mask in self._unique_upper.values()
                ),
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": _STATE_SCHEMA,
            "schema_version": _STATE_VERSION,
            "configuration": self.configuration,
            "population_sha256": self._population_sha256,
            "population_fingerprint": self._population.fingerprint,
            "optimizer": self._optimizer.state_dict(),
            "step_count": self._step_count,
            "lower_projection_count": dict(self._lower_projection_count),
            "upper_projection_count": dict(self._upper_projection_count),
            "unique_lower": {
                key: value.detach().cpu().clone()
                for key, value in self._unique_lower.items()
            },
            "unique_upper": {
                key: value.detach().cpu().clone()
                for key, value in self._unique_upper.items()
            },
            "initial_report": deepcopy(self._initial_report),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise ValueError(
                "Expected IBM OM FP32 bounds optimizer state to be an object."
            )
        expected = {
            "schema",
            "schema_version",
            "configuration",
            "population_sha256",
            "population_fingerprint",
            "optimizer",
            "step_count",
            "lower_projection_count",
            "upper_projection_count",
            "unique_lower",
            "unique_upper",
            "initial_report",
        }
        if set(state_dict) != expected:
            raise ValueError(
                "Expected exact IBM OM FP32 bounds optimizer state fields. "
                f"Provided value: {sorted(state_dict, key=repr)!r}."
            )
        if (
            state_dict["schema"] != _STATE_SCHEMA
            or state_dict["schema_version"] != _STATE_VERSION
            or dict(state_dict["configuration"]) != self._configuration
            or state_dict["population_sha256"] != self._population_sha256
            or state_dict["population_fingerprint"] != self._population.fingerprint
            or state_dict["initial_report"] != self._initial_report
        ):
            raise ValueError(
                "Expected resumed IBM OM FP32 bounds provenance to match the "
                "current runtime exactly."
            )
        keys = tuple(binding.key for binding in self._bindings)
        for name in ("lower_projection_count", "upper_projection_count", "unique_lower", "unique_upper"):
            value = state_dict[name]
            if not isinstance(value, Mapping) or set(value) != set(keys):
                raise ValueError(
                    f"Expected resumed {name} to contain exactly {keys!r}."
                )
        step_count = state_dict["step_count"]
        if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
            raise ValueError("Expected resumed step_count to be a non-negative integer.")
        lower_counts: dict[str, int] = {}
        upper_counts: dict[str, int] = {}
        lower_masks: dict[str, torch.Tensor] = {}
        upper_masks: dict[str, torch.Tensor] = {}
        for binding in self._bindings:
            key = binding.key
            for name, output in (
                ("lower_projection_count", lower_counts),
                ("upper_projection_count", upper_counts),
            ):
                count = state_dict[name][key]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"Expected resumed {name}[{key!r}] to be non-negative.")
                output[key] = count
            for name, output in (("unique_lower", lower_masks), ("unique_upper", upper_masks)):
                mask = state_dict[name][key]
                if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(binding.state.shape):
                    raise ValueError(
                        f"Expected resumed {name}[{key!r}] to be a matching boolean tensor."
                    )
                output[key] = mask.detach().to(binding.state.device).clone()

        self._optimizer.load_state_dict(deepcopy(state_dict["optimizer"]))
        self._step_count = step_count
        self._lower_projection_count = lower_counts
        self._upper_projection_count = upper_counts
        self._unique_lower = lower_masks
        self._unique_upper = upper_masks

    def __str__(self) -> str:
        return (
            "IBM OM per-cell-bounded FP32 oracle -- "
            f"optimizer={self._optimizer}, assignment_seed="
            f"{self._population.assignment_seed}, steps={self._step_count}"
        )


__all__ = [
    "IbmOmFp32BoundsOptimizer",
    "effective_controller_bounds",
]
