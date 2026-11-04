"""Measured cohort conductance projection for exploratory ReRAM training.

The optimizer keeps an ideal digital shadow for every trainable dense weight.
SGD updates that shadow, after which the live model tensor is replaced by the
globally nearest conductance on a deterministic virtual-device curve.  Each
virtual curve is a pointwise convex interpolation of two distinct measured
traces from the selected physical-device cohort.  Biases remain ordinary
digital SGD parameters.

The standard cohort optimizers are deliberately not sequential pulse models:
pulse indices are labels on the measured graph and every update may choose any
point on that graph. ``MeasuredCohortAOnePulseDownOptimizer`` is the explicit
exception; it traverses an isotonic trace locally by at most one pulse index.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from networks.resistive.builders import ParameterBinding, ParameterCatalog
from networks.variable.parameter import DenseWeight
from hardware.devices.trace_bank import MeasuredTraceBank


_STATE_VERSION = 4
_DEADBAND_STATE_VERSION = 3
_PREVIOUS_STATE_VERSION = 2
_LEGACY_STATE_VERSION = 1
_CACHE_CHUNK_CELLS = 1024


@dataclass(frozen=True)
class MeasuredTraceConfig:
    curve_preprocessing: str
    split_seed: int
    assignment_seed: int
    formed_resistance_max_ohm: float
    cohort_fraction: float
    cohort: str
    source_traces_per_cell: int
    initial_pulse_index: int
    projection: str
    expected_trace_length: int
    initial_target_mapping: str
    programming_deadband_mode: str
    programming_deadband_relative: float
    probabilistic_write_mode: str
    probabilistic_write_probability: float
    probabilistic_write_scale_relative: float
    probabilistic_write_seed: int
    dual_rail_layout_by_parameter: tuple[tuple[str, str], ...] | None
    positive_gradient_threshold_by_parameter: (
        tuple[tuple[str, float], ...] | None
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MeasuredTraceConfig":
        if not isinstance(value, Mapping):
            raise ValueError(
                "Expected measured-cohort parameters to be an object. "
                f"Provided value: {value!r}."
            )
        normalized = dict(value)
        normalized.setdefault("initial_target_mapping", "literal")
        normalized.setdefault("programming_deadband_mode", "none")
        normalized.setdefault("programming_deadband_relative", 0.0)
        normalized.setdefault("probabilistic_write_mode", "none")
        normalized.setdefault("probabilistic_write_probability", 1.0)
        normalized.setdefault("probabilistic_write_scale_relative", 0.0)
        normalized.setdefault("probabilistic_write_seed", 0)
        normalized.setdefault("dual_rail_layout_by_parameter", None)
        normalized.setdefault("positive_gradient_threshold_by_parameter", None)
        raw_layouts = normalized["dual_rail_layout_by_parameter"]
        if isinstance(raw_layouts, Mapping):
            normalized["dual_rail_layout_by_parameter"] = tuple(
                sorted(raw_layouts.items())
            )
        elif isinstance(raw_layouts, (list, tuple)):
            try:
                normalized["dual_rail_layout_by_parameter"] = tuple(
                    (item[0], item[1]) for item in raw_layouts
                )
            except (IndexError, TypeError) as error:
                raise ValueError(
                    "Expected dual_rail_layout_by_parameter to be null or "
                    "an object mapping stable parameter keys to 'halves' or "
                    f"'paired'. Provided value: {raw_layouts!r}."
                ) from error
        raw_thresholds = normalized[
            "positive_gradient_threshold_by_parameter"
        ]
        if isinstance(raw_thresholds, Mapping):
            normalized["positive_gradient_threshold_by_parameter"] = tuple(
                sorted(raw_thresholds.items())
            )
        elif isinstance(raw_thresholds, (list, tuple)):
            try:
                normalized["positive_gradient_threshold_by_parameter"] = tuple(
                    (item[0], item[1]) for item in raw_thresholds
                )
            except (IndexError, TypeError) as error:
                raise ValueError(
                    "Expected positive_gradient_threshold_by_parameter to be "
                    "null or an object mapping stable parameter keys to "
                    "finite non-negative raw-gradient thresholds. Provided "
                    f"value: {raw_thresholds!r}."
                ) from error
        expected = set(cls.__dataclass_fields__)
        if set(normalized) != expected:
            raise ValueError(
                "Expected measured-cohort parameters to contain exactly "
                f"{sorted(expected)!r}. Provided value: "
                f"{sorted(value, key=repr)!r}."
            )
        config = cls(**normalized)
        if config.curve_preprocessing not in {
            "raw",
            "isotonic_nonincreasing",
        }:
            raise ValueError(
                "Expected curve_preprocessing to be 'raw' or "
                "'isotonic_nonincreasing'. Provided value: "
                f"{config.curve_preprocessing!r}."
            )
        for name in ("split_seed", "assignment_seed", "initial_pulse_index"):
            item = getattr(config, name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(
                    f"Expected {name} to be a non-negative integer. "
                    f"Provided value: {item!r}."
                )
        if (
            isinstance(config.expected_trace_length, bool)
            or not isinstance(config.expected_trace_length, int)
            or config.expected_trace_length < 2
        ):
            raise ValueError(
                "Expected expected_trace_length to be an integer >= 2. "
                f"Provided value: {config.expected_trace_length!r}."
            )
        if (
            not math.isfinite(float(config.formed_resistance_max_ohm))
            or config.formed_resistance_max_ohm <= 0.0
        ):
            raise ValueError(
                "Expected formed_resistance_max_ohm to be positive and "
                f"finite. Provided value: {config.formed_resistance_max_ohm!r}."
            )
        if config.cohort not in {"A", "B"}:
            raise ValueError(
                "Expected cohort to be 'A' or 'B'. Provided value: "
                f"{config.cohort!r}."
            )
        exact = {
            "cohort_fraction": 0.5,
            "source_traces_per_cell": 2,
            "projection": "global_nearest",
        }
        for name, expected_value in exact.items():
            if getattr(config, name) != expected_value:
                raise ValueError(
                    f"Expected {name} to equal {expected_value!r} for the "
                    "measured two-cohort protocol. Provided value: "
                    f"{getattr(config, name)!r}."
                )
        if config.initial_pulse_index >= config.expected_trace_length:
            raise ValueError(
                "Expected initial_pulse_index to be smaller than "
                "expected_trace_length. Provided value: "
                f"initial_pulse_index={config.initial_pulse_index!r}, "
                f"expected_trace_length={config.expected_trace_length!r}."
            )
        if config.initial_target_mapping not in {
            "literal",
            "per_device_affine",
            "paired_affine_common_window",
            "dual_rail_pairwise_common_window",
            "dual_rail_quad_common_window",
        }:
            raise ValueError(
                "Expected initial_target_mapping to be 'literal', "
                "'per_device_affine', 'paired_affine_common_window', or "
                "a dual-rail common-window mapping "
                "('dual_rail_pairwise_common_window' or "
                "'dual_rail_quad_common_window'). "
                f"Provided value: {config.initial_target_mapping!r}."
            )
        if (
            config.cohort == "B"
            and config.initial_target_mapping
            in {
                "per_device_affine",
                "dual_rail_pairwise_common_window",
            }
        ):
            raise ValueError(
                "Expected cohort-B initial_target_mapping to be 'literal' "
                "or a matched common-window mapping "
                "('paired_affine_common_window' or "
                "'dual_rail_quad_common_window'). Provided value: "
                f"{config.initial_target_mapping!r}."
            )
        layouts = config.dual_rail_layout_by_parameter
        if config.initial_target_mapping in {
            "dual_rail_pairwise_common_window",
            "dual_rail_quad_common_window",
        }:
            if (
                not layouts
                or len({key for key, _layout in layouts}) != len(layouts)
                or any(
                    not isinstance(key, str)
                    or layout not in {"halves", "paired"}
                    for key, layout in layouts
                )
            ):
                raise ValueError(
                    "Expected dual_rail_layout_by_parameter to map unique "
                    "stable parameter keys to 'halves' or 'paired' when "
                    "initial_target_mapping is a dual-rail common-window "
                    "mapping. Provided value: "
                    f"{layouts!r}."
                )
        elif layouts is not None:
            raise ValueError(
                "Expected dual_rail_layout_by_parameter to be null unless "
                "initial_target_mapping is a dual-rail common-window "
                "mapping. Provided value: "
                f"{layouts!r}."
            )
        thresholds = config.positive_gradient_threshold_by_parameter
        if thresholds is not None and (
            len({key for key, _threshold in thresholds}) != len(thresholds)
            or any(
                not isinstance(key, str)
                or not key
                or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
                or float(threshold) < 0.0
                for key, threshold in thresholds
            )
        ):
            raise ValueError(
                "Expected positive_gradient_threshold_by_parameter to be "
                "null or map unique non-empty stable parameter keys to "
                "finite non-negative raw-gradient thresholds. Provided "
                f"value: {thresholds!r}."
            )
        if config.programming_deadband_mode not in {
            "none",
            "accumulated_shadow_relative_rms",
        }:
            raise ValueError(
                "Expected programming_deadband_mode to be 'none' or "
                "'accumulated_shadow_relative_rms'. Provided value: "
                f"{config.programming_deadband_mode!r}."
            )
        deadband_relative = config.programming_deadband_relative
        if (
            isinstance(deadband_relative, bool)
            or not isinstance(deadband_relative, (int, float))
            or not math.isfinite(float(deadband_relative))
            or float(deadband_relative) < 0.0
        ):
            raise ValueError(
                "Expected programming_deadband_relative to be finite and "
                f"non-negative. Provided value: {deadband_relative!r}."
            )
        if (
            config.programming_deadband_mode == "none"
            and float(deadband_relative) != 0.0
        ):
            raise ValueError(
                "Expected programming_deadband_relative to equal 0.0 when "
                "programming_deadband_mode is 'none'. Provided value: "
                f"{deadband_relative!r}."
            )
        if (
            config.programming_deadband_mode
            == "accumulated_shadow_relative_rms"
            and (config.cohort != "B" or float(deadband_relative) <= 0.0)
        ):
            raise ValueError(
                "Expected accumulated_shadow_relative_rms deadband to select "
                "cohort B and a positive programming_deadband_relative. "
                "Provided value: "
                f"cohort={config.cohort!r}, "
                f"programming_deadband_relative={deadband_relative!r}."
            )
        if config.probabilistic_write_mode not in {
            "none",
            "uniform_bernoulli",
            "displacement_proportional",
        }:
            raise ValueError(
                "Expected probabilistic_write_mode to be 'none', "
                "'uniform_bernoulli', or 'displacement_proportional'. "
                f"Provided value: {config.probabilistic_write_mode!r}."
            )
        probability = config.probabilistic_write_probability
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError(
                "Expected probabilistic_write_probability to be finite and "
                f"in [0, 1]. Provided value: {probability!r}."
            )
        probability_scale = config.probabilistic_write_scale_relative
        if (
            isinstance(probability_scale, bool)
            or not isinstance(probability_scale, (int, float))
            or not math.isfinite(float(probability_scale))
            or float(probability_scale) < 0.0
        ):
            raise ValueError(
                "Expected probabilistic_write_scale_relative to be finite "
                f"and non-negative. Provided value: {probability_scale!r}."
            )
        probability_seed = config.probabilistic_write_seed
        if (
            isinstance(probability_seed, bool)
            or not isinstance(probability_seed, int)
            or not 0 <= probability_seed < 2**63
        ):
            raise ValueError(
                "Expected probabilistic_write_seed to be an integer in "
                f"[0, 2**63). Provided value: {probability_seed!r}."
            )
        mode = config.probabilistic_write_mode
        if mode == "none" and (
            float(probability) != 1.0
            or float(probability_scale) != 0.0
            or probability_seed != 0
        ):
            raise ValueError(
                "Expected probabilistic-write settings to equal probability "
                "1.0, scale 0.0, and seed 0 when mode is 'none'. Provided "
                f"value: probability={probability!r}, "
                f"scale={probability_scale!r}, seed={probability_seed!r}."
            )
        if mode != "none" and (
            config.cohort != "B"
            or config.programming_deadband_mode != "none"
        ):
            raise ValueError(
                "Expected probabilistic writing only for cohort B and without "
                "a deterministic deadband. Provided value: "
                f"cohort={config.cohort!r}, deadband="
                f"{config.programming_deadband_mode!r}."
            )
        if mode == "uniform_bernoulli" and (
            not 0.0 < float(probability) < 1.0
            or float(probability_scale) != 0.0
        ):
            raise ValueError(
                "Expected uniform_bernoulli to use a probability in (0, 1) "
                "and scale 0.0. Provided value: "
                f"probability={probability!r}, scale={probability_scale!r}."
            )
        if mode == "displacement_proportional" and (
            float(probability) != 1.0 or float(probability_scale) <= 0.0
        ):
            raise ValueError(
                "Expected displacement_proportional to use probability 1.0 "
                "and a positive relative scale. Provided value: "
                f"probability={probability!r}, scale={probability_scale!r}."
            )
        return config


from hardware.devices.measured_trace import _TraceDataset


from hardware.devices.measured_trace import _ProjectionTable


from hardware.devices.measured_trace import _sha256_file


from hardware.devices.measured_trace import _pava_nonincreasing


from hardware.devices.measured_trace import _load_trace_dataset


from hardware.devices.measured_trace import _assignment_seed


from hardware.devices.measured_trace import _programming_seed


from hardware.devices.measured_trace import _assignment_digest


class MeasuredTraceOptimizer:
    """Wrap direct SGD with deterministic measured-state projection."""

    _SUPPORTS_POSITIVE_GRADIENT_THRESHOLDS = False

    def __init__(
        self,
        optimizer: Any,
        catalog: ParameterCatalog,
        parameters: Mapping[str, Any],
        device_data_path: Path,
        *,
        projection_bindings: tuple[ParameterBinding, ...] | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._config = MeasuredTraceConfig.from_mapping(parameters)
        self._bindings = tuple(
            binding
            for binding in catalog.trainable
            if isinstance(binding.parameter, DenseWeight)
        )
        if not self._bindings:
            raise ValueError(
                "Expected a measured-cohort backend to target at least one trainable "
                "DenseWeight. Provided value: none."
            )
        if self._config.initial_target_mapping in {
            "dual_rail_pairwise_common_window",
            "dual_rail_quad_common_window",
        }:
            expected_layout_keys = {binding.key for binding in self._bindings}
            provided_layout_keys = {
                key
                for key, _layout in (
                    self._config.dual_rail_layout_by_parameter or ()
                )
            }
            if provided_layout_keys != expected_layout_keys:
                raise ValueError(
                    "Expected dual_rail_layout_by_parameter keys to equal "
                    "the stable trainable dense parameter keys. Provided "
                    f"value: expected={sorted(expected_layout_keys)!r}, "
                    f"provided={sorted(provided_layout_keys)!r}."
                )
        thresholds = self._config.positive_gradient_threshold_by_parameter
        if thresholds is not None and not (
            self._SUPPORTS_POSITIVE_GRADIENT_THRESHOLDS
        ):
            raise ValueError(
                "Expected positive_gradient_threshold_by_parameter to be "
                "null for this measured update backend. Provided value: "
                f"{dict(thresholds)!r}."
            )
        if thresholds is not None:
            expected_threshold_keys = {
                binding.key for binding in self._bindings
            }
            provided_threshold_keys = {
                key for key, _threshold in thresholds
            }
            if provided_threshold_keys != expected_threshold_keys:
                raise ValueError(
                    "Expected positive_gradient_threshold_by_parameter keys "
                    "to equal the stable trainable dense parameter keys. "
                    "Provided value: "
                    f"expected={sorted(expected_threshold_keys)!r}, "
                    f"provided={sorted(provided_threshold_keys)!r}."
                )
        self._projection_bindings = (
            self._bindings
            if projection_bindings is None
            else tuple(projection_bindings)
        )
        if (
            not self._projection_bindings
            or any(
                not isinstance(binding.parameter, DenseWeight)
                for binding in self._projection_bindings
            )
            or len({binding.key for binding in self._projection_bindings})
            != len(self._projection_bindings)
            or not {binding.key for binding in self._bindings}.issubset(
                {binding.key for binding in self._projection_bindings}
            )
        ):
            raise ValueError(
                "Expected projection_bindings to contain unique DenseWeight "
                "bindings including every trainable measured parameter. "
                f"Provided value: {self._projection_bindings!r}."
            )
        self._dataset = _load_trace_dataset(
            Path(device_data_path),
            self._config,
        )
        self._validate_binding_bounds()
        self._raw_projection_cache_report = (
            self._resolve_raw_projection_cache()
        )
        self._plant = MeasuredTraceBank(self._config.curve_preprocessing)
        self._plant.source_curves: dict[str, torch.Tensor] = {}
        self._plant.tables: dict[str, _ProjectionTable] = {}
        self._build_projection_tables()
        self._shadows: dict[str, torch.Tensor] = {}
        self._last_programmed_shadows: dict[str, torch.Tensor] = {}
        self._programming_threshold_s: dict[str, float] = {}
        self._probabilistic_write_scale_s: dict[str, float] = {}
        self._probabilistic_write_generators: dict[
            str, torch.Generator
        ] = {}
        if self._config.probabilistic_write_mode != "none":
            for binding in self._bindings:
                generator = torch.Generator(device=binding.state.device)
                generator.manual_seed(
                    _programming_seed(
                        self._config.probabilistic_write_seed,
                        binding.key,
                    )
                )
                self._probabilistic_write_generators[binding.key] = generator
        self._plant.pulse_indices: dict[str, torch.Tensor] = {}
        self._accumulators: dict[str, dict[str, torch.Tensor]] = {}
        self._initialization_reports: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._step_count = 0

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def cohort(self) -> str:
        return self._config.cohort

    @property
    def configuration(self) -> dict[str, Any]:
        return asdict(self._config)

    @property
    def data_report(self) -> dict[str, Any]:
        conductance = self._dataset.conductance_s
        cohort_a_used = self._config.cohort == "A"
        cohort_b_used = self._config.cohort == "B"
        return {
            "semantics": (
                f"cohort_{self._config.cohort.lower()}_virtual_device_"
                "global_nearest"
            ),
            "pulse_model": False,
            "active_cohort": self._config.cohort,
            "source_path": self._dataset.source_path,
            "source_sha256": self._dataset.source_sha256,
            "source_trace_count": self._dataset.source_trace_count,
            "formed_trace_count": self._dataset.formed_trace_count,
            "screened_trace_names": list(self._dataset.screened_trace_names),
            "physical_cell_count": self._dataset.physical_cell_count,
            "cohort_a_cell_count": self._dataset.cohort_a_cell_count,
            "cohort_b_cell_count": self._dataset.cohort_b_cell_count,
            "cohort_a_trace_count": len(self._dataset.cohort_a_trace_names),
            "cohort_b_trace_count": len(self._dataset.cohort_b_trace_names),
            "active_trace_count": len(self._dataset.active_trace_names),
            "active_trace_names": list(self._dataset.active_trace_names),
            "cohort_a_used": cohort_a_used,
            "cohort_b_used": cohort_b_used,
            "cohort_a_reserved_not_used": not cohort_a_used,
            "cohort_b_reserved_not_used": not cohort_b_used,
            "trace_length": int(conductance.shape[1]),
            "conductance_min_s": float(conductance.min()),
            "conductance_max_s": float(conductance.max()),
            "curve_preprocessing": self._config.curve_preprocessing,
            "initial_target_mapping": self._config.initial_target_mapping,
            "dual_rail_layout_by_parameter": (
                None
                if self._config.dual_rail_layout_by_parameter is None
                else dict(self._config.dual_rail_layout_by_parameter)
            ),
            "positive_gradient_threshold_by_parameter": (
                None
                if self._config.positive_gradient_threshold_by_parameter
                is None
                else {
                    key: float(value)
                    for key, value in (
                        self._config.positive_gradient_threshold_by_parameter
                    )
                }
            ),
            "raw_projection_cache": dict(
                self._raw_projection_cache_report
            ),
            "programming_deadband_mode": (
                self._config.programming_deadband_mode
            ),
            "programming_deadband_relative": float(
                self._config.programming_deadband_relative
            ),
            "probabilistic_write_mode": (
                self._config.probabilistic_write_mode
            ),
            "probabilistic_write_probability": float(
                self._config.probabilistic_write_probability
            ),
            "probabilistic_write_scale_relative": float(
                self._config.probabilistic_write_scale_relative
            ),
            "probabilistic_write_seed": (
                self._config.probabilistic_write_seed
            ),
            "assignment_sha256_by_parameter": {
                key: table.assignment_sha256
                for key, table in sorted(self._plant.tables.items())
            },
        }

    @property
    def programming_report(self) -> dict[str, Any]:
        return {
            **self.data_report,
            "initialized": self._initialized,
            "step_count": self._step_count,
            "parameters": {
                binding.key: self._binding_report(binding.key)
                for binding in self._bindings
            },
        }

    def set_learning_rates(self, values: list[float] | tuple[float, ...]) -> None:
        rates = tuple(float(value) for value in values)
        if len(rates) != len(self.param_groups) or any(
            not math.isfinite(value) or value < 0.0 for value in rates
        ):
            raise ValueError(
                "Expected one finite non-negative learning rate per optimizer "
                f"parameter group ({len(self.param_groups)}). Provided value: "
                f"{rates!r}."
            )
        for group, rate in zip(self.param_groups, rates):
            group["lr"] = rate

    def learning_rates(self) -> tuple[float, ...]:
        return tuple(float(group["lr"]) for group in self.param_groups)

    def _curve_ranges(self, key):
        return self._plant.curve_ranges(key)

    def _mapped_initial_targets(
        self,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, dict[str, Any]],
    ]:
        nominal: dict[str, torch.Tensor] = {}
        clipped: dict[str, torch.Tensor] = {}
        targets: dict[str, torch.Tensor] = {}
        details: dict[str, dict[str, Any]] = {}
        for binding in self._bindings:
            lower = float(binding.parameter.min_cond)
            upper = float(binding.parameter.max_cond)
            loaded = binding.state.detach().clone()
            clipped[binding.key] = (loaded < lower) | (loaded > upper)
            nominal[binding.key] = loaded.clamp(min=lower, max=upper)

        mode = self._config.initial_target_mapping
        if mode == "literal":
            for binding in self._bindings:
                targets[binding.key] = nominal[binding.key]
                details[binding.key] = {"initial_target_mapping": mode}
            return nominal, clipped, targets, details

        if mode == "per_device_affine":
            for binding in self._bindings:
                key = binding.key
                lower = float(binding.parameter.min_cond)
                upper = float(binding.parameter.max_cond)
                curve_min, curve_max = self._curve_ranges(key)
                fraction = nominal[key].reshape(-1).sub(lower).div(
                    upper - lower
                )
                span = curve_max - curve_min
                targets[key] = (
                    curve_min + fraction * span
                ).reshape(binding.state.shape)
                details[key] = {
                    "initial_target_mapping": mode,
                    "device_curve_min_mean_s": float(curve_min.mean().item()),
                    "device_curve_max_mean_s": float(curve_max.mean().item()),
                    "device_curve_span_min_s": float(span.min().item()),
                    "device_curve_span_mean_s": float(span.mean().item()),
                    "device_curve_span_max_s": float(span.max().item()),
                }
            return nominal, clipped, targets, details

        if mode == "dual_rail_pairwise_common_window":
            layouts = dict(self._config.dual_rail_layout_by_parameter or ())
            for binding in self._bindings:
                key = binding.key
                shape = tuple(binding.state.shape)
                if (
                    len(shape) != 2
                    or shape[0] % 2
                    or shape[1] % 2
                ):
                    raise ValueError(
                        "Expected dual-rail pairwise common-window bindings "
                        "to be even-by-even rank-2 tensors. Provided value: "
                        f"key={key!r}, shape={shape!r}."
                    )
                layout = layouts[key]
                input_count = shape[0] // 2
                output_count = shape[1] // 2
                if layout == "halves":
                    plus_columns = torch.arange(
                        output_count,
                        device=binding.state.device,
                    )
                    minus_columns = plus_columns + output_count
                else:
                    plus_columns = torch.arange(
                        output_count,
                        device=binding.state.device,
                    ) * 2
                    minus_columns = plus_columns + 1
                lower = float(binding.parameter.min_cond)
                upper = float(binding.parameter.max_cond)
                fraction = nominal[key].sub(lower).div(upper - lower)
                curve_min, curve_max = self._curve_ranges(key)
                curve_min = curve_min.reshape(shape)
                curve_max = curve_max.reshape(shape)
                target = torch.empty_like(nominal[key])
                baselines = []
                spans = []
                overlaps = []
                for row_offset in (0, input_count):
                    rows = torch.arange(
                        input_count,
                        device=binding.state.device,
                    ) + row_offset
                    first_min = curve_min[rows[:, None], plus_columns]
                    second_min = curve_min[rows[:, None], minus_columns]
                    first_max = curve_max[rows[:, None], plus_columns]
                    second_max = curve_max[rows[:, None], minus_columns]
                    common_low = torch.maximum(first_min, second_min)
                    common_high = torch.minimum(first_max, second_max)
                    overlap = common_high >= common_low
                    baseline = torch.where(
                        overlap,
                        common_low,
                        0.5 * (common_low + common_high),
                    )
                    span = (common_high - common_low).clamp_min(0.0)
                    target[rows[:, None], plus_columns] = (
                        baseline
                        + fraction[rows[:, None], plus_columns] * span
                    )
                    target[rows[:, None], minus_columns] = (
                        baseline
                        + fraction[rows[:, None], minus_columns] * span
                    )
                    baselines.append(baseline)
                    spans.append(span)
                    overlaps.append(overlap)
                baseline = torch.cat(tuple(item.reshape(-1) for item in baselines))
                span = torch.cat(tuple(item.reshape(-1) for item in spans))
                overlap = torch.cat(tuple(item.reshape(-1) for item in overlaps))
                targets[key] = target
                details[key] = {
                    "initial_target_mapping": mode,
                    "dual_rail_layout": layout,
                    "pair_count": int(overlap.numel()),
                    "common_window_empty_fraction": float(
                        (~overlap).to(torch.float64).mean().item()
                    ),
                    "common_window_baseline_mean_s": float(
                        baseline.mean().item()
                    ),
                    "common_window_span_min_s": float(span.min().item()),
                    "common_window_span_mean_s": float(span.mean().item()),
                    "common_window_span_max_s": float(span.max().item()),
                    "nominal_fraction_min": float(fraction.min().item()),
                    "nominal_fraction_max": float(fraction.max().item()),
                }
            return nominal, clipped, targets, details

        if mode == "dual_rail_quad_common_window":
            layouts = dict(self._config.dual_rail_layout_by_parameter or ())
            for binding in self._bindings:
                key = binding.key
                shape = tuple(binding.state.shape)
                if (
                    len(shape) != 2
                    or shape[0] % 2
                    or shape[1] % 2
                ):
                    raise ValueError(
                        "Expected dual-rail quad common-window bindings to "
                        "be even-by-even rank-2 tensors. Provided value: "
                        f"key={key!r}, shape={shape!r}."
                    )
                layout = layouts[key]
                input_count = shape[0] // 2
                output_count = shape[1] // 2
                plus_rows = torch.arange(
                    input_count,
                    device=binding.state.device,
                )
                minus_rows = plus_rows + input_count
                if layout == "halves":
                    plus_columns = torch.arange(
                        output_count,
                        device=binding.state.device,
                    )
                    minus_columns = plus_columns + output_count
                else:
                    plus_columns = torch.arange(
                        output_count,
                        device=binding.state.device,
                    ) * 2
                    minus_columns = plus_columns + 1

                lower = float(binding.parameter.min_cond)
                upper = float(binding.parameter.max_cond)
                fraction = nominal[key].sub(lower).div(upper - lower)
                curve_min, curve_max = self._curve_ranges(key)
                curve_min = curve_min.reshape(shape)
                curve_max = curve_max.reshape(shape)
                # One logical synapse occupies the Cartesian product of its
                # two source rails and two destination rails.  Sharing all
                # four cells' window prevents a source-row-dependent
                # baseline from reappearing as signed forward drive.
                row_groups = (plus_rows, minus_rows)
                column_groups = (plus_columns, minus_columns)
                group_minimum = torch.stack(
                    tuple(
                        curve_min[rows[:, None], columns]
                        for rows in row_groups
                        for columns in column_groups
                    )
                )
                group_maximum = torch.stack(
                    tuple(
                        curve_max[rows[:, None], columns]
                        for rows in row_groups
                        for columns in column_groups
                    )
                )
                common_low = group_minimum.amax(dim=0)
                common_high = group_maximum.amin(dim=0)
                overlap = common_high >= common_low
                baseline = torch.where(
                    overlap,
                    common_low,
                    0.5 * (common_low + common_high),
                )
                span = (common_high - common_low).clamp_min(0.0)
                target = torch.empty_like(nominal[key])
                for rows in row_groups:
                    for columns in column_groups:
                        target[rows[:, None], columns] = (
                            baseline
                            + fraction[rows[:, None], columns] * span
                        )
                targets[key] = target
                details[key] = {
                    "initial_target_mapping": mode,
                    "dual_rail_layout": layout,
                    "group_size": 4,
                    "quad_count": int(overlap.numel()),
                    "logical_synapse_count": int(overlap.numel()),
                    "common_window_empty_fraction": float(
                        (~overlap).to(torch.float64).mean().item()
                    ),
                    "common_window_baseline_mean_s": float(
                        baseline.mean().item()
                    ),
                    "common_window_span_min_s": float(span.min().item()),
                    "common_window_span_mean_s": float(span.mean().item()),
                    "common_window_span_max_s": float(span.max().item()),
                    "nominal_fraction_min": float(fraction.min().item()),
                    "nominal_fraction_max": float(fraction.max().item()),
                }
            return nominal, clipped, targets, details

        if len(self._bindings) % 2:
            raise ValueError(
                "Expected paired_affine_common_window to receive adjacent "
                "conductance_plus/conductance_minus bindings. Provided "
                f"value: binding_count={len(self._bindings)}."
            )
        for index in range(0, len(self._bindings), 2):
            plus = self._bindings[index]
            minus = self._bindings[index + 1]
            if (
                plus.role != "conductance_plus"
                or minus.role != "conductance_minus"
                or plus.key.rsplit(".", 1)[-1]
                != minus.key.rsplit(".", 1)[-1]
                or tuple(plus.state.shape) != tuple(minus.state.shape)
                or float(plus.parameter.min_cond)
                != float(minus.parameter.min_cond)
                or float(plus.parameter.max_cond)
                != float(minus.parameter.max_cond)
            ):
                raise ValueError(
                    "Expected paired_affine_common_window to receive "
                    "shape- and bound-matched adjacent plus/minus bindings "
                    f"for each layer. Provided value: plus={plus.key!r}, "
                    f"minus={minus.key!r}."
                )
            plus_min, plus_max = self._curve_ranges(plus.key)
            minus_min, minus_max = self._curve_ranges(minus.key)
            common_low = torch.maximum(plus_min, minus_min)
            common_high = torch.minimum(plus_max, minus_max)
            overlap = common_high >= common_low
            baseline = torch.where(
                overlap,
                common_low,
                0.5 * (common_low + common_high),
            )
            common_span = (common_high - common_low).clamp_min(0.0)
            lower = float(plus.parameter.min_cond)
            upper = float(plus.parameter.max_cond)
            for binding in (plus, minus):
                fraction = nominal[binding.key].reshape(-1).sub(lower).div(
                    upper - lower
                )
                targets[binding.key] = (
                    baseline + fraction * common_span
                ).reshape(binding.state.shape)
                details[binding.key] = {
                    "initial_target_mapping": mode,
                    "paired_binding": (
                        minus.key if binding is plus else plus.key
                    ),
                    "common_window_empty_fraction": float(
                        (~overlap).to(torch.float64).mean().item()
                    ),
                    "common_window_baseline_mean_s": float(
                        baseline.mean().item()
                    ),
                    "common_window_span_min_s": float(
                        common_span.min().item()
                    ),
                    "common_window_span_mean_s": float(
                        common_span.mean().item()
                    ),
                    "common_window_span_max_s": float(
                        common_span.max().item()
                    ),
                }
        return nominal, clipped, targets, details

    def initialize_at_pulse_zero(self) -> dict[str, Any]:
        if self._config.cohort != "A":
            raise RuntimeError(
                "Expected pulse-zero initialization only for cohort A. "
                f"Provided value: cohort={self._config.cohort!r}."
            )
        if self._initialized:
            raise RuntimeError(
                "Expected measured-cohort initialization exactly once. "
                "Provided value: optimizer is already initialized."
            )
        for binding in self._bindings:
            table = self._plant.tables[binding.key]
            source = self._plant.source_curves[binding.key]
            pulse = self._config.initial_pulse_index
            realized = (
                table.alpha * source[table.source_left, pulse]
                + (1.0 - table.alpha) * source[table.source_right, pulse]
            ).reshape(table.shape)
            with torch.no_grad():
                binding.state.copy_(realized)
            self._shadows[binding.key] = realized.detach().clone()
            self._initialize_programming_state(binding.key, realized)
            self._plant.pulse_indices[binding.key] = torch.full(
                (realized.numel(),),
                pulse,
                dtype=torch.int16,
                device=realized.device,
            )
            self._accumulators[binding.key] = self._empty_accumulators(
                realized.device
            )
            self._initialization_reports[binding.key] = {
                "kind": "pulse_zero",
                "target_clipped_fraction": 0.0,
                "projection_mean_abs_error_s": 0.0,
                "projection_rms_error_s": 0.0,
                "projection_max_abs_error_s": 0.0,
                "pulse_index_min": pulse,
                "pulse_index_max": pulse,
                "pulse_index_mean": float(pulse),
                "pulse_zero_fraction": 1.0,
                "last_pulse_fraction": 0.0,
            }
        self._initialized = True
        return self.programming_report

    def preview_reset_targets(self) -> dict[str, torch.Tensor]:
        """Project the current cohort-A targets without mutating optimizer state.

        This supports calibration-only selection among explicitly declared
        teacher mappings.  The selected candidate is still initialized later
        through :meth:`initialize_from_reset_targets`, which records the one
        accounted RESET-to-target write.
        """

        if self._config.cohort != "A":
            raise RuntimeError(
                "Expected reset-target preview only for cohort A. Provided "
                f"value: cohort={self._config.cohort!r}."
            )
        if self._initialized:
            raise RuntimeError(
                "Expected reset-target preview before measured optimizer "
                "initialization. Provided value: optimizer is initialized."
            )
        with torch.no_grad():
            _nominal, _clipped, targets, _mapping_details = (
                self._mapped_initial_targets()
            )
            return {
                binding.key: self._project(
                    binding.key,
                    targets[binding.key].reshape(-1),
                )[0].reshape(binding.state.shape)
                for binding in self._bindings
            }

    def initialize_from_reset_targets(self) -> dict[str, Any]:
        """RESET cohort-A cells, then perform one nearest-state target write.

        The live tensors are expected to contain the desired mapped targets
        when this method is called.  Every virtual cell is first accounted at
        ``initial_pulse_index`` and is then programmed exactly once to the
        globally nearest measured state.  This is distinct from the legacy
        cohort-A from-scratch protocol, which leaves every cell at RESET.
        """

        if self._config.cohort != "A":
            raise RuntimeError(
                "Expected reset-target initialization only for cohort A. "
                f"Provided value: cohort={self._config.cohort!r}."
            )
        if self._initialized:
            raise RuntimeError(
                "Expected measured-cohort initialization exactly once. "
                "Provided value: optimizer is already initialized."
            )
        with torch.no_grad():
            nominal, clipped, targets, mapping_details = (
                self._mapped_initial_targets()
            )
            for binding in self._bindings:
                key = binding.key
                table = self._plant.tables[key]
                source = self._plant.source_curves[key]
                pulse = self._config.initial_pulse_index
                reset = (
                    table.alpha * source[table.source_left, pulse]
                    + (1.0 - table.alpha)
                    * source[table.source_right, pulse]
                )
                nominal_target = nominal[key]
                target = targets[key]
                realized, pulses = self._project(key, target.reshape(-1))
                binding.state.copy_(realized.reshape(binding.state.shape))
                self._shadows[key] = target.clone()
                self._initialize_programming_state(key, target)
                self._plant.pulse_indices[key] = pulses.to(torch.int16)
                self._accumulators[key] = self._empty_accumulators(
                    realized.device
                )
                error = target.reshape(-1).sub(realized).to(torch.float64)
                reset_distance = target.reshape(-1).sub(reset).to(
                    torch.float64
                )
                pulse_long = pulses.to(torch.long)
                self._initialization_reports[key] = {
                    "kind": "reset_then_loaded_target_global_nearest",
                    "reset_pulse_index": pulse,
                    "initial_write_count_per_cell": 1,
                    "target_clipped_fraction": float(
                        clipped[key].to(torch.float64).mean().item()
                    ),
                    "nominal_target_conductance_min_s": float(
                        nominal_target.min().item()
                    ),
                    "nominal_target_conductance_max_s": float(
                        nominal_target.max().item()
                    ),
                    "nominal_target_conductance_rms_s": float(
                        torch.sqrt(
                            nominal_target.to(torch.float64).square().mean()
                        ).item()
                    ),
                    "target_conductance_min_s": float(target.min().item()),
                    "target_conductance_max_s": float(target.max().item()),
                    "target_conductance_rms_s": float(
                        torch.sqrt(target.to(torch.float64).square().mean())
                        .item()
                    ),
                    "reset_to_target_mean_abs_distance_s": float(
                        reset_distance.abs().mean().item()
                    ),
                    "reset_to_target_rms_distance_s": float(
                        torch.sqrt(reset_distance.square().mean()).item()
                    ),
                    "projection_mean_abs_error_s": float(
                        error.abs().mean().item()
                    ),
                    "projection_rms_error_s": float(
                        torch.sqrt(error.square().mean()).item()
                    ),
                    "projection_max_abs_error_s": float(
                        error.abs().max().item()
                    ),
                    "pulse_index_min": int(pulse_long.min().item()),
                    "pulse_index_max": int(pulse_long.max().item()),
                    "pulse_index_mean": float(
                        pulse_long.to(torch.float64).mean().item()
                    ),
                    "pulse_zero_fraction": float(
                        (pulse_long == pulse).to(torch.float64).mean().item()
                    ),
                    "last_pulse_fraction": float(
                        (
                            pulse_long
                            == self._config.expected_trace_length - 1
                        )
                        .to(torch.float64)
                        .mean()
                        .item()
                    ),
                    **mapping_details[key],
                }
        self._initialized = True
        return self.programming_report

    def initialize_from_loaded_targets(self) -> dict[str, Any]:
        """Deploy loaded cohort-A targets onto independently held-out B curves."""

        if self._config.cohort != "B":
            raise RuntimeError(
                "Expected loaded-target deployment only for cohort B. "
                f"Provided value: cohort={self._config.cohort!r}."
            )
        if self._initialized:
            raise RuntimeError(
                "Expected measured-cohort initialization exactly once. "
                "Provided value: optimizer is already initialized."
            )
        with torch.no_grad():
            nominal, clipped, targets, mapping_details = (
                self._mapped_initial_targets()
            )
            for binding in self._bindings:
                key = binding.key
                nominal_target = nominal[key]
                target = targets[key]
                realized, pulses = self._project(key, target.reshape(-1))
                realized_shaped = realized.reshape(binding.state.shape)
                binding.state.copy_(realized_shaped)
                self._shadows[key] = target.clone()
                self._initialize_programming_state(key, target)
                self._plant.pulse_indices[key] = pulses.to(torch.int16)
                self._accumulators[key] = self._empty_accumulators(
                    realized.device
                )
                error = (target.reshape(-1) - realized).to(torch.float64)
                pulse_long = pulses.to(torch.long)
                self._initialization_reports[key] = {
                    "kind": (
                        "loaded_target_global_nearest"
                        if self._config.initial_target_mapping == "literal"
                        else "mapped_loaded_target_global_nearest"
                    ),
                    "target_clipped_fraction": float(
                        clipped[key].to(torch.float64).mean().item()
                    ),
                    "nominal_target_conductance_min_s": float(
                        nominal_target.min().item()
                    ),
                    "nominal_target_conductance_max_s": float(
                        nominal_target.max().item()
                    ),
                    "nominal_target_conductance_rms_s": float(
                        torch.sqrt(
                            nominal_target.to(torch.float64).square().mean()
                        ).item()
                    ),
                    "target_conductance_min_s": float(target.min().item()),
                    "target_conductance_max_s": float(target.max().item()),
                    "target_conductance_rms_s": float(
                        torch.sqrt(
                            target.to(torch.float64).square().mean()
                        ).item()
                    ),
                    "realized_conductance_min_s": float(realized.min().item()),
                    "realized_conductance_max_s": float(realized.max().item()),
                    "realized_conductance_rms_s": float(
                        torch.sqrt(
                            realized.to(torch.float64).square().mean()
                        ).item()
                    ),
                    "projection_mean_abs_error_s": float(
                        error.abs().mean().item()
                    ),
                    "projection_rms_error_s": float(
                        torch.sqrt(error.square().mean()).item()
                    ),
                    "projection_max_abs_error_s": float(
                        error.abs().max().item()
                    ),
                    "pulse_index_min": int(pulse_long.min().item()),
                    "pulse_index_max": int(pulse_long.max().item()),
                    "pulse_index_mean": float(
                        pulse_long.to(torch.float64).mean().item()
                    ),
                    "pulse_zero_fraction": float(
                        (pulse_long == 0).to(torch.float64).mean().item()
                    ),
                    "last_pulse_fraction": float(
                        (pulse_long == self._config.expected_trace_length - 1)
                        .to(torch.float64)
                        .mean()
                        .item()
                    ),
                    **mapping_details[key],
                }
        self._initialized = True
        return self.programming_report

    def step(self, closure=None):
        if closure is not None:
            raise ValueError(
                "Expected measured-cohort updates without an optimizer "
                f"closure. Provided value: {closure!r}."
            )
        if not self._initialized:
            raise RuntimeError(
                "Expected cohort-appropriate measured initialization before "
                "the first optimizer step. Provided value: uninitialized."
            )
        previous_realized: dict[str, torch.Tensor] = {}
        previous_shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for binding in self._bindings:
                previous_realized[binding.key] = binding.state.detach().clone()
                previous_shadow[binding.key] = self._shadows[binding.key].clone()
                binding.state.copy_(self._shadows[binding.key])
        result = self._optimizer.step()
        with torch.no_grad():
            for binding in self._bindings:
                key = binding.key
                proposed = binding.state
                lower = float(binding.parameter.min_cond)
                upper = float(binding.parameter.max_cond)
                clipped = (proposed < lower) | (proposed > upper)
                proposed.clamp_(min=lower, max=upper)
                self._shadows[key].copy_(proposed)
                targets = proposed.reshape(-1)
                threshold = self._programming_threshold_s[key]
                probabilistic_mode = self._config.probabilistic_write_mode
                write_probability: float | torch.Tensor | None = None
                if threshold == 0.0 and probabilistic_mode == "none":
                    # Keep the historical path byte-for-byte equivalent for
                    # the zero-deadband control.
                    realized, pulses = self._project(key, targets)
                    eligible = None
                else:
                    last_programmed = self._last_programmed_shadows[
                        key
                    ].reshape(-1)
                    pending = targets.sub(last_programmed).abs()
                    if threshold > 0.0:
                        eligible = pending >= threshold
                    else:
                        generator = self._probabilistic_write_generators[key]
                        if probabilistic_mode == "uniform_bernoulli":
                            write_probability = float(
                                self._config.probabilistic_write_probability
                            )
                        else:
                            write_probability = pending.div(
                                self._probabilistic_write_scale_s[key]
                            ).clamp_(max=1.0)
                        random = torch.rand(
                            targets.shape,
                            dtype=targets.dtype,
                            device=targets.device,
                            generator=generator,
                        )
                        eligible = random < write_probability
                    projected, projected_pulses = self._project(key, targets)
                    realized = torch.where(
                        eligible,
                        projected,
                        previous_realized[key].reshape(-1),
                    )
                    pulses = torch.where(
                        eligible,
                        projected_pulses,
                        self._plant.pulse_indices[key],
                    )
                    last_programmed.copy_(
                        torch.where(eligible, targets, last_programmed)
                    )
                binding.state.copy_(realized.reshape(binding.state.shape))
                self._accumulate(
                    key,
                    previous_shadow=previous_shadow[key].reshape(-1),
                    previous_realized=previous_realized[key].reshape(-1),
                    realized=realized,
                    pulses=pulses,
                    clipped=clipped.reshape(-1),
                    programming_eligible=eligible,
                    write_probability=write_probability,
                )
                self._plant.pulse_indices[key] = pulses.to(torch.int16)
        self._step_count += 1
        return result

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "configuration": self.configuration,
            "device_data_sha256": self._dataset.source_sha256,
            "active_cohort_trace_names": self._dataset.active_trace_names,
            "assignment_sha256_by_parameter": {
                key: table.assignment_sha256
                for key, table in self._plant.tables.items()
            },
            "optimizer": self._optimizer.state_dict(),
            "initialized": self._initialized,
            "step_count": self._step_count,
            "shadows": {
                key: value.detach().cpu().clone()
                for key, value in self._shadows.items()
            },
            "last_programmed_shadows": {
                key: value.detach().cpu().clone()
                for key, value in self._last_programmed_shadows.items()
            },
            "programming_threshold_s": dict(self._programming_threshold_s),
            "probabilistic_write_scale_s": dict(
                self._probabilistic_write_scale_s
            ),
            "probabilistic_write_rng_states": {
                key: generator.get_state().detach().cpu().clone()
                for key, generator in self._probabilistic_write_generators.items()
            },
            "pulse_indices": {
                key: value.detach().cpu().clone()
                for key, value in self._plant.pulse_indices.items()
            },
            "accumulators": {
                key: {
                    name: value.detach().cpu().clone()
                    for name, value in accumulators.items()
                }
                for key, accumulators in self._accumulators.items()
            },
            "initialization_reports": deepcopy(self._initialization_reports),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise ValueError(
                "Expected measured-cohort optimizer state to be an object. "
                f"Provided value: {state_dict!r}."
            )
        version = state_dict.get("version")
        legacy_expected = {
            "version",
            "configuration",
            "device_data_sha256",
            "cohort_a_trace_names",
            "assignment_sha256_by_parameter",
            "optimizer",
            "initialized",
            "step_count",
            "shadows",
            "pulse_indices",
            "accumulators",
        }
        previous_expected = (
            legacy_expected
            - {"cohort_a_trace_names"}
            | {"active_cohort_trace_names", "initialization_reports"}
        )
        deadband_expected = previous_expected | {
            "last_programmed_shadows",
            "programming_threshold_s",
        }
        current_expected = deadband_expected | {
            "probabilistic_write_scale_s",
            "probabilistic_write_rng_states",
        }
        if version == _LEGACY_STATE_VERSION:
            expected = legacy_expected
        elif version == _PREVIOUS_STATE_VERSION:
            expected = previous_expected
        elif version == _DEADBAND_STATE_VERSION:
            expected = deadband_expected
        elif version == _STATE_VERSION:
            expected = current_expected
        else:
            raise ValueError(
                "Expected measured-cohort optimizer state version to equal "
                f"{_LEGACY_STATE_VERSION}, {_PREVIOUS_STATE_VERSION}, "
                f"{_DEADBAND_STATE_VERSION}, or {_STATE_VERSION}. Provided "
                f"value: {version!r}."
            )
        if set(state_dict) != expected:
            raise ValueError(
                "Expected measured-cohort optimizer state to contain exactly "
                f"{sorted(expected)!r}. Provided value: "
                f"{sorted(state_dict, key=repr)!r}."
            )
        if version == _LEGACY_STATE_VERSION and self._config.cohort != "A":
            raise ValueError(
                "Expected legacy measured optimizer state only for cohort A. "
                f"Provided value: cohort={self._config.cohort!r}."
            )
        saved_config = MeasuredTraceConfig.from_mapping(
            state_dict["configuration"]
        )
        if saved_config != self._config:
            raise ValueError(
                "Expected measured-cohort checkpoint configuration to match "
                "the current configuration. Provided value: "
                f"saved={asdict(saved_config)!r}, current={self.configuration!r}."
            )
        saved_trace_names = (
            state_dict["cohort_a_trace_names"]
            if version == _LEGACY_STATE_VERSION
            else state_dict["active_cohort_trace_names"]
        )
        identity = (
            state_dict["device_data_sha256"] == self._dataset.source_sha256
            and tuple(saved_trace_names) == self._dataset.active_trace_names
            and dict(state_dict["assignment_sha256_by_parameter"])
            == {
                key: table.assignment_sha256
                for key, table in self._plant.tables.items()
            }
        )
        if not identity:
            raise ValueError(
                "Expected measured-cohort checkpoint data and virtual-device "
                "assignments to match this runtime. Provided value: mismatch."
            )
        initialized = state_dict["initialized"]
        step_count = state_dict["step_count"]
        if not isinstance(initialized, bool):
            raise ValueError(
                "Expected measured optimizer initialized to be a boolean. "
                f"Provided value: {initialized!r}."
            )
        if (
            isinstance(step_count, bool)
            or not isinstance(step_count, int)
            or step_count < 0
        ):
            raise ValueError(
                "Expected measured optimizer step_count to be a non-negative "
                f"integer. Provided value: {step_count!r}."
            )
        expected_keys = {binding.key for binding in self._bindings}
        state_keys = expected_keys if initialized else set()
        for name in ("shadows", "pulse_indices", "accumulators"):
            raw = state_dict[name]
            if not isinstance(raw, Mapping) or set(raw) != state_keys:
                raise ValueError(
                    f"Expected measured optimizer {name} keys to equal "
                    f"{sorted(state_keys)!r}. Provided value: {raw!r}."
                )
        raw_initialization_reports = (
            {}
            if version == _LEGACY_STATE_VERSION
            else state_dict["initialization_reports"]
        )
        if version in {
            _PREVIOUS_STATE_VERSION,
            _DEADBAND_STATE_VERSION,
            _STATE_VERSION,
        } and (
            not isinstance(raw_initialization_reports, Mapping)
            or set(raw_initialization_reports) != state_keys
        ):
            raise ValueError(
                "Expected measured optimizer initialization_reports keys to "
                f"equal {sorted(state_keys)!r}. Provided value: "
                f"{raw_initialization_reports!r}."
            )
        raw_last_programmed_shadows = (
            state_dict["last_programmed_shadows"]
            if version in {_DEADBAND_STATE_VERSION, _STATE_VERSION}
            else state_dict["shadows"]
        )
        raw_programming_thresholds = (
            state_dict["programming_threshold_s"]
            if version in {_DEADBAND_STATE_VERSION, _STATE_VERSION}
            else {key: 0.0 for key in state_keys}
        )
        raw_probability_scales = (
            state_dict["probabilistic_write_scale_s"]
            if version == _STATE_VERSION
            else {key: 0.0 for key in state_keys}
        )
        raw_rng_states = (
            state_dict["probabilistic_write_rng_states"]
            if version == _STATE_VERSION
            else {}
        )
        for name, raw in (
            ("last_programmed_shadows", raw_last_programmed_shadows),
            ("programming_threshold_s", raw_programming_thresholds),
            ("probabilistic_write_scale_s", raw_probability_scales),
        ):
            if not isinstance(raw, Mapping) or set(raw) != state_keys:
                raise ValueError(
                    f"Expected measured optimizer {name} keys to equal "
                    f"{sorted(state_keys)!r}. Provided value: {raw!r}."
                )
        expected_rng_keys = (
            expected_keys
            if self._config.probabilistic_write_mode != "none"
            else set()
        )
        if not isinstance(raw_rng_states, Mapping) or set(
            raw_rng_states
        ) != expected_rng_keys:
            raise ValueError(
                "Expected measured optimizer probabilistic_write_rng_states "
                f"keys to equal {sorted(expected_rng_keys)!r}. Provided "
                f"value: {raw_rng_states!r}."
            )
        shadows: dict[str, torch.Tensor] = {}
        last_programmed_shadows: dict[str, torch.Tensor] = {}
        programming_thresholds: dict[str, float] = {}
        probability_scales: dict[str, float] = {}
        pulses: dict[str, torch.Tensor] = {}
        accumulators: dict[str, dict[str, torch.Tensor]] = {}
        for binding in self._bindings:
            if not initialized:
                break
            key = binding.key
            shadow = state_dict["shadows"][key]
            last_programmed_shadow = raw_last_programmed_shadows[key]
            pulse = state_dict["pulse_indices"][key]
            if (
                not isinstance(shadow, torch.Tensor)
                or tuple(shadow.shape) != tuple(binding.state.shape)
                or shadow.dtype != binding.state.dtype
            ):
                raise ValueError(
                    "Expected each measured shadow to match its live tensor "
                    f"shape and dtype. Provided value: key={key!r}, "
                    f"shadow={shadow!r}."
                )
            if (
                not isinstance(last_programmed_shadow, torch.Tensor)
                or tuple(last_programmed_shadow.shape)
                != tuple(binding.state.shape)
                or last_programmed_shadow.dtype != binding.state.dtype
            ):
                raise ValueError(
                    "Expected each measured last-programmed shadow to match "
                    "its live tensor shape and dtype. Provided value: "
                    f"key={key!r}, shadow={last_programmed_shadow!r}."
                )
            threshold = raw_programming_thresholds[key]
            probability_scale = raw_probability_scales[key]
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
                or float(threshold) < 0.0
            ):
                raise ValueError(
                    "Expected each measured programming threshold to be a "
                    "finite non-negative number in siemens. Provided value: "
                    f"key={key!r}, threshold={threshold!r}."
                )
            if (
                self._config.programming_deadband_mode == "none"
                and float(threshold) != 0.0
            ) or (
                self._config.programming_deadband_mode
                == "accumulated_shadow_relative_rms"
                and float(threshold) <= 0.0
            ):
                raise ValueError(
                    "Expected each saved programming threshold to match the "
                    "configured deadband mode. Provided value: "
                    f"key={key!r}, mode="
                    f"{self._config.programming_deadband_mode!r}, "
                    f"threshold={threshold!r}."
                )
            if (
                isinstance(probability_scale, bool)
                or not isinstance(probability_scale, (int, float))
                or not math.isfinite(float(probability_scale))
                or float(probability_scale) < 0.0
                or (
                    self._config.probabilistic_write_mode
                    == "displacement_proportional"
                    and float(probability_scale) <= 0.0
                )
                or (
                    self._config.probabilistic_write_mode
                    != "displacement_proportional"
                    and float(probability_scale) != 0.0
                )
            ):
                raise ValueError(
                    "Expected each saved probabilistic write scale to match "
                    "the configured policy. Provided value: "
                    f"key={key!r}, mode="
                    f"{self._config.probabilistic_write_mode!r}, "
                    f"scale={probability_scale!r}."
                )
            if (
                not isinstance(pulse, torch.Tensor)
                or pulse.dtype != torch.int16
                or pulse.shape != (binding.state.numel(),)
                or bool((pulse < 0).any())
                or bool((pulse >= self._config.expected_trace_length).any())
            ):
                raise ValueError(
                    "Expected each measured pulse-index tensor to be int16, "
                    "flat, in range, and match the weight size. Provided "
                    f"value: key={key!r}, pulses={pulse!r}."
                )
            raw_accumulators = state_dict["accumulators"][key]
            template = self._empty_accumulators(binding.state.device)
            expected_accumulator_names = set(template)
            if version in {
                _LEGACY_STATE_VERSION,
                _PREVIOUS_STATE_VERSION,
                _DEADBAND_STATE_VERSION,
            }:
                expected_accumulator_names -= {
                    "programming_eligible_count",
                    "programming_suppressed_count",
                    "write_probability_sum",
                    "write_probability_count",
                }
                if version == _DEADBAND_STATE_VERSION:
                    expected_accumulator_names |= {
                        "programming_eligible_count",
                        "programming_suppressed_count",
                    }
            if (
                not isinstance(raw_accumulators, Mapping)
                or set(raw_accumulators) != expected_accumulator_names
            ):
                raise ValueError(
                    "Expected measured projection accumulators to have the "
                    f"current schema. Provided value: key={key!r}."
                )
            shadows[key] = shadow.to(binding.state.device).clone()
            last_programmed_shadows[key] = last_programmed_shadow.to(
                binding.state.device
            ).clone()
            programming_thresholds[key] = float(threshold)
            probability_scales[key] = float(probability_scale)
            pulses[key] = pulse.to(binding.state.device).clone()
            accumulator = {
                name: raw_accumulators[name]
                .to(binding.state.device)
                .clone()
                for name in expected_accumulator_names
            }
            if version in {_LEGACY_STATE_VERSION, _PREVIOUS_STATE_VERSION}:
                accumulator["programming_eligible_count"] = accumulator[
                    "element_updates"
                ].clone()
                accumulator["programming_suppressed_count"] = torch.zeros_like(
                    accumulator["element_updates"]
                )
            if version in {
                _LEGACY_STATE_VERSION,
                _PREVIOUS_STATE_VERSION,
                _DEADBAND_STATE_VERSION,
            }:
                accumulator["write_probability_sum"] = torch.zeros_like(
                    accumulator["element_updates"]
                )
                accumulator["write_probability_count"] = torch.zeros_like(
                    accumulator["element_updates"]
                )
            accumulators[key] = accumulator
        self._optimizer.load_state_dict(deepcopy(state_dict["optimizer"]))
        self._initialized = initialized
        self._step_count = step_count
        self._shadows = shadows
        self._last_programmed_shadows = last_programmed_shadows
        self._programming_threshold_s = programming_thresholds
        self._probabilistic_write_scale_s = probability_scales
        self._plant.pulse_indices = pulses
        self._accumulators = accumulators
        self._initialization_reports = (
            {
                key: {"kind": "legacy_cohort_a_checkpoint"}
                for key in state_keys
            }
            if version == _LEGACY_STATE_VERSION
            else deepcopy(dict(raw_initialization_reports))
        )
        for key, raw_state in raw_rng_states.items():
            if (
                not isinstance(raw_state, torch.Tensor)
                or raw_state.dtype != torch.uint8
                or raw_state.ndim != 1
            ):
                raise ValueError(
                    "Expected every probabilistic write RNG state to be a "
                    "flat uint8 tensor. Provided value: "
                    f"key={key!r}, state={raw_state!r}."
                )
            self._probabilistic_write_generators[key].set_state(
                raw_state.detach().cpu()
            )

    def _initialize_programming_state(
        self,
        key: str,
        shadow: torch.Tensor,
    ) -> None:
        self._last_programmed_shadows[key] = shadow.detach().clone()
        needs_initial_rms = (
            self._config.programming_deadband_mode
            == "accumulated_shadow_relative_rms"
            or self._config.probabilistic_write_mode
            == "displacement_proportional"
        )
        initial_rms = (
            torch.sqrt(
                shadow.detach().to(torch.float64).square().mean()
            ).item()
            if needs_initial_rms
            else None
        )
        if self._config.programming_deadband_mode == "none":
            threshold = 0.0
        else:
            threshold = (
                float(self._config.programming_deadband_relative)
                * float(initial_rms)
            )
            if not math.isfinite(threshold) or threshold <= 0.0:
                raise ValueError(
                    "Expected the initial dense-weight RMS to produce a "
                    "positive finite programming deadband. Provided value: "
                    f"key={key!r}, initial_rms_s={initial_rms!r}, "
                    f"relative={self._config.programming_deadband_relative!r}."
                )
        self._programming_threshold_s[key] = threshold
        if self._config.probabilistic_write_mode != "displacement_proportional":
            probability_scale = 0.0
        else:
            probability_scale = (
                float(self._config.probabilistic_write_scale_relative)
                * float(initial_rms)
            )
            if not math.isfinite(probability_scale) or probability_scale <= 0.0:
                raise ValueError(
                    "Expected the initial dense-weight RMS to produce a "
                    "positive finite probabilistic write scale. Provided "
                    f"value: key={key!r}, initial_rms_s={initial_rms!r}, "
                    "relative="
                    f"{self._config.probabilistic_write_scale_relative!r}."
                )
        self._probabilistic_write_scale_s[key] = probability_scale

    def _validate_binding_bounds(self) -> None:
        source_min = float(self._dataset.conductance_s.min())
        source_max = float(self._dataset.conductance_s.max())
        for binding in self._projection_bindings:
            lower = binding.parameter.min_cond
            upper = binding.parameter.max_cond
            valid = (
                lower is not None
                and upper is not None
                and math.isfinite(float(lower))
                and math.isfinite(float(upper))
                and float(lower) < float(upper)
                and float(lower) <= source_min
                and float(upper) >= source_max
            )
            if not valid:
                raise ValueError(
                    "Expected every measured dense-weight bound interval to be "
                    "finite, ordered, and contain all active-cohort conductances. "
                    "Provided value: "
                    f"key={binding.key!r}, bounds={(lower, upper)!r}, "
                    f"source_range={(source_min, source_max)!r}."
                )

    def _build_projection_tables(self):
        layout = [(b.key, tuple(b.state.shape), b.state.device, b.state.dtype) for b in self._projection_bindings]
        self._plant.assign(layout, self._dataset.conductance_s,
                           assignment_seed=self._config.assignment_seed,
                           cohort=self._config.cohort,
                           expected_trace_length=self._config.expected_trace_length,
                           cache_mode=self._raw_projection_cache_report["mode"])

    def _project(self, key, targets):
        return self._plant.project(key, targets)

    def _resolve_raw_projection_cache(self) -> dict[str, Any]:
        cells = sum(
            binding.state.numel()
            for binding in self._projection_bindings
        )
        required = (
            cells
            * self._config.expected_trace_length
            * (torch.tensor([], dtype=torch.float32).element_size() + 2)
        )
        if self._config.curve_preprocessing != "raw":
            return {
                "mode": "not_applicable",
                "estimated_bytes": 0,
                "available_bytes_at_construction": None,
                "safety_fraction": None,
            }
        device = self._projection_bindings[0].state.device
        available = None
        safety_fraction = 0.7
        if device.type == "cuda":
            available, _total = torch.cuda.mem_get_info(device)
            use_cache = required <= int(available * safety_fraction)
        else:
            # Unit tests and deliberately small CPU jobs retain the fast
            # cache. Full measured studies are required to run on CUDA.
            use_cache = required <= 512 * 1024**2
        return {
            "mode": "cached" if use_cache else "exact_chunked",
            "estimated_bytes": int(required),
            "available_bytes_at_construction": (
                None if available is None else int(available)
            ),
            "safety_fraction": safety_fraction,
        }

    @staticmethod
    def _empty_accumulators(device: torch.device) -> dict[str, torch.Tensor]:
        zeros = lambda: torch.zeros((), dtype=torch.float64, device=device)
        return {
            "projection_abs_error_sum": zeros(),
            "projection_squared_error_sum": zeros(),
            "projection_abs_error_max": zeros(),
            "shadow_clipped_count": zeros(),
            "pulse_jump_sum": zeros(),
            "pulse_jump_max": zeros(),
            "pulse_changed_count": zeros(),
            "proposed_update_squared_sum": zeros(),
            "applied_update_squared_sum": zeros(),
            "programming_eligible_count": zeros(),
            "programming_suppressed_count": zeros(),
            "write_probability_sum": zeros(),
            "write_probability_count": zeros(),
            "element_updates": zeros(),
        }

    def _accumulate(
        self,
        key: str,
        *,
        previous_shadow: torch.Tensor,
        previous_realized: torch.Tensor,
        realized: torch.Tensor,
        pulses: torch.Tensor,
        clipped: torch.Tensor,
        programming_eligible: torch.Tensor | None,
        write_probability: float | torch.Tensor | None,
    ) -> None:
        accumulator = self._accumulators[key]
        shadow = self._shadows[key].reshape(-1)
        error = (shadow - realized).to(torch.float64)
        jump = (
            pulses.to(torch.int32)
            - self._plant.pulse_indices[key].to(torch.int32)
        ).abs().to(torch.float64)
        proposal = (shadow - previous_shadow).to(torch.float64)
        applied = (realized - previous_realized).to(torch.float64)
        accumulator["projection_abs_error_sum"].add_(error.abs().sum())
        accumulator["projection_squared_error_sum"].add_(error.square().sum())
        accumulator["projection_abs_error_max"].copy_(
            torch.maximum(
                accumulator["projection_abs_error_max"],
                error.abs().max(),
            )
        )
        accumulator["shadow_clipped_count"].add_(
            clipped.to(torch.float64).sum()
        )
        accumulator["pulse_jump_sum"].add_(jump.sum())
        accumulator["pulse_jump_max"].copy_(
            torch.maximum(accumulator["pulse_jump_max"], jump.max())
        )
        accumulator["pulse_changed_count"].add_((jump > 0).sum())
        accumulator["proposed_update_squared_sum"].add_(proposal.square().sum())
        accumulator["applied_update_squared_sum"].add_(applied.square().sum())
        if programming_eligible is None:
            accumulator["programming_eligible_count"].add_(
                float(realized.numel())
            )
        else:
            eligible_count = programming_eligible.to(torch.float64).sum()
            accumulator["programming_eligible_count"].add_(eligible_count)
            accumulator["programming_suppressed_count"].add_(
                float(realized.numel()) - eligible_count
            )
        if write_probability is not None:
            probability_sum = (
                float(write_probability) * float(realized.numel())
                if isinstance(write_probability, float)
                else write_probability.to(torch.float64).sum()
            )
            accumulator["write_probability_sum"].add_(probability_sum)
            accumulator["write_probability_count"].add_(
                float(realized.numel())
            )
        accumulator["element_updates"].add_(float(realized.numel()))

    def _binding_report(self, key: str) -> dict[str, Any]:
        table = self._plant.tables[key]
        if not self._initialized:
            return {
                "shape": list(table.shape),
                "assignment_sha256": table.assignment_sha256,
                "initialized": False,
            }
        accumulator = self._accumulators[key]
        count = float(accumulator["element_updates"].item())
        shadow = self._shadows[key]
        binding = next(item for item in self._bindings if item.key == key)
        lower = float(binding.parameter.min_cond)
        upper = float(binding.parameter.max_cond)
        proposed_sq = float(accumulator["proposed_update_squared_sum"].item())
        applied_sq = float(accumulator["applied_update_squared_sum"].item())
        gate_active = (
            self._config.programming_deadband_mode != "none"
            or self._config.probabilistic_write_mode != "none"
        )
        pending = (
            (shadow - self._last_programmed_shadows[key])
            .to(torch.float64)
            .abs()
            if gate_active
            else None
        )
        probability_count = float(
            accumulator["write_probability_count"].item()
        )
        return {
            "shape": list(table.shape),
            "assignment_sha256": table.assignment_sha256,
            "initialized": True,
            "initial_write": deepcopy(
                self._initialization_reports.get(key)
            ),
            "programming_deadband_mode": (
                self._config.programming_deadband_mode
            ),
            "programming_deadband_relative": float(
                self._config.programming_deadband_relative
            ),
            "programming_threshold_s": self._programming_threshold_s[key],
            "probabilistic_write_mode": (
                self._config.probabilistic_write_mode
            ),
            "probabilistic_write_probability": float(
                self._config.probabilistic_write_probability
            ),
            "probabilistic_write_scale_relative": float(
                self._config.probabilistic_write_scale_relative
            ),
            "probabilistic_write_scale_s": (
                self._probabilistic_write_scale_s[key]
            ),
            "pending_programming_mean_abs_s": (
                0.0 if pending is None else float(pending.mean().item())
            ),
            "pending_programming_rms_s": (
                0.0
                if pending is None
                else float(torch.sqrt(pending.square().mean()).item())
            ),
            "pending_programming_max_abs_s": (
                0.0 if pending is None else float(pending.max().item())
            ),
            "shadow_bound_fraction": float(
                ((shadow <= lower) | (shadow >= upper))
                .to(torch.float64)
                .mean()
                .item()
            ),
            "projection_mean_abs_error_s": (
                None
                if count == 0.0
                else float(accumulator["projection_abs_error_sum"].item())
                / count
            ),
            "projection_rms_error_s": (
                None
                if count == 0.0
                else math.sqrt(
                    float(accumulator["projection_squared_error_sum"].item())
                    / count
                )
            ),
            "projection_max_abs_error_s": float(
                accumulator["projection_abs_error_max"].item()
            ),
            "shadow_clipped_fraction": (
                None
                if count == 0.0
                else float(accumulator["shadow_clipped_count"].item()) / count
            ),
            "mean_abs_pulse_jump": (
                None
                if count == 0.0
                else float(accumulator["pulse_jump_sum"].item()) / count
            ),
            "max_abs_pulse_jump": int(accumulator["pulse_jump_max"].item()),
            "pulse_changed_fraction": (
                None
                if count == 0.0
                else float(accumulator["pulse_changed_count"].item()) / count
            ),
            "programming_event_fraction": (
                None
                if count == 0.0
                else float(
                    accumulator["programming_eligible_count"].item()
                )
                / count
            ),
            "programming_suppressed_fraction": (
                None
                if count == 0.0
                else float(
                    accumulator["programming_suppressed_count"].item()
                )
                / count
            ),
            "mean_write_probability": (
                None
                if probability_count == 0.0
                else float(accumulator["write_probability_sum"].item())
                / probability_count
            ),
            "projection_efficiency_rms_ratio": (
                None
                if proposed_sq == 0.0
                else math.sqrt(applied_sq / proposed_sq)
            ),
        }


class MeasuredCohortAOptimizer(MeasuredTraceOptimizer):
    """Measured optimizer constrained to the off-chip training cohort A."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.cohort != "A":
            raise ValueError(
                "Expected measured_cohort_a parameters.cohort to equal 'A'. "
                f"Provided value: {self.cohort!r}."
            )


class MeasuredCohortAOnePulseDownOptimizer(MeasuredCohortAOptimizer):
    """Apply only one local pulse toward lower conductance per update.

    The teacher-mapped initialization remains a single global-nearest write.
    Fine-tuning has no digital target accumulator: a gradient strictly above
    its parameter's configured threshold advances the assigned virtual device
    by exactly one pulse, while every other gradient holds the device at its
    current pulse. Isotonic preprocessing is required so increasing the pulse
    index can never increase conductance.
    """

    _UPDATE_RULE = "positive_gradient_one_pulse_down_else_hold"
    _SUPPORTS_POSITIVE_GRADIENT_THRESHOLDS = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        expected = {
            "curve_preprocessing": "isotonic_nonincreasing",
            "source_traces_per_cell": 2,
            "initial_pulse_index": 0,
            "projection": "global_nearest",
            "programming_deadband_mode": "none",
            "programming_deadband_relative": 0.0,
            "probabilistic_write_mode": "none",
            "probabilistic_write_probability": 1.0,
            "probabilistic_write_scale_relative": 0.0,
        }
        mismatches = {
            name: {
                "expected": value,
                "provided": getattr(self._config, name),
            }
            for name, value in expected.items()
            if getattr(self._config, name) != value
        }
        allowed_mappings = {
            "paired_affine_common_window",
            "dual_rail_quad_common_window",
        }
        if self._config.initial_target_mapping not in allowed_mappings:
            mismatches["initial_target_mapping"] = {
                "expected": sorted(allowed_mappings),
                "provided": self._config.initial_target_mapping,
            }
        if mismatches:
            raise ValueError(
                "Expected measured_cohort_a_one_pulse_down to use the strict "
                "isotonic four- or eight-device local-pulse protocol. "
                f"Provided value: {mismatches!r}."
            )
        configured = dict(
            self._config.positive_gradient_threshold_by_parameter or ()
        )
        self._positive_gradient_threshold_by_parameter = {
            binding.key: float(configured.get(binding.key, 0.0))
            for binding in self._bindings
        }

    @property
    def _update_rule(self) -> str:
        if any(
            threshold > 0.0
            for threshold in (
                self._positive_gradient_threshold_by_parameter.values()
            )
        ):
            return (
                "gradient_above_parameter_threshold_one_pulse_down_else_hold"
            )
        return self._UPDATE_RULE

    @property
    def data_report(self) -> dict[str, Any]:
        report = super().data_report
        report.update(
            {
                "semantics": "cohort_a_isotonic_one_pulse_down_or_hold",
                "pulse_model": True,
                "fine_tuning_update_rule": self._update_rule,
                "gradient_gate": (
                    "raw_gradient_strictly_greater_than_parameter_threshold"
                ),
                "positive_gradient_threshold_by_parameter": dict(
                    self._positive_gradient_threshold_by_parameter
                ),
                "pulse_index_direction": (
                    "an increment of one selects the next non-increasing "
                    "conductance state"
                ),
                "global_nearest_initialization": True,
                "global_nearest_fine_tuning": False,
                "digital_shadow_accumulation": False,
                "learning_rate_magnitude_used": False,
            }
        )
        return report

    @staticmethod
    def _empty_accumulators(device: torch.device) -> dict[str, torch.Tensor]:
        result = MeasuredTraceOptimizer._empty_accumulators(device)
        zeros = lambda: torch.zeros((), dtype=torch.float64, device=device)
        result.update(
            {
                "down_request_count": zeros(),
                "down_applied_count": zeros(),
                "down_blocked_last_pulse_count": zeros(),
                "negative_gradient_hold_count": zeros(),
                "zero_gradient_hold_count": zeros(),
                "conductance_decrease_count": zeros(),
                "conductance_hold_count": zeros(),
                "conductance_increase_count": zeros(),
                "max_conductance_increase_s": zeros(),
            }
        )
        return result

    def _values_at_pulses(self, key, pulses):
        return self._plant.values_at_pulses(key, pulses)

    @staticmethod
    def _monotonic_tolerance(
        previous: torch.Tensor,
        realized: torch.Tensor,
    ) -> float:
        scale = max(
            float(previous.detach().abs().max().item()),
            float(realized.detach().abs().max().item()),
            torch.finfo(previous.dtype).tiny,
        )
        return 8.0 * torch.finfo(previous.dtype).eps * scale

    def initialize_from_reset_targets(self) -> dict[str, Any]:
        super().initialize_from_reset_targets()
        with torch.no_grad():
            for binding in self._bindings:
                key = binding.key
                realized = binding.state.detach().clone()
                # Fine-tuning starts from the physical state itself.  The
                # mapped target used for the one initialization write is not
                # retained as a digital accumulator.
                self._shadows[key] = realized.clone()
                self._last_programmed_shadows[key] = realized.clone()
                self._initialization_reports[key].update(
                    {
                        "fine_tuning_update_rule": self._update_rule,
                        "positive_gradient_threshold": (
                            self._positive_gradient_threshold_by_parameter[key]
                        ),
                        "fine_tuning_global_nearest_projection": False,
                        "fine_tuning_digital_shadow_accumulation": False,
                    }
                )
        return self.programming_report

    def initialize_at_pulse_zero(self) -> dict[str, Any]:
        raise RuntimeError(
            "Expected measured_cohort_a_one_pulse_down to use teacher-mapped "
            "reset-target initialization. Provided value: pulse-zero-only "
            "initialization."
        )

    def step(self, closure=None):
        if closure is not None:
            raise ValueError(
                "Expected one-pulse-down updates without an optimizer "
                f"closure. Provided value: {closure!r}."
            )
        if not self._initialized:
            raise RuntimeError(
                "Expected cohort-A reset-target initialization before the "
                "first one-pulse-down step. Provided value: uninitialized."
            )

        with torch.no_grad():
            for binding in self._bindings:
                gradient = binding.state.grad
                if gradient is None:
                    continue
                if (
                    tuple(gradient.shape) != tuple(binding.state.shape)
                    or not bool(torch.isfinite(gradient).all())
                ):
                    raise ValueError(
                        "Expected every one-pulse-down gradient to be finite "
                        "and match its conductance tensor. Provided value: "
                        f"key={binding.key!r}, shape={tuple(gradient.shape)!r}, "
                        f"expected_shape={tuple(binding.state.shape)!r}."
                    )

                key = binding.key
                previous = binding.state.detach().reshape(-1).clone()
                previous_pulses = self._plant.pulse_indices[key].to(torch.long)
                indexed_previous = self._values_at_pulses(
                    key,
                    previous_pulses,
                )
                tolerance = self._monotonic_tolerance(
                    previous,
                    indexed_previous,
                )
                if bool((previous - indexed_previous).abs().gt(tolerance).any()):
                    maximum = float(
                        (previous - indexed_previous).abs().max().item()
                    )
                    raise RuntimeError(
                        "Expected the live conductance to equal its recorded "
                        "pulse state before a one-pulse-down update. Provided "
                        f"value: key={key!r}, max_abs_error_s={maximum!r}."
                    )

                flat_gradient = gradient.detach().reshape(-1)
                threshold = self._positive_gradient_threshold_by_parameter[key]
                down_requested = flat_gradient > threshold
                at_last_pulse = (
                    previous_pulses
                    == self._config.expected_trace_length - 1
                )
                down_applied = down_requested & ~at_last_pulse
                pulses = previous_pulses + down_applied.to(torch.long)
                realized = self._values_at_pulses(key, pulses)
                tolerance = self._monotonic_tolerance(previous, realized)
                increases = realized > previous + tolerance
                if bool(increases.any()):
                    maximum = float((realized - previous).max().item())
                    raise RuntimeError(
                        "Expected an isotonic one-pulse-down update never to "
                        "increase conductance. Provided value: "
                        f"key={key!r}, max_increase_s={maximum!r}."
                    )

                shaped = realized.reshape(binding.state.shape)
                binding.state.copy_(shaped)
                self._shadows[key].copy_(shaped)
                self._last_programmed_shadows[key].copy_(shaped)
                self._accumulate(
                    key,
                    previous_shadow=previous,
                    previous_realized=previous,
                    realized=realized,
                    pulses=pulses,
                    clipped=torch.zeros_like(realized, dtype=torch.bool),
                    programming_eligible=down_applied,
                    write_probability=None,
                )
                accumulator = self._accumulators[key]
                accumulator["down_request_count"].add_(
                    down_requested.to(torch.float64).sum()
                )
                accumulator["down_applied_count"].add_(
                    down_applied.to(torch.float64).sum()
                )
                accumulator["down_blocked_last_pulse_count"].add_(
                    (down_requested & at_last_pulse).to(torch.float64).sum()
                )
                accumulator["negative_gradient_hold_count"].add_(
                    (flat_gradient < 0.0).to(torch.float64).sum()
                )
                accumulator["zero_gradient_hold_count"].add_(
                    (flat_gradient == 0.0).to(torch.float64).sum()
                )
                decrease = realized < previous - tolerance
                hold = ~decrease & ~increases
                accumulator["conductance_decrease_count"].add_(
                    decrease.to(torch.float64).sum()
                )
                accumulator["conductance_hold_count"].add_(
                    hold.to(torch.float64).sum()
                )
                accumulator["conductance_increase_count"].add_(
                    increases.to(torch.float64).sum()
                )
                accumulator["max_conductance_increase_s"].copy_(
                    torch.maximum(
                        accumulator["max_conductance_increase_s"],
                        (realized - previous).clamp_min(0.0).max().to(
                            torch.float64
                        ),
                    )
                )
                self._plant.pulse_indices[key] = pulses.to(torch.int16)
        self._step_count += 1
        return None

    def _binding_report(self, key: str) -> dict[str, Any]:
        report = super()._binding_report(key)
        if not self._initialized:
            return report
        accumulator = self._accumulators[key]
        count = float(accumulator["element_updates"].item())

        def fraction(name: str) -> float | None:
            return (
                None
                if count == 0.0
                else float(accumulator[name].item()) / count
            )

        pulses = self._plant.pulse_indices[key].to(torch.long)
        down_request_fraction = fraction("down_request_count")
        negative_fraction = fraction("negative_gradient_hold_count")
        zero_fraction = fraction("zero_gradient_hold_count")
        positive_fraction = (
            None
            if count == 0.0
            else 1.0 - float(negative_fraction) - float(zero_fraction)
        )
        report.update(
            {
                "fine_tuning_update_rule": self._update_rule,
                "gradient_gate": (
                    "raw_gradient_strictly_greater_than_parameter_threshold"
                ),
                "positive_gradient_threshold": (
                    self._positive_gradient_threshold_by_parameter[key]
                ),
                "learning_rate_magnitude_used": False,
                "digital_shadow_accumulation": False,
                "down_request_fraction": down_request_fraction,
                "threshold_eligible_fraction": down_request_fraction,
                "positive_gradient_fraction": positive_fraction,
                "positive_gradient_suppressed_by_threshold_fraction": (
                    None
                    if positive_fraction is None
                    else positive_fraction - float(down_request_fraction)
                ),
                "down_pulse_applied_fraction": fraction(
                    "down_applied_count"
                ),
                "down_request_blocked_at_last_pulse_fraction": fraction(
                    "down_blocked_last_pulse_count"
                ),
                "negative_gradient_hold_fraction": negative_fraction,
                "zero_gradient_hold_fraction": zero_fraction,
                "conductance_decrease_fraction": fraction(
                    "conductance_decrease_count"
                ),
                "conductance_hold_fraction": fraction(
                    "conductance_hold_count"
                ),
                "conductance_increase_fraction": fraction(
                    "conductance_increase_count"
                ),
                "max_conductance_increase_s": float(
                    accumulator["max_conductance_increase_s"].item()
                ),
                "one_pulse_down_invariant_passed": (
                    int(accumulator["pulse_jump_max"].item()) <= 1
                    and float(
                        accumulator["conductance_increase_count"].item()
                    )
                    == 0.0
                ),
                "current_pulse_index_min": int(pulses.min().item()),
                "current_pulse_index_max": int(pulses.max().item()),
                "current_pulse_index_mean": float(
                    pulses.to(torch.float64).mean().item()
                ),
                "current_last_pulse_fraction": float(
                    (
                        pulses
                        == self._config.expected_trace_length - 1
                    )
                    .to(torch.float64)
                    .mean()
                    .item()
                ),
            }
        )
        return report

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if not self._initialized:
            return
        with torch.no_grad():
            for binding in self._bindings:
                key = binding.key
                initial = self._initialization_reports.get(key, {})
                if initial.get("fine_tuning_update_rule") != self._update_rule:
                    raise ValueError(
                        "Expected a one-pulse-down checkpoint initialization "
                        "marker. Provided value: "
                        f"key={key!r}, report={initial!r}."
                    )
                expected = self._values_at_pulses(
                    key,
                    self._plant.pulse_indices[key].to(torch.long),
                ).reshape(binding.state.shape)
                tolerance = self._monotonic_tolerance(
                    binding.state,
                    expected,
                )
                values = (
                    binding.state,
                    self._shadows[key],
                    self._last_programmed_shadows[key],
                )
                if any(
                    bool((value - expected).abs().gt(tolerance).any())
                    for value in values
                ):
                    raise ValueError(
                        "Expected one-pulse-down live conductance, shadow, and "
                        "recorded pulse state to agree on resume. Provided "
                        f"value: key={key!r}, mismatch=true."
                    )


class MeasuredCohortBOptimizer(MeasuredTraceOptimizer):
    """Measured optimizer for deployment and fine-tuning on held-out cohort B."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.cohort != "B":
            raise ValueError(
                "Expected measured_cohort_b parameters.cohort to equal 'B'. "
                f"Provided value: {self.cohort!r}."
            )


class MeasuredCohortBLoRAOptimizer(MeasuredTraceOptimizer):
    """Freeze a cohort-B base and update fully reset physical LoRA factors."""

    _BASE_KEYS = (
        "base.dense_weight.0",
        "base.dense_weight.1",
    )
    _ADAPTER_KEYS = (
        "adapter.input_factor.0",
        "adapter.output_factor.0",
        "adapter.input_factor.1",
        "adapter.output_factor.1",
    )

    def __init__(
        self,
        optimizer: Any,
        catalog: ParameterCatalog,
        parameters: Mapping[str, Any],
        device_data_path: Path,
    ) -> None:
        base_bindings = tuple(
            catalog.by_key[key]
            for key in self._BASE_KEYS
            if key in catalog.by_key
        )
        adapter_bindings = tuple(
            binding
            for binding in catalog.trainable
            if isinstance(binding.parameter, DenseWeight)
        )
        provided_base = tuple(binding.key for binding in base_bindings)
        provided_adapter = tuple(binding.key for binding in adapter_bindings)
        if (
            provided_base != self._BASE_KEYS
            or provided_adapter != self._ADAPTER_KEYS
            or any(binding.trainable for binding in base_bindings)
        ):
            raise ValueError(
                "Expected measured_cohort_b_lora to expose exactly two frozen "
                "base dense weights and four trainable layerwise adapter "
                "factors in stable order. Provided value: "
                f"base={provided_base!r}, adapter={provided_adapter!r}."
            )
        self._base_bindings = base_bindings
        self._frozen_base_reports: dict[str, dict[str, Any]] = {}
        super().__init__(
            optimizer,
            catalog,
            parameters,
            device_data_path,
            projection_bindings=base_bindings + adapter_bindings,
        )
        if self.cohort != "B":
            raise ValueError(
                "Expected measured_cohort_b_lora parameters.cohort to equal "
                f"'B'. Provided value: {self.cohort!r}."
            )

    @property
    def frozen_base_report(self) -> dict[str, Any]:
        return deepcopy(self._frozen_base_reports)

    def initialize_from_loaded_targets(self) -> dict[str, Any]:
        raise RuntimeError(
            "Expected measured_cohort_b_lora to deploy the frozen base and "
            "reset all adapter devices together via "
            "initialize_from_base_and_reset_adapters(). Provided value: "
            "generic loaded-target initialization."
        )

    def initialize_from_base_and_reset_adapters(self) -> dict[str, Any]:
        """Map the loaded base once and put every LoRA device at RESET end."""

        if self._initialized:
            raise RuntimeError(
                "Expected measured LoRA initialization exactly once. "
                "Provided value: optimizer is already initialized."
            )
        with torch.no_grad():
            for binding in self._base_bindings:
                key = binding.key
                lower = float(binding.parameter.min_cond)
                upper = float(binding.parameter.max_cond)
                loaded = binding.state.detach().clone()
                clipped = (loaded < lower) | (loaded > upper)
                target = loaded.clamp(min=lower, max=upper)
                realized, pulses = self._project(key, target.reshape(-1))
                binding.state.copy_(realized.reshape(binding.state.shape))
                error = target.reshape(-1).sub(realized).to(torch.float64)
                pulse_long = pulses.to(torch.long)
                self._frozen_base_reports[key] = {
                    "kind": "loaded_target_global_nearest_frozen",
                    "shape": list(binding.state.shape),
                    "assignment_sha256": self._plant.tables[key].assignment_sha256,
                    "target_clipped_fraction": float(
                        clipped.to(torch.float64).mean().item()
                    ),
                    "target_conductance_min_s": float(target.min().item()),
                    "target_conductance_max_s": float(target.max().item()),
                    "realized_conductance_min_s": float(realized.min().item()),
                    "realized_conductance_max_s": float(realized.max().item()),
                    "projection_mean_abs_error_s": float(
                        error.abs().mean().item()
                    ),
                    "projection_rms_error_s": float(
                        torch.sqrt(error.square().mean()).item()
                    ),
                    "projection_max_abs_error_s": float(
                        error.abs().max().item()
                    ),
                    "pulse_index_min": int(pulse_long.min().item()),
                    "pulse_index_max": int(pulse_long.max().item()),
                    "pulse_index_mean": float(
                        pulse_long.to(torch.float64).mean().item()
                    ),
                    "pulse_zero_fraction": float(
                        (pulse_long == 0).to(torch.float64).mean().item()
                    ),
                    "last_pulse_fraction": float(
                        (
                            pulse_long
                            == self._config.expected_trace_length - 1
                        )
                        .to(torch.float64)
                        .mean()
                        .item()
                    ),
                }

            pulse = self._config.initial_pulse_index
            for binding in self._bindings:
                key = binding.key
                table = self._plant.tables[key]
                source = self._plant.source_curves[key]
                realized = (
                    table.alpha * source[table.source_left, pulse]
                    + (1.0 - table.alpha)
                    * source[table.source_right, pulse]
                ).reshape(table.shape)
                binding.state.copy_(realized)
                self._shadows[key] = realized.detach().clone()
                self._initialize_programming_state(key, realized)
                self._plant.pulse_indices[key] = torch.full(
                    (realized.numel(),),
                    pulse,
                    dtype=torch.int16,
                    device=realized.device,
                )
                self._accumulators[key] = self._empty_accumulators(
                    realized.device
                )
                self._initialization_reports[key] = {
                    "kind": "fully_reset_trace_endpoint",
                    "target_clipped_fraction": 0.0,
                    "projection_mean_abs_error_s": 0.0,
                    "projection_rms_error_s": 0.0,
                    "projection_max_abs_error_s": 0.0,
                    "pulse_index_min": pulse,
                    "pulse_index_max": pulse,
                    "pulse_index_mean": float(pulse),
                    "pulse_zero_fraction": float(pulse == 0),
                    "last_pulse_fraction": float(
                        pulse == self._config.expected_trace_length - 1
                    ),
                }
        self._initialized = True
        return {
            "semantics": (
                "frozen loaded base globally projected once onto cohort-B "
                "curves; every trainable LoRA device initialized at the "
                "configured fully reset trace endpoint"
            ),
            "pulse_model": False,
            "frozen_base": self.frozen_base_report,
            "lora": self.programming_report,
        }


__all__ = [
    "MeasuredCohortAOptimizer",
    "MeasuredCohortAOnePulseDownOptimizer",
    "MeasuredCohortBOptimizer",
    "MeasuredCohortBLoRAOptimizer",
    "MeasuredTraceOptimizer",
    "MeasuredTraceConfig",
]
