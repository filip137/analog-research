"""Pure, strict schema for ``mnist_relu_drn_kd.v1``."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping

from experiments.mnist_relu.config import (
    DataSettings,
    RuntimeSettings,
    _integer,
    _keys,
    _number,
    _object,
    _optional_batches,
    _parse_data,
    _parse_runtime,
)
from experiments.schema import RunMode, config_error, freeze_json
from model.resistive.device_config import (
    AIHWKIT_RERAM_CMO,
    IBM_AFM2025_PCM,
    WAN2022_PHYSICAL,
    device_programming_to_mapping,
    parse_device_programming_config,
)
from training.ibm_om.spec import (
    EXPLICIT_LAYER_FIELDS,
    NestedConfigError,
    flatten_nested,
    is_nested_config,
    legacy_explicit_defaults,
)


EXPERIMENT_ID = "mnist_relu_drn_kd.v1"
SCHEMA_VERSION = 1
MEASURED_COHORT_A_BACKENDS = frozenset(
    {
        "measured_cohort_a",
        "measured_cohort_a_sign_sgd",
        "measured_cohort_a_one_pulse_down",
    }
)
MEASURED_COHORT_B_BACKENDS = frozenset({"measured_cohort_b"})
MEASURED_BACKENDS = MEASURED_COHORT_A_BACKENDS | MEASURED_COHORT_B_BACKENDS
IBM_OM_DEPLOYED_RECOVERY_BACKEND = "ibm_om_deployed_recovery"
IBM_OM_DEPLOYED_RECOVERY_TTV2_AIHWKIT_METHOD = (
    "ttv2_aihwkit_1p1_minibatch_equation"
)
IBM_OM_DEPLOYED_RECOVERY_METHODS = frozenset(
    {
        "rail_refresh",
        "direct_pulse",
        "tiki_taka",
        "ttv2",
        IBM_OM_DEPLOYED_RECOVERY_TTV2_AIHWKIT_METHOD,
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS = frozenset(
    {
        "dual_rail_quad_common_window",
        "differential_pair_common_window",
        "shared_reset_relative_quad",
        "raw_active_p90_quad",
    }
)
_MEASURED_KEYS = {
    "curve_preprocessing",
    "split_seed",
    "assignment_seed",
    "formed_resistance_max_ohm",
    "cohort_fraction",
    "cohort",
    "source_traces_per_cell",
    "initial_pulse_index",
    "projection",
    "expected_trace_length",
    "initial_target_mapping",
    "programming_deadband_mode",
    "programming_deadband_relative",
    "probabilistic_write_mode",
    "probabilistic_write_probability",
    "probabilistic_write_scale_relative",
    "probabilistic_write_seed",
}
_MEASURED_OPTIONAL_KEYS = {
    "dual_rail_layout_by_parameter",
    "positive_gradient_threshold_by_parameter",
}


@dataclass(frozen=True)
class StudentModelSettings:
    dims: tuple[int, int, int]
    input_gain: float
    conductance_min: float
    conductance_max: float
    voltage_amp: float
    current_amp: float
    encoding: str
    include_biases: bool
    non_linearity: str
    quadratic_diode_param: Mapping[str, Any]
    exponential_diode_param: Mapping[str, Any]
    hard_sigmoid_param: Mapping[str, Any]
    amplification_indexing: str = "logical"


@dataclass(frozen=True)
class SolverSettings:
    inference_iterations: int
    training_iterations: int
    mode: str
    overrelaxation_factor: float


@dataclass(frozen=True)
class MappingSettings:
    scale_fractions: tuple[float, ...]
    scale_fraction_pairs: tuple[tuple[float, float], ...] | None
    range_placement: str
    calibration_examples: int
    calibration_batch_size: int
    logit_gain_min: float
    logit_gain_max: float
    logit_gain_steps: int


@dataclass(frozen=True)
class UpdateBackendSettings:
    type: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class TeacherSettings:
    type: str
    initialization: str
    conductance_bounds: tuple[float, float] | None


@dataclass(frozen=True)
class StudentTrainSettings:
    num_epochs: int
    learning_rates: tuple[float, float]
    temperature: float
    log_every: int
    max_batches: int | None
    max_validation_batches: int | None
    minimum_relative_kl_improvement: float
    selection_evaluation: str
    selection_metric: str
    selection_noise_repeats: int
    weight_modifier: UpdateBackendSettings
    selection_weight_modifier: UpdateBackendSettings
    update_backend: UpdateBackendSettings


@dataclass(frozen=True)
class StudentValidateSettings:
    split: str
    sample_limit: int | None
    weight_modifier: UpdateBackendSettings
    noise_repeats: int


@dataclass(frozen=True)
class StudentConfig:
    schema_version: int
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    teacher: TeacherSettings
    model: StudentModelSettings
    solver: SolverSettings
    mapping: MappingSettings
    modes: Mapping[str, Any]


@dataclass(frozen=True)
class StudentTrainSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    teacher: TeacherSettings
    model: StudentModelSettings
    solver: SolverSettings
    mapping: MappingSettings
    settings: StudentTrainSettings


@dataclass(frozen=True)
class StudentValidateSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    teacher: TeacherSettings
    model: StudentModelSettings
    solver: SolverSettings
    mapping: MappingSettings
    settings: StudentValidateSettings


def _empty_object(value: Any, path: str) -> Mapping[str, Any]:
    raw = _object(value, path)
    if raw:
        raise config_error(path, "to be an explicit empty JSON object", dict(raw))
    return MappingProxyType({})


def _parse_teacher(value: Any) -> TeacherSettings:
    path = "config.teacher"
    raw = _object(value, path)
    if "type" not in raw:
        raise config_error(path, "to contain 'type'", dict(raw))
    teacher_type = raw["type"]
    if teacher_type not in {"bias_free_relu", "bounded_drn"}:
        raise config_error(
            f"{path}.type",
            "to be 'bias_free_relu' or 'bounded_drn'",
            teacher_type,
        )
    expected_keys = {"type", "initialization"}
    if teacher_type == "bounded_drn":
        expected_keys.add("conductance_bounds")
    _keys(raw, path, expected_keys)
    expected_initialization = {
        "bias_free_relu": "signed_weight_mapping",
        "bounded_drn": "literal_named_weight_copy",
    }[teacher_type]
    if raw["initialization"] != expected_initialization:
        raise config_error(
            f"{path}.initialization",
            f"to equal {expected_initialization!r} for teacher type "
            f"{teacher_type!r}",
            raw["initialization"],
        )
    conductance_bounds = None
    if teacher_type == "bounded_drn":
        raw_bounds = raw["conductance_bounds"]
        if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 2:
            raise config_error(
                f"{path}.conductance_bounds",
                "to be [minimum, maximum]",
                raw_bounds,
            )
        lower = _number(
            raw_bounds[0],
            f"{path}.conductance_bounds[0]",
            minimum=0.0,
        )
        upper = _number(
            raw_bounds[1],
            f"{path}.conductance_bounds[1]",
            minimum=0.0,
        )
        if lower >= upper:
            raise config_error(
                f"{path}.conductance_bounds",
                "to satisfy 0 <= minimum < maximum",
                raw_bounds,
            )
        conductance_bounds = (lower, upper)
    return TeacherSettings(
        type=teacher_type,
        initialization=expected_initialization,
        conductance_bounds=conductance_bounds,
    )


def _parse_model(value: Any) -> StudentModelSettings:
    path = "config.model"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "dims",
            "input_gain",
            "conductance_min",
            "conductance_max",
            "voltage_amp",
            "current_amp",
            "encoding",
            "include_biases",
            "non_linearity",
        },
    )
    if not isinstance(raw["dims"], (list, tuple)) or tuple(raw["dims"]) != (1568, 100, 20):
        raise config_error(f"{path}.dims", "to equal [1568, 100, 20]", raw["dims"])
    if raw["encoding"] not in {"single", "differential"}:
        raise config_error(f"{path}.encoding", "to be 'single' or 'differential'", raw["encoding"])
    if raw["include_biases"] is not False:
        raise config_error(f"{path}.include_biases", "to be false", raw["include_biases"])
    if _number(raw["input_gain"], f"{path}.input_gain") != 100.0:
        raise config_error(f"{path}.input_gain", "to equal 100.0", raw["input_gain"])
    voltage_amp = _number(raw["voltage_amp"], f"{path}.voltage_amp")
    if voltage_amp <= 0.0:
        raise config_error(
            f"{path}.voltage_amp",
            "to be a finite number > 0",
            raw["voltage_amp"],
        )
    current_amp = _number(raw["current_amp"], f"{path}.current_amp")
    if current_amp <= 0.0:
        raise config_error(
            f"{path}.current_amp",
            "to be a finite number > 0",
            raw["current_amp"],
        )
    if raw["encoding"] == "single" and voltage_amp != 4.0:
        raise config_error(
            f"{path}.voltage_amp",
            "to equal 4.0 for legacy single-device encoding",
            raw["voltage_amp"],
        )
    if raw["encoding"] == "single" and current_amp != 0.25:
        raise config_error(
            f"{path}.current_amp",
            "to equal 0.25 for legacy single-device encoding",
            raw["current_amp"],
        )
    lower = _number(raw["conductance_min"], f"{path}.conductance_min")
    upper = _number(raw["conductance_max"], f"{path}.conductance_max")
    if lower >= upper:
        raise config_error(path, "to have conductance_min < conductance_max", dict(raw))
    non_linearity = _object(raw["non_linearity"], f"{path}.non_linearity")
    _keys(
        non_linearity,
        f"{path}.non_linearity",
        {"type", "quadratic_diode_param", "exponential_diode_param", "hard_sigmoid_param"},
    )
    if non_linearity["type"] != "perfect_diode":
        raise config_error(f"{path}.non_linearity.type", "to equal 'perfect_diode'", non_linearity["type"])
    return StudentModelSettings(
        dims=(1568, 100, 20),
        input_gain=100.0,
        conductance_min=lower,
        conductance_max=upper,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        encoding=raw["encoding"],
        include_biases=False,
        non_linearity="perfect_diode",
        quadratic_diode_param=_empty_object(non_linearity["quadratic_diode_param"], f"{path}.non_linearity.quadratic_diode_param"),
        exponential_diode_param=_empty_object(non_linearity["exponential_diode_param"], f"{path}.non_linearity.exponential_diode_param"),
        hard_sigmoid_param=_empty_object(non_linearity["hard_sigmoid_param"], f"{path}.non_linearity.hard_sigmoid_param"),
    )


def _parse_solver(value: Any) -> SolverSettings:
    path = "config.solver"
    raw = _object(value, path)
    _keys(raw, path, {"inference_iterations", "training_iterations", "mode", "overrelaxation_factor"})
    if raw["mode"] != "asynchronous":
        raise config_error(f"{path}.mode", "to equal 'asynchronous'", raw["mode"])
    return SolverSettings(
        inference_iterations=_integer(raw["inference_iterations"], f"{path}.inference_iterations", minimum=1),
        training_iterations=_integer(raw["training_iterations"], f"{path}.training_iterations", minimum=1),
        mode="asynchronous",
        overrelaxation_factor=_number(raw["overrelaxation_factor"], f"{path}.overrelaxation_factor"),
    )


def _parse_mapping(value: Any) -> MappingSettings:
    path = "config.mapping"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "scale_fractions",
            "calibration_examples",
            "calibration_batch_size",
            "logit_gain_min",
            "logit_gain_max",
            "logit_gain_steps",
        },
        {"scale_fraction_pairs", "range_placement"},
    )
    fractions = raw["scale_fractions"]
    if not isinstance(fractions, (list, tuple)) or not fractions:
        raise config_error(f"{path}.scale_fractions", "to be a non-empty list of values in (0, 1]", fractions)
    parsed = tuple(_number(item, f"{path}.scale_fractions[{index}]") for index, item in enumerate(fractions))
    if any(item <= 0.0 or item > 1.0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise config_error(f"{path}.scale_fractions", "to contain unique values in (0, 1]", fractions)
    raw_pairs = raw.get("scale_fraction_pairs")
    parsed_pairs = None
    if raw_pairs is not None:
        if not isinstance(raw_pairs, (list, tuple)) or not raw_pairs:
            raise config_error(
                f"{path}.scale_fraction_pairs",
                "to be null or a non-empty list of unique two-value lists in (0, 1]",
                raw_pairs,
            )
        candidate_pairs = []
        for index, pair in enumerate(raw_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise config_error(
                    f"{path}.scale_fraction_pairs[{index}]",
                    "to contain exactly two values in (0, 1]",
                    pair,
                )
            candidate = tuple(
                _number(item, f"{path}.scale_fraction_pairs[{index}][{offset}]")
                for offset, item in enumerate(pair)
            )
            if any(item <= 0.0 or item > 1.0 for item in candidate):
                raise config_error(
                    f"{path}.scale_fraction_pairs[{index}]",
                    "to contain exactly two values in (0, 1]",
                    pair,
                )
            candidate_pairs.append((candidate[0], candidate[1]))
        if len(set(candidate_pairs)) != len(candidate_pairs):
            raise config_error(
                f"{path}.scale_fraction_pairs",
                "to contain unique two-value lists",
                raw_pairs,
            )
        parsed_pairs = tuple(candidate_pairs)
    range_placement = raw.get("range_placement", "lower")
    if range_placement not in {"lower", "centered"}:
        raise config_error(
            f"{path}.range_placement",
            "to be 'lower' or 'centered'",
            range_placement,
        )
    gain_min = _number(raw["logit_gain_min"], f"{path}.logit_gain_min")
    gain_max = _number(raw["logit_gain_max"], f"{path}.logit_gain_max")
    if gain_min <= 0.0 or gain_min >= gain_max:
        raise config_error(path, "to have 0 < logit_gain_min < logit_gain_max", dict(raw))
    return MappingSettings(
        scale_fractions=parsed,
        scale_fraction_pairs=parsed_pairs,
        range_placement=range_placement,
        calibration_examples=_integer(raw["calibration_examples"], f"{path}.calibration_examples", minimum=1),
        calibration_batch_size=_integer(raw["calibration_batch_size"], f"{path}.calibration_batch_size", minimum=1),
        logit_gain_min=gain_min,
        logit_gain_max=gain_max,
        logit_gain_steps=_integer(raw["logit_gain_steps"], f"{path}.logit_gain_steps", minimum=3),
    )


def _parse_measured(
    value: Any,
    path: str,
    *,
    backend_type: str,
) -> Mapping[str, Any]:
    raw = _object(value, path)
    _keys(raw, path, _MEASURED_KEYS, _MEASURED_OPTIONAL_KEYS)
    if raw["curve_preprocessing"] not in {"raw", "isotonic_nonincreasing"}:
        raise config_error(f"{path}.curve_preprocessing", "to be 'raw' or 'isotonic_nonincreasing'", raw["curve_preprocessing"])
    if raw["initial_target_mapping"] not in {
        "literal",
        "per_device_affine",
        "paired_affine_common_window",
        "dual_rail_pairwise_common_window",
        "dual_rail_quad_common_window",
    }:
        raise config_error(
            f"{path}.initial_target_mapping",
            "to be 'literal', 'per_device_affine', "
            "'paired_affine_common_window', or "
            "a dual-rail common-window mapping "
            "('dual_rail_pairwise_common_window' or "
            "'dual_rail_quad_common_window')",
            raw["initial_target_mapping"],
        )
    expected_cohort = {
        "measured_cohort_a": "A",
        "measured_cohort_a_sign_sgd": "A",
        "measured_cohort_a_one_pulse_down": "A",
        "measured_cohort_b": "B",
    }[backend_type]
    if (
        expected_cohort == "B"
        and raw["initial_target_mapping"]
        in {
            "per_device_affine",
            "dual_rail_pairwise_common_window",
        }
    ):
        raise config_error(
            f"{path}.initial_target_mapping",
            "to be 'literal' or a matched common-window mapping "
            "('paired_affine_common_window' or "
            "'dual_rail_quad_common_window') for cohort B",
            raw["initial_target_mapping"],
        )
    layouts = raw.get("dual_rail_layout_by_parameter")
    if raw["initial_target_mapping"] in {
        "dual_rail_pairwise_common_window",
        "dual_rail_quad_common_window",
    }:
        if not isinstance(layouts, Mapping) or not layouts:
            raise config_error(
                f"{path}.dual_rail_layout_by_parameter",
                "to map stable single-conductance parameter keys to "
                "'halves' or 'paired' for a dual-rail common-window mapping",
                layouts,
            )
        invalid = {
            key: layout
            for key, layout in layouts.items()
            if not isinstance(key, str)
            or layout not in {"halves", "paired"}
        }
        if invalid:
            raise config_error(
                f"{path}.dual_rail_layout_by_parameter",
                "to map stable parameter keys to 'halves' or 'paired'",
                layouts,
            )
    elif layouts is not None:
        raise config_error(
            f"{path}.dual_rail_layout_by_parameter",
            "to be omitted or null unless initial_target_mapping is "
            "a dual-rail common-window mapping",
            layouts,
        )
    exact = {
        "cohort": expected_cohort,
        "cohort_fraction": 0.5,
        "source_traces_per_cell": 2,
        "initial_pulse_index": 0,
        "projection": "global_nearest",
        "programming_deadband_mode": "none",
        "programming_deadband_relative": 0.0,
        "probabilistic_write_mode": "none",
        "probabilistic_write_probability": 1.0,
        "probabilistic_write_scale_relative": 0.0,
        "probabilistic_write_seed": 0,
    }
    for name, expected in exact.items():
        if raw[name] != expected:
            raise config_error(f"{path}.{name}", f"to equal {expected!r}", raw[name])
    for name in ("split_seed", "assignment_seed"):
        _integer(raw[name], f"{path}.{name}")
    _integer(raw["expected_trace_length"], f"{path}.expected_trace_length", minimum=2)
    if _number(raw["formed_resistance_max_ohm"], f"{path}.formed_resistance_max_ohm") <= 0.0:
        raise config_error(f"{path}.formed_resistance_max_ohm", "to be positive", raw["formed_resistance_max_ohm"])
    normalized = dict(raw)
    normalized.setdefault("dual_rail_layout_by_parameter", None)
    raw_thresholds = raw.get("positive_gradient_threshold_by_parameter")
    if backend_type != "measured_cohort_a_one_pulse_down":
        if raw_thresholds is not None:
            raise config_error(
                f"{path}.positive_gradient_threshold_by_parameter",
                "to be omitted or null unless update_backend.type is "
                "'measured_cohort_a_one_pulse_down'",
                raw_thresholds,
            )
        normalized["positive_gradient_threshold_by_parameter"] = None
    elif raw_thresholds is None:
        # Omission is the exact zero-threshold historical control.
        normalized["positive_gradient_threshold_by_parameter"] = None
    else:
        mapping = raw["initial_target_mapping"]
        expected_threshold_keys = (
            {
                "base.conductance_plus.0",
                "base.conductance_minus.0",
                "base.conductance_plus.1",
                "base.conductance_minus.1",
            }
            if mapping == "paired_affine_common_window"
            else {
                "base.dense_weight.0",
                "base.dense_weight.1",
            }
        )
        if not isinstance(raw_thresholds, Mapping) or (
            set(raw_thresholds) != expected_threshold_keys
        ):
            raise config_error(
                f"{path}.positive_gradient_threshold_by_parameter",
                "to map exactly the stable parameter keys "
                f"{sorted(expected_threshold_keys)!r} to finite "
                "non-negative raw-gradient thresholds",
                raw_thresholds,
            )
        parsed_thresholds = {
            key: _number(
                raw_thresholds[key],
                f"{path}.positive_gradient_threshold_by_parameter.{key}",
            )
            for key in sorted(expected_threshold_keys)
        }
        if any(value < 0.0 for value in parsed_thresholds.values()):
            raise config_error(
                f"{path}.positive_gradient_threshold_by_parameter",
                "to map exactly the stable parameter keys "
                f"{sorted(expected_threshold_keys)!r} to finite "
                "non-negative raw-gradient thresholds",
                raw_thresholds,
            )
        normalized["positive_gradient_threshold_by_parameter"] = (
            parsed_thresholds
        )
    if backend_type == "measured_cohort_a_one_pulse_down":
        if normalized["curve_preprocessing"] != "isotonic_nonincreasing":
            raise config_error(
                f"{path}.curve_preprocessing",
                "to equal 'isotonic_nonincreasing' for "
                "measured_cohort_a_one_pulse_down",
                normalized["curve_preprocessing"],
            )
        allowed_mappings = {
            "paired_affine_common_window",
            "dual_rail_quad_common_window",
        }
        if normalized["initial_target_mapping"] not in allowed_mappings:
            raise config_error(
                f"{path}.initial_target_mapping",
                "to be 'paired_affine_common_window' or "
                "'dual_rail_quad_common_window' for "
                "measured_cohort_a_one_pulse_down",
                normalized["initial_target_mapping"],
            )
    return freeze_json(normalized, path=path)


def _normalize_explicit_layer_fields(
    parameters: dict[str, Any],
    *,
    target_mapping: str,
    path: str,
) -> None:
    """Declare the coordinate, start, tolerance, clamp and characterization.

    Absent fields take the value ``target_mapping`` supports; declared
    fields must equal it, because no other combination is implemented.
    """

    supported = legacy_explicit_defaults(target_mapping)
    for name in EXPLICIT_LAYER_FIELDS:
        value = parameters.get(name)
        if value is None:
            value = supported[name]
        elif name == "clamp_to_parameter_bounds":
            if not isinstance(value, bool):
                raise config_error(f"{path}.{name}", "to be a boolean", value)
        elif name == "verify_tolerance_absolute":
            value = _number(value, f"{path}.{name}")
        elif not isinstance(value, str):
            raise config_error(f"{path}.{name}", "to be a string", value)
        if value != supported[name]:
            raise config_error(
                f"{path}.{name}",
                f"to equal {supported[name]!r} for target_mapping "
                f"{target_mapping!r}",
                value,
            )
        parameters[name] = value


def _parse_weight_modifier(value: Any, path: str) -> UpdateBackendSettings:
    raw = _object(value, path)
    _keys(raw, path, {"type", "parameters"})
    modifier_type = raw["type"]
    parameters_path = f"{path}.parameters"
    parameters = _object(raw["parameters"], parameters_path)
    if modifier_type == "none":
        return UpdateBackendSettings(
            type="none",
            parameters=_empty_object(parameters, parameters_path),
        )
    if modifier_type == "ibm_reram_om_program_verify":
        if is_nested_config(parameters):
            # Nested sections are an input format; everything downstream
            # (metadata, checkpoints, bundles) sees the canonical flat dict.
            try:
                parameters = flatten_nested(parameters)
            except NestedConfigError as error:
                raise config_error(
                    f"{parameters_path}.{error.location}",
                    error.expectation,
                    error.value,
                ) from error
        required = {
            "execution",
            "assignment_seed",
            "endpoint_seed",
            "corruption_policy",
            "noisy_evaluation",
            "endpoint_policy",
            "target_out_of_support",
            "preset",
            "controller",
            "start_protocol",
            "tolerance_step_ratio",
            "maximum_program_pulses",
        }
        mapping_fields = {
            "target_mapping",
            "dual_rail_layout_by_parameter",
            "common_window_margin_fraction",
        }
        reset_relative_fields = {
            "reset_relative_mode",
            "reset_relative_contrast_step",
            "reset_read_samples",
            "reset_guard_standard_errors",
        }
        raw_active_fields = {
            "raw_active_mode",
            "raw_active_unsupported_quad_policy",
        }
        _keys(
            parameters,
            parameters_path,
            required,
            mapping_fields
            | reset_relative_fields
            | raw_active_fields
            | {"forward_logit_gain"}
            | set(EXPLICIT_LAYER_FIELDS),
        )
        provided_mapping_fields = mapping_fields & set(parameters)
        if provided_mapping_fields and provided_mapping_fields != mapping_fields:
            raise config_error(
                parameters_path,
                "to provide all three target-mapping fields together: "
                "target_mapping, dual_rail_layout_by_parameter, and "
                "common_window_margin_fraction",
                dict(parameters),
            )
        normalized_parameters = dict(parameters)
        if not provided_mapping_fields:
            # Legacy pilot configs retain their literal global-coordinate path.
            normalized_parameters.update(
                {
                    "target_mapping": "literal_global",
                    "dual_rail_layout_by_parameter": None,
                    "common_window_margin_fraction": 0.0,
                }
            )
        target_mapping = normalized_parameters["target_mapping"]
        if target_mapping not in {
            "literal_global",
            "dual_rail_quad_common_window",
            "differential_pair_common_window",
            "shared_reset_relative_quad",
            "raw_active_p90_quad",
        }:
            raise config_error(
                f"{parameters_path}.target_mapping",
                "to be 'literal_global' or "
                "an array-specific common-window mapping "
                "('dual_rail_quad_common_window', "
                "'differential_pair_common_window', or "
                "'shared_reset_relative_quad', or "
                "'raw_active_p90_quad')",
                target_mapping,
            )
        margin = _number(
            normalized_parameters["common_window_margin_fraction"],
            f"{parameters_path}.common_window_margin_fraction",
        )
        if margin < 0.0 or margin >= 0.5:
            raise config_error(
                f"{parameters_path}.common_window_margin_fraction",
                "to be a finite number in [0, 0.5)",
                margin,
            )
        normalized_parameters["common_window_margin_fraction"] = margin
        raw_layouts = normalized_parameters["dual_rail_layout_by_parameter"]
        if target_mapping in {
            "dual_rail_quad_common_window",
            "shared_reset_relative_quad",
            "raw_active_p90_quad",
        }:
            expected_layouts = {
                "base.dense_weight.0": "halves",
                "base.dense_weight.1": "paired",
            }
            if not isinstance(raw_layouts, Mapping) or dict(raw_layouts) != expected_layouts:
                raise config_error(
                    f"{parameters_path}.dual_rail_layout_by_parameter",
                    "to equal the canonical W1/W2 layout "
                    f"{expected_layouts!r}",
                    raw_layouts,
                )
            normalized_parameters["dual_rail_layout_by_parameter"] = dict(
                raw_layouts
            )
        if target_mapping == "shared_reset_relative_quad":
            if margin != 0.0:
                raise config_error(
                    f"{parameters_path}.common_window_margin_fraction",
                    "to equal 0.0 for shared_reset_relative_quad",
                    margin,
                )
            if not reset_relative_fields <= set(parameters):
                raise config_error(
                    parameters_path,
                    "to provide all RESET-relative commissioning fields for "
                    "shared_reset_relative_quad",
                    dict(parameters),
                )
            mode = normalized_parameters["reset_relative_mode"]
            if mode not in {"continuous", "quantized_9_level"}:
                raise config_error(
                    f"{parameters_path}.reset_relative_mode",
                    "to be 'continuous' or 'quantized_9_level'",
                    mode,
                )
            step = _number(
                normalized_parameters["reset_relative_contrast_step"],
                f"{parameters_path}.reset_relative_contrast_step",
            )
            if step <= 0.0 or step > 0.5:
                raise config_error(
                    f"{parameters_path}.reset_relative_contrast_step",
                    "to be in (0, 0.5]",
                    step,
                )
            normalized_parameters["reset_relative_contrast_step"] = step
            normalized_parameters["reset_read_samples"] = _integer(
                normalized_parameters["reset_read_samples"],
                f"{parameters_path}.reset_read_samples",
                minimum=2,
            )
            guard = _number(
                normalized_parameters["reset_guard_standard_errors"],
                f"{parameters_path}.reset_guard_standard_errors",
            )
            if guard < 0.0:
                raise config_error(
                    f"{parameters_path}.reset_guard_standard_errors",
                    "to be non-negative",
                    guard,
                )
            normalized_parameters["reset_guard_standard_errors"] = guard
            normalized_parameters.setdefault("forward_logit_gain", None)
        elif target_mapping == "raw_active_p90_quad":
            if margin != 0.0:
                raise config_error(
                    f"{parameters_path}.common_window_margin_fraction",
                    "to equal 0.0 for raw_active_p90_quad",
                    margin,
                )
            if not raw_active_fields <= set(parameters):
                raise config_error(
                    parameters_path,
                    "to provide raw_active_mode and "
                    "raw_active_unsupported_quad_policy for "
                    "raw_active_p90_quad",
                    dict(parameters),
                )
            mode = normalized_parameters["raw_active_mode"]
            if mode not in {"continuous", "quantized_7_level"}:
                raise config_error(
                    f"{parameters_path}.raw_active_mode",
                    "to be 'continuous' or 'quantized_7_level'",
                    mode,
                )
            policy = normalized_parameters[
                "raw_active_unsupported_quad_policy"
            ]
            if policy != "structural_failure":
                raise config_error(
                    f"{parameters_path}.raw_active_unsupported_quad_policy",
                    "to equal 'structural_failure' until a donor-stream "
                    "assignment protocol is versioned",
                    policy,
                )
            if reset_relative_fields & set(parameters):
                raise config_error(
                    parameters_path,
                    "to omit RESET-relative fields for raw_active_p90_quad",
                    dict(parameters),
                )
            normalized_parameters.setdefault("forward_logit_gain", None)
        elif reset_relative_fields & set(parameters):
            raise config_error(
                parameters_path,
                "to omit RESET-relative commissioning fields unless "
                "target_mapping is shared_reset_relative_quad",
                dict(parameters),
            )
        elif raw_active_fields & set(parameters):
            raise config_error(
                parameters_path,
                "to omit raw-active fields unless target_mapping is "
                "raw_active_p90_quad",
                dict(parameters),
            )
        elif target_mapping == "differential_pair_common_window":
            if raw_layouts is not None:
                raise config_error(
                    f"{parameters_path}.dual_rail_layout_by_parameter",
                    "to be null for differential_pair_common_window; the "
                    "core mapper enforces the canonical adjacent plus/minus "
                    "catalog",
                    raw_layouts,
                )
        elif target_mapping == "literal_global" and (
            raw_layouts is not None or margin != 0.0
        ):
            raise config_error(
                parameters_path,
                "to use null dual_rail_layout_by_parameter and zero "
                "common_window_margin_fraction for literal_global",
                dict(normalized_parameters),
            )
        if "forward_logit_gain" in normalized_parameters:
            raw_gain = normalized_parameters["forward_logit_gain"]
            if raw_gain is not None:
                gain = _number(
                    raw_gain,
                    f"{parameters_path}.forward_logit_gain",
                )
                if gain <= 0.0:
                    raise config_error(
                        f"{parameters_path}.forward_logit_gain",
                        "to be null or positive",
                        gain,
                    )
                normalized_parameters["forward_logit_gain"] = gain
        raw_active = target_mapping == "raw_active_p90_quad"
        exact = {
            "preset": "reram_array_om",
            "controller": "one_pulse" if raw_active else "adaptive",
            "start_protocol": "lower_to_target",
            "tolerance_step_ratio": 0.5,
            "maximum_program_pulses": 128,
            "endpoint_policy": "preserve" if raw_active else "clip_0_1",
            "target_out_of_support": "error",
        }
        for name, expected in exact.items():
            if normalized_parameters[name] != expected:
                raise config_error(
                    f"{parameters_path}.{name}",
                    f"to equal {expected!r}",
                    normalized_parameters[name],
                )
        if normalized_parameters["execution"] not in {
            "mapped_target",
            "compact_endpoint",
            "pulse_resolved",
        }:
            raise config_error(
                f"{parameters_path}.execution",
                "to be 'mapped_target', 'compact_endpoint', or "
                "'pulse_resolved'",
                normalized_parameters["execution"],
            )
        if (
            normalized_parameters["execution"] == "mapped_target"
            and not raw_active
        ):
            raise config_error(
                f"{parameters_path}.execution",
                "to use mapped_target only for raw_active_p90_quad",
                normalized_parameters["execution"],
            )
        if raw_active and normalized_parameters["execution"] == "compact_endpoint":
            raise config_error(
                f"{parameters_path}.execution",
                "not to reuse the historical q-coordinate compact endpoint "
                "model for raw_active_p90_quad",
                normalized_parameters["execution"],
            )
        if normalized_parameters["corruption_policy"] not in {
            "counterfactual_repaired",
            "published",
        }:
            raise config_error(
                f"{parameters_path}.corruption_policy",
                "to be 'counterfactual_repaired' or 'published'",
                normalized_parameters["corruption_policy"],
            )
        for name in ("assignment_seed", "endpoint_seed"):
            seed = _integer(
                normalized_parameters[name], f"{parameters_path}.{name}"
            )
            if seed >= 2**63:
                raise config_error(
                    f"{parameters_path}.{name}",
                    "to be an integer in [0, 2**63)",
                    seed,
                )
        if not isinstance(normalized_parameters["noisy_evaluation"], bool):
            raise config_error(
                f"{parameters_path}.noisy_evaluation",
                "to be a boolean",
                normalized_parameters["noisy_evaluation"],
            )
        _normalize_explicit_layer_fields(
            normalized_parameters,
            target_mapping=target_mapping,
            path=parameters_path,
        )
        return UpdateBackendSettings(
            type=modifier_type,
            parameters=freeze_json(
                normalized_parameters,
                path=parameters_path,
            ),
        )
    if modifier_type != "add_normal":
        raise config_error(
            f"{path}.type",
            "to be 'none', 'add_normal', or "
            "'ibm_reram_om_program_verify'",
            modifier_type,
        )
    _keys(
        parameters,
        parameters_path,
        {"std_dev", "seed", "noisy_evaluation", "scale_mode"},
    )
    std_dev = _number(parameters["std_dev"], f"{parameters_path}.std_dev")
    seed = _integer(parameters["seed"], f"{parameters_path}.seed")
    if seed > 2**64 - 1:
        raise config_error(
            f"{parameters_path}.seed",
            "to be an integer in [0, 2**64 - 1]",
            seed,
        )
    if not isinstance(parameters["noisy_evaluation"], bool):
        raise config_error(
            f"{parameters_path}.noisy_evaluation",
            "to be a boolean",
            parameters["noisy_evaluation"],
        )
    scale_mode = parameters["scale_mode"]
    if scale_mode not in {"tensor_abs_max", "output_channel_abs_max"}:
        raise config_error(
            f"{parameters_path}.scale_mode",
            "to be 'tensor_abs_max' or 'output_channel_abs_max'",
            scale_mode,
        )
    return UpdateBackendSettings(
        type="add_normal",
        parameters=freeze_json(
            {
                "std_dev": std_dev,
                "seed": seed,
                "noisy_evaluation": parameters["noisy_evaluation"],
                "scale_mode": scale_mode,
            },
            path=parameters_path,
        ),
    )


def _parse_program_verify(value: Any, path: str) -> Mapping[str, Any]:
    raw = _object(value, path)
    _keys(raw, path, {"device"})
    try:
        device = parse_device_programming_config(
            raw["device"],
            path=f"{path}.device",
        )
    except ValueError as error:
        raise config_error(
            f"{path}.device",
            "to contain a valid repeated-write endpoint device model",
            raw["device"],
        ) from error
    if device.type not in {
        IBM_AFM2025_PCM,
        WAN2022_PHYSICAL,
        AIHWKIT_RERAM_CMO,
    }:
        raise config_error(
            f"{path}.device.type",
            "to select a device model with repeated-write support",
            device.type,
        )
    return freeze_json(
        {"device": device_programming_to_mapping(device)},
        path=path,
    )


def _parse_optional_seed(value: Any, path: str) -> int | None:
    if value is None:
        return None
    result = _integer(value, path, minimum=0)
    if result >= 2**63:
        raise config_error(path, "to be an integer in [0, 2**63)", result)
    return result


def _parse_optional_positive(value: Any, path: str) -> float | None:
    if value is None:
        return None
    result = _number(value, path)
    if result <= 0.0:
        raise config_error(path, "to be a positive finite number or null", value)
    return result


def _parse_ibm_om_deployed_recovery(value: Any, path: str) -> Mapping[str, Any]:
    raw = _object(value, path)
    keys = {
        "method",
        "slow_pulse_budget_per_cell",
        "pulse_step_ratio",
        "update_seed",
        "allow_off_grid",
        "recalibrate_logit_gain_each_epoch",
        "terminal_epoch",
        "source_forward_logit_gain",
        "expected_source_weights_sha256",
        "expected_source_selection_epoch",
        "expected_device_model_sha256",
        "expected_assignment_seed",
        "expected_endpoint_seed",
        "expected_target_mapping",
        "expected_reset_relative_mode",
        "expected_reset_relative_contrast_step",
        "expected_corruption_policy",
        "expected_reset_read_samples",
        "expected_reset_guard_standard_errors",
        "source_dual_rail_layout_by_parameter",
        "direct_probability_calibration_batches",
        "direct_probability_scale",
        "fast_backend",
        "fast_assignment_seed",
        "fast_endpoint_seed",
        "fast_reset_read_samples",
        "fast_reset_guard_standard_errors",
    }
    optional_keys = {
        "direct_gradient_magnitude_percentile",
        "direct_probability_scale_source_report_sha256",
        "ttv2_transfer_every",
        "ttv2_gamma0",
        "ttv2_fast_update_model",
        "ttv2_fast_weight_limit",
        "ttv2_scan_mode",
        "ttv2_buffer_threshold",
        "ttv2_buffer_residual_mode",
        "ttv2_fast_lr_by_parameter",
        "ttv2_transfer_lr",
        "ttv2_scale_transfer_lr",
        "ttv2_units_in_mbatch",
        "ttv2_auto_scale",
        "ttv2_fast_granularity_by_parameter",
        "ttv2_buffer_granularity",
        "ttv2_auto_granularity",
        "ttv2_correct_gradient_magnitudes",
        "ttv2_desired_bl",
        "ttv2_momentum",
        "ttv2_forget_buffer",
        "ttv2_cap_scope",
        "ttv2_cursor_policy",
        "ttv2_in_chop_probability",
    }
    _keys(raw, path, keys, optional_keys)
    method = raw["method"]
    if method not in IBM_OM_DEPLOYED_RECOVERY_METHODS:
        raise config_error(
            f"{path}.method",
            "to be 'rail_refresh', 'direct_pulse', 'tiki_taka', 'ttv2', "
            "or 'ttv2_aihwkit_1p1_minibatch_equation'",
            method,
        )
    budget = _number(raw["slow_pulse_budget_per_cell"], f"{path}.slow_pulse_budget_per_cell")
    if budget not in {0.1, 1.0, 4.0}:
        raise config_error(
            f"{path}.slow_pulse_budget_per_cell",
            "to equal 0.1, 1, or 4",
            budget,
        )
    pulse_step = _number(raw["pulse_step_ratio"], f"{path}.pulse_step_ratio")
    if pulse_step != 0.04745:
        raise config_error(f"{path}.pulse_step_ratio", "to equal 0.04745", pulse_step)
    update_seed = _parse_optional_seed(raw["update_seed"], f"{path}.update_seed")
    if update_seed is None:
        raise config_error(f"{path}.update_seed", "to be an integer seed", None)
    exact = {
        "allow_off_grid": True,
        "recalibrate_logit_gain_each_epoch": True,
        "expected_target_mapping": "shared_reset_relative_quad",
        "expected_reset_relative_mode": "quantized_9_level",
        "expected_reset_relative_contrast_step": 0.095849,
        "expected_corruption_policy": "counterfactual_repaired",
        "expected_reset_read_samples": 8,
        "expected_reset_guard_standard_errors": 3.0,
    }
    for name, expected in exact.items():
        if raw[name] != expected:
            raise config_error(f"{path}.{name}", f"to equal {expected!r}", raw[name])
    terminal_epoch = _integer(raw["terminal_epoch"], f"{path}.terminal_epoch", minimum=1)
    source_gain = _number(raw["source_forward_logit_gain"], f"{path}.source_forward_logit_gain")
    if source_gain <= 0.0:
        raise config_error(f"{path}.source_forward_logit_gain", "to be positive", source_gain)
    digests = {}
    for name in ("expected_source_weights_sha256", "expected_device_model_sha256"):
        digest = raw[name]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise config_error(f"{path}.{name}", "to be a lowercase SHA-256 digest", digest)
        digests[name] = digest
    selection_epoch = _integer(
        raw["expected_source_selection_epoch"],
        f"{path}.expected_source_selection_epoch",
        minimum=-1,
    )
    assignment_seed = _parse_optional_seed(raw["expected_assignment_seed"], f"{path}.expected_assignment_seed")
    endpoint_seed = _parse_optional_seed(raw["expected_endpoint_seed"], f"{path}.expected_endpoint_seed")
    assert assignment_seed is not None and endpoint_seed is not None
    source_layouts = _object(
        raw["source_dual_rail_layout_by_parameter"],
        f"{path}.source_dual_rail_layout_by_parameter",
    )
    expected_source_layouts = {
        "base.dense_weight.0": "halves",
        "base.dense_weight.1": "paired",
    }
    if source_layouts != expected_source_layouts:
        raise config_error(
            f"{path}.source_dual_rail_layout_by_parameter",
            f"to equal {expected_source_layouts!r}",
            source_layouts,
        )

    calibration_batches = raw["direct_probability_calibration_batches"]
    probability_scale = _parse_optional_positive(raw["direct_probability_scale"], f"{path}.direct_probability_scale")
    gradient_percentile = raw.get("direct_gradient_magnitude_percentile")
    scale_source_digest = raw.get(
        "direct_probability_scale_source_report_sha256"
    )
    if method == "direct_pulse":
        calibration_batches = _integer(
            calibration_batches,
            f"{path}.direct_probability_calibration_batches",
            minimum=1,
        )
        if calibration_batches != 64:
            raise config_error(
                f"{path}.direct_probability_calibration_batches",
                "to equal the frozen 64-minibatch calibration cohort",
                calibration_batches,
            )
        if gradient_percentile is not None:
            gradient_percentile = _number(
                gradient_percentile,
                f"{path}.direct_gradient_magnitude_percentile",
            )
            if gradient_percentile not in {90.0, 95.0, 99.0}:
                raise config_error(
                    f"{path}.direct_gradient_magnitude_percentile",
                    "to equal 90, 95, or 99 when provided",
                    gradient_percentile,
                )
        if probability_scale is None:
            if scale_source_digest is not None:
                raise config_error(
                    f"{path}.direct_probability_scale_source_report_sha256",
                    "to be omitted when direct_probability_scale is null",
                    scale_source_digest,
                )
        elif (
            not isinstance(scale_source_digest, str)
            or _SHA256.fullmatch(scale_source_digest) is None
        ):
            raise config_error(
                f"{path}.direct_probability_scale_source_report_sha256",
                "to be the source recovery-report SHA-256 when using a frozen scale",
                scale_source_digest,
            )
    elif any(
        value is not None
        for value in (
            calibration_batches,
            probability_scale,
            gradient_percentile,
            scale_source_digest,
        )
    ):
        raise config_error(
            path,
            "to set direct probability and gradient-gate fields to null outside direct_pulse",
            dict(raw),
        )

    fast_backend = raw["fast_backend"]
    fast_assignment_seed = _parse_optional_seed(raw["fast_assignment_seed"], f"{path}.fast_assignment_seed")
    fast_endpoint_seed = _parse_optional_seed(raw["fast_endpoint_seed"], f"{path}.fast_endpoint_seed")
    fast_read_samples = raw["fast_reset_read_samples"]
    fast_guard = raw["fast_reset_guard_standard_errors"]
    ttv2_methods = {
        "ttv2",
        IBM_OM_DEPLOYED_RECOVERY_TTV2_AIHWKIT_METHOD,
    }
    if method not in {"tiki_taka", *ttv2_methods}:
        if any(item is not None for item in (fast_backend, fast_assignment_seed, fast_endpoint_seed, fast_read_samples, fast_guard)):
            raise config_error(
                path,
                "to set every fast-array field to null outside tiki_taka/TTv2",
                dict(raw),
            )
    elif fast_backend == "ideal":
        if any(item is not None for item in (fast_assignment_seed, fast_endpoint_seed, fast_read_samples, fast_guard)):
            raise config_error(path, "to omit physical fast-array fields for the ideal backend", dict(raw))
    elif fast_backend == "physical_om" and method == "tiki_taka":
        if fast_assignment_seed is None or fast_endpoint_seed is None:
            raise config_error(path, "to provide physical fast-array assignment and endpoint seeds", dict(raw))
        fast_read_samples = _integer(fast_read_samples, f"{path}.fast_reset_read_samples", minimum=2)
        fast_guard = _number(fast_guard, f"{path}.fast_reset_guard_standard_errors")
        if fast_guard < 0.0:
            raise config_error(f"{path}.fast_reset_guard_standard_errors", "to be non-negative", fast_guard)
    else:
        raise config_error(
            f"{path}.fast_backend",
            "to be 'ideal' for TTv2 methods or 'ideal'/'physical_om' "
            "for legacy tiki_taka",
            fast_backend,
        )

    common_ttv2_names = (
        "ttv2_transfer_every",
        "ttv2_fast_update_model",
        "ttv2_fast_weight_limit",
        "ttv2_scan_mode",
        "ttv2_buffer_threshold",
    )
    legacy_ttv2_names = (
        "ttv2_gamma0",
        "ttv2_buffer_residual_mode",
    )
    aihwkit_ttv2_names = (
        "ttv2_fast_lr_by_parameter",
        "ttv2_transfer_lr",
        "ttv2_scale_transfer_lr",
        "ttv2_units_in_mbatch",
        "ttv2_auto_scale",
        "ttv2_fast_granularity_by_parameter",
        "ttv2_buffer_granularity",
        "ttv2_auto_granularity",
        "ttv2_correct_gradient_magnitudes",
        "ttv2_desired_bl",
        "ttv2_momentum",
        "ttv2_forget_buffer",
        "ttv2_cap_scope",
        "ttv2_cursor_policy",
        "ttv2_in_chop_probability",
    )
    normalized_ttv2: dict[str, Any] = {}
    if method in ttv2_methods:
        transfer_every = _integer(
            raw.get("ttv2_transfer_every"),
            f"{path}.ttv2_transfer_every",
            minimum=1,
        )
        if transfer_every != 1:
            raise config_error(
                f"{path}.ttv2_transfer_every",
                "to equal the predeclared one-minibatch transfer period",
                transfer_every,
            )
        fast_update_model = raw.get("ttv2_fast_update_model")
        if fast_update_model != "symmetric_soft_bounds":
            raise config_error(
                f"{path}.ttv2_fast_update_model",
                "to equal 'symmetric_soft_bounds'",
                fast_update_model,
            )
        fast_weight_limit = _number(
            raw.get("ttv2_fast_weight_limit"),
            f"{path}.ttv2_fast_weight_limit",
        )
        if fast_weight_limit != 1.0:
            raise config_error(
                f"{path}.ttv2_fast_weight_limit",
                "to equal 1",
                fast_weight_limit,
            )
        scan_mode = raw.get("ttv2_scan_mode")
        if scan_mode != "dual_rail_input_pair":
            raise config_error(
                f"{path}.ttv2_scan_mode",
                "to equal 'dual_rail_input_pair'",
                scan_mode,
            )
        buffer_threshold = _number(
            raw.get("ttv2_buffer_threshold"),
            f"{path}.ttv2_buffer_threshold",
        )
        if buffer_threshold != 1.0:
            raise config_error(
                f"{path}.ttv2_buffer_threshold",
                "to equal one slow-pulse unit",
                buffer_threshold,
            )

    if method == "ttv2":
        provided = {
            name: raw[name]
            for name in aihwkit_ttv2_names
            if name in raw
        }
        if provided:
            raise config_error(
                path,
                "to omit AIHWKit-1.1 TTv2 fields for legacy method='ttv2'",
                provided,
            )
        gamma0 = _number(raw.get("ttv2_gamma0"), f"{path}.ttv2_gamma0")
        if gamma0 != 10000.0:
            raise config_error(
                f"{path}.ttv2_gamma0",
                "to equal the predeclared 10000 transfer-scale constant",
                gamma0,
            )
        residual_mode = raw.get("ttv2_buffer_residual_mode")
        if residual_mode != "subtract_dispatched":
            raise config_error(
                f"{path}.ttv2_buffer_residual_mode",
                "to equal 'subtract_dispatched'",
                residual_mode,
            )
    elif method == IBM_OM_DEPLOYED_RECOVERY_TTV2_AIHWKIT_METHOD:
        forbidden = {
            name: raw[name]
            for name in legacy_ttv2_names
            if name in raw
        }
        if forbidden:
            raise config_error(
                path,
                "to omit ttv2_gamma0 and ttv2_buffer_residual_mode for "
                "method='ttv2_aihwkit_1p1_minibatch_equation'",
                forbidden,
            )
        missing = sorted(name for name in aihwkit_ttv2_names if name not in raw)
        if missing:
            raise config_error(
                path,
                "to contain every required AIHWKit-1.1 TTv2 field "
                f"{sorted(aihwkit_ttv2_names)!r}",
                dict(raw),
            )

        expected_parameter_keys = set(expected_source_layouts)

        def positive_parameter_mapping(name: str) -> dict[str, float]:
            mapping = raw[name]
            if not isinstance(mapping, Mapping) or set(mapping) != expected_parameter_keys:
                raise config_error(
                    f"{path}.{name}",
                    "to map exactly the source layout keys "
                    f"{sorted(expected_parameter_keys)!r} to positive finite numbers",
                    mapping,
                )
            parsed = {
                key: _number(mapping[key], f"{path}.{name}.{key}")
                for key in sorted(expected_parameter_keys)
            }
            if any(value <= 0.0 for value in parsed.values()):
                raise config_error(
                    f"{path}.{name}",
                    "to map exactly the source layout keys "
                    f"{sorted(expected_parameter_keys)!r} to positive finite numbers",
                    mapping,
                )
            return parsed

        normalized_ttv2["ttv2_fast_lr_by_parameter"] = (
            positive_parameter_mapping("ttv2_fast_lr_by_parameter")
        )
        normalized_ttv2["ttv2_fast_granularity_by_parameter"] = (
            positive_parameter_mapping("ttv2_fast_granularity_by_parameter")
        )
        for name in (
            "ttv2_transfer_lr",
            "ttv2_buffer_granularity",
            "ttv2_auto_granularity",
        ):
            parsed = _number(raw[name], f"{path}.{name}")
            if parsed <= 0.0:
                raise config_error(f"{path}.{name}", "to be positive", raw[name])
            normalized_ttv2[name] = parsed

        scale_transfer_lr = raw["ttv2_scale_transfer_lr"]
        if not isinstance(scale_transfer_lr, bool):
            raise config_error(
                f"{path}.ttv2_scale_transfer_lr",
                "to be a boolean",
                scale_transfer_lr,
            )
        normalized_ttv2["ttv2_scale_transfer_lr"] = scale_transfer_lr

        exact_aihwkit = {
            "ttv2_units_in_mbatch": True,
            "ttv2_auto_scale": False,
            "ttv2_correct_gradient_magnitudes": True,
            "ttv2_forget_buffer": True,
            "ttv2_cap_scope": "per_parameter_proportional",
            "ttv2_cursor_policy": "zero",
        }
        for name, expected in exact_aihwkit.items():
            if raw[name] != expected or (
                isinstance(expected, bool) and not isinstance(raw[name], bool)
            ):
                raise config_error(
                    f"{path}.{name}",
                    f"to equal {expected!r}",
                    raw[name],
                )
            normalized_ttv2[name] = expected

        desired_bl = _integer(
            raw["ttv2_desired_bl"],
            f"{path}.ttv2_desired_bl",
            minimum=1,
        )
        if desired_bl != 1:
            raise config_error(
                f"{path}.ttv2_desired_bl",
                "to equal 1",
                desired_bl,
            )
        normalized_ttv2["ttv2_desired_bl"] = desired_bl

        momentum = _number(raw["ttv2_momentum"], f"{path}.ttv2_momentum")
        if momentum != 0.0:
            raise config_error(
                f"{path}.ttv2_momentum",
                "to equal 0",
                momentum,
            )
        normalized_ttv2["ttv2_momentum"] = momentum

        in_chop_probability = _number(
            raw["ttv2_in_chop_probability"],
            f"{path}.ttv2_in_chop_probability",
        )
        if in_chop_probability != 0.0:
            raise config_error(
                f"{path}.ttv2_in_chop_probability",
                "to equal 0",
                in_chop_probability,
            )
        normalized_ttv2["ttv2_in_chop_probability"] = in_chop_probability
    else:
        legacy_provided = {
            name: raw.get(name)
            for name in (*common_ttv2_names, *legacy_ttv2_names)
            if raw.get(name) is not None
        }
        aihwkit_provided = {
            name: raw[name]
            for name in aihwkit_ttv2_names
            if name in raw
        }
        provided = {**legacy_provided, **aihwkit_provided}
        if provided:
            raise config_error(
                path,
                "to omit every TTv2 field outside a TTv2 method",
                provided,
            )

    normalized = {
            **dict(raw),
            "slow_pulse_budget_per_cell": budget,
            "pulse_step_ratio": pulse_step,
            "update_seed": update_seed,
            "terminal_epoch": terminal_epoch,
            "source_forward_logit_gain": source_gain,
            **digests,
            "expected_source_selection_epoch": selection_epoch,
            "expected_assignment_seed": assignment_seed,
            "expected_endpoint_seed": endpoint_seed,
            "source_dual_rail_layout_by_parameter": source_layouts,
            "direct_probability_calibration_batches": calibration_batches,
            "direct_probability_scale": probability_scale,
            "fast_assignment_seed": fast_assignment_seed,
            "fast_endpoint_seed": fast_endpoint_seed,
            "fast_reset_read_samples": fast_read_samples,
            "fast_reset_guard_standard_errors": fast_guard,
            **normalized_ttv2,
        }
    if "direct_gradient_magnitude_percentile" in raw:
        normalized["direct_gradient_magnitude_percentile"] = gradient_percentile
    if "direct_probability_scale_source_report_sha256" in raw:
        normalized[
            "direct_probability_scale_source_report_sha256"
        ] = scale_source_digest
    return freeze_json(
        normalized,
        path=path,
    )


def _parse_train(value: Any) -> StudentTrainSettings:
    path = "config.modes.train"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "num_epochs",
            "learning_rates",
            "temperature",
            "log_every",
            "max_batches",
            "max_validation_batches",
            "minimum_relative_kl_improvement",
            "update_backend",
        },
        {
            "weight_modifier",
            "selection_weight_modifier",
            "selection_evaluation",
            "selection_metric",
            "selection_noise_repeats",
        },
    )
    rates = raw["learning_rates"]
    if not isinstance(rates, (list, tuple)) or len(rates) != 2:
        raise config_error(f"{path}.learning_rates", "to contain exactly two non-negative rates", rates)
    parsed_rates = tuple(_number(item, f"{path}.learning_rates[{index}]") for index, item in enumerate(rates))
    if any(rate < 0.0 for rate in parsed_rates):
        raise config_error(
            f"{path}.learning_rates",
            "to contain exactly two non-negative rates",
            rates,
        )
    if _number(raw["temperature"], f"{path}.temperature") != 1.0:
        raise config_error(f"{path}.temperature", "to equal 1.0", raw["temperature"])
    relative = _number(raw["minimum_relative_kl_improvement"], f"{path}.minimum_relative_kl_improvement")
    if relative > 1.0:
        raise config_error(f"{path}.minimum_relative_kl_improvement", "to be in [0, 1]", relative)
    backend = _object(raw["update_backend"], f"{path}.update_backend")
    _keys(backend, f"{path}.update_backend", {"type", "parameters"})
    if backend["type"] not in {
        "ideal",
        "measured_cohort_a",
        "measured_cohort_a_sign_sgd",
        "measured_cohort_a_one_pulse_down",
        "measured_cohort_b",
        "program_verify",
        IBM_OM_DEPLOYED_RECOVERY_BACKEND,
    }:
        raise config_error(
            f"{path}.update_backend.type",
            "to be 'ideal', 'measured_cohort_a', "
            "'measured_cohort_a_sign_sgd', "
            "'measured_cohort_a_one_pulse_down', 'measured_cohort_b', "
            "'program_verify', or 'ibm_om_deployed_recovery'",
            backend["type"],
        )
    if backend["type"] == "ideal":
        parameters = _empty_object(backend["parameters"], f"{path}.update_backend.parameters")
    elif backend["type"] == "program_verify":
        parameters = _parse_program_verify(
            backend["parameters"],
            f"{path}.update_backend.parameters",
        )
    elif backend["type"] == IBM_OM_DEPLOYED_RECOVERY_BACKEND:
        parameters = _parse_ibm_om_deployed_recovery(
            backend["parameters"],
            f"{path}.update_backend.parameters",
        )
    else:
        parameters = _parse_measured(
            backend["parameters"],
            f"{path}.update_backend.parameters",
            backend_type=backend["type"],
        )
    if (
        backend["type"] == "measured_cohort_a_one_pulse_down"
        and parsed_rates != (0.0, 0.0)
    ):
        raise config_error(
            f"{path}.learning_rates",
            "to equal [0.0, 0.0] because "
            "measured_cohort_a_one_pulse_down ignores learning-rate magnitude",
            rates,
        )
    weight_modifier = _parse_weight_modifier(
        raw.get(
            "weight_modifier",
            {"type": "none", "parameters": {}},
        ),
        f"{path}.weight_modifier",
    )
    selection_weight_modifier = _parse_weight_modifier(
        raw.get(
            "selection_weight_modifier",
            {"type": "none", "parameters": {}},
        ),
        f"{path}.selection_weight_modifier",
    )
    selection_evaluation = raw.get("selection_evaluation", "clean")
    if selection_evaluation not in {"clean", "modifier"}:
        raise config_error(
            f"{path}.selection_evaluation",
            "to be 'clean' or 'modifier'",
            selection_evaluation,
        )
    selection_metric = raw.get("selection_metric", "kl_teacher_student")
    if selection_metric not in {"kl_teacher_student", "student_accuracy"}:
        raise config_error(
            f"{path}.selection_metric",
            "to be 'kl_teacher_student' or 'student_accuracy'",
            selection_metric,
        )
    selection_noise_repeats = _integer(
        raw.get("selection_noise_repeats", 1),
        f"{path}.selection_noise_repeats",
        minimum=1,
    )
    if backend["type"] == IBM_OM_DEPLOYED_RECOVERY_BACKEND:
        parsed_epochs = _integer(
            raw["num_epochs"], f"{path}.num_epochs", minimum=0
        )
        if parsed_rates != (
            0.004411914893617021,
            0.000011974808510638298,
        ):
            raise config_error(
                f"{path}.learning_rates",
                "to equal the frozen RESET-relative QAT learning rates",
                rates,
            )
        if (
            parsed_epochs != 10
            or parameters["terminal_epoch"] != parsed_epochs
            or raw["max_batches"] is not None
            or raw["max_validation_batches"] is not None
            or relative != 0.0
            or weight_modifier.type != "none"
            or selection_weight_modifier.type != "none"
            or selection_evaluation != "clean"
            or selection_metric != "student_accuracy"
            or selection_noise_repeats != 1
        ):
            raise config_error(
                path,
                "to use the full ten-epoch fixed-terminal deployed-recovery "
                "cohort with no weight modifier or validation selection",
                dict(raw),
            )
    if selection_evaluation == "modifier":
        selected_modifier = (
            selection_weight_modifier
            if selection_weight_modifier.type != "none"
            else weight_modifier
        )
        valid_modifier = bool(
            selected_modifier.parameters.get("noisy_evaluation", False)
        )
        if not valid_modifier:
            raise config_error(
                f"{path}.selection_evaluation",
                "to select a weight modifier with "
                "noisy_evaluation=true when set to 'modifier'",
                selection_evaluation,
            )
    elif selection_noise_repeats != 1:
        raise config_error(
            f"{path}.selection_noise_repeats",
            "to equal 1 when selection_evaluation is 'clean'",
            selection_noise_repeats,
        )
    if (
        selection_weight_modifier.type != "none"
        and selection_evaluation != "modifier"
    ):
        raise config_error(
            f"{path}.selection_weight_modifier",
            "to be 'none' unless selection_evaluation is 'modifier'",
            selection_weight_modifier.type,
        )
    if weight_modifier.type == "ibm_reram_om_program_verify":
        training_parameters = weight_modifier.parameters
        expected_training_execution = (
            "mapped_target"
            if training_parameters["target_mapping"]
            == "raw_active_p90_quad"
            else "compact_endpoint"
        )
        if (
            training_parameters["execution"]
            != expected_training_execution
            or training_parameters["noisy_evaluation"]
        ):
            raise config_error(
                f"{path}.weight_modifier",
                f"to use {expected_training_execution} with "
                "noisy_evaluation=false for off-chip minibatches",
                dict(training_parameters),
            )
    if selection_weight_modifier.type == "ibm_reram_om_program_verify":
        selection_parameters = selection_weight_modifier.parameters
        selection_is_array_specific = (
            selection_parameters["target_mapping"]
            in _IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS
        )
        expected_selection_corruption = (
            "counterfactual_repaired"
            if selection_is_array_specific
            else "published"
        )
        if (
            selection_parameters["execution"] != "pulse_resolved"
            or selection_parameters["corruption_policy"]
            != expected_selection_corruption
            or not selection_parameters["noisy_evaluation"]
        ):
            raise config_error(
                f"{path}.selection_weight_modifier",
                "to use pulse_resolved, noisy_evaluation=true, and "
                f"{expected_selection_corruption} corruption for its "
                "declared target mapping",
                dict(selection_parameters),
            )
    if (
        weight_modifier.type == "ibm_reram_om_program_verify"
        and weight_modifier.parameters["target_mapping"]
        in _IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS
    ):
        if selection_weight_modifier.type != "ibm_reram_om_program_verify":
            raise config_error(
                f"{path}.selection_weight_modifier",
                "to be an IBM OM modifier matched to the array-specific "
                "training modifier",
                selection_weight_modifier.type,
            )
        matched_fields = (
            "assignment_seed",
            "corruption_policy",
            "target_mapping",
            "dual_rail_layout_by_parameter",
            "common_window_margin_fraction",
        )
        if (
            weight_modifier.parameters["target_mapping"]
            == "shared_reset_relative_quad"
        ):
            matched_fields += (
                "reset_relative_mode",
                "reset_relative_contrast_step",
                "reset_read_samples",
                "reset_guard_standard_errors",
                "forward_logit_gain",
            )
        elif (
            weight_modifier.parameters["target_mapping"]
            == "raw_active_p90_quad"
        ):
            matched_fields += (
                "raw_active_mode",
                "raw_active_unsupported_quad_policy",
                "forward_logit_gain",
            )
        mismatches = {
            name: {
                "training": weight_modifier.parameters[name],
                "selection": selection_weight_modifier.parameters[name],
            }
            for name in matched_fields
            if weight_modifier.parameters[name]
            != selection_weight_modifier.parameters[name]
        }
        if mismatches:
            raise config_error(
                f"{path}.selection_weight_modifier",
                "to match the array-specific training modifier's fixed "
                "array, corruption policy, mapping, layout, and "
                "inner-window margin",
                mismatches,
            )
    return StudentTrainSettings(
        num_epochs=_integer(raw["num_epochs"], f"{path}.num_epochs", minimum=0),
        learning_rates=(parsed_rates[0], parsed_rates[1]),
        temperature=1.0,
        log_every=_integer(raw["log_every"], f"{path}.log_every", minimum=1),
        max_batches=_optional_batches(raw["max_batches"], f"{path}.max_batches"),
        max_validation_batches=_optional_batches(raw["max_validation_batches"], f"{path}.max_validation_batches"),
        minimum_relative_kl_improvement=relative,
        selection_evaluation=selection_evaluation,
        selection_metric=selection_metric,
        selection_noise_repeats=selection_noise_repeats,
        weight_modifier=weight_modifier,
        selection_weight_modifier=selection_weight_modifier,
        update_backend=UpdateBackendSettings(type=backend["type"], parameters=parameters),
    )


def _parse_validate(value: Any) -> StudentValidateSettings:
    path = "config.modes.validate"
    raw = _object(value, path)
    _keys(raw, path, {"split", "sample_limit"}, {"weight_modifier", "noise_repeats"})
    if raw["split"] not in {"validation", "test"}:
        raise config_error(f"{path}.split", "to be 'validation' or 'test'", raw["split"])
    modifier = _parse_weight_modifier(
        raw.get("weight_modifier", {"type": "none", "parameters": {}}),
        f"{path}.weight_modifier",
    )
    repeats = _integer(
        raw.get("noise_repeats", 1),
        f"{path}.noise_repeats",
        minimum=1,
    )
    if modifier.type == "none" and repeats != 1:
        raise config_error(
            f"{path}.noise_repeats",
            "to equal 1 when weight_modifier is 'none'",
            repeats,
        )
    if modifier.type != "none" and not bool(
        modifier.parameters.get("noisy_evaluation", False)
    ):
        raise config_error(
            f"{path}.weight_modifier",
            "to set noisy_evaluation=true for validation",
            dict(modifier.parameters),
        )
    if modifier.type == "ibm_reram_om_program_verify":
        array_specific = (
            modifier.parameters["target_mapping"]
            in _IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS
        )
        expected_corruption = (
            "counterfactual_repaired" if array_specific else "published"
        )
        if (
            modifier.parameters["execution"] != "pulse_resolved"
            or modifier.parameters["corruption_policy"]
            != expected_corruption
        ):
            raise config_error(
                f"{path}.weight_modifier",
                "to use pulse_resolved with "
                f"{expected_corruption} corruption for its declared target "
                "mapping during physical deployment validation",
                dict(modifier.parameters),
            )
    return StudentValidateSettings(
        split=raw["split"],
        sample_limit=_optional_batches(raw["sample_limit"], f"{path}.sample_limit"),
        weight_modifier=modifier,
        noise_repeats=repeats,
    )


def parse_student_config(payload: Mapping[str, Any]) -> StudentConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {
            "schema_version",
            "experiment_id",
            "runtime",
            "data",
            "model",
            "solver",
            "mapping",
            "modes",
        },
        {"teacher"},
    )
    if raw["schema_version"] != 1:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error("config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"])
    modes_raw = _object(raw["modes"], "config.modes")
    if not modes_raw or set(modes_raw) - {"train", "validate"}:
        raise config_error("config.modes", "to contain one or both of 'train' and 'validate'", dict(modes_raw))
    modes: dict[str, Any] = {}
    if "train" in modes_raw:
        modes["train"] = _parse_train(modes_raw["train"])
    if "validate" in modes_raw:
        modes["validate"] = _parse_validate(modes_raw["validate"])
    teacher = _parse_teacher(
        raw.get(
            "teacher",
            {
                "type": "bias_free_relu",
                "initialization": "signed_weight_mapping",
            },
        )
    )
    model = _parse_model(raw["model"])
    ibm_modifiers: list[tuple[str, UpdateBackendSettings]] = []
    if isinstance(modes.get("train"), StudentTrainSettings):
        train_settings = modes["train"]
        ibm_modifiers.extend(
            (
                (
                    "config.modes.train.weight_modifier",
                    train_settings.weight_modifier,
                ),
                (
                    "config.modes.train.selection_weight_modifier",
                    train_settings.selection_weight_modifier,
                ),
            )
        )
    if isinstance(modes.get("validate"), StudentValidateSettings):
        ibm_modifiers.append(
            (
                "config.modes.validate.weight_modifier",
                modes["validate"].weight_modifier,
            )
        )
    for modifier_path, modifier in ibm_modifiers:
        if modifier.type != "ibm_reram_om_program_verify":
            continue
        target_mapping = modifier.parameters["target_mapping"]
        if (
            target_mapping
            in {
                "dual_rail_quad_common_window",
                "shared_reset_relative_quad",
                "raw_active_p90_quad",
            }
            and model.encoding != "single"
        ):
            raise config_error(
                f"{modifier_path}.parameters.target_mapping",
                f"to select model.encoding='single' for {target_mapping}",
                target_mapping,
            )
        if target_mapping == "raw_active_p90_quad" and (
            model.conductance_min != 0.0 or model.conductance_max != 1.0
        ):
            raise config_error(
                f"{modifier_path}.parameters.target_mapping",
                "to use model conductance bounds [0, 1] so the frozen "
                "array-wide g coordinate is not rescaled a second time",
                {
                    "conductance_min": model.conductance_min,
                    "conductance_max": model.conductance_max,
                },
            )
        if (
            target_mapping == "differential_pair_common_window"
            and model.encoding != "differential"
        ):
            raise config_error(
                f"{modifier_path}.parameters.target_mapping",
                "to select model.encoding='differential' for "
                "differential_pair_common_window",
                target_mapping,
            )
    if teacher.type == "bounded_drn":
        teacher_bounds = teacher.conductance_bounds
        if teacher_bounds is None:  # pragma: no cover - parser guarantees it
            raise RuntimeError("Expected bounded_drn teacher bounds after parsing.")
        if (
            model.conductance_min > teacher_bounds[0]
            or model.conductance_max < teacher_bounds[1]
        ):
            raise config_error(
                "config.model",
                "to contain the bounded_drn teacher conductance interval "
                f"{list(teacher_bounds)!r}",
                {
                    "conductance_min": model.conductance_min,
                    "conductance_max": model.conductance_max,
                },
            )
    train = modes.get("train")
    if (
        teacher.type == "bounded_drn"
        and isinstance(train, StudentTrainSettings)
        and train.update_backend.type in MEASURED_BACKENDS
    ):
        raise config_error(
            "config.modes.train.update_backend.type",
            "to be 'ideal' or 'program_verify' for a bounded_drn teacher",
            train.update_backend.type,
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type
        in MEASURED_BACKENDS
        and train.update_backend.parameters["initial_target_mapping"]
        == "paired_affine_common_window"
        and model.encoding != "differential"
    ):
        raise config_error(
            "config.modes.train.update_backend.parameters.initial_target_mapping",
            "to select differential model encoding when using "
            "'paired_affine_common_window'",
            train.update_backend.parameters["initial_target_mapping"],
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type
        in MEASURED_BACKENDS
        and train.update_backend.parameters["initial_target_mapping"]
        in {
            "dual_rail_pairwise_common_window",
            "dual_rail_quad_common_window",
        }
        and model.encoding != "single"
    ):
        raise config_error(
            "config.modes.train.update_backend.parameters.initial_target_mapping",
            "to select single model encoding when using "
            "a dual-rail common-window mapping",
            train.update_backend.parameters["initial_target_mapping"],
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type
        in MEASURED_BACKENDS
        and train.update_backend.parameters["initial_target_mapping"]
        in {
            "dual_rail_pairwise_common_window",
            "dual_rail_quad_common_window",
        }
    ):
        layouts = train.update_backend.parameters[
            "dual_rail_layout_by_parameter"
        ]
        expected_layouts = {
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        }
        if dict(layouts) != expected_layouts:
            raise config_error(
                "config.modes.train.update_backend.parameters."
                "dual_rail_layout_by_parameter",
                "to equal the model-local dual-rail layouts "
                f"{expected_layouts!r}",
                dict(layouts),
            )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type == "measured_cohort_b"
        and model.encoding == "single"
        and train.update_backend.parameters["initial_target_mapping"]
        != "dual_rail_quad_common_window"
    ):
        raise config_error(
            "config.modes.train.update_backend.parameters."
            "initial_target_mapping",
            "to equal 'dual_rail_quad_common_window' for a four-device "
            "single-encoding cohort-B deployment",
            train.update_backend.parameters["initial_target_mapping"],
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type == "measured_cohort_a_sign_sgd"
        and (
            model.encoding != "single"
            or train.update_backend.parameters["initial_target_mapping"]
            != "dual_rail_quad_common_window"
        )
    ):
        raise config_error(
            "config.modes.train.update_backend.type",
            "to select the single-encoding four-device "
            "dual_rail_quad_common_window protocol when using "
            "'measured_cohort_a_sign_sgd'",
            train.update_backend.type,
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type
        == "measured_cohort_a_one_pulse_down"
    ):
        expected_mapping = (
            "dual_rail_quad_common_window"
            if model.encoding == "single"
            else "paired_affine_common_window"
        )
        provided_mapping = train.update_backend.parameters[
            "initial_target_mapping"
        ]
        if provided_mapping != expected_mapping:
            raise config_error(
                "config.modes.train.update_backend.parameters."
                "initial_target_mapping",
                f"to equal {expected_mapping!r} for {model.encoding!r} "
                "encoding with measured_cohort_a_one_pulse_down",
                provided_mapping,
            )
    mapping = _parse_mapping(raw["mapping"])
    if teacher.type == "bounded_drn":
        constraints = {
            "config.model.encoding": (model.encoding, "single"),
            "config.mapping.scale_fractions": (
                mapping.scale_fractions,
                (1.0,),
            ),
            "config.mapping.scale_fraction_pairs": (
                mapping.scale_fraction_pairs,
                None,
            ),
            "config.mapping.range_placement": (
                mapping.range_placement,
                "lower",
            ),
        }
        for constraint_path, (provided, expected) in constraints.items():
            if provided != expected:
                raise config_error(
                    constraint_path,
                    f"to equal {expected!r} for a bounded_drn teacher",
                    provided,
                )
    if mapping.range_placement == "centered" and model.encoding != "single":
        raise config_error(
            "config.mapping.range_placement",
            "to select single model encoding when using 'centered'",
            mapping.range_placement,
        )
    return StudentConfig(
        schema_version=1,
        experiment_id=EXPERIMENT_ID,
        runtime=_parse_runtime(raw["runtime"]),
        data=_parse_data(raw["data"]),
        teacher=teacher,
        model=model,
        solver=_parse_solver(raw["solver"]),
        mapping=mapping,
        modes=MappingProxyType(modes),
    )


def resolve_student_spec(document: StudentConfig, mode: RunMode) -> Any:
    key = mode.value
    if key not in document.modes:
        raise config_error("config.modes", f"to define requested mode {key!r}", tuple(document.modes))
    common = {
        "experiment_id": document.experiment_id,
        "runtime": document.runtime,
        "data": document.data,
        "teacher": document.teacher,
        "model": document.model,
        "solver": document.solver,
        "mapping": document.mapping,
        "settings": document.modes[key],
    }
    if mode is RunMode.TRAIN:
        return StudentTrainSpec(**common)
    if mode is RunMode.VALIDATE:
        return StudentValidateSpec(**common)
    raise config_error("the requested run mode", "to be 'train' or 'validate'", key)


__all__ = [
    "EXPERIMENT_ID",
    "MEASURED_BACKENDS",
    "MEASURED_COHORT_A_BACKENDS",
    "MEASURED_COHORT_B_BACKENDS",
    "IBM_OM_DEPLOYED_RECOVERY_BACKEND",
    "IBM_OM_DEPLOYED_RECOVERY_METHODS",
    "IBM_OM_DEPLOYED_RECOVERY_TTV2_AIHWKIT_METHOD",
    "SCHEMA_VERSION",
    "StudentConfig",
    "StudentTrainSpec",
    "StudentValidateSpec",
    "TeacherSettings",
    "parse_student_config",
    "resolve_student_spec",
]
