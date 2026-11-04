"""Pure, strict schemas for RESET-trained measured-device MNIST DRNs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from ebl.schema.mnist_relu import (
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
from ebl.schema.mnist_relu_drn import (
    SolverSettings,
    StudentModelSettings,
    UpdateBackendSettings,
    _parse_model as _parse_student_model,
    _parse_solver as _parse_student_solver,
)
from ebl.schema.base import RunMode, config_error, freeze_json
from ebl.schema.small_network import (
    ComponentSettings,
    _learning_rate_selection,
)


EXPERIMENT_ID = "mnist_relu_drn_reset.v1"
BIAS_EXPERIMENT_ID = "mnist_relu_drn_reset_bias.v1"
LEGACY_BIAS_EXPERIMENT_ID = "mnist_relu_drn_reset_bias_legacy.v1"
FACTORIAL_EXPERIMENT_ID = "mnist_relu_drn_reset_factorial.v1"
DIFFERENTIAL_EXPERIMENT_ID = "mnist_relu_drn_reset_differential.v1"
SCHEMA_VERSION = 1

_RESET_MEASURED_KEYS = {
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
}


@dataclass(frozen=True)
class InitializationSettings:
    type: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class LogitGainSettings:
    type: str
    value: float


@dataclass(frozen=True)
class ResetTrainSettings:
    objective: str
    num_epochs: int
    learning_rates: tuple[float, ...]
    learning_rate_selection: ComponentSettings
    initialization: InitializationSettings
    logit_gain: LogitGainSettings
    temperature: float
    reset_input_between_batches: bool
    log_every: int
    max_batches: int | None
    max_validation_batches: int | None
    minimum_validation_accuracy: float
    update_backend: UpdateBackendSettings


@dataclass(frozen=True)
class ResetValidateSettings:
    objective: str
    split: str
    sample_limit: int | None


@dataclass(frozen=True)
class ResetStudentConfig:
    schema_version: int
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: StudentModelSettings
    solver: SolverSettings
    modes: Mapping[str, Any]


@dataclass(frozen=True)
class ResetTrainSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: StudentModelSettings
    solver: SolverSettings
    settings: ResetTrainSettings


@dataclass(frozen=True)
class ResetValidateSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: StudentModelSettings
    solver: SolverSettings
    settings: ResetValidateSettings


def _empty_object(value: Any, path: str) -> Mapping[str, Any]:
    raw = _object(value, path)
    if raw:
        raise config_error(path, "to be an explicit empty JSON object", dict(raw))
    return MappingProxyType({})


def _parse_model(
    value: Any,
    *,
    include_biases: bool,
    amplification_indexing: str,
    encoding: str = "single",
    explicit_amplification_indexing: bool = False,
) -> StudentModelSettings:
    raw = _object(value, "config.model")
    if "include_biases" not in raw:
        # Delegate missing-key reporting to the complete parent schema.
        return _parse_student_model(value)
    if raw["include_biases"] is not include_biases:
        raise config_error(
            "config.model.include_biases",
            f"to be {str(include_biases).lower()}",
            raw["include_biases"],
        )
    parent_value = dict(raw)
    provided_indexing = parent_value.pop("amplification_indexing", None)
    if explicit_amplification_indexing:
        if provided_indexing != amplification_indexing:
            raise config_error(
                "config.model.amplification_indexing",
                f"to equal {amplification_indexing!r}",
                provided_indexing,
            )
    elif amplification_indexing == "legacy_process_global":
        if provided_indexing != amplification_indexing:
            raise config_error(
                "config.model.amplification_indexing",
                "to equal 'legacy_process_global'",
                provided_indexing,
            )
    elif provided_indexing is not None:
        raise config_error(
            "config.model.amplification_indexing",
            "to be absent for logical indexing",
            provided_indexing,
        )
    parent_value["include_biases"] = False
    model = replace(
        _parse_student_model(parent_value),
        include_biases=include_biases,
        amplification_indexing=amplification_indexing,
    )
    if model.encoding != encoding:
        encoding_description = (
            "one device per physical edge"
            if encoding == "single"
            else "one G+/G- pair per physical edge"
        )
        raise config_error(
            "config.model.encoding",
            f"to equal {encoding!r} for {encoding_description}",
            model.encoding,
        )
    return model


def _parse_initialization(value: Any, path: str) -> InitializationSettings:
    raw = _object(value, path)
    _keys(raw, path, {"type", "parameters"})
    if raw["type"] != "measured_reset":
        raise config_error(f"{path}.type", "to equal 'measured_reset'", raw["type"])
    return InitializationSettings(
        type="measured_reset",
        parameters=_empty_object(raw["parameters"], f"{path}.parameters"),
    )


def _parse_logit_gain(value: Any, path: str) -> LogitGainSettings:
    raw = _object(value, path)
    _keys(raw, path, {"type", "value"})
    if raw["type"] != "fixed":
        raise config_error(f"{path}.type", "to equal 'fixed'", raw["type"])
    gain = _number(raw["value"], f"{path}.value")
    if gain != 1.0:
        raise config_error(f"{path}.value", "to equal 1.0", raw["value"])
    return LogitGainSettings(type="fixed", value=1.0)


def _parse_reset_measured(value: Any, path: str) -> Mapping[str, Any]:
    raw = _object(value, path)
    _keys(raw, path, _RESET_MEASURED_KEYS)
    exact = {
        "curve_preprocessing": "raw",
        "cohort_fraction": 0.5,
        "cohort": "A",
        "source_traces_per_cell": 2,
        "initial_pulse_index": 0,
        "projection": "global_nearest",
    }
    for name, expected in exact.items():
        if raw[name] != expected:
            raise config_error(f"{path}.{name}", f"to equal {expected!r}", raw[name])
    for name in ("split_seed", "assignment_seed"):
        _integer(raw[name], f"{path}.{name}")
    _integer(raw["expected_trace_length"], f"{path}.expected_trace_length", minimum=2)
    if _number(
        raw["formed_resistance_max_ohm"],
        f"{path}.formed_resistance_max_ohm",
    ) <= 0.0:
        raise config_error(
            f"{path}.formed_resistance_max_ohm",
            "to be positive",
            raw["formed_resistance_max_ohm"],
        )
    return freeze_json(raw, path=path)


def _objective_description(allowed: frozenset[str]) -> str:
    order = ("teacher_kl", "cross_entropy", "paired_squared_error")
    return "to be " + ", ".join(
        repr(item) for item in order if item in allowed
    )


def _parse_train(
    value: Any,
    *,
    allowed_objectives: frozenset[str],
    include_biases: bool,
    required_reset_input: bool | None = None,
) -> ResetTrainSettings:
    path = "config.modes.train"
    raw = _object(value, path)
    allowed_keys = {
        "objective",
        "num_epochs",
        "learning_rate_selection",
        "initialization",
        "logit_gain",
        "temperature",
        "log_every",
        "max_batches",
        "max_validation_batches",
        "minimum_validation_accuracy",
        "update_backend",
    }
    if required_reset_input is not None:
        allowed_keys.add("reset_input_between_batches")
    _keys(
        raw,
        path,
        allowed_keys,
    )
    objective = raw["objective"]
    if objective not in allowed_objectives:
        raise config_error(
            f"{path}.objective",
            _objective_description(allowed_objectives),
            objective,
        )
    temperature = _number(raw["temperature"], f"{path}.temperature")
    if temperature != 1.0:
        raise config_error(f"{path}.temperature", "to equal 1.0", raw["temperature"])
    if required_reset_input is None:
        reset_input_between_batches = not include_biases
    else:
        provided_reset_input = raw["reset_input_between_batches"]
        if provided_reset_input is not required_reset_input:
            raise config_error(
                f"{path}.reset_input_between_batches",
                f"to be {str(required_reset_input).lower()}",
                provided_reset_input,
            )
        reset_input_between_batches = required_reset_input
    minimum_accuracy = _number(
        raw["minimum_validation_accuracy"],
        f"{path}.minimum_validation_accuracy",
    )
    if minimum_accuracy > 1.0:
        raise config_error(
            f"{path}.minimum_validation_accuracy",
            "to be in [0, 1]",
            raw["minimum_validation_accuracy"],
        )
    backend = _object(raw["update_backend"], f"{path}.update_backend")
    _keys(backend, f"{path}.update_backend", {"type", "parameters"})
    if backend["type"] != "measured_cohort_a":
        raise config_error(
            f"{path}.update_backend.type",
            "to equal 'measured_cohort_a'",
            backend["type"],
        )
    selection = _learning_rate_selection(
        raw["learning_rate_selection"],
        f"{path}.learning_rate_selection",
        weight_count=2,
    )
    if selection.type != "bounded_relative_update_grid":
        raise config_error(
            f"{path}.learning_rate_selection.type",
            "to equal 'bounded_relative_update_grid'",
            selection.type,
        )
    return ResetTrainSettings(
        objective=objective,
        num_epochs=_integer(raw["num_epochs"], f"{path}.num_epochs", minimum=1),
        learning_rates=(0.0,) * (3 if include_biases else 2),
        learning_rate_selection=selection,
        initialization=_parse_initialization(
            raw["initialization"], f"{path}.initialization"
        ),
        logit_gain=_parse_logit_gain(raw["logit_gain"], f"{path}.logit_gain"),
        temperature=1.0,
        reset_input_between_batches=reset_input_between_batches,
        log_every=_integer(raw["log_every"], f"{path}.log_every", minimum=1),
        max_batches=_optional_batches(raw["max_batches"], f"{path}.max_batches"),
        max_validation_batches=_optional_batches(
            raw["max_validation_batches"], f"{path}.max_validation_batches"
        ),
        minimum_validation_accuracy=minimum_accuracy,
        update_backend=UpdateBackendSettings(
            type="measured_cohort_a",
            parameters=_parse_reset_measured(
                backend["parameters"], f"{path}.update_backend.parameters"
            ),
        ),
    )


def _parse_validate(
    value: Any,
    *,
    allowed_objectives: frozenset[str],
) -> ResetValidateSettings:
    path = "config.modes.validate"
    raw = _object(value, path)
    _keys(raw, path, {"objective", "split", "sample_limit"})
    if raw["objective"] not in allowed_objectives:
        raise config_error(
            f"{path}.objective",
            _objective_description(allowed_objectives),
            raw["objective"],
        )
    if raw["split"] not in {"validation", "test"}:
        raise config_error(
            f"{path}.split", "to be 'validation' or 'test'", raw["split"]
        )
    return ResetValidateSettings(
        objective=raw["objective"],
        split=raw["split"],
        sample_limit=_optional_batches(raw["sample_limit"], f"{path}.sample_limit"),
    )


def _parse_reset_student_config(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    include_biases: bool,
    amplification_indexing: str,
    allowed_objectives: frozenset[str],
    encoding: str = "single",
    required_reset_input: bool | None = None,
    explicit_amplification_indexing: bool = False,
) -> ResetStudentConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {"schema_version", "experiment_id", "runtime", "data", "model", "solver", "modes"},
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise config_error(
            "config.schema_version", "to equal 1", raw["schema_version"]
        )
    if raw["experiment_id"] != experiment_id:
        raise config_error(
            "config.experiment_id",
            f"to equal {experiment_id!r}",
            raw["experiment_id"],
        )
    modes_raw = _object(raw["modes"], "config.modes")
    if not modes_raw or set(modes_raw) - {"train", "validate"}:
        raise config_error(
            "config.modes",
            "to contain one or both of 'train' and 'validate'",
            dict(modes_raw),
        )
    modes: dict[str, Any] = {}
    if "train" in modes_raw:
        modes["train"] = _parse_train(
            modes_raw["train"],
            allowed_objectives=allowed_objectives,
            include_biases=include_biases,
            required_reset_input=required_reset_input,
        )
    if "validate" in modes_raw:
        modes["validate"] = _parse_validate(
            modes_raw["validate"],
            allowed_objectives=allowed_objectives,
        )
    if (
        "train" in modes
        and "validate" in modes
        and modes["train"].objective != modes["validate"].objective
    ):
        raise config_error(
            "config.modes.train.objective and config.modes.validate.objective",
            "to match",
            {
                "train": modes["train"].objective,
                "validate": modes["validate"].objective,
            },
        )
    return ResetStudentConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        runtime=_parse_runtime(raw["runtime"]),
        data=_parse_data(raw["data"]),
        model=_parse_model(
            raw["model"],
            include_biases=include_biases,
            amplification_indexing=amplification_indexing,
            encoding=encoding,
            explicit_amplification_indexing=explicit_amplification_indexing,
        ),
        solver=_parse_student_solver(raw["solver"]),
        modes=MappingProxyType(modes),
    )


def parse_reset_student_config(payload: Mapping[str, Any]) -> ResetStudentConfig:
    return _parse_reset_student_config(
        payload,
        experiment_id=EXPERIMENT_ID,
        include_biases=False,
        amplification_indexing="logical",
        allowed_objectives=frozenset({"teacher_kl", "cross_entropy"}),
    )


def parse_reset_differential_student_config(
    payload: Mapping[str, Any],
) -> ResetStudentConfig:
    """Parse the model-local differential RESET training experiment."""

    document = _parse_reset_student_config(
        payload,
        experiment_id=DIFFERENTIAL_EXPERIMENT_ID,
        include_biases=False,
        amplification_indexing="logical",
        encoding="differential",
        allowed_objectives=frozenset({"paired_squared_error"}),
        required_reset_input=True,
        explicit_amplification_indexing=True,
    )
    if document.model.voltage_amp != 4.0:
        raise config_error(
            "config.model.voltage_amp",
            "to equal 4.0 for the differential RESET comparison",
            document.model.voltage_amp,
        )
    if document.model.current_amp != 0.25:
        raise config_error(
            "config.model.current_amp",
            "to equal 0.25 for the differential RESET comparison",
            document.model.current_amp,
        )
    return document


def parse_reset_bias_student_config(
    payload: Mapping[str, Any],
) -> ResetStudentConfig:
    return _parse_reset_student_config(
        payload,
        experiment_id=BIAS_EXPERIMENT_ID,
        include_biases=True,
        amplification_indexing="logical",
        allowed_objectives=frozenset(
            {"teacher_kl", "cross_entropy", "paired_squared_error"}
        ),
    )


def parse_reset_legacy_bias_student_config(
    payload: Mapping[str, Any],
) -> ResetStudentConfig:
    return _parse_reset_student_config(
        payload,
        experiment_id=LEGACY_BIAS_EXPERIMENT_ID,
        include_biases=True,
        amplification_indexing="legacy_process_global",
        allowed_objectives=frozenset(
            {"teacher_kl", "cross_entropy", "paired_squared_error"}
        ),
    )


def parse_reset_factorial_student_config(
    payload: Mapping[str, Any],
) -> ResetStudentConfig:
    """Parse one arm of the controlled bias/loss/indexing factorial."""

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
            "modes",
        },
    )
    model_raw = _object(raw["model"], "config.model")
    if "include_biases" not in model_raw:
        raise config_error(
            "config.model",
            "to contain the explicit boolean key 'include_biases'",
            dict(model_raw),
        )
    include_biases = model_raw["include_biases"]
    if not isinstance(include_biases, bool):
        raise config_error(
            "config.model.include_biases",
            "to be a boolean",
            include_biases,
        )
    if "amplification_indexing" not in model_raw:
        raise config_error(
            "config.model",
            "to contain the explicit key 'amplification_indexing'",
            dict(model_raw),
        )
    amplification_indexing = model_raw["amplification_indexing"]
    if amplification_indexing not in {"logical", "legacy_process_global"}:
        raise config_error(
            "config.model.amplification_indexing",
            "to be 'logical' or 'legacy_process_global'",
            amplification_indexing,
        )
    return _parse_reset_student_config(
        payload,
        experiment_id=FACTORIAL_EXPERIMENT_ID,
        include_biases=include_biases,
        amplification_indexing=amplification_indexing,
        allowed_objectives=frozenset({"teacher_kl", "paired_squared_error"}),
        required_reset_input=True,
        explicit_amplification_indexing=True,
    )


def resolve_reset_student_spec(document: ResetStudentConfig, mode: RunMode) -> Any:
    key = mode.value
    if key not in document.modes:
        raise config_error(
            "config.modes", f"to define requested mode {key!r}", tuple(document.modes)
        )
    common = {
        "experiment_id": document.experiment_id,
        "runtime": document.runtime,
        "data": document.data,
        "model": document.model,
        "solver": document.solver,
        "settings": document.modes[key],
    }
    if mode is RunMode.TRAIN:
        return ResetTrainSpec(**common)
    if mode is RunMode.VALIDATE:
        return ResetValidateSpec(**common)
    raise config_error("the requested run mode", "to be 'train' or 'validate'", key)


__all__ = [
    "BIAS_EXPERIMENT_ID",
    "DIFFERENTIAL_EXPERIMENT_ID",
    "FACTORIAL_EXPERIMENT_ID",
    "LEGACY_BIAS_EXPERIMENT_ID",
    "EXPERIMENT_ID",
    "SCHEMA_VERSION",
    "ResetStudentConfig",
    "ResetTrainSpec",
    "ResetValidateSpec",
    "parse_reset_student_config",
    "parse_reset_differential_student_config",
    "parse_reset_bias_student_config",
    "parse_reset_legacy_bias_student_config",
    "parse_reset_factorial_student_config",
    "resolve_reset_student_spec",
]
