"""Strict, immutable configuration for the ``small_drn.v1`` experiment.

This module describes scientific intent only.  Paths to input checkpoints,
output directories, resume snapshots, and other operational concerns belong
to the CLI request, not this document.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from ebl.schema.base import (
    ConfigError,
    ExtensionSelection,
    RunMode,
    config_error,
    freeze_json,
)
from networks.resistive.low_rank_config import (
    parse_passive_low_rank_adapter,
)
from networks.resistive.digital_low_rank_config import (
    parse_digital_low_rank_adapter,
)
from hardware.config import (
    AIHWKIT_RERAM_CMO,
    IBM_AFM2025_PCM,
    WAN2022_PHYSICAL,
    device_programming_to_mapping,
    parse_device_programming_config,
)
from networks.resistive.passive_layerwise_low_rank_config import (
    parse_passive_layerwise_low_rank_adapter,
    passive_layerwise_low_rank_to_mapping,
)


EXPERIMENT_ID = "small_drn.v1"
SCHEMA_VERSION = 1
_MAX_TORCH_SEED = 2**64 - 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RuntimeSettings:
    seed: Optional[int]
    data_seed: Optional[int]
    device: str
    dtype: str


from networks.config.small import DataSettings


from networks.config.small import NonLinearitySettings


from core.components import ComponentSettings


from networks.config.small import ModelSettings


from networks.config.small import UpdaterSettings


from networks.config.small import ToleranceSettings


from networks.config.small import PolishSettings


from networks.config.small import AndersonSettings


from networks.config.small import OverrelaxationSettings


from networks.config.small import ExperimentalExponentialSettings


from networks.config.small import SolverSettings


@dataclass(frozen=True)
class CommonSettings:
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    solver: SolverSettings


from methods.config.small import TrainSettings


@dataclass(frozen=True)
class LinspaceSettings:
    minimum: float
    maximum: float
    samples: int
    record_states: bool


@dataclass(frozen=True)
class ValidateSettings:
    split: str
    sample_limit: Optional[int]
    record_states: bool


@dataclass(frozen=True)
class SmallDrnConfig:
    schema_version: int
    experiment_id: str
    common: CommonSettings
    train: Optional[TrainSettings]
    linspace: Optional[LinspaceSettings]
    validate: Optional[ValidateSettings]


@dataclass(frozen=True)
class TrainSpec:
    schema_version: int
    experiment_id: str
    common: CommonSettings
    settings: TrainSettings

    @property
    def extensions(self) -> ExtensionSelection:
        return ExtensionSelection(
            model_adapter=self.common.model.adapter.type,
            weight_modifier=self.settings.weight_modifier.type,
            update_backend=self.settings.update_backend.type,
            algorithm=self.settings.algorithm,
        )


@dataclass(frozen=True)
class LinspaceSpec:
    schema_version: int
    experiment_id: str
    common: CommonSettings
    settings: LinspaceSettings


@dataclass(frozen=True)
class ValidateSpec:
    schema_version: int
    experiment_id: str
    common: CommonSettings
    settings: ValidateSettings


SmallDrnSpec = Union[TrainSpec, LinspaceSpec, ValidateSpec]


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise config_error(path, "to be a JSON object", value)
    for key in value:
        if not isinstance(key, str):
            raise config_error(
                path,
                "to be a JSON object with string keys",
                value,
            )
    return value


def _check_keys(
    value: Mapping[str, Any],
    path: str,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise config_error(
            path,
            "to define the required keys "
            + ", ".join(repr(item) for item in required),
            dict(value),
        )
    allowed = set(required) | set(optional)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise config_error(
            path,
            "to contain only the keys "
            + ", ".join(repr(item) for item in sorted(allowed)),
            unknown,
        )


def _string(
    value: Any,
    path: str,
    *,
    choices: Optional[Sequence[str]] = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise config_error(path, "to be a non-empty string", value)
    if choices is not None and value not in choices:
        raise config_error(
            path,
            "to be one of " + ", ".join(repr(item) for item in choices),
            value,
        )
    return value


def _optional_string(value: Any, path: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise config_error(path, "to be a boolean", value)
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise config_error(path, "to be an integer", value)
    if minimum is not None and value < minimum:
        raise config_error(path, f"to be an integer >= {minimum}", value)
    return value


def _optional_integer(
    value: Any,
    path: str,
    *,
    minimum: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    return _integer(value, path, minimum=minimum)


def _number(
    value: Any,
    path: str,
    *,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise config_error(path, "to be a finite number", value)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise config_error(path, "to be a finite number", value)
    if strictly_positive and parsed <= 0.0:
        raise config_error(path, "to be a finite number > 0", value)
    if minimum is not None and parsed < minimum:
        raise config_error(
            path,
            f"to be a finite number >= {minimum}",
            value,
        )
    return parsed


def _optional_number(
    value: Any,
    path: str,
    *,
    minimum: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return None
    return _number(value, path, minimum=minimum)


def _number_tuple(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise config_error(path, "to be a JSON array of finite numbers", value)
    if nonempty and not value:
        raise config_error(
            path,
            "to be a non-empty JSON array of finite numbers",
            value,
        )
    return tuple(
        _number(
            item,
            f"{path}[{index}]",
            minimum=minimum,
            strictly_positive=strictly_positive,
        )
        for index, item in enumerate(value)
    )


def _numeric_parameter_object(
    value: Any,
    path: str,
) -> Mapping[str, Union[int, float]]:
    """Require an explicit diode dictionary and freeze its numeric values."""

    parsed = _object(value, path)
    numeric = {}
    for key, item in parsed.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise config_error(
                f"{path}.{key}",
                "to be a finite number",
                item,
            )
        if not math.isfinite(float(item)):
            raise config_error(
                f"{path}.{key}",
                "to be a finite number",
                item,
            )
        numeric[key] = item
    return freeze_json(numeric, path=path)


def _component(
    value: Any,
    path: str,
    *,
    allowed_types: Sequence[str],
) -> ComponentSettings:
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("type", "parameters"),
    )
    component_type = _string(
        parsed["type"],
        f"{path}.type",
        choices=allowed_types,
    )
    parameters = _object(parsed["parameters"], f"{path}.parameters")
    if component_type == "none" and parameters:
        raise config_error(
            f"{path}.parameters",
            "to be an empty object when type is 'none'",
            dict(parameters),
        )
    return ComponentSettings(
        type=component_type,
        parameters=freeze_json(
            parameters,
            path=f"{path}.parameters",
        ),
    )


def _weight_modifier(value: Any, path: str) -> ComponentSettings:
    """Parse the focused hardware-aware weight modifier."""

    parsed = _object(value, path)
    _check_keys(parsed, path, required=("type", "parameters"))
    modifier_type = _string(
        parsed["type"],
        f"{path}.type",
        choices=("none", "add_normal"),
    )
    parameters_path = f"{path}.parameters"
    parameters = _object(parsed["parameters"], parameters_path)
    if modifier_type == "none":
        if parameters:
            raise config_error(
                parameters_path,
                "to be an empty object when type is 'none'",
                dict(parameters),
            )
        return ComponentSettings(
            type="none",
            parameters=freeze_json({}, path=parameters_path),
        )

    _check_keys(
        parameters,
        parameters_path,
        required=("std_dev",),
        optional=("seed", "noisy_evaluation", "scale_mode"),
    )
    seed = _optional_integer(
        parameters.get("seed"),
        f"{parameters_path}.seed",
        minimum=0,
    )
    if seed is not None and seed > _MAX_TORCH_SEED:
        raise config_error(
            f"{parameters_path}.seed",
            f"to be an integer in [0, {_MAX_TORCH_SEED}] or null",
            seed,
        )
    normalized = {
        "std_dev": _number(
            parameters["std_dev"],
            f"{parameters_path}.std_dev",
            minimum=0.0,
        ),
        "seed": seed,
        "noisy_evaluation": _boolean(
            parameters.get("noisy_evaluation", False),
            f"{parameters_path}.noisy_evaluation",
        ),
        "scale_mode": _string(
            parameters.get("scale_mode", "tensor_abs_max"),
            f"{parameters_path}.scale_mode",
            choices=("tensor_abs_max", "output_channel_abs_max"),
        ),
    }
    return ComponentSettings(
        type="add_normal",
        parameters=freeze_json(normalized, path=parameters_path),
    )


def _adam_parameters(value: Any, path: str) -> dict[str, Any]:
    parameters = _object(value, path)
    _check_keys(
        parameters,
        path,
        required=(
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "amsgrad",
        ),
    )
    beta1 = _number(parameters["beta1"], f"{path}.beta1", minimum=0.0)
    beta2 = _number(parameters["beta2"], f"{path}.beta2", minimum=0.0)
    if beta1 >= 1.0 or beta2 >= 1.0:
        raise config_error(
            f"{path}.beta1 and {path}.beta2",
            "to lie in [0, 1)",
            {"beta1": beta1, "beta2": beta2},
        )
    return {
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": _number(
            parameters["epsilon"],
            f"{path}.epsilon",
            strictly_positive=True,
        ),
        "weight_decay": _number(
            parameters["weight_decay"],
            f"{path}.weight_decay",
            minimum=0.0,
        ),
        "amsgrad": _boolean(parameters["amsgrad"], f"{path}.amsgrad"),
    }


def _update_backend(value: Any, path: str) -> ComponentSettings:
    """Parse the explicitly supported gradient-application backends."""

    parsed = _object(value, path)
    _check_keys(parsed, path, required=("type", "parameters"))
    backend_type = _string(
        parsed["type"],
        f"{path}.type",
        choices=(
            "direct",
            "direct_adam",
            "ibm_om_fp32_bounds",
            "tiki_taka",
            "program_verify",
            "measured_cohort_a",
            "measured_cohort_b",
            "measured_cohort_b_lora",
        ),
    )
    parameters_path = f"{path}.parameters"
    parameters = _object(parsed["parameters"], parameters_path)
    if backend_type == "direct_adam":
        normalized = _adam_parameters(parameters, parameters_path)
        return ComponentSettings(
            type=backend_type,
            parameters=freeze_json(normalized, path=parameters_path),
        )
    if backend_type == "ibm_om_fp32_bounds":
        _check_keys(
            parameters,
            parameters_path,
            required=(
                "preset",
                "assignment_seed",
                "corruption_policy",
                "bounds_coordinate",
                "initialization_distribution",
                "initialization_seed",
                "expected_population_sha256",
                "expected_population_fingerprint",
                "optimizer",
            ),
        )
        exact_strings = {
            "preset": "reram_array_om",
            "bounds_coordinate": "controller_clipped_0_1",
            "initialization_distribution": (
                "uniform_per_cell_effective_bounds"
            ),
        }
        normalized: dict[str, Any] = {}
        for name, expected in exact_strings.items():
            normalized[name] = _string(
                parameters[name],
                f"{parameters_path}.{name}",
                choices=(expected,),
            )
        normalized["assignment_seed"] = _integer(
            parameters["assignment_seed"],
            f"{parameters_path}.assignment_seed",
            minimum=0,
        )
        normalized["corruption_policy"] = _string(
            parameters["corruption_policy"],
            f"{parameters_path}.corruption_policy",
            choices=("published", "counterfactual_repaired"),
        )
        normalized["initialization_seed"] = _integer(
            parameters["initialization_seed"],
            f"{parameters_path}.initialization_seed",
            minimum=0,
        )
        for name in (
            "expected_population_sha256",
            "expected_population_fingerprint",
        ):
            digest = parameters[name]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise config_error(
                    f"{parameters_path}.{name}",
                    "to be a lowercase SHA-256 digest",
                    digest,
                )
            normalized[name] = digest
        optimizer_path = f"{parameters_path}.optimizer"
        optimizer = _object(parameters["optimizer"], optimizer_path)
        _check_keys(
            optimizer,
            optimizer_path,
            required=("type", "parameters"),
        )
        optimizer_type = _string(
            optimizer["type"],
            f"{optimizer_path}.type",
            choices=("sgd", "adam"),
        )
        optimizer_parameters_path = f"{optimizer_path}.parameters"
        if optimizer_type == "adam":
            optimizer_parameters = _adam_parameters(
                optimizer["parameters"],
                optimizer_parameters_path,
            )
        else:
            raw_sgd = _object(
                optimizer["parameters"],
                optimizer_parameters_path,
            )
            _check_keys(
                raw_sgd,
                optimizer_parameters_path,
                required=("momentum", "weight_decay"),
            )
            momentum = _number(
                raw_sgd["momentum"],
                f"{optimizer_parameters_path}.momentum",
                minimum=0.0,
            )
            if momentum >= 1.0:
                raise config_error(
                    f"{optimizer_parameters_path}.momentum",
                    "to lie in [0, 1)",
                    momentum,
                )
            optimizer_parameters = {
                "momentum": momentum,
                "weight_decay": _number(
                    raw_sgd["weight_decay"],
                    f"{optimizer_parameters_path}.weight_decay",
                    minimum=0.0,
                ),
            }
        normalized["optimizer"] = {
            "type": optimizer_type,
            "parameters": optimizer_parameters,
        }
        return ComponentSettings(
            type=backend_type,
            parameters=freeze_json(normalized, path=parameters_path),
        )
    if backend_type in {
        "measured_cohort_a",
        "measured_cohort_b",
        "measured_cohort_b_lora",
    }:
        _check_keys(
            parameters,
            parameters_path,
            required=(
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
            ),
            optional=(
                "programming_deadband_mode",
                "programming_deadband_relative",
                "probabilistic_write_mode",
                "probabilistic_write_probability",
                "probabilistic_write_scale_relative",
                "probabilistic_write_seed",
            ),
        )
        normalized = {
            "curve_preprocessing": _string(
                parameters["curve_preprocessing"],
                f"{parameters_path}.curve_preprocessing",
                choices=("raw", "isotonic_nonincreasing"),
            ),
            "split_seed": _integer(
                parameters["split_seed"],
                f"{parameters_path}.split_seed",
                minimum=0,
            ),
            "assignment_seed": _integer(
                parameters["assignment_seed"],
                f"{parameters_path}.assignment_seed",
                minimum=0,
            ),
            "formed_resistance_max_ohm": _number(
                parameters["formed_resistance_max_ohm"],
                f"{parameters_path}.formed_resistance_max_ohm",
                strictly_positive=True,
            ),
            "cohort_fraction": _number(
                parameters["cohort_fraction"],
                f"{parameters_path}.cohort_fraction",
                strictly_positive=True,
            ),
            "cohort": _string(
                parameters["cohort"],
                f"{parameters_path}.cohort",
                choices=("A", "B"),
            ),
            "source_traces_per_cell": _integer(
                parameters["source_traces_per_cell"],
                f"{parameters_path}.source_traces_per_cell",
                minimum=1,
            ),
            "initial_pulse_index": _integer(
                parameters["initial_pulse_index"],
                f"{parameters_path}.initial_pulse_index",
                minimum=0,
            ),
            "projection": _string(
                parameters["projection"],
                f"{parameters_path}.projection",
                choices=("global_nearest",),
            ),
            "expected_trace_length": _integer(
                parameters["expected_trace_length"],
                f"{parameters_path}.expected_trace_length",
                minimum=2,
            ),
            "programming_deadband_mode": _string(
                parameters.get("programming_deadband_mode", "none"),
                f"{parameters_path}.programming_deadband_mode",
                choices=(
                    "none",
                    "accumulated_shadow_relative_rms",
                ),
            ),
            "programming_deadband_relative": _number(
                parameters.get("programming_deadband_relative", 0.0),
                f"{parameters_path}.programming_deadband_relative",
                minimum=0.0,
            ),
            "probabilistic_write_mode": _string(
                parameters.get("probabilistic_write_mode", "none"),
                f"{parameters_path}.probabilistic_write_mode",
                choices=(
                    "none",
                    "uniform_bernoulli",
                    "displacement_proportional",
                ),
            ),
            "probabilistic_write_probability": _number(
                parameters.get("probabilistic_write_probability", 1.0),
                f"{parameters_path}.probabilistic_write_probability",
                minimum=0.0,
            ),
            "probabilistic_write_scale_relative": _number(
                parameters.get("probabilistic_write_scale_relative", 0.0),
                f"{parameters_path}.probabilistic_write_scale_relative",
                minimum=0.0,
            ),
            "probabilistic_write_seed": _integer(
                parameters.get("probabilistic_write_seed", 0),
                f"{parameters_path}.probabilistic_write_seed",
                minimum=0,
            ),
        }
        exact = {
            "cohort_fraction": 0.5,
            "source_traces_per_cell": 2,
            "initial_pulse_index": (
                normalized["expected_trace_length"] - 1
                if backend_type == "measured_cohort_b_lora"
                else 0
            ),
            "cohort": "A" if backend_type == "measured_cohort_a" else "B",
        }
        for name, expected in exact.items():
            if normalized[name] != expected:
                raise config_error(
                    f"{parameters_path}.{name}",
                    f"to equal {expected!r} for the {backend_type} protocol",
                    normalized[name],
                )
        deadband_mode = normalized["programming_deadband_mode"]
        deadband_relative = normalized["programming_deadband_relative"]
        if deadband_mode == "none" and deadband_relative != 0.0:
            raise config_error(
                f"{parameters_path}.programming_deadband_relative",
                "to equal 0.0 when programming_deadband_mode is 'none'",
                deadband_relative,
            )
        if deadband_mode == "accumulated_shadow_relative_rms" and (
            backend_type
            not in {"measured_cohort_b", "measured_cohort_b_lora"}
            or deadband_relative <= 0.0
        ):
            raise config_error(
                f"{parameters_path}.programming_deadband_mode and "
                f"{parameters_path}.programming_deadband_relative",
                "to select measured_cohort_b or measured_cohort_b_lora and "
                "a positive relative "
                "threshold for accumulated_shadow_relative_rms",
                {
                    "backend_type": backend_type,
                    "programming_deadband_mode": deadband_mode,
                    "programming_deadband_relative": deadband_relative,
                },
            )
        probability_mode = normalized["probabilistic_write_mode"]
        probability = normalized["probabilistic_write_probability"]
        probability_scale = normalized[
            "probabilistic_write_scale_relative"
        ]
        probability_seed = normalized["probabilistic_write_seed"]
        if probability > 1.0:
            raise config_error(
                f"{parameters_path}.probabilistic_write_probability",
                "to be no greater than 1.0",
                probability,
            )
        if probability_seed >= 2**63:
            raise config_error(
                f"{parameters_path}.probabilistic_write_seed",
                "to be an integer in [0, 2**63)",
                probability_seed,
            )
        if probability_mode == "none" and (
            probability != 1.0
            or probability_scale != 0.0
            or probability_seed != 0
        ):
            raise config_error(
                f"{parameters_path}.probabilistic_write_probability, "
                f"{parameters_path}.probabilistic_write_scale_relative, and "
                f"{parameters_path}.probabilistic_write_seed",
                "to equal 1.0, 0.0, and 0 when probabilistic_write_mode is "
                "'none'",
                {
                    "probability": probability,
                    "scale_relative": probability_scale,
                    "seed": probability_seed,
                },
            )
        if probability_mode != "none" and (
            backend_type
            not in {"measured_cohort_b", "measured_cohort_b_lora"}
            or deadband_mode != "none"
        ):
            raise config_error(
                f"{parameters_path}.probabilistic_write_mode",
                "to select measured_cohort_b or measured_cohort_b_lora "
                "without a deterministic "
                "programming deadband",
                {
                    "backend_type": backend_type,
                    "programming_deadband_mode": deadband_mode,
                    "probabilistic_write_mode": probability_mode,
                },
            )
        if probability_mode == "uniform_bernoulli" and (
            not 0.0 < probability < 1.0 or probability_scale != 0.0
        ):
            raise config_error(
                f"{parameters_path}.probabilistic_write_probability and "
                f"{parameters_path}.probabilistic_write_scale_relative",
                "to use a probability in (0, 1) and scale 0.0 for "
                "uniform_bernoulli",
                {
                    "probability": probability,
                    "scale_relative": probability_scale,
                },
            )
        if probability_mode == "displacement_proportional" and (
            probability != 1.0 or probability_scale <= 0.0
        ):
            raise config_error(
                f"{parameters_path}.probabilistic_write_probability and "
                f"{parameters_path}.probabilistic_write_scale_relative",
                "to use probability 1.0 and a positive scale for "
                "displacement_proportional",
                {
                    "probability": probability,
                    "scale_relative": probability_scale,
                },
            )
        return ComponentSettings(
            type=backend_type,
            parameters=freeze_json(normalized, path=parameters_path),
        )
    if backend_type != "program_verify":
        return ComponentSettings(
            type=backend_type,
            parameters=freeze_json(parameters, path=parameters_path),
        )
    _check_keys(
        parameters,
        parameters_path,
        required=("device",),
    )
    try:
        device = parse_device_programming_config(
            parameters["device"],
            path=f"{parameters_path}.device",
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if device.type not in (
        IBM_AFM2025_PCM,
        WAN2022_PHYSICAL,
        AIHWKIT_RERAM_CMO,
    ):
        raise config_error(
            f"{parameters_path}.device.type",
            "to select an endpoint model with a repeated-write "
            "implementation",
            device.type,
        )
    return ComponentSettings(
        type="program_verify",
        parameters=freeze_json(
            {"device": device_programming_to_mapping(device)},
            path=parameters_path,
        ),
    )


def _learning_rate_selection(
    value: Any,
    path: str,
    *,
    weight_count: int,
) -> ComponentSettings:
    parsed = _object(value, path)
    _check_keys(parsed, path, required=("type", "parameters"))
    selection_type = _string(
        parsed["type"],
        f"{path}.type",
        choices=("none", "bounded_relative_update_grid"),
    )
    parameters_path = f"{path}.parameters"
    parameters = _object(parsed["parameters"], parameters_path)
    if selection_type == "none":
        if parameters:
            raise config_error(
                parameters_path,
                "to be an empty object when type is 'none'",
                dict(parameters),
            )
        return ComponentSettings(
            type="none",
            parameters=freeze_json({}, path=parameters_path),
        )

    _check_keys(
        parameters,
        parameters_path,
        required=(
            "probe_batches",
            "stability_tolerance",
            "weight_relative_update_targets",
            "bias_quantile",
            "canary_batches",
            "candidate_epochs",
            "grid_multipliers",
            "max_center_reductions",
            "plateau_relative_tolerance",
            "safety",
        ),
    )
    raw_probe_batches = parameters["probe_batches"]
    if not isinstance(raw_probe_batches, (list, tuple)):
        raise config_error(
            f"{parameters_path}.probe_batches",
            "to be a strictly increasing array of positive even integers",
            raw_probe_batches,
        )
    probe_batches = tuple(
        _integer(
            item,
            f"{parameters_path}.probe_batches[{index}]",
            minimum=2,
        )
        for index, item in enumerate(raw_probe_batches)
    )
    if (
        not probe_batches
        or any(value % 2 for value in probe_batches)
        or tuple(sorted(set(probe_batches))) != probe_batches
    ):
        raise config_error(
            f"{parameters_path}.probe_batches",
            "to be a strictly increasing array of positive even integers",
            list(probe_batches),
        )
    targets = _number_tuple(
        parameters["weight_relative_update_targets"],
        f"{parameters_path}.weight_relative_update_targets",
        nonempty=True,
        strictly_positive=True,
    )
    if len(targets) != weight_count:
        raise config_error(
            f"{parameters_path}.weight_relative_update_targets",
            f"to contain exactly {weight_count} values",
            list(targets),
        )
    multipliers = _number_tuple(
        parameters["grid_multipliers"],
        f"{parameters_path}.grid_multipliers",
        nonempty=True,
        strictly_positive=True,
    )
    if tuple(sorted(set(multipliers))) != multipliers or 1.0 not in multipliers:
        raise config_error(
            f"{parameters_path}.grid_multipliers",
            "to be strictly increasing, unique, and contain 1.0",
            list(multipliers),
        )
    safety_path = f"{parameters_path}.safety"
    safety = _object(parameters["safety"], safety_path)
    _check_keys(
        safety,
        safety_path,
        required=(
            "warmup_batches",
            "loss_ema_decay",
            "loss_growth_factor",
            "gradient_growth_factor",
            "persistence_batches",
        ),
    )
    ema_decay = _number(
        safety["loss_ema_decay"],
        f"{safety_path}.loss_ema_decay",
        minimum=0.0,
    )
    if ema_decay >= 1.0:
        raise config_error(
            f"{safety_path}.loss_ema_decay",
            "to be smaller than 1.0",
            ema_decay,
        )
    bias_quantile = _number(
        parameters["bias_quantile"],
        f"{parameters_path}.bias_quantile",
        strictly_positive=True,
    )
    if bias_quantile > 1.0:
        raise config_error(
            f"{parameters_path}.bias_quantile",
            "to be in (0, 1]",
            bias_quantile,
        )
    normalized = {
        "probe_batches": probe_batches,
        "stability_tolerance": _number(
            parameters["stability_tolerance"],
            f"{parameters_path}.stability_tolerance",
            minimum=0.0,
        ),
        "weight_relative_update_targets": targets,
        "bias_quantile": bias_quantile,
        "canary_batches": _integer(
            parameters["canary_batches"],
            f"{parameters_path}.canary_batches",
            minimum=1,
        ),
        "candidate_epochs": _integer(
            parameters["candidate_epochs"],
            f"{parameters_path}.candidate_epochs",
            minimum=1,
        ),
        "grid_multipliers": multipliers,
        "max_center_reductions": _integer(
            parameters["max_center_reductions"],
            f"{parameters_path}.max_center_reductions",
            minimum=0,
        ),
        "plateau_relative_tolerance": _number(
            parameters["plateau_relative_tolerance"],
            f"{parameters_path}.plateau_relative_tolerance",
            minimum=0.0,
        ),
        "safety": {
            "warmup_batches": _integer(
                safety["warmup_batches"],
                f"{safety_path}.warmup_batches",
                minimum=1,
            ),
            "loss_ema_decay": ema_decay,
            "loss_growth_factor": _number(
                safety["loss_growth_factor"],
                f"{safety_path}.loss_growth_factor",
                strictly_positive=True,
            ),
            "gradient_growth_factor": _number(
                safety["gradient_growth_factor"],
                f"{safety_path}.gradient_growth_factor",
                strictly_positive=True,
            ),
            "persistence_batches": _integer(
                safety["persistence_batches"],
                f"{safety_path}.persistence_batches",
                minimum=1,
            ),
        },
    }
    return ComponentSettings(
        type=selection_type,
        parameters=freeze_json(normalized, path=parameters_path),
    )


def _parse_adapter(value: Any) -> ComponentSettings:
    path = "config.model.adapter"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("type", "parameters"),
    )
    adapter_type = _string(
        parsed["type"],
        f"{path}.type",
        choices=(
            "none",
            "passive_low_rank",
            "digital_low_rank",
            "passive_layerwise_low_rank",
        ),
    )
    parameters = _object(parsed["parameters"], f"{path}.parameters")
    if adapter_type == "none":
        if parameters:
            raise config_error(
                f"{path}.parameters",
                "to be an empty object when type is 'none'",
                dict(parameters),
            )
        normalized_parameters: Mapping[str, Any] = {}
    elif adapter_type == "passive_low_rank":
        try:
            normalized = parse_passive_low_rank_adapter(
                parameters,
                path=f"{path}.parameters",
            )
        except ValueError as error:
            raise ConfigError(str(error)) from error
        if normalized is None:  # pragma: no cover - non-null mapping above
            raise AssertionError("passive_low_rank normalization returned null")
        normalized_parameters = asdict(normalized)
    elif adapter_type == "digital_low_rank":
        try:
            normalized = parse_digital_low_rank_adapter(
                parameters,
                path=f"{path}.parameters",
            )
        except ValueError as error:
            raise ConfigError(str(error)) from error
        if normalized is None:  # pragma: no cover - non-null mapping above
            raise AssertionError("digital_low_rank normalization returned null")
        normalized_parameters = asdict(normalized)
    else:
        try:
            normalized_layerwise = (
                parse_passive_layerwise_low_rank_adapter(
                    parameters,
                    path=f"{path}.parameters",
                )
            )
        except ValueError as error:
            raise ConfigError(str(error)) from error
        if normalized_layerwise is None:  # pragma: no cover - non-null above
            raise AssertionError(
                "passive_layerwise_low_rank normalization returned null"
            )
        normalized_parameters = passive_layerwise_low_rank_to_mapping(
            normalized_layerwise
        )
    return ComponentSettings(
        type=adapter_type,
        parameters=freeze_json(
            normalized_parameters,
            path=f"{path}.parameters",
        ),
    )


def _parse_runtime(value: Any) -> RuntimeSettings:
    path = "config.runtime"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("seed", "data_seed", "device", "dtype"),
    )
    return RuntimeSettings(
        seed=_optional_integer(parsed["seed"], f"{path}.seed", minimum=0),
        data_seed=_optional_integer(
            parsed["data_seed"],
            f"{path}.data_seed",
            minimum=0,
        ),
        device=_string(parsed["device"], f"{path}.device"),
        dtype=_string(
            parsed["dtype"],
            f"{path}.dtype",
            choices=("float32", "float64"),
        ),
    )


def _parse_data(value: Any) -> DataSettings:
    path = "config.data"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("dataset", "batch_size", "num_points", "shuffle"),
        optional=("validation_points",),
    )
    return DataSettings(
        dataset=_string(
            parsed["dataset"],
            f"{path}.dataset",
            choices=("moons", "yinyang", "digits", "mnist"),
        ),
        batch_size=_integer(
            parsed["batch_size"],
            f"{path}.batch_size",
            minimum=1,
        ),
        num_points=_optional_integer(
            parsed["num_points"],
            f"{path}.num_points",
            minimum=1,
        ),
        validation_points=_optional_integer(
            parsed.get("validation_points"),
            f"{path}.validation_points",
            minimum=1,
        ),
        shuffle=_boolean(parsed["shuffle"], f"{path}.shuffle"),
    )


def _parse_non_linearity(value: Any) -> NonLinearitySettings:
    path = "config.model.non_linearity"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=(
            "type",
            "quadratic_diode_param",
            "exponential_diode_param",
            "hard_sigmoid_param",
            "iv_data_path",
        ),
    )
    iv_data_path = parsed["iv_data_path"]
    if iv_data_path is not None:
        iv_data_path = _string(iv_data_path, f"{path}.iv_data_path")
    return NonLinearitySettings(
        type=_string(
            parsed["type"],
            f"{path}.type",
            choices=(
                "perfect_diode",
                "lpw_diode",
                "double_diode",
                "double_diode_quadratic",
                "double_diode_exponential",
                "single_diode_exponential",
                "hard_sigmoid",
                "linear",
                "experimental",
            ),
        ),
        quadratic_diode_param=_numeric_parameter_object(
            parsed["quadratic_diode_param"],
            f"{path}.quadratic_diode_param",
        ),
        exponential_diode_param=_numeric_parameter_object(
            parsed["exponential_diode_param"],
            f"{path}.exponential_diode_param",
        ),
        hard_sigmoid_param=_numeric_parameter_object(
            parsed["hard_sigmoid_param"],
            f"{path}.hard_sigmoid_param",
        ),
        iv_data_path=iv_data_path,
    )


def _parse_model(value: Any) -> ModelSettings:
    path = "config.model"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=(
            "dims",
            "input_gain",
            "weight_gains",
            "weight_min",
            "weight_max",
            "voltage_amp",
            "current_amp",
            "non_linearity",
            "adapter",
        ),
        optional=("weight_init_mode", "include_biases"),
    )

    dims_value = parsed["dims"]
    if not isinstance(dims_value, (list, tuple)) or len(dims_value) < 2:
        raise config_error(
            f"{path}.dims",
            "to be an array of at least two positive integers",
            dims_value,
        )
    dims = tuple(
        _integer(item, f"{path}.dims[{index}]", minimum=1)
        for index, item in enumerate(dims_value)
    )
    weight_gains = _number_tuple(
        parsed["weight_gains"],
        f"{path}.weight_gains",
        nonempty=True,
        minimum=0.0,
    )
    expected_layers = len(dims) - 1
    if len(weight_gains) != expected_layers:
        raise config_error(
            f"{path}.weight_gains",
            f"to contain exactly {expected_layers} values "
            "(one per weight layer)",
            list(weight_gains),
        )

    weight_min = _number(parsed["weight_min"], f"{path}.weight_min")
    weight_max = _number(parsed["weight_max"], f"{path}.weight_max")
    if weight_min > weight_max:
        raise config_error(
            f"{path}.weight_min and {path}.weight_max",
            "to satisfy weight_min <= weight_max",
            {"weight_min": weight_min, "weight_max": weight_max},
        )

    voltage_amp = _number(
        parsed["voltage_amp"],
        f"{path}.voltage_amp",
        strictly_positive=True,
    )
    current_amp = _number(
        parsed["current_amp"],
        f"{path}.current_amp",
        strictly_positive=True,
    )
    adapter = _parse_adapter(parsed["adapter"])
    if adapter.type == "passive_low_rank":
        if len(dims) != 2:
            raise config_error(
                f"{path}.dims",
                "to contain exactly [input_width, output_width] when "
                "model.adapter.type is 'passive_low_rank'",
                list(dims),
            )
        if voltage_amp != 1.0 or current_amp != 1.0:
            raise config_error(
                f"{path}.voltage_amp and {path}.current_amp",
                "to both equal 1.0 when model.adapter.type is "
                "'passive_low_rank'",
                {
                    "voltage_amp": voltage_amp,
                    "current_amp": current_amp,
                },
            )
    elif adapter.type == "digital_low_rank":
        if len(dims) != 2:
            raise config_error(
                f"{path}.dims",
                "to contain exactly [input_width, output_width] when "
                "model.adapter.type is 'digital_low_rank'",
                list(dims),
            )
        device_noise = adapter.parameters["device_noise"]
        reference = device_noise.get("drn_conductance_at_g_max")
        if reference is not None and reference > weight_max:
            raise config_error(
                f"{path}.adapter.parameters.device_noise."
                "drn_conductance_at_g_max",
                "to be no greater than config.model.weight_max",
                reference,
            )
    elif adapter.type == "passive_layerwise_low_rank":
        if len(dims) != 3:
            raise config_error(
                f"{path}.dims",
                "to contain exactly [input_width, hidden_width, "
                "output_width] when model.adapter.type is "
                "'passive_layerwise_low_rank'",
                list(dims),
            )
        for parameter_key, layer in adapter.parameters["layers"].items():
            device_noise = layer["device_noise"]
            reference = (
                None
                if device_noise is None
                else device_noise.get("drn_conductance_at_g_max")
            )
            if reference is not None and reference > weight_max:
                raise config_error(
                    f"{path}.adapter.parameters.layers.{parameter_key}."
                    "device_noise.drn_conductance_at_g_max",
                    "to be no greater than config.model.weight_max",
                    reference,
                )

    return ModelSettings(
        dims=dims,
        input_gain=_number(
            parsed["input_gain"],
            f"{path}.input_gain",
            strictly_positive=True,
        ),
        weight_gains=weight_gains,
        weight_min=weight_min,
        weight_max=weight_max,
        weight_init_mode=_string(
            parsed.get("weight_init_mode", "kaiming_uniform"),
            f"{path}.weight_init_mode",
            choices=(
                "kaiming_uniform",
                "bounded_uniform",
                "bounded_range_uniform",
                "floor_shifted_kaiming_uniform",
                "om_cell_bounds_uniform",
            ),
        ),
        include_biases=_boolean(
            parsed.get("include_biases", True),
            f"{path}.include_biases",
        ),
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        non_linearity=_parse_non_linearity(parsed["non_linearity"]),
        adapter=adapter,
    )


def _parse_solver(value: Any) -> SolverSettings:
    path = "config.solver"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=(
            "inference_iterations",
            "training_iterations",
            "minimizer_impl",
            "minimizer_mode",
            "adaptive_equilibrium",
            "random_initialization",
            "updaters",
            "tolerances",
            "polish",
            "anderson",
            "overrelaxation",
            "experimental_exponential",
        ),
    )

    updaters_path = f"{path}.updaters"
    updaters = _object(parsed["updaters"], updaters_path)
    _check_keys(
        updaters,
        updaters_path,
        required=("double_diode", "single_diode"),
    )

    tolerances_path = f"{path}.tolerances"
    tolerances = _object(parsed["tolerances"], tolerances_path)
    _check_keys(
        tolerances,
        tolerances_path,
        required=("relative", "voltage", "residual_current"),
    )

    polish_path = f"{path}.polish"
    polish = _object(parsed["polish"], polish_path)
    _check_keys(
        polish,
        polish_path,
        required=(
            "enabled",
            "dynamic",
            "max_newton_iterations",
            "z_threshold",
            "exponential_clip",
        ),
    )

    anderson_path = f"{path}.anderson"
    anderson = _object(parsed["anderson"], anderson_path)
    _check_keys(
        anderson,
        anderson_path,
        required=(
            "memory",
            "omega",
            "tolerance_floor",
            "regularization",
        ),
    )

    overrelaxation_path = f"{path}.overrelaxation"
    overrelaxation = _object(
        parsed["overrelaxation"],
        overrelaxation_path,
    )
    _check_keys(
        overrelaxation,
        overrelaxation_path,
        required=(
            "factor",
            "reject_steps",
            "reject_max_tries",
            "reject_shrink",
            "reject_epsilon",
        ),
    )

    experimental_path = f"{path}.experimental_exponential"
    experimental = _object(
        parsed["experimental_exponential"],
        experimental_path,
    )
    _check_keys(
        experimental,
        experimental_path,
        required=(
            "damping",
            "newton_max_steps",
            "progressive_tolerance",
            "tolerance_start",
            "tolerance_end",
            "tolerance_switch_high",
            "tolerance_switch_low",
        ),
    )

    settings = SolverSettings(
        inference_iterations=_integer(
            parsed["inference_iterations"],
            f"{path}.inference_iterations",
            minimum=1,
        ),
        training_iterations=_integer(
            parsed["training_iterations"],
            f"{path}.training_iterations",
            minimum=1,
        ),
        minimizer_impl=_string(
            parsed["minimizer_impl"],
            f"{path}.minimizer_impl",
            choices=("custom", "anderson"),
        ),
        minimizer_mode=_string(
            parsed["minimizer_mode"],
            f"{path}.minimizer_mode",
            choices=(
                "forward",
                "backward",
                "synchronous",
                "asynchronous",
            ),
        ),
        adaptive_equilibrium=_boolean(
            parsed["adaptive_equilibrium"],
            f"{path}.adaptive_equilibrium",
        ),
        random_initialization=_boolean(
            parsed["random_initialization"],
            f"{path}.random_initialization",
        ),
        updaters=UpdaterSettings(
            double_diode=_optional_string(
                updaters["double_diode"],
                f"{updaters_path}.double_diode",
            ),
            single_diode=_optional_string(
                updaters["single_diode"],
                f"{updaters_path}.single_diode",
            ),
        ),
        tolerances=ToleranceSettings(
            relative=_number(
                tolerances["relative"],
                f"{tolerances_path}.relative",
                minimum=0.0,
            ),
            voltage=_number(
                tolerances["voltage"],
                f"{tolerances_path}.voltage",
                minimum=0.0,
            ),
            residual_current=_optional_number(
                tolerances["residual_current"],
                f"{tolerances_path}.residual_current",
                minimum=0.0,
            ),
        ),
        polish=PolishSettings(
            enabled=_boolean(
                polish["enabled"],
                f"{polish_path}.enabled",
            ),
            dynamic=_boolean(
                polish["dynamic"],
                f"{polish_path}.dynamic",
            ),
            max_newton_iterations=_integer(
                polish["max_newton_iterations"],
                f"{polish_path}.max_newton_iterations",
                minimum=1,
            ),
            z_threshold=_number(
                polish["z_threshold"],
                f"{polish_path}.z_threshold",
                strictly_positive=True,
            ),
            exponential_clip=_number(
                polish["exponential_clip"],
                f"{polish_path}.exponential_clip",
                strictly_positive=True,
            ),
        ),
        anderson=AndersonSettings(
            memory=_integer(
                anderson["memory"],
                f"{anderson_path}.memory",
                minimum=1,
            ),
            omega=_number(
                anderson["omega"],
                f"{anderson_path}.omega",
                strictly_positive=True,
            ),
            tolerance_floor=_number(
                anderson["tolerance_floor"],
                f"{anderson_path}.tolerance_floor",
                minimum=0.0,
            ),
            regularization=_number(
                anderson["regularization"],
                f"{anderson_path}.regularization",
                minimum=0.0,
            ),
        ),
        overrelaxation=OverrelaxationSettings(
            factor=_number(
                overrelaxation["factor"],
                f"{overrelaxation_path}.factor",
                strictly_positive=True,
            ),
            reject_steps=_boolean(
                overrelaxation["reject_steps"],
                f"{overrelaxation_path}.reject_steps",
            ),
            reject_max_tries=_integer(
                overrelaxation["reject_max_tries"],
                f"{overrelaxation_path}.reject_max_tries",
                minimum=1,
            ),
            reject_shrink=_number(
                overrelaxation["reject_shrink"],
                f"{overrelaxation_path}.reject_shrink",
                strictly_positive=True,
            ),
            reject_epsilon=_number(
                overrelaxation["reject_epsilon"],
                f"{overrelaxation_path}.reject_epsilon",
                minimum=0.0,
            ),
        ),
        experimental_exponential=ExperimentalExponentialSettings(
            damping=_number(
                experimental["damping"],
                f"{experimental_path}.damping",
                strictly_positive=True,
            ),
            newton_max_steps=_integer(
                experimental["newton_max_steps"],
                f"{experimental_path}.newton_max_steps",
                minimum=1,
            ),
            progressive_tolerance=_boolean(
                experimental["progressive_tolerance"],
                f"{experimental_path}.progressive_tolerance",
            ),
            tolerance_start=_number(
                experimental["tolerance_start"],
                f"{experimental_path}.tolerance_start",
                strictly_positive=True,
            ),
            tolerance_end=_number(
                experimental["tolerance_end"],
                f"{experimental_path}.tolerance_end",
                strictly_positive=True,
            ),
            tolerance_switch_high=_number(
                experimental["tolerance_switch_high"],
                f"{experimental_path}.tolerance_switch_high",
                strictly_positive=True,
            ),
            tolerance_switch_low=_number(
                experimental["tolerance_switch_low"],
                f"{experimental_path}.tolerance_switch_low",
                strictly_positive=True,
            ),
        ),
    )
    if settings.overrelaxation.reject_shrink >= 1.0:
        raise config_error(
            f"{overrelaxation_path}.reject_shrink",
            "to be smaller than 1.0",
            settings.overrelaxation.reject_shrink,
        )
    if (
        settings.experimental_exponential.tolerance_switch_high
        <= settings.experimental_exponential.tolerance_switch_low
    ):
        raise config_error(
            f"{experimental_path}.tolerance_switch_high and "
            f"{experimental_path}.tolerance_switch_low",
            "to satisfy tolerance_switch_high > tolerance_switch_low > 0",
            {
                "tolerance_switch_high": (
                    settings.experimental_exponential.tolerance_switch_high
                ),
                "tolerance_switch_low": (
                    settings.experimental_exponential.tolerance_switch_low
                ),
            },
        )
    return settings


def _parse_train(
    value: Any,
    *,
    weight_count: int,
    bias_count: int,
) -> TrainSettings:
    path = "config.modes.train"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=(
            "num_epochs",
            "algorithm",
            "learning_rates",
            "bias_learning_rates",
            "nudging",
            "log_every",
            "max_batches",
            "max_validation_batches",
            "weight_modifier",
            "update_backend",
        ),
        optional=("learning_rate_selection",),
    )
    learning_rates = _number_tuple(
        parsed["learning_rates"],
        f"{path}.learning_rates",
        nonempty=True,
        minimum=0.0,
    )
    if len(learning_rates) != weight_count:
        raise config_error(
            f"{path}.learning_rates",
            f"to contain exactly {weight_count} values "
            "(one per weight layer or trainable adapter factor)",
            list(learning_rates),
        )
    bias_learning_rates = _number_tuple(
        parsed["bias_learning_rates"],
        f"{path}.bias_learning_rates",
        minimum=0.0,
    )
    if len(bias_learning_rates) != bias_count:
        raise config_error(
            f"{path}.bias_learning_rates",
            f"to contain exactly {bias_count} values "
            "(one per hidden-layer bias; use 0.0 to freeze one)",
            list(bias_learning_rates),
        )
    algorithm = _string(
        parsed["algorithm"],
        f"{path}.algorithm",
        choices=("ep", "backprop", "digital"),
    )
    nudging = _number(parsed["nudging"], f"{path}.nudging")
    if algorithm == "ep" and nudging == 0.0:
        raise config_error(
            f"{path}.nudging",
            "to be non-zero for equilibrium propagation",
            nudging,
        )
    if algorithm == "digital" and nudging != 0.0:
        raise config_error(
            f"{path}.nudging",
            "to equal 0.0 for direct digital-readout training",
            nudging,
        )
    update_backend = _update_backend(
        parsed["update_backend"],
        f"{path}.update_backend",
    )
    learning_rate_selection = _learning_rate_selection(
        parsed.get(
            "learning_rate_selection",
            {"type": "none", "parameters": {}},
        ),
        f"{path}.learning_rate_selection",
        weight_count=weight_count,
    )
    if learning_rate_selection.type != "none" and any(
        value != 0.0 for value in learning_rates + bias_learning_rates
    ):
        raise config_error(
            f"{path}.learning_rates and {path}.bias_learning_rates",
            "to contain only 0.0 placeholders when automatic selection is enabled",
            {
                "learning_rates": list(learning_rates),
                "bias_learning_rates": list(bias_learning_rates),
            },
        )
    if (
        update_backend.type == "measured_cohort_a"
        and learning_rate_selection.type != "bounded_relative_update_grid"
    ):
        raise config_error(
            f"{path}.learning_rate_selection.type",
            "to equal 'bounded_relative_update_grid' for measured_cohort_a",
            learning_rate_selection.type,
        )
    if (
        update_backend.type == "measured_cohort_b_lora"
        and learning_rate_selection.type != "none"
    ):
        raise config_error(
            f"{path}.learning_rate_selection.type",
            "to equal 'none' for the exploratory four-factor measured LoRA "
            "protocol",
            learning_rate_selection.type,
        )
    if (
        learning_rate_selection.type == "bounded_relative_update_grid"
        and update_backend.type
        not in {"measured_cohort_a", "measured_cohort_b"}
    ):
        raise config_error(
            f"{path}.update_backend.type",
            "to select a measured-cohort backend when bounded relative-update selection is enabled",
            update_backend.type,
        )
    return TrainSettings(
        num_epochs=_integer(
            parsed["num_epochs"],
            f"{path}.num_epochs",
            minimum=1,
        ),
        algorithm=algorithm,
        learning_rates=learning_rates,
        bias_learning_rates=bias_learning_rates,
        nudging=nudging,
        log_every=_integer(
            parsed["log_every"],
            f"{path}.log_every",
            minimum=1,
        ),
        max_batches=_optional_integer(
            parsed["max_batches"],
            f"{path}.max_batches",
            minimum=1,
        ),
        max_validation_batches=_optional_integer(
            parsed["max_validation_batches"],
            f"{path}.max_validation_batches",
            minimum=1,
        ),
        weight_modifier=_weight_modifier(
            parsed["weight_modifier"],
            f"{path}.weight_modifier",
        ),
        update_backend=update_backend,
        learning_rate_selection=learning_rate_selection,
    )


def _parse_linspace(value: Any) -> LinspaceSettings:
    path = "config.modes.linspace"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("minimum", "maximum", "samples", "record_states"),
    )
    minimum = _number(parsed["minimum"], f"{path}.minimum")
    maximum = _number(parsed["maximum"], f"{path}.maximum")
    if minimum >= maximum:
        raise config_error(
            f"{path}.minimum and {path}.maximum",
            "to satisfy minimum < maximum",
            {"minimum": minimum, "maximum": maximum},
        )
    return LinspaceSettings(
        minimum=minimum,
        maximum=maximum,
        samples=_integer(
            parsed["samples"],
            f"{path}.samples",
            minimum=2,
        ),
        record_states=_boolean(
            parsed["record_states"],
            f"{path}.record_states",
        ),
    )


def _parse_validate(value: Any) -> ValidateSettings:
    path = "config.modes.validate"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("split", "sample_limit", "record_states"),
    )
    return ValidateSettings(
        split=_string(
            parsed["split"],
            f"{path}.split",
            choices=("train", "validation", "test"),
        ),
        sample_limit=_optional_integer(
            parsed["sample_limit"],
            f"{path}.sample_limit",
            minimum=1,
        ),
        record_states=_boolean(
            parsed["record_states"],
            f"{path}.record_states",
        ),
    )


def parse_small_drn_config(payload: Mapping[str, Any]) -> SmallDrnConfig:
    """Parse one complete ``small_drn.v1`` document.

    Every supplied mode is validated, while a document may intentionally
    contain only the modes used by a particular campaign.
    """

    root = _object(payload, "config")
    _check_keys(
        root,
        "config",
        required=(
            "schema_version",
            "experiment_id",
            "runtime",
            "data",
            "model",
            "solver",
            "modes",
        ),
    )
    schema_version = _integer(
        root["schema_version"],
        "config.schema_version",
        minimum=1,
    )
    if schema_version != SCHEMA_VERSION:
        raise config_error(
            "config.schema_version",
            f"to equal {SCHEMA_VERSION} for {EXPERIMENT_ID!r}",
            schema_version,
        )
    experiment_id = _string(
        root["experiment_id"],
        "config.experiment_id",
    )
    if experiment_id != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id",
            f"to equal {EXPERIMENT_ID!r}",
            experiment_id,
        )

    model = _parse_model(root["model"])
    runtime = _parse_runtime(root["runtime"])
    data = _parse_data(root["data"])
    if model.dims[0] % 2:
        raise config_error(
            "config.model.dims[0]",
            "to be even because inputs use positive/negative paired nodes",
            model.dims[0],
        )
    logical_input = model.dims[0] // 2
    expected_inputs = {"yinyang": 2, "digits": 64, "mnist": 784}
    if data.dataset in expected_inputs and logical_input != expected_inputs[
        data.dataset
    ]:
        raise config_error(
            "config.model.dims[0]",
            f"to equal {2 * expected_inputs[data.dataset]} for "
            f"dataset {data.dataset!r}",
            model.dims[0],
        )
    class_count = {
        "moons": 2,
        "yinyang": 3,
        "digits": 10,
        "mnist": 10,
    }[data.dataset]
    if model.dims[-1] not in (class_count, 2 * class_count):
        raise config_error(
            "config.model.dims[-1]",
            f"to equal {class_count} (standard output) or "
            f"{2 * class_count} (differential output) for "
            f"dataset {data.dataset!r}",
            model.dims[-1],
        )
    if (
        model.adapter.type == "digital_low_rank"
        and model.dims[-1] != 2 * class_count
    ):
        raise config_error(
            "config.model.dims[-1]",
            f"to equal {2 * class_count} for differential outputs when "
            "model.adapter.type is 'digital_low_rank'",
            model.dims[-1],
        )
    if data.dataset in {"moons", "yinyang"} and data.num_points is None:
        raise config_error(
            "config.data.num_points",
            f"to be an integer for dataset {data.dataset!r}",
            data.num_points,
        )
    if data.validation_points is not None and data.dataset != "mnist":
        raise config_error(
            "config.data.validation_points",
            "to be null unless config.data.dataset is 'mnist'",
            data.validation_points,
        )
    if data.dataset == "mnist" and (
        data.validation_points is not None
        and data.validation_points >= 60000
    ):
        raise config_error(
            "config.data.validation_points",
            "to be smaller than the 60000-sample MNIST training set",
            data.validation_points,
        )
    common = CommonSettings(
        runtime=runtime,
        data=data,
        model=model,
        solver=_parse_solver(root["solver"]),
    )

    modes = _object(root["modes"], "config.modes")
    _check_keys(
        modes,
        "config.modes",
        required=(),
        optional=("train", "linspace", "validate"),
    )
    if not modes:
        raise config_error(
            "config.modes",
            "to define at least one of 'train', 'linspace', or 'validate'",
            dict(modes),
        )
    if data.dataset == "mnist" and "linspace" in modes:
        raise config_error(
            "config.modes",
            "to contain only 'train' and/or 'validate' for dataset 'mnist'",
            sorted(modes),
        )
    if model.adapter.type in {"passive_low_rank", "digital_low_rank"}:
        weight_count = 2
        bias_count = 0
    elif model.adapter.type == "passive_layerwise_low_rank":
        weight_count = 4
        bias_count = 0
    else:
        weight_count = len(model.dims) - 1
        bias_count = (
            max(0, weight_count - 1) if model.include_biases else 0
        )
    train_settings = (
        _parse_train(
            modes["train"],
            weight_count=weight_count,
            bias_count=bias_count,
        )
        if "train" in modes
        else None
    )
    if train_settings is not None:
        om_bounds_backend = (
            train_settings.update_backend.type == "ibm_om_fp32_bounds"
        )
        om_bounds_initialization = (
            model.weight_init_mode == "om_cell_bounds_uniform"
        )
        if om_bounds_backend != om_bounds_initialization:
            raise config_error(
                "config.model.weight_init_mode and "
                "config.modes.train.update_backend.type",
                "to pair 'om_cell_bounds_uniform' exactly with "
                "'ibm_om_fp32_bounds'",
                {
                    "weight_init_mode": model.weight_init_mode,
                    "update_backend": train_settings.update_backend.type,
                },
            )
        if om_bounds_backend:
            constraints = {
                "config.data.dataset": (data.dataset, "mnist"),
                "config.model.dims": (model.dims, (1568, 100, 20)),
                "config.model.weight_min": (model.weight_min, 0.0),
                "config.model.weight_max": (model.weight_max, 1.0),
                "config.model.include_biases": (model.include_biases, False),
                "config.model.adapter.type": (model.adapter.type, "none"),
                "config.modes.train.algorithm": (
                    train_settings.algorithm,
                    "backprop",
                ),
                "config.modes.train.weight_modifier.type": (
                    train_settings.weight_modifier.type,
                    "none",
                ),
            }
            for constraint_path, (provided, expected) in constraints.items():
                if provided != expected:
                    raise config_error(
                        constraint_path,
                        "to equal "
                        f"{expected!r} for ibm_om_fp32_bounds",
                        provided,
                    )
            if data.validation_points is None:
                raise config_error(
                    "config.data.validation_points",
                    "to define a held-out MNIST validation split for "
                    "ibm_om_fp32_bounds",
                    data.validation_points,
                )
    if train_settings is not None and (
        train_settings.update_backend.type
        in {
            "measured_cohort_a",
            "measured_cohort_b",
            "measured_cohort_b_lora",
        }
    ):
        expected_adapter = (
            "passive_layerwise_low_rank"
            if train_settings.update_backend.type
            == "measured_cohort_b_lora"
            else "none"
        )
        constraints = {
            "config.data.dataset": (data.dataset, "mnist"),
            "config.model.adapter.type": (
                model.adapter.type,
                expected_adapter,
            ),
            "config.modes.train.algorithm": (
                train_settings.algorithm,
                "backprop",
            ),
            "config.modes.train.weight_modifier.type": (
                train_settings.weight_modifier.type,
                "none",
            ),
        }
        for constraint_path, (provided, expected) in constraints.items():
            if provided != expected:
                raise config_error(
                    constraint_path,
                    "to equal "
                    f"{expected!r} for {train_settings.update_backend.type}",
                    provided,
                )
        if data.validation_points is None:
            raise config_error(
                "config.data.validation_points",
                "to define a held-out MNIST validation split for measured-cohort training",
                data.validation_points,
            )
        if model.weight_min != 0.0:
            raise config_error(
                "config.model.weight_min",
                "to equal 0.0 for measured non-negative conductances",
                model.weight_min,
            )
    if model.adapter.type == "passive_layerwise_low_rank":
        layers = model.adapter.parameters["layers"]
        noise_by_parameter = {
            key: layer["device_noise"] for key, layer in layers.items()
        }
        measured_lora = (
            train_settings is not None
            and train_settings.update_backend.type
            == "measured_cohort_b_lora"
        )
        invalid = {
            key: value
            for key, value in noise_by_parameter.items()
            if (value is not None) == measured_lora
        }
        if invalid:
            expected = (
                "to be null because --device-data supplies the measured "
                "cohort-B base and LoRA curves"
                if measured_lora
                else "to define a device-programming object outside the "
                "measured_cohort_b_lora protocol"
            )
            raise config_error(
                "config.model.adapter.parameters.layers.*.device_noise",
                expected,
                invalid,
            )
    return SmallDrnConfig(
        schema_version=schema_version,
        experiment_id=experiment_id,
        common=common,
        train=train_settings,
        linspace=(
            _parse_linspace(modes["linspace"])
            if "linspace" in modes
            else None
        ),
        validate=(
            _parse_validate(modes["validate"])
            if "validate" in modes
            else None
        ),
    )


def resolve_small_drn_spec(
    document: SmallDrnConfig,
    mode: RunMode,
) -> SmallDrnSpec:
    """Resolve a parsed document to the selected immutable mode spec."""

    if mode is RunMode.TRAIN:
        if document.train is None:
            raise config_error(
                "config.modes.train",
                "to be present for the 'train' command",
                None,
            )
        return TrainSpec(
            schema_version=document.schema_version,
            experiment_id=document.experiment_id,
            common=document.common,
            settings=document.train,
        )
    if mode is RunMode.LINSPACE:
        if document.linspace is None:
            raise config_error(
                "config.modes.linspace",
                "to be present for the 'linspace' command",
                None,
            )
        return LinspaceSpec(
            schema_version=document.schema_version,
            experiment_id=document.experiment_id,
            common=document.common,
            settings=document.linspace,
        )
    if mode is RunMode.VALIDATE:
        if document.validate is None:
            raise config_error(
                "config.modes.validate",
                "to be present for the 'validate' command",
                None,
            )
        return ValidateSpec(
            schema_version=document.schema_version,
            experiment_id=document.experiment_id,
            common=document.common,
            settings=document.validate,
        )
    raise ConfigError(
        "Expected the requested run mode to be 'train', 'linspace', or "
        f"'validate'. Provided value: {mode!r}."
    )
