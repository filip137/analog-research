"""Strict staged schema for the large IBM-OM MNIST crossbar study.

The v2 composition root deliberately names exactly one scientific boundary per
configuration.  Input artifacts are supplied by the public CLI; configuration
documents contain no checkpoint, device-state, receipt, or output paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, TypeAlias

from experiments.mnist_relu.config import _integer, _keys, _number, _object
from experiments.schema import RunMode, config_error


EXPERIMENT_ID = "mnist_ibm_om_crossbar_relu.v2"
SCHEMA_VERSION = 2

ADAM_LEARNING_RATE_GRID = (3e-4, 1e-3, 3e-3)
ADAM_PULSE_CAP_GRID = (128, None)
ADAM_DIAGNOSTIC_LEARNING_RATE_GRID = (
    3e-7,
    1e-6,
    0.0,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    2e-4,
    3e-4,
)
ADAM_DIAGNOSTIC_PULSE_CAP_PER_CELL = 640
ADAM_DIAGNOSTIC_OBJECTIVES = ("supervised_cross_entropy", "teacher_kl")
STAGE_KINDS = (
    "offchip_hwa",
    "deploy",
    "apply_corruption",
    "on_chip_adam",
    "on_chip_adam_diagnostic",
    "fresh_apparent_diagnostic",
)
DEPLOY_SOURCE_KINDS = ("teacher", "hwa_master", "scratch")
EVALUATION_PROFILES = (
    "tuning_validation_only",
    "diagnostic_validation_only",
    "production_full_validation_and_test",
)


@dataclass(frozen=True)
class RuntimeSettings:
    seed: int
    data_seed: int
    device: str
    dtype: str
    required_aihwkit_version: str


@dataclass(frozen=True)
class DataSettings:
    batch_size: int
    validation_points: int
    num_points: int | None
    shuffle: bool


@dataclass(frozen=True)
class ModelSettings:
    dims: tuple[int, int, int]
    bias: bool
    digital_non_linearity: str
    maximum_input_size: int
    maximum_output_size: int


@dataclass(frozen=True)
class TeacherSourceSettings:
    experiment_id: str
    architecture: str
    minimum_validation_accuracy: float
    accuracy_comparison: str


@dataclass(frozen=True)
class DeviceSettings:
    preset: str
    evidence_class: str
    device_pair_topology: str
    effective_state: str
    assignment_seed: int
    endpoint_seed: int
    healthy_programming_corruption_policy: str
    fault_source_corruption_policy: str
    bound_policy: str
    reference_policy: str
    deterministic_codebook_pulses: int
    maximum_programming_pulses: int
    verify_tolerance_x: float
    controller: str
    fault_transition: str
    fault_source_preset_default_corrupt_devices_prob: float
    fault_source_enabled_corrupt_devices_prob: float
    fault_source_corrupt_devices_range: float


@dataclass(frozen=True)
class MappingSettings:
    trained_source_weight_scaling_omega: tuple[float, float]
    trained_source_scaling_policy: str
    scratch_weight_scaling_omega: tuple[float, float]
    scratch_scaling_policy: str
    scratch_initialization: str
    scratch_initialization_seed: int
    out_of_bounds_policy: str
    codebook_tie_rule: str


@dataclass(frozen=True)
class EvaluationSettings:
    profile: str
    evaluate_validation: bool
    evaluate_test: bool
    validation_points: int
    test_points: int | None
    selection_metric: str
    primary_state: str
    secondary_state: str


@dataclass(frozen=True)
class OffchipForwardNoiseSettings:
    model: str
    base_state: str
    equation: str
    resampling: str
    samples_per_minibatch: int
    relative_scale: float
    seed: int
    gradient_estimator: str


@dataclass(frozen=True)
class OffchipHwaStageSettings:
    kind: str
    policy: str
    training_protocol: str
    epochs: int
    optimizer: str
    learning_rate: float
    beta_1: float
    beta_2: float
    epsilon: float
    objective: str
    master_q_bounds: tuple[float, float]
    maximum_batches: int
    checkpoint_policy: str
    deployment_target: str
    evaluation_forward_policy: str
    forward_noise: OffchipForwardNoiseSettings


@dataclass(frozen=True)
class DeployStageSettings:
    kind: str
    source_kind: str


@dataclass(frozen=True)
class ApplyCorruptionStageSettings:
    kind: str
    source_kind: str
    input_state_kind: str
    fault_source: str


@dataclass(frozen=True)
class LiteralAdamHyperparameters:
    source: str
    learning_rate: float
    pulse_cap_per_cell: int | None


@dataclass(frozen=True)
class SelectionReceiptAdamHyperparameters:
    source: str
    allowed_learning_rates: tuple[float, float, float]
    allowed_pulse_caps_per_cell: tuple[int | None, int | None]
    required_start_states: tuple[str, str, str, str]
    score: str
    tie_break_policy: str


AdamHyperparameters: TypeAlias = (
    LiteralAdamHyperparameters | SelectionReceiptAdamHyperparameters
)


@dataclass(frozen=True)
class OnChipAdamStageSettings:
    kind: str
    start_state: str
    epochs: int
    optimizer: str
    objective: str
    repair_examples: int
    maximum_batches: int
    beta_1: float
    beta_2: float
    epsilon: float
    layer_scope: str
    forward_state: str
    write_state: str
    gradient_estimator: str
    checkpoint_policy: str
    hyperparameters: AdamHyperparameters


@dataclass(frozen=True)
class DiagnosticLiteralAdamHyperparameters:
    source: str
    learning_rate: float
    pulse_cap_per_cell: int


@dataclass(frozen=True)
class OnChipAdamDiagnosticStageSettings:
    kind: str
    start_state: str
    epochs: int
    optimizer: str
    objective: str
    repair_examples: int
    maximum_batches: int
    beta_1: float
    beta_2: float
    epsilon: float
    layer_scope: str
    forward_state: str
    write_state: str
    gradient_estimator: str
    checkpoint_policy: str
    hyperparameters: DiagnosticLiteralAdamHyperparameters
    learning_rate_schedule: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FreshApparentDiagnosticStageSettings:
    kind: str
    start_state: str
    intervention: str
    draws: int
    relative_scale: float
    seed: int
    mutation_policy: str
    interpretation: str


StageSettings: TypeAlias = (
    OffchipHwaStageSettings
    | DeployStageSettings
    | ApplyCorruptionStageSettings
    | OnChipAdamStageSettings
    | OnChipAdamDiagnosticStageSettings
    | FreshApparentDiagnosticStageSettings
)


@dataclass(frozen=True)
class StagedCrossbarConfig:
    schema_version: int
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    source: TeacherSourceSettings
    device: DeviceSettings
    mapping: MappingSettings
    evaluation: EvaluationSettings
    stage: StageSettings


@dataclass(frozen=True)
class StagedCrossbarTrainSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    source: TeacherSourceSettings
    device: DeviceSettings
    mapping: MappingSettings
    evaluation: EvaluationSettings
    stage: StageSettings


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise config_error(path, "to be a boolean", value)
    return value


def _exact_number(value: Any, path: str, expected: float) -> float:
    parsed = _number(value, path, minimum=0.0)
    if not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1e-15):
        raise config_error(path, f"to equal {expected!r}", value)
    return expected


def _number_pair(
    value: Any,
    path: str,
    *,
    minimum: float,
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise config_error(path, "to contain exactly two finite numbers", value)
    return (
        _number(value[0], f"{path}[0]", minimum=minimum),
        _number(value[1], f"{path}[1]", minimum=minimum),
    )


def _parse_runtime(value: Any) -> RuntimeSettings:
    path = "config.runtime"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {"seed", "data_seed", "device", "dtype", "required_aihwkit_version"},
    )
    if raw["seed"] != 42 or raw["data_seed"] != 42:
        raise config_error(path, "to use matched seed=42 and data_seed=42", dict(raw))
    if raw["device"] != "cuda":
        raise config_error(f"{path}.device", "to equal 'cuda'", raw["device"])
    if raw["dtype"] != "float32":
        raise config_error(f"{path}.dtype", "to equal 'float32'", raw["dtype"])
    if raw["required_aihwkit_version"] != "1.1.0":
        raise config_error(
            f"{path}.required_aihwkit_version",
            "to equal '1.1.0'",
            raw["required_aihwkit_version"],
        )
    return RuntimeSettings(
        seed=_integer(raw["seed"], f"{path}.seed"),
        data_seed=_integer(raw["data_seed"], f"{path}.data_seed"),
        device="cuda",
        dtype="float32",
        required_aihwkit_version="1.1.0",
    )


def _parse_data(value: Any) -> DataSettings:
    path = "config.data"
    raw = _object(value, path)
    _keys(raw, path, {"batch_size", "validation_points", "num_points", "shuffle"})
    batch_size = _integer(raw["batch_size"], f"{path}.batch_size", minimum=1)
    validation_points = _integer(
        raw["validation_points"], f"{path}.validation_points", minimum=1
    )
    shuffle = _boolean(raw["shuffle"], f"{path}.shuffle")
    if (
        batch_size != 16
        or validation_points != 5000
        or raw["num_points"] is not None
        or not shuffle
    ):
        raise config_error(
            path,
            (
                "to use batch_size=16, validation_points=5000, "
                "num_points=null, and shuffle=true"
            ),
            dict(raw),
        )
    return DataSettings(16, 5000, None, True)


def _parse_model(value: Any) -> ModelSettings:
    path = "config.model"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "dims",
            "bias",
            "digital_non_linearity",
            "maximum_input_size",
            "maximum_output_size",
        },
    )
    if not isinstance(raw["dims"], (list, tuple)) or tuple(raw["dims"]) != (
        784,
        256,
        10,
    ):
        raise config_error(f"{path}.dims", "to equal [784, 256, 10]", raw["dims"])
    if raw["bias"] is not False:
        raise config_error(f"{path}.bias", "to be false", raw["bias"])
    if raw["digital_non_linearity"] != "relu":
        raise config_error(
            f"{path}.digital_non_linearity",
            "to equal 'relu'",
            raw["digital_non_linearity"],
        )
    for name in ("maximum_input_size", "maximum_output_size"):
        parsed = _integer(raw[name], f"{path}.{name}", minimum=1)
        if parsed != 512:
            raise config_error(f"{path}.{name}", "to equal 512", raw[name])
    return ModelSettings((784, 256, 10), False, "relu", 512, 512)


def _parse_source(value: Any) -> TeacherSourceSettings:
    path = "config.source"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "teacher_experiment_id",
            "teacher_architecture",
            "minimum_validation_accuracy",
            "accuracy_comparison",
        },
    )
    expected_text = {
        "teacher_experiment_id": "mnist_relu.v2",
        "teacher_architecture": "bias_free_relu_784_256_10",
        "accuracy_comparison": ">",
    }
    for name, expected in expected_text.items():
        if raw[name] != expected:
            raise config_error(f"{path}.{name}", f"to equal {expected!r}", raw[name])
    threshold = _exact_number(
        raw["minimum_validation_accuracy"],
        f"{path}.minimum_validation_accuracy",
        0.97,
    )
    return TeacherSourceSettings(
        experiment_id=expected_text["teacher_experiment_id"],
        architecture=expected_text["teacher_architecture"],
        minimum_validation_accuracy=threshold,
        accuracy_comparison=expected_text["accuracy_comparison"],
    )


def _parse_device(value: Any) -> DeviceSettings:
    path = "config.device"
    raw = _object(value, path)
    required = {
        "preset",
        "evidence_class",
        "device_pair_topology",
        "effective_state",
        "assignment_seed",
        "endpoint_seed",
        "healthy_programming_corruption_policy",
        "fault_source_corruption_policy",
        "bound_policy",
        "reference_policy",
        "deterministic_codebook_pulses",
        "maximum_programming_pulses",
        "verify_tolerance_x",
        "controller",
        "fault_transition",
        "fault_source_preset_default_corrupt_devices_prob",
        "fault_source_enabled_corrupt_devices_prob",
        "fault_source_corrupt_devices_range",
    }
    _keys(raw, path, required)
    expected_text = {
        "preset": "ReRamArrayOMPresetDevice",
        "evidence_class": "model_based_aihwkit_preset",
        "device_pair_topology": "one_active_om_plus_fixed_intrinsic_reference",
        "effective_state": "q_equals_a_minus_r",
        "healthy_programming_corruption_policy": "counterfactual_repaired",
        "fault_source_corruption_policy": "published",
        "bound_policy": "native_sampled_bounds",
        "reference_policy": "fixed_sampled_intrinsic_reference_q_equals_a_minus_r",
        "controller": "one_pulse_apparent_verify_persistent_handoff",
        "fault_transition": "post_deployment_published_companion_replay",
    }
    for name, expected in expected_text.items():
        if raw[name] != expected:
            raise config_error(f"{path}.{name}", f"to equal {expected!r}", raw[name])
    for name in ("deterministic_codebook_pulses", "maximum_programming_pulses"):
        parsed = _integer(raw[name], f"{path}.{name}", minimum=1)
        if parsed != 128:
            raise config_error(f"{path}.{name}", "to equal 128", raw[name])
    tolerance = _exact_number(
        raw["verify_tolerance_x"], f"{path}.verify_tolerance_x", 0.023725
    )
    default_corruption = _exact_number(
        raw["fault_source_preset_default_corrupt_devices_prob"],
        f"{path}.fault_source_preset_default_corrupt_devices_prob",
        0.0,
    )
    enabled_corruption = _exact_number(
        raw["fault_source_enabled_corrupt_devices_prob"],
        f"{path}.fault_source_enabled_corrupt_devices_prob",
        0.1348,
    )
    corruption_range = _exact_number(
        raw["fault_source_corrupt_devices_range"],
        f"{path}.fault_source_corrupt_devices_range",
        0.01,
    )
    return DeviceSettings(
        preset=expected_text["preset"],
        evidence_class=expected_text["evidence_class"],
        device_pair_topology=expected_text["device_pair_topology"],
        effective_state=expected_text["effective_state"],
        assignment_seed=_integer(
            raw["assignment_seed"], f"{path}.assignment_seed", minimum=1
        ),
        endpoint_seed=_integer(
            raw["endpoint_seed"], f"{path}.endpoint_seed", minimum=1
        ),
        healthy_programming_corruption_policy=expected_text[
            "healthy_programming_corruption_policy"
        ],
        fault_source_corruption_policy=expected_text[
            "fault_source_corruption_policy"
        ],
        bound_policy=expected_text["bound_policy"],
        reference_policy=expected_text["reference_policy"],
        deterministic_codebook_pulses=128,
        maximum_programming_pulses=128,
        verify_tolerance_x=tolerance,
        controller=expected_text["controller"],
        fault_transition=expected_text["fault_transition"],
        fault_source_preset_default_corrupt_devices_prob=default_corruption,
        fault_source_enabled_corrupt_devices_prob=enabled_corruption,
        fault_source_corrupt_devices_range=corruption_range,
    )


def _parse_mapping(value: Any) -> MappingSettings:
    path = "config.mapping"
    raw = _object(value, path)
    required = {
        "trained_source_weight_scaling_omega",
        "trained_source_scaling_policy",
        "scratch_weight_scaling_omega",
        "scratch_scaling_policy",
        "scratch_initialization",
        "scratch_initialization_seed",
        "out_of_bounds_policy",
        "codebook_tie_rule",
    }
    _keys(raw, path, required)
    trained_omega = _number_pair(
        raw["trained_source_weight_scaling_omega"],
        f"{path}.trained_source_weight_scaling_omega",
        minimum=0.0,
    )
    scratch_omega = _number_pair(
        raw["scratch_weight_scaling_omega"],
        f"{path}.scratch_weight_scaling_omega",
        minimum=0.0,
    )
    if trained_omega != (1.0, 1.0):
        raise config_error(
            f"{path}.trained_source_weight_scaling_omega",
            "to equal [1, 1]",
            raw["trained_source_weight_scaling_omega"],
        )
    if scratch_omega != (0.0, 0.0):
        raise config_error(
            f"{path}.scratch_weight_scaling_omega",
            "to equal [0, 0]",
            raw["scratch_weight_scaling_omega"],
        )
    expected = {
        "trained_source_scaling_policy": "one_shared_absmax_scale_per_logical_layer",
        "scratch_scaling_policy": "aihwkit_default_no_weight_scaling_direct_q",
        "scratch_initialization": "aihwkit_analog_linear_default_kaiming_uniform",
        "out_of_bounds_policy": "retain_request_and_report_endpoint_saturation",
        "codebook_tie_rule": "lowest_pulse_index",
    }
    for name, required_value in expected.items():
        if raw[name] != required_value:
            raise config_error(
                f"{path}.{name}", f"to equal {required_value!r}", raw[name]
            )
    scratch_seed = _integer(
        raw["scratch_initialization_seed"],
        f"{path}.scratch_initialization_seed",
    )
    if scratch_seed != 42:
        raise config_error(
            f"{path}.scratch_initialization_seed",
            "to equal 42",
            raw["scratch_initialization_seed"],
        )
    return MappingSettings(
        trained_source_weight_scaling_omega=(1.0, 1.0),
        trained_source_scaling_policy=expected["trained_source_scaling_policy"],
        scratch_weight_scaling_omega=(0.0, 0.0),
        scratch_scaling_policy=expected["scratch_scaling_policy"],
        scratch_initialization=expected["scratch_initialization"],
        scratch_initialization_seed=42,
        out_of_bounds_policy=expected["out_of_bounds_policy"],
        codebook_tie_rule=expected["codebook_tie_rule"],
    )


def _parse_evaluation(value: Any) -> EvaluationSettings:
    path = "config.evaluation"
    raw = _object(value, path)
    _keys(raw, path, {"profile"})
    profile = raw["profile"]
    if profile not in EVALUATION_PROFILES:
        raise config_error(
            f"{path}.profile", f"to be one of {list(EVALUATION_PROFILES)!r}", profile
        )
    production = profile == "production_full_validation_and_test"
    diagnostic = profile == "diagnostic_validation_only"
    return EvaluationSettings(
        profile=profile,
        evaluate_validation=True,
        evaluate_test=production,
        validation_points=5000,
        test_points=10000 if production else None,
        selection_metric=(
            "stage_declared_held_apparent_validation"
            if diagnostic
            else "fixed_final_epoch_no_selection"
        ),
        primary_state="held_apparent_q",
        secondary_state="hidden_persistent_q_diagnostic",
    )


def _parse_forward_noise(value: Any, path: str) -> OffchipForwardNoiseSettings:
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "model",
            "base_state",
            "equation",
            "resampling",
            "samples_per_minibatch",
            "relative_scale",
            "seed",
            "gradient_estimator",
        },
    )
    expected = {
        "model": "aihwkit_1_1_0_softbounds_reference_additive_write_noise",
        "base_state": "support_clamped_digital_master_q_surrogate",
        "equation": (
            "q_apparent_equals_q_digital_master_surrogate_plus_write_noise_std_times_"
            "nominal_dw_min_times_standard_normal"
        ),
        "resampling": "independent_full_array_draw_per_minibatch",
        "gradient_estimator": "identity_straight_through",
    }
    for name, required in expected.items():
        if raw[name] != required:
            raise config_error(f"{path}.{name}", f"to equal {required!r}", raw[name])
    samples = _integer(
        raw["samples_per_minibatch"],
        f"{path}.samples_per_minibatch",
        minimum=1,
    )
    if samples != 1:
        raise config_error(
            f"{path}.samples_per_minibatch",
            "to equal 1",
            raw["samples_per_minibatch"],
        )
    relative_scale = _exact_number(
        raw["relative_scale"], f"{path}.relative_scale", 1.0
    )
    return OffchipForwardNoiseSettings(
        model=expected["model"],
        base_state=expected["base_state"],
        equation=expected["equation"],
        resampling=expected["resampling"],
        samples_per_minibatch=1,
        relative_scale=relative_scale,
        seed=_integer(raw["seed"], f"{path}.seed", minimum=1),
        gradient_estimator=expected["gradient_estimator"],
    )


def _parse_offchip_hwa_stage(
    raw: Mapping[str, Any], path: str
) -> OffchipHwaStageSettings:
    required = {
        "kind",
        "policy",
        "training_protocol",
        "epochs",
        "optimizer",
        "learning_rate",
        "betas",
        "epsilon",
        "objective",
        "master_q_bounds",
        "maximum_batches",
        "checkpoint_policy",
        "deployment_target",
        "evaluation_forward_policy",
        "forward_noise",
    }
    _keys(raw, path, required)
    expected = {
        "policy": "stochastic_apparent_hwa",
        "training_protocol": "full_mnist_ten_epoch_fixed_final",
        "optimizer": "adam",
        "objective": "teacher_kl",
        "checkpoint_policy": "fixed_final_epoch_no_selection",
        "deployment_target": "fixed_final_master_fault_blind_pv",
        "evaluation_forward_policy": (
            "one_sampled_held_apparent_q_per_evaluation"
        ),
    }
    for name, required_value in expected.items():
        if raw[name] != required_value:
            raise config_error(
                f"{path}.{name}", f"to equal {required_value!r}", raw[name]
            )
    epochs = _integer(raw["epochs"], f"{path}.epochs", minimum=1)
    maximum_batches = _integer(
        raw["maximum_batches"], f"{path}.maximum_batches", minimum=1
    )
    if epochs != 10:
        raise config_error(f"{path}.epochs", "to equal 10", raw["epochs"])
    if maximum_batches != 3438:
        raise config_error(
            f"{path}.maximum_batches", "to equal 3438", raw["maximum_batches"]
        )
    rate = _exact_number(raw["learning_rate"], f"{path}.learning_rate", 1e-4)
    betas = _number_pair(raw["betas"], f"{path}.betas", minimum=0.0)
    if betas != (0.9, 0.999):
        raise config_error(f"{path}.betas", "to equal [0.9, 0.999]", raw["betas"])
    epsilon = _exact_number(raw["epsilon"], f"{path}.epsilon", 1e-8)
    bounds = _number_pair(raw["master_q_bounds"], f"{path}.master_q_bounds", minimum=-1.0)
    if bounds != (-1.0, 1.0):
        raise config_error(
            f"{path}.master_q_bounds", "to equal [-1, 1]", raw["master_q_bounds"]
        )
    return OffchipHwaStageSettings(
        kind="offchip_hwa",
        policy=expected["policy"],
        training_protocol=expected["training_protocol"],
        epochs=10,
        optimizer="adam",
        learning_rate=rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=epsilon,
        objective="teacher_kl",
        master_q_bounds=(-1.0, 1.0),
        maximum_batches=maximum_batches,
        checkpoint_policy=expected["checkpoint_policy"],
        deployment_target=expected["deployment_target"],
        evaluation_forward_policy=expected["evaluation_forward_policy"],
        forward_noise=_parse_forward_noise(raw["forward_noise"], f"{path}.forward_noise"),
    )


def _parse_deploy_stage(raw: Mapping[str, Any], path: str) -> DeployStageSettings:
    _keys(raw, path, {"kind", "source_kind"})
    source_kind = raw["source_kind"]
    if source_kind not in DEPLOY_SOURCE_KINDS:
        raise config_error(
            f"{path}.source_kind",
            f"to be one of {list(DEPLOY_SOURCE_KINDS)!r}",
            source_kind,
        )
    return DeployStageSettings(kind="deploy", source_kind=source_kind)


def _parse_apply_corruption_stage(
    raw: Mapping[str, Any], path: str
) -> ApplyCorruptionStageSettings:
    _keys(raw, path, {"kind", "source_kind", "input_state_kind", "fault_source"})
    source_kind = raw["source_kind"]
    if source_kind not in DEPLOY_SOURCE_KINDS:
        raise config_error(
            f"{path}.source_kind",
            f"to be one of {list(DEPLOY_SOURCE_KINDS)!r}",
            source_kind,
        )
    expected = {
        "input_state_kind": "healthy_p0",
        "fault_source": "published_companion",
    }
    for name, required in expected.items():
        if raw[name] != required:
            raise config_error(f"{path}.{name}", f"to equal {required!r}", raw[name])
    return ApplyCorruptionStageSettings(
        kind="apply_corruption",
        source_kind=source_kind,
        input_state_kind=expected["input_state_kind"],
        fault_source=expected["fault_source"],
    )


def _canonical_learning_rate(value: Any, path: str) -> float:
    parsed = _number(value, path, minimum=0.0)
    for candidate in ADAM_LEARNING_RATE_GRID:
        if math.isclose(parsed, candidate, rel_tol=0.0, abs_tol=1e-15):
            return candidate
    raise config_error(path, f"to be one of {list(ADAM_LEARNING_RATE_GRID)!r}", value)


def _pulse_cap(value: Any, path: str) -> int | None:
    if value is None:
        return None
    parsed = _integer(value, path, minimum=1)
    if parsed != 128:
        raise config_error(path, "to be 128 or null", value)
    return 128


def _parse_adam_hyperparameters(value: Any, path: str) -> AdamHyperparameters:
    raw = _object(value, path)
    source = raw.get("source")
    if source == "literal":
        _keys(raw, path, {"source", "learning_rate", "pulse_cap_per_cell"})
        return LiteralAdamHyperparameters(
            source="literal",
            learning_rate=_canonical_learning_rate(
                raw["learning_rate"], f"{path}.learning_rate"
            ),
            pulse_cap_per_cell=_pulse_cap(
                raw["pulse_cap_per_cell"], f"{path}.pulse_cap_per_cell"
            ),
        )
    if source == "strict_selection_receipt":
        _keys(
            raw,
            path,
            {
                "source",
                "allowed_learning_rates",
                "allowed_pulse_caps_per_cell",
                "required_start_states",
                "score",
                "tie_break_policy",
            },
        )
        rates_raw = raw["allowed_learning_rates"]
        if not isinstance(rates_raw, (list, tuple)) or len(rates_raw) != 3:
            raise config_error(
                f"{path}.allowed_learning_rates",
                f"to equal {list(ADAM_LEARNING_RATE_GRID)!r}",
                rates_raw,
            )
        rates = tuple(
            _canonical_learning_rate(item, f"{path}.allowed_learning_rates[{index}]")
            for index, item in enumerate(rates_raw)
        )
        if rates != ADAM_LEARNING_RATE_GRID:
            raise config_error(
                f"{path}.allowed_learning_rates",
                f"to equal {list(ADAM_LEARNING_RATE_GRID)!r}",
                rates_raw,
            )
        caps_raw = raw["allowed_pulse_caps_per_cell"]
        if not isinstance(caps_raw, (list, tuple)) or len(caps_raw) != 2:
            raise config_error(
                f"{path}.allowed_pulse_caps_per_cell",
                "to equal [128, null]",
                caps_raw,
            )
        caps = tuple(
            _pulse_cap(item, f"{path}.allowed_pulse_caps_per_cell[{index}]")
            for index, item in enumerate(caps_raw)
        )
        if caps != ADAM_PULSE_CAP_GRID:
            raise config_error(
                f"{path}.allowed_pulse_caps_per_cell",
                "to equal [128, null]",
                caps_raw,
            )
        expected_starts = (
            "hwa_healthy_p0",
            "hwa_published_fault",
            "scratch_healthy_p0",
            "scratch_published_fault",
        )
        starts_raw = raw["required_start_states"]
        if not isinstance(starts_raw, (list, tuple)) or tuple(starts_raw) != expected_starts:
            raise config_error(
                f"{path}.required_start_states",
                f"to equal {list(expected_starts)!r}",
                starts_raw,
            )
        expected_text = {
            "score": (
                "mean_final_apparent_state_validation_cross_entropy_"
                "across_four_matched_starts"
            ),
            "tie_break_policy": "prefer_pulse_cap_128_then_lower_learning_rate",
        }
        for name, required in expected_text.items():
            if raw[name] != required:
                raise config_error(f"{path}.{name}", f"to equal {required!r}", raw[name])
        return SelectionReceiptAdamHyperparameters(
            source="strict_selection_receipt",
            allowed_learning_rates=ADAM_LEARNING_RATE_GRID,
            allowed_pulse_caps_per_cell=ADAM_PULSE_CAP_GRID,
            required_start_states=expected_starts,
            score=expected_text["score"],
            tie_break_policy=expected_text["tie_break_policy"],
        )
    raise config_error(
        f"{path}.source",
        "to be 'literal' or 'strict_selection_receipt'",
        source,
    )


def _parse_on_chip_adam_stage(
    raw: Mapping[str, Any], path: str
) -> OnChipAdamStageSettings:
    required = {
        "kind",
        "start_state",
        "epochs",
        "optimizer",
        "objective",
        "repair_examples",
        "maximum_batches",
        "betas",
        "epsilon",
        "layer_scope",
        "forward_state",
        "write_state",
        "gradient_estimator",
        "checkpoint_policy",
        "hyperparameters",
    }
    _keys(raw, path, required)
    start_state = raw["start_state"]
    allowed_start_states = (
        "hwa_healthy_p0",
        "hwa_published_fault",
        "scratch_healthy_p0",
        "scratch_published_fault",
    )
    if start_state not in allowed_start_states:
        raise config_error(
            f"{path}.start_state",
            f"to be one of {list(allowed_start_states)!r}",
            start_state,
        )
    expected = {
        "optimizer": "pulse_adam",
        "objective": "supervised_cross_entropy",
        "layer_scope": "all",
        "forward_state": "held_apparent_q",
        "write_state": "persistent_q",
        "gradient_estimator": (
            "identity_ste_apparent_q_to_persistent_pulse_update"
        ),
        "checkpoint_policy": "epoch_boundary_exact_resume_fixed_final",
    }
    for name, required_value in expected.items():
        if raw[name] != required_value:
            raise config_error(
                f"{path}.{name}", f"to equal {required_value!r}", raw[name]
            )
    epochs = _integer(raw["epochs"], f"{path}.epochs", minimum=1)
    repair_examples = _integer(
        raw["repair_examples"], f"{path}.repair_examples", minimum=1
    )
    maximum_batches = _integer(
        raw["maximum_batches"], f"{path}.maximum_batches", minimum=1
    )
    if epochs != 10:
        raise config_error(f"{path}.epochs", "to equal 10", raw["epochs"])
    if repair_examples != 55000:
        raise config_error(
            f"{path}.repair_examples", "to equal 55000", raw["repair_examples"]
        )
    if maximum_batches != 3438:
        raise config_error(
            f"{path}.maximum_batches", "to equal 3438", raw["maximum_batches"]
        )
    betas = _number_pair(raw["betas"], f"{path}.betas", minimum=0.0)
    if betas != (0.9, 0.999):
        raise config_error(f"{path}.betas", "to equal [0.9, 0.999]", raw["betas"])
    epsilon = _exact_number(raw["epsilon"], f"{path}.epsilon", 1e-8)
    return OnChipAdamStageSettings(
        kind="on_chip_adam",
        start_state=start_state,
        epochs=10,
        optimizer="pulse_adam",
        objective="supervised_cross_entropy",
        repair_examples=repair_examples,
        maximum_batches=maximum_batches,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=epsilon,
        layer_scope="all",
        forward_state="held_apparent_q",
        write_state="persistent_q",
        gradient_estimator=expected["gradient_estimator"],
        checkpoint_policy=expected["checkpoint_policy"],
        hyperparameters=_parse_adam_hyperparameters(
            raw["hyperparameters"], f"{path}.hyperparameters"
        ),
    )


def _diagnostic_learning_rate(value: Any, path: str) -> float:
    parsed = _number(value, path, minimum=0.0)
    for candidate in ADAM_DIAGNOSTIC_LEARNING_RATE_GRID:
        if math.isclose(parsed, candidate, rel_tol=0.0, abs_tol=1e-15):
            return candidate
    raise config_error(
        path,
        f"to be one of {list(ADAM_DIAGNOSTIC_LEARNING_RATE_GRID)!r}",
        value,
    )


def _parse_on_chip_adam_diagnostic_stage(
    raw: Mapping[str, Any], path: str
) -> OnChipAdamDiagnosticStageSettings:
    required = {
        "kind",
        "start_state",
        "epochs",
        "optimizer",
        "objective",
        "repair_examples",
        "maximum_batches",
        "betas",
        "epsilon",
        "layer_scope",
        "forward_state",
        "write_state",
        "gradient_estimator",
        "checkpoint_policy",
        "hyperparameters",
    }
    _keys(raw, path, required | ({'learning_rate_schedule'} if 'learning_rate_schedule' in raw else set()))
    start_state = raw["start_state"]
    allowed_start_states = ("hwa_healthy_p0", "hwa_published_fault")
    if start_state not in allowed_start_states:
        raise config_error(
            f"{path}.start_state",
            f"to be one of {list(allowed_start_states)!r}",
            start_state,
        )
    expected = {
        "optimizer": "pulse_adam",
        "layer_scope": "all",
        "forward_state": "held_apparent_q",
        "write_state": "persistent_q",
        "gradient_estimator": (
            "identity_ste_apparent_q_to_persistent_pulse_update"
        ),
        "checkpoint_policy": (
            "best_held_apparent_validation_accuracy_then_objective_"
            "then_earlier_epoch_including_epoch0"
        ),
    }
    kl_policy = 'best_held_apparent_validation_objective_then_accuracy_then_earlier_epoch_including_epoch0'
    if raw['checkpoint_policy'] == kl_policy:
        expected['checkpoint_policy'] = kl_policy
    for name, required_value in expected.items():
        if raw[name] != required_value:
            raise config_error(
                f"{path}.{name}", f"to equal {required_value!r}", raw[name]
            )
    objective = raw["objective"]
    if objective not in ADAM_DIAGNOSTIC_OBJECTIVES:
        raise config_error(
            f"{path}.objective",
            f"to be one of {list(ADAM_DIAGNOSTIC_OBJECTIVES)!r}",
            objective,
        )
    epochs = _integer(raw["epochs"], f"{path}.epochs", minimum=1)
    repair_examples = _integer(
        raw["repair_examples"], f"{path}.repair_examples", minimum=1
    )
    maximum_batches = _integer(
        raw["maximum_batches"], f"{path}.maximum_batches", minimum=1
    )
    if epochs > 60:
        raise config_error(f'{path}.epochs', 'to be between 1 and 60', raw['epochs'])
    schedule = raw.get('learning_rate_schedule')
    if schedule is not None:
        schedule = _object(schedule, f'{path}.learning_rate_schedule')
        _keys(schedule, f'{path}.learning_rate_schedule', {'kind', 'final_factor', 'decay_epochs'})
        if schedule['kind'] not in {'constant', 'exponential'}:
            raise config_error(f'{path}.learning_rate_schedule.kind', 'to be constant or exponential', schedule['kind'])
        factor = _number(schedule['final_factor'], f'{path}.learning_rate_schedule.final_factor', minimum=0.0)
        horizon = _integer(schedule['decay_epochs'], f'{path}.learning_rate_schedule.decay_epochs', minimum=2)
        if not 0 < factor <= 1 or (schedule['kind'] == 'constant' and factor != 1):
            raise ValueError('Expected 0 < final_factor <= 1, with factor 1 for a constant schedule.')
        schedule = dict(kind=schedule['kind'], final_factor=factor, decay_epochs=horizon)
    if repair_examples != 55000:
        raise config_error(
            f"{path}.repair_examples", "to equal 55000", raw["repair_examples"]
        )
    if maximum_batches != 3438:
        raise config_error(
            f"{path}.maximum_batches", "to equal 3438", raw["maximum_batches"]
        )
    betas = _number_pair(raw["betas"], f"{path}.betas", minimum=0.0)
    if betas != (0.9, 0.999):
        raise config_error(f"{path}.betas", "to equal [0.9, 0.999]", raw["betas"])
    epsilon = _exact_number(raw["epsilon"], f"{path}.epsilon", 1e-8)
    hyperparameters = _object(raw["hyperparameters"], f"{path}.hyperparameters")
    _keys(
        hyperparameters,
        f"{path}.hyperparameters",
        {"source", "learning_rate", "pulse_cap_per_cell"},
    )
    if hyperparameters["source"] != "literal_diagnostic_grid":
        raise config_error(
            f"{path}.hyperparameters.source",
            "to equal 'literal_diagnostic_grid'",
            hyperparameters["source"],
        )
    pulse_cap = _integer(
        hyperparameters["pulse_cap_per_cell"],
        f"{path}.hyperparameters.pulse_cap_per_cell",
        minimum=1,
    )
    if pulse_cap != ADAM_DIAGNOSTIC_PULSE_CAP_PER_CELL:
        raise config_error(
            f"{path}.hyperparameters.pulse_cap_per_cell",
            f"to equal {ADAM_DIAGNOSTIC_PULSE_CAP_PER_CELL}",
            hyperparameters["pulse_cap_per_cell"],
        )
    return OnChipAdamDiagnosticStageSettings(
        kind="on_chip_adam_diagnostic",
        start_state=start_state,
        epochs=epochs,
        optimizer="pulse_adam",
        objective=objective,
        repair_examples=55000,
        maximum_batches=3438,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=epsilon,
        layer_scope="all",
        forward_state="held_apparent_q",
        write_state="persistent_q",
        gradient_estimator=expected["gradient_estimator"],
        checkpoint_policy=expected["checkpoint_policy"],
        learning_rate_schedule=schedule,
        hyperparameters=DiagnosticLiteralAdamHyperparameters(
            source="literal_diagnostic_grid",
            learning_rate=_diagnostic_learning_rate(
                hyperparameters["learning_rate"],
                f"{path}.hyperparameters.learning_rate",
            ),
            pulse_cap_per_cell=ADAM_DIAGNOSTIC_PULSE_CAP_PER_CELL,
        ),
    )


def _parse_fresh_apparent_diagnostic_stage(
    raw: Mapping[str, Any], path: str
) -> FreshApparentDiagnosticStageSettings:
    required = {
        "kind",
        "start_state",
        "intervention",
        "draws",
        "relative_scale",
        "seed",
        "mutation_policy",
        "interpretation",
    }
    _keys(raw, path, required)
    start_state = raw["start_state"]
    allowed_start_states = ("hwa_healthy_p0", "hwa_published_fault")
    if start_state not in allowed_start_states:
        raise config_error(
            f"{path}.start_state",
            f"to be one of {list(allowed_start_states)!r}",
            start_state,
        )
    expected = {
        "intervention": (
            "counterfactual_post_write_apparent_noise_redraw_without_device_write"
        ),
        "mutation_policy": "do_not_mutate_held_apparent_or_persistent_state",
        "interpretation": "diagnostic_only_not_physical_inference_read_noise",
    }
    for name, required_value in expected.items():
        if raw[name] != required_value:
            raise config_error(
                f"{path}.{name}", f"to equal {required_value!r}", raw[name]
            )
    draws = _integer(raw["draws"], f"{path}.draws", minimum=1)
    if draws != 4:
        raise config_error(f"{path}.draws", "to equal 4", raw["draws"])
    relative_scale = _exact_number(
        raw["relative_scale"], f"{path}.relative_scale", 1.0
    )
    return FreshApparentDiagnosticStageSettings(
        kind="fresh_apparent_diagnostic",
        start_state=start_state,
        intervention=expected["intervention"],
        draws=4,
        relative_scale=relative_scale,
        seed=_integer(raw["seed"], f"{path}.seed", minimum=1),
        mutation_policy=expected["mutation_policy"],
        interpretation=expected["interpretation"],
    )


def _parse_stage(value: Any) -> StageSettings:
    path = "config.stage"
    raw = _object(value, path)
    kind = raw.get("kind")
    if kind == "offchip_hwa":
        return _parse_offchip_hwa_stage(raw, path)
    if kind == "deploy":
        return _parse_deploy_stage(raw, path)
    if kind == "apply_corruption":
        return _parse_apply_corruption_stage(raw, path)
    if kind == "on_chip_adam":
        return _parse_on_chip_adam_stage(raw, path)
    if kind == "on_chip_adam_diagnostic":
        return _parse_on_chip_adam_diagnostic_stage(raw, path)
    if kind == "fresh_apparent_diagnostic":
        return _parse_fresh_apparent_diagnostic_stage(raw, path)
    raise config_error(f"{path}.kind", f"to be one of {list(STAGE_KINDS)!r}", kind)


def parse_staged_crossbar_config(payload: Mapping[str, Any]) -> StagedCrossbarConfig:
    """Parse one immutable stage document without touching input artifacts."""

    raw = _object(payload, "config")
    required = {
        "schema_version",
        "experiment_id",
        "runtime",
        "data",
        "model",
        "source",
        "device",
        "mapping",
        "evaluation",
        "stage",
    }
    _keys(raw, "config", required)
    schema_version = _integer(raw["schema_version"], "config.schema_version")
    if schema_version != SCHEMA_VERSION:
        raise config_error(
            "config.schema_version", f"to equal {SCHEMA_VERSION}", raw["schema_version"]
        )
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"]
        )
    runtime = _parse_runtime(raw["runtime"])
    data = _parse_data(raw["data"])
    model = _parse_model(raw["model"])
    source = _parse_source(raw["source"])
    device = _parse_device(raw["device"])
    mapping = _parse_mapping(raw["mapping"])
    evaluation = _parse_evaluation(raw["evaluation"])
    stage = _parse_stage(raw["stage"])
    if (
        isinstance(stage, OffchipHwaStageSettings)
        and evaluation.profile != "tuning_validation_only"
    ):
        raise config_error(
            "config.evaluation.profile",
            "to equal 'tuning_validation_only' for off-chip HWA",
            evaluation.profile,
        )
    if isinstance(
        stage,
        (OnChipAdamDiagnosticStageSettings, FreshApparentDiagnosticStageSettings),
    ) and evaluation.profile != "diagnostic_validation_only":
        raise config_error(
            "config.evaluation.profile",
            "to equal 'diagnostic_validation_only' for diagnostic stages",
            evaluation.profile,
        )
    if evaluation.profile == "diagnostic_validation_only" and not isinstance(
        stage,
        (OnChipAdamDiagnosticStageSettings, FreshApparentDiagnosticStageSettings),
    ):
        raise config_error(
            "config.evaluation.profile",
            "to be used only by a diagnostic stage",
            evaluation.profile,
        )
    return StagedCrossbarConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        runtime=runtime,
        data=data,
        model=model,
        source=source,
        device=device,
        mapping=mapping,
        evaluation=evaluation,
        stage=stage,
    )


def resolve_staged_crossbar_spec(
    document: StagedCrossbarConfig,
    mode: RunMode,
) -> StagedCrossbarTrainSpec:
    """Resolve the sole supported public mode for a staged v2 document."""

    if not isinstance(document, StagedCrossbarConfig):
        raise TypeError(
            "Expected a parsed StagedCrossbarConfig. "
            f"Provided value: {type(document).__name__}."
        )
    if mode is not RunMode.TRAIN:
        raise config_error("mode", "to equal 'train'", mode.value)
    return StagedCrossbarTrainSpec(
        experiment_id=document.experiment_id,
        runtime=document.runtime,
        data=document.data,
        model=document.model,
        source=document.source,
        device=document.device,
        mapping=document.mapping,
        evaluation=document.evaluation,
        stage=document.stage,
    )


__all__ = [
    "ADAM_DIAGNOSTIC_LEARNING_RATE_GRID",
    "ADAM_DIAGNOSTIC_OBJECTIVES",
    "ADAM_DIAGNOSTIC_PULSE_CAP_PER_CELL",
    "ADAM_LEARNING_RATE_GRID",
    "ADAM_PULSE_CAP_GRID",
    "DEPLOY_SOURCE_KINDS",
    "EVALUATION_PROFILES",
    "EXPERIMENT_ID",
    "SCHEMA_VERSION",
    "STAGE_KINDS",
    "AdamHyperparameters",
    "ApplyCorruptionStageSettings",
    "DataSettings",
    "DeployStageSettings",
    "DeviceSettings",
    "DiagnosticLiteralAdamHyperparameters",
    "EvaluationSettings",
    "LiteralAdamHyperparameters",
    "MappingSettings",
    "ModelSettings",
    "OffchipForwardNoiseSettings",
    "OffchipHwaStageSettings",
    "FreshApparentDiagnosticStageSettings",
    "OnChipAdamDiagnosticStageSettings",
    "OnChipAdamStageSettings",
    "RuntimeSettings",
    "SelectionReceiptAdamHyperparameters",
    "StageSettings",
    "StagedCrossbarConfig",
    "StagedCrossbarTrainSpec",
    "TeacherSourceSettings",
    "parse_staged_crossbar_config",
    "resolve_staged_crossbar_spec",
]
