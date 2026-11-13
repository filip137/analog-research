"""The explicit registry of executable experiment definitions.

Registration is deliberately a plain dictionary.  There is no import-time
discovery, decorator side effect, entry-point scan, or dynamic module path in
the config file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

from ebl.schema.base import (
    ConfigError,
    ExperimentDefinition,
    ExtensionSelection,
    RunMode,
    ValidatedCombination,
    config_error,
    to_plain_data,
)
from ebl.schema.small_network import (
    EXPERIMENT_ID,
    SCHEMA_VERSION,
    SmallDrnConfig,
    SmallDrnSpec,
    TrainSpec,
    parse_small_drn_config,
    resolve_small_drn_spec,
)
from ebl.schema.mnist_relu import (
    EXPERIMENT_ID as MNIST_RELU_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_RELU_SCHEMA_VERSION,
    parse_teacher_config,
    resolve_teacher_spec,
)
from ebl.schema.mnist_relu_drn import (
    EXPERIMENT_ID as MNIST_RELU_DRN_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_RELU_DRN_SCHEMA_VERSION,
    StudentTrainSpec,
    parse_student_config,
    resolve_student_spec,
)
from ebl.schema.mnist_relu_drn_reset import (
    BIAS_EXPERIMENT_ID as MNIST_RELU_DRN_RESET_BIAS_EXPERIMENT_ID,
    DIFFERENTIAL_EXPERIMENT_ID as MNIST_RELU_DRN_RESET_DIFFERENTIAL_EXPERIMENT_ID,
    EXPERIMENT_ID as MNIST_RELU_DRN_RESET_EXPERIMENT_ID,
    FACTORIAL_EXPERIMENT_ID as MNIST_RELU_DRN_RESET_FACTORIAL_EXPERIMENT_ID,
    LEGACY_BIAS_EXPERIMENT_ID as MNIST_RELU_DRN_RESET_LEGACY_BIAS_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_RELU_DRN_RESET_SCHEMA_VERSION,
    ResetTrainSpec,
    parse_reset_bias_student_config,
    parse_reset_differential_student_config,
    parse_reset_factorial_student_config,
    parse_reset_legacy_bias_student_config,
    parse_reset_student_config,
    resolve_reset_student_spec,
)
from ebl.schema.reram_program_verify import (
    EXPERIMENT_ID as RERAM_PROGRAM_VERIFY_EXPERIMENT_ID,
    SCHEMA_VERSION as RERAM_PROGRAM_VERIFY_SCHEMA_VERSION,
    parse_reram_program_verify_config,
    resolve_reram_program_verify_spec,
)
from ebl.schema.raw_active_cell import (
    EXPERIMENT_ID as RAW_ACTIVE_CELL_PROGRAM_VERIFY_EXPERIMENT_ID,
    SCHEMA_VERSION as RAW_ACTIVE_CELL_PROGRAM_VERIFY_SCHEMA_VERSION,
    parse_raw_active_cell_program_verify_config,
    resolve_raw_active_cell_program_verify_spec,
)
from ebl.schema.opt_cmo_drn_drift_regimes import (
    parse_config as parse_opt_cmo_drn_drift_regimes_config,
    resolve_config as resolve_opt_cmo_drn_drift_regimes_config,
)
from ebl.schema.opt_cmo_drn import (
    parse_config as parse_opt_cmo_drn_config,
    resolve_config as resolve_opt_cmo_drn_config,
)


_SMALL_DRN_COMBINATIONS: Tuple[ValidatedCombination, ...] = (
    ValidatedCombination(
        ExtensionSelection("none", "none", "direct", "ep"),
        "validated",
        "Reference equilibrium-propagation path.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "none", "direct", "backprop"),
        "validated",
        "Reference backpropagation path.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "none", "direct_adam", "backprop"),
        "experimental",
        "Bounded backpropagation with ordinary digital Adam updates.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "none",
            "none",
            "ibm_om_fp32_bounds",
            "backprop",
        ),
        "experimental",
        "Array-specific ideal FP32 training projected into each sampled IBM "
        "OM cell's effective controller bounds.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "none", "tiki_taka", "ep"),
        "experimental",
        "Tiki-taka backend with either ideal or AIHWKit parameters.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "none", "tiki_taka", "backprop"),
        "experimental",
        "Tiki-taka backend with backpropagation.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "add_normal", "direct", "backprop"),
        "experimental",
        "Additive Gaussian weight noise with backpropagation.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "none", "program_verify", "backprop"),
        "experimental",
        "BPTT with each dense-weight update reprogrammed through a measured "
        "endpoint noise model.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "none",
            "none",
            "measured_cohort_a",
            "backprop",
        ),
        "experimental",
        "BPTT with digital shadows projected onto interpolated measured "
        "cohort-A ReRAM conductance traces.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "none",
            "none",
            "measured_cohort_b",
            "backprop",
        ),
        "experimental",
        "BPTT after projecting a named cohort-A checkpoint onto independently "
        "held-out interpolated cohort-B ReRAM conductance traces.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "passive_low_rank",
            "none",
            "direct",
            "ep",
        ),
        "experimental",
        "Passive low-rank factors trained by direct EP updates.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "passive_low_rank",
            "none",
            "tiki_taka",
            "ep",
        ),
        "experimental",
        "Passive low-rank factors trained by ideal-tensor Tiki-Taka.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "digital_low_rank",
            "none",
            "direct",
            "digital",
        ),
        "experimental",
        "Ideal FP32 logit residual recovering a frozen programmed DRN.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "passive_layerwise_low_rank",
            "none",
            "direct",
            "ep",
        ),
        "experimental",
        "Two ideal passive conductance branches across frozen DRN edges.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "passive_layerwise_low_rank",
            "none",
            "direct",
            "backprop",
        ),
        "experimental",
        "Two ideal passive conductance branches trained through unrolled "
        "DRN minimization.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "passive_layerwise_low_rank",
            "none",
            "program_verify",
            "backprop",
        ),
        "experimental",
        "Frozen programmed base edges with BPTT LoRA factors reprogrammed "
        "noisily after every update.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "passive_layerwise_low_rank",
            "none",
            "measured_cohort_b_lora",
            "backprop",
        ),
        "experimental",
        "Frozen cohort-B deployed base edges with four fully reset measured "
        "ReRAM low-rank factor arrays trained for accuracy recovery.",
    ),
)


def _resolve_small_drn(
    document: SmallDrnConfig,
    mode: RunMode,
) -> SmallDrnSpec:
    spec = resolve_small_drn_spec(document, mode)
    if isinstance(spec, TrainSpec):
        combination = next(
            (
                item
                for item in _SMALL_DRN_COMBINATIONS
                if item.selection == spec.extensions
            ),
            None,
        )
        if combination is None:
            raise config_error(
                "the train extension combination "
                "(model.adapter, weight_modifier, update_backend, algorithm)",
                "to be listed explicitly by the 'small_drn.v1' definition",
                (
                    spec.extensions.model_adapter,
                    spec.extensions.weight_modifier,
                    spec.extensions.update_backend,
                    spec.extensions.algorithm,
                ),
            )
        if (
            spec.extensions.model_adapter == "passive_low_rank"
            and spec.extensions.update_backend == "tiki_taka"
            and spec.settings.update_backend.parameters.get(
                "aihwkit_preset"
            )
            is not None
        ):
            raise config_error(
                "config.modes.train.update_backend.parameters.aihwkit_preset",
                "to be null or omitted for passive_low_rank because only the "
                "ideal-tensor Tiki-Taka backend is validated",
                spec.settings.update_backend.parameters.get(
                    "aihwkit_preset"
                ),
            )
    return spec


SMALL_DRN_V1 = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    schema_version=SCHEMA_VERSION,
    description=(
        "Small dissipative-resistive-network training, linspace analysis, "
        "and checkpoint validation."
    ),
    supported_modes=(
        RunMode.TRAIN,
        RunMode.LINSPACE,
        RunMode.VALIDATE,
    ),
    parser=parse_small_drn_config,
    resolver=_resolve_small_drn,
    combinations=_SMALL_DRN_COMBINATIONS,
    legacy_names=("labs.small_network",),
)


MNIST_RELU_V1 = ExperimentDefinition(
    experiment_id=MNIST_RELU_EXPERIMENT_ID,
    schema_version=MNIST_RELU_SCHEMA_VERSION,
    description=(
        "Bias-free 784-50-10 ReLU MNIST teacher training and validation."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_teacher_config,
    resolver=resolve_teacher_spec,
    combinations=(
        ValidatedCombination(
            ExtensionSelection(
                "relu_teacher",
                "none",
                "adam",
                "cross_entropy",
            ),
            "validated",
            "Bias-free digital teacher selected by validation cross-entropy.",
        ),
    ),
)


RERAM_PROGRAM_VERIFY_V1 = ExperimentDefinition(
    experiment_id=RERAM_PROGRAM_VERIFY_EXPERIMENT_ID,
    schema_version=RERAM_PROGRAM_VERIFY_SCHEMA_VERSION,
    description=(
        "Pulse-resolved characterization of IBM ReRAM array presets with "
        "one-pulse and adaptive program-and-verify controllers."
    ),
    supported_modes=(RunMode.CHARACTERIZE,),
    parser=parse_reram_program_verify_config,
    resolver=resolve_reram_program_verify_spec,
)


RAW_ACTIVE_CELL_PROGRAM_VERIFY_V1 = ExperimentDefinition(
    experiment_id=RAW_ACTIVE_CELL_PROGRAM_VERIFY_EXPERIMENT_ID,
    schema_version=RAW_ACTIVE_CELL_PROGRAM_VERIFY_SCHEMA_VERSION,
    description=(
        "Pulse-resolved single-active-state IBM OM characterization of one "
        "array-wide p90 nine-level physical-cell codebook."
    ),
    supported_modes=(RunMode.CHARACTERIZE,),
    parser=parse_raw_active_cell_program_verify_config,
    resolver=resolve_raw_active_cell_program_verify_spec,
)


_MNIST_RELU_DRN_COMBINATIONS: Tuple[ValidatedCombination, ...] = tuple(
    ValidatedCombination(
        ExtensionSelection(encoding, "none", backend, "teacher_kl"),
        "experimental",
        (
            "Teacher-mapped DRN with pure KL distillation and "
            + (
                (
                    "one device per physical edge; the measured cohort-A "
                    "path includes strict model-local dual-rail pairwise "
                    "and four-cell-block common-window initializations."
                    if backend == "measured_cohort_a"
                    else "one device per physical edge."
                )
                if encoding == "single"
                else (
                    "a differential G+/G- pair per physical edge with "
                    "finite positive voltage/current amplifier magnitudes."
                )
            )
        ),
    )
    for encoding in ("single", "differential")
    for backend in ("ideal", "measured_cohort_a")
) + (
    *(
        ValidatedCombination(
            ExtensionSelection(
                encoding,
                "add_normal",
                "ideal",
                "teacher_kl",
            ),
            "experimental",
            (
                "Teacher-mapped DRN hardware-aware training with temporary "
                "output-channel-scaled additive Gaussian conductance noise."
            ),
        )
        for encoding in ("single", "differential")
    ),
    ValidatedCombination(
        ExtensionSelection(
            "single",
            "ibm_reram_om_program_verify",
            "ideal",
            "teacher_kl",
        ),
        "experimental",
        (
            "Off-chip BPTT with fixed IBM OM per-cell bounds and corruption "
            "identity, using a fresh calibrated cap-128 programming endpoint "
            "for each minibatch."
        ),
    ),
    ValidatedCombination(
        ExtensionSelection(
            "differential",
            "ibm_reram_om_program_verify",
            "ideal",
            "teacher_kl",
        ),
        "experimental",
        (
            "Off-chip BPTT for the eight-device differential DRN using one "
            "fixed IBM OM assignment, canonical adjacent G+/G- pair common "
            "windows, and a fresh calibrated cap-128 programming endpoint "
            "for each minibatch."
        ),
    ),
    *(
        ValidatedCombination(
            ExtensionSelection(
                encoding,
                "none",
                "program_verify",
                "teacher_kl",
            ),
            "experimental",
            (
                "Teacher-mapped DRN deployment and full BPTT with a fresh "
                "program-and-verify endpoint-device realization after every "
                "conductance update."
            ),
        )
        for encoding in ("single", "differential")
    ),
    ValidatedCombination(
        ExtensionSelection(
            "single",
            "none",
            "measured_cohort_b",
            "teacher_kl",
        ),
        "experimental",
        (
            "Four-device single-conductance deployment and fine-tuning on "
            "independently held-out measured cohort-B traces using one "
            "common reachable window per dual-rail four-cell block."
        ),
    ),
    ValidatedCombination(
        ExtensionSelection(
            "differential",
            "none",
            "measured_cohort_b",
            "teacher_kl",
        ),
        "experimental",
        (
            "Differential G+/G- deployment and fine-tuning on independently "
            "held-out measured cohort-B device traces."
        ),
    ),
    ValidatedCombination(
        ExtensionSelection(
            "single",
            "none",
            "measured_cohort_a_sign_sgd",
            "teacher_kl",
        ),
        "experimental",
        (
            "Four-device cohort-A quad-common-window training with "
            "fixed-magnitude signSGD shadow updates followed by measured "
            "global-nearest projection."
        ),
    ),
    ValidatedCombination(
        ExtensionSelection(
            "single",
            "none",
            "measured_cohort_a_one_pulse_down",
            "teacher_kl",
        ),
        "experimental",
        (
            "Four-device cohort-A quad-common-window initialization followed "
            "by strict isotonic local updates: gradients above explicit "
            "per-parameter thresholds advance exactly one pulse toward lower "
            "conductance and all other gradients hold. Omitted thresholds "
            "define the zero-threshold control."
        ),
    ),
    ValidatedCombination(
        ExtensionSelection(
            "differential",
            "none",
            "measured_cohort_a_one_pulse_down",
            "teacher_kl",
        ),
        "experimental",
        (
            "Eight-device cohort-A paired-common-window initialization "
            "followed by the same threshold-gated one-pulse-down rule on "
            "each physical G+/G- conductance tensor."
        ),
    ),
    ValidatedCombination(
        ExtensionSelection(
            "single",
            "none",
            "ibm_om_deployed_recovery",
            "teacher_kl",
        ),
        "experimental",
        (
            "Pulse-compatible recovery from one exact saved IBM OM apparent/"
            "persistent deployment using rail refresh, direct pulses, or a "
            "matched Tiki-Taka transfer path."
        ),
    ),
)


def _resolve_mnist_relu_drn(document, mode: RunMode):
    spec = resolve_student_spec(document, mode)
    if isinstance(spec, StudentTrainSpec):
        selection = ExtensionSelection(
            spec.model.encoding,
            spec.settings.weight_modifier.type,
            spec.settings.update_backend.type,
            "teacher_kl",
        )
        if not any(
            item.selection == selection
            for item in _MNIST_RELU_DRN_COMBINATIONS
        ):
            raise config_error(
                "the train extension combination",
                "to be listed explicitly by the "
                "'mnist_relu_drn_kd.v1' definition",
                to_plain_data(selection),
            )
    return spec


MNIST_RELU_DRN_KD_V1 = ExperimentDefinition(
    experiment_id=MNIST_RELU_DRN_EXPERIMENT_ID,
    schema_version=MNIST_RELU_DRN_SCHEMA_VERSION,
    description=(
        "Teacher-initialized MNIST DRN trained with pure teacher-to-student KL."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_student_config,
    resolver=_resolve_mnist_relu_drn,
    combinations=_MNIST_RELU_DRN_COMBINATIONS,
)


_MNIST_RELU_DRN_RESET_COMBINATIONS: Tuple[ValidatedCombination, ...] = (
    ValidatedCombination(
        ExtensionSelection(
            "single", "none", "measured_cohort_a", "teacher_kl"
        ),
        "experimental",
        "One measured device per physical dual-rail edge, trained from RESET "
        "against a frozen ReLU teacher.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "single", "none", "measured_cohort_a", "cross_entropy"
        ),
        "experimental",
        "Matched hard-label control using the same RESET device assignment.",
    ),
)


def _resolve_mnist_relu_drn_reset(document, mode: RunMode):
    spec = resolve_reset_student_spec(document, mode)
    if isinstance(spec, ResetTrainSpec):
        selection = ExtensionSelection(
            spec.model.encoding,
            "none",
            spec.settings.update_backend.type,
            spec.settings.objective,
        )
        if not any(
            item.selection == selection
            for item in _MNIST_RELU_DRN_RESET_COMBINATIONS
        ):
            raise config_error(
                "the train extension combination",
                "to be listed explicitly by the "
                "'mnist_relu_drn_reset.v1' definition",
                to_plain_data(selection),
            )
    return spec


MNIST_RELU_DRN_RESET_V1 = ExperimentDefinition(
    experiment_id=MNIST_RELU_DRN_RESET_EXPERIMENT_ID,
    schema_version=MNIST_RELU_DRN_RESET_SCHEMA_VERSION,
    description=(
        "Single-device dual-rail MNIST DRN trained from measured cohort-A "
        "RESET using teacher KL or matched label supervision."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_reset_student_config,
    resolver=_resolve_mnist_relu_drn_reset,
    combinations=_MNIST_RELU_DRN_RESET_COMBINATIONS,
)


_MNIST_RELU_DRN_RESET_DIFFERENTIAL_COMBINATIONS: Tuple[
    ValidatedCombination, ...
] = (
    ValidatedCombination(
        ExtensionSelection(
            "differential_logical",
            "none",
            "measured_cohort_a",
            "paired_squared_error",
        ),
        "experimental",
        "Bias-free model-local G+/G- pairs trained from measured cohort-A "
        "RESET using paired-output squared error.",
    ),
)


def _resolve_mnist_relu_drn_reset_differential(document, mode: RunMode):
    spec = resolve_reset_student_spec(document, mode)
    if isinstance(spec, ResetTrainSpec):
        selection = ExtensionSelection(
            "differential_logical",
            "none",
            spec.settings.update_backend.type,
            spec.settings.objective,
        )
        if not any(
            item.selection == selection
            for item in _MNIST_RELU_DRN_RESET_DIFFERENTIAL_COMBINATIONS
        ):
            raise config_error(
                "the train extension combination",
                "to be listed explicitly by the "
                "'mnist_relu_drn_reset_differential.v1' definition",
                to_plain_data(selection),
            )
    return spec


MNIST_RELU_DRN_RESET_DIFFERENTIAL_V1 = ExperimentDefinition(
    experiment_id=MNIST_RELU_DRN_RESET_DIFFERENTIAL_EXPERIMENT_ID,
    schema_version=MNIST_RELU_DRN_RESET_SCHEMA_VERSION,
    description=(
        "Differential-pair dual-rail MNIST DRN trained from measured "
        "cohort-A RESET with model-local weighted equilibria."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_reset_differential_student_config,
    resolver=_resolve_mnist_relu_drn_reset_differential,
    combinations=_MNIST_RELU_DRN_RESET_DIFFERENTIAL_COMBINATIONS,
)


_MNIST_RELU_DRN_RESET_BIAS_COMBINATIONS: Tuple[
    ValidatedCombination, ...
] = tuple(
    ValidatedCombination(
        ExtensionSelection(
            "single_bias", "none", "measured_cohort_a", objective
        ),
        "experimental",
        note,
    )
    for objective, note in (
        (
            "paired_squared_error",
            "Historical paired-output squared-error control with the original "
            "trainable digital hidden bias.",
        ),
        (
            "cross_entropy",
            "Matched hard-label cross-entropy arm with the original trainable "
            "digital hidden bias.",
        ),
        (
            "teacher_kl",
            "Teacher-KL arm with the original trainable digital hidden bias.",
        ),
    )
)


def _resolve_mnist_relu_drn_reset_bias(document, mode: RunMode):
    spec = resolve_reset_student_spec(document, mode)
    if isinstance(spec, ResetTrainSpec):
        selection = ExtensionSelection(
            f"{spec.model.encoding}_bias",
            "none",
            spec.settings.update_backend.type,
            spec.settings.objective,
        )
        if not any(
            item.selection == selection
            for item in _MNIST_RELU_DRN_RESET_BIAS_COMBINATIONS
        ):
            raise config_error(
                "the train extension combination",
                "to be listed explicitly by the "
                "'mnist_relu_drn_reset_bias.v1' definition",
                to_plain_data(selection),
            )
    return spec


MNIST_RELU_DRN_RESET_BIAS_V1 = ExperimentDefinition(
    experiment_id=MNIST_RELU_DRN_RESET_BIAS_EXPERIMENT_ID,
    schema_version=MNIST_RELU_DRN_RESET_SCHEMA_VERSION,
    description=(
        "Single-device dual-rail MNIST DRN with the historical trainable "
        "digital hidden bias, trained from measured cohort-A RESET under "
        "paired squared error, cross-entropy, or teacher KL."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_reset_bias_student_config,
    resolver=_resolve_mnist_relu_drn_reset_bias,
    combinations=_MNIST_RELU_DRN_RESET_BIAS_COMBINATIONS,
)


_MNIST_RELU_DRN_RESET_LEGACY_BIAS_COMBINATIONS: Tuple[
    ValidatedCombination, ...
] = tuple(
    ValidatedCombination(
        ExtensionSelection(
            "single_bias_legacy_process_index",
            "none",
            "measured_cohort_a",
            objective,
        ),
        "historical_control",
        "Explicit replay of the archived process-global amplifier indexing; "
        "this is a provenance control, not the intended logical 4/0.25 circuit.",
    )
    for objective in (
        "paired_squared_error",
        "cross_entropy",
        "teacher_kl",
    )
)


def _resolve_mnist_relu_drn_reset_legacy_bias(document, mode: RunMode):
    spec = resolve_reset_student_spec(document, mode)
    if isinstance(spec, ResetTrainSpec):
        selection = ExtensionSelection(
            "single_bias_legacy_process_index",
            "none",
            spec.settings.update_backend.type,
            spec.settings.objective,
        )
        if not any(
            item.selection == selection
            for item in _MNIST_RELU_DRN_RESET_LEGACY_BIAS_COMBINATIONS
        ):
            raise config_error(
                "the train extension combination",
                "to be listed explicitly by the "
                "'mnist_relu_drn_reset_bias_legacy.v1' definition",
                to_plain_data(selection),
            )
    return spec


MNIST_RELU_DRN_RESET_LEGACY_BIAS_V1 = ExperimentDefinition(
    experiment_id=MNIST_RELU_DRN_RESET_LEGACY_BIAS_EXPERIMENT_ID,
    schema_version=MNIST_RELU_DRN_RESET_SCHEMA_VERSION,
    description=(
        "Historical-control replay of the RESET-trained single-device MNIST "
        "DRN with trainable digital bias and archived process-global "
        "amplifier indexing, under three supervision losses."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_reset_legacy_bias_student_config,
    resolver=_resolve_mnist_relu_drn_reset_legacy_bias,
    combinations=_MNIST_RELU_DRN_RESET_LEGACY_BIAS_COMBINATIONS,
)


_MNIST_RELU_DRN_RESET_FACTORIAL_COMBINATIONS: Tuple[
    ValidatedCombination, ...
] = tuple(
    ValidatedCombination(
        ExtensionSelection(
            (
                "single_bias" if include_biases else "single"
            )
            + (
                "_logical"
                if amplification_indexing == "logical"
                else "_legacy_process_index"
            ),
            "none",
            "measured_cohort_a",
            objective,
        ),
        "experimental",
        "Controlled factorial arm with explicit minibatch input reset and "
        "an identical reseed/rebuild production lifecycle.",
    )
    for include_biases in (False, True)
    for amplification_indexing in ("logical", "legacy_process_global")
    for objective in ("paired_squared_error", "teacher_kl")
)


def _resolve_mnist_relu_drn_reset_factorial(document, mode: RunMode):
    spec = resolve_reset_student_spec(document, mode)
    if isinstance(spec, ResetTrainSpec):
        adapter = "single_bias" if spec.model.include_biases else "single"
        adapter += (
            "_logical"
            if spec.model.amplification_indexing == "logical"
            else "_legacy_process_index"
        )
        selection = ExtensionSelection(
            adapter,
            "none",
            spec.settings.update_backend.type,
            spec.settings.objective,
        )
        if not any(
            item.selection == selection
            for item in _MNIST_RELU_DRN_RESET_FACTORIAL_COMBINATIONS
        ):
            raise config_error(
                "the train extension combination",
                "to be listed explicitly by the "
                "'mnist_relu_drn_reset_factorial.v1' definition",
                to_plain_data(selection),
            )
    return spec


MNIST_RELU_DRN_RESET_FACTORIAL_V1 = ExperimentDefinition(
    experiment_id=MNIST_RELU_DRN_RESET_FACTORIAL_EXPERIMENT_ID,
    schema_version=MNIST_RELU_DRN_RESET_SCHEMA_VERSION,
    description=(
        "Controlled RESET-trained MNIST factorial crossing digital hidden "
        "bias, paired-MSE versus teacher-KL supervision, and logical versus "
        "archived process-global amplifier indexing."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_reset_factorial_student_config,
    resolver=_resolve_mnist_relu_drn_reset_factorial,
    combinations=_MNIST_RELU_DRN_RESET_FACTORIAL_COMBINATIONS,
)


# This dictionary is the complete registration mechanism.
OPT_CMO_DRN_HWA_V1 = ExperimentDefinition(
    experiment_id="opt_cmo_drn_hwa.v1",
    schema_version=1,
    description="CMO-initialized OPT-125M with six coupled perfect-diode DRNs, whole-model HWA and matched physical recovery.",
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_opt_cmo_drn_config,
    resolver=resolve_opt_cmo_drn_config,
)

OPT_CMO_DRN_DRIFT_REGIMES_V1 = ExperimentDefinition(
    experiment_id="opt_cmo_drn_drift_regimes.v1",
    schema_version=1,
    description="exp-003 drift-regimes screen on the frozen OPT CMO DRN: matched P&V deployment, power-law drift family and maintenance arms.",
    supported_modes=(RunMode.TRAIN,),
    parser=parse_opt_cmo_drn_drift_regimes_config,
    resolver=resolve_opt_cmo_drn_drift_regimes_config,
)

EXPERIMENT_REGISTRY: Dict[str, ExperimentDefinition] = {
    OPT_CMO_DRN_HWA_V1.experiment_id: OPT_CMO_DRN_HWA_V1,
    OPT_CMO_DRN_DRIFT_REGIMES_V1.experiment_id: OPT_CMO_DRN_DRIFT_REGIMES_V1,
    SMALL_DRN_V1.experiment_id: SMALL_DRN_V1,
    MNIST_RELU_V1.experiment_id: MNIST_RELU_V1,
    MNIST_RELU_DRN_KD_V1.experiment_id: MNIST_RELU_DRN_KD_V1,
    MNIST_RELU_DRN_RESET_V1.experiment_id: MNIST_RELU_DRN_RESET_V1,
    MNIST_RELU_DRN_RESET_DIFFERENTIAL_V1.experiment_id: MNIST_RELU_DRN_RESET_DIFFERENTIAL_V1,
    MNIST_RELU_DRN_RESET_BIAS_V1.experiment_id: MNIST_RELU_DRN_RESET_BIAS_V1,
    MNIST_RELU_DRN_RESET_LEGACY_BIAS_V1.experiment_id: MNIST_RELU_DRN_RESET_LEGACY_BIAS_V1,
    MNIST_RELU_DRN_RESET_FACTORIAL_V1.experiment_id: MNIST_RELU_DRN_RESET_FACTORIAL_V1,
    RERAM_PROGRAM_VERIFY_V1.experiment_id: RERAM_PROGRAM_VERIFY_V1,
    RAW_ACTIVE_CELL_PROGRAM_VERIFY_V1.experiment_id: (
        RAW_ACTIVE_CELL_PROGRAM_VERIFY_V1
    ),
}

from ebl.schema.sparse_equilibrium import parse_config as _sparse_parse, resolve_config as _sparse_resolve
EXPERIMENT_REGISTRY['sparse_equilibrium_pilot.v1'] = ExperimentDefinition(
    experiment_id='sparse_equilibrium_pilot.v1', schema_version=1,
    description='Bounded exploratory sparse-equilibrium graph and counterfactual-sentiment pilots.',
    supported_modes=(RunMode.TRAIN,), parser=_sparse_parse, resolver=_sparse_resolve,
)


from ebl.schema.analog_energy_transformer import (
    parse_config as _energy_attention_parse,
    resolve_config as _energy_attention_resolve,
)
EXPERIMENT_REGISTRY["analog_energy_transformer.v1"] = ExperimentDefinition(
    experiment_id="analog_energy_transformer.v1", schema_version=1,
    description="Ideal signed reciprocal softmax circuit and equilibrium-propagation verification.",
    supported_modes=(RunMode.CHARACTERIZE,),
    parser=_energy_attention_parse, resolver=_energy_attention_resolve,
)


from ebl.schema.convergent_energy_transformer import (
    parse_config as _convergent_et_parse,
    resolve_config as _convergent_et_resolve,
)
EXPERIMENT_REGISTRY["convergent_energy_transformer.v1"] = ExperimentDefinition(
    experiment_id="convergent_energy_transformer.v1", schema_version=1,
    description="Convergent multi-token Energy Transformer analog circuit and EP checks.",
    supported_modes=(RunMode.CHARACTERIZE,),
    parser=_convergent_et_parse, resolver=_convergent_et_resolve,
)


from ebl.schema.mos_attention_learning import (
    parse_config as _mos_learning_parse, resolve_config as _mos_learning_resolve,
)
EXPERIMENT_REGISTRY["mos_attention_learning.v1"] = ExperimentDefinition(
    experiment_id="mos_attention_learning.v1", schema_version=1,
    description="Native MOS attention with measured-SPICE equilibrium-propagation task learning.",
    supported_modes=(RunMode.TRAIN,), parser=_mos_learning_parse, resolver=_mos_learning_resolve,
)


def list_definitions() -> Tuple[ExperimentDefinition, ...]:
    """Return definitions in stable identifier order."""

    return tuple(
        EXPERIMENT_REGISTRY[key]
        for key in sorted(EXPERIMENT_REGISTRY)
    )


def get_definition(experiment_id: str) -> ExperimentDefinition:
    """Return one registered definition or raise a user-facing config error."""

    try:
        return EXPERIMENT_REGISTRY[experiment_id]
    except KeyError as error:
        raise config_error(
            "config.experiment_id",
            "to be one of "
            + ", ".join(repr(key) for key in sorted(EXPERIMENT_REGISTRY)),
            experiment_id,
        ) from error


def parse_experiment_config(
    payload: Mapping[str, Any],
) -> Tuple[ExperimentDefinition, Any]:
    """Select the definition from the document and parse it strictly."""

    if not isinstance(payload, Mapping):
        raise config_error("config", "to be a JSON object", payload)
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise config_error(
            "config.experiment_id",
            "to be a non-empty registered identifier",
            experiment_id,
        )
    definition = get_definition(experiment_id)
    return definition, definition.parse(payload)


def load_experiment_config(
    path: Union[str, Path],
) -> Tuple[ExperimentDefinition, Any]:
    """Read and parse one versioned JSON experiment document."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(
            "Expected --config to reference a readable JSON file. "
            f"Provided value: {str(config_path)!r}. {error}"
        ) from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(
            "Expected --config to contain valid JSON. "
            f"Provided value: {str(config_path)!r} "
            f"(line {error.lineno}, column {error.colno}: {error.msg})."
        ) from error
    return parse_experiment_config(payload)


def resolve_experiment_config(
    path: Union[str, Path],
    mode: Union[str, RunMode],
) -> Tuple[ExperimentDefinition, Any]:
    """Load a config and resolve the selected mode-specific immutable spec."""

    try:
        run_mode = mode if isinstance(mode, RunMode) else RunMode(mode)
    except ValueError as error:
        raise config_error(
            "the requested run mode",
            "to be 'train', 'linspace', or 'validate'",
            mode,
        ) from error
    definition, document = load_experiment_config(path)
    return definition, definition.resolve(document, run_mode)
