"""The explicit registry of executable experiment definitions.

Registration is deliberately a plain dictionary.  There is no import-time
discovery, decorator side effect, entry-point scan, or dynamic module path in
the config file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

from experiments.schema import (
    ConfigError,
    ExperimentDefinition,
    ExtensionSelection,
    RunMode,
    ValidatedCombination,
    config_error,
    to_plain_data,
)
from experiments.small_network.config import (
    EXPERIMENT_ID,
    SCHEMA_VERSION,
    SmallDrnConfig,
    SmallDrnSpec,
    TrainSpec,
    parse_small_drn_config,
    resolve_small_drn_spec,
)
from experiments.mnist_relu.config import (
    EXPERIMENT_ID as MNIST_RELU_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_RELU_SCHEMA_VERSION,
    V2_EXPERIMENT_ID as MNIST_RELU_V2_EXPERIMENT_ID,
    parse_teacher_config,
    parse_teacher_v2_config,
    resolve_teacher_spec,
)
from experiments.mnist_analog_relu.config import (
    EXPERIMENT_ID as MNIST_IBM_OM_CROSSBAR_RELU_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_IBM_OM_CROSSBAR_RELU_SCHEMA_VERSION,
    parse_crossbar_config,
    resolve_crossbar_spec,
)
from experiments.mnist_analog_relu.staged_config import (
    EXPERIMENT_ID as MNIST_IBM_OM_CROSSBAR_RELU_V2_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_IBM_OM_CROSSBAR_RELU_V2_SCHEMA_VERSION,
    parse_staged_crossbar_config,
    resolve_staged_crossbar_spec,
)
from experiments.mnist_relu_drn.config import (
    EXPERIMENT_ID as MNIST_RELU_DRN_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_RELU_DRN_SCHEMA_VERSION,
    StudentTrainSpec,
    parse_student_config,
    resolve_student_spec,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection_config import (
    EXPERIMENT_ID as IBM_OM_BASELINE_SELECTION_EXPERIMENT_ID,
    SCHEMA_VERSION as IBM_OM_BASELINE_SELECTION_SCHEMA_VERSION,
    parse_baseline_selection_config,
    resolve_baseline_selection_spec,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_config import (
    EXPERIMENT_ID as IBM_OM_BASELINE_SPACING_PV_EXPERIMENT_ID,
    SCHEMA_VERSION as IBM_OM_BASELINE_SPACING_PV_SCHEMA_VERSION,
    parse_baseline_spacing_pv_config,
    resolve_baseline_spacing_pv_spec,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_no_clip_config import (
    EXPERIMENT_ID as IBM_OM_BASELINE_SPACING_PV_NO_CLIP_EXPERIMENT_ID,
    SCHEMA_VERSION as IBM_OM_BASELINE_SPACING_PV_NO_CLIP_SCHEMA_VERSION,
    parse_baseline_spacing_pv_no_clip_config,
    resolve_baseline_spacing_pv_no_clip_spec,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_truncated_nominal_config import (
    EXPERIMENT_ID as IBM_OM_BASELINE_SPACING_PV_TRUNCATED_NOMINAL_EXPERIMENT_ID,
    SCHEMA_VERSION as IBM_OM_BASELINE_SPACING_PV_TRUNCATED_NOMINAL_SCHEMA_VERSION,
    parse_baseline_spacing_pv_truncated_nominal_config,
    resolve_baseline_spacing_pv_truncated_nominal_spec,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_config import (
    EXPERIMENT_ID as IBM_OM_WINSORIZED_QAT_EXPERIMENT_ID,
    SCHEMA_VERSION as IBM_OM_WINSORIZED_QAT_SCHEMA_VERSION,
    parse_winsorized_qat_config,
    resolve_winsorized_qat_spec,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_config import (
    EXPERIMENT_ID as IBM_OM_WINSORIZED_MULTI_ASSIGNMENT_QAT_EXPERIMENT_ID,
    SCHEMA_VERSION as IBM_OM_WINSORIZED_MULTI_ASSIGNMENT_QAT_SCHEMA_VERSION,
    parse_winsorized_multi_assignment_qat_config,
    resolve_winsorized_multi_assignment_qat_spec,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat_config import (
    EXPERIMENT_ID as IBM_OM_WINSORIZED_PV_ENSEMBLE_QAT_EXPERIMENT_ID,
    SCHEMA_VERSION as IBM_OM_WINSORIZED_PV_ENSEMBLE_QAT_SCHEMA_VERSION,
    parse_winsorized_pv_ensemble_qat_config,
    resolve_winsorized_pv_ensemble_qat_spec,
)
from experiments.mnist_relu_drn.ibm_om_four_reference_balance_config import (
    EXPERIMENT_ID as IBM_OM_FOUR_REFERENCE_BALANCE_EXPERIMENT_ID,
    SCHEMA_VERSION as IBM_OM_FOUR_REFERENCE_BALANCE_SCHEMA_VERSION,
    parse_balance_config,
    resolve_balance_spec,
)
from experiments.mnist_relu_drn.ibm_om_local_reference_compensation_config import (
    EXPERIMENT_ID as IBM_OM_LOCAL_REFERENCE_COMPENSATION_EXPERIMENT_ID,
    SCHEMA_VERSION as IBM_OM_LOCAL_REFERENCE_COMPENSATION_SCHEMA_VERSION,
    parse_local_reference_compensation_config,
    resolve_local_reference_compensation_spec,
)
from experiments.mnist_relu_drn_reset.config import (
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
from experiments.reram_program_verify.config import (
    EXPERIMENT_ID as RERAM_PROGRAM_VERIFY_EXPERIMENT_ID,
    SCHEMA_VERSION as RERAM_PROGRAM_VERIFY_SCHEMA_VERSION,
    parse_reram_program_verify_config,
    resolve_reram_program_verify_spec,
)
from experiments.cifar_crossbar.sweep_config import (
    parse_config as parse_cifar_sweep_config,
    resolve_spec as resolve_cifar_sweep_spec,
)
from experiments.cifar_crossbar.closed_loop_lr_config import (
    parse_config as parse_cifar_closed_loop_lr_config,
    resolve_spec as resolve_cifar_closed_loop_lr_spec,
)
from experiments.cifar_crossbar.om_open_loop_config import (
    parse_config as parse_cifar_om_open_loop_config,
    resolve_spec as resolve_cifar_om_open_loop_spec,
)
from experiments.cifar_crossbar.config import (
    parse_config as parse_cifar_suffix_config,
    resolve_spec as resolve_cifar_suffix_spec,
)
from experiments.cifar_crossbar.fault_config import (
    parse_config as parse_cifar_pcm_fault_config,
    resolve_spec as resolve_cifar_pcm_fault_spec,
)
from experiments.cifar_crossbar.hwa_fault_config import (
    parse_config as parse_cifar_pcm_hwa_config,
    resolve_spec as resolve_cifar_pcm_hwa_spec,
)
from experiments.cifar_crossbar.epoch_config import (
    parse_config as parse_cifar_pcm_epochs_config,
    resolve_spec as resolve_cifar_pcm_epochs_spec,
)
from experiments.cifar_crossbar.full_epoch_config import (
    parse_config as parse_cifar_full_epochs_config,
    resolve_spec as resolve_cifar_full_epochs_spec,
)
from experiments.cifar10_crossbar.config import (
    parse_config as parse_cifar10_crossbar,
    resolve_config as resolve_cifar10_crossbar,
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


MNIST_RELU_V2 = ExperimentDefinition(
    experiment_id=MNIST_RELU_V2_EXPERIMENT_ID,
    schema_version=MNIST_RELU_SCHEMA_VERSION,
    description=(
        "Configurable bias-free 784-H-10 ReLU MNIST teacher training and "
        "validation with a strict validation-accuracy acceptance gate."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_teacher_v2_config,
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
            (
                "Bias-free digital teacher selected by validation "
                "cross-entropy and accepted only above the configured "
                "validation-accuracy threshold."
            ),
        ),
    ),
)


MNIST_IBM_OM_CROSSBAR_RELU_V1 = ExperimentDefinition(
    experiment_id=MNIST_IBM_OM_CROSSBAR_RELU_EXPERIMENT_ID,
    schema_version=MNIST_IBM_OM_CROSSBAR_RELU_SCHEMA_VERSION,
    description=(
        "Matched IBM-OM standard-crossbar MVM, digital-ReLU, standard-crossbar "
        "MVM deployment and pulse-recovery comparison against the DRN."
    ),
    supported_modes=(RunMode.TRAIN,),
    parser=parse_crossbar_config,
    resolver=resolve_crossbar_spec,
    combinations=(
        ValidatedCombination(
            ExtensionSelection(
                "standard_crossbar_fixed_reference",
                "none",
                "pulse_adam",
                "teacher_kl",
            ),
            "experimental",
            (
                "Model-based AIHWKit 1.1.0 OM control with apparent-forward "
                "program-and-verify, hidden persistent update state, and "
                "open-loop pulse recovery."
            ),
        ),
    ),
)


MNIST_IBM_OM_CROSSBAR_RELU_V2 = ExperimentDefinition(
    experiment_id=MNIST_IBM_OM_CROSSBAR_RELU_V2_EXPERIMENT_ID,
    schema_version=MNIST_IBM_OM_CROSSBAR_RELU_V2_SCHEMA_VERSION,
    description=(
        "Staged 784-256-10 IBM-OM standard-crossbar experiment separating "
        "off-chip HWA, P&V deployment, post-deployment corruption, and "
        "same-array pulse-Adam recovery."
    ),
    supported_modes=(RunMode.TRAIN,),
    parser=parse_staged_crossbar_config,
    resolver=resolve_staged_crossbar_spec,
    combinations=(),
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


IBM_OM_FOUR_REFERENCE_BALANCE_V1 = ExperimentDefinition(
    experiment_id=IBM_OM_FOUR_REFERENCE_BALANCE_EXPERIMENT_ID,
    schema_version=IBM_OM_FOUR_REFERENCE_BALANCE_SCHEMA_VERSION,
    description=(
        "Validate ideal continuous four-device OM initialization under matched "
        "sampled-order and intrinsic-reference-balanced identity bindings."
    ),
    supported_modes=(RunMode.VALIDATE,),
    parser=parse_balance_config,
    resolver=resolve_balance_spec,
)


IBM_OM_LOCAL_REFERENCE_COMPENSATION_V1 = ExperimentDefinition(
    experiment_id=IBM_OM_LOCAL_REFERENCE_COMPENSATION_EXPERIMENT_ID,
    schema_version=IBM_OM_LOCAL_REFERENCE_COMPENSATION_SCHEMA_VERSION,
    description=(
        "Validate ideal continuous four-device OM initialization with fixed "
        "identity binding and a local minimum-change exact-zero baseline."
    ),
    supported_modes=(RunMode.VALIDATE,),
    parser=parse_local_reference_compensation_config,
    resolver=resolve_local_reference_compensation_spec,
)


IBM_OM_BASELINE_SELECTION_V1 = ExperimentDefinition(
    experiment_id=IBM_OM_BASELINE_SELECTION_EXPERIMENT_ID,
    schema_version=IBM_OM_BASELINE_SELECTION_SCHEMA_VERSION,
    description=(
        "Validate four matched IBM OM baseline policies at ideal bounded "
        "continuous initialization with a frozen four-delta diagnostic."
    ),
    supported_modes=(RunMode.VALIDATE,),
    parser=parse_baseline_selection_config,
    resolver=resolve_baseline_selection_spec,
)


IBM_OM_BASELINE_SPACING_PV_V1 = ExperimentDefinition(
    experiment_id=IBM_OM_BASELINE_SPACING_PV_EXPERIMENT_ID,
    schema_version=IBM_OM_BASELINE_SPACING_PV_SCHEMA_VERSION,
    description=(
        "Validate matched four-device IBM OM shared-destination baseline "
        "positions and uniform spacings at ideal and persistent P&V endpoints."
    ),
    supported_modes=(RunMode.VALIDATE,),
    parser=parse_baseline_spacing_pv_config,
    resolver=resolve_baseline_spacing_pv_spec,
)


IBM_OM_BASELINE_SPACING_PV_NO_CLIP_V1 = ExperimentDefinition(
    experiment_id=IBM_OM_BASELINE_SPACING_PV_NO_CLIP_EXPERIMENT_ID,
    schema_version=IBM_OM_BASELINE_SPACING_PV_NO_CLIP_SCHEMA_VERSION,
    description=(
        "Validate the exploratory four-device IBM OM baseline/spacing matrix "
        "with one frozen raw-support affine translation and no circuit "
        "handoff projection."
    ),
    supported_modes=(RunMode.VALIDATE,),
    parser=parse_baseline_spacing_pv_no_clip_config,
    resolver=resolve_baseline_spacing_pv_no_clip_spec,
)


IBM_OM_BASELINE_SPACING_PV_TRUNCATED_NOMINAL_V1 = ExperimentDefinition(
    experiment_id=IBM_OM_BASELINE_SPACING_PV_TRUNCATED_NOMINAL_EXPERIMENT_ID,
    schema_version=IBM_OM_BASELINE_SPACING_PV_TRUNCATED_NOMINAL_SCHEMA_VERSION,
    description=(
        "Validate an exploratory four-device IBM OM baseline/spacing matrix "
        "after counterfactual per-identity nominal-bound Winsorization to "
        "raw a in [-1, 1] before RESET commissioning and programming."
    ),
    supported_modes=(RunMode.VALIDATE,),
    parser=parse_baseline_spacing_pv_truncated_nominal_config,
    resolver=resolve_baseline_spacing_pv_truncated_nominal_spec,
)


IBM_OM_WINSORIZED_QAT_V1 = ExperimentDefinition(
    experiment_id=IBM_OM_WINSORIZED_QAT_EXPERIMENT_ID,
    schema_version=IBM_OM_WINSORIZED_QAT_SCHEMA_VERSION,
    description=(
        "Train the alpha-zero Winsorized IBM OM four-device DRN with a "
        "deterministic uniform codebook and straight-through QAT."
    ),
    supported_modes=(RunMode.TRAIN,),
    parser=parse_winsorized_qat_config,
    resolver=resolve_winsorized_qat_spec,
)


IBM_OM_WINSORIZED_MULTI_ASSIGNMENT_QAT_V1 = ExperimentDefinition(
    experiment_id=IBM_OM_WINSORIZED_MULTI_ASSIGNMENT_QAT_EXPERIMENT_ID,
    schema_version=IBM_OM_WINSORIZED_MULTI_ASSIGNMENT_QAT_SCHEMA_VERSION,
    description=(
        "Train the alpha-zero Winsorized IBM OM four-device DRN by cycling "
        "two frozen deterministic device codebooks per global minibatch."
    ),
    supported_modes=(RunMode.TRAIN,),
    parser=parse_winsorized_multi_assignment_qat_config,
    resolver=resolve_winsorized_multi_assignment_qat_spec,
)


IBM_OM_WINSORIZED_PV_ENSEMBLE_QAT_V1 = ExperimentDefinition(
    experiment_id=IBM_OM_WINSORIZED_PV_ENSEMBLE_QAT_EXPERIMENT_ID,
    schema_version=IBM_OM_WINSORIZED_PV_ENSEMBLE_QAT_SCHEMA_VERSION,
    description=(
        "Compare deterministic, mean-P&V, and tail-robust QAT using exact "
        "precomputed persistent IBM OM endpoints on frozen device codebooks."
    ),
    supported_modes=(RunMode.TRAIN,),
    parser=parse_winsorized_pv_ensemble_qat_config,
    resolver=resolve_winsorized_pv_ensemble_qat_spec,
)


CIFAR10_CROSSBAR_V1 = ExperimentDefinition(
    experiment_id="cifar10_ibm_om_crossbar.v1",
    schema_version=1,
    description="CIFAR-10 dense IBM OM crossbar HWA and paired pulse recovery.",
    supported_modes=(RunMode.TRAIN,),
    parser=parse_cifar10_crossbar,
    resolver=resolve_cifar10_crossbar,
    combinations=(),
)


# This dictionary is the complete registration mechanism.
EXPERIMENT_REGISTRY: Dict[str, ExperimentDefinition] = {
    CIFAR10_CROSSBAR_V1.experiment_id: CIFAR10_CROSSBAR_V1,
    "cifar_om_closed_loop_lr.v1": ExperimentDefinition(
        experiment_id="cifar_om_closed_loop_lr.v1", schema_version=1,
        description="Development-selected uncapped closed-loop OM learning rates with paired confirmation.",
        supported_modes=(RunMode.TRAIN,), parser=parse_cifar_closed_loop_lr_config,
        resolver=resolve_cifar_closed_loop_lr_spec, combinations=(),
    ),
    "cifar_om_open_loop.v1": ExperimentDefinition(
        experiment_id="cifar_om_open_loop.v1", schema_version=1,
        description="Uncapped open-loop OM recovery from saved deployment or fresh open-loop RESET programming.",
        supported_modes=(RunMode.TRAIN,), parser=parse_cifar_om_open_loop_config,
        resolver=resolve_cifar_om_open_loop_spec, combinations=(),
    ),
    "cifar_crossbar_fault_sweep.v1": ExperimentDefinition(
        experiment_id="cifar_crossbar_fault_sweep.v1", schema_version=1,
        description="Mixed-rate corrupt-device HWA and five-epoch recovery at four/eight analog convolutions.",
        supported_modes=(RunMode.TRAIN,), parser=parse_cifar_sweep_config,
        resolver=resolve_cifar_sweep_spec, combinations=(),
    ),
    "cifar_crossbar_full_epochs.v1": ExperimentDefinition(
        experiment_id="cifar_crossbar_full_epochs.v1",
        schema_version=1,
        description="Five full CIFAR adaptation epochs on fixed PCM and OM arrays after digital or device-specific HWA deployment.",
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_full_epochs_config,
        resolver=resolve_cifar_full_epochs_spec,
        combinations=(),
    ),
    "cifar_pcm_recovery_epochs.v1": ExperimentDefinition(
        experiment_id="cifar_pcm_recovery_epochs.v1",
        schema_version=1,
        description="Development-selected schedules and many-pass fixed-array PCM recovery from digital and HWA sources.",
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_pcm_epochs_config,
        resolver=resolve_cifar_pcm_epochs_spec,
        combinations=(),
    ),
    "cifar_pcm_hwa_comparison.v1": ExperimentDefinition(
        experiment_id="cifar_pcm_hwa_comparison.v1",
        schema_version=1,
        description="Development-selected generic HWA and corrupt-device training followed by matched fresh-array Gaussian PCM recovery.",
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_pcm_hwa_config,
        resolver=resolve_cifar_pcm_hwa_spec,
        combinations=(),
    ),
    "cifar_pcm_fault_recovery.v1": ExperimentDefinition(
        experiment_id="cifar_pcm_fault_recovery.v1",
        schema_version=1,
        description="Permanent PCM failures, Gaussian programming endpoints and one-epoch teacher-KL recovery on CIFAR ResNet suffixes.",
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_pcm_fault_config,
        resolver=resolve_cifar_pcm_fault_spec,
        combinations=(),
    ),
    "cifar_resnet_suffix_recovery.v1": ExperimentDefinition(
        experiment_id="cifar_resnet_suffix_recovery.v1",
        schema_version=1,
        description="Pretrained CIFAR ResNet suffix HWA, fresh-array deployment and matched teacher-KL pulse recovery.",
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_suffix_config,
        resolver=resolve_cifar_suffix_spec,
        combinations=(),
    ),
    IBM_OM_BASELINE_SELECTION_V1.experiment_id: IBM_OM_BASELINE_SELECTION_V1,
    IBM_OM_BASELINE_SPACING_PV_V1.experiment_id: IBM_OM_BASELINE_SPACING_PV_V1,
    IBM_OM_BASELINE_SPACING_PV_NO_CLIP_V1.experiment_id: IBM_OM_BASELINE_SPACING_PV_NO_CLIP_V1,
    IBM_OM_BASELINE_SPACING_PV_TRUNCATED_NOMINAL_V1.experiment_id: IBM_OM_BASELINE_SPACING_PV_TRUNCATED_NOMINAL_V1,
    IBM_OM_WINSORIZED_QAT_V1.experiment_id: IBM_OM_WINSORIZED_QAT_V1,
    IBM_OM_WINSORIZED_MULTI_ASSIGNMENT_QAT_V1.experiment_id: IBM_OM_WINSORIZED_MULTI_ASSIGNMENT_QAT_V1,
    IBM_OM_WINSORIZED_PV_ENSEMBLE_QAT_V1.experiment_id: IBM_OM_WINSORIZED_PV_ENSEMBLE_QAT_V1,
    IBM_OM_FOUR_REFERENCE_BALANCE_V1.experiment_id: IBM_OM_FOUR_REFERENCE_BALANCE_V1,
    IBM_OM_LOCAL_REFERENCE_COMPENSATION_V1.experiment_id: IBM_OM_LOCAL_REFERENCE_COMPENSATION_V1,
    SMALL_DRN_V1.experiment_id: SMALL_DRN_V1,
    MNIST_RELU_V1.experiment_id: MNIST_RELU_V1,
    MNIST_RELU_V2.experiment_id: MNIST_RELU_V2,
    MNIST_IBM_OM_CROSSBAR_RELU_V1.experiment_id: MNIST_IBM_OM_CROSSBAR_RELU_V1,
    MNIST_IBM_OM_CROSSBAR_RELU_V2.experiment_id: MNIST_IBM_OM_CROSSBAR_RELU_V2,
    MNIST_RELU_DRN_KD_V1.experiment_id: MNIST_RELU_DRN_KD_V1,
    MNIST_RELU_DRN_RESET_V1.experiment_id: MNIST_RELU_DRN_RESET_V1,
    MNIST_RELU_DRN_RESET_DIFFERENTIAL_V1.experiment_id: MNIST_RELU_DRN_RESET_DIFFERENTIAL_V1,
    MNIST_RELU_DRN_RESET_BIAS_V1.experiment_id: MNIST_RELU_DRN_RESET_BIAS_V1,
    MNIST_RELU_DRN_RESET_LEGACY_BIAS_V1.experiment_id: MNIST_RELU_DRN_RESET_LEGACY_BIAS_V1,
    MNIST_RELU_DRN_RESET_FACTORIAL_V1.experiment_id: MNIST_RELU_DRN_RESET_FACTORIAL_V1,
    RERAM_PROGRAM_VERIFY_V1.experiment_id: RERAM_PROGRAM_VERIFY_V1,
}


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
