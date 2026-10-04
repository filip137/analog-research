"""Explicit registry of crossbar experiments and their digital controls."""

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
)
from experiments.mnist_relu.config import (
    EXPERIMENT_ID as MNIST_RELU_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_RELU_SCHEMA_VERSION,
    V2_EXPERIMENT_ID as MNIST_RELU_V2_EXPERIMENT_ID,
    parse_teacher_config,
    parse_teacher_v2_config,
    resolve_teacher_spec,
)
from experiments.mnist_analog_relu.staged_config import (
    EXPERIMENT_ID as MNIST_IBM_OM_CROSSBAR_RELU_V2_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_IBM_OM_CROSSBAR_RELU_V2_SCHEMA_VERSION,
    parse_staged_crossbar_config,
    resolve_staged_crossbar_spec,
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


CIFAR10_CROSSBAR_V1 = ExperimentDefinition(
    experiment_id="cifar10_ibm_om_crossbar.v1",
    schema_version=1,
    description="CIFAR-10 dense IBM OM crossbar HWA and paired pulse recovery.",
    supported_modes=(RunMode.TRAIN,),
    parser=parse_cifar10_crossbar,
    resolver=resolve_cifar10_crossbar,
    combinations=(),
)


EXPERIMENT_REGISTRY: Dict[str, ExperimentDefinition] = {
    CIFAR10_CROSSBAR_V1.experiment_id: CIFAR10_CROSSBAR_V1,
    'cifar_om_closed_loop_lr.v1': ExperimentDefinition(
        experiment_id='cifar_om_closed_loop_lr.v1',
        schema_version=1,
        description=(
            'Development-selected uncapped closed-loop OM learning rates with paired '
            'confirmation.'
        ),
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_closed_loop_lr_config,
        resolver=resolve_cifar_closed_loop_lr_spec,
        combinations=(),
    ),
    'cifar_om_open_loop.v1': ExperimentDefinition(
        experiment_id='cifar_om_open_loop.v1',
        schema_version=1,
        description=(
            'Uncapped open-loop OM recovery from saved deployment or fresh open-loop '
            'RESET programming.'
        ),
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_om_open_loop_config,
        resolver=resolve_cifar_om_open_loop_spec,
        combinations=(),
    ),
    'cifar_crossbar_fault_sweep.v1': ExperimentDefinition(
        experiment_id='cifar_crossbar_fault_sweep.v1',
        schema_version=1,
        description=(
            'Mixed-rate corrupt-device HWA and five-epoch recovery at four/eight '
            'analog convolutions.'
        ),
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_sweep_config,
        resolver=resolve_cifar_sweep_spec,
        combinations=(),
    ),
    'cifar_crossbar_full_epochs.v1': ExperimentDefinition(
        experiment_id='cifar_crossbar_full_epochs.v1',
        schema_version=1,
        description=(
            'Five full CIFAR adaptation epochs on fixed PCM and OM arrays after '
            'digital or device-specific HWA deployment.'
        ),
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_full_epochs_config,
        resolver=resolve_cifar_full_epochs_spec,
        combinations=(),
    ),
    'cifar_pcm_recovery_epochs.v1': ExperimentDefinition(
        experiment_id='cifar_pcm_recovery_epochs.v1',
        schema_version=1,
        description=(
            'Development-selected schedules and many-pass fixed-array PCM recovery '
            'from digital and HWA sources.'
        ),
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_pcm_epochs_config,
        resolver=resolve_cifar_pcm_epochs_spec,
        combinations=(),
    ),
    'cifar_pcm_hwa_comparison.v1': ExperimentDefinition(
        experiment_id='cifar_pcm_hwa_comparison.v1',
        schema_version=1,
        description=(
            'Development-selected generic HWA and corrupt-device training followed by '
            'matched fresh-array Gaussian PCM recovery.'
        ),
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_pcm_hwa_config,
        resolver=resolve_cifar_pcm_hwa_spec,
        combinations=(),
    ),
    'cifar_pcm_fault_recovery.v1': ExperimentDefinition(
        experiment_id='cifar_pcm_fault_recovery.v1',
        schema_version=1,
        description=(
            'Permanent PCM failures, Gaussian programming endpoints and one-epoch '
            'teacher-KL recovery on CIFAR ResNet suffixes.'
        ),
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_pcm_fault_config,
        resolver=resolve_cifar_pcm_fault_spec,
        combinations=(),
    ),
    'cifar_resnet_suffix_recovery.v1': ExperimentDefinition(
        experiment_id='cifar_resnet_suffix_recovery.v1',
        schema_version=1,
        description=(
            'Pretrained CIFAR ResNet suffix HWA, fresh-array deployment and matched '
            'teacher-KL pulse recovery.'
        ),
        supported_modes=(RunMode.TRAIN,),
        parser=parse_cifar_suffix_config,
        resolver=resolve_cifar_suffix_spec,
        combinations=(),
    ),
    MNIST_RELU_V1.experiment_id: MNIST_RELU_V1,
    MNIST_RELU_V2.experiment_id: MNIST_RELU_V2,
    MNIST_IBM_OM_CROSSBAR_RELU_V2.experiment_id: MNIST_IBM_OM_CROSSBAR_RELU_V2,
    RERAM_PROGRAM_VERIFY_V1.experiment_id: RERAM_PROGRAM_VERIFY_V1,
}


def list_definitions() -> Tuple[ExperimentDefinition, ...]:
    """Return definitions in stable identifier order."""

    return tuple(
        EXPERIMENT_REGISTRY[key]
        for key in sorted(EXPERIMENT_REGISTRY)
    )


RETIRED_EXPERIMENT_IDS = frozenset({
    'ibm_om_four_reference_balance.v1',
    'ibm_om_local_reference_compensation.v1',
    'mnist_ibm_om_baseline_selection.v1',
    'mnist_ibm_om_baseline_spacing_pv.v1',
    'mnist_ibm_om_baseline_spacing_pv_no_clip.v1',
    'mnist_ibm_om_baseline_spacing_pv_truncated_nominal.v1',
    'mnist_ibm_om_crossbar_relu.v1',
    'mnist_ibm_om_winsorized_multi_assignment_qat.v1',
    'mnist_ibm_om_winsorized_pv_ensemble_qat.v1',
    'mnist_ibm_om_winsorized_qat.v1',
    'mnist_relu_drn_kd.v1',
    'mnist_relu_drn_reset.v1',
    'mnist_relu_drn_reset_bias.v1',
    'mnist_relu_drn_reset_bias_legacy.v1',
    'mnist_relu_drn_reset_differential.v1',
    'mnist_relu_drn_reset_factorial.v1',
    'small_drn.v1',
})


def get_definition(experiment_id: str) -> ExperimentDefinition:
    """Return one registered definition or raise a user-facing config error."""

    if experiment_id in RETIRED_EXPERIMENT_IDS:
        raise ConfigError(
            f"Experiment {experiment_id!r} is retired from this crossbar worktree. "
            "Use pre-cleanup commit 16938bf for its historical implementation; "
            "see docs/crossbar_scope.md."
        )
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
            "to be 'train', 'validate', or 'characterize'",
            mode,
        ) from error
    definition, document = load_experiment_config(path)
    return definition, definition.resolve(document, run_mode)
