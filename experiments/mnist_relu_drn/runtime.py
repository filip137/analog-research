"""Application runtime for teacher-mapped DRN KL distillation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import torch
import torch.nn.functional as F

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_relu_drn.components import (
    BoundedDrnTeacher,
    StudentStack,
    build_student_stack,
    collect_calibration,
    conductance_statistics,
    fit_positive_logit_gain,
    measured_optimizer_type,
    select_mapping_and_gain,
)
from experiments.mnist_relu_drn.config import (
    IBM_OM_DEPLOYED_RECOVERY_BACKEND,
    MEASURED_BACKENDS,
    MEASURED_COHORT_A_BACKENDS,
    MEASURED_COHORT_B_BACKENDS,
    StudentTrainSpec,
    StudentValidateSpec,
)
from experiments.mnist_shared import build_mnist_loaders, teacher_batches
from experiments.schema import to_plain_data
from model.resistive.interaction import DenseResistive, SignedDenseResistive
from model.resistive.device_config import parse_device_programming_config
from training.add_normal import AddNormalConfig, build_add_normal_modifier
from training.checkpoint import (
    atomic_torch_save,
    encode_named_weights,
    load_epoch_boundary_checkpoint,
    load_named_weights,
    save_encoded_named_weights,
    save_epoch_boundary_checkpoint,
)
from training.measured_trace import (
    MeasuredCohortBOptimizer,
    MeasuredTraceOptimizer,
)
from training.ibm_om.spec import backfill_weight_modifier_metadata
from training.ibm_reram_hwa import (
    IbmReramHwaConfig,
    IbmReramHwaParameterModifier,
    build_ibm_reram_hwa_modifier,
)
from training.core.engine import (
    EvaluationComponents,
    ExperimentComponents,
    evaluate,
    train_epoch,
)
from training.core.guards import FiniteGradientGuard
from training.core.modifier import ComposedModifier, SplitParameterModifier
from training.program_verify import ProgramVerifyOptimizer

if TYPE_CHECKING:
    from ebl.cli import TrainRequest, ValidateRequest


_ROOT = Path(__file__).resolve().parents[2]
_IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS = frozenset(
    {
        "dual_rail_quad_common_window",
        "differential_pair_common_window",
        "shared_reset_relative_quad",
        "raw_active_p90_quad",
    }
)


def _build_one_weight_modifier(
    stack: StudentStack,
    settings,
    *,
    spec: StudentTrainSpec | StudentValidateSpec,
    device_model_path: Path | None,
    population_artifact_dir: Path | None = None,
    population_role: str | None = None,
):
    if settings.type == "none":
        return None
    if settings.type == "ibm_reram_om_program_verify":
        if device_model_path is None:
            raise ValueError(
                "Expected IBM OM HWA to receive an explicit --device-model. "
                "Provided value: None."
            )
        if (population_artifact_dir is None) != (population_role is None):
            raise ValueError(
                "Expected IBM OM population artifact directory and role together."
            )
        population_path = (
            population_artifact_dir / f"ibm_om_population.{population_role}.npz"
            if population_artifact_dir is not None
            else None
        )
        receipt_path = (
            population_artifact_dir
            / f"ibm_om_population.{population_role}.receipt.json"
            if population_artifact_dir is not None
            else None
        )
        modifier = build_ibm_reram_hwa_modifier(
            stack.bundle.catalog.trainable,
            IbmReramHwaConfig(**dict(settings.parameters)),
            device_model_path=device_model_path,
            conductance_min=spec.model.conductance_min,
            conductance_max=spec.model.conductance_max,
            population_path=population_path,
            population_receipt_path=receipt_path,
        )
        commissioning = modifier.reset_commissioning_bundle
        commissioning_report = modifier.reset_commissioning_report
        if commissioning is not None:
            if population_artifact_dir is None or population_role is None:
                return modifier
            commissioning_path = (
                population_artifact_dir
                / f"ibm_om_reset_commissioning.{population_role}.pt"
            )
            commissioning_receipt_path = (
                population_artifact_dir
                / f"ibm_om_reset_commissioning.{population_role}.receipt.json"
            )
            atomic_torch_save(commissioning, commissioning_path)
            if commissioning_report is None:  # pragma: no cover - defensive
                raise RuntimeError(
                    "Expected RESET commissioning receipt with its bundle."
                )
            atomic_write_json(
                commissioning_receipt_path,
                commissioning_report,
            )
            modifier.register_reset_commissioning_artifacts(
                commissioning_path,
                commissioning_receipt_path,
            )
        return modifier
    if settings.type != "add_normal":  # pragma: no cover - schema rejects it
        raise ValueError(
            "Expected a registered MNIST KD weight modifier. Provided value: "
            f"{settings.type!r}."
        )
    parameters = settings.parameters
    return build_add_normal_modifier(
        stack.bundle.catalog.trainable_parameters,
        AddNormalConfig(
            std_dev=float(parameters["std_dev"]),
            seed=int(parameters["seed"]),
            noisy_evaluation=bool(parameters["noisy_evaluation"]),
            scale_mode=str(parameters["scale_mode"]),
        ),
        run_seed=spec.runtime.seed,
    )


def _build_weight_modifier(
    stack: StudentStack,
    spec: StudentTrainSpec,
    *,
    device_model_path: Path | None = None,
    population_artifact_dir: Path | None = None,
):
    training = _build_one_weight_modifier(
        stack,
        spec.settings.weight_modifier,
        spec=spec,
        device_model_path=device_model_path,
        population_artifact_dir=population_artifact_dir,
        population_role="training" if population_artifact_dir is not None else None,
    )
    selection = _build_one_weight_modifier(
        stack,
        spec.settings.selection_weight_modifier,
        spec=spec,
        device_model_path=device_model_path,
        population_artifact_dir=population_artifact_dir,
        population_role="selection" if population_artifact_dir is not None else None,
    )
    if selection is None:
        return training
    return SplitParameterModifier(training=training, evaluation=selection)


def _evaluation_modifier(modifier):
    return modifier.evaluation if isinstance(modifier, SplitParameterModifier) else modifier


def _training_modifier(modifier):
    return modifier.training if isinstance(modifier, SplitParameterModifier) else modifier


class ForwardGainModifier:
    """Apply an IBM OM modifier's forward logit gain to the KD cost.

    Training and evaluation each take the gain of their own half of a split
    modifier. The previous cost gain is restored when the context exits.
    """

    def __init__(self, cost: Any, modifier) -> None:
        self._cost = cost
        self._training_gain = _forward_logit_gain(_training_modifier(modifier))
        self._evaluation_gain = _forward_logit_gain(_evaluation_modifier(modifier))

    def training_context(self):
        return self._applied(self._training_gain)

    def evaluation_context(self):
        return self._applied(self._evaluation_gain)

    @contextmanager
    def _applied(self, gain: float | None):
        original = float(self._cost.gain)
        try:
            if gain is not None:
                self._cost.gain = float(gain)
            yield self
        finally:
            self._cost.gain = original

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict:
            raise ValueError(
                "Expected the stateless forward-gain modifier state to be empty. "
                f"Provided value: {state_dict!r}."
            )


def _forward_logit_gain(candidate) -> float | None:
    if isinstance(candidate, IbmReramHwaParameterModifier):
        return candidate.config.forward_logit_gain
    return None


def _kd_modifier(stack: StudentStack, modifier) -> ComposedModifier:
    return ComposedModifier(modifier, ForwardGainModifier(stack.cost, modifier))


def _ibm_population_fingerprints(modifier) -> dict[str, str | None]:
    training = _training_modifier(modifier)
    selection = _evaluation_modifier(modifier)
    return {
        "training": (
            training.population_fingerprint
            if isinstance(training, IbmReramHwaParameterModifier)
            else None
        ),
        "selection": (
            selection.population_fingerprint
            if isinstance(selection, IbmReramHwaParameterModifier)
            else None
        ),
    }


def _validate_array_specific_modifier_population_parity(modifier) -> None:
    training = _training_modifier(modifier)
    if not isinstance(training, IbmReramHwaParameterModifier) or (
        training.config.target_mapping
        not in _IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS
    ):
        return
    selection = _evaluation_modifier(modifier)
    if not isinstance(selection, IbmReramHwaParameterModifier):
        raise RuntimeError(
            "Expected array-specific IBM OM HWA to have an IBM OM "
            "selection modifier on the same fixed array."
        )
    if training.population_fingerprint != selection.population_fingerprint:
        raise RuntimeError(
            "Expected array-specific IBM OM training and selection to use "
            "the same fixed population fingerprint. Provided value: "
            f"training={training.population_fingerprint!r}, "
            f"selection={selection.population_fingerprint!r}."
        )


def _ibm_target_mapping_preflights(modifier) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for role, candidate in (
        ("training", _training_modifier(modifier)),
        ("selection", _evaluation_modifier(modifier)),
    ):
        if not isinstance(candidate, IbmReramHwaParameterModifier) or (
            candidate.config.target_mapping
            not in _IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS
        ):
            continue
        _targets, report = candidate.preflight_target_mapping()
        reports[role] = report
    return reports


def _ibm_population_artifact_records(
    store: RunStore,
    modifier,
) -> tuple:
    candidates = (
        (modifier.training, modifier.evaluation)
        if isinstance(modifier, SplitParameterModifier)
        else (modifier,)
    )
    records = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not isinstance(candidate, IbmReramHwaParameterModifier):
            continue
        paths = candidate.population_artifact_paths
        if paths is None or paths[0] in seen:
            continue
        seen.add(paths[0])
        records.extend(
            (
                store.artifact_record(
                    paths[0], kind="ibm_om_array_population"
                ),
                store.artifact_record(
                    paths[1], kind="ibm_om_array_population_receipt"
                ),
            )
        )
        commissioning_paths = candidate.reset_commissioning_artifact_paths
        if commissioning_paths is not None:
            records.extend(
                (
                    store.artifact_record(
                        commissioning_paths[0],
                        kind="ibm_om_reset_commissioning",
                    ),
                    store.artifact_record(
                        commissioning_paths[1],
                        kind="ibm_om_reset_commissioning_receipt",
                    ),
                )
            )
    return tuple(records)


def _wrap_program_verify(
    stack: StudentStack,
    spec: StudentTrainSpec,
) -> StudentStack:
    return replace(
        stack,
        optimizer=ProgramVerifyOptimizer(
            stack.optimizer,
            stack.bundle.catalog,
            parse_device_programming_config(
                spec.settings.update_backend.parameters["device"],
                path=(
                    "config.modes.train.update_backend.parameters.device"
                ),
            ),
        ),
    )


def _amplification_index_report(stack: StudentStack) -> list[dict[str, Any]]:
    """Record model-local stage semantics independently of generated names."""

    report = []
    for interaction in stack.bundle.energy._interactions:
        if not isinstance(
            interaction, (DenseResistive, SignedDenseResistive)
        ):
            continue
        item = {
            "pre_layer": interaction._layer_pre.name,
            "post_layer": interaction._layer_post.name,
            "resolved_pre_index": interaction._logical_pre_index,
            "resolved_post_index": interaction._logical_post_index,
        }
        if isinstance(interaction, SignedDenseResistive):
            item.update(
                {
                    "interaction": "differential_pair",
                    "voltage_amp": interaction._voltage_amp,
                    "current_amp": interaction._current_amp,
                    "forward_gain": interaction._forward_gain(),
                    "post_layer_metric": interaction._post_metric(),
                }
            )
        else:
            logical_pre_index = interaction._logical_pre_index
            item.update(
                {
                    "interaction": "single_conductance",
                    "voltage_amp": interaction._voltage_amp,
                    "current_amp": interaction._current_amp,
                    "forward_gain": (
                        1.0
                        if logical_pre_index == 0
                        else float(interaction._voltage_amp)
                    ),
                    "post_layer_metric": (
                        float(interaction._current_amp)
                        / float(interaction._voltage_amp)
                    )
                    ** logical_pre_index,
                }
            )
        report.append(item)
    return report


def _differential_topology_signature(
    report: Any,
) -> tuple[tuple[int, int, float, float], ...] | None:
    """Project recorded topology onto its model-local numerical semantics."""

    if not isinstance(report, (list, tuple)):
        return None
    try:
        return tuple(
            (
                int(item["resolved_pre_index"]),
                int(item["resolved_post_index"]),
                float(item["forward_gain"]),
                float(item["post_layer_metric"]),
            )
            for item in report
            if item.get("interaction") == "differential_pair"
        )
    except (KeyError, TypeError, ValueError):
        return None


def _single_topology_signature(
    report: Any,
    *,
    voltage_amp: float,
    current_amp: float,
) -> tuple[tuple[int, int, float, float], ...] | None:
    """Project a one-conductance stack onto model-local stage semantics.

    Older named checkpoints recorded only the explicit logical indices. The
    single-conductance schema fixes the amplifier magnitudes, so missing stage
    scales can be reconstructed without consulting generated layer names.
    New checkpoints record the resolved scales directly and are checked
    exactly.
    """

    if not isinstance(report, (list, tuple)):
        return None
    try:
        signature = []
        for item in report:
            interaction = item.get("interaction")
            if interaction not in {None, "single_conductance"}:
                return None
            pre_index = int(item["resolved_pre_index"])
            post_index = int(item["resolved_post_index"])
            forward_gain = float(
                item.get("forward_gain", 1.0 if pre_index == 0 else voltage_amp)
            )
            post_metric = float(
                item.get(
                    "post_layer_metric",
                    (current_amp / voltage_amp) ** pre_index,
                )
            )
            signature.append(
                (pre_index, post_index, forward_gain, post_metric)
            )
        return tuple(signature)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _weight_modifier_metadata(settings: Any) -> dict[str, Any]:
    """Return a mutable JSON representation at the artifact boundary."""

    return {
        "type": settings.type,
        "parameters": to_plain_data(settings.parameters),
    }


def _model_checkpoint_metadata(
    stack: StudentStack,
    *,
    spec: StudentTrainSpec,
    teacher_sha256: str,
    mapping: dict[str, Any] | None,
    deployment_source: dict[str, Any] | None = None,
    device_model_sha256: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "encoding": spec.model.encoding,
        "include_biases": spec.model.include_biases,
        "amplification_indexing": spec.model.amplification_indexing,
        "amplification_indices": _amplification_index_report(stack),
        "objective": "teacher_kl",
        "initialization": "teacher_mapped",
        "teacher_type": spec.teacher.type,
        "teacher_initialization": spec.teacher.initialization,
        "fixed_logit_gain": stack.cost.gain,
        "temperature": spec.settings.temperature,
        "selection_evaluation": spec.settings.selection_evaluation,
        "selection_metric": spec.settings.selection_metric,
        "selection_noise_repeats": spec.settings.selection_noise_repeats,
        "conductance_bounds_s": [
            spec.model.conductance_min,
            spec.model.conductance_max,
        ],
        "mapping_range_placement": spec.mapping.range_placement,
        "mapping_scale_fraction_pairs": (
            None
            if spec.mapping.scale_fraction_pairs is None
            else [list(pair) for pair in spec.mapping.scale_fraction_pairs]
        ),
        "teacher_sha256": teacher_sha256,
        "mapping": mapping,
        "weight_modifier": _weight_modifier_metadata(
            spec.settings.weight_modifier
        ),
        "update_backend": spec.settings.update_backend.type,
        "deployment_source": deployment_source,
    }
    if spec.settings.selection_weight_modifier.type != "none":
        metadata["selection_weight_modifier"] = _weight_modifier_metadata(
            spec.settings.selection_weight_modifier
        )
    if device_model_sha256 is not None:
        metadata["device_model_sha256"] = device_model_sha256
    if isinstance(stack.optimizer, MeasuredTraceOptimizer):
        data_report = stack.optimizer.data_report
        metadata.update(
            {
                "initial_target_mapping": spec.settings.update_backend.parameters[
                    "initial_target_mapping"
                ],
                "dual_rail_layout_by_parameter": (
                    None
                    if spec.settings.update_backend.parameters.get(
                        "dual_rail_layout_by_parameter"
                    )
                    is None
                    else dict(
                        spec.settings.update_backend.parameters[
                            "dual_rail_layout_by_parameter"
                        ]
                    )
                ),
                "positive_gradient_threshold_by_parameter": (
                    data_report.get(
                        "positive_gradient_threshold_by_parameter"
                    )
                ),
                "device_data_sha256": data_report["source_sha256"],
                "device_assignment_sha256_by_parameter": data_report[
                    "assignment_sha256_by_parameter"
                ],
            }
        )
    elif isinstance(stack.optimizer, ProgramVerifyOptimizer):
        metadata.update(
            {
                "initial_target_mapping": (
                    "teacher_mapping_then_endpoint_program_verify"
                ),
                "device_programming": stack.optimizer.configuration,
            }
        )
    else:
        metadata["initial_target_mapping"] = (
            "literal_named_weight_copy"
            if spec.teacher.type == "bounded_drn"
            else "ideal_teacher_mapping"
        )
    return metadata


def _validate_checkpoint_metadata(
    metadata: Any,
    *,
    spec: StudentValidateSpec | StudentTrainSpec,
    teacher_sha256: str,
    expected_amplification_indices: Any,
) -> None:
    provided = metadata if isinstance(metadata, dict) else {}
    expected = {
        "experiment_id": spec.experiment_id,
        "encoding": spec.model.encoding,
        "include_biases": spec.model.include_biases,
        "amplification_indexing": spec.model.amplification_indexing,
        "objective": "teacher_kl",
        "initialization": "teacher_mapped",
        "temperature": 1.0,
        "teacher_sha256": teacher_sha256,
    }
    if spec.teacher.type == "bounded_drn":
        expected.update(
            {
                "teacher_type": "bounded_drn",
                "teacher_initialization": "literal_named_weight_copy",
                "conductance_bounds_s": [
                    spec.model.conductance_min,
                    spec.model.conductance_max,
                ],
                "mapping_range_placement": spec.mapping.range_placement,
                "mapping_scale_fraction_pairs": None,
            }
        )
    if (
        spec.mapping.range_placement != "lower"
        or spec.mapping.scale_fraction_pairs is not None
    ):
        expected.update(
            {
                "conductance_bounds_s": [
                    spec.model.conductance_min,
                    spec.model.conductance_max,
                ],
                "mapping_range_placement": spec.mapping.range_placement,
                "mapping_scale_fraction_pairs": (
                    None
                    if spec.mapping.scale_fraction_pairs is None
                    else [
                        list(pair)
                        for pair in spec.mapping.scale_fraction_pairs
                    ]
                ),
            }
        )
    train_settings = (
        spec.settings
        if isinstance(spec, StudentTrainSpec)
        else None
    )
    if train_settings is not None:
        backend = train_settings.update_backend.type
        initial_target_mapping = train_settings.update_backend.parameters.get(
            "initial_target_mapping"
        )
        provenance_required = (
            backend == "measured_cohort_a_one_pulse_down"
            or (
                backend in MEASURED_COHORT_A_BACKENDS
                and initial_target_mapping
                in {
                    "dual_rail_pairwise_common_window",
                    "dual_rail_quad_common_window",
                }
            )
        )
        if provenance_required:
            expected["initial_target_mapping"] = initial_target_mapping
        if provenance_required and initial_target_mapping in {
            "dual_rail_pairwise_common_window",
            "dual_rail_quad_common_window",
        }:
            expected["dual_rail_layout_by_parameter"] = dict(
                train_settings.update_backend.parameters[
                    "dual_rail_layout_by_parameter"
                ]
            )
    if (
        train_settings is not None
        and train_settings.update_backend.type
        == "measured_cohort_a_one_pulse_down"
        and train_settings.update_backend.parameters.get(
            "positive_gradient_threshold_by_parameter"
        )
        is not None
    ):
        expected["positive_gradient_threshold_by_parameter"] = dict(
            train_settings.update_backend.parameters[
                "positive_gradient_threshold_by_parameter"
            ]
        )
    mismatches = {
        name: {"expected": value, "provided": provided.get(name)}
        for name, value in expected.items()
        if provided.get(name) != value
    }
    if spec.model.encoding == "differential":
        expected_signature = _differential_topology_signature(
            expected_amplification_indices
        )
        provided_signature = _differential_topology_signature(
            provided.get("amplification_indices")
        )
        if provided_signature != expected_signature:
            mismatches["amplification_indices"] = {
                "expected": expected_signature,
                "provided": provided_signature,
            }
    else:
        expected_signature = _single_topology_signature(
            expected_amplification_indices,
            voltage_amp=spec.model.voltage_amp,
            current_amp=spec.model.current_amp,
        )
        provided_signature = _single_topology_signature(
            provided.get("amplification_indices"),
            voltage_amp=spec.model.voltage_amp,
            current_amp=spec.model.current_amp,
        )
        if provided_signature != expected_signature:
            mismatches["amplification_indices"] = {
                "expected": expected_signature,
                "provided": provided_signature,
            }
    if mismatches:
        raise ValueError(
            "Expected KD checkpoint metadata to match the requested model, "
            f"topology, and teacher. Provided value: {mismatches!r}."
        )


def _validate_cohort_b_source_metadata(
    metadata: Any,
    *,
    spec: StudentTrainSpec,
    device_data_sha256: str,
) -> None:
    """Require cohort-B deployment to originate from the matched A protocol."""

    provided = metadata if isinstance(metadata, dict) else {}
    destination_mapping = spec.settings.update_backend.parameters[
        "initial_target_mapping"
    ]
    source_protocols = {
        ("differential", "literal"): "paired_affine_common_window",
        (
            "differential",
            "paired_affine_common_window",
        ): "paired_affine_common_window",
        (
            "single",
            "dual_rail_quad_common_window",
        ): "dual_rail_quad_common_window",
    }
    source_mapping = source_protocols.get(
        (spec.model.encoding, destination_mapping)
    )
    if source_mapping is None:
        raise ValueError(
            "Expected cohort-B deployment to select a registered source "
            "protocol for its encoding and target mapping. Provided value: "
            f"encoding={spec.model.encoding!r}, "
            f"initial_target_mapping={destination_mapping!r}."
        )
    source_layouts = (
        dict(
            spec.settings.update_backend.parameters[
                "dual_rail_layout_by_parameter"
            ]
        )
        if spec.model.encoding == "single"
        else None
    )
    expected = {
        "encoding": spec.model.encoding,
        "initialization": "teacher_mapped",
        "update_backend": "measured_cohort_a",
        "initial_target_mapping": source_mapping,
        "device_data_sha256": device_data_sha256,
    }
    if source_layouts is not None:
        expected["dual_rail_layout_by_parameter"] = source_layouts
    mismatches = {
        name: {"expected": value, "provided": provided.get(name)}
        for name, value in expected.items()
        if provided.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "Expected cohort-B deployment weights to come from the matched "
            "cohort-A common-window protocol. Provided value: "
            f"{mismatches!r}."
        )


def _validate_resume_backend_metadata(
    metadata: Any,
    *,
    spec: StudentTrainSpec,
    device_data_sha256: str | None,
    device_model_sha256: str | None,
) -> None:
    provided = metadata if isinstance(metadata, dict) else {}
    backend = spec.settings.update_backend.type
    expected = {"update_backend": backend}
    if backend in MEASURED_BACKENDS:
        expected.update(
            {
                "initial_target_mapping": spec.settings.update_backend.parameters[
                    "initial_target_mapping"
                ],
                "device_data_sha256": device_data_sha256,
            }
        )
        if (
            spec.settings.update_backend.parameters["initial_target_mapping"]
            in {
                "dual_rail_pairwise_common_window",
                "dual_rail_quad_common_window",
            }
        ):
            expected["dual_rail_layout_by_parameter"] = dict(
                spec.settings.update_backend.parameters[
                    "dual_rail_layout_by_parameter"
                ]
            )
        if (
            backend == "measured_cohort_a_one_pulse_down"
            and spec.settings.update_backend.parameters.get(
                "positive_gradient_threshold_by_parameter"
            )
            is not None
        ):
            expected["positive_gradient_threshold_by_parameter"] = dict(
                spec.settings.update_backend.parameters[
                    "positive_gradient_threshold_by_parameter"
                ]
            )
    elif backend == "program_verify":
        expected.update(
            {
                "initial_target_mapping": (
                    "teacher_mapping_then_endpoint_program_verify"
                ),
                "device_programming": dict(
                    spec.settings.update_backend.parameters["device"]
                ),
            }
        )
    expected["weight_modifier"] = _weight_modifier_metadata(
        spec.settings.weight_modifier
    )
    if spec.settings.selection_weight_modifier.type != "none":
        expected["selection_weight_modifier"] = _weight_modifier_metadata(
            spec.settings.selection_weight_modifier
        )
    if device_model_sha256 is not None:
        expected["device_model_sha256"] = device_model_sha256
    if spec.teacher.type == "bounded_drn":
        expected.update(
            {
                "selection_evaluation": spec.settings.selection_evaluation,
                "selection_noise_repeats": (
                    spec.settings.selection_noise_repeats
                ),
            }
        )
    if spec.settings.selection_metric != "kl_teacher_student":
        expected.update(
            {
                "selection_evaluation": spec.settings.selection_evaluation,
                "selection_metric": spec.settings.selection_metric,
                "selection_noise_repeats": (
                    spec.settings.selection_noise_repeats
                ),
            }
        )
    if (
        spec.mapping.range_placement != "lower"
        or spec.mapping.scale_fraction_pairs is not None
    ):
        expected.update(
            {
                "conductance_bounds_s": [
                    spec.model.conductance_min,
                    spec.model.conductance_max,
                ],
                "mapping_range_placement": spec.mapping.range_placement,
                "mapping_scale_fraction_pairs": (
                    None
                    if spec.mapping.scale_fraction_pairs is None
                    else [
                        list(pair)
                        for pair in spec.mapping.scale_fraction_pairs
                    ]
                ),
            }
        )
    provided = {
        name: (
            backfill_weight_modifier_metadata(value)
            if name in {"weight_modifier", "selection_weight_modifier"}
            else value
        )
        for name, value in provided.items()
    }
    mismatches = {
        name: {"expected": value, "provided": provided.get(name)}
        for name, value in expected.items()
        if provided.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "Expected resume checkpoint backend provenance to match the "
            f"requested training protocol. Provided value: {mismatches!r}."
        )


def _input(role: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected --{role.replace('_', '-')} to name an existing file. "
            f"Provided value: {str(resolved)!r}."
        )
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def _validate_train_request(request: Any, spec: StudentTrainSpec) -> None:
    if request.teacher_weights is None:
        raise ValueError(
            "Expected mnist_relu_drn_kd.v1 training to receive "
            "--teacher-weights. Provided value: None."
        )
    if request.base_weights is not None:
        raise ValueError(
            "Expected mnist_relu_drn_kd.v1 not to use --base-weights. "
            f"Provided value: {str(request.base_weights)!r}."
        )
    selected = [request.weights is not None, request.resume is not None]
    if sum(selected) > 1:
        raise ValueError(
            "Expected at most one of --weights or --resume. Provided value: both."
        )
    backend = spec.settings.update_backend.type
    deployed_recovery = backend == IBM_OM_DEPLOYED_RECOVERY_BACKEND
    requested_deployment = getattr(request, "deployment", None)
    if deployed_recovery:
        if sum(selected) != 1:
            raise ValueError(
                "Expected deployed recovery to receive exactly one of "
                "--weights or --resume."
            )
        if requested_deployment is None:
            raise ValueError(
                "Expected deployed recovery to receive an exact --deployment."
            )
    elif requested_deployment is not None:
        raise ValueError(
            "Expected --deployment only for ibm_om_deployed_recovery. "
            f"Provided value: {requested_deployment!r}."
        )
    measured = backend in MEASURED_BACKENDS
    if measured != (request.device_data is not None):
        expected = "an explicit --device-data" if measured else "no --device-data"
        raise ValueError(
            f"Expected update backend {spec.settings.update_backend.type!r} to use {expected}. "
            f"Provided value: {request.device_data!r}."
        )
    modifiers = (
        spec.settings.weight_modifier,
        spec.settings.selection_weight_modifier,
    )
    uses_ibm_device_model = deployed_recovery or any(
        modifier.type == "ibm_reram_om_program_verify"
        for modifier in modifiers
    )
    requested_device_model = getattr(request, "device_model", None)
    if uses_ibm_device_model != (requested_device_model is not None):
        expected = (
            "an explicit --device-model"
            if uses_ibm_device_model
            else "no --device-model"
        )
        raise ValueError(
            "Expected IBM OM weight modifiers to use "
            f"{expected}. Provided value: {requested_device_model!r}."
        )
    if (
        backend in MEASURED_COHORT_B_BACKENDS
        and request.weights is None
        and request.resume is None
    ):
        raise ValueError(
            "Expected measured_cohort_b training to receive cohort-A "
            "--weights or an exact --resume checkpoint. Provided value: "
            "weights=None, resume=None."
        )


def _validate_bounded_drn_teacher_metadata(
    metadata: Any,
    *,
    spec: StudentTrainSpec | StudentValidateSpec,
    stack: StudentStack,
) -> None:
    provided = metadata if isinstance(metadata, dict) else {}
    model = provided.get("model")
    model = model if isinstance(model, dict) else {}
    solver = provided.get("solver")
    solver = solver if isinstance(solver, dict) else {}
    dataset = provided.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    expected = {
        "architecture": "deep_resistive_network",
        "checkpoint_role": "supervised_drn",
        "objective": "supervised_paired_squared_error",
        "output_semantics": "adjacent_pair_difference",
        "model.dims": list(spec.model.dims),
        "model.logical_input_dim": 784,
        "model.num_classes": 10,
        "model.input_gain": spec.model.input_gain,
        "model.weight_bounds": [
            spec.model.conductance_min,
            spec.model.conductance_max,
        ],
        "model.include_biases": spec.model.include_biases,
        "model.voltage_amp": spec.model.voltage_amp,
        "model.current_amp": spec.model.current_amp,
        "model.adapter": {"type": "none", "parameters": {}},
        "model.amplification_indexing": "model_local",
        "solver.inference_iterations": spec.solver.inference_iterations,
        "solver.training_iterations": spec.solver.training_iterations,
        "solver.minimizer_mode": spec.solver.mode,
        "dataset.name": "mnist",
        "dataset.validation_points": spec.data.validation_points,
        "dataset.data_seed": spec.runtime.data_seed,
    }
    actual = {
        "architecture": provided.get("architecture"),
        "checkpoint_role": provided.get("checkpoint_role"),
        "objective": provided.get("objective"),
        "output_semantics": provided.get("output_semantics"),
        "model.dims": model.get("dims"),
        "model.logical_input_dim": model.get("logical_input_dim"),
        "model.num_classes": model.get("num_classes"),
        "model.input_gain": model.get("input_gain"),
        "model.weight_bounds": model.get("weight_bounds"),
        "model.include_biases": model.get("include_biases"),
        "model.voltage_amp": model.get("voltage_amp"),
        "model.current_amp": model.get("current_amp"),
        "model.adapter": model.get("adapter"),
        "model.amplification_indexing": model.get(
            "amplification_indexing"
        ),
        "solver.inference_iterations": solver.get("inference_iterations"),
        "solver.training_iterations": solver.get("training_iterations"),
        "solver.minimizer_mode": solver.get("minimizer_mode"),
        "dataset.name": dataset.get("name"),
        "dataset.validation_points": dataset.get("validation_points"),
        "dataset.data_seed": dataset.get("data_seed"),
    }
    mismatches = {
        key: {"expected": value, "provided": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    expected_topology = _single_topology_signature(
        _amplification_index_report(stack),
        voltage_amp=spec.model.voltage_amp,
        current_amp=spec.model.current_amp,
    )
    provided_topology = _single_topology_signature(
        model.get("amplification_indices"),
        voltage_amp=spec.model.voltage_amp,
        current_amp=spec.model.current_amp,
    )
    if provided_topology != expected_topology:
        mismatches["model.amplification_indices"] = {
            "expected": expected_topology,
            "provided": provided_topology,
        }
    overrelaxation = solver.get("overrelaxation")
    provided_factor = (
        overrelaxation.get("factor")
        if isinstance(overrelaxation, dict)
        else None
    )
    if provided_factor != spec.solver.overrelaxation_factor:
        mismatches["solver.overrelaxation.factor"] = {
            "expected": spec.solver.overrelaxation_factor,
            "provided": provided_factor,
        }
    if mismatches:
        raise ValueError(
            "Expected bounded DRN teacher metadata to match the requested "
            "student topology, solver, bounds, and MNIST split. Provided "
            f"value: {mismatches!r}."
        )


def _load_teacher(
    path: Path,
    *,
    device: torch.device,
    spec: StudentTrainSpec | StudentValidateSpec,
) -> tuple[BiasFreeReluTeacher | BoundedDrnTeacher, dict[str, Any]]:
    if spec.teacher.type == "bias_free_relu":
        teacher = BiasFreeReluTeacher(device=device)
        loaded = load_named_weights(path, teacher.catalog)
        architecture = loaded.metadata.get("architecture")
        if architecture != "bias_free_relu_784_50_10":
            raise ValueError(
                "Expected --teacher-weights metadata architecture to equal "
                "'bias_free_relu_784_50_10'. "
                f"Provided value: {architecture!r}."
            )
        return teacher, loaded.metadata

    stack = build_student_stack(spec, enable_measured=False)
    loaded = load_named_weights(path, stack.bundle.catalog)
    _validate_bounded_drn_teacher_metadata(
        loaded.metadata,
        spec=spec,
        stack=stack,
    )
    return BoundedDrnTeacher(
        stack=stack,
        checkpoint_metadata=dict(loaded.metadata),
    ), loaded.metadata


def _literal_bounded_drn_initialization(
    stack: StudentStack,
    teacher: BoundedDrnTeacher,
    loader: Iterable,
    spec: StudentTrainSpec,
) -> dict[str, Any]:
    """Copy stable teacher tensors and prove gain-one logit parity."""

    teacher.copy_named_weights_to(stack.bundle.catalog)
    raw_scores, teacher_logits = collect_calibration(
        stack,
        teacher,
        loader,
    )
    difference = raw_scores.to(torch.float64) - teacher_logits.to(
        torch.float64
    )
    max_abs_difference = float(difference.abs().max().item())
    if max_abs_difference > 1e-6:
        raise RuntimeError(
            "Expected literal bounded-DRN initialization to reproduce "
            "teacher logits within 1e-6 before training. Provided value: "
            f"max_abs_logit_difference={max_abs_difference!r}."
        )
    teacher_log_prob = F.log_softmax(teacher_logits.to(torch.float64), dim=1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(raw_scores.to(torch.float64), dim=1)
    kl = float(
        (
            teacher_prob
            * (teacher_log_prob - student_log_prob)
        )
        .sum(dim=1)
        .mean()
        .item()
    )
    calibration = {
        "gain": 1.0,
        "raw_kl": kl,
        "calibrated_kl": kl,
        "score_rms": float(
            torch.sqrt(raw_scores.to(torch.float64).square().mean()).item()
        ),
        "teacher_logit_rms": float(
            torch.sqrt(
                teacher_logits.to(torch.float64).square().mean()
            ).item()
        ),
        "gain_at_grid_boundary": False,
        "max_abs_logit_difference": max_abs_difference,
        "mean_abs_logit_difference": float(
            difference.abs().mean().item()
        ),
    }
    selected = {
        "scale_fractions": [1.0, 1.0],
        "mapping": {
            "encoding": "single",
            "strategy": "literal_named_weight_copy",
            "conductance_bounds": [
                spec.model.conductance_min,
                spec.model.conductance_max,
            ],
            "floor_subtracted": False,
            "additive_remapping": False,
        },
        "calibration": calibration,
    }
    return {
        "search_examples": spec.mapping.calibration_examples,
        "selection_domain": "literal_named_weight_copy",
        "candidates": [selected],
        "selected_index": 0,
        "selected": selected,
    }


class KdEvaluationProbe:
    """Teacher-student metrics over one KD evaluation pass at the applied gain."""

    name = "kd_evaluation"

    def reset(self) -> None:
        self._totals = {
            "kl": 0.0,
            "raw_kl": 0.0,
            "student_correct": 0,
            "teacher_correct": 0,
            "agreement": 0,
            "raw_score_squared": 0.0,
            "calibrated_score_squared": 0.0,
            "teacher_score_squared": 0.0,
        }
        self._examples = 0
        self._applied_gain: float | None = None

    def observe(self, event: Any) -> None:
        cost = event.components.cost_fn
        teacher_logits = event.batch.targets.logits
        labels = event.batch.targets.labels
        totals = self._totals
        self._applied_gain = float(cost.gain)
        raw_scores = cost.student_logits() / cost.gain
        student_logits = raw_scores * cost.gain
        teacher_log_prob = F.log_softmax(teacher_logits, dim=1)
        teacher_prob = teacher_log_prob.exp()
        student_log_prob = F.log_softmax(student_logits, dim=1)
        raw_student_log_prob = F.log_softmax(raw_scores, dim=1)
        totals["kl"] += float(
            (teacher_prob * (teacher_log_prob - student_log_prob)).sum().item()
        )
        totals["raw_kl"] += float(
            (teacher_prob * (teacher_log_prob - raw_student_log_prob)).sum().item()
        )
        student_prediction = student_logits.argmax(dim=1)
        teacher_prediction = teacher_logits.argmax(dim=1)
        totals["student_correct"] += int(student_prediction.eq(labels).sum().item())
        totals["teacher_correct"] += int(teacher_prediction.eq(labels).sum().item())
        totals["agreement"] += int(student_prediction.eq(teacher_prediction).sum().item())
        totals["raw_score_squared"] += float(raw_scores.square().sum().item())
        totals["calibrated_score_squared"] += float(student_logits.square().sum().item())
        totals["teacher_score_squared"] += float(teacher_logits.square().sum().item())
        self._examples += int(labels.shape[0])

    def result(self) -> dict[str, Any]:
        examples = self._examples
        totals = self._totals
        if examples == 0:
            raise ValueError("Expected KD evaluation to process examples. Provided value: 0.")
        score_values = examples * 10
        return {
            "examples": examples,
            "kl_teacher_student": totals["kl"] / examples,
            "raw_kl_teacher_student": totals["raw_kl"] / examples,
            "student_accuracy": totals["student_correct"] / examples,
            "teacher_accuracy": totals["teacher_correct"] / examples,
            "teacher_agreement": totals["agreement"] / examples,
            "raw_score_rms": (totals["raw_score_squared"] / score_values) ** 0.5,
            "calibrated_score_rms": (totals["calibrated_score_squared"] / score_values) ** 0.5,
            "teacher_logit_rms": (totals["teacher_score_squared"] / score_values) ** 0.5,
            "fixed_logit_gain": self._applied_gain,
        }


class TeacherAgreementProbe:
    """KL, accuracy and teacher agreement at each training free phase."""

    name = "teacher_agreement"

    def reset(self) -> None:
        self._kl_sum = 0.0
        self._student_correct = 0
        self._agreement = 0
        self._examples = 0

    def observe(self, event: Any) -> None:
        cost = event.components.cost_fn
        teacher_logits = event.batch.targets.logits
        labels = event.batch.targets.labels
        with torch.no_grad():
            batch_kl = cost.eval()
            student_prediction = cost.student_logits().argmax(dim=1)
            teacher_prediction = teacher_logits.argmax(dim=1)
            self._kl_sum += float(batch_kl.sum().item())
            self._student_correct += int(student_prediction.eq(labels).sum().item())
            self._agreement += int(student_prediction.eq(teacher_prediction).sum().item())
        self._examples += int(labels.shape[0])

    def result(self) -> dict[str, Any]:
        examples = self._examples
        if examples == 0:
            raise ValueError("Expected KD training to process examples. Provided value: 0.")
        return {
            "examples": examples,
            "kl_teacher_student": self._kl_sum / examples,
            "student_accuracy": self._student_correct / examples,
            "teacher_agreement": self._agreement / examples,
        }


def _evaluate(
    stack: StudentStack,
    teacher: BiasFreeReluTeacher | BoundedDrnTeacher,
    loader: Iterable,
    *,
    maximum_batches: int | None = None,
    sample_limit: int | None = None,
    modifier=None,
) -> dict[str, Any]:
    probe = KdEvaluationProbe()
    with torch.no_grad():
        result = evaluate(
            EvaluationComponents(stack.network, stack.cost, stack.minimizer),
            teacher_batches(
                loader,
                teacher,
                stack.device,
                maximum_batches=maximum_batches,
                sample_limit=sample_limit,
            ),
            modifier=_kd_modifier(stack, modifier),
            probes=(probe,),
        )
    return result.probe_value(probe.name)


def _selection_evaluate(
    stack: StudentStack,
    teacher: BiasFreeReluTeacher | BoundedDrnTeacher,
    loader: Iterable,
    *,
    spec: StudentTrainSpec,
    modifier,
    clean: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if spec.settings.selection_evaluation == "clean":
        return (
            clean
            if clean is not None
            else _evaluate(
                stack,
                teacher,
                loader,
                maximum_batches=spec.settings.max_validation_batches,
            )
        )
    if modifier is None:
        raise RuntimeError(
            "Expected modifier-based selection to have a parameter "
            "modifier. Provided value: None."
        )
    snapshot = modifier.state_dict()
    reports = []
    programming_reports = []
    try:
        for _repeat in range(spec.settings.selection_noise_repeats):
            reports.append(
                _evaluate(
                    stack,
                    teacher,
                    loader,
                    maximum_batches=(
                        spec.settings.max_validation_batches
                    ),
                    modifier=modifier,
                )
            )
            evaluation = _evaluation_modifier(modifier)
            if isinstance(evaluation, IbmReramHwaParameterModifier):
                programming_reports.append(evaluation.programming_report)
    finally:
        modifier.load_state_dict(snapshot)
    scalar_keys = (
        "kl_teacher_student",
        "raw_kl_teacher_student",
        "student_accuracy",
        "teacher_accuracy",
        "teacher_agreement",
        "raw_score_rms",
        "calibrated_score_rms",
        "teacher_logit_rms",
    )
    result = {
        "examples": reports[0]["examples"],
        "fixed_logit_gain": reports[0]["fixed_logit_gain"],
        "selection_evaluation": "modifier_fixed_sequence_average",
        "selection_noise_repeats": len(reports),
        "repeat_kl_teacher_student": [
            report["kl_teacher_student"] for report in reports
        ],
        "repeat_student_accuracy": [
            report["student_accuracy"] for report in reports
        ],
        "repeat_teacher_agreement": [
            report["teacher_agreement"] for report in reports
        ],
    }
    if programming_reports:
        result["repeat_device_programming"] = programming_reports
    result.update(
        {
            key: sum(float(report[key]) for report in reports) / len(reports)
            for key in scalar_keys
        }
    )
    return result


def _train_epoch(
    stack: StudentStack,
    teacher: BiasFreeReluTeacher | BoundedDrnTeacher,
    loader: Iterable,
    *,
    maximum_batches: int | None,
    modifier=None,
) -> tuple[dict[str, Any], int]:
    probe = TeacherAgreementProbe()
    result = train_epoch(
        ExperimentComponents(
            stack.network,
            stack.cost,
            stack.minimizer,
            tuple(stack.bundle.energy.params()),
            stack.differentiator,
            stack.optimizer,
        ),
        teacher_batches(loader, teacher, stack.device, maximum_batches=maximum_batches),
        modifier=_kd_modifier(stack, modifier),
        probes=(probe,),
        event_handlers=(
            FiniteGradientGuard(
                tuple(binding.key for binding in stack.bundle.catalog.trainable),
                label="KD",
            ),
        ),
        reset_input=True,
    )
    return result.probe_value(probe.name), result.batch_count


def _selected_payload(
    stack: StudentStack,
    *,
    spec: StudentTrainSpec,
    teacher_path: Path,
    teacher_sha256: str,
    mapping: dict[str, Any] | None,
    deployment_source: dict[str, Any] | None,
    device_model_sha256: str | None,
    epoch: int,
    validation: dict[str, Any],
) -> dict[str, Any]:
    metadata = _model_checkpoint_metadata(
        stack,
        spec=spec,
        teacher_sha256=teacher_sha256,
        mapping=mapping,
        deployment_source=deployment_source,
        device_model_sha256=device_model_sha256,
    )
    metadata.update(
        {
            "teacher_path": str(teacher_path.expanduser().resolve()),
            "selection_metric": (
                "validation_modifier_mean."
                f"{spec.settings.selection_metric}"
                if spec.settings.selection_evaluation == "modifier"
                else "validation_clean."
                f"{spec.settings.selection_metric}"
            ),
            "selection_value": validation[spec.settings.selection_metric],
            "selection_epoch": epoch,
            "selection_student_accuracy": validation["student_accuracy"],
            "selection_teacher_agreement": validation["teacher_agreement"],
        }
    )
    return encode_named_weights(
        stack.bundle.catalog,
        metadata=metadata,
    )


def _selection_improved(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    metric: str,
) -> bool:
    candidate_value = float(candidate[metric])
    incumbent_value = float(incumbent[metric])
    if metric == "kl_teacher_student":
        return candidate_value < incumbent_value
    if metric == "student_accuracy":
        return candidate_value > incumbent_value
    raise ValueError(f"Unsupported selection metric: {metric!r}.")


def _adaptation_summary(
    baseline: dict[str, Any],
    selected: dict[str, Any],
    last: dict[str, Any],
    *,
    selected_epoch: int,
) -> dict[str, Any]:
    baseline_kl = float(baseline["kl_teacher_student"])

    def comparison(candidate: dict[str, Any]) -> dict[str, Any]:
        kl_reduction = baseline_kl - float(candidate["kl_teacher_student"])
        return {
            "student_accuracy_change": (
                float(candidate["student_accuracy"])
                - float(baseline["student_accuracy"])
            ),
            "teacher_agreement_change": (
                float(candidate["teacher_agreement"])
                - float(baseline["teacher_agreement"])
            ),
            "kl_reduction": kl_reduction,
            "relative_kl_reduction": (
                kl_reduction / baseline_kl if baseline_kl > 0.0 else 0.0
            ),
        }

    return {
        "baseline": "post_deployment_calibrated_validation",
        "selected_epoch": selected_epoch,
        "selected": comparison(selected),
        "after_ten_epochs": comparison(last),
    }


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, StudentTrainSpec):
        raise TypeError(
            "Expected mnist_relu_drn_kd.v1 train to resolve StudentTrainSpec. "
            f"Provided value: {type(spec).__name__}."
        )
    _validate_train_request(request, spec)
    if spec.settings.update_backend.type == IBM_OM_DEPLOYED_RECOVERY_BACKEND:
        from experiments.mnist_relu_drn.ibm_om_deployed_recovery import (
            run_recovery_train,
        )

        return run_recovery_train(request)
    input_artifacts = [_input("teacher_weights", request.teacher_weights)]
    for role in ("weights", "resume", "device_data", "device_model"):
        value = getattr(request, role)
        if value is not None:
            input_artifacts.append(_input(role, value))
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=input_artifacts,
    )
    try:
        torch.manual_seed(spec.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(spec.runtime.seed)
        device = torch.device(spec.runtime.device)
        data = build_mnist_loaders(
            spec.data,
            data_seed=spec.runtime.data_seed,
            calibration_examples=spec.mapping.calibration_examples,
            calibration_batch_size=spec.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            request.teacher_weights,
            device=device,
            spec=spec,
        )
        teacher_sha = sha256_file(request.teacher_weights)
        backend = spec.settings.update_backend.type
        measured = backend in MEASURED_BACKENDS
        cohort_b = backend in MEASURED_COHORT_B_BACKENDS
        program_verify = backend == "program_verify"
        measured_optimizer = (
            measured_optimizer_type(backend) if measured else None
        )
        device_data_sha = (
            sha256_file(request.device_data)
            if request.device_data is not None
            else None
        )
        device_model_sha = (
            sha256_file(request.device_model)
            if request.device_model is not None
            else None
        )
        stack = build_student_stack(
            spec,
            device_data_path=request.device_data,
            enable_measured=request.resume is not None,
        )
        modifier = _build_weight_modifier(
            stack,
            spec,
            device_model_path=request.device_model,
            population_artifact_dir=store.run_dir / "artifacts",
        )
        _validate_array_specific_modifier_population_parity(modifier)
        ibm_population_fingerprints = _ibm_population_fingerprints(modifier)
        ibm_target_mapping_preflights: dict[str, dict[str, Any]] = {}
        weights_path = store.run_dir / "checkpoints" / "weights.pt"
        resume_path = store.run_dir / "checkpoints" / "resume.pt"
        deployment_bundle_path: Path | None = None
        training_programming_report_path: Path | None = None
        start_epoch = 0
        global_step = 0
        mapping_report: dict[str, Any] | None = None
        programming_report = None
        selected_weights = None
        selected_epoch = -1
        selected_validation = None
        deployment_source: dict[str, Any] | None = None
        deployment_metrics: dict[str, Any] | None = None
        initial: dict[str, Any] | None = None
        initial_clean_validation: dict[str, Any] | None = None
        initial_conductances: dict[str, Any] | None = None
        resumed_last_train: dict[str, Any] | None = None
        resumed_last_validation: dict[str, Any] | None = None
        resumed_last_clean_validation: dict[str, Any] | None = None
        resumed_run = request.resume is not None

        if request.resume is not None:
            resumed = load_epoch_boundary_checkpoint(
                request.resume,
                catalog=stack.bundle.catalog,
                optimizer=stack.optimizer,
                modifier=modifier,
                dataloader_generators={"train": data.train_generator},
            )
            _validate_checkpoint_metadata(
                resumed.metadata,
                spec=spec,
                teacher_sha256=teacher_sha,
                expected_amplification_indices=(
                    _amplification_index_report(stack)
                ),
            )
            _validate_resume_backend_metadata(
                resumed.metadata,
                spec=spec,
                device_data_sha256=device_data_sha,
                device_model_sha256=device_model_sha,
            )
            start_epoch = resumed.epoch
            global_step = resumed.global_step
            stack.cost.gain = float(resumed.progress_state["fixed_logit_gain"])
            selected_epoch = int(resumed.progress_state["selected_epoch"])
            selected_validation = dict(resumed.progress_state["selected_validation"])
            mapping_report = deepcopy(resumed.progress_state.get("mapping"))
            deployment_source = deepcopy(
                resumed.progress_state.get("deployment_source")
            )
            deployment_metrics = deepcopy(
                resumed.progress_state.get("deployment")
            )
            initial = dict(resumed.progress_state["initial_validation"])
            initial_clean_validation = dict(
                resumed.progress_state.get(
                    "initial_clean_validation",
                    initial,
                )
            )
            initial_conductances = deepcopy(
                resumed.progress_state.get("initial_conductances")
            )
            resumed_last_train = deepcopy(
                resumed.progress_state.get("last_train")
            )
            resumed_last_validation = deepcopy(
                resumed.progress_state.get("last_validation")
            )
            resumed_last_clean_validation = deepcopy(
                resumed.progress_state.get("last_clean_validation")
            )
            selected_weights = resumed.selected_weights
            if selected_weights is not None:
                save_encoded_named_weights(weights_path, selected_weights, catalog=stack.bundle.catalog)
        elif request.weights is not None:
            loaded = load_named_weights(request.weights, stack.bundle.catalog)
            _validate_checkpoint_metadata(
                loaded.metadata,
                spec=spec,
                teacher_sha256=teacher_sha,
                expected_amplification_indices=(
                    _amplification_index_report(stack)
                ),
            )
            stack.cost.gain = float(loaded.metadata["fixed_logit_gain"])
            mapping_report = deepcopy(loaded.metadata.get("mapping"))
            if cohort_b:
                if device_data_sha is None:
                    raise RuntimeError(
                        "Expected cohort-B device data hash after request "
                        "validation. Provided value: None."
                    )
                _validate_cohort_b_source_metadata(
                    loaded.metadata,
                    spec=spec,
                    device_data_sha256=device_data_sha,
                )
                source_gain = stack.cost.gain
                pre_deployment = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                pre_deployment_conductances = conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                )
                deployment_source = {
                    "path": str(request.weights.expanduser().resolve()),
                    "sha256": sha256_file(request.weights),
                    "encoding": loaded.metadata.get("encoding"),
                    "update_backend": loaded.metadata.get("update_backend"),
                    "initial_target_mapping": loaded.metadata.get(
                        "initial_target_mapping"
                    ),
                    "dual_rail_layout_by_parameter": deepcopy(
                        loaded.metadata.get("dual_rail_layout_by_parameter")
                    ),
                    "device_data_sha256": loaded.metadata.get(
                        "device_data_sha256"
                    ),
                    "device_assignment_sha256_by_parameter": deepcopy(
                        loaded.metadata.get(
                            "device_assignment_sha256_by_parameter"
                        )
                    ),
                    "selection_epoch": loaded.metadata.get("selection_epoch"),
                    "fixed_logit_gain": source_gain,
                }
                optimizer = MeasuredCohortBOptimizer(
                    stack.optimizer,
                    stack.bundle.catalog,
                    spec.settings.update_backend.parameters,
                    request.device_data,
                )
                stack = replace(stack, optimizer=optimizer)
                programming_report = optimizer.initialize_from_loaded_targets()
                post_deployment_source_gain = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                post_deployment_conductances = conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                )
                raw_scores, teacher_logits = collect_calibration(
                    stack,
                    teacher,
                    data.calibration,
                )
                deployment_calibration = fit_positive_logit_gain(
                    raw_scores,
                    teacher_logits,
                    gain_min=spec.mapping.logit_gain_min,
                    gain_max=spec.mapping.logit_gain_max,
                    steps=spec.mapping.logit_gain_steps,
                )
                stack.cost.gain = float(deployment_calibration["gain"])
                post_deployment_calibrated = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                if mapping_report is None:
                    mapping_report = {}
                mapping_report["cohort_b_deployment"] = {
                    "initial_target_mapping": spec.settings.update_backend.parameters[
                        "initial_target_mapping"
                    ],
                    "source_fixed_logit_gain": source_gain,
                    "post_deployment_calibration": deployment_calibration,
                }
                deployment_metrics = {
                    "source": deployment_source,
                    "initial_target_mapping": spec.settings.update_backend.parameters[
                        "initial_target_mapping"
                    ],
                    "pre_deployment": {
                        "validation": pre_deployment,
                        "conductances": pre_deployment_conductances,
                    },
                    "post_deployment_source_gain": {
                        "validation": post_deployment_source_gain,
                        "conductances": post_deployment_conductances,
                    },
                    "post_deployment_calibrated": {
                        "calibration": deployment_calibration,
                        "validation": post_deployment_calibrated,
                        "conductances": post_deployment_conductances,
                    },
                }
                initial = dict(post_deployment_calibrated)
                initial_conductances = post_deployment_conductances
            elif measured:
                if measured_optimizer is None:  # pragma: no cover - guarded above
                    raise AssertionError("measured optimizer type is missing")
                optimizer = measured_optimizer(
                    stack.optimizer,
                    stack.bundle.catalog,
                    spec.settings.update_backend.parameters,
                    request.device_data,
                )
                stack = replace(stack, optimizer=optimizer)
                programming_report = optimizer.initialize_from_reset_targets()
            elif program_verify:
                source_gain = stack.cost.gain
                pre_deployment = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                pre_deployment_conductances = conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                )
                deployment_source = {
                    "path": str(request.weights.expanduser().resolve()),
                    "sha256": sha256_file(request.weights),
                    "update_backend": loaded.metadata.get("update_backend"),
                    "weight_modifier": deepcopy(
                        loaded.metadata.get("weight_modifier")
                    ),
                    "initial_target_mapping": loaded.metadata.get(
                        "initial_target_mapping"
                    ),
                    "selection_epoch": loaded.metadata.get("selection_epoch"),
                    "fixed_logit_gain": source_gain,
                }
                stack = _wrap_program_verify(stack, spec)
                programming_report = (
                    stack.optimizer.initialize_from_loaded_targets()
                )
                post_deployment_source_gain = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                post_deployment_conductances = conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                )
                raw_scores, teacher_logits = collect_calibration(
                    stack,
                    teacher,
                    data.calibration,
                )
                deployment_calibration = fit_positive_logit_gain(
                    raw_scores,
                    teacher_logits,
                    gain_min=spec.mapping.logit_gain_min,
                    gain_max=spec.mapping.logit_gain_max,
                    steps=spec.mapping.logit_gain_steps,
                )
                stack.cost.gain = float(deployment_calibration["gain"])
                post_deployment_calibrated = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                if mapping_report is None:
                    mapping_report = {}
                mapping_report["endpoint_deployment"] = {
                    "device": dict(
                        spec.settings.update_backend.parameters["device"]
                    ),
                    "source_fixed_logit_gain": source_gain,
                    "post_deployment_calibration": deployment_calibration,
                }
                deployment_metrics = {
                    "source": deployment_source,
                    "initial_target_mapping": (
                        "teacher_mapping_then_endpoint_program_verify"
                    ),
                    "pre_deployment": {
                        "validation": pre_deployment,
                        "conductances": pre_deployment_conductances,
                    },
                    "post_deployment_source_gain": {
                        "validation": post_deployment_source_gain,
                        "conductances": post_deployment_conductances,
                    },
                    "post_deployment_calibrated": {
                        "calibration": deployment_calibration,
                        "validation": post_deployment_calibrated,
                        "conductances": post_deployment_conductances,
                    },
                }
                initial = dict(post_deployment_calibrated)
                initial_conductances = post_deployment_conductances
        else:
            measured_candidate_projector = None
            if isinstance(teacher, BoundedDrnTeacher):
                if measured:
                    raise ValueError(
                        "Expected bounded_drn teacher initialization to use "
                        "the ideal or program_verify update backend. Provided "
                        f"value: {backend!r}."
                    )
                mapping_report = _literal_bounded_drn_initialization(
                    stack,
                    teacher,
                    data.calibration,
                    spec,
                )
            else:
                if (
                    measured
                    and spec.settings.update_backend.parameters[
                        "initial_target_mapping"
                    ]
                    in {
                        "dual_rail_pairwise_common_window",
                        "dual_rail_quad_common_window",
                    }
                ):
                    if measured_optimizer is None:  # pragma: no cover
                        raise AssertionError(
                            "measured optimizer type is missing"
                        )
                    measured_candidate_projector = measured_optimizer(
                        stack.optimizer,
                        stack.bundle.catalog,
                        spec.settings.update_backend.parameters,
                        request.device_data,
                    )
                    stack = replace(
                        stack,
                        optimizer=measured_candidate_projector,
                    )
                mapping_report, _targets = select_mapping_and_gain(
                    stack,
                    teacher,
                    data.calibration,
                    spec,
                    measured_candidate_projector=(
                        measured_candidate_projector
                    ),
                )
            stack.cost.gain = float(mapping_report["selected"]["calibration"]["gain"])
            if measured:
                if measured_candidate_projector is None:
                    if measured_optimizer is None:  # pragma: no cover
                        raise AssertionError("measured optimizer type is missing")
                    optimizer = measured_optimizer(
                        stack.optimizer,
                        stack.bundle.catalog,
                        spec.settings.update_backend.parameters,
                        request.device_data,
                    )
                    stack = replace(stack, optimizer=optimizer)
                else:
                    optimizer = measured_candidate_projector
                programming_report = (
                    optimizer.initialize_from_reset_targets()
                )
                raw_scores, teacher_logits = collect_calibration(
                    stack,
                    teacher,
                    data.calibration,
                )
                actual_calibration = fit_positive_logit_gain(
                    raw_scores,
                    teacher_logits,
                    gain_min=spec.mapping.logit_gain_min,
                    gain_max=spec.mapping.logit_gain_max,
                    steps=spec.mapping.logit_gain_steps,
                )
                mapping_report["post_programming_calibration"] = actual_calibration
                stack.cost.gain = float(actual_calibration["gain"])
            elif program_verify:
                pre_deployment = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                pre_deployment_conductances = conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                )
                source_gain = stack.cost.gain
                stack = _wrap_program_verify(stack, spec)
                programming_report = (
                    stack.optimizer.initialize_from_loaded_targets()
                )
                post_deployment_source_gain = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                post_deployment_conductances = conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                )
                raw_scores, teacher_logits = collect_calibration(
                    stack,
                    teacher,
                    data.calibration,
                )
                actual_calibration = fit_positive_logit_gain(
                    raw_scores,
                    teacher_logits,
                    gain_min=spec.mapping.logit_gain_min,
                    gain_max=spec.mapping.logit_gain_max,
                    steps=spec.mapping.logit_gain_steps,
                )
                mapping_report["post_programming_calibration"] = (
                    actual_calibration
                )
                stack.cost.gain = float(actual_calibration["gain"])
                post_deployment_calibrated = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                deployment_metrics = {
                    "source": {
                        "teacher_sha256": teacher_sha,
                        "fixed_logit_gain": source_gain,
                    },
                    "initial_target_mapping": (
                        "teacher_mapping_then_endpoint_program_verify"
                    ),
                    "pre_deployment": {
                        "validation": pre_deployment,
                        "conductances": pre_deployment_conductances,
                    },
                    "post_deployment_source_gain": {
                        "validation": post_deployment_source_gain,
                        "conductances": post_deployment_conductances,
                    },
                    "post_deployment_calibrated": {
                        "calibration": actual_calibration,
                        "validation": post_deployment_calibrated,
                        "conductances": post_deployment_conductances,
                    },
                }
                initial = dict(post_deployment_calibrated)
                initial_conductances = post_deployment_conductances

        ibm_target_mapping_preflights = _ibm_target_mapping_preflights(modifier)
        if initial_clean_validation is None:
            initial_clean_validation = _evaluate(
                stack,
                teacher,
                data.validation,
                maximum_batches=spec.settings.max_validation_batches,
            )
        if initial is None:
            initial = _selection_evaluate(
                stack,
                teacher,
                data.validation,
                spec=spec,
                modifier=modifier,
                clean=initial_clean_validation,
            )
        if initial_conductances is None:
            initial_conductances = conductance_statistics(
                stack.bundle.catalog,
                encoding=spec.model.encoding,
            )
        if deployment_metrics is not None and not resumed_run:
            for mode in (
                "pre_deployment",
                "post_deployment_source_gain",
                "post_deployment_calibrated",
            ):
                store.append_metric(
                    {
                        "mode": mode,
                        **deployment_metrics[mode],
                    }
                )
        store.append_metric(
            {
                "mode": "initialization",
                "mapping": mapping_report,
                "programming": programming_report,
                "validation": initial,
                "validation_clean": initial_clean_validation,
                "selection_evaluation": (
                    spec.settings.selection_evaluation
                ),
                "conductances": initial_conductances,
            }
        )
        if selected_validation is None:
            selected_validation = dict(initial)
            selected_epoch = -1
            selected_weights = _selected_payload(
                stack,
                spec=spec,
                teacher_path=request.teacher_weights,
                teacher_sha256=teacher_sha,
                mapping=mapping_report,
                deployment_source=deployment_source,
                device_model_sha256=device_model_sha,
                epoch=-1,
                validation=initial,
            )
            save_encoded_named_weights(weights_path, selected_weights, catalog=stack.bundle.catalog)

        last_train = resumed_last_train
        last_validation = (
            resumed_last_validation
            if resumed_last_validation is not None
            else initial
        )
        last_clean_validation = (
            resumed_last_clean_validation
            if resumed_last_clean_validation is not None
            else initial_clean_validation
        )
        for epoch in range(start_epoch, spec.settings.num_epochs):
            last_train, batches = _train_epoch(
                stack,
                teacher,
                data.train,
                maximum_batches=spec.settings.max_batches,
                modifier=modifier,
            )
            global_step += batches
            last_clean_validation = _evaluate(
                stack,
                teacher,
                data.validation,
                maximum_batches=spec.settings.max_validation_batches,
            )
            last_validation = _selection_evaluate(
                stack,
                teacher,
                data.validation,
                spec=spec,
                modifier=modifier,
                clean=last_clean_validation,
            )
            improved = _selection_improved(
                last_validation,
                selected_validation,
                metric=spec.settings.selection_metric,
            )
            if improved:
                selected_epoch = epoch
                selected_validation = dict(last_validation)
                selected_weights = _selected_payload(
                    stack,
                    spec=spec,
                    teacher_path=request.teacher_weights,
                    teacher_sha256=teacher_sha,
                    mapping=mapping_report,
                    deployment_source=deployment_source,
                    device_model_sha256=device_model_sha,
                    epoch=epoch,
                    validation=last_validation,
                )
                save_encoded_named_weights(weights_path, selected_weights, catalog=stack.bundle.catalog)
            if selected_weights is None:
                raise RuntimeError("Expected selected KD weights at every epoch boundary.")
            progress = {
                "fixed_logit_gain": stack.cost.gain,
                "selected_epoch": selected_epoch,
                "selected_validation": selected_validation,
                "mapping": mapping_report,
                "initial_validation": initial,
                "initial_clean_validation": initial_clean_validation,
                "initial_conductances": initial_conductances,
                "deployment_source": deployment_source,
                "deployment": deployment_metrics,
                "last_train": last_train,
                "last_validation": last_validation,
                "last_clean_validation": last_clean_validation,
            }
            save_epoch_boundary_checkpoint(
                resume_path,
                catalog=stack.bundle.catalog,
                epoch=epoch + 1,
                global_step=global_step,
                optimizer=stack.optimizer,
                modifier=modifier,
                progress_state=progress,
                selected_weights=selected_weights,
                dataloader_generators={"train": data.train_generator},
                metadata=_model_checkpoint_metadata(
                    stack,
                    spec=spec,
                    teacher_sha256=teacher_sha,
                    mapping=mapping_report,
                    deployment_source=deployment_source,
                    device_model_sha256=device_model_sha,
                ),
            )
            if (epoch + 1) % spec.settings.log_every == 0 or epoch + 1 == spec.settings.num_epochs:
                metric = {
                    "mode": "train",
                    "epoch": epoch,
                    "completed_epochs": epoch + 1,
                    "global_step": global_step,
                    "train": last_train,
                    "validation": last_validation,
                    "validation_clean": last_clean_validation,
                    "selection_evaluation": (
                        spec.settings.selection_evaluation
                    ),
                    "selected": improved,
                    "selected_epoch": selected_epoch,
                    "selection_metric": spec.settings.selection_metric,
                    "selected_value": selected_validation[
                        spec.settings.selection_metric
                    ],
                    "selected_kl": selected_validation["kl_teacher_student"],
                    "conductances": conductance_statistics(
                        stack.bundle.catalog,
                        encoding=spec.model.encoding,
                    ),
                }
                if isinstance(stack.optimizer, MeasuredTraceOptimizer):
                    metric["measured_projection"] = stack.optimizer.programming_report
                elif isinstance(stack.optimizer, ProgramVerifyOptimizer):
                    metric["program_verify"] = stack.optimizer.programming_report
                store.append_metric(metric)

        if selected_weights is None or selected_validation is None:
            raise RuntimeError("Expected training to provide selected KD weights.")
        if not resume_path.exists():
            save_epoch_boundary_checkpoint(
                resume_path,
                catalog=stack.bundle.catalog,
                epoch=start_epoch,
                global_step=global_step,
                optimizer=stack.optimizer,
                modifier=modifier,
                progress_state={
                    "fixed_logit_gain": stack.cost.gain,
                    "selected_epoch": selected_epoch,
                    "selected_validation": selected_validation,
                    "mapping": mapping_report,
                    "initial_validation": initial,
                    "initial_clean_validation": initial_clean_validation,
                    "initial_conductances": initial_conductances,
                    "deployment_source": deployment_source,
                    "deployment": deployment_metrics,
                    "last_train": last_train,
                    "last_validation": last_validation,
                    "last_clean_validation": last_clean_validation,
                },
                selected_weights=selected_weights,
                dataloader_generators={"train": data.train_generator},
                metadata=_model_checkpoint_metadata(
                    stack,
                    spec=spec,
                    teacher_sha256=teacher_sha,
                    mapping=mapping_report,
                    deployment_source=deployment_source,
                    device_model_sha256=device_model_sha,
                ),
            )
        evaluation_modifier = _evaluation_modifier(modifier)
        if isinstance(evaluation_modifier, IbmReramHwaParameterModifier):
            load_named_weights(weights_path, stack.bundle.catalog)
            with evaluation_modifier.evaluation_context():
                pass
            deployment = evaluation_modifier.last_deployment_bundle
            if deployment is None:
                raise RuntimeError(
                    "Expected pulse-resolved selection to produce a persistent "
                    "deployment bundle."
                )
            deployment["selected_weights_sha256"] = sha256_file(weights_path)
            deployment["selected_epoch"] = selected_epoch
            deployment_bundle_path = (
                store.run_dir / "artifacts" / "ibm_om_deployment.pt"
            )
            atomic_torch_save(deployment, deployment_bundle_path)
        relative_improvement = (
            initial["kl_teacher_student"] - selected_validation["kl_teacher_student"]
        ) / initial["kl_teacher_student"] if initial["kl_teacher_student"] > 0.0 else 0.0
        if spec.settings.selection_metric == "student_accuracy":
            selection_improvement = (
                float(selected_validation["student_accuracy"])
                - float(initial["student_accuracy"])
            )
            selection_gate = {
                "metric": "student_accuracy",
                "direction": "maximize",
                "minimum_absolute_improvement": 0.0,
                "observed_absolute_improvement": selection_improvement,
                "passed": selection_improvement >= 0.0,
            }
        else:
            selection_gate = {
                "minimum_relative_kl_improvement": spec.settings.minimum_relative_kl_improvement,
                "passed": relative_improvement
                >= spec.settings.minimum_relative_kl_improvement,
            }
        final_programming = (
            stack.optimizer.programming_report
            if isinstance(
                stack.optimizer,
                (MeasuredTraceOptimizer, ProgramVerifyOptimizer),
            )
            else None
        )
        training_modifier = _training_modifier(modifier)
        training_device_programming = (
            training_modifier.programming_report
            if isinstance(training_modifier, IbmReramHwaParameterModifier)
            else None
        )
        if training_device_programming is not None:
            training_programming_report_path = (
                store.run_dir
                / "artifacts"
                / "ibm_om_training_device_programming.json"
            )
            atomic_write_json(
                training_programming_report_path,
                training_device_programming,
            )
        adaptation = (
            _adaptation_summary(
                initial,
                selected_validation,
                last_validation,
                selected_epoch=selected_epoch,
            )
            if cohort_b or program_verify
            else None
        )
        store.complete(
            metrics={
                "teacher": {"sha256": teacher_sha, "metadata": teacher_metadata},
                "encoding": spec.model.encoding,
                "include_biases": spec.model.include_biases,
                "amplification_indexing": spec.model.amplification_indexing,
                "amplification_indices": _amplification_index_report(stack),
                "objective": "teacher_kl",
                "initialization": "teacher_mapped",
                "weight_modifier": _weight_modifier_metadata(
                    spec.settings.weight_modifier
                ),
                "selection_weight_modifier": _weight_modifier_metadata(
                    spec.settings.selection_weight_modifier
                ),
                "device_model_sha256": device_model_sha,
                "ibm_om_population_fingerprints": ibm_population_fingerprints,
                "ibm_om_target_mapping_preflights": (
                    ibm_target_mapping_preflights
                ),
                "fixed_logit_gain": stack.cost.gain,
                "temperature": spec.settings.temperature,
                "update_backend": spec.settings.update_backend.type,
                "mapping": mapping_report,
                "deployment": deployment_metrics,
                "analog_adaptation": adaptation,
                "initial_validation": initial,
                "initial_clean_validation": initial_clean_validation,
                "last_train": last_train,
                "last_validation": last_validation,
                "last_clean_validation": last_clean_validation,
                "selection_evaluation": (
                    spec.settings.selection_evaluation
                ),
                "selection_metric": spec.settings.selection_metric,
                "selection_noise_repeats": (
                    spec.settings.selection_noise_repeats
                ),
                "selected": {"epoch": selected_epoch, **selected_validation},
                "relative_kl_recovery_from_initialization": relative_improvement,
                "acceptance_gate": selection_gate,
                "initial_conductances": initial_conductances,
                "final_conductances": conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                ),
                "measured_programming": final_programming,
                "device_programming": (
                    final_programming if program_verify else None
                ),
                "selection_device_programming": (
                    evaluation_modifier.programming_report
                    if isinstance(
                        evaluation_modifier,
                        IbmReramHwaParameterModifier,
                    )
                    else None
                ),
                "training_device_programming": training_device_programming,
            },
            artifacts=(
                store.artifact_record(weights_path, kind="selected_named_weights"),
                store.artifact_record(resume_path, kind="epoch_boundary_resume"),
                *_ibm_population_artifact_records(store, modifier),
                *(
                    (
                        store.artifact_record(
                            training_programming_report_path,
                            kind="ibm_om_training_device_programming",
                        ),
                    )
                    if training_programming_report_path is not None
                    else ()
                ),
                *(
                    (
                        store.artifact_record(
                            deployment_bundle_path,
                            kind="ibm_om_persistent_deployment",
                        ),
                    )
                    if deployment_bundle_path is not None
                    else ()
                ),
            ),
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


def run_validate(request: "ValidateRequest") -> int:
    spec = request.spec
    if not isinstance(spec, StudentValidateSpec):
        raise TypeError(
            "Expected mnist_relu_drn_kd.v1 validate to resolve StudentValidateSpec. "
            f"Provided value: {type(spec).__name__}."
        )
    if request.teacher_weights is None:
        raise ValueError(
            "Expected mnist_relu_drn_kd.v1 validation to receive "
            "--teacher-weights. Provided value: None."
        )
    uses_device_model = (
        spec.settings.weight_modifier.type
        == "ibm_reram_om_program_verify"
    )
    if uses_device_model != (request.device_model is not None):
        expected = (
            "an explicit --device-model"
            if uses_device_model
            else "no --device-model"
        )
        raise ValueError(
            f"Expected validation modifier to use {expected}. "
            f"Provided value: {request.device_model!r}."
        )
    validation_inputs = [
        _input("weights", request.weights),
        _input("teacher_weights", request.teacher_weights),
    ]
    if request.device_model is not None:
        validation_inputs.append(_input("device_model", request.device_model))
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=validation_inputs,
    )
    try:
        torch.manual_seed(spec.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(spec.runtime.seed)
        device = torch.device(spec.runtime.device)
        data = build_mnist_loaders(
            spec.data,
            data_seed=spec.runtime.data_seed,
            calibration_examples=spec.mapping.calibration_examples,
            calibration_batch_size=spec.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            request.teacher_weights,
            device=device,
            spec=spec,
        )
        teacher_sha = sha256_file(request.teacher_weights)
        stack = build_student_stack(spec, enable_measured=False)
        loaded = load_named_weights(request.weights, stack.bundle.catalog)
        _validate_checkpoint_metadata(
            loaded.metadata,
            spec=spec,
            teacher_sha256=teacher_sha,
            expected_amplification_indices=_amplification_index_report(stack),
        )
        stack.cost.gain = float(loaded.metadata["fixed_logit_gain"])
        loader = data.validation if spec.settings.split == "validation" else data.test
        modifier = _build_one_weight_modifier(
            stack,
            spec.settings.weight_modifier,
            spec=spec,
            device_model_path=request.device_model,
            population_artifact_dir=store.run_dir / "artifacts",
            population_role="validation",
        )
        if modifier is None:
            reports = [
                _evaluate(
                    stack,
                    teacher,
                    loader,
                    sample_limit=spec.settings.sample_limit,
                )
            ]
        else:
            reports = [
                _evaluate(
                    stack,
                    teacher,
                    loader,
                    sample_limit=spec.settings.sample_limit,
                    modifier=modifier,
                )
                for _repeat in range(spec.settings.noise_repeats)
            ]
        scalar_keys = (
            "kl_teacher_student",
            "raw_kl_teacher_student",
            "student_accuracy",
            "teacher_accuracy",
            "teacher_agreement",
            "raw_score_rms",
            "calibrated_score_rms",
            "teacher_logit_rms",
        )
        metrics = dict(reports[0])
        if len(reports) > 1:
            metrics.update(
                {
                    key: sum(float(report[key]) for report in reports)
                    / len(reports)
                    for key in scalar_keys
                }
            )
        metrics["noise_repeats"] = len(reports)
        metrics["repeat_kl_teacher_student"] = [
            report["kl_teacher_student"] for report in reports
        ]
        deployment_path = None
        if isinstance(modifier, IbmReramHwaParameterModifier):
            metrics["device_programming"] = modifier.programming_report
            deployment = modifier.last_deployment_bundle
            if deployment is None:
                raise RuntimeError(
                    "Expected physical validation to produce a deployment bundle."
                )
            deployment["source_weights_sha256"] = sha256_file(request.weights)
            deployment_path = (
                store.run_dir / "artifacts" / "ibm_om_deployment.pt"
            )
            atomic_torch_save(deployment, deployment_path)
        store.append_metric({"mode": "validate", "split": spec.settings.split, **metrics})
        store.complete(
            metrics={
                "split": spec.settings.split,
                **metrics,
                "encoding": spec.model.encoding,
                "include_biases": spec.model.include_biases,
                "amplification_indexing": spec.model.amplification_indexing,
                "amplification_indices": _amplification_index_report(stack),
                "objective": "teacher_kl",
                "initialization": "teacher_mapped",
                "teacher": {"sha256": teacher_sha, "metadata": teacher_metadata},
                "checkpoint_metadata": loaded.metadata,
                "weight_modifier": _weight_modifier_metadata(
                    spec.settings.weight_modifier
                ),
                "device_model_sha256": (
                    sha256_file(request.device_model)
                    if request.device_model is not None
                    else None
                ),
                "conductances": conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                ),
            },
            artifacts=(
                *_ibm_population_artifact_records(store, modifier),
                *(
                    (
                        store.artifact_record(
                            deployment_path,
                            kind="ibm_om_persistent_deployment",
                        ),
                    )
                    if deployment_path is not None
                    else ()
                ),
            ),
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_train", "run_validate"]
