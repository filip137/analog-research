"""Application handlers for ``small_drn.v1``.

The public functions in this module own one command/run lifecycle each.  They
compose the numerical objects from :mod:`components`, invoke the shared
training/evaluation engine, and persist only through the versioned RunStore
and checkpoint contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from itertools import islice
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.artifacts import (
    ArtifactRecord,
    RunStore,
    atomic_write_json,
    content_hash,
    sha256_file,
)
from experiments.schema import to_plain_data
from experiments.small_network.components import (
    EvaluationRuntime,
    TrainRuntime,
    ValidationRuntime,
    build_evaluation_runtime,
    build_model_stack,
    build_train_runtime,
    build_validation_runtime,
    configured_resume_capability,
    seed_runtime,
)
from experiments.small_network.config import (
    LinspaceSpec,
    SmallDrnConfig,
    TrainSpec,
    ValidateSpec,
)
from labs.datasets import MoonsDataset
from model.resistive.builders import ParameterCatalog
from model.resistive.interaction import DenseResistive, SignedDenseResistive
from model.resistive.digital_low_rank_config import (
    parse_digital_low_rank_adapter,
)
from model.resistive.device_config import Wan2022ProgrammingConfig
from model.resistive.passive_layerwise_low_rank_config import (
    parse_passive_layerwise_low_rank_adapter,
)
from training.checkpoint import (
    LEGACY_BASE_ONLY,
    LEGACY_FULL,
    encode_named_weights,
    load_epoch_boundary_checkpoint,
    load_legacy_positional_weights,
    load_named_weights,
    save_encoded_named_weights,
    save_epoch_boundary_checkpoint,
    save_named_weights,
)
from training.core.engine import (
    FreePhaseEvent,
    evaluate,
    train_epoch,
)
from training.core.guards import FiniteGradientGuard
from training.device_programming import (
    program_device_base_conductances,
    program_wan2022_base_conductance,
    program_wan2022_base_conductances,
)
from training.core.probes import (
    MeanCostProbe,
    MeanErrorProbe,
    ResidualInfinityNormProbe,
    SettledLayerStatesProbe,
    SolverIterationCountsProbe,
)
from training.program_verify import ProgramVerifyOptimizer
from training.ibm_om_fp32_bounds import IbmOmFp32BoundsOptimizer
from training.measured_trace import (
    MeasuredCohortAOptimizer,
    MeasuredCohortBOptimizer,
    MeasuredCohortBLoRAOptimizer,
    MeasuredTraceOptimizer,
)

if TYPE_CHECKING:
    from ebl.cli import (
        ImportLegacyCheckpointRequest,
        LinspaceRequest,
        TrainRequest,
        ValidateRequest,
    )


@dataclass(frozen=True)
class TrainingEpochReport:
    """Typed, immutable measurements available after one completed epoch.

    Error and accuracy values are fractions in ``[0, 1]``.  Selection fields
    always describe the global clean-validation best known at this boundary,
    including a best restored from a resume checkpoint.
    """

    epoch_index: int
    completed_epochs: int
    global_step: int
    train_examples: int
    train_mean_cost: float | None
    train_error_fraction: float | None
    train_accuracy: float | None
    validation_examples: int
    validation_mean_cost: float
    validation_error_fraction: float
    validation_accuracy: float
    selected_this_epoch: bool
    selected_epoch_index: int
    selected_validation_cost: float
    selected_validation_error_fraction: float | None
    selected_validation_accuracy: float | None


@dataclass(frozen=True)
class TrainingOutcome:
    """Typed completion record for programmatic training callers."""

    run_dir: Path
    result_path: Path
    weights_path: Path
    resume_path: Path
    completed_epochs: int
    global_step: int
    selected_epoch_index: int
    selected_validation_cost: float
    selected_validation_error_fraction: float | None
    selected_validation_accuracy: float | None
    last_epoch_report: TrainingEpochReport | None
    resume_capability: str


TrainingObserver = Callable[[TrainingEpochReport], None]


def run_train(request: "TrainRequest") -> int:
    """CLI-compatible wrapper around :func:`execute_train`."""

    execute_train(request)
    return 0


def execute_train(
    request: "TrainRequest",
    observers: Iterable[TrainingObserver] = (),
) -> TrainingOutcome:
    """Train one run and return typed results to programmatic callers.

    Observers run exactly once after every completed epoch, after clean
    validation, global-best selection, resume persistence, and normal metric
    logging for that boundary.  Their cadence is independent of
    ``settings.log_every``.  An observer exception is recorded as the run's
    terminal failure and then re-raised unchanged.
    """

    spec = _expect_spec(request.spec, TrainSpec, mode="train")
    _validate_training_initialization(request, spec)
    active_observers = _validated_observers(observers)
    resume_capability = configured_resume_capability(spec)
    store = _create_run_store(
        request,
        spec,
        input_artifacts=_training_input_artifacts(request),
        resume_capability=resume_capability,
        runtime_protocol={
            "selection_evaluation": "clean",
            "validation_requested": "validation",
            "validation_effective": (
                "held_out_validation"
                if spec.common.data.validation_points is not None
                else "held_out_test"
            ),
            "parameter_modifier": {
                "type": spec.settings.weight_modifier.type,
                "parameters": dict(
                    spec.settings.weight_modifier.parameters
                ),
            },
            "noisy_evaluation_order": "before_clean",
            "checkpoint_weights": "checkpoints/weights.pt",
            "checkpoint_resume": "checkpoints/resume.pt",
        },
    )
    selection_report: Mapping[str, Any] | None = None
    selection_path: Path | None = None
    try:
        seed_runtime(spec.common.runtime.seed)
        if (
            request.resume is None
            and spec.settings.learning_rate_selection.type
            == "bounded_relative_update_grid"
        ):
            from experiments.small_network.learning_rate_selection import (
                select_measured_learning_rates,
            )

            selection_runtime = build_train_runtime(
                spec,
                device_data_path=request.device_data,
            )
            selection_report = select_measured_learning_rates(
                spec,
                selection_runtime,
                store=store,
                initial_weights_path=request.weights,
            )
            selection_path = store.run_dir / "artifacts" / "lr_selection.json"
            atomic_write_json(selection_path, selection_report)
            selected_rates = tuple(
                float(value)
                for value in selection_report["selection"][
                    "selected_learning_rate_vector"
                ]
            )
            del selection_runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Production is an exact restart, including model initialization,
            # data-shuffle state, and virtual-device assignment.
            seed_runtime(spec.common.runtime.seed)
            runtime = build_train_runtime(
                spec,
                device_data_path=request.device_data,
            )
            if not isinstance(runtime.optimizer, MeasuredTraceOptimizer):
                raise RuntimeError(
                    "Expected the selected measured production runtime to "
                    "expose MeasuredTraceOptimizer. Provided value: "
                    f"{type(runtime.optimizer).__name__}."
                )
            runtime.optimizer.set_learning_rates(selected_rates)
        else:
            runtime = build_train_runtime(
                spec,
                device_data_path=request.device_data,
            )
        if runtime.resume_capability != resume_capability:
            raise RuntimeError(
                "Expected preflight and numerical resume capabilities to "
                "match. Provided value: "
                f"preflight={resume_capability!r}, "
                f"numerical={runtime.resume_capability!r}."
            )
        metrics, artifacts, last_epoch_report = _execute_training(
            request,
            spec,
            runtime,
            store,
            observers=active_observers,
        )
        if selection_report is not None and selection_path is not None:
            selection_summary = dict(selection_report["selection"])
            metrics = {
                **metrics,
                "learning_rate_selection": selection_summary,
            }
            artifacts = (
                *artifacts,
                store.artifact_record(
                    selection_path,
                    kind="learning_rate_selection",
                ),
            )
        result_path = store.complete(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        store.fail(exc)
        raise
    selected = metrics["selected"]
    return TrainingOutcome(
        run_dir=store.run_dir,
        result_path=result_path,
        weights_path=store.run_dir / "checkpoints" / "weights.pt",
        resume_path=store.run_dir / "checkpoints" / "resume.pt",
        completed_epochs=int(metrics["completed_epochs"]),
        global_step=int(metrics["global_step"]),
        selected_epoch_index=int(selected["epoch"]),
        selected_validation_cost=float(selected["value"]),
        selected_validation_error_fraction=_optional_float(
            selected.get("error_fraction")
        ),
        selected_validation_accuracy=_optional_float(
            selected.get("accuracy")
        ),
        last_epoch_report=last_epoch_report,
        resume_capability=str(metrics["resume_capability"]),
    )


def run_linspace(request: "LinspaceRequest") -> int:
    """Evaluate explicit named weights on the configured two-dimensional grid."""

    spec = _expect_spec(request.spec, LinspaceSpec, mode="linspace")
    store = _create_run_store(
        request,
        spec,
        input_artifacts=(_input_artifact("weights", request.weights),),
        resume_capability="unsupported",
        runtime_protocol={
            "input_assignment": "standard_train_mode",
            "input_gain": 1.0,
            "legacy_test_mode_effect": "no_op",
        },
    )
    try:
        seed_runtime(spec.common.runtime.seed)
        runtime = build_evaluation_runtime(spec.common)
        load_named_weights(request.weights, runtime.stack.bundle.catalog)
        metrics, artifacts = _execute_linspace(spec, runtime, store)
        store.append_metric({"mode": "linspace", **metrics})
        store.complete(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        store.fail(exc)
        raise
    return 0


def run_validate(request: "ValidateRequest") -> int:
    """Evaluate explicit named weights on one configured dataset split."""

    spec = _expect_spec(request.spec, ValidateSpec, mode="validate")
    if spec.settings.split == "train":
        effective_split = "train"
    elif (
        spec.settings.split == "validation"
        and spec.common.data.validation_points is not None
    ):
        effective_split = "held_out_validation"
    else:
        effective_split = "held_out_test"
    store = _create_run_store(
        request,
        spec,
        input_artifacts=(_input_artifact("weights", request.weights),),
        resume_capability="unsupported",
        runtime_protocol={
            "split_requested": spec.settings.split,
            "split_effective": effective_split,
            "evaluation": "clean",
        },
    )
    try:
        seed_runtime(spec.common.runtime.seed)
        runtime = build_validation_runtime(spec.common)
        load_named_weights(request.weights, runtime.stack.bundle.catalog)
        metrics, artifacts = _execute_validation(spec, runtime, store)
        store.append_metric({"mode": "validate", **metrics})
        store.complete(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        store.fail(exc)
        raise
    return 0


def import_legacy_checkpoint(
    request: "ImportLegacyCheckpointRequest",
) -> int:
    """Convert one explicitly profiled positional checkpoint to named weights."""

    if not isinstance(request.document, SmallDrnConfig):
        raise TypeError(
            "Expected checkpoint import document to be a SmallDrnConfig. "
            f"Provided value: {type(request.document).__name__}."
        )
    source = Path(request.source).expanduser().resolve()
    output = Path(request.output).expanduser().resolve()
    if source == output:
        raise ValueError(
            "Expected checkpoint import output to differ from its source. "
            f"Provided value: {str(output)!r}."
        )
    seed_runtime(request.document.common.runtime.seed)
    stack = build_model_stack(request.document.common)
    profile = LEGACY_FULL if request.kind == "full" else LEGACY_BASE_ONLY
    output_catalog = (
        stack.bundle.catalog
        if request.kind == "full"
        else _base_checkpoint_catalog(stack.bundle.catalog)
    )
    loaded = load_legacy_positional_weights(
        source,
        stack.bundle.catalog,
        profile=profile,
    )
    save_named_weights(
        output,
        output_catalog,
        metadata={
            "experiment_id": request.definition.experiment_id,
            "legacy_profile": profile.name,
            "legacy_schema": loaded.schema,
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "config_sha256": content_hash(to_plain_data(request.document)),
        },
    )
    return 0


def _execute_training(
    request: Any,
    spec: TrainSpec,
    runtime: TrainRuntime,
    store: RunStore,
    *,
    observers: tuple[TrainingObserver, ...],
) -> tuple[
    dict[str, Any],
    tuple[ArtifactRecord, ...],
    TrainingEpochReport | None,
]:
    catalog = runtime.stack.bundle.catalog
    weights_path = store.run_dir / "checkpoints" / "weights.pt"
    resume_path = store.run_dir / "checkpoints" / "resume.pt"
    generators = runtime.data.dataloader_generators

    start_epoch = 0
    global_step = 0
    selected_cost: float | None = None
    selected_epoch: int | None = None
    selected_error: float | None = None
    selected_accuracy: float | None = None
    selected_weights: Mapping[str, Any] | None = None
    last_epoch_report: TrainingEpochReport | None = None
    device_programming: Mapping[str, Any] | None = None
    pre_deployment_validation: Mapping[str, Any] | None = None
    initial_validation: Mapping[str, Any] | None = None

    if request.resume is not None:
        resumed = load_epoch_boundary_checkpoint(
            request.resume,
            catalog=catalog,
            optimizer=runtime.optimizer,
            modifier=runtime.modifier,
            scheduler=None,
            runtime_state=runtime.runtime_state,
            dataloader_generators=generators,
        )
        if resumed.resume_capability != runtime.resume_capability:
            raise ValueError(
                "Expected resume checkpoint capability to match this runtime. "
                f"Provided value: checkpoint={resumed.resume_capability!r}, "
                f"runtime={runtime.resume_capability!r}."
            )
        expected_resume_config = _resume_config_sha256(spec)
        provided_resume_config = resumed.metadata.get(
            "resume_config_sha256"
        )
        if provided_resume_config != expected_resume_config:
            raise ValueError(
                "Expected the resume checkpoint numerical configuration to "
                "match the requested training configuration except for "
                "modes.train.num_epochs. "
                f"Provided value: checkpoint={provided_resume_config!r}, "
                f"requested={expected_resume_config!r}."
            )
        if resumed.epoch > spec.settings.num_epochs:
            raise ValueError(
                "Expected modes.train.num_epochs to be greater than or equal "
                "to the completed epoch stored in the resume checkpoint. "
                f"Provided value: configured={spec.settings.num_epochs}, "
                f"checkpoint={resumed.epoch}."
            )
        if resumed.selected_weights is None:
            raise ValueError(
                "Expected an epoch-boundary resume checkpoint to include its "
                "clean-evaluation selected_weights snapshot. "
                "Provided value: null."
            )
        selected_weights = resumed.selected_weights
        (
            selected_cost,
            selected_epoch,
            selected_error,
            selected_accuracy,
        ) = _selection_from_progress(resumed.progress_state)
        save_encoded_named_weights(
            weights_path,
            selected_weights,
            catalog=catalog,
        )
        start_epoch = resumed.epoch
        global_step = resumed.global_step
    elif request.weights is not None:
        load_named_weights(request.weights, catalog)
    elif request.base_weights is not None:
        load_named_weights(
            request.base_weights,
            _base_checkpoint_catalog(catalog),
        )
        if spec.common.model.adapter.type == "digital_low_rank":
            adapter_config = parse_digital_low_rank_adapter(
                dict(spec.common.model.adapter.parameters),
                path="config.model.adapter.parameters",
            )
            if adapter_config is None:  # pragma: no cover - schema guarantees it
                raise AssertionError("digital_low_rank config resolved to null")
            if isinstance(
                adapter_config.device_noise,
                Wan2022ProgrammingConfig,
            ):
                device_programming = program_wan2022_base_conductance(
                    catalog,
                    adapter_config.device_noise,
                )
            else:
                dense_base_keys = tuple(
                    binding.key
                    for binding in catalog.for_group(
                        "base",
                        checkpointed_only=True,
                    )
                    if binding.role == "dense_weight"
                )
                if len(dense_base_keys) != 1:
                    raise ValueError(
                        "Expected digital_low_rank to contain exactly one "
                        "base dense weight. Provided value: "
                        f"{dense_base_keys!r}."
                    )
                device_programming = program_device_base_conductances(
                    catalog,
                    {dense_base_keys[0]: adapter_config.device_noise},
                )
        elif (
            spec.common.model.adapter.type
            == "passive_layerwise_low_rank"
        ):
            if isinstance(
                runtime.optimizer,
                MeasuredCohortBLoRAOptimizer,
            ):
                # The measured LoRA optimizer owns the one-time cohort-B base
                # deployment as well as RESET-endpoint adapter initialization.
                adapter_config = None
            else:
                adapter_config = parse_passive_layerwise_low_rank_adapter(
                    dict(spec.common.model.adapter.parameters),
                    path="config.model.adapter.parameters",
                )
            if (
                adapter_config is None
                and not isinstance(
                    runtime.optimizer,
                    MeasuredCohortBLoRAOptimizer,
                )
            ):  # pragma: no cover - schema guarantees it
                raise AssertionError(
                    "passive_layerwise_low_rank config resolved to null"
                )
            if adapter_config is not None:
                layer_configs = {
                    layer.parameter_key: layer.device_noise
                    for layer in adapter_config.layers
                }
                if all(
                    isinstance(value, Wan2022ProgrammingConfig)
                    for value in layer_configs.values()
                ):
                    device_programming = program_wan2022_base_conductances(
                        catalog,
                        layer_configs,  # type: ignore[arg-type]
                    )
                else:
                    device_programming = program_device_base_conductances(
                        catalog,
                        layer_configs,  # type: ignore[arg-type]
                    )

    if (
        isinstance(runtime.optimizer, ProgramVerifyOptimizer)
        and request.resume is None
    ):
        trainable_programming = (
            runtime.optimizer.initialize_from_loaded_targets()
        )
        device_programming = {
            "base": (
                None
                if device_programming is None
                else dict(device_programming)
            ),
            "trainable_initial_write": trainable_programming,
            "semantics": (
                "digital shadow target with a noisy program-and-verify "
                "write after every optimizer step"
            ),
            "pulse_model": False,
        }

    if isinstance(runtime.optimizer, MeasuredCohortBOptimizer) and request.resume is None:
        layer_snapshot = runtime.runtime_state.state_dict()
        source_result = evaluate(
            runtime.evaluation_components,
            _limit_batches(
                runtime.data.held_out_loader,
                spec.settings.max_validation_batches,
            ),
            modifier=None,
            probes=(MeanCostProbe(), MeanErrorProbe()),
            epoch=-1,
            split="pre_deployment_validation",
            reset_input=True,
        )
        pre_deployment_validation = _classification_metrics(source_result)
        runtime.runtime_state.load_state_dict(layer_snapshot)
        store.append_metric(
            {
                "mode": "pre_deployment",
                "source_weights": {
                    "path": str(Path(request.weights).expanduser().resolve()),
                    "sha256": sha256_file(request.weights),
                },
                "validation": dict(pre_deployment_validation),
            }
        )

    if isinstance(runtime.optimizer, MeasuredTraceOptimizer) and request.resume is None:
        if isinstance(runtime.optimizer, MeasuredCohortBLoRAOptimizer):
            measured_initialization = (
                runtime.optimizer.initialize_from_base_and_reset_adapters()
            )
            semantics = (
                "loaded cohort-A base frozen after one cohort-B projection; "
                "four physical low-rank factor arrays start at the fully "
                "reset trace endpoint and alone receive measured writes"
            )
        elif isinstance(runtime.optimizer, MeasuredCohortAOptimizer):
            measured_initialization = (
                runtime.optimizer.initialize_at_pulse_zero()
            )
            semantics = (
                "digital shadow with global-nearest projection onto a "
                "deterministic interpolated cohort-A measured curve"
            )
        else:
            measured_initialization = (
                runtime.optimizer.initialize_from_loaded_targets()
            )
            semantics = (
                "loaded cohort-A targets retained as digital shadows and "
                "globally projected onto deterministic interpolated "
                "cohort-B measured curves before fine-tuning"
            )
        device_programming = {
            "semantics": semantics,
            "pulse_model": False,
            "cohort_a_used": runtime.optimizer.cohort == "A",
            "cohort_b_used": runtime.optimizer.cohort == "B",
            "source_checkpoint": (
                None
                if request.weights is None and request.base_weights is None
                else {
                    "path": str(
                        Path(
                            request.weights
                            if request.weights is not None
                            else request.base_weights
                        )
                        .expanduser()
                        .resolve()
                    ),
                    "sha256": sha256_file(
                        request.weights
                        if request.weights is not None
                        else request.base_weights
                    ),
                }
            ),
            "initialization": measured_initialization,
        }

    if (
        spec.common.model.adapter.type
        in {"digital_low_rank", "passive_layerwise_low_rank"}
        or isinstance(
            runtime.optimizer,
            (
                ProgramVerifyOptimizer,
                MeasuredTraceOptimizer,
                IbmOmFp32BoundsOptimizer,
            ),
        )
    ) and request.resume is None:
        layer_snapshot = runtime.runtime_state.state_dict()
        initial_result = evaluate(
            runtime.evaluation_components,
            _limit_batches(
                runtime.data.held_out_loader,
                spec.settings.max_validation_batches,
            ),
            modifier=None,
            probes=(MeanCostProbe(), MeanErrorProbe()),
            epoch=-1,
            split="initial_validation",
            reset_input=True,
        )
        initial_validation = _classification_metrics(initial_result)
        runtime.runtime_state.load_state_dict(layer_snapshot)
        store.append_metric(
            {
                "mode": "initialization",
                "device_programming": (
                    None
                    if device_programming is None
                    else dict(device_programming)
                ),
                "om_fp32_bounds": (
                    runtime.optimizer.bounds_report
                    if isinstance(
                        runtime.optimizer,
                        IbmOmFp32BoundsOptimizer,
                    )
                    else None
                ),
                "validation": dict(initial_validation),
            }
        )

    last_validation: dict[str, Any] | None = None
    last_noisy_validation: dict[str, Any] | None = None
    completed_epoch = start_epoch
    for epoch in range(start_epoch, spec.settings.num_epochs):
        train_metrics = _TrainingMetrics()
        training_loader = _limit_batches(
            runtime.data.train_loader,
            spec.settings.max_batches,
        )
        trained = train_epoch(
            runtime.training_components,
            training_loader,
            modifier=runtime.modifier,
            event_handlers=(train_metrics, FiniteGradientGuard()),
            epoch=epoch,
            start_global_step=global_step,
            reset_input=False,
        )
        global_step = trained.next_global_step
        completed_epoch = epoch + 1

        if runtime.noisy_evaluation:
            noisy_validation = evaluate(
                runtime.evaluation_components,
                _limit_batches(
                    runtime.data.held_out_loader,
                    spec.settings.max_validation_batches,
                ),
                modifier=runtime.modifier,
                probes=(MeanCostProbe(), MeanErrorProbe()),
                epoch=epoch,
                split="validation_noisy",
                reset_input=True,
            )
            last_noisy_validation = _classification_metrics(
                noisy_validation
            )

        validation = evaluate(
            runtime.evaluation_components,
            _limit_batches(
                runtime.data.held_out_loader,
                spec.settings.max_validation_batches,
            ),
            modifier=None,
            probes=(MeanCostProbe(), MeanErrorProbe()),
            epoch=epoch,
            split="validation",
            reset_input=True,
        )
        last_validation = _classification_metrics(validation)
        validation_cost = last_validation["mean_cost"]
        if validation_cost is None:
            raise ValueError(
                "Expected clean validation to evaluate at least one example. "
                f"Provided value: {validation.example_count} examples."
            )

        improved = selected_cost is None or validation_cost < selected_cost
        if improved:
            selected_cost = validation_cost
            selected_epoch = epoch
            selected_error = last_validation["mean_error"]
            selected_accuracy = last_validation["accuracy"]
            selected_weights = encode_named_weights(
                catalog,
                metadata={
                    **_selected_model_metadata(spec, runtime),
                    "selection_metric": "validation.mean_cost",
                    "selection_value": selected_cost,
                    "selection_epoch": selected_epoch,
                    "selection_error_fraction": selected_error,
                    "selection_accuracy": selected_accuracy,
                },
            )
            save_encoded_named_weights(
                weights_path,
                selected_weights,
                catalog=catalog,
            )

        if selected_weights is None or selected_epoch is None:
            raise RuntimeError(
                "Expected clean validation to select a named-weights snapshot. "
                "Provided value: no selected snapshot."
            )
        progress_state = _selection_progress(
            cost=selected_cost,
            epoch=selected_epoch,
            error_fraction=selected_error,
            accuracy=selected_accuracy,
        )
        save_epoch_boundary_checkpoint(
            resume_path,
            catalog=catalog,
            epoch=completed_epoch,
            global_step=global_step,
            optimizer=runtime.optimizer,
            modifier=runtime.modifier,
            scheduler=None,
            runtime_state=runtime.runtime_state,
            progress_state=progress_state,
            selected_weights=selected_weights,
            dataloader_generators=generators,
            resume_capability=runtime.resume_capability,
            metadata=_resume_metadata(spec, runtime),
        )

        should_log = (
            completed_epoch % spec.settings.log_every == 0
            or completed_epoch == spec.settings.num_epochs
        )
        train_values = train_metrics.result()
        last_epoch_report = _epoch_report(
            epoch=epoch,
            completed_epoch=completed_epoch,
            global_step=global_step,
            train=train_values,
            validation=last_validation,
            improved=improved,
            selected_epoch=selected_epoch,
            selected_cost=selected_cost,
            selected_error=selected_error,
            selected_accuracy=selected_accuracy,
        )
        if should_log:
            epoch_metric = {
                "mode": "train",
                "epoch": epoch,
                "completed_epochs": completed_epoch,
                "global_step": global_step,
                "train": train_values,
                "validation": last_validation,
                "validation_noisy": last_noisy_validation,
                "selected": improved,
                "selected_epoch": selected_epoch,
                "selected_cost": selected_cost,
            }
            if isinstance(runtime.optimizer, MeasuredTraceOptimizer):
                epoch_metric["measured_projection"] = (
                    runtime.optimizer.programming_report
                )
            store.append_metric(epoch_metric)
        for observer in observers:
            observer(last_epoch_report)

    if selected_weights is None:
        raise ValueError(
            "Expected training or its resume checkpoint to provide selected "
            "clean-evaluation weights. Provided value: null."
        )

    # A continuation whose configured target epoch is already complete still
    # owns a fresh, self-contained resume artifact.
    if not resume_path.exists():
        save_epoch_boundary_checkpoint(
            resume_path,
            catalog=catalog,
            epoch=completed_epoch,
            global_step=global_step,
            optimizer=runtime.optimizer,
            modifier=runtime.modifier,
            scheduler=None,
            runtime_state=runtime.runtime_state,
            progress_state=_selection_progress(
                cost=selected_cost,
                epoch=selected_epoch,
                error_fraction=selected_error,
                accuracy=selected_accuracy,
            ),
            selected_weights=selected_weights,
            dataloader_generators=generators,
            resume_capability=runtime.resume_capability,
            metadata=_resume_metadata(spec, runtime),
        )

    om_bounds_report = (
        runtime.optimizer.bounds_report
        if isinstance(runtime.optimizer, IbmOmFp32BoundsOptimizer)
        else None
    )
    metrics = {
        "completed_epochs": completed_epoch,
        "global_step": global_step,
        "selected": {
            "metric": "validation.mean_cost",
            "value": selected_cost,
            "epoch": selected_epoch,
            "error_fraction": selected_error,
            "accuracy": selected_accuracy,
            "evaluation": "clean",
        },
        "last_validation": last_validation,
        "last_noisy_validation": last_noisy_validation,
        "evaluation_protocol": {
            "validation_requested": "validation",
            "validation_effective": (
                "held_out_validation"
                if spec.common.data.validation_points is not None
                else "held_out_test"
            ),
            "selection_evaluation": "clean",
            "noisy_evaluation_order": "before_clean",
            "parameter_modifier": _modifier_provenance(spec, runtime),
        },
        "resume_capability": runtime.resume_capability,
        "device_programming": (
            None if device_programming is None else dict(device_programming)
        ),
        "update_programming": (
            runtime.optimizer.programming_report
            if isinstance(
                runtime.optimizer,
                (ProgramVerifyOptimizer, MeasuredTraceOptimizer),
            )
            else None
        ),
        "om_fp32_bounds": om_bounds_report,
        "initial_validation": (
            None if initial_validation is None else dict(initial_validation)
        ),
        "pre_deployment_validation": (
            None
            if pre_deployment_validation is None
            else dict(pre_deployment_validation)
        ),
    }
    extra_artifacts: tuple[ArtifactRecord, ...] = ()
    if om_bounds_report is not None:
        bounds_path = store.run_dir / "artifacts" / "om_fp32_bounds.json"
        atomic_write_json(bounds_path, om_bounds_report)
        extra_artifacts = (
            store.artifact_record(bounds_path, kind="om_fp32_bounds"),
        )
    artifacts = (
        store.artifact_record(weights_path, kind="weights"),
        store.artifact_record(resume_path, kind="resume"),
        *extra_artifacts,
    )
    return metrics, artifacts, last_epoch_report


def _execute_linspace(
    spec: LinspaceSpec,
    runtime: EvaluationRuntime,
    store: RunStore,
) -> tuple[dict[str, Any], tuple[ArtifactRecord, ...]]:
    raw_points, model_points = _linspace_points(spec, runtime)
    labels = torch.zeros(
        model_points.shape[0],
        dtype=torch.long,
        device=runtime.stack.device,
    )
    loader = DataLoader(
        TensorDataset(model_points, labels),
        batch_size=spec.common.data.batch_size,
        shuffle=False,
    )
    probes: list[Any] = [
        ResidualInfinityNormProbe(),
        SolverIterationCountsProbe(),
    ]
    if spec.settings.record_states:
        probes.append(SettledLayerStatesProbe())

    input_layer = runtime.stack.network.layers()[0]
    original_gain = getattr(input_layer, "_gain", None)
    if original_gain is None:
        raise ValueError(
            "Expected the small-DRN input layer to expose its gain. "
            f"Provided value: {input_layer!r}."
        )
    input_layer._gain = 1.0
    try:
        result = evaluate(
            runtime.components,
            loader,
            probes=tuple(probes),
            epoch=0,
            split="linspace",
            reset_input=True,
        )
    finally:
        input_layer._gain = original_gain

    inputs_path = store.run_dir / "artifacts" / "linspace_inputs.npz"
    residuals_path = (
        store.run_dir / "artifacts" / "linspace_residual_currents.npz"
    )
    iterations_path = (
        store.run_dir / "artifacts" / "linspace_iteration_counts.npz"
    )
    _atomic_npz(inputs_path, inputs=raw_points.detach().cpu().numpy())
    residuals = result.probe_value("residual_inf_norm")
    _atomic_npz(residuals_path, **residuals)
    iteration_counts = result.probe_value("solver_iteration_counts")
    _atomic_npz(
        iterations_path,
        iteration_counts=iteration_counts,
        iteration_grid=iteration_counts.reshape(
            spec.settings.samples,
            spec.settings.samples,
        ),
        linspace_samples=np.asarray(spec.settings.samples),
    )

    artifacts = [
        store.artifact_record(inputs_path, kind="linspace_inputs"),
        store.artifact_record(residuals_path, kind="residuals"),
        store.artifact_record(iterations_path, kind="solver_iterations"),
    ]
    if spec.settings.record_states:
        states_path = store.run_dir / "artifacts" / "linspace_states.npz"
        _atomic_npz(
            states_path,
            **_numpy_mapping(result.probe_value("settled_layer_states")),
        )
        artifacts.append(store.artifact_record(states_path, kind="states"))

    residual_metrics = _residual_metrics(
        residuals,
        tolerance=spec.common.solver.tolerances.residual_current,
    )
    metrics = {
        "examples": result.example_count,
        "batches": result.batch_count,
        "grid": {
            "minimum": spec.settings.minimum,
            "maximum": spec.settings.maximum,
            "samples_per_axis": spec.settings.samples,
        },
        "residuals": residual_metrics,
        "solver_iterations": _numeric_summary(iteration_counts),
        "input_protocol": {
            "configured_gain": spec.common.model.input_gain,
            "effective_gain": 1.0,
            "assignment_mode": "train",
            "legacy_test_assignment_effect": "no_op",
        },
    }
    return metrics, tuple(artifacts)


def _execute_validation(
    spec: ValidateSpec,
    runtime: ValidationRuntime,
    store: RunStore,
) -> tuple[dict[str, Any], tuple[ArtifactRecord, ...]]:
    if spec.settings.split == "train":
        loader = runtime.data.train_loader
        effective_split = "train"
    elif (
        spec.settings.split == "validation"
        and spec.common.data.validation_points is not None
    ):
        loader = runtime.data.held_out_loader
        effective_split = "held_out_validation"
    else:
        loader = runtime.data.test_loader
        effective_split = "held_out_test"
    limited = _limit_examples(loader, spec.settings.sample_limit)
    capture = _BatchCapture()
    probes: list[Any] = [
        MeanCostProbe(),
        MeanErrorProbe(),
        ResidualInfinityNormProbe(),
        SolverIterationCountsProbe(),
    ]
    if spec.settings.record_states:
        probes.append(SettledLayerStatesProbe())
    result = evaluate(
        runtime.components,
        limited,
        probes=tuple(probes),
        event_handlers=(capture,),
        epoch=0,
        split=spec.settings.split,
        reset_input=True,
    )
    metrics = {
        **_classification_metrics(result),
        "residuals": _residual_metrics(
            result.probe_value("residual_inf_norm"),
            tolerance=spec.common.solver.tolerances.residual_current,
        ),
        "solver_iterations": _numeric_summary(
            result.probe_value("solver_iteration_counts")
        ),
        "split_protocol": {
            "requested": spec.settings.split,
            "effective": effective_split,
        },
        "evaluation": "clean",
    }

    capture_values = capture.result()
    inputs_path = store.run_dir / "artifacts" / "validation_inputs.npz"
    _atomic_npz(inputs_path, **capture_values)
    residuals_path = (
        store.run_dir / "artifacts" / "validation_residual_currents.npz"
    )
    _atomic_npz(
        residuals_path,
        **result.probe_value("residual_inf_norm"),
    )
    iterations_path = (
        store.run_dir / "artifacts" / "validation_iteration_counts.npz"
    )
    _atomic_npz(
        iterations_path,
        iteration_counts=result.probe_value("solver_iteration_counts"),
    )
    artifacts = [
        store.artifact_record(inputs_path, kind="validation_inputs"),
        store.artifact_record(residuals_path, kind="residuals"),
        store.artifact_record(iterations_path, kind="solver_iterations"),
    ]
    if spec.settings.record_states:
        states_path = store.run_dir / "artifacts" / "validation_states.npz"
        _atomic_npz(
            states_path,
            **_numpy_mapping(result.probe_value("settled_layer_states")),
        )
        artifacts.append(store.artifact_record(states_path, kind="states"))
    return metrics, tuple(artifacts)


def _linspace_points(
    spec: LinspaceSpec,
    runtime: EvaluationRuntime,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(
        spec.settings.minimum,
        spec.settings.maximum,
        spec.settings.samples,
        dtype=runtime.stack.dtype,
        device=runtime.stack.device,
    )
    x_axis, y_axis = torch.meshgrid(axis, axis, indexing="ij")
    raw = torch.stack((x_axis.flatten(), y_axis.flatten()), dim=1)
    logical_width = runtime.stack.logical_input_dim
    if logical_width == 2:
        return raw, raw
    if spec.common.data.dataset == "moons":
        return raw, MoonsDataset._expand_xy_features(raw, logical_width)
    raise ValueError(
        "Expected linspace evaluation to use a two-dimensional logical input "
        "or the legacy expanded-moons representation. "
        f"Provided value: dataset={spec.common.data.dataset!r}, "
        f"logical_input_dim={logical_width!r}."
    )


def _classification_metrics(result: Any) -> dict[str, Any]:
    cost = result.probe_value("mean_cost")
    error = result.probe_value("mean_error")
    mean_error = error["mean"]
    return {
        "examples": result.example_count,
        "batches": result.batch_count,
        "mean_cost": cost["mean"],
        "mean_error": mean_error,
        "accuracy": None if mean_error is None else 1.0 - mean_error,
    }


@dataclass
class _TrainingMetrics:
    cost_sum: float = 0.0
    error_sum: float = 0.0
    count: int = 0

    def __call__(self, event: Any) -> None:
        if not isinstance(event, FreePhaseEvent):
            return
        cost = event.components.cost_fn.eval()
        error = event.components.cost_fn.error_fn()
        count = int(cost.numel())
        self.cost_sum += float(cost.sum().item())
        self.error_sum += float(error.sum().item())
        self.count += count

    def result(self) -> dict[str, Any]:
        return {
            "examples": self.count,
            "mean_cost": self.cost_sum / self.count if self.count else None,
            "mean_error": self.error_sum / self.count if self.count else None,
            "accuracy": (
                1.0 - self.error_sum / self.count if self.count else None
            ),
        }


class _BatchCapture:
    def __init__(self) -> None:
        self._inputs: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._indices: list[np.ndarray] = []

    def __call__(self, event: Any) -> None:
        self._inputs.append(_as_numpy(event.batch.inputs))
        self._labels.append(_as_numpy(event.batch.targets))
        if event.batch.indices is not None:
            self._indices.append(_as_numpy(event.batch.indices))

    def result(self) -> dict[str, np.ndarray]:
        values = {
            "inputs": _concatenate(self._inputs),
            "labels": _concatenate(self._labels),
        }
        if self._indices:
            values["indices"] = _concatenate(self._indices)
        return values


def _modifier_provenance(
    spec: TrainSpec,
    runtime: TrainRuntime,
) -> dict[str, Any]:
    parameters = dict(spec.settings.weight_modifier.parameters)
    return {
        "type": spec.settings.weight_modifier.type,
        "parameters": parameters,
        "active_during_training": runtime.modifier is not None,
        "resolved_seed": runtime.modifier_resolved_seed,
        "noisy_evaluation_requested": bool(
            parameters.get("noisy_evaluation", False)
        ),
        "noisy_evaluation_executed": runtime.noisy_evaluation,
    }


def _selection_from_progress(
    progress: Mapping[str, Any],
) -> tuple[float, int, float | None, float | None]:
    metric = progress.get("selected_metric")
    value = progress.get("selected_value")
    epoch = progress.get("selected_epoch")
    valid_value = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
    valid_epoch = (
        not isinstance(epoch, bool) and isinstance(epoch, int) and epoch >= 0
    )
    if metric != "validation.mean_cost" or not valid_value or not valid_epoch:
        raise ValueError(
            "Expected resume progress to describe a finite clean-validation "
            "selection with keys selected_metric, selected_value, and "
            f"selected_epoch. Provided value: {dict(progress)!r}."
        )
    error_fraction = _optional_fraction(
        progress.get("selected_error_fraction"),
        name="resume progress selected_error_fraction",
    )
    accuracy = _optional_fraction(
        progress.get("selected_accuracy"),
        name="resume progress selected_accuracy",
    )
    return float(value), epoch, error_fraction, accuracy


def _selection_progress(
    *,
    cost: float | None,
    epoch: int | None,
    error_fraction: float | None,
    accuracy: float | None,
) -> dict[str, Any]:
    if cost is None or epoch is None:
        raise RuntimeError(
            "Expected selected clean-validation cost and epoch before "
            "persisting progress. "
            f"Provided value: cost={cost!r}, epoch={epoch!r}."
        )
    return {
        "selected_metric": "validation.mean_cost",
        "selected_value": float(cost),
        "selected_epoch": int(epoch),
        "selected_error_fraction": error_fraction,
        "selected_accuracy": accuracy,
    }


def _epoch_report(
    *,
    epoch: int,
    completed_epoch: int,
    global_step: int,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    improved: bool,
    selected_epoch: int | None,
    selected_cost: float | None,
    selected_error: float | None,
    selected_accuracy: float | None,
) -> TrainingEpochReport:
    if selected_epoch is None or selected_cost is None:
        raise RuntimeError(
            "Expected clean validation to define selection fields before "
            "reporting an epoch. "
            f"Provided value: epoch={selected_epoch!r}, cost={selected_cost!r}."
        )
    validation_cost = validation.get("mean_cost")
    validation_error = validation.get("mean_error")
    validation_accuracy = validation.get("accuracy")
    if (
        validation_cost is None
        or validation_error is None
        or validation_accuracy is None
    ):
        raise RuntimeError(
            "Expected completed clean validation metrics to include cost, "
            "error fraction, and accuracy. "
            f"Provided value: {dict(validation)!r}."
        )
    return TrainingEpochReport(
        epoch_index=epoch,
        completed_epochs=completed_epoch,
        global_step=global_step,
        train_examples=int(train["examples"]),
        train_mean_cost=_optional_float(train.get("mean_cost")),
        train_error_fraction=_optional_fraction(
            train.get("mean_error"),
            name="training error fraction",
        ),
        train_accuracy=_optional_fraction(
            train.get("accuracy"),
            name="training accuracy",
        ),
        validation_examples=int(validation["examples"]),
        validation_mean_cost=float(validation_cost),
        validation_error_fraction=float(validation_error),
        validation_accuracy=float(validation_accuracy),
        selected_this_epoch=improved,
        selected_epoch_index=selected_epoch,
        selected_validation_cost=float(selected_cost),
        selected_validation_error_fraction=selected_error,
        selected_validation_accuracy=selected_accuracy,
    )


def _validated_observers(
    observers: Iterable[TrainingObserver],
) -> tuple[TrainingObserver, ...]:
    try:
        resolved = tuple(observers)
    except TypeError as exc:
        raise TypeError(
            "Expected observers to be an iterable of callables accepting one "
            f"TrainingEpochReport. Provided value: {observers!r}."
        ) from exc
    for index, observer in enumerate(resolved):
        if not callable(observer):
            raise TypeError(
                "Expected every observer to be callable with one "
                f"TrainingEpochReport. Provided value: observers[{index}]="
                f"{observer!r}."
            )
    return resolved


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_fraction(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    valid = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )
    if not valid:
        raise ValueError(
            f"Expected {name} to be a finite fraction in [0, 1] or null. "
            f"Provided value: {value!r}."
        )
    return float(value)


def _resume_metadata(
    spec: TrainSpec,
    runtime: TrainRuntime,
) -> dict[str, Any]:
    return {
        **_selected_model_metadata(spec, runtime),
        "resume_config_sha256": _resume_config_sha256(spec),
    }


def _amplification_index_report(runtime: TrainRuntime) -> list[dict[str, Any]]:
    """Record the model-local numerical topology of every dense edge."""

    report: list[dict[str, Any]] = []
    for interaction in runtime.stack.bundle.energy._interactions:
        if not isinstance(
            interaction,
            (DenseResistive, SignedDenseResistive),
        ):
            continue
        pre_index = int(interaction._logical_pre_index)
        post_index = int(interaction._logical_post_index)
        item: dict[str, Any] = {
            "resolved_pre_index": pre_index,
            "resolved_post_index": post_index,
        }
        if isinstance(interaction, SignedDenseResistive):
            item.update(
                {
                    "interaction": "differential_pair",
                    "voltage_amp": float(interaction._voltage_amp),
                    "current_amp": float(interaction._current_amp),
                    "forward_gain": float(interaction._forward_gain()),
                    "post_layer_metric": float(interaction._post_metric()),
                }
            )
        else:
            voltage_amp = float(interaction._voltage_amp)
            current_amp = float(interaction._current_amp)
            item.update(
                {
                    "interaction": "single_conductance",
                    "voltage_amp": voltage_amp,
                    "current_amp": current_amp,
                    "forward_gain": (
                        1.0 if pre_index == 0 else voltage_amp
                    ),
                    "post_layer_metric": (
                        (current_amp / voltage_amp) ** pre_index
                    ),
                }
            )
        report.append(item)
    return report


def _selected_model_metadata(
    spec: TrainSpec,
    runtime: TrainRuntime,
) -> dict[str, Any]:
    """Describe enough numerical semantics to consume a DRN as a teacher."""

    model = spec.common.model
    paired_outputs = model.dims[-1] == 2 * runtime.stack.num_classes
    return {
        "experiment_id": spec.experiment_id,
        "schema_version": spec.schema_version,
        "checkpoint_role": "supervised_drn",
        "architecture": "deep_resistive_network",
        "objective": (
            "supervised_paired_squared_error"
            if paired_outputs
            else "supervised_squared_error"
        ),
        "output_semantics": (
            "adjacent_pair_difference"
            if paired_outputs
            else "direct_output"
        ),
        "model": {
            "dims": list(model.dims),
            "logical_input_dim": runtime.stack.logical_input_dim,
            "num_classes": runtime.stack.num_classes,
            "input_gain": model.input_gain,
            "weight_gains": list(model.weight_gains),
            "weight_bounds": [model.weight_min, model.weight_max],
            "weight_init_mode": model.weight_init_mode,
            "include_biases": model.include_biases,
            "voltage_amp": model.voltage_amp,
            "current_amp": model.current_amp,
            "non_linearity": to_plain_data(model.non_linearity),
            "adapter": to_plain_data(model.adapter),
            "amplification_indexing": "model_local",
            "amplification_indices": _amplification_index_report(runtime),
        },
        "solver": to_plain_data(spec.common.solver),
        "dataset": {
            "name": spec.common.data.dataset,
            "validation_points": spec.common.data.validation_points,
            "data_seed": spec.common.runtime.data_seed,
        },
    }


def _resume_config_sha256(spec: TrainSpec) -> str:
    """Fingerprint every numerical setting except the target epoch horizon."""

    value = to_plain_data(spec)
    settings = value.get("settings")
    if not isinstance(settings, dict) or "num_epochs" not in settings:
        raise TypeError(
            "Expected a plain TrainSpec to contain settings.num_epochs. "
            f"Provided value: {value!r}."
        )
    settings = dict(settings)
    del settings["num_epochs"]
    value["settings"] = settings
    return content_hash(value)


def _limit_batches(
    loader: Iterable[Any],
    maximum: int | None,
) -> Iterable[Any]:
    return loader if maximum is None else islice(loader, maximum)


def _limit_examples(
    loader: Iterable[Any],
    maximum: int | None,
) -> Iterable[Any]:
    if maximum is None:
        return loader

    def limited():
        remaining = maximum
        for raw_batch in loader:
            if remaining <= 0:
                return
            if not isinstance(raw_batch, (tuple, list)) or len(raw_batch) not in (
                2,
                3,
            ):
                raise ValueError(
                    "Expected validation loader batches to be pairs or "
                    f"triples. Provided value: {raw_batch!r}."
                )
            batch_size = int(raw_batch[0].shape[0])
            take = min(batch_size, remaining)
            if take == batch_size:
                yield raw_batch
            else:
                yield tuple(item[:take] for item in raw_batch)
            remaining -= take

    return limited()


def _residual_metrics(
    values: Mapping[str, np.ndarray],
    *,
    tolerance: float | None,
) -> dict[str, Any]:
    per_layer = {
        name: _numeric_summary(array, tolerance=tolerance)
        for name, array in values.items()
    }
    # The first ordered layer is the clamped input and is not an equilibrium
    # residual.  Do not rely on legacy process-global layer-name counters.
    comparable = [
        np.asarray(array).reshape(-1)
        for index, array in enumerate(values.values())
        if index != 0 and np.asarray(array).size
    ]
    overall = (
        _numeric_summary(
            np.concatenate(comparable),
            tolerance=tolerance,
        )
        if comparable
        else None
    )
    return {"layers": per_layer, "overall": overall}


def _numeric_summary(
    value: Any,
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size and not np.isfinite(array).all():
        raise FloatingPointError(
            "Expected diagnostic values to be finite. "
            f"Provided value: {array!r}."
        )
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    result: dict[str, Any] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
    }
    if tolerance is not None:
        result.update(
            {
                "tolerance": float(tolerance),
                "pass_rate": float(np.mean(array <= tolerance)),
                "pass_all": bool(np.all(array <= tolerance)),
            }
        )
    return result


def _training_input_artifacts(request: Any) -> tuple[Mapping[str, Any], ...]:
    records = []
    for role in ("weights", "base_weights", "resume"):
        path = getattr(request, role)
        if path is not None:
            records.append(_input_artifact(role, path))
    device_data = getattr(request, "device_data", None)
    if device_data is not None:
        records.append(_input_artifact("device_data", device_data))
    return tuple(records)


def _validate_training_initialization(
    request: Any,
    spec: TrainSpec,
) -> None:
    sources = tuple(
        name
        for name in ("weights", "base_weights", "resume")
        if getattr(request, name) is not None
    )
    if len(sources) > 1:
        raise ValueError(
            "Expected at most one of --weights, --base-weights, or --resume. "
            f"Provided value: {sources!r}."
        )
    if (
        spec.settings.update_backend.type == "measured_cohort_b_lora"
        and not sources
    ):
        raise ValueError(
            "Expected measured_cohort_b_lora to initialize every new run "
            "from an explicit named cohort-A base checkpoint via "
            "--base-weights, or to use --resume. Provided value: neither."
        )
    has_adapter = spec.common.model.adapter.type in {
        "passive_low_rank",
        "digital_low_rank",
        "passive_layerwise_low_rank",
    }
    if has_adapter and len(sources) != 1:
        raise ValueError(
            "Expected low-rank adapter training to initialize from exactly "
            "one of --weights, --base-weights, or --resume. "
            f"Provided value: {sources!r}."
        )
    if not has_adapter and request.base_weights is not None:
        raise ValueError(
            "Expected --base-weights only when model.adapter.type is "
            "'passive_low_rank', 'digital_low_rank', or "
            "'passive_layerwise_low_rank'. Provided value: "
            f"{str(request.base_weights)!r}."
        )
    measured_backend = spec.settings.update_backend.type
    measured = measured_backend in {
        "measured_cohort_a",
        "measured_cohort_b",
        "measured_cohort_b_lora",
    }
    om_bounds = measured_backend == "ibm_om_fp32_bounds"
    device_data_backend = measured or om_bounds
    device_data = getattr(request, "device_data", None)
    if device_data_backend and device_data is None:
        raise ValueError(
            "Expected --device-data when modes.train.update_backend.type is "
            f"{measured_backend!r}. Provided value: None."
        )
    if not device_data_backend and device_data is not None:
        raise ValueError(
            "Expected --device-data only when modes.train.update_backend.type "
            "is a measured-cohort or IBM OM FP32-bounds backend. Provided "
            "value: "
            f"{str(device_data)!r}."
        )
    if om_bounds and request.resume is None and sources:
        raise ValueError(
            "Expected ibm_om_fp32_bounds to initialize every new run from "
            "its frozen per-cell uniform distribution, or to use --resume. "
            f"Provided value: {sources!r}."
        )
    if measured_backend == "measured_cohort_a" and (
        request.weights is not None or request.base_weights is not None
    ):
        raise ValueError(
            "Expected measured_cohort_a to start every new run at pulse index "
            "0, or to use --resume. Provided value: --weights/--base-weights."
        )
    if (
        measured_backend == "measured_cohort_b"
        and request.resume is None
        and request.weights is None
    ):
        raise ValueError(
            "Expected measured_cohort_b to initialize every new run from an "
            "explicit named cohort-A checkpoint via --weights, or to use "
            "--resume. Provided value: neither."
        )
    if measured_backend == "measured_cohort_b_lora":
        if request.weights is not None:
            raise ValueError(
                "Expected measured_cohort_b_lora new deployment to use "
                "--base-weights (or --resume), not a full --weights "
                f"checkpoint. Provided value: {str(request.weights)!r}."
            )
        if request.resume is None and request.base_weights is None:
            raise ValueError(
                "Expected measured_cohort_b_lora to initialize every new run "
                "from an explicit named cohort-A base checkpoint via "
                "--base-weights, or to use --resume. Provided value: neither."
            )


def _base_checkpoint_catalog(
    catalog: ParameterCatalog,
) -> ParameterCatalog:
    bindings = catalog.for_group("base", checkpointed_only=True)
    if not bindings:
        raise ValueError(
            "Expected the model catalog to contain at least one checkpointed "
            "base parameter. Provided value: empty base group."
        )
    return ParameterCatalog(bindings)


def _input_artifact(role: str, path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            "Expected input artifact path to name an existing file. "
            f"Provided value: {str(source)!r}."
        )
    return {
        "role": role,
        "path": str(source),
        "sha256": sha256_file(source),
    }


def _create_run_store(
    request: Any,
    spec: Any,
    *,
    input_artifacts: Iterable[Mapping[str, Any]],
    resume_capability: str,
    runtime_protocol: Mapping[str, Any],
) -> RunStore:
    resolved = to_plain_data(spec)
    resolved["runtime_protocol"] = dict(runtime_protocol)
    return RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=resolved,
        command=request.command,
        repo_root=Path(__file__).resolve().parents[2],
        input_artifacts=input_artifacts,
        resume_capability=resume_capability,
    )


def _expect_spec(value: Any, expected: type, *, mode: str) -> Any:
    if not isinstance(value, expected):
        raise TypeError(
            f"Expected {mode} request spec to be {expected.__name__}. "
            f"Provided value: {type(value).__name__}."
        )
    return value


def _atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _numpy_mapping(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {name: _as_numpy(value) for name, value in values.items()}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _concatenate(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(values, axis=0)


__all__ = [
    "TrainingEpochReport",
    "TrainingObserver",
    "TrainingOutcome",
    "execute_train",
    "import_legacy_checkpoint",
    "run_linspace",
    "run_train",
    "run_validate",
]
