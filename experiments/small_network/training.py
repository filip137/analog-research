"""The ``small_drn.v1`` train phase: start state, deployment, epochs, results.

Selection is the strict global-best clean validation ``mean_cost``.  Training
keeps ``reset_input=False``, so each epoch's first minibatch starts from the
layer states its preceding clean validation left; the optional noisy
validation therefore runs first (``noisy_evaluation_order: before_clean``).
"""

from __future__ import annotations

from typing import Any

from experiments.artifacts import ArtifactRecord, RunStore, atomic_write_json
from experiments.lifecycle import BestCheckpoint, EpochResult, run_epochs
from experiments.lifecycle.train_phase import lower
from experiments.small_network.components import TrainRuntime
from experiments.small_network.config import TrainSpec
from experiments.small_network.deployment import deploy
from experiments.small_network.initialization import load_start_state
from experiments.small_network.provenance import (
    modifier_provenance,
    resume_metadata,
    selected_model_metadata,
    selection_progress,
)
from experiments.small_network.reporting import (
    TrainingEpochReport,
    TrainingMetrics,
    TrainingObserver,
    epoch_report,
    held_out_metrics,
    limit_batches,
)
from training.checkpoint import encode_named_weights, save_epoch_boundary_checkpoint
from training.core.engine import train_epoch
from training.core.guards import FiniteGradientGuard
from training.ibm_om_fp32_bounds import IbmOmFp32BoundsOptimizer
from training.measured_trace import MeasuredTraceOptimizer
from training.program_verify import ProgramVerifyOptimizer


def execute_training(
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
    best = BestCheckpoint(weights_path, catalog, better=lower("mean_cost"))
    last_epoch_report: TrainingEpochReport | None = None

    start = load_start_state(request, spec, runtime, best)
    deployment = deploy(request, spec, runtime, store)

    def train(epoch: int, step: int) -> tuple[dict[str, Any], int]:
        train_metrics = TrainingMetrics()
        trained = train_epoch(
            runtime.training_components,
            limit_batches(
                runtime.data.train_loader,
                spec.settings.max_batches,
            ),
            modifier=runtime.modifier,
            event_handlers=(train_metrics, FiniteGradientGuard()),
            epoch=epoch,
            start_global_step=step,
            reset_input=False,
        )
        return train_metrics.result(), trained.next_global_step

    def validate(
        epoch: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        noisy = None
        if runtime.noisy_evaluation:
            noisy = held_out_metrics(
                runtime,
                maximum_batches=spec.settings.max_validation_batches,
                modifier=runtime.modifier,
                epoch=epoch,
                split="validation_noisy",
            )
        clean = held_out_metrics(
            runtime,
            maximum_batches=spec.settings.max_validation_batches,
            modifier=None,
            epoch=epoch,
            split="validation",
        )
        if clean["mean_cost"] is None:
            raise ValueError(
                "Expected clean validation to evaluate at least one example. "
                f"Provided value: {clean['examples']} examples."
            )
        return noisy, clean

    def selection(
        validation: tuple[dict[str, Any] | None, dict[str, Any]],
    ) -> dict[str, Any]:
        clean = validation[1]
        return {
            "mean_cost": clean["mean_cost"],
            "mean_error": clean["mean_error"],
            "accuracy": clean["accuracy"],
        }

    def payload(
        epoch: int,
        validation: tuple[dict[str, Any] | None, dict[str, Any]],
    ) -> dict[str, Any]:
        chosen = selection(validation)
        return encode_named_weights(
            catalog,
            metadata={
                **selected_model_metadata(spec, runtime),
                "selection_metric": "validation.mean_cost",
                "selection_value": chosen["mean_cost"],
                "selection_epoch": epoch,
                "selection_error_fraction": chosen["mean_error"],
                "selection_accuracy": chosen["accuracy"],
            },
        )

    def save_resume(completed: int, step: int, _last: EpochResult | None) -> None:
        chosen = best.selection or {}
        save_epoch_boundary_checkpoint(
            resume_path,
            catalog=catalog,
            epoch=completed,
            global_step=step,
            optimizer=runtime.optimizer,
            modifier=runtime.modifier,
            scheduler=None,
            runtime_state=runtime.runtime_state,
            progress_state=selection_progress(
                cost=chosen.get("mean_cost"),
                epoch=best.epoch,
                error_fraction=chosen.get("mean_error"),
                accuracy=chosen.get("accuracy"),
            ),
            selected_weights=best.payload,
            dataloader_generators=runtime.data.dataloader_generators,
            resume_capability=runtime.resume_capability,
            metadata=resume_metadata(spec, runtime),
        )

    def record(result: EpochResult) -> dict[str, Any]:
        noisy, clean = result.validation
        epoch_metric = {
            "mode": "train",
            "epoch": result.epoch,
            "completed_epochs": result.completed_epochs,
            "global_step": result.global_step,
            "train": result.train,
            "validation": clean,
            "validation_noisy": noisy,
            "selected": result.improved,
            "selected_epoch": best.epoch,
            "selected_cost": best.selection["mean_cost"],
        }
        if isinstance(runtime.optimizer, MeasuredTraceOptimizer):
            epoch_metric["measured_projection"] = (
                runtime.optimizer.programming_report
            )
        return epoch_metric

    def report(result: EpochResult) -> None:
        nonlocal last_epoch_report
        last_epoch_report = epoch_report(
            epoch=result.epoch,
            completed_epoch=result.completed_epochs,
            global_step=result.global_step,
            train=result.train,
            validation=result.validation[1],
            improved=result.improved,
            selected_epoch=best.epoch,
            selected_cost=best.selection["mean_cost"],
            selected_error=best.selection["mean_error"],
            selected_accuracy=best.selection["accuracy"],
        )
        for observer in observers:
            observer(last_epoch_report)

    last, completed_epoch, global_step = run_epochs(
        store=store,
        start_epoch=start.epoch,
        num_epochs=spec.settings.num_epochs,
        global_step=start.global_step,
        log_every=spec.settings.log_every,
        train_epoch=train,
        validate=validate,
        best=best,
        selection=selection,
        payload=payload,
        save_resume=save_resume,
        resume_path=resume_path,
        record=record,
        after_epoch=(report,),
        missing_selection=(
            "Expected training or its resume checkpoint to provide selected "
            "clean-evaluation weights. Provided value: null."
        ),
    )
    last_noisy_validation, last_validation = (
        (None, None) if last is None else last.validation
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
            "value": best.selection["mean_cost"],
            "epoch": best.epoch,
            "error_fraction": best.selection["mean_error"],
            "accuracy": best.selection["accuracy"],
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
            "parameter_modifier": modifier_provenance(spec, runtime),
        },
        "resume_capability": runtime.resume_capability,
        "device_programming": _plain(deployment.device_programming),
        "update_programming": (
            runtime.optimizer.programming_report
            if isinstance(
                runtime.optimizer,
                (ProgramVerifyOptimizer, MeasuredTraceOptimizer),
            )
            else None
        ),
        "om_fp32_bounds": om_bounds_report,
        "initial_validation": _plain(deployment.initial_validation),
        "pre_deployment_validation": _plain(
            deployment.pre_deployment_validation
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


def _plain(value: Any) -> dict[str, Any] | None:
    return None if value is None else dict(value)
