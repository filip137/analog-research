"""Command lifecycles for ``small_drn.v1``.

Each public function owns one command: it validates the request, creates the
RunStore, and records completion or failure.  The numerical bodies live
beside it:

- ``initialization``: the train run's one start source (resume, weights,
  base weights, or the seeded model);
- ``deployment``: one-time device programming of a new run;
- ``backends``: the hardware-aware modifier and update backend;
- ``training``: the epoch loop hooks and train results;
- ``validate`` and ``linspace``: the fixed-weight diagnostics.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import torch

from experiments.artifacts import (
    ArtifactRecord,
    RunStore,
    atomic_write_json,
    content_hash,
    sha256_file,
)
from experiments.lifecycle import run_phase
from experiments.schema import to_plain_data
from experiments.small_network.backends import configured_resume_capability
from experiments.small_network.components import (
    build_evaluation_runtime,
    build_model_stack,
    build_train_runtime,
    build_validation_runtime,
    seed_runtime,
)
from experiments.small_network.config import (
    LinspaceSpec,
    SmallDrnConfig,
    TrainSpec,
    ValidateSpec,
)
from experiments.small_network.initialization import (
    base_checkpoint_catalog,
    validate_training_initialization,
)
from experiments.small_network.linspace import execute_linspace
from experiments.small_network.reporting import (
    TrainingEpochReport,
    TrainingObserver,
    TrainingOutcome,
    optional_float,
    validated_observers,
)
from experiments.small_network.training import execute_training
from experiments.small_network.validate import effective_split, execute_validation
from training.checkpoint import (
    LEGACY_BASE_ONLY,
    LEGACY_FULL,
    load_legacy_positional_weights,
    load_named_weights,
    save_named_weights,
)
from training.measured_trace import MeasuredTraceOptimizer

if TYPE_CHECKING:
    from ebl.cli import (
        ImportLegacyCheckpointRequest,
        LinspaceRequest,
        TrainRequest,
        ValidateRequest,
    )


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
    validate_training_initialization(request, spec)
    active_observers = validated_observers(observers)
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
    metrics: dict[str, Any] = {}
    last_epoch_report: TrainingEpochReport | None = None

    def phase() -> tuple[dict[str, Any], tuple[ArtifactRecord, ...]]:
        nonlocal metrics, selection_report, selection_path, last_epoch_report
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
        metrics, artifacts, last_epoch_report = execute_training(
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
        return metrics, artifacts

    result_path = run_phase(store, phase)
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
        selected_validation_error_fraction=optional_float(
            selected.get("error_fraction")
        ),
        selected_validation_accuracy=optional_float(
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
        metrics, artifacts = execute_linspace(spec, runtime, store)
        store.append_metric({"mode": "linspace", **metrics})
        store.complete(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        store.fail(exc)
        raise
    return 0


def run_validate(request: "ValidateRequest") -> int:
    """Evaluate explicit named weights on one configured dataset split."""

    spec = _expect_spec(request.spec, ValidateSpec, mode="validate")
    store = _create_run_store(
        request,
        spec,
        input_artifacts=(_input_artifact("weights", request.weights),),
        resume_capability="unsupported",
        runtime_protocol={
            "split_requested": spec.settings.split,
            "split_effective": effective_split(spec),
            "evaluation": "clean",
        },
    )
    try:
        seed_runtime(spec.common.runtime.seed)
        runtime = build_validation_runtime(spec.common)
        load_named_weights(request.weights, runtime.stack.bundle.catalog)
        metrics, artifacts = execute_validation(spec, runtime, store)
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
        else base_checkpoint_catalog(stack.bundle.catalog)
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
