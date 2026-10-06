"""Clean evaluation of explicit named weights on one ``small_drn.v1`` split."""

from __future__ import annotations

from typing import Any

from experiments.artifacts import ArtifactRecord, RunStore
from experiments.small_network.components import ValidationRuntime
from experiments.small_network.config import ValidateSpec
from experiments.small_network.reporting import (
    BatchCapture,
    atomic_npz,
    classification_metrics,
    limit_examples,
    numeric_summary,
    numpy_mapping,
    residual_metrics,
)
from training.core.engine import evaluate
from training.core.probes import (
    MeanCostProbe,
    MeanErrorProbe,
    ResidualInfinityNormProbe,
    SettledLayerStatesProbe,
    SolverIterationCountsProbe,
)


def effective_split(spec: ValidateSpec) -> str:
    if spec.settings.split == "train":
        return "train"
    if (
        spec.settings.split == "validation"
        and spec.common.data.validation_points is not None
    ):
        return "held_out_validation"
    return "held_out_test"


def execute_validation(
    spec: ValidateSpec,
    runtime: ValidationRuntime,
    store: RunStore,
) -> tuple[dict[str, Any], tuple[ArtifactRecord, ...]]:
    split = effective_split(spec)
    loader = {
        "train": runtime.data.train_loader,
        "held_out_validation": runtime.data.held_out_loader,
        "held_out_test": runtime.data.test_loader,
    }[split]
    limited = limit_examples(loader, spec.settings.sample_limit)
    capture = BatchCapture()
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
        **classification_metrics(result),
        "residuals": residual_metrics(
            result.probe_value("residual_inf_norm"),
            tolerance=spec.common.solver.tolerances.residual_current,
        ),
        "solver_iterations": numeric_summary(
            result.probe_value("solver_iteration_counts")
        ),
        "split_protocol": {
            "requested": spec.settings.split,
            "effective": split,
        },
        "evaluation": "clean",
    }

    inputs_path = store.run_dir / "artifacts" / "validation_inputs.npz"
    atomic_npz(inputs_path, **capture.result())
    residuals_path = (
        store.run_dir / "artifacts" / "validation_residual_currents.npz"
    )
    atomic_npz(
        residuals_path,
        **result.probe_value("residual_inf_norm"),
    )
    iterations_path = (
        store.run_dir / "artifacts" / "validation_iteration_counts.npz"
    )
    atomic_npz(
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
        atomic_npz(
            states_path,
            **numpy_mapping(result.probe_value("settled_layer_states")),
        )
        artifacts.append(store.artifact_record(states_path, kind="states"))
    return metrics, tuple(artifacts)
