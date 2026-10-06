"""Settled-state diagnostics of named weights on a 2-D ``small_drn.v1`` grid."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.artifacts import ArtifactRecord, RunStore
from experiments.small_network.components import EvaluationRuntime
from experiments.small_network.config import LinspaceSpec
from experiments.small_network.reporting import (
    atomic_npz,
    numeric_summary,
    numpy_mapping,
    residual_metrics,
)
from labs.datasets import MoonsDataset
from training.core.engine import evaluate
from training.core.probes import (
    ResidualInfinityNormProbe,
    SettledLayerStatesProbe,
    SolverIterationCountsProbe,
)


def execute_linspace(
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
    atomic_npz(inputs_path, inputs=raw_points.detach().cpu().numpy())
    residuals = result.probe_value("residual_inf_norm")
    atomic_npz(residuals_path, **residuals)
    iteration_counts = result.probe_value("solver_iteration_counts")
    atomic_npz(
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
        atomic_npz(
            states_path,
            **numpy_mapping(result.probe_value("settled_layer_states")),
        )
        artifacts.append(store.artifact_record(states_path, kind="states"))

    metrics = {
        "examples": result.example_count,
        "batches": result.batch_count,
        "grid": {
            "minimum": spec.settings.minimum,
            "maximum": spec.settings.maximum,
            "samples_per_axis": spec.settings.samples,
        },
        "residuals": residual_metrics(
            residuals,
            tolerance=spec.common.solver.tolerances.residual_current,
        ),
        "solver_iterations": numeric_summary(iteration_counts),
        "input_protocol": {
            "configured_gain": spec.common.model.input_gain,
            "effective_gain": 1.0,
            "assignment_mode": "train",
            "legacy_test_assignment_effect": "no_op",
        },
    }
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
