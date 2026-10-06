"""Checkpoint metadata and the resume-progress codec for ``small_drn.v1``.

Both the code that writes an epoch-boundary checkpoint and the code that
resumes from one use these records, so they live apart from either.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from experiments.artifacts import content_hash
from experiments.schema import to_plain_data
from experiments.small_network.components import TrainRuntime
from experiments.small_network.config import TrainSpec
from experiments.small_network.reporting import optional_fraction
from model.resistive.interaction import DenseResistive, SignedDenseResistive


def amplification_index_report(runtime: TrainRuntime) -> list[dict[str, Any]]:
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


def selected_model_metadata(
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
            "amplification_indices": amplification_index_report(runtime),
        },
        "solver": to_plain_data(spec.common.solver),
        "dataset": {
            "name": spec.common.data.dataset,
            "validation_points": spec.common.data.validation_points,
            "data_seed": spec.common.runtime.data_seed,
        },
    }


def resume_config_sha256(spec: TrainSpec) -> str:
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


def resume_metadata(
    spec: TrainSpec,
    runtime: TrainRuntime,
) -> dict[str, Any]:
    return {
        **selected_model_metadata(spec, runtime),
        "resume_config_sha256": resume_config_sha256(spec),
    }


def selection_progress(
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


def selection_from_progress(
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
    error_fraction = optional_fraction(
        progress.get("selected_error_fraction"),
        name="resume progress selected_error_fraction",
    )
    accuracy = optional_fraction(
        progress.get("selected_accuracy"),
        name="resume progress selected_accuracy",
    )
    return float(value), epoch, error_fraction, accuracy


def modifier_provenance(
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
