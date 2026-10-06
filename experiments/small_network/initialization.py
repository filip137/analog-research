"""The logical start state of a ``small_drn.v1`` train run.

A run starts from exactly one explicit source: an epoch-boundary resume
checkpoint, full named weights, named base weights (adapter runs), or the
seeded model itself.  This module validates that choice against the update
backend and loads it; programming the loaded state onto a device is
:mod:`deployment`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.lifecycle import BestCheckpoint
from experiments.small_network.components import TrainRuntime
from experiments.small_network.config import TrainSpec
from experiments.small_network.provenance import (
    resume_config_sha256,
    selection_from_progress,
)
from model.resistive.builders import ParameterCatalog
from training.checkpoint import load_epoch_boundary_checkpoint, load_named_weights


@dataclass(frozen=True)
class StartState:
    epoch: int
    global_step: int


def validate_training_initialization(
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


def base_checkpoint_catalog(
    catalog: ParameterCatalog,
) -> ParameterCatalog:
    bindings = catalog.for_group("base", checkpointed_only=True)
    if not bindings:
        raise ValueError(
            "Expected the model catalog to contain at least one checkpointed "
            "base parameter. Provided value: empty base group."
        )
    return ParameterCatalog(bindings)


def load_start_state(
    request: Any,
    spec: TrainSpec,
    runtime: TrainRuntime,
    best: BestCheckpoint,
) -> StartState:
    """Load the run's one start source; a resume also restores ``best``."""

    catalog = runtime.stack.bundle.catalog
    if request.resume is not None:
        return _resume(request, spec, runtime, best)
    if request.weights is not None:
        load_named_weights(request.weights, catalog)
    elif request.base_weights is not None:
        load_named_weights(request.base_weights, base_checkpoint_catalog(catalog))
    return StartState(epoch=0, global_step=0)


def _resume(
    request: Any,
    spec: TrainSpec,
    runtime: TrainRuntime,
    best: BestCheckpoint,
) -> StartState:
    resumed = load_epoch_boundary_checkpoint(
        request.resume,
        catalog=runtime.stack.bundle.catalog,
        optimizer=runtime.optimizer,
        modifier=runtime.modifier,
        scheduler=None,
        runtime_state=runtime.runtime_state,
        dataloader_generators=runtime.data.dataloader_generators,
    )
    if resumed.resume_capability != runtime.resume_capability:
        raise ValueError(
            "Expected resume checkpoint capability to match this runtime. "
            f"Provided value: checkpoint={resumed.resume_capability!r}, "
            f"runtime={runtime.resume_capability!r}."
        )
    expected_resume_config = resume_config_sha256(spec)
    provided_resume_config = resumed.metadata.get("resume_config_sha256")
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
    (
        selected_cost,
        selected_epoch,
        selected_error,
        selected_accuracy,
    ) = selection_from_progress(resumed.progress_state)
    best.restore(
        epoch=selected_epoch,
        selection={
            "mean_cost": selected_cost,
            "mean_error": selected_error,
            "accuracy": selected_accuracy,
        },
        payload=resumed.selected_weights,
    )
    return StartState(epoch=resumed.epoch, global_step=resumed.global_step)
