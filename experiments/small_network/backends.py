"""Hardware-aware modifier and update-backend selection for ``small_drn.v1``.

``modes.train.weight_modifier`` chooses what the forward pass sees during
training (hardware-aware noise); ``modes.train.update_backend`` chooses how a
computed gradient becomes a parameter update (digital, Tiki-Taka, Adam,
program-verify, measured cohort traces, or IBM OM FP32 bounds).  Both are
resolved here and nowhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from experiments.small_network.config import TrainSpec
from model.resistive.device_config import parse_device_programming_config
from training.add_normal import AddNormalConfig, build_add_normal_modifier
from training.adam import AdamOptimizer
from training.core.modifier import ParameterModifier
from training.core.optimizers import build_optimizer
from training.ibm_om_fp32_bounds import IbmOmFp32BoundsOptimizer
from training.measured_trace import (
    MeasuredCohortAOptimizer,
    MeasuredCohortBLoRAOptimizer,
    MeasuredCohortBOptimizer,
)
from training.program_verify import ProgramVerifyOptimizer
from training.tiki_taka import parse_update_pipeline

if TYPE_CHECKING:
    from experiments.small_network.components import ModelStack


_MEASURED_OPTIMIZERS = {
    "measured_cohort_a": MeasuredCohortAOptimizer,
    "measured_cohort_b": MeasuredCohortBOptimizer,
    "measured_cohort_b_lora": MeasuredCohortBLoRAOptimizer,
}


def build_weight_modifier(
    spec: TrainSpec,
    stack: "ModelStack",
) -> tuple[ParameterModifier | None, bool]:
    """Return the training modifier and whether evaluation also runs noisy."""

    settings = spec.settings.weight_modifier
    if settings.type == "none":
        return None, False
    if settings.type != "add_normal":
        raise NotImplementedError(
            "Expected the hardware-aware branch to use weight modifier "
            "'none' or 'add_normal'. Provided value: "
            f"{settings.type!r}."
        )
    config = AddNormalConfig(
        std_dev=float(settings.parameters["std_dev"]),
        seed=settings.parameters["seed"],
        noisy_evaluation=bool(settings.parameters["noisy_evaluation"]),
        scale_mode=str(settings.parameters["scale_mode"]),
    )
    modifier = build_add_normal_modifier(
        stack.bundle.catalog.trainable_parameters,
        config,
        run_seed=spec.common.runtime.seed,
    )
    return modifier, modifier is not None and config.noisy_evaluation


def build_update_optimizer(
    spec: TrainSpec,
    stack: "ModelStack",
    learning_rates: Sequence[float],
    *,
    device_data_path: Path | None,
) -> tuple[Any, str]:
    """Return the update backend and its resume capability."""

    backend = spec.settings.update_backend
    energy = stack.bundle.energy
    cost_fn = stack.cost_fn
    program_verify_config = None
    measured_parameters = None
    om_bounds_parameters = None
    if backend.type == "program_verify":
        program_verify_config = parse_device_programming_config(
            backend.parameters["device"],
            path="config.modes.train.update_backend.parameters.device",
        )
        parsed_pipeline = None
    elif backend.type in _MEASURED_OPTIMIZERS:
        if device_data_path is None:
            raise ValueError(
                "Expected --device-data for update backend "
                f"{backend.type!r}. Provided value: None."
            )
        measured_parameters = backend.parameters
        parsed_pipeline = None
    elif backend.type == "direct_adam":
        parsed_pipeline = None
    elif backend.type == "ibm_om_fp32_bounds":
        if device_data_path is None:
            raise ValueError(
                "Expected --device-data for update backend "
                "'ibm_om_fp32_bounds'. Provided value: None."
            )
        om_bounds_parameters = backend.parameters
        parsed_pipeline = None
    else:
        parsed_pipeline = parse_update_pipeline(
            {"type": backend.type, **dict(backend.parameters)}
        )
    if (
        spec.common.model.adapter.type
        in {"passive_low_rank", "passive_layerwise_low_rank"}
        and parsed_pipeline is not None
        and parsed_pipeline.aihwkit_preset is not None
    ):
        raise ValueError(
            "Expected low-rank energy-adapter training to use direct updates "
            "or the ideal-tensor Tiki-Taka backend. Provided value: "
            f"aihwkit_preset={parsed_pipeline.aihwkit_preset!r}."
        )
    if backend.type == "direct_adam":
        optimizer = _adam(energy, cost_fn, learning_rates, backend.parameters)
    elif om_bounds_parameters is not None and (
        om_bounds_parameters["optimizer"]["type"] == "adam"
    ):
        optimizer = _adam(
            energy,
            cost_fn,
            learning_rates,
            om_bounds_parameters["optimizer"]["parameters"],
        )
    else:
        sgd_parameters = (
            om_bounds_parameters["optimizer"]["parameters"]
            if om_bounds_parameters is not None
            else {}
        )
        optimizer = build_optimizer(
            energy,
            cost_fn,
            learning_rates,
            update_pipeline=parsed_pipeline,
            momentum=float(sgd_parameters.get("momentum", 0.0)),
            weight_decay=float(sgd_parameters.get("weight_decay", 0.0)),
        )
    if program_verify_config is not None:
        optimizer = ProgramVerifyOptimizer(
            optimizer,
            stack.bundle.catalog,
            program_verify_config,
        )
    elif measured_parameters is not None:
        optimizer = _MEASURED_OPTIMIZERS[backend.type](
            optimizer,
            stack.bundle.catalog,
            measured_parameters,
            device_data_path,
        )
    elif om_bounds_parameters is not None:
        optimizer = IbmOmFp32BoundsOptimizer(
            optimizer,
            stack.bundle.catalog,
            om_bounds_parameters,
            device_data_path,
        )
    resume_capability = (
        "stateful_nondeterministic"
        if parsed_pipeline is not None
        and parsed_pipeline.aihwkit_preset is not None
        else "exact"
    )
    return optimizer, resume_capability


def configured_resume_capability(spec: TrainSpec) -> str:
    """Resolve capability before importing optional numerical backends."""

    if spec.settings.update_backend.type != "tiki_taka":
        return "exact"
    preset = spec.settings.update_backend.parameters.get("aihwkit_preset")
    return "stateful_nondeterministic" if preset is not None else "exact"


def _adam(
    energy: Any,
    cost_fn: Any,
    learning_rates: Sequence[float],
    parameters: Any,
) -> AdamOptimizer:
    return AdamOptimizer(
        energy,
        cost_fn,
        learning_rates,
        betas=(float(parameters["beta1"]), float(parameters["beta2"])),
        eps=float(parameters["epsilon"]),
        weight_decay=float(parameters["weight_decay"]),
        amsgrad=bool(parameters["amsgrad"]),
    )
