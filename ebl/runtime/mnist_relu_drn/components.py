"""Numerical composition and mapping for teacher-initialized DRN KD."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from networks.teachers.mnist import BiasFreeReluTeacher
from ebl.schema.mnist_relu_drn import (
    MEASURED_BACKENDS,
    StudentTrainSpec,
)
from networks.resistive.solver import CustomQuadraticMinimizer, MinimizerSettings
from networks.function.interaction import Function
from networks.function.network import Network
from networks.resistive.builders import ModelBundle, ParameterCatalog, build_deep_resistive_energy
from hardware.config import parse_device_programming_config
from networks.variable.parameter import Bias, DenseWeight
from methods.measured_trace import (
    MeasuredCohortAOptimizer,
    MeasuredCohortAOnePulseDownOptimizer,
    MeasuredCohortBOptimizer,
)
from methods.program_verify import ProgramVerifyOptimizer
from methods.teacher_batches import teacher_batches
from methods.core.engine import EvaluationComponents, evaluate
from methods.core.probes import PerBatchProbe
from methods.core.sgd import Backprop
from methods.sign_sgd import SignSGD


from networks.resistive.mnist import paired_scores, build_student_network


from methods.objectives import TeacherKLDivergence


@dataclass(frozen=True)
class StudentStack:
    bundle: ModelBundle
    network: Network
    minimizer: Any
    training_minimizer: Any
    cost: Any
    differentiator: Backprop
    optimizer: Any
    device: torch.device


from networks.teachers.drn import BoundedDrnTeacher


from networks.resistive.mnist import _minimizer


def _logical_optimizer(
    catalog: ParameterCatalog,
    *,
    encoding: str,
    rates: tuple[float, ...],
    sign_only: bool = False,
) -> torch.optim.Optimizer:
    bindings = catalog.trainable
    if encoding == "single":
        dense = tuple(
            binding
            for binding in bindings
            if isinstance(binding.parameter, DenseWeight)
        )
        biases = tuple(
            binding
            for binding in bindings
            if isinstance(binding.parameter, Bias)
        )
        if (
            len(dense) != 2
            or len(biases) not in {0, 1}
            or len(bindings) != len(dense) + len(biases)
            or len(rates) != len(bindings)
        ):
            raise ValueError(
                "Expected encoding='single' to expose two dense weights, "
                "zero or one hidden bias, and one learning rate per stable "
                "catalog binding. Provided value: "
                f"bindings={len(bindings)}, dense={len(dense)}, "
                f"biases={len(biases)}, rates={len(rates)}."
            )
        groups = [
            {"params": [binding.state], "lr": rates[index]}
            for index, binding in enumerate(bindings)
        ]
    else:
        if (
            len(bindings) != 4
            or any(
                not isinstance(binding.parameter, DenseWeight)
                for binding in bindings
            )
            or len(rates) != 2
        ):
            raise ValueError(
                "Expected encoding='differential' to expose four dense "
                "conductance tensors and two logical learning rates. "
                f"Provided value: bindings={len(bindings)}, rates={len(rates)}."
            )
        groups = [
            {
                "params": [bindings[2 * index].state, bindings[2 * index + 1].state],
                "lr": rates[index],
            }
            for index in range(2)
        ]
    if sign_only:
        return SignSGD(groups, lr=0.0)
    return torch.optim.SGD(groups, momentum=0.0, weight_decay=0.0)


def measured_optimizer_type(update_backend_type: str) -> type:
    """Resolve the explicitly registered measured optimizer implementation."""

    implementations = {
        "measured_cohort_a": MeasuredCohortAOptimizer,
        "measured_cohort_a_sign_sgd": MeasuredCohortAOptimizer,
        "measured_cohort_a_one_pulse_down": (
            MeasuredCohortAOnePulseDownOptimizer
        ),
        "measured_cohort_b": MeasuredCohortBOptimizer,
    }
    if update_backend_type not in implementations:
        raise ValueError(
            "Expected a registered MNIST measured update backend. Provided "
            f"value: {update_backend_type!r}."
        )
    return implementations[update_backend_type]


def build_student_stack(
    spec: Any,
    *,
    device_data_path=None,
    enable_measured: bool = True,
) -> StudentStack:
    bundle, network, inference, training, device = build_student_network(spec.model, spec.solver, spec.runtime.device)
    cost = TeacherKLDivergence(bundle.energy.layers()[-1])
    differentiator = Backprop(
        bundle.energy.params(),
        list(bundle.energy.layers()[1:]),
        cost,
        training,
    )
    rates = getattr(
        spec.settings,
        "learning_rates",
        (0.0,) * (3 if spec.model.include_biases else 2),
    )
    optimizer: Any = _logical_optimizer(
        bundle.catalog,
        encoding=spec.model.encoding,
        rates=rates,
        sign_only=(
            getattr(spec.settings, "update_backend", None) is not None
            and spec.settings.update_backend.type
            == "measured_cohort_a_sign_sgd"
        ),
    )
    update_backend = getattr(spec.settings, "update_backend", None)
    if (
        enable_measured
        and update_backend is not None
        and update_backend.type in MEASURED_BACKENDS
    ):
        if device_data_path is None:
            raise ValueError(
                f"Expected --device-data for {update_backend.type}. "
                "Provided value: None."
            )
        optimizer_type = measured_optimizer_type(update_backend.type)
        optimizer = optimizer_type(
            optimizer,
            bundle.catalog,
            update_backend.parameters,
            device_data_path,
        )
    elif (
        enable_measured
        and update_backend is not None
        and update_backend.type == "program_verify"
    ):
        optimizer = ProgramVerifyOptimizer(
            optimizer,
            bundle.catalog,
            parse_device_programming_config(
                update_backend.parameters["device"],
                path=(
                    "config.modes.train.update_backend.parameters.device"
                ),
            ),
        )
    return StudentStack(
        bundle=bundle,
        network=network,
        minimizer=inference,
        training_minimizer=training,
        cost=cost,
        differentiator=differentiator,
        optimizer=optimizer,
        device=device,
    )


from networks.resistive.mnist import signed_dual_rail_lift


from networks.resistive.mnist import signed_differential_lift


from networks.resistive.mnist import collapse_clamped_dual_rail_input


from networks.resistive.mnist import _normalized


from networks.resistive.mnist import mapped_conductances


from networks.resistive.mnist import apply_targets


from methods.calibration.mnist import fit_positive_logit_gain


from methods.calibration.mnist import collect_calibration


from methods.calibration.mnist import select_mapping_and_gain


from networks.resistive.mnist import conductance_statistics


__all__ = [
    "BoundedDrnTeacher",
    "StudentStack",
    "TeacherKLDivergence",
    "apply_targets",
    "build_student_stack",
    "collect_calibration",
    "conductance_statistics",
    "fit_positive_logit_gain",
    "mapped_conductances",
    "measured_optimizer_type",
    "paired_scores",
    "select_mapping_and_gain",
    "collapse_clamped_dual_rail_input",
    "signed_differential_lift",
    "signed_dual_rail_lift",
]
