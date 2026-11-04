"""Numerical composition for the versioned small-DRN experiment.

This module translates ``small_drn.v1`` configuration objects into legacy
datasets, energy functions, minimizers and gradient estimators; the
hardware-aware modifier and the update backend come from ``backends.py``.
Command lifecycles live in ``runtime.py``; the generic training loop remains
configuration-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ebl.runtime.small_network.backends import (
    build_update_optimizer,
    build_weight_modifier,
)
from ebl.schema.small_network import CommonSettings, TrainSpec
from networks.resistive.solver import (
    CustomAnderssonMinimizer,
    CustomQuadraticMinimizer,
    MinimizerSettings,
)
from networks.data.reference import DigitsDataset, MnistDataset, MoonsDataset, YinYangDataset
from networks.function.cost import SquaredError, SquaredErrorPairedOutputs
from networks.function.network import Network
from networks.resistive.builders import ModelBundle, build_deep_resistive_energy
from networks.resistive.digital_low_rank import DigitalLowRankReadout
from networks.variable.parameter import Bias
from methods.core.engine import EvaluationComponents, ExperimentComponents
from methods.core.modifier import ParameterModifier
from methods.core.sgd import (
    AugmentedFunction,
    Backprop,
    DirectReadoutGradient,
    EquilibriumProp,
)


from networks.resistive.small import _CLASS_COUNTS
from networks.resistive.small import _DTYPES


from networks.data.small import _FlattenedImageDataset


from networks.resistive.small import TypedNetwork


from networks.resistive.small import LayerStateCheckpoint


from networks.data.small import DataBundle


from networks.resistive.small import ModelStack


@dataclass(frozen=True)
class EvaluationRuntime:
    stack: ModelStack
    components: EvaluationComponents


@dataclass(frozen=True)
class ValidationRuntime(EvaluationRuntime):
    data: DataBundle


@dataclass(frozen=True)
class TrainRuntime:
    stack: ModelStack
    data: DataBundle
    training_components: ExperimentComponents
    evaluation_components: EvaluationComponents
    optimizer: Any
    modifier: ParameterModifier | None
    noisy_evaluation: bool
    runtime_state: LayerStateCheckpoint
    resume_capability: str

    @property
    def modifier_resolved_seed(self) -> int | None:
        seed = getattr(self.modifier, "resolved_seed", None)
        return None if seed is None else int(seed)


def seed_runtime(seed: int | None) -> None:
    """Match the legacy training entry point's process RNG initialization."""

    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


from networks.resistive.small import build_model_stack


from networks.data.small import build_data


from networks.data.small import _stratified_split_indices


def build_evaluation_runtime(common: CommonSettings) -> EvaluationRuntime:
    stack = build_model_stack(common)
    minimizer = _build_minimizer(
        common,
        stack,
        function=stack.bundle.energy,
        iterations=common.solver.inference_iterations,
    )
    _reset_minimizer_statistics(minimizer)
    return EvaluationRuntime(
        stack=stack,
        components=EvaluationComponents(
            network=stack.network,
            cost_fn=stack.cost_fn,
            energy_minimizer=minimizer,
        ),
    )


def build_validation_runtime(common: CommonSettings) -> ValidationRuntime:
    evaluation = build_evaluation_runtime(common)
    return ValidationRuntime(
        stack=evaluation.stack,
        components=evaluation.components,
        data=build_data(common),
    )


def build_train_runtime(
    spec: TrainSpec,
    *,
    device_data_path: Path | None = None,
) -> TrainRuntime:
    """Compose the authoritative engine with legacy numerical primitives."""

    stack = build_model_stack(spec.common)
    data = build_data(spec.common)
    energy = stack.bundle.energy
    cost_fn = stack.cost_fn
    modifier, noisy_evaluation = build_weight_modifier(spec, stack)

    inference_minimizer = _build_minimizer(
        spec.common,
        stack,
        function=energy,
        iterations=spec.common.solver.inference_iterations,
    )
    if spec.settings.algorithm == "ep":
        training_function = AugmentedFunction(energy, cost_fn)
        training_minimizer = _build_minimizer(
            spec.common,
            stack,
            function=training_function,
            iterations=spec.common.solver.training_iterations,
        )
        differentiator = EquilibriumProp(
            energy.params(),
            list(stack.free_layers),
            training_function,
            cost_fn,
            training_minimizer,
            variant="centered",
            nudging=spec.settings.nudging,
        )
    elif spec.settings.algorithm == "backprop":
        training_minimizer = _build_minimizer(
            spec.common,
            stack,
            function=energy,
            iterations=spec.common.solver.training_iterations,
        )
        differentiator = Backprop(
            energy.params(),
            list(stack.free_layers),
            cost_fn,
            training_minimizer,
        )
    elif spec.settings.algorithm == "digital":
        if spec.common.model.adapter.type != "digital_low_rank":
            raise ValueError(
                "Expected algorithm='digital' only with "
                "model.adapter.type='digital_low_rank'. Provided value: "
                f"adapter={spec.common.model.adapter.type!r}."
            )
        training_minimizer = _build_minimizer(
            spec.common,
            stack,
            function=energy,
            iterations=spec.common.solver.training_iterations,
        )
        differentiator = DirectReadoutGradient(cost_fn)
    else:  # pragma: no cover - rejected by the pure schema
        raise ValueError(
            "Expected config.modes.train.algorithm to be 'ep', 'backprop', "
            "or 'digital'. Provided value: "
            f"{spec.settings.algorithm!r}."
        )

    parameters = tuple(energy.params()) + tuple(cost_fn.params())
    learning_rates = _parameter_learning_rates(
        parameters,
        weight_rates=spec.settings.learning_rates,
        bias_rates=spec.settings.bias_learning_rates,
    )
    optimizer, resume_capability = build_update_optimizer(
        spec,
        stack,
        learning_rates,
        device_data_path=device_data_path,
    )
    _reset_minimizer_statistics(inference_minimizer)

    evaluation_components = EvaluationComponents(
        network=stack.network,
        cost_fn=cost_fn,
        energy_minimizer=inference_minimizer,
    )
    return TrainRuntime(
        stack=stack,
        data=data,
        training_components=ExperimentComponents(
            network=stack.network,
            cost_fn=cost_fn,
            energy_minimizer=inference_minimizer,
            parameters=parameters,
            differentiator=differentiator,
            optimizer=optimizer,
        ),
        evaluation_components=evaluation_components,
        optimizer=optimizer,
        modifier=modifier,
        noisy_evaluation=noisy_evaluation,
        runtime_state=LayerStateCheckpoint(stack.network.layers()),
        resume_capability=resume_capability,
    )


from networks.resistive.small import _build_minimizer


def _parameter_learning_rates(
    parameters: tuple[Any, ...],
    *,
    weight_rates: tuple[float, ...],
    bias_rates: tuple[float, ...],
) -> list[float]:
    weights = [parameter for parameter in parameters if not isinstance(parameter, Bias)]
    biases = [parameter for parameter in parameters if isinstance(parameter, Bias)]
    if len(weights) != len(weight_rates):
        raise ValueError(
            "Expected one configured learning rate per non-bias trainable "
            f"parameter ({len(weights)}). Provided value: {len(weight_rates)} rates."
        )
    if len(biases) != len(bias_rates):
        raise ValueError(
            "Expected one configured bias learning rate per trainable Bias "
            f"parameter ({len(biases)}). Provided value: {len(bias_rates)} rates."
        )
    weight_iter = iter(weight_rates)
    bias_iter = iter(bias_rates)
    return [
        float(next(bias_iter) if isinstance(parameter, Bias) else next(weight_iter))
        for parameter in parameters
    ]


from networks.resistive.small import _validate_dataset_shape


from networks.resistive.small import _resolve_device


from networks.resistive.small import _reset_minimizer_statistics


__all__ = [
    "DataBundle",
    "EvaluationRuntime",
    "LayerStateCheckpoint",
    "ModelStack",
    "TrainRuntime",
    "TypedNetwork",
    "ValidationRuntime",
    "build_data",
    "build_evaluation_runtime",
    "build_model_stack",
    "build_train_runtime",
    "build_validation_runtime",
    "seed_runtime",
]
