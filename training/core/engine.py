"""Protocol-driven training and evaluation loops.

This module is intentionally independent of experiment configuration,
hardware-aware policies, persistence, and the ``training.lab.epoch`` runners.
It defines the ordering contract that those higher-level layers can reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Tuple,
    Union,
)

from training.core.batch import Batch, RawBatch, as_batch
from training.core.modifier import ParameterModifier, modifier_or_default
from training.core.probes import EvaluationProbe, ProbeResult


DEFAULT_TRAIN_RESET_INPUT = False
DEFAULT_EVALUATION_RESET_INPUT = True


@dataclass(frozen=True)
class EvaluationComponents:
    """Objects required to settle and measure a model.

    ``cost_fn=None`` supports unlabelled state and residual diagnostics.
    """

    network: Any
    cost_fn: Any
    energy_minimizer: Any


@dataclass(frozen=True)
class ExperimentComponents(EvaluationComponents):
    """Objects required for one complete parameter-update step."""

    parameters: Tuple[Any, ...]
    differentiator: Any
    optimizer: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))


@dataclass(frozen=True)
class FreePhaseEvent:
    """A settled free phase, emitted before gradient computation."""

    epoch: int
    batch_index: int
    global_step: int
    batch: Batch[Any, Any, Any]
    components: ExperimentComponents


@dataclass(frozen=True)
class GradientsReadyEvent:
    """Computed gradients, emitted while the parameter modifier is active."""

    epoch: int
    batch_index: int
    global_step: int
    batch: Batch[Any, Any, Any]
    components: ExperimentComponents
    gradients: Tuple[Any, ...]


@dataclass(frozen=True)
class BeforeUpdateEvent:
    """Restored parameters immediately before the update backend runs."""

    epoch: int
    batch_index: int
    global_step: int
    batch: Batch[Any, Any, Any]
    components: ExperimentComponents


@dataclass(frozen=True)
class AfterUpdateEvent:
    """Parameters after the backend update and model-bound clamping."""

    epoch: int
    batch_index: int
    global_step: int
    batch: Batch[Any, Any, Any]
    components: ExperimentComponents


TrainingEvent = Union[
    FreePhaseEvent, GradientsReadyEvent, BeforeUpdateEvent, AfterUpdateEvent
]
TrainingEventHandler = Callable[[TrainingEvent], None]


@dataclass(frozen=True)
class EvaluationBatchEvent:
    """A settled evaluation minibatch shared by handlers and probes."""

    epoch: int
    split: str
    batch_index: int
    batch: Batch[Any, Any, Any]
    components: EvaluationComponents


EvaluationEventHandler = Callable[[EvaluationBatchEvent], None]


@dataclass(frozen=True)
class TrainingEpochResult:
    """Progress produced by one call to :func:`train_epoch`."""

    epoch: int
    batch_count: int
    example_count: int
    start_global_step: int
    next_global_step: int


@dataclass(frozen=True)
class EvaluationResult:
    """Progress and named probe values from one evaluation pass."""

    epoch: int
    split: str
    batch_count: int
    example_count: int
    probe_results: Tuple[ProbeResult[Any], ...]

    def probe_value(self, name: str) -> Any:
        """Return one probe value by name."""

        for probe_result in self.probe_results:
            if probe_result.name == name:
                return probe_result.value
        raise KeyError(name)


def train_epoch(
    components: ExperimentComponents,
    dataloader: Iterable[RawBatch],
    *,
    modifier: Optional[ParameterModifier] = None,
    event_handlers: Iterable[TrainingEventHandler] = (),
    epoch: int = 0,
    start_global_step: int = 0,
    reset_input: bool = DEFAULT_TRAIN_RESET_INPUT,
) -> TrainingEpochResult:
    """Train for one loader traversal using the authoritative update order.

    The modifier covers input assignment, the free phase, target assignment,
    gradient computation, and both metric/event points.  It exits before
    ``optimizer.step()``, after which every parameter is clamped. Before/after
    update events observe restored and clamped parameters respectively. Training
    deliberately defaults to ``reset_input=False`` to preserve the legacy
    small-network continuation between equal-sized minibatches.
    """

    _validate_reset_input(reset_input)
    active_modifier = modifier_or_default(modifier)
    handlers = tuple(event_handlers)
    batch_count = 0
    example_count = 0

    for batch_index, raw_batch in enumerate(dataloader):
        batch = as_batch(raw_batch)
        global_step = start_global_step + batch_index

        with active_modifier.training_context():
            components.network.set_input(batch.inputs, reset=reset_input)
            components.energy_minimizer.compute_equilibrium()
            components.cost_fn.set_target(batch.targets)

            free_event = FreePhaseEvent(
                epoch=epoch,
                batch_index=batch_index,
                global_step=global_step,
                batch=batch,
                components=components,
            )
            _emit(handlers, free_event)

            raw_gradients = components.differentiator.compute_gradient()
            try:
                gradients = tuple(raw_gradients)
            except TypeError as error:
                raise ValueError(
                    "Expected differentiator.compute_gradient() to return "
                    "one gradient per experiment parameter. "
                    f"Provided value: {raw_gradients!r}."
                ) from error

            if len(gradients) != len(components.parameters):
                raise ValueError(
                    "Expected differentiator.compute_gradient() to return "
                    f"{len(components.parameters)} gradients. "
                    f"Provided value: {len(gradients)} gradients."
                )

            for parameter, gradient in zip(
                components.parameters, gradients
            ):
                parameter.state.grad = gradient

            gradient_event = GradientsReadyEvent(
                epoch=epoch,
                batch_index=batch_index,
                global_step=global_step,
                batch=batch,
                components=components,
                gradients=gradients,
            )
            _emit(handlers, gradient_event)

        _emit(handlers, BeforeUpdateEvent(
            epoch, batch_index, global_step, batch, components
        ))
        components.optimizer.step()
        for parameter in components.parameters:
            parameter.clamp_()
        _emit(handlers, AfterUpdateEvent(
            epoch, batch_index, global_step, batch, components
        ))

        batch_count += 1
        example_count += batch.example_count

    return TrainingEpochResult(
        epoch=epoch,
        batch_count=batch_count,
        example_count=example_count,
        start_global_step=start_global_step,
        next_global_step=start_global_step + batch_count,
    )


def evaluate(
    components: EvaluationComponents,
    dataloader: Iterable[RawBatch],
    *,
    modifier: Optional[ParameterModifier] = None,
    probes: Iterable[EvaluationProbe[Any]] = (),
    event_handlers: Iterable[EvaluationEventHandler] = (),
    epoch: int = 0,
    split: str = "evaluation",
    reset_input: bool = DEFAULT_EVALUATION_RESET_INPUT,
) -> EvaluationResult:
    """Evaluate once while updating every probe in the same loader loop.

    The modifier context spans the entire loader, allowing one temporary
    parameter realization to remain fixed for a complete evaluation pass.
    Evaluation deliberately defaults to ``reset_input=True`` so samples do not
    inherit settled states from a previous minibatch.
    """

    _validate_reset_input(reset_input)
    active_modifier = modifier_or_default(modifier)
    active_probes = tuple(probes)
    handlers = tuple(event_handlers)
    _validate_probes(active_probes)

    for probe in active_probes:
        probe.reset()

    batch_count = 0
    example_count = 0
    with active_modifier.evaluation_context():
        for batch_index, raw_batch in enumerate(dataloader):
            batch = as_batch(raw_batch)
            components.network.set_input(batch.inputs, reset=reset_input)
            components.energy_minimizer.compute_equilibrium()
            if components.cost_fn is not None:
                components.cost_fn.set_target(batch.targets)

            event = EvaluationBatchEvent(
                epoch=epoch,
                split=split,
                batch_index=batch_index,
                batch=batch,
                components=components,
            )
            _emit(handlers, event)
            for probe in active_probes:
                probe.observe(event)

            batch_count += 1
            example_count += batch.example_count

    probe_results = tuple(
        ProbeResult(name=probe.name, value=probe.result())
        for probe in active_probes
    )
    return EvaluationResult(
        epoch=epoch,
        split=split,
        batch_count=batch_count,
        example_count=example_count,
        probe_results=probe_results,
    )


def _emit(handlers: Tuple[Callable[[Any], None], ...], event: Any) -> None:
    for handler in handlers:
        handler(event)


def _validate_probes(probes: Tuple[EvaluationProbe[Any], ...]) -> None:
    names = []
    for probe in probes:
        name = probe.name
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Expected every evaluation probe name to be a non-empty "
                f"string. Provided value: {name!r}."
            )
        names.append(name)

    duplicates = sorted(
        {name for name in names if names.count(name) > 1}
    )
    if duplicates:
        raise ValueError(
            "Expected evaluation probe names to be unique. "
            f"Provided value: duplicate names {duplicates!r}."
        )


def _validate_reset_input(reset_input: bool) -> None:
    if not isinstance(reset_input, bool):
        raise ValueError(
            "Expected reset_input to be a boolean. "
            f"Provided value: {reset_input!r}."
        )
