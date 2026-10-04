from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from training.core.engine import (
    ExperimentComponents,
    FreePhaseEvent,
    GradientsReadyEvent,
    train_epoch,
)


@dataclass
class _State:
    value: float
    grad: float = 0.0


class _Parameter:
    def __init__(self, value, log):
        self.state = _State(value)
        self.log = log
        self.clamp_count = 0

    def clamp_(self):
        self.log.append(("clamp", self.state.value))
        self.clamp_count += 1
        self.state.value = min(max(self.state.value, 0.0), 1.0)


class _Modifier:
    def __init__(self, parameter, log):
        self.parameter = parameter
        self.log = log

    @contextmanager
    def training_context(self):
        clean = self.parameter.state.value
        self.log.append(("modifier_enter", clean))
        self.parameter.state.value += 10.0
        try:
            yield self
        finally:
            self.parameter.state.value = clean
            self.log.append(("modifier_exit", clean))

    @contextmanager
    def evaluation_context(self):
        yield self

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


class _Network:
    def __init__(self, parameter, log):
        self.parameter = parameter
        self.log = log

    def set_input(self, inputs, reset):
        self.log.append(
            ("set_input", self.parameter.state.value, inputs, reset)
        )


class _Minimizer:
    def __init__(self, parameter, log):
        self.parameter = parameter
        self.log = log

    def compute_equilibrium(self):
        self.log.append(("free_phase", self.parameter.state.value))


class _Cost:
    def __init__(self, parameter, log):
        self.parameter = parameter
        self.log = log

    def set_target(self, targets):
        self.log.append(
            ("set_target", self.parameter.state.value, targets)
        )


class _Differentiator:
    def __init__(self, parameter, log):
        self.parameter = parameter
        self.log = log

    def compute_gradient(self):
        self.log.append(("gradient", self.parameter.state.value))
        return (-0.5,)


class _Optimizer:
    def __init__(self, parameter, log):
        self.parameter = parameter
        self.log = log
        self.step_count = 0

    def step(self):
        self.log.append(
            (
                "optimizer_step",
                self.parameter.state.value,
                self.parameter.state.grad,
            )
        )
        self.step_count += 1
        self.parameter.state.value -= self.parameter.state.grad


def _components(log):
    parameter = _Parameter(0.8, log)
    optimizer = _Optimizer(parameter, log)
    components = ExperimentComponents(
        network=_Network(parameter, log),
        cost_fn=_Cost(parameter, log),
        energy_minimizer=_Minimizer(parameter, log),
        parameters=[parameter],
        differentiator=_Differentiator(parameter, log),
        optimizer=optimizer,
    )
    return components, parameter, optimizer


def test_train_epoch_keeps_metrics_modified_then_steps_clean_and_clamps():
    log = []
    components, parameter, optimizer = _components(log)
    modifier = _Modifier(parameter, log)

    def measure(event):
        if isinstance(event, FreePhaseEvent):
            log.append(("free_metric", parameter.state.value))
        elif isinstance(event, GradientsReadyEvent):
            log.append(
                (
                    "gradient_metric",
                    parameter.state.value,
                    parameter.state.grad,
                )
            )

    result = train_epoch(
        components,
        [(["x0", "x1"], [0, 1])],
        modifier=modifier,
        event_handlers=[measure],
        epoch=3,
        start_global_step=7,
    )

    assert log == [
        ("modifier_enter", 0.8),
        ("set_input", 10.8, ["x0", "x1"], False),
        ("free_phase", 10.8),
        ("set_target", 10.8, [0, 1]),
        ("free_metric", 10.8),
        ("gradient", 10.8),
        ("gradient_metric", 10.8, -0.5),
        ("modifier_exit", 0.8),
        ("optimizer_step", 0.8, -0.5),
        ("clamp", 1.3),
    ]
    assert parameter.state.value == 1.0
    assert optimizer.step_count == 1
    assert parameter.clamp_count == 1
    assert components.parameters == (parameter,)
    assert result.epoch == 3
    assert result.batch_count == 1
    assert result.example_count == 2
    assert result.start_global_step == 7
    assert result.next_global_step == 8


def test_train_epoch_can_explicitly_reset_each_input_batch():
    log = []
    components, _parameter, _optimizer = _components(log)

    train_epoch(
        components,
        [(["x"], [0])],
        reset_input=True,
    )

    set_input = next(item for item in log if item[0] == "set_input")
    assert set_input[-1] is True


def test_train_epoch_rejects_non_boolean_reset_policy():
    log = []
    components, _parameter, _optimizer = _components(log)

    with pytest.raises(ValueError, match="reset_input to be a boolean"):
        train_epoch(
            components,
            [(["x"], [0])],
            reset_input=1,
        )

    assert log == []


def test_train_epoch_restores_before_propagating_metric_exception():
    log = []
    components, parameter, optimizer = _components(log)
    modifier = _Modifier(parameter, log)

    def fail_after_gradient(event):
        if isinstance(event, GradientsReadyEvent):
            raise RuntimeError("metric failed")

    with pytest.raises(RuntimeError, match="metric failed"):
        train_epoch(
            components,
            [(["x"], [0])],
            modifier=modifier,
            event_handlers=[fail_after_gradient],
        )

    assert parameter.state.value == 0.8
    assert optimizer.step_count == 0
    assert parameter.clamp_count == 0
    assert log[-1] == ("modifier_exit", 0.8)


def test_train_epoch_rejects_gradient_count_mismatch_before_update():
    log = []
    components, parameter, optimizer = _components(log)
    components.differentiator.compute_gradient = lambda: ()

    with pytest.raises(ValueError, match="return 1 gradients"):
        train_epoch(
            components,
            [(["x"], [0])],
            modifier=_Modifier(parameter, log),
        )

    assert parameter.state.value == 0.8
    assert optimizer.step_count == 0
