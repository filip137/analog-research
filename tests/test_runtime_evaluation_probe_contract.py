from contextlib import contextmanager
from dataclasses import dataclass

import pytest

from training.core.engine import EvaluationComponents, evaluate


@dataclass
class _State:
    value: float


class _Parameter:
    def __init__(self, value):
        self.state = _State(value)


class _EvaluationModifier:
    def __init__(self, parameter, log):
        self.parameter = parameter
        self.log = log

    @contextmanager
    def training_context(self):
        yield self

    @contextmanager
    def evaluation_context(self):
        clean = self.parameter.state.value
        self.log.append(("modifier_enter", clean))
        self.parameter.state.value += 10.0
        try:
            yield self
        finally:
            self.parameter.state.value = clean
            self.log.append(("modifier_exit", clean))

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


class _Network:
    def __init__(self, parameter):
        self.parameter = parameter
        self.observations = []

    def set_input(self, inputs, reset):
        self.observations.append(
            (inputs, reset, self.parameter.state.value)
        )


class _Minimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.observations = []

    def compute_equilibrium(self):
        self.observations.append(self.parameter.state.value)


class _Cost:
    def __init__(self, parameter):
        self.parameter = parameter
        self.observations = []

    def set_target(self, target):
        self.observations.append((target, self.parameter.state.value))


class _Probe:
    def __init__(self, name, parameter, log):
        self._name = name
        self.parameter = parameter
        self.log = log
        self.reset_count = 0
        self.observations = []

    @property
    def name(self):
        return self._name

    def reset(self):
        self.reset_count += 1
        self.observations = []
        self.log.append(("probe_reset", self.name))

    def observe(self, event):
        observation = (
            event.batch_index,
            event.batch.indices,
            self.parameter.state.value,
        )
        self.observations.append(observation)
        self.log.append(("probe_observe", self.name, event.batch_index))

    def result(self):
        self.log.append(
            ("probe_result", self.name, self.parameter.state.value)
        )
        return tuple(self.observations)


def test_evaluate_shares_one_loader_pass_and_one_modifier_context():
    log = []
    parameter = _Parameter(0.25)
    network = _Network(parameter)
    minimizer = _Minimizer(parameter)
    cost = _Cost(parameter)
    components = EvaluationComponents(
        network=network,
        cost_fn=cost,
        energy_minimizer=minimizer,
    )
    first_probe = _Probe("first", parameter, log)
    second_probe = _Probe("second", parameter, log)
    handler_events = []

    result = evaluate(
        components,
        [
            (["x0", "x1"], [0, 1]),
            (["x2"], [2], [8]),
        ],
        modifier=_EvaluationModifier(parameter, log),
        probes=[first_probe, second_probe],
        event_handlers=[
            lambda event: handler_events.append(
                (
                    event.epoch,
                    event.split,
                    event.batch_index,
                    event.batch.indices,
                    parameter.state.value,
                )
            )
        ],
        epoch=4,
        split="validation",
    )

    assert len(network.observations) == 2
    assert len(minimizer.observations) == 2
    assert len(cost.observations) == 2
    assert [item[2] for item in network.observations] == [10.25, 10.25]
    assert minimizer.observations == [10.25, 10.25]
    assert handler_events == [
        (4, "validation", 0, None, 10.25),
        (4, "validation", 1, [8], 10.25),
    ]
    assert first_probe.reset_count == 1
    assert second_probe.reset_count == 1
    assert len(first_probe.observations) == 2
    assert len(second_probe.observations) == 2
    assert parameter.state.value == 0.25

    assert log.count(("modifier_enter", 0.25)) == 1
    assert log.count(("modifier_exit", 0.25)) == 1
    exit_index = log.index(("modifier_exit", 0.25))
    assert log.index(("probe_result", "first", 0.25)) > exit_index
    assert log.index(("probe_result", "second", 0.25)) > exit_index

    assert result.epoch == 4
    assert result.split == "validation"
    assert result.batch_count == 2
    assert result.example_count == 3
    assert result.probe_value("first") == (
        (0, None, 10.25),
        (1, [8], 10.25),
    )


def test_evaluate_rejects_duplicate_probe_names_before_running():
    log = []
    parameter = _Parameter(0.25)
    network = _Network(parameter)
    components = EvaluationComponents(
        network=network,
        cost_fn=_Cost(parameter),
        energy_minimizer=_Minimizer(parameter),
    )

    with pytest.raises(ValueError, match="probe names to be unique"):
        evaluate(
            components,
            [(["x"], [0])],
            modifier=_EvaluationModifier(parameter, log),
            probes=[
                _Probe("duplicate", parameter, log),
                _Probe("duplicate", parameter, log),
            ],
        )

    assert log == []
    assert network.observations == []


def test_evaluate_can_preserve_previous_state_when_explicitly_requested():
    parameter = _Parameter(0.25)
    network = _Network(parameter)
    components = EvaluationComponents(
        network=network,
        cost_fn=_Cost(parameter),
        energy_minimizer=_Minimizer(parameter),
    )

    evaluate(
        components,
        [(["x"], [0])],
        reset_input=False,
    )

    assert network.observations == [(["x"], False, 0.25)]


def test_evaluate_rejects_non_boolean_reset_policy():
    parameter = _Parameter(0.25)
    network = _Network(parameter)
    components = EvaluationComponents(
        network=network,
        cost_fn=_Cost(parameter),
        energy_minimizer=_Minimizer(parameter),
    )

    with pytest.raises(ValueError, match="reset_input to be a boolean"):
        evaluate(
            components,
            [(["x"], [0])],
            reset_input="yes",
        )

    assert network.observations == []


def test_probe_failure_restores_evaluation_modifier():
    log = []
    parameter = _Parameter(0.25)
    components = EvaluationComponents(
        network=_Network(parameter),
        cost_fn=_Cost(parameter),
        energy_minimizer=_Minimizer(parameter),
    )

    class _FailingProbe(_Probe):
        def observe(self, event):
            raise RuntimeError("probe failed")

    with pytest.raises(RuntimeError, match="probe failed"):
        evaluate(
            components,
            [(["x"], [0])],
            modifier=_EvaluationModifier(parameter, log),
            probes=[_FailingProbe("failure", parameter, log)],
        )

    assert parameter.state.value == 0.25
    assert log[-1] == ("modifier_exit", 0.25)
