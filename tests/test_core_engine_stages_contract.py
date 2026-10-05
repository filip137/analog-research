"""Contracts for the engine's optional stages and the shared loop parts.

The MNIST DRN families select their loop by leaving the differentiator or
the optimizer out of the components; these tests pin what each selection
runs, plus the batch adapter, composed modifier, probes and named guard.
"""

from contextlib import contextmanager

import pytest
import torch

from experiments.mnist_shared import TeacherTargets, teacher_batches
from training.core.batch import Batch, prepared_batches
from training.core.engine import ExperimentComponents, train_epoch
from training.core.guards import FiniteGradientGuard
from training.core.modifier import ComposedModifier
from training.core.probes import PerBatchProbe


class _Parameter:
    def __init__(self, log):
        self.state = torch.tensor([2.0])
        self.log = log

    def clamp_(self):
        self.log.append("clamp")
        self.state.clamp_(0.0, 1.0)


def _components(log, *, differentiator=True, optimizer=True, clamp=True):
    parameter = _Parameter(log)
    network = type("Network", (), {"set_input": lambda self, x, reset: log.append("set_input")})()
    minimizer = type("Minimizer", (), {"compute_equilibrium": lambda self: log.append("settle")})()
    cost = type("Cost", (), {"set_target": lambda self, t: log.append("target")})()
    gradient = type(
        "Differentiator",
        (),
        {"compute_gradient": lambda self: (log.append("gradient"), (torch.tensor([0.5]),))[1]},
    )()
    step = type("Optimizer", (), {"step": lambda self: log.append("step")})()
    return parameter, ExperimentComponents(
        network,
        cost,
        minimizer,
        (parameter,),
        gradient if differentiator else None,
        step if optimizer else None,
        clamp_after_update=clamp,
    )


def _events(log):
    def handler(event):
        log.append(type(event).__name__)

    return handler


_BATCH = [(torch.zeros(1, 2), torch.zeros(1))]


def test_full_step_runs_every_stage_in_order():
    log = []
    _parameter, components = _components(log)
    train_epoch(components, _BATCH, event_handlers=(_events(log),))
    assert log == [
        "set_input", "settle", "target", "FreePhaseEvent", "gradient",
        "GradientsReadyEvent", "BeforeUpdateEvent", "step", "clamp",
        "AfterUpdateEvent",
    ]


def test_gradient_pass_assigns_gradients_without_updating():
    log = []
    parameter, components = _components(log, optimizer=False)
    train_epoch(components, _BATCH, event_handlers=(_events(log),))
    assert log == [
        "set_input", "settle", "target", "FreePhaseEvent", "gradient",
        "GradientsReadyEvent",
    ]
    assert torch.equal(parameter.state.grad, torch.tensor([0.5]))
    assert torch.equal(parameter.state, torch.tensor([2.0]))


def test_update_without_gradient_skips_gradients_and_optionally_clamping():
    log = []
    parameter, components = _components(log, differentiator=False, clamp=False)
    train_epoch(components, _BATCH, event_handlers=(_events(log),))
    assert log == [
        "set_input", "settle", "target", "FreePhaseEvent",
        "BeforeUpdateEvent", "step", "AfterUpdateEvent",
    ]
    assert parameter.state.grad is None
    assert torch.equal(parameter.state, torch.tensor([2.0]))


def test_training_probes_observe_each_free_phase_after_handlers():
    log = []
    _parameter, components = _components(log, optimizer=False)
    probe = PerBatchProbe("phase", lambda event: (log.append("probe"), event.batch_index)[1])
    result = train_epoch(
        components, _BATCH * 3, probes=(probe,), event_handlers=(_events(log),)
    )
    assert result.probe_value("phase") == [0, 1, 2]
    first = log[: log.index("gradient")]
    assert first[-2:] == ["FreePhaseEvent", "probe"]
    again = train_epoch(components, _BATCH, probes=(probe,))
    assert again.probe_value("phase") == [0]


def test_named_guard_rejects_nonfinite_gradients_before_later_handlers():
    log = []
    _parameter, components = _components(log, optimizer=False)
    components = ExperimentComponents(
        components.network, components.cost_fn, components.energy_minimizer,
        components.parameters,
        type("Nan", (), {"compute_gradient": lambda self: (torch.tensor([float("nan")]),)})(),
        None,
    )
    with pytest.raises(FloatingPointError, match=r"finite KD gradients.*'base\.w', batch=0"):
        train_epoch(
            components,
            _BATCH,
            event_handlers=(FiniteGradientGuard(("base.w",), label="KD"), _events(log)),
        )
    assert "GradientsReadyEvent" not in log


def test_composed_modifier_enters_in_order_and_exits_in_reverse():
    log = []

    class Named:
        def __init__(self, name):
            self.name = name

        @contextmanager
        def _context(self, phase):
            log.append(f"enter {self.name} {phase}")
            try:
                yield self
            finally:
                log.append(f"exit {self.name} {phase}")

        def training_context(self):
            return self._context("train")

        def evaluation_context(self):
            return self._context("eval")

        def state_dict(self):
            return {"name": self.name}

        def load_state_dict(self, state):
            log.append(f"load {state['name']}")

    composed = ComposedModifier(Named("a"), None, Named("b"))
    with composed.training_context():
        log.append("body")
    with composed.evaluation_context():
        pass
    assert log == [
        "enter a train", "enter b train", "body", "exit b train", "exit a train",
        "enter a eval", "enter b eval", "exit b eval", "exit a eval",
    ]
    state = composed.state_dict()
    composed.load_state_dict(state)
    assert log[-2:] == ["load a", "load b"]
    with pytest.raises(ValueError):
        composed.load_state_dict({"version": 1, "modifiers": [{}]})


class _CountingLoader:
    def __init__(self, batches):
        self.batches = batches
        self.pulls = 0
        self.iterators = 0

    def __iter__(self):
        self.iterators += 1
        for batch in self.batches:
            self.pulls += 1
            yield batch


def _pairs(sizes):
    return [(torch.arange(size), torch.arange(size)) for size in sizes]


def test_prepared_batches_never_pulls_past_the_batch_limit_and_starts_lazily():
    loader = _CountingLoader(_pairs([2, 2, 2, 2]))
    batches = prepared_batches(loader, maximum_batches=2)
    assert loader.iterators == 0
    assert len(list(batches)) == 2
    assert loader.pulls == 2


def test_prepared_batches_checks_stop_before_each_pull():
    loader = _CountingLoader(_pairs([2, 2, 2, 2]))
    seen = []
    for batch in prepared_batches(loader, stop=lambda: len(seen) == 2):
        seen.append(batch)
    assert loader.pulls == 2


def test_prepared_batches_truncates_to_the_sample_limit_and_drops_one_extra_pull():
    loader = _CountingLoader(_pairs([3, 3, 3]))
    batches = list(prepared_batches(loader, sample_limit=4))
    assert [batch.example_count for batch in batches] == [3, 1]
    assert loader.pulls == 3
    exact = _CountingLoader(_pairs([2, 2, 2]))
    assert len(list(prepared_batches(exact, sample_limit=4))) == 2
    assert exact.pulls == 3


def test_teacher_batches_count_examples_and_transfer_before_the_teacher():
    seen = []

    class Teacher:
        def logits(self, inputs):
            seen.append((inputs.dtype, torch.is_grad_enabled()))
            return torch.zeros(inputs.shape[0], 2)

    raw = [(torch.zeros(3, 4, dtype=torch.float64), torch.tensor([0, 1, 1]))]
    hashed = []
    (batch,) = teacher_batches(
        raw, Teacher(), "cpu", before_transfer=lambda x, y: hashed.append(x.dtype)
    )
    assert isinstance(batch, Batch) and isinstance(batch.targets, TeacherTargets)
    assert batch.example_count == 3
    assert batch.targets.labels.dtype == torch.long
    assert seen == [(torch.float32, False)]
    assert hashed == [torch.float64]
