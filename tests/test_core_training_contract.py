from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from training.core.engine import (
    AfterUpdateEvent, BeforeUpdateEvent, EvaluationComponents, ExperimentComponents,
    FreePhaseEvent, GradientsReadyEvent, train_epoch,
)
from training.core.guards import FiniteGradientGuard
from training.core.optimizers import SGDOptimizer
from training.core.sgd import AugmentedFunction, Backprop, EquilibriumProp
from training.lab.diagnostics import BetaSize, GradientUpdateObserver
from training.lab.epoch import Evaluator, Trainer

from test_small_network_numerical_parity import (
    _build_stack, _inputs, _labels, _load_oracle, _minimizer,
)


@pytest.mark.parametrize("algorithm", ["ep", "backprop"])
@pytest.mark.parametrize("reset_input", [False, True])
def test_epoch_runner_matches_explicit_numerical_sequence(algorithm, reset_input):
    """Check the old batch sequence against the shared runner on real DRNs."""
    oracle = _load_oracle()

    def execute(use_runner):
        stack = _build_stack(oracle)
        energy, cost = stack.bundle.energy, stack.cost_fn
        parameters = tuple(energy.params()) + tuple(cost.params())
        original_tensors = tuple(parameter.state for parameter in parameters)
        free_solver = _minimizer(oracle, energy, stack.free_layers)
        if algorithm == "ep":
            augmented = AugmentedFunction(energy, cost)
            differentiator = EquilibriumProp(
                energy.params(), list(stack.free_layers), augmented, cost,
                _minimizer(oracle, augmented, stack.free_layers), nudging=0.1,
            )
        else:
            differentiator = Backprop(
                energy.params(), list(stack.free_layers), cost,
                _minimizer(oracle, energy, stack.free_layers),
            )
        optimizer = SGDOptimizer(energy, cost, [0.02] * len(parameters))
        inputs, labels = _inputs(oracle, stack), _labels(oracle, stack)
        batches = [
            (inputs, labels), (inputs * 0.7, labels.roll(1)),
            (inputs[:1] * 0.6, labels[:1]),
        ]
        if use_runner:
            runner = Trainer(
                ExperimentComponents(stack.network, cost, free_solver, parameters,
                                     differentiator, optimizer),
                batches, reset_input=reset_input,
            )
            result = runner.run()
            assert result.batch_count == 3
            assert runner.global_step == 3
        else:
            for x, y in batches:
                stack.network.set_input(x, reset=reset_input)
                free_solver.compute_equilibrium()
                cost.set_target(y)
                gradients = differentiator.compute_gradient()
                for parameter, gradient in zip(parameters, gradients):
                    parameter.state.grad = gradient
                optimizer.step()
                for parameter in parameters:
                    parameter.clamp_()
        assert all(parameter.state is state for parameter, state in zip(parameters, original_tensors))
        return [parameter.state.detach().clone() for parameter in parameters] + [
            layer.state.detach().clone() for layer in stack.network.layers()
        ]

    expected, actual = execute(False), execute(True)
    for reference, value in zip(expected, actual):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_backprop_restores_parameter_flags_even_when_solver_fails(requires_grad, failure):
    parameter = SimpleNamespace(state=torch.tensor([0.5], requires_grad=requires_grad))
    layer = SimpleNamespace(state=torch.tensor([2.0]))
    original = parameter.state

    def settle():
        layer.state = parameter.state * 2
        if failure:
            raise RuntimeError("injected solver failure")

    cost = SimpleNamespace(params=lambda: [], eval=lambda: layer.state.square())
    estimator = Backprop([parameter], [layer], cost, SimpleNamespace(compute_equilibrium=settle))
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            estimator.compute_gradient()
    else:
        torch.testing.assert_close(estimator.compute_gradient()[0], torch.tensor([4.0]))
    assert parameter.state is original
    assert parameter.state.requires_grad is requires_grad
    assert layer.state.grad_fn is None


@pytest.mark.parametrize("field,bad_value", [
    ("nudging", 0), ("nudging", float("inf")),
    ("nudging", float("nan")), ("nudging", True), ("nudging", "0.1"),
    ("variant", "unknown"), ("use_alternative_formula", 1),
])
def test_ep_constructor_and_setters_share_validation(field, bad_value):
    args = ([], [], object(), object(), object())
    with pytest.raises(ValueError):
        EquilibriumProp(*args, **{field: bad_value})
    estimator = EquilibriumProp(*args)
    previous = getattr(estimator, field)
    with pytest.raises(ValueError):
        setattr(estimator, field, bad_value)
    assert getattr(estimator, field) == previous


def test_ep_accepts_negative_nudging_like_the_config_parser():
    args = ([], [], object(), object(), object())
    estimator = EquilibriumProp(*args, nudging=-0.1, variant="positive")
    assert (estimator._first_nudging, estimator._second_nudging) == (0.0, -0.1)
    estimator.nudging = -0.2
    assert estimator.nudging == -0.2


class _Parameter:
    name = "weight"

    def __init__(self):
        self.state = torch.tensor([0.8])

    def clamp_(self):
        self.state.clamp_(0, 1)


def _toy_components(gradient=-0.5):
    parameter = _Parameter()
    layer = SimpleNamespace(name="layer", state=torch.tensor([[1.0]]))
    resets = []
    energy = SimpleNamespace(layers=lambda: [layer], grad_layer_fn=lambda _: lambda: layer.state)
    network = SimpleNamespace(
        set_input=lambda x, reset: resets.append(reset), layers=lambda: [layer], _function=energy,
    )
    cost = SimpleNamespace(set_target=lambda _: None)
    optimizer = SimpleNamespace(calls=0)

    def step():
        optimizer.calls += 1
        parameter.state.add_(-parameter.state.grad)

    optimizer.step = step
    components = ExperimentComponents(
        network, cost, SimpleNamespace(compute_equilibrium=lambda: None),
        (parameter,), SimpleNamespace(compute_gradient=lambda: (torch.tensor([gradient]),)),
        optimizer,
    )
    return components, parameter, resets


class _OffsetModifier:
    @contextmanager
    def training_context(self):
        clean = self.parameter.state.clone()
        self.parameter.state.add_(10)
        try:
            yield
        finally:
            self.parameter.state.copy_(clean)

    def __init__(self, parameter):
        self.parameter = parameter


def test_update_observer_measures_restored_master_to_clamped_update():
    components, parameter, _ = _toy_components()
    observer = GradientUpdateObserver()
    events = []
    train_epoch(components, [([0], [0])], modifier=_OffsetModifier(parameter),
                event_handlers=(observer, events.append))
    assert [type(event) for event in events] == [
        FreePhaseEvent, GradientsReadyEvent, BeforeUpdateEvent, AfterUpdateEvent,
    ]
    assert components.optimizer.calls == 1
    assert parameter.state.item() == 1
    assert observer.last_result["update_stats"]["weight"]["norm"] == pytest.approx(0.2)
    assert observer._before is None


def test_engine_does_not_copy_parameters_without_update_observer(monkeypatch):
    components, _, _ = _toy_components()

    def forbidden_clone(*args, **kwargs):
        raise AssertionError("unexpected snapshot in the default training path")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden_clone)
    train_epoch(components, [([0], [0])])
    assert components.optimizer.calls == 1


def test_finite_guard_restores_modifier_and_prevents_update():
    components, parameter, _ = _toy_components(float("nan"))
    with pytest.raises(FloatingPointError, match="finite tensor"):
        train_epoch(components, [([0], [0])], modifier=_OffsetModifier(parameter),
                    event_handlers=(FiniteGradientGuard(),))
    assert components.optimizer.calls == 0
    assert parameter.state.item() == pytest.approx(0.8)


def test_lab_wrappers_use_shared_loops_and_reset_observers():
    from labs.common import CustomTrainer, CustomEvaluator

    components, _, resets = _toy_components()
    # Projected solver residuals have different units and layer coverage;
    # the lab observer must keep measuring the raw energy gradient.
    components.energy_minimizer.residual_currents_inf = lambda: {"projected": 99.0}
    requested = ("store_states", "calc_residual_current")
    trainer = CustomTrainer(components, [([0], [0])], reset_input=False,
                            record_statistics=requested)
    trainer.run()
    trainer.run()
    assert len(trainer.layer_states["layer"]) == 1
    assert trainer.res_currents == {"layer": [1.0]}
    assert resets == [False, False]
    evaluator = CustomEvaluator(
        EvaluationComponents(components.network, components.cost_fn, components.energy_minimizer),
        [([0], [0], [7])], reset_input=True, record_statistics=requested,
    )
    evaluator.run()
    assert evaluator.idx == [7]
    assert evaluator.res_currents == {"layer": [1.0]}
    assert len(evaluator.layer_states["layer"]) == 1
    assert resets[-1] is True


@pytest.mark.parametrize("variant", ["positive", "negative", "centered"])
@pytest.mark.parametrize("failure", [False, True])
def test_beta_diagnostic_uses_nonzero_phase_and_restores_state(variant, failure):
    layer = SimpleNamespace(name="layer", state=torch.tensor([9.0, 11.0]))
    original_state = layer.state
    force = torch.tensor([3.0])
    nudging_term = SimpleNamespace(_force=force)
    augmented = SimpleNamespace(nudging=0.17, _nudging=nudging_term)
    augmented.prepare_nudging = lambda: setattr(nudging_term, "_force", torch.tensor([4.0]))
    observed_nudging = []

    def perturbed_solve():
        observed_nudging.append(augmented.nudging)
        layer.state.add_(augmented.nudging)
        if failure:
            raise RuntimeError("injected beta failure")
        return {layer.name: layer.state}

    estimator = SimpleNamespace(
        _augmented_fn=augmented,
        _first_nudging=0 if variant == "positive" else -0.2,
        _second_nudging=0 if variant == "negative" else 0.2,
        _energy_minimizer=SimpleNamespace(compute_equilibrium=perturbed_solve),
    )
    network = SimpleNamespace(layers=lambda: [layer], set_input=lambda x, reset: None)
    components = EvaluationComponents(
        network, SimpleNamespace(set_target=lambda y: None),
        SimpleNamespace(compute_equilibrium=lambda: layer.state.copy_(torch.tensor([2.0, 4.0]))),
    )
    diagnostic = BetaSize(components, [([0], [0], [1])], estimator)
    if failure:
        with pytest.raises(RuntimeError, match="injected beta"):
            diagnostic.run()
    else:
        diagnostic.run()
        assert diagnostic.summary()["overall_mean"] == pytest.approx(0.2 / 3)
    assert observed_nudging == [0.2 if variant == "positive" else -0.2]
    assert layer.state is original_state
    torch.testing.assert_close(layer.state, torch.tensor([9.0, 11.0]))
    assert augmented.nudging == 0.17
    assert nudging_term._force is force
