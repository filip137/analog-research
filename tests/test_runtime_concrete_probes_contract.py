import numpy as np
import pytest
import torch

from training.core.engine import EvaluationComponents, evaluate
from training.core.probes import (
    MeanCostProbe,
    MeanErrorProbe,
    ResidualInfinityNormProbe,
    SettledLayerStatesProbe,
    SolverIterationCountsProbe,
)


class _Layer:
    def __init__(self, name):
        self.name = name
        self.state = torch.empty((0, 1))


class _Function:
    def __init__(self, layers):
        self._layers = layers
        self.gradient_calls = {layer.name: 0 for layer in layers}

    def layers(self):
        return self._layers

    def grad_layer_fn(self, layer):
        def gradient():
            self.gradient_calls[layer.name] += 1
            return layer.state * (
                1.0 if layer.name == "Layer_0" else -2.0
            )

        return gradient


class _Network:
    def __init__(self):
        self._layers = [_Layer("Layer_0"), _Layer("Layer_1")]
        self._function = _Function(self._layers)
        self.reset_values = []

    def layers(self):
        return self._layers

    def set_input(self, inputs, reset):
        self.reset_values.append(reset)
        self._layers[0].state = inputs.detach().clone()
        self._layers[1].state = inputs.detach().clone() + 0.5


class _Cost:
    def __init__(self, network):
        self.network = network
        self.targets = None

    def set_target(self, targets):
        self.targets = targets

    def eval(self):
        prediction = self.network.layers()[-1].state.reshape(-1)
        return (prediction - self.targets.float()).square()

    def error_fn(self):
        prediction = self.network.layers()[-1].state.reshape(-1)
        return prediction.round().long() != self.targets


class _SampleIterationMinimizer:
    def __init__(self, network):
        self.network = network
        self.calls = 0
        self.sample_iteration_calls = 0
        self._sample_iterations = None

    def compute_equilibrium(self):
        self.calls += 1
        self.network.layers()[-1].state.add_(0.5)
        batch_size = self.network.layers()[-1].state.shape[0]
        self._sample_iterations = torch.arange(
            self.calls,
            self.calls + batch_size,
        )

    def equilibrium_sample_iterations(self):
        self.sample_iteration_calls += 1
        return self._sample_iterations


def _components():
    network = _Network()
    cost = _Cost(network)
    minimizer = _SampleIterationMinimizer(network)
    return (
        EvaluationComponents(
            network=network,
            cost_fn=cost,
            energy_minimizer=minimizer,
        ),
        network,
        minimizer,
    )


def test_concrete_probes_share_one_settled_pass_and_return_memory_values():
    components, network, minimizer = _components()
    cost_probe = MeanCostProbe()
    error_probe = MeanErrorProbe()
    states_probe = SettledLayerStatesProbe()
    residual_probe = ResidualInfinityNormProbe()
    iterations_probe = SolverIterationCountsProbe()

    result = evaluate(
        components,
        [
            (
                torch.tensor([[0.0], [1.0]]),
                torch.tensor([1, 1]),
            ),
            (
                torch.tensor([[2.0]]),
                torch.tensor([3]),
                torch.tensor([7]),
            ),
        ],
        probes=[
            cost_probe,
            error_probe,
            states_probe,
            residual_probe,
            iterations_probe,
        ],
    )

    assert minimizer.calls == 2
    assert minimizer.sample_iteration_calls == 2
    assert network.reset_values == [True, True]

    assert result.probe_value("mean_cost") == {
        "mean": pytest.approx(1.0 / 3.0),
        "sum": pytest.approx(1.0),
        "count": 3,
    }
    assert result.probe_value("mean_error") == {
        "mean": pytest.approx(1.0 / 3.0),
        "sum": pytest.approx(1.0),
        "count": 3,
    }

    states = result.probe_value("settled_layer_states")
    torch.testing.assert_close(
        states["Layer_0"],
        torch.tensor([[0.0], [1.0], [2.0]]),
    )
    torch.testing.assert_close(
        states["Layer_1"],
        torch.tensor([[1.0], [2.0], [3.0]]),
    )
    assert states_probe.summary() == {
        "Layer_0": {"shape": [3, 1], "dtype": "torch.float32"},
        "Layer_1": {"shape": [3, 1], "dtype": "torch.float32"},
    }

    residuals = result.probe_value("residual_inf_norm")
    np.testing.assert_allclose(
        residuals["Layer_0"],
        np.asarray([1.0, 2.0]),
    )
    np.testing.assert_allclose(
        residuals["Layer_1"],
        np.asarray([4.0, 6.0]),
    )
    assert network._function.gradient_calls == {
        "Layer_0": 2,
        "Layer_1": 2,
    }
    assert residual_probe.summary() == {
        "Layer_0": {
            "count": 2,
            "mean": 1.5,
            "min": 1.0,
            "max": 2.0,
        },
        "Layer_1": {
            "count": 2,
            "mean": 5.0,
            "min": 4.0,
            "max": 6.0,
        },
    }

    np.testing.assert_array_equal(
        result.probe_value("solver_iteration_counts"),
        np.asarray([1, 2, 2], dtype=np.int64),
    )
    assert iterations_probe.summary() == {
        "count": 3,
        "mean": pytest.approx(5.0 / 3.0),
        "min": 1.0,
        "max": 2.0,
    }


class _DirectResidualMinimizer:
    def __init__(self):
        self.num_iterations = 4
        self.residual_calls = 0

    def compute_equilibrium(self):
        return None

    def residual_currents_inf(self):
        self.residual_calls += 1
        return {
            "Layer_0": torch.tensor(0.25),
            "Layer_1": 0.5,
        }


def test_residual_prefers_minimizer_api_once_per_batch_and_fixed_iterations():
    network = _Network()
    minimizer = _DirectResidualMinimizer()
    components = EvaluationComponents(
        network=network,
        cost_fn=_Cost(network),
        energy_minimizer=minimizer,
    )
    residual_probe = ResidualInfinityNormProbe()
    iteration_probe = SolverIterationCountsProbe()

    result = evaluate(
        components,
        [
            (torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1])),
            (torch.tensor([[2.0]]), torch.tensor([2])),
        ],
        probes=[residual_probe, iteration_probe],
    )

    assert minimizer.residual_calls == 2
    assert network._function.gradient_calls == {
        "Layer_0": 0,
        "Layer_1": 0,
    }
    np.testing.assert_array_equal(
        result.probe_value("solver_iteration_counts"),
        np.asarray([4, 4, 4]),
    )


class _StatsIterationMinimizer:
    def __init__(self):
        self.stats_calls = []

    def compute_equilibrium(self):
        return None

    def equilibrium_sample_iterations(self):
        return None

    def equilibrium_iteration_stats(self, reset):
        self.stats_calls.append(reset)
        return {"last_iterations": 6}


def test_iteration_probe_falls_back_to_non_resetting_solver_stats():
    network = _Network()
    minimizer = _StatsIterationMinimizer()
    components = EvaluationComponents(
        network=network,
        cost_fn=_Cost(network),
        energy_minimizer=minimizer,
    )

    result = evaluate(
        components,
        [(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1]))],
        probes=[SolverIterationCountsProbe()],
    )

    assert minimizer.stats_calls == [False]
    np.testing.assert_array_equal(
        result.probe_value("solver_iteration_counts"),
        np.asarray([6, 6]),
    )


def test_iteration_probe_rejects_shape_mismatch():
    components, _network, minimizer = _components()
    minimizer.equilibrium_sample_iterations = lambda: torch.tensor([1, 2, 3])

    with pytest.raises(
        ValueError,
        match="one value per example",
    ):
        evaluate(
            components,
            [(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1]))],
            probes=[SolverIterationCountsProbe()],
        )
