"""Numerical parity for the finite-conductance-floor replay utility."""

from __future__ import annotations

import pytest
import torch

from labs.tools.analyze_mnist_conductance_floor import replay
from labs.tools.analyze_mnist_mapping_kl import paired_scores
from networks.function.network import Network
from networks.resistive.minimizer import QuadraticMinimizer
from networks.resistive.network import DeepResistiveEnergy
from networks.variable.parameter import Bias, DenseWeight


def test_floor_replay_matches_the_runtime_coordinate_updates() -> None:
    energy = DeepResistiveEnergy(
        layer_shapes=[(4,), (4,), (4,)],
        weight_gains=[0.2, 0.2],
        input_gain=2.5,
        non_linearity="perfect_diode",
        quadratic_diode_param={},
        exponential_diode_param={},
        hard_sigmoid_param={},
        voltage_amp=1.0,
        current_amp=1.0,
        weight_min=1e-7,
        weight_max=2.0,
    )
    energy.set_device(torch.device("cpu"))
    dense = [
        parameter
        for parameter in energy.params()
        if isinstance(parameter, DenseWeight)
    ]
    biases = [
        parameter
        for parameter in energy.params()
        if isinstance(parameter, Bias)
    ]
    assert len(dense) == 2
    assert len(biases) == 1

    w1 = torch.tensor(
        [
            [0.4, 0.2, 0.3, 0.1],
            [0.1, 0.5, 0.2, 0.4],
            [0.3, 0.1, 0.6, 0.2],
            [0.2, 0.4, 0.1, 0.5],
        ]
    )
    w2 = torch.tensor(
        [
            [0.5, 0.2, 0.1, 0.4],
            [0.2, 0.6, 0.3, 0.1],
            [0.4, 0.1, 0.5, 0.2],
            [0.1, 0.3, 0.2, 0.6],
        ]
    )
    bias = torch.tensor([0.03, -0.02, 0.01, -0.04])
    with torch.no_grad():
        dense[0].state.copy_(w1)
        dense[1].state.copy_(w2)
        biases[0].state.copy_(bias)

    inputs = torch.tensor(
        [
            [0.8, 0.1],
            [0.2, 0.9],
            [0.6, 0.4],
        ]
    )
    targets = torch.tensor([0, 1, 0])
    iterations = 4
    network = Network(energy)
    network.set_input(inputs, reset=True)
    QuadraticMinimizer(
        fn=energy,
        free_layers=network.free_layers(),
        num_iterations=iterations,
        mode="asynchronous",
        non_linearity="perfect_diode",
        quadratic_diode_param={},
        exponential_diode_param={},
        hard_sigmoid_param={},
        voltage_amp=1.0,
        current_amp=1.0,
    ).compute_equilibrium()

    hidden = network.layers()[1].state
    output = network.layers()[2].state
    scores = output.reshape(output.shape[0], 2, 2)
    predictions = (scores[..., 0] - scores[..., 1]).argmax(1)
    expected_accuracy = float((predictions == targets).float().mean())

    result = replay(
        inputs,
        targets,
        numerator_w1=w1,
        numerator_w2=w2,
        bias=bias,
        input_gain=2.5,
        iterations=iterations,
        batch_size=inputs.shape[0],
    )
    replayed_scores = paired_scores(
        inputs,
        w1=w1,
        w2=w2,
        bias=bias,
        input_gain=2.5,
        iterations=iterations,
        batch_size=inputs.shape[0],
    )

    torch.testing.assert_close(
        replayed_scores,
        scores[..., 0] - scores[..., 1],
    )
    assert result["accuracy"] == pytest.approx(expected_accuracy)
    assert result["hidden_voltage_rms"] == pytest.approx(
        float(torch.sqrt(hidden.square().mean()))
    )
    assert result["output_voltage_rms"] == pytest.approx(
        float(torch.sqrt(output.square().mean()))
    )
    assert result["hidden_degree_mean"] == pytest.approx(
        float((w1.sum(0) + w2.sum(1)).mean())
    )
    assert result["output_degree_mean"] == pytest.approx(
        float(w2.sum(0).mean())
    )
