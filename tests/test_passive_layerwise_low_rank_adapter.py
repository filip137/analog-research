from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from model.function.cost import SquaredErrorPairedOutputs
from model.resistive.builders import build_deep_resistive_energy
from model.resistive.interaction import DenseResistive
from model.resistive.low_rank import PassiveLowRankDenseWeight
from model.resistive.minimizer import QuadraticMinimizer
from model.resistive.layer import NonlinearResistiveLayer
from model.variable.layer import LinearLayer
from model.resistive.digital_low_rank_config import Wan2022ProgrammingConfig
from training.device_programming import (
    program_wan2022_base_conductances,
)
from training.core.sgd import AugmentedFunction, EquilibriumProp
from training.core.optimizers import build_optimizer


_DIODE = {
    "quadratic_diode_param": {
        "diode_conductance": 1.0,
        "v_min": -1.0,
        "v_max": 1.0,
    },
    "exponential_diode_param": {
        "I_s": 1e-6,
        "V_t": 0.05,
        "V_off": 1.0,
    },
    "hard_sigmoid_param": {
        "g_on": 1.0,
        "g_off": 0.1,
        "v_min": -1.0,
        "v_max": 1.0,
    },
}


def _noise(seed: int) -> dict:
    return {
        "type": "aihwkit_reram_wan2022",
        "programming_seed": seed,
        "g_max_us": 40.0,
        "drn_conductance_at_g_max": 1.0,
        "noise_scale": 1.0,
        "t_inference_seconds": 1.0,
    }


def _layer_config(
    seed: int,
    *,
    rank: int,
    input_min: float,
    maximum: float,
) -> dict:
    return {
        "rank": rank,
        "input_factor_gain": 0.4,
        "input_factor_min": input_min,
        "conductance_max": maximum,
        "output_factor_init": "zero",
        "device_noise": _noise(seed),
    }


def _bundle():
    return build_deep_resistive_energy(
        layer_shapes=[(4,), (3,), (4,)],
        weight_gains=[0.2, 0.3],
        input_gain=1.0,
        non_linearity="linear",
        voltage_amp=1.0,
        current_amp=1.0,
        weight_min=0.01,
        weight_max=1.0,
        passive_layerwise_low_rank_adapter={
            "layers": {
                "base.dense_weight.0": _layer_config(
                    13,
                    rank=2,
                    input_min=0.02,
                    maximum=0.8,
                ),
                "base.dense_weight.1": _layer_config(
                    17,
                    rank=1,
                    input_min=0.03,
                    maximum=0.7,
                ),
            }
        },
        **_DIODE,
    )


def _copy(parameter, value) -> None:
    with torch.no_grad():
        parameter.state.copy_(
            torch.as_tensor(value, dtype=parameter.state.dtype).reshape_as(
                parameter.state
            )
        )


def _minimizer(fn, free_layers, iterations=250):
    return QuadraticMinimizer(
        fn=fn,
        free_layers=free_layers,
        num_iterations=iterations,
        mode="forward",
        non_linearity="linear",
        voltage_amp=1.0,
        current_amp=1.0,
        **_DIODE,
    )


def _known_conductances(energy):
    first_base, second_base, bias = energy.base_params()
    first_input, first_output, second_input, second_output = (
        energy.adapter_params()
    )
    _copy(
        first_base,
        [
            [0.4, 0.2, 0.3],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.4],
            [0.5, 0.2, 0.4],
        ],
    )
    _copy(
        second_base,
        [
            [0.3, 0.2, 0.4, 0.5],
            [0.5, 0.4, 0.2, 0.3],
            [0.2, 0.5, 0.3, 0.4],
        ],
    )
    bias.state.zero_()
    _copy(
        first_input,
        [
            [0.4, 0.2],
            [0.3, 0.5],
            [0.2, 0.4],
            [0.5, 0.3],
        ],
    )
    _copy(
        first_output,
        [
            [0.3, 0.2, 0.4],
            [0.2, 0.5, 0.3],
        ],
    )
    _copy(second_input, [[0.4], [0.2], [0.5]])
    _copy(second_output, [[0.3, 0.4, 0.2, 0.5]])
    return (
        first_base.state,
        second_base.state,
        first_input.state,
        first_output.state,
        second_input.state,
        second_output.state,
    )


def test_physical_layerwise_topology_catalog_and_individual_bounds() -> None:
    bundle = _bundle()
    energy = bundle.energy
    factors = energy.adapter_params()
    dense_interactions = [
        interaction
        for interaction in energy._interactions
        if isinstance(interaction, DenseResistive)
    ]

    assert [layer.shape for layer in energy.layers()] == [
        (4,),
        (2,),
        (3,),
        (1,),
        (4,),
    ]
    assert isinstance(energy.layers()[1], LinearLayer)
    assert isinstance(energy.layers()[2], NonlinearResistiveLayer)
    assert isinstance(energy.layers()[3], LinearLayer)
    assert len(dense_interactions) == 6
    assert all(len(interaction.params()) == 1 for interaction in dense_interactions)
    assert [parameter.shape for parameter in factors] == [
        (4, 2),
        (2, 3),
        (3, 1),
        (1, 4),
    ]
    assert all(
        isinstance(parameter, PassiveLowRankDenseWeight)
        for parameter in factors
    )
    assert [parameter.min_cond for parameter in factors] == [
        0.02,
        0.0,
        0.03,
        0.0,
    ]
    assert [parameter.max_cond for parameter in factors] == [
        0.8,
        0.8,
        0.7,
        0.7,
    ]
    assert all(torch.all(parameter.state >= 0.0) for parameter in factors)
    assert torch.count_nonzero(factors[1].state) == 0
    assert torch.count_nonzero(factors[3].state) == 0
    assert energy.params() == factors

    assert [binding.key for binding in bundle.catalog.all] == [
        "base.dense_weight.0",
        "base.dense_weight.1",
        "base.bias.0",
        "adapter.input_factor.0",
        "adapter.output_factor.0",
        "adapter.input_factor.1",
        "adapter.output_factor.1",
    ]
    assert [binding.key for binding in bundle.catalog.trainable] == [
        "adapter.input_factor.0",
        "adapter.output_factor.0",
        "adapter.input_factor.1",
        "adapter.output_factor.1",
    ]


def test_each_conductance_matrix_clips_its_own_parameter_state() -> None:
    energy = _bundle().energy
    first_base, second_base, _ = energy.base_params()
    factors = energy.adapter_params()
    for parameter in (first_base, second_base, *factors):
        parameter.state.flatten()[0] = -2.0
        parameter.state.flatten()[-1] = 2.0
        parameter.clamp_()

    for base in (first_base, second_base):
        assert float(base.state.min()) >= 0.01 - 1e-6
        assert float(base.state.max()) <= 1.0 + 1e-6
    for factor in factors:
        assert float(factor.state.min()) >= factor.min_cond - 1e-6
        assert float(factor.state.max()) <= factor.max_cond + 1e-6


def test_energy_and_coordinate_descent_match_explicit_five_layer_kcl() -> None:
    energy = _bundle().energy
    w1, w2, a1, b1, a2, b2 = _known_conductances(energy)
    inputs = torch.tensor(
        [[0.3, -0.2, 0.8, 0.1], [-0.6, 0.5, 0.1, 0.4]]
    )
    rank1 = torch.tensor([[0.1, 0.5], [-0.2, 0.3]])
    hidden = torch.tensor([[0.2, -0.1, 0.4], [0.3, 0.1, -0.2]])
    rank2 = torch.tensor([[0.25], [-0.15]])
    output = torch.tensor(
        [[0.7, -0.4, 0.1, 0.2], [0.2, 0.6, -0.3, 0.4]]
    )
    states = (inputs, rank1, hidden, rank2, output)
    for layer, state in zip(energy.layers(), states):
        layer.state = state.clone()

    expected_energy = 0.5 * (
        ((inputs.unsqueeze(2) - hidden.unsqueeze(1)).square() * w1).sum((1, 2))
        + ((hidden.unsqueeze(2) - output.unsqueeze(1)).square() * w2).sum((1, 2))
        + ((inputs.unsqueeze(2) - rank1.unsqueeze(1)).square() * a1).sum((1, 2))
        + ((rank1.unsqueeze(2) - hidden.unsqueeze(1)).square() * b1).sum((1, 2))
        + ((hidden.unsqueeze(2) - rank2.unsqueeze(1)).square() * a2).sum((1, 2))
        + ((rank2.unsqueeze(2) - output.unsqueeze(1)).square() * b2).sum((1, 2))
    )
    torch.testing.assert_close(energy.eval(), expected_energy)

    rank1_diag = torch.diag(a1.sum(dim=0) + b1.sum(dim=1))
    hidden_diag = torch.diag(
        w1.sum(dim=0)
        + w2.sum(dim=1)
        + b1.sum(dim=0)
        + a2.sum(dim=1)
    )
    rank2_diag = torch.diag(a2.sum(dim=0) + b2.sum(dim=1))
    output_diag = torch.diag(w2.sum(dim=0) + b2.sum(dim=0))
    zeros_r1_r2 = torch.zeros(a1.shape[1], a2.shape[1])
    zeros_r1_y = torch.zeros(a1.shape[1], w2.shape[1])
    zeros_r2_r1 = zeros_r1_r2.T
    zeros_y_r1 = zeros_r1_y.T
    kcl = torch.cat(
        (
            torch.cat(
                (rank1_diag, -b1, zeros_r1_r2, zeros_r1_y),
                dim=1,
            ),
            torch.cat(
                (-b1.T, hidden_diag, -a2, -w2),
                dim=1,
            ),
            torch.cat(
                (zeros_r2_r1, -a2.T, rank2_diag, -b2),
                dim=1,
            ),
            torch.cat(
                (zeros_y_r1, -w2.T, -b2.T, output_diag),
                dim=1,
            ),
        ),
        dim=0,
    )
    rhs = torch.cat(
        (
            inputs @ a1,
            inputs @ w1,
            torch.zeros(inputs.shape[0], a2.shape[1]),
            torch.zeros(inputs.shape[0], w2.shape[1]),
        ),
        dim=1,
    )
    expected = torch.linalg.solve(kcl, rhs.T).T
    for layer in energy.layers()[1:]:
        layer.state.zero_()
    _minimizer(
        energy,
        energy.layers()[1:],
        iterations=500,
    ).compute_equilibrium()

    widths = [a1.shape[1], w1.shape[1], a2.shape[1], w2.shape[1]]
    expected_states = torch.split(expected, widths, dim=1)
    for layer, expected_state in zip(
        energy.layers()[1:],
        expected_states,
    ):
        torch.testing.assert_close(
            layer.state,
            expected_state,
            rtol=3e-5,
            atol=3e-5,
        )


def test_ep_updates_only_individually_bounded_adapter_conductances() -> None:
    energy = _bundle().energy
    _known_conductances(energy)
    base_before = [
        parameter.state.clone() for parameter in energy.base_params()
    ]
    inputs = torch.tensor(
        [
            [0.6, 0.1, -0.6, -0.1],
            [0.1, 0.7, -0.1, -0.7],
        ]
    )
    energy.layers()[0].state = inputs
    for layer in energy.layers()[1:]:
        layer.init_state(inputs.shape[0], torch.device("cpu"))
    free_layers = energy.layers()[1:]
    _minimizer(energy, free_layers, iterations=40).compute_equilibrium()
    cost = SquaredErrorPairedOutputs(energy.layers()[-1], num_classes=2)
    cost.set_target(torch.tensor([0, 1]))
    augmented = AugmentedFunction(energy, cost)
    differentiator = EquilibriumProp(
        energy.params(),
        free_layers,
        augmented,
        cost,
        _minimizer(augmented, free_layers, iterations=40),
        variant="centered",
        nudging=0.05,
    )
    gradients = differentiator.compute_gradient()

    assert len(gradients) == 4
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    factor_before = [
        parameter.state.clone() for parameter in energy.adapter_params()
    ]
    optimizer = build_optimizer(
        energy,
        cost,
        [0.01, 0.01, 0.01, 0.01],
    )
    for parameter, gradient in zip(energy.params(), gradients):
        parameter.state.grad = gradient
    optimizer.step()
    for parameter in energy.params():
        parameter.clamp_()

    for parameter, expected in zip(energy.base_params(), base_before):
        torch.testing.assert_close(
            parameter.state,
            expected,
            rtol=0.0,
            atol=0.0,
        )
    assert any(
        not torch.equal(parameter.state, before)
        for parameter, before in zip(energy.adapter_params(), factor_before)
    )
    for factor in energy.adapter_params():
        assert float(factor.state.min()) >= factor.min_cond
        assert float(factor.state.max()) <= factor.max_cond


def test_two_matrix_wan_programming_is_stable_keyed_and_deterministic(
    monkeypatch,
) -> None:
    bundle = _bundle()
    catalog = bundle.catalog
    dense = [
        catalog.by_key["base.dense_weight.0"],
        catalog.by_key["base.dense_weight.1"],
    ]
    for index, binding in enumerate(dense):
        binding.state.fill_(0.2 + 0.1 * index)
    clean = [binding.state.clone() for binding in dense]
    factor_before = [
        binding.state.clone()
        for binding in catalog.for_group("adapter")
    ]

    class FakeWan:
        def __init__(self, *, g_max, noise_scale):
            self.g_max = g_max
            self.noise_scale = noise_scale

        def apply_drift_noise_to_conductance(self, first, second, age):
            torch.testing.assert_close(first, second)
            assert age == 1.0
            return first + self.noise_scale * torch.randn_like(first)

    root = ModuleType("aihwkit")
    root.__version__ = "test"
    inference = ModuleType("aihwkit.inference")
    noise = ModuleType("aihwkit.inference.noise")
    reram = ModuleType("aihwkit.inference.noise.reram")
    reram.ReRamWan2022NoiseModel = FakeWan
    monkeypatch.setitem(sys.modules, "aihwkit", root)
    monkeypatch.setitem(sys.modules, "aihwkit.inference", inference)
    monkeypatch.setitem(sys.modules, "aihwkit.inference.noise", noise)
    monkeypatch.setitem(sys.modules, "aihwkit.inference.noise.reram", reram)

    configs = {
        key: Wan2022ProgrammingConfig(**_noise(seed))
        for key, seed in (
            ("base.dense_weight.0", 31),
            ("base.dense_weight.1", 37),
        )
    }
    first = program_wan2022_base_conductances(catalog, configs)
    first_states = [binding.state.clone() for binding in dense]
    for binding, state in zip(dense, clean):
        binding.state.copy_(state)
    second = program_wan2022_base_conductances(catalog, configs)

    assert first == second
    assert first["parameter_keys"] == [
        "base.dense_weight.0",
        "base.dense_weight.1",
    ]
    assert [
        report["programming_seed"] for report in first["parameters"]
    ] == [31, 37]
    for binding, expected in zip(dense, first_states):
        torch.testing.assert_close(
            binding.state,
            expected,
            rtol=0.0,
            atol=0.0,
        )
    for binding, expected in zip(
        catalog.for_group("adapter"),
        factor_before,
    ):
        torch.testing.assert_close(
            binding.state,
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_two_matrix_programming_rejects_missing_stable_key() -> None:
    bundle = _bundle()
    config = Wan2022ProgrammingConfig(**_noise(3))
    with pytest.raises(ValueError, match="every checkpointed"):
        program_wan2022_base_conductances(
            bundle.catalog,
            {"base.dense_weight.0": config},
        )


def test_real_aihwkit_zero_noise_programming_is_neutral_for_both_edges() -> None:
    pytest.importorskip("aihwkit")
    bundle = _bundle()
    dense = [
        bundle.catalog.by_key["base.dense_weight.0"],
        bundle.catalog.by_key["base.dense_weight.1"],
    ]
    for binding in dense:
        binding.state.uniform_(0.01, 0.4)
    clean = [binding.state.clone() for binding in dense]
    configs = {
        key: Wan2022ProgrammingConfig(
            **{**_noise(seed), "noise_scale": 0.0}
        )
        for key, seed in (
            ("base.dense_weight.0", 41),
            ("base.dense_weight.1", 43),
        )
    }

    report = program_wan2022_base_conductances(bundle.catalog, configs)

    assert report["parameter_keys"] == [
        "base.dense_weight.0",
        "base.dense_weight.1",
    ]
    assert report["error_abs_max"] <= 1e-7
    for binding, expected in zip(dense, clean):
        torch.testing.assert_close(
            binding.state,
            expected,
            rtol=1e-6,
            atol=1e-7,
        )
