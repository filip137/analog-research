from __future__ import annotations

import pytest
import torch

from model.resistive.interaction import DenseResistive
from model.resistive.low_rank import (
    PassiveLowRankAdapterConfig,
    PassiveLowRankDenseWeight,
)
from model.resistive.minimizer import QuadraticMinimizer
from model.resistive.network import DeepResistiveEnergy
from model.variable.layer import LinearLayer
from model.variable.parameter import Bias, DenseWeight
from training.core.optimizers import build_optimizer
from training.tiki_taka import TikiTakaOptimizer


_QUADRATIC_DIODE_PARAM = {
    "diode_conductance": 10.0,
    "v_min": -1.0,
    "v_max": 1.0,
}
_EXPONENTIAL_DIODE_PARAM = {
    "I_s": 1e-6,
    "V_t": 0.05,
    "V_off": 0.5,
}
_HARD_SIGMOID_PARAM = {
    "g_on": 10.0,
    "g_off": 10.0,
    "v_min": -1.0,
    "v_max": 1.0,
}
_DEFAULT_ADAPTER = object()


def _adapter_config(**overrides) -> PassiveLowRankAdapterConfig:
    values = {
        "rank": 2,
        "input_factor_gain": 0.4,
        "input_factor_min": 0.02,
        "conductance_max": 2.0,
        "output_factor_init": "zero",
        "output_off_conductance": None,
    }
    values.update(overrides)
    return PassiveLowRankAdapterConfig(**values)


def _build_energy(
    *,
    adapter=_DEFAULT_ADAPTER,
    layer_shapes=((3,), (2,)),
    weight_gains=None,
    voltage_amp=1.0,
    current_amp=1.0,
    conv_pipeline=None,
) -> DeepResistiveEnergy:
    if adapter is _DEFAULT_ADAPTER:
        adapter = _adapter_config()
    if weight_gains is None:
        weight_gains = [0.3] * (len(layer_shapes) - 1)
    energy = DeepResistiveEnergy(
        layer_shapes=list(layer_shapes),
        weight_gains=weight_gains,
        input_gain=1.0,
        non_linearity="linear",
        quadratic_diode_param=_QUADRATIC_DIODE_PARAM,
        exponential_diode_param=_EXPONENTIAL_DIODE_PARAM,
        hard_sigmoid_param=_HARD_SIGMOID_PARAM,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        weight_min=0.0,
        weight_max=2.0,
        conv_pipeline=conv_pipeline,
        passive_low_rank_adapter=adapter,
    )
    energy.set_device(torch.device("cpu"))
    return energy


def _copy_state(parameter, value) -> None:
    with torch.no_grad():
        parameter.state.copy_(
            torch.as_tensor(
                value,
                dtype=parameter.state.dtype,
            ).reshape_as(parameter.state)
        )


def _set_fixed_input(energy: DeepResistiveEnergy, inputs) -> None:
    values = torch.as_tensor(inputs, dtype=torch.float32)
    energy.layers()[0].state = values.clone()
    for layer in energy.layers()[1:]:
        layer.init_state(values.shape[0], torch.device("cpu"))


def _linear_minimizer(
    function,
    *,
    num_iterations: int = 250,
) -> QuadraticMinimizer:
    return QuadraticMinimizer(
        fn=function,
        free_layers=function.layers()[1:],
        num_iterations=num_iterations,
        mode="forward",
        non_linearity="linear",
        quadratic_diode_param=_QUADRATIC_DIODE_PARAM,
        exponential_diode_param=_EXPONENTIAL_DIODE_PARAM,
        hard_sigmoid_param=_HARD_SIGMOID_PARAM,
        voltage_amp=1.0,
        current_amp=1.0,
    )


def _known_conductances(energy: DeepResistiveEnergy):
    base_weight = energy.base_params()[0]
    input_factor, output_factor = energy.adapter_params()
    _copy_state(
        base_weight,
        [[1.0, 0.4], [0.3, 1.2], [0.7, 0.5]],
    )
    _copy_state(
        input_factor,
        [[0.8, 0.2], [0.4, 1.0], [0.6, 0.3]],
    )
    _copy_state(output_factor, [[0.5, 0.9], [0.7, 0.4]])
    return base_weight.state, input_factor.state, output_factor.state


def test_adapter_topology_roles_bounds_and_initialization() -> None:
    energy = _build_energy(adapter=_adapter_config(rank=3))
    base_params = energy.base_params()
    adapter_params = energy.adapter_params()

    assert [layer.shape for layer in energy.layers()] == [(3,), (3,), (2,)]
    assert isinstance(energy.layers()[1], LinearLayer)
    assert energy.layers()[-1].shape == (2,)
    assert len(base_params) == 1
    assert isinstance(base_params[0], DenseWeight)
    assert [parameter.adapter_role for parameter in adapter_params] == [
        "input_factor",
        "output_factor",
    ]
    assert all(
        isinstance(parameter, PassiveLowRankDenseWeight)
        for parameter in adapter_params
    )
    assert [parameter.shape for parameter in adapter_params] == [
        (3, 3),
        (3, 2),
    ]
    assert energy.params() == adapter_params
    assert energy._params == base_params + adapter_params
    assert not any(isinstance(parameter, Bias) for parameter in energy._params)
    assert all(
        isinstance(interaction, DenseResistive)
        for interaction in energy._interactions
    )
    assert torch.all(adapter_params[0].state >= 0.02)
    assert torch.count_nonzero(adapter_params[1].state) == 0

    off_energy = _build_energy(
        adapter=_adapter_config(
            output_factor_init="off_conductance",
            output_off_conductance=0.07,
        )
    )
    off_factor = off_energy.adapter_params()[1]
    torch.testing.assert_close(
        off_factor.state,
        torch.full_like(off_factor.state, 0.07),
    )
    assert off_factor.min_cond == pytest.approx(0.07)


@pytest.mark.parametrize(
    "adapter",
    [
        {
            "rank": 2,
            "input_factor_gain": 0.4,
            "input_factor_min": 0.02,
            "conductance_max": 2.0,
            "output_factor_init": "zero",
            "unknown": 1,
        },
        {
            "rank": 2,
            "input_factor_gain": 0.4,
            "input_factor_min": 0.02,
            "conductance_max": 2.0,
        },
        {
            "rank": 0,
            "input_factor_gain": 0.4,
            "input_factor_min": 0.02,
            "conductance_max": 2.0,
            "output_factor_init": "zero",
        },
        {
            "rank": 2,
            "input_factor_gain": 0.4,
            "input_factor_min": 2.0,
            "conductance_max": 2.0,
            "output_factor_init": "zero",
        },
        {
            "rank": 2,
            "input_factor_gain": 0.4,
            "input_factor_min": 0.02,
            "conductance_max": 2.0,
            "output_factor_init": "off_conductance",
        },
    ],
)
def test_adapter_config_validation_reports_expected_then_provided(
    adapter,
) -> None:
    with pytest.raises(ValueError) as raised:
        _build_energy(adapter=adapter)

    message = str(raised.value)
    assert message.index("Expected") < message.index("Provided value:")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_shapes": ((2, 2), (2,))},
        {
            "layer_shapes": ((3,), (4,), (2,)),
            "weight_gains": [0.3, 0.3],
        },
        {"voltage_amp": 2.0},
        {"current_amp": 0.5},
        {
            "layer_shapes": ((1, 2, 2), (1, 1, 1), (2,)),
            "weight_gains": [0.3, 0.3],
            "conv_pipeline": [
                {
                    "mode": "convolution",
                    "kernel": (2, 2),
                    "stride": 1,
                    "padding": 0,
                }
            ],
        },
    ],
)
def test_adapter_rejects_unsupported_topology(kwargs) -> None:
    with pytest.raises(ValueError) as raised:
        _build_energy(**kwargs)

    message = str(raised.value)
    assert "Expected" in message
    assert "Provided value:" in message


def test_energy_and_coordinate_descent_match_explicit_block_kcl() -> None:
    energy = _build_energy()
    base, input_factor, output_factor = _known_conductances(energy)
    inputs = torch.tensor([[0.3, -0.2, 0.8], [-0.6, 0.5, 0.1]])
    rank_state = torch.tensor([[0.1, 0.5], [-0.2, 0.3]])
    output_state = torch.tensor([[0.7, -0.4], [0.2, 0.6]])
    _set_fixed_input(energy, inputs)
    energy.layers()[1].state = rank_state.clone()
    energy.layers()[2].state = output_state.clone()

    expected_energy = 0.5 * (
        (
            (inputs.unsqueeze(2) - output_state.unsqueeze(1)).square()
            * base
        ).sum((1, 2))
        + (
            (inputs.unsqueeze(2) - rank_state.unsqueeze(1)).square()
            * input_factor
        ).sum((1, 2))
        + (
            (rank_state.unsqueeze(2) - output_state.unsqueeze(1)).square()
            * output_factor
        ).sum((1, 2))
    )
    torch.testing.assert_close(energy.eval(), expected_energy)

    rank_diagonal = torch.diag(
        input_factor.sum(dim=0) + output_factor.sum(dim=1)
    )
    output_diagonal = torch.diag(
        base.sum(dim=0) + output_factor.sum(dim=0)
    )
    kcl_matrix = torch.cat(
        (
            torch.cat((rank_diagonal, -output_factor), dim=1),
            torch.cat(
                (-output_factor.transpose(0, 1), output_diagonal),
                dim=1,
            ),
        ),
        dim=0,
    )
    rhs = torch.cat(
        (inputs @ input_factor, inputs @ base),
        dim=1,
    )
    expected = torch.linalg.solve(kcl_matrix, rhs.transpose(0, 1)).T

    for layer in energy.layers()[1:]:
        layer.state.zero_()
    _linear_minimizer(energy).compute_equilibrium()
    torch.testing.assert_close(
        energy.layers()[1].state,
        expected[:, : input_factor.shape[1]],
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        energy.layers()[-1].state,
        expected[:, input_factor.shape[1] :],
        rtol=2e-5,
        atol=2e-5,
    )


def test_kron_reduction_is_low_rank_but_not_naive_algebraic_lora() -> None:
    energy = _build_energy()
    base, input_factor, output_factor = _known_conductances(energy)
    inputs = torch.tensor([[0.3, -0.2, 0.8], [-0.6, 0.5, 0.1]])

    rank_inverse = torch.diag(
        1.0 / (input_factor.sum(dim=0) + output_factor.sum(dim=1))
    )
    cross_term = input_factor @ rank_inverse @ output_factor
    output_loading = (
        torch.diag(output_factor.sum(dim=0))
        - output_factor.transpose(0, 1)
        @ rank_inverse
        @ output_factor
    )
    assert torch.linalg.matrix_rank(cross_term).item() <= input_factor.shape[1]
    torch.testing.assert_close(
        output_loading,
        output_loading.transpose(0, 1),
    )
    assert output_loading[0, 1] < 0.0

    _set_fixed_input(energy, inputs)
    _linear_minimizer(energy).compute_equilibrium()
    naive = base + cross_term
    naive_output = (inputs @ naive) / naive.sum(dim=0)
    assert not torch.allclose(
        energy.layers()[-1].state,
        naive_output,
        rtol=1e-3,
        atol=1e-3,
    )


def test_zero_output_factor_is_output_neutral_with_finite_rank_nodes() -> None:
    base_energy = _build_energy(adapter=None)
    adapted_energy = _build_energy()
    base = torch.tensor([[1.0, 0.4], [0.3, 1.2], [0.7, 0.5]])
    input_factor = torch.tensor(
        [[0.8, 0.2], [0.4, 1.0], [0.6, 0.3]]
    )
    _copy_state(base_energy.base_params()[0], base)
    _copy_state(adapted_energy.base_params()[0], base)
    _copy_state(adapted_energy.adapter_params()[0], input_factor)
    _copy_state(adapted_energy.adapter_params()[1], torch.zeros(2, 2))
    inputs = torch.tensor([[0.3, -0.2, 0.8], [-0.6, 0.5, 0.1]])

    _set_fixed_input(base_energy, inputs)
    _set_fixed_input(adapted_energy, inputs)
    _linear_minimizer(base_energy, num_iterations=4).compute_equilibrium()
    _linear_minimizer(adapted_energy, num_iterations=4).compute_equilibrium()

    torch.testing.assert_close(
        adapted_energy.layers()[-1].state,
        base_energy.layers()[-1].state,
    )
    expected_rank = (inputs @ input_factor) / input_factor.sum(dim=0)
    torch.testing.assert_close(
        adapted_energy.layers()[1].state,
        expected_rank,
    )
    assert torch.isfinite(adapted_energy.layers()[1].state).all()


class _NoParameters:
    @staticmethod
    def params():
        return []


@pytest.mark.parametrize("pipeline", ["direct", "tiki_taka"])
def test_only_adapter_factors_update_with_direct_and_ideal_tiki(
    pipeline: str,
) -> None:
    energy = _build_energy()
    base_before = energy.base_params()[0].state.clone()
    adapter_before = [
        parameter.state.clone() for parameter in energy.adapter_params()
    ]
    update_pipeline = None
    if pipeline == "tiki_taka":
        update_pipeline = {
            "type": "tiki_taka",
            "fast_lr": 1.0,
            "transfer_every": 1,
            "n_reads_per_transfer": 64,
            "transfer_lr": 0.05,
            "scale_transfer_lr": False,
            "with_reset_prob": 1.0,
        }
    optimizer = build_optimizer(
        energy,
        _NoParameters(),
        [0.05, 0.05],
        update_pipeline=update_pipeline,
    )
    if pipeline == "tiki_taka":
        assert isinstance(optimizer, TikiTakaOptimizer)

    for parameter in energy.params():
        parameter.state.grad = -torch.ones_like(parameter.state)
    optimizer.step()
    for parameter in energy.params():
        parameter.clamp_()

    assert torch.equal(energy.base_params()[0].state, base_before)
    assert all(
        not torch.equal(parameter.state, before)
        for parameter, before in zip(
            energy.adapter_params(),
            adapter_before,
        )
    )
