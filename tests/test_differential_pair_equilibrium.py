from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from labs.custom_minimizer import (
    CustomMinimizer,
    CustomQuadraticMinimizer,
    MinimizerSettings,
)
from model.resistive.layer import NonlinearResistiveLayer
from model.resistive.network import DeepResistiveEnergy
from model.variable.layer import LinearLayer
from model.function.cost import SquaredError
from training.core.sgd import AugmentedFunction


DTYPE = torch.float64


def _energy(
    *,
    voltage_amp: float,
    current_amp: float,
    non_linearity: str = "linear",
    num_layers: int = 3,
    differential_dense_edges: tuple[int, ...] | None = None,
    include_biases: bool = False,
) -> DeepResistiveEnergy:
    edge_count = num_layers - 1
    return DeepResistiveEnergy(
        layer_shapes=[(2,)] * num_layers,
        weight_gains=[0.1] * edge_count,
        input_gain=1.0,
        non_linearity=non_linearity,
        exponential_diode_param={},
        quadratic_diode_param={},
        hard_sigmoid_param={},
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        weight_min=0.0,
        weight_max=1.0,
        differential_dense_edges=(
            tuple(range(edge_count))
            if differential_dense_edges is None
            else differential_dense_edges
        ),
        include_biases=include_biases,
    )


def _set_pair_conductances(
    energy: DeepResistiveEnergy,
    pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    *,
    dtype: torch.dtype = DTYPE,
) -> None:
    parameters = energy._all_params
    assert len(parameters) == 2 * len(pairs)
    for parameter, value in zip(
        parameters,
        (value for pair in pairs for value in pair),
    ):
        parameter.state = value.to(dtype=dtype).clone()


def _settings() -> MinimizerSettings:
    return MinimizerSettings(
        rel_tol=1e-12,
        vn_tol=1e-12,
        use_polish=False,
        max_newton_iters=8,
        z_thresh=1e10,
        exp_clip=1e5,
        dynamic_polish=False,
        overrelaxation_reject_steps=False,
        overrelaxation_reject_max_tries=3,
        overrelaxation_reject_shrink=0.5,
        overrelaxation_reject_eps=0.0,
    )


def _minimizer(
    energy: DeepResistiveEnergy,
    *,
    non_linearity: str,
    num_iterations: int,
) -> CustomQuadraticMinimizer:
    return CustomQuadraticMinimizer(
        fn=energy,
        free_layers=list(energy.layers()[1:]),
        num_iterations=num_iterations,
        mode="forward",
        non_linearity=non_linearity,
        quadratic_diode_param={},
        exponential_diode_param={},
        voltage_amp=energy._voltage_amp,
        current_amp=energy._current_amp,
        hard_sigmoid_param={},
        iv_data=None,
        iv_data_path=None,
        double_diode_updater=None,
        adaptive_equilibrium=False,
        overrelaxation_factor=1.0,
        single_diode_updater=None,
        minimizer_settings=_settings(),
    )


@pytest.mark.parametrize(
    ("voltage_amp", "current_amp"),
    [(1.0, 1.0), (4.0, 0.25), (3.0, 2.0), (0.5, 2.0)],
)
def test_linear_equilibrium_converges_to_zero_physical_kcl_residual(
    voltage_amp: float,
    current_amp: float,
) -> None:
    energy = _energy(
        voltage_amp=voltage_amp,
        current_amp=current_amp,
    )
    _set_pair_conductances(
        energy,
        (
            (
                torch.tensor([[0.80, 0.15], [0.20, 0.70]]),
                torch.tensor([[0.30, 0.25], [0.10, 0.20]]),
            ),
            (
                torch.tensor([[0.65, 0.10], [0.25, 0.75]]),
                torch.tensor([[0.20, 0.30], [0.15, 0.35]]),
            ),
        ),
    )
    input_layer, hidden_layer, output_layer = energy.layers()
    input_layer.state = torch.tensor([[0.7, -0.4]], dtype=DTYPE)
    hidden_layer.state = torch.tensor([[0.9, -0.8]], dtype=DTYPE)
    output_layer.state = torch.tensor([[-0.6, 0.5]], dtype=DTYPE)
    minimizer = _minimizer(
        energy,
        non_linearity="linear",
        num_iterations=1,
    )

    initial_residuals = minimizer.residual_currents_inf()
    for layer in (hidden_layer, output_layer):
        expected_physical_current = (
            energy.grad_layer_fn(layer)()
            / energy.layer_energy_scale(layer)
        )
        assert initial_residuals[layer.name] == pytest.approx(
            expected_physical_current.abs().max().item()
        )
    initial = max(initial_residuals.values())
    sampled_residuals = []
    for sweep in range(1, 129):
        minimizer.compute_equilibrium()
        if sweep & (sweep - 1) == 0:
            sampled_residuals.append(
                max(minimizer.residual_currents_inf().values())
            )
    final_residuals = minimizer.residual_currents_inf()
    final = max(final_residuals.values())

    assert input_layer.name not in final_residuals
    assert set(final_residuals) == {hidden_layer.name, output_layer.name}
    assert sampled_residuals[0] < initial
    assert all(
        later <= earlier + 1e-14
        for earlier, later in zip(sampled_residuals, sampled_residuals[1:])
    )
    assert final < 1e-10


@pytest.mark.parametrize(
    ("voltage_amp", "current_amp"),
    [(4.0, 0.25), (3.0, 2.0), (0.5, 2.0)],
)
def test_perfect_diode_equilibrium_has_zero_projected_kkt_residual(
    voltage_amp: float,
    current_amp: float,
) -> None:
    energy = _energy(
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        non_linearity="perfect_diode",
    )
    diagonal_plus = torch.diag(torch.tensor([0.8, 0.8]))
    diagonal_minus = torch.diag(torch.tensor([0.2, 0.2]))
    common = torch.diag(torch.tensor([0.4, 0.4]))
    _set_pair_conductances(
        energy,
        ((diagonal_plus, diagonal_minus), (common, common)),
    )
    input_layer, hidden_layer, output_layer = energy.layers()
    input_layer.state = torch.tensor([[-1.0, 1.0]], dtype=DTYPE)
    hidden_layer.state = torch.tensor([[0.3, -0.3]], dtype=DTYPE)
    output_layer.state = torch.tensor([[0.2, -0.2]], dtype=DTYPE)
    minimizer = _minimizer(
        energy,
        non_linearity="perfect_diode",
        num_iterations=8,
    )

    minimizer.compute_equilibrium()
    raw_hidden_current = (
        energy.grad_layer_fn(hidden_layer)()
        / energy.layer_energy_scale(hidden_layer)
    )
    projected_residuals = minimizer.residual_currents_inf()

    torch.testing.assert_close(
        hidden_layer.state,
        torch.zeros_like(hidden_layer.state),
    )
    assert raw_hidden_current.abs().max().item() > 0.1
    assert max(projected_residuals.values()) < 1e-10
    assert input_layer.name not in projected_residuals
    assert output_layer.name in projected_residuals


def test_float32_linear_equilibrium_reaches_numerical_kcl_tolerance() -> None:
    energy = _energy(voltage_amp=3.0, current_amp=2.0)
    _set_pair_conductances(
        energy,
        (
            (
                torch.tensor([[0.80, 0.15], [0.20, 0.70]]),
                torch.tensor([[0.30, 0.25], [0.10, 0.20]]),
            ),
            (
                torch.tensor([[0.65, 0.10], [0.25, 0.75]]),
                torch.tensor([[0.20, 0.30], [0.15, 0.35]]),
            ),
        ),
        dtype=torch.float32,
    )
    input_layer, hidden_layer, output_layer = energy.layers()
    input_layer.state = torch.tensor([[0.7, -0.4]], dtype=torch.float32)
    hidden_layer.state = torch.tensor([[0.9, -0.8]], dtype=torch.float32)
    output_layer.state = torch.tensor([[-0.6, 0.5]], dtype=torch.float32)
    minimizer = _minimizer(
        energy,
        non_linearity="linear",
        num_iterations=128,
    )

    minimizer.compute_equilibrium()
    assert max(minimizer.residual_currents_inf().values()) < 1e-6


def test_float32_projected_residual_preserves_small_interior_current() -> None:
    diagnostic = object.__new__(CustomMinimizer)
    current = torch.tensor([[1e-8, -1e-8]], dtype=torch.float32)

    linear = LinearLayer((2,), batch_size=1, device="cpu")
    linear.state = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    linear_result = diagnostic._projected_current_residual(
        SimpleNamespace(_layer=linear, _a=lambda: torch.full((1, 2), 0.5)),
        current,
        1.0,
    )
    torch.testing.assert_close(linear_result, current, rtol=0.0, atol=0.0)

    diode = NonlinearResistiveLayer(
        (2,),
        batch_size=1,
        device="cpu",
        non_linearity="perfect_diode",
    )
    diode.state = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    diode_result = diagnostic._projected_current_residual(
        SimpleNamespace(_layer=diode, _a=lambda: torch.full((1, 2), 0.5)),
        current,
        1.0,
    )
    torch.testing.assert_close(diode_result, current, rtol=0.0, atol=0.0)


def test_layer_energy_metric_follows_current_to_voltage_gain_ratio() -> None:
    energy = _energy(
        voltage_amp=3.0,
        current_amp=2.0,
        num_layers=4,
    )
    input_layer, first, second, third = energy.layers()
    assert energy.layer_energy_scale(input_layer) == pytest.approx(1.0)
    assert energy.layer_energy_scale(first) == pytest.approx(1.0)
    assert energy.layer_energy_scale(second) == pytest.approx(2.0 / 3.0)
    assert energy.layer_energy_scale(third) == pytest.approx((2.0 / 3.0) ** 2)

    copied = energy.to("cpu")
    assert copied.layer_energy_scale(copied.layers()[2]) == pytest.approx(
        2.0 / 3.0
    )


def test_current_nudging_uses_output_layer_metric() -> None:
    energy = _energy(voltage_amp=3.0, current_amp=2.0)
    output_layer = energy.layers()[-1]
    augmented = AugmentedFunction(
        energy,
        SquaredError(output_layer),
        nudging_mode="current",
        current_scale="auto",
    )
    assert augmented._current_scale == pytest.approx(
        energy.layer_energy_scale(output_layer)
    )


def test_partial_differential_topology_fails_closed() -> None:
    with pytest.raises(ValueError, match="select every dense edge"):
        _energy(
            voltage_amp=3.0,
            current_amp=2.0,
            differential_dense_edges=(0,),
        )


@pytest.mark.parametrize(
    "edges",
    [0, (True,), (-1,), (0, 0), ("0",)],
)
def test_invalid_differential_edge_selection_fails_closed(edges: object) -> None:
    with pytest.raises(ValueError, match="unique non-negative integer"):
        _energy(
            voltage_amp=3.0,
            current_amp=2.0,
            differential_dense_edges=edges,
        )


def test_differential_biases_fail_closed() -> None:
    with pytest.raises(ValueError, match="include_biases=false"):
        _energy(
            voltage_amp=3.0,
            current_amp=2.0,
            include_biases=True,
        )


@pytest.mark.parametrize("include_biases", [0, 1, "false", None])
def test_include_biases_requires_an_explicit_boolean(
    include_biases: object,
) -> None:
    with pytest.raises(ValueError, match="include_biases to be a bool"):
        _energy(
            voltage_amp=3.0,
            current_amp=2.0,
            include_biases=include_biases,
        )


def test_unscaled_nonlinearity_fails_closed() -> None:
    with pytest.raises(ValueError, match="'linear' or 'perfect_diode'"):
        _energy(
            voltage_amp=3.0,
            current_amp=2.0,
            non_linearity="hard_sigmoid",
        )


def test_process_global_amplifier_indexing_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match="legacy_process_index_amplification=false",
    ):
        DeepResistiveEnergy(
            layer_shapes=[(2,), (2,), (2,)],
            weight_gains=[0.1, 0.1],
            input_gain=1.0,
            non_linearity="linear",
            exponential_diode_param={},
            quadratic_diode_param={},
            hard_sigmoid_param={},
            voltage_amp=3.0,
            current_amp=2.0,
            weight_min=0.0,
            weight_max=1.0,
            differential_dense_edges=(0, 1),
            include_biases=False,
            legacy_process_index_amplification=True,
        )


def test_differential_network_requires_one_gain_per_edge() -> None:
    with pytest.raises(ValueError, match="one gain per dense edge"):
        DeepResistiveEnergy(
            layer_shapes=[(2,), (2,), (2,)],
            weight_gains=[0.1],
            input_gain=1.0,
            non_linearity="linear",
            exponential_diode_param={},
            quadratic_diode_param={},
            hard_sigmoid_param={},
            voltage_amp=3.0,
            current_amp=2.0,
            weight_min=0.0,
            weight_max=1.0,
            differential_dense_edges=(0, 1),
            include_biases=False,
        )


def test_unrepresentable_layer_metric_fails_closed() -> None:
    with pytest.raises(ValueError, match="finite positive layer-energy scale"):
        _energy(
            voltage_amp=1e30,
            current_amp=1e-30,
            num_layers=3,
        )


def test_unrepresentable_interior_energy_scale_fails_closed() -> None:
    with pytest.raises(ValueError, match="energy coefficient scales"):
        _energy(
            voltage_amp=1e20,
            current_amp=1e20,
        )


@pytest.mark.parametrize(
    ("weight_min", "weight_max"),
    [(-1.0, 1.0), (0.0, -0.1), (0.2, 0.2), (0.2, 0.1)],
)
def test_differential_conductance_bounds_fail_closed(
    weight_min: float,
    weight_max: float,
) -> None:
    with pytest.raises(ValueError, match="weight_(?:min|max)"):
        DeepResistiveEnergy(
            layer_shapes=[(2,), (2,), (2,)],
            weight_gains=[0.1, 0.1],
            input_gain=1.0,
            non_linearity="linear",
            exponential_diode_param={},
            quadratic_diode_param={},
            hard_sigmoid_param={},
            voltage_amp=3.0,
            current_amp=2.0,
            weight_min=weight_min,
            weight_max=weight_max,
            differential_dense_edges=(0, 1),
            include_biases=False,
        )
