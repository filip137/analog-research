import pytest
import torch

from networks.resistive.solver import CustomQuadraticMinimizer, MinimizerSettings
from networks.function.interaction import DoubleQuadraticNonLinearInteraction
from networks.resistive.layer import NonlinearResistiveLayer, ResistiveInputLayer
from networks.resistive.minimizer import QuadraticMinimizer
from networks.variable.layer import Layer


VOLTAGE_AMP = 4.0
CURRENT_AMP = 0.25
EXPECTED_DEPTH_FACTORS = [1.0, 1.0 / 16.0, 1.0 / 256.0]


class _LayerwiseQuadraticEnergy:
    def __init__(self, layers, *, a_by_layer=None, b_by_layer=None):
        self._layers = list(layers)
        self._a_by_layer = a_by_layer or {layer: 1.0 for layer in layers}
        self._b_by_layer = b_by_layer or {layer: 0.0 for layer in layers}

    def layers(self):
        return self._layers

    def params(self):
        return []

    def a_coef_fn(self, layer):
        return lambda: torch.full_like(layer.state, self._a_by_layer[layer])

    def b_coef_fn(self, layer):
        return lambda: torch.full_like(layer.state, self._b_by_layer[layer])

    def grad_layer_fn(self, layer):
        return lambda: (
            2.0 * self.a_coef_fn(layer)() * layer.state
            + self.b_coef_fn(layer)()
        )

    def eval(self):
        return sum(
            (
                self._a_by_layer[layer] * layer.state.square()
                + self._b_by_layer[layer] * layer.state
            ).flatten(start_dim=1).sum(dim=1)
            for layer in self._layers
        )


def _make_hidden_layers(non_linearity):
    Layer._counter = 0
    ResistiveInputLayer((2,), gain=1.0, batch_size=1, device="cpu")
    return [
        NonlinearResistiveLayer(
            (2,),
            batch_size=1,
            device="cpu",
            non_linearity=non_linearity,
        )
        for _ in range(3)
    ]


@pytest.mark.parametrize(
    ("non_linearity", "strength_attr", "base_strength"),
    [
        ("lpw_diode", "_diode_conductance", 10.0),
        ("double_diode_quadratic", "_diode_conductance", 10.0),
        ("double_diode_exponential", "_Is", 1.0e-6),
        ("single_diode_exponential", "_Is", 1.0e-6),
    ],
)
def test_core_specialized_updaters_scale_only_diode_strength_by_depth(
    non_linearity,
    strength_attr,
    base_strength,
):
    layers = _make_hidden_layers(non_linearity)
    energy = _LayerwiseQuadraticEnergy(layers)
    quadratic_params = {"diode_conductance": 10.0, "v_off": 1.25}
    exponential_params = {"I_s": 1.0e-6, "V_t": 0.05, "V_off": 0.5}

    minimizer = QuadraticMinimizer(
        energy,
        layers,
        num_iterations=1,
        mode="asynchronous",
        non_linearity=non_linearity,
        quadratic_diode_param=quadratic_params,
        exponential_diode_param=exponential_params,
        voltage_amp=VOLTAGE_AMP,
        current_amp=CURRENT_AMP,
    )

    assert [layer.name for layer in layers] == ["Layer_1", "Layer_2", "Layer_3"]
    assert [
        getattr(updater, strength_attr) for updater in minimizer._updaters
    ] == pytest.approx(
        [base_strength * factor for factor in EXPECTED_DEPTH_FACTORS]
    )

    if strength_attr == "_diode_conductance":
        assert [updater._v_off for updater in minimizer._updaters] == pytest.approx(
            [1.25, 1.25, 1.25]
        )
        assert quadratic_params == {"diode_conductance": 10.0, "v_off": 1.25}
    else:
        assert [updater._Vt for updater in minimizer._updaters] == pytest.approx(
            [0.05, 0.05, 0.05]
        )
        assert [updater._v_off for updater in minimizer._updaters] == pytest.approx(
            [0.5, 0.5, 0.5]
        )
        assert exponential_params == {"I_s": 1.0e-6, "V_t": 0.05, "V_off": 0.5}


def test_quadratic_updater_is_stationary_for_physical_kcl_energy_at_each_depth():
    layers = _make_hidden_layers("double_diode_quadratic")
    a_by_layer = {
        layer: 2.0 * factor
        for layer, factor in zip(layers, EXPECTED_DEPTH_FACTORS)
    }
    b_by_layer = {
        layer: -10.0 * factor
        for layer, factor in zip(layers, EXPECTED_DEPTH_FACTORS)
    }
    energy = _LayerwiseQuadraticEnergy(
        layers,
        a_by_layer=a_by_layer,
        b_by_layer=b_by_layer,
    )
    params = {"diode_conductance": 1.0, "v_off": 1.0}
    nonlinear_interactions = [
        DoubleQuadraticNonLinearInteraction(
            layer,
            params,
            voltage_amp=VOLTAGE_AMP,
            current_amp=CURRENT_AMP,
        )
        for layer in layers
    ]
    minimizer = QuadraticMinimizer(
        energy,
        layers,
        num_iterations=1,
        mode="asynchronous",
        non_linearity="double_diode_quadratic",
        quadratic_diode_param=params,
        exponential_diode_param={"I_s": 1.0e-6, "V_t": 0.05, "V_off": 0.5},
        voltage_amp=VOLTAGE_AMP,
        current_amp=CURRENT_AMP,
    )

    updated_states = [updater.pre_activate() for updater in minimizer._updaters]
    for layer, updated_state, nonlinear_interaction in zip(
        layers,
        updated_states,
        nonlinear_interactions,
    ):
        layer.state = updated_state
        total_gradient = (
            energy.grad_layer_fn(layer)()
            + nonlinear_interaction.grad_layer_fn(layer)()
        )
        assert torch.allclose(
            total_gradient,
            torch.zeros_like(total_gradient),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    assert torch.allclose(updated_states[0], updated_states[1])
    assert torch.allclose(updated_states[1], updated_states[2])


def test_custom_exponential_factory_uses_physical_kcl_depth_factors():
    layers = _make_hidden_layers("double_diode_exponential")
    energy = _LayerwiseQuadraticEnergy(layers)
    exponential_params = {"I_s": 1.0e-6, "V_t": 0.05, "V_off": 0.5}

    minimizer = CustomQuadraticMinimizer(
        energy,
        layers,
        num_iterations=1,
        mode="asynchronous",
        non_linearity="double_diode_exponential",
        quadratic_diode_param={"diode_conductance": 10.0, "v_off": 1.0},
        exponential_diode_param=exponential_params,
        voltage_amp=VOLTAGE_AMP,
        current_amp=CURRENT_AMP,
        hard_sigmoid_param={},
        iv_data=None,
        iv_data_path=None,
        double_diode_updater="custom",
        adaptive_equilibrium=False,
        overrelaxation_factor=1.0,
        single_diode_updater="custom",
        minimizer_settings=MinimizerSettings(
            rel_tol=1.0e-5,
            vn_tol=1.0e-6,
            use_polish=True,
            max_newton_iters=64,
            z_thresh=1.0e10,
            exp_clip=1.0e4,
            dynamic_polish=False,
            overrelaxation_reject_steps=False,
            overrelaxation_reject_max_tries=3,
            overrelaxation_reject_shrink=0.5,
            overrelaxation_reject_eps=0.0,
        ),
    )

    assert [updater._Is for updater in minimizer._updaters] == pytest.approx(
        [exponential_params["I_s"] * factor for factor in EXPECTED_DEPTH_FACTORS]
    )
    assert [updater._Vt for updater in minimizer._updaters] == pytest.approx(
        [0.05, 0.05, 0.05]
    )
    assert [updater._v_off for updater in minimizer._updaters] == pytest.approx(
        [0.5, 0.5, 0.5]
    )
