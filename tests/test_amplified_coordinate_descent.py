"""Amplified circuit equations and actual equilibrium residuals.

The dense/conv convention follows lean-experiment-workflow/model:
E_edge = 0.5 * (Ai/Av)**p * g * (alpha*v_pre - v_post)**2.
Pooling uses the same convention rather than the reference's legacy scaling.
"""
from __future__ import annotations

import pytest
import torch

from labs.custom_classes import ConvResistive as FlexibleConvResistive
from labs.custom_minimizer import CustomQuadraticMinimizer, MinimizerSettings
from model.function.cost import SquaredError
from model.resistive.interaction import (
    AveragePoolResistive, ConvResistive, DenseResistive, MaxPoolResistive,
)
from model.resistive.minimizer import QuadraticMinimizer
from model.resistive.network import DeepResistiveEnergy
from model.variable.layer import LinearLayer
from model.variable.parameter import Bias, ConvWeight, DenseWeight, PoolWeight
from training.core.sgd import AugmentedFunction


DTYPE = torch.float64
QUADRATIC = {"diode_conductance": 10.0, "v_off": 0.05}
EXPONENTIAL = {"I_s": 0.01, "V_t": 0.3, "V_off": 0.05}
HARD = {"g_on": 10.0, "g_off": 0.0, "v_min": -0.05, "v_max": 0.05}


@pytest.mark.parametrize("kind", ["dense", "conv", "flexible_conv", "avg_pool", "max_pool"])
@pytest.mark.parametrize("depth", [0, 1, 2, 10])
@pytest.mark.parametrize("av,ai", [(1., 1.), (4., .25), (1., 4.), (2., 3.)])
def test_edge_coefficients_and_weight_gradients_are_energy_derivatives(kind, depth, av, ai):
    generator = torch.Generator().manual_seed(39)
    if kind == "dense":
        pre, post = LinearLayer((2,)), LinearLayer((2,))
        weight = DenseWeight(pre.shape, post.shape, 1., "cpu", clamp=True)
    else:
        pre, post = LinearLayer((2, 4, 4)), LinearLayer((2, 2, 2))
        weight_cls = PoolWeight if "pool" in kind else ConvWeight
        weight = weight_cls((2, 2, 2, 2), 1., "cpu", clamp=True, clamp_min=0., clamp_max=10.)
    pre._logical_index, post._logical_index = depth, depth + 1
    for layer in (pre, post):
        layer.state = torch.randn((2, *layer.shape), generator=generator, dtype=DTYPE, requires_grad=True)
    weight.state = weight.state.to(DTYPE)
    if "pool" not in kind:
        weight.state = torch.rand(weight.state.shape, generator=generator, dtype=DTYPE) + .1
    weight.state.requires_grad_(True)
    if kind == "dense":
        edge = DenseResistive(pre, post, weight, av, ai)
        alpha = 1. if depth == 0 else av
        expected = .5 * (ai / av)**depth * (
            (alpha * pre.state.unsqueeze(-1) - post.state.unsqueeze(1)).square()
            * weight.get()
        ).sum((1, 2))
        torch.testing.assert_close(edge.eval(), expected, rtol=1e-12, atol=1e-12)
    elif "pool" in kind:
        cls = AveragePoolResistive if kind == "avg_pool" else MaxPoolResistive
        edge = cls(pre, post, weight, 2, av, ai)
    else:
        cls = ConvResistive if kind == "conv" else FlexibleConvResistive
        edge = cls(pre, post, weight, 0, 2, 1, av, ai)
    grad_pre, grad_post, grad_weight = torch.autograd.grad(
        edge.eval().sum(), (pre.state, post.state, weight.state), allow_unused=True,
    )
    for layer, expected in ((pre, grad_pre), (post, grad_post)):
        actual = 2 * edge.a_coef_fn(layer)() * layer.state + edge.b_coef_fn(layer)()
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    if "pool" not in kind:
        torch.testing.assert_close(edge.grad_param_fn(weight)(), grad_weight / 2, rtol=1e-11, atol=1e-12)


def make_energy(kind, nonlinearity, av, ai, dtype=DTYPE):
    if kind == "dense":
        shapes, pipeline = [(2,)] * 4, None
    elif kind == "conv":
        shapes = [(2, 4, 4), (2, 3, 3), (2, 2, 2), (2,)]
        pipeline = [dict(kernel=(2, 2), stride=1, padding=0, mode="convolution")] * 2
    else:
        shapes = [(2, 4, 4), (2, 3, 3), (2, 1, 1), (2,)]
        pipeline = [dict(kernel=(2, 2), stride=1, padding=0, mode="convolution"),
                    dict(kernel=(2, 2), stride=2, padding=0, mode="pooling")]
    energy = DeepResistiveEnergy(
        shapes, [1.] * 3, 1., nonlinearity, EXPONENTIAL, QUADRATIC, HARD,
        av, ai, weight_min=0., weight_max=10., conv_pipeline=pipeline,
    )
    generator = torch.Generator().manual_seed(721)
    for param in energy._all_params:
        if isinstance(param, Bias):
            param.state = torch.full(param.shape, .02, dtype=dtype)
        elif isinstance(param, PoolWeight):
            param.state = param.state.to(dtype)
        else:
            param.state = torch.rand(param.shape, generator=generator, dtype=dtype) + .1
    for layer in energy.layers():
        layer.state = torch.randn((2, *layer.shape), generator=generator, dtype=dtype) * .3
        if nonlinearity == "perfect_diode" and layer is not energy.layers()[0]:
            layer.state = layer.activate()
    return energy


def make_solver(energy, nonlinearity, av, ai, custom=True, fn=None):
    kwargs = dict(
        fn=fn or energy, free_layers=energy.layers()[1:], num_iterations=1,
        mode="asynchronous", non_linearity=nonlinearity,
        quadratic_diode_param=QUADRATIC, exponential_diode_param=EXPONENTIAL,
        hard_sigmoid_param=HARD, voltage_amp=av, current_amp=ai,
    )
    if not custom:
        return QuadraticMinimizer(**kwargs)
    return CustomQuadraticMinimizer(
        **kwargs, iv_data=None, iv_data_path=None,
        double_diode_updater="custom", single_diode_updater="custom",
        adaptive_equilibrium=False, overrelaxation_factor=1.,
        minimizer_settings=MinimizerSettings(
            rel_tol=1e-12, vn_tol=1e-12, use_polish=True, max_newton_iters=64,
            z_thresh=1e10, exp_clip=1e4, dynamic_polish=False,
            overrelaxation_reject_steps=False, overrelaxation_reject_max_tries=3,
            overrelaxation_reject_shrink=.5, overrelaxation_reject_eps=0.,
        ),
    )


def independent_residual(energy, fn=None):
    """Differentiate E independently of updater coefficients; remove depth units.

    At a perfect-diode boundary only current violating the feasible half-line
    is a residual. A diode may supply a nonzero normal reaction at equilibrium.
    """
    fn = fn or energy
    layers = energy.layers()[1:]
    saved = [layer.state for layer in layers]
    try:
        for layer in layers:
            layer.state = layer.state.detach().clone().requires_grad_(True)
        gradients = torch.autograd.grad(fn.eval().sum(), [layer.state for layer in layers])
        result = {}
        for depth, (layer, grad) in enumerate(zip(layers, gradients), 1):
            scale = (energy._current_amp / energy._voltage_amp) ** (depth - 1)
            current = grad / scale
            if getattr(layer, "non_linearity", None) == "perfect_diode":
                half = layer.shape[0] // 2
                exc = torch.where(layer.state[:, :half] == 0, current[:, :half].clamp(max=0), current[:, :half])
                inh = torch.where(layer.state[:, half:] == 0, current[:, half:].clamp(min=0), current[:, half:])
                current = torch.cat((exc, inh), dim=1)
            result[layer.name] = float(current.detach().abs().max())
        return result
    finally:
        for layer, state in zip(layers, saved):
            layer.state = state


def settle_with_residuals(energy, solver, fn=None, tolerance=1e-10):
    samples = {0: max(independent_residual(energy, fn).values())}
    previous = 0
    for sweeps in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        solver.num_iterations = sweeps - previous
        with torch.no_grad():
            solver.compute_equilibrium()
        samples[sweeps] = max(independent_residual(energy, fn).values())
        if samples[sweeps] < tolerance:
            break
        previous = sweeps
    return samples


@pytest.mark.parametrize("custom", [False, True], ids=["core", "custom"])
@pytest.mark.parametrize("kind", ["dense", "conv", "conv_pool"])
@pytest.mark.parametrize("nonlinearity", ["linear", "perfect_diode", "double_diode_quadratic", "hard_sigmoid", "lpw_diode"])
@pytest.mark.parametrize("av,ai", [(1., 1.), (4., .25), (.5, 2.)])
def test_equilibrium_residual_converges_to_zero(custom, kind, nonlinearity, av, ai):
    energy = make_energy(kind, nonlinearity, av, ai)
    solver = make_solver(energy, nonlinearity, av, ai, custom)
    samples = settle_with_residuals(energy, solver)
    assert samples[0] > 1e-3
    assert list(samples.values())[-1] < 1e-10, samples
    if custom:
        reported = solver.residual_currents_inf()
        independent = independent_residual(energy)
        for layer, residual in reported.items():
            assert residual == pytest.approx(independent[layer], abs=1e-11)


@pytest.mark.parametrize("nonlinearity", ["linear", "perfect_diode"])
def test_cost_nudged_equilibrium_and_float32_residual(nonlinearity):
    energy = make_energy("dense", nonlinearity, 4., .25, torch.float32)
    cost = SquaredError(energy.layers()[-1])
    cost.set_target(torch.tensor([0, 1]))
    augmented = AugmentedFunction(energy, cost, nudging_mode="cost")
    augmented.nudging = .05
    solver = make_solver(energy, nonlinearity, 4., .25, fn=augmented)
    samples = settle_with_residuals(energy, solver, augmented, tolerance=2e-6)
    assert list(samples.values())[-1] < 2e-6, samples
    assert max(solver.residual_currents_inf().values()) < 2e-6


@pytest.mark.parametrize("kind", ["dense", "conv", "conv_pool"])
def test_building_another_network_does_not_change_amplifier_depth(kind):
    first = make_energy(kind, "double_diode_quadratic", 4., .25)
    second = make_energy(kind, "double_diode_quadratic", 4., .25)
    assert first.layers()[0].name != second.layers()[0].name
    for energy in (first, second):
        solver = make_solver(energy, "double_diode_quadratic", 4., .25)
        with torch.no_grad():
            solver.compute_equilibrium()
    for left, right in zip(first.layers(), second.layers()):
        torch.testing.assert_close(left.state, right.state, rtol=0, atol=0)


def test_physical_residual_equals_direct_dense_kcl():
    av, ai = 4., .25
    energy = make_energy("dense", "linear", av, ai)
    solver = make_solver(energy, "linear", av, ai)
    layers = energy.layers()
    edges = [x for x in energy._interactions if isinstance(x, DenseResistive)]
    biases = {x._layer: x._bias.get() for x in energy._interactions if hasattr(x, "_bias")}
    actual = solver.residual_currents_inf()
    for depth, layer in enumerate(layers[1:], 1):
        incoming = edges[depth - 1]._weight.get()
        alpha = 1. if depth == 1 else av
        current = incoming.sum(0) * layer.state - alpha * (layers[depth - 1].state @ incoming)
        if depth < len(layers) - 1:
            outgoing = edges[depth]._weight.get()
            current = current + ai * (av * outgoing.sum(1) * layer.state - layers[depth + 1].state @ outgoing.T)
        if layer in biases:
            current = current - biases[layer] / (ai / av)**(depth - 1)
        assert actual[layer.name] == pytest.approx(float(current.abs().max()), rel=1e-12)


@pytest.mark.parametrize("custom", [False, True], ids=["core", "custom"])
def test_hard_sigmoid_leakage_has_zero_residual(monkeypatch, custom):
    # Stay within the off region to check its nonzero leakage independently
    # of switching: d(0.5*g_off*v**2)/dv = g_off*v, not 2*g_off*v.
    for key, value in {"g_off": 2., "v_min": -100., "v_max": 100.}.items():
        monkeypatch.setitem(HARD, key, value)
    energy = make_energy("dense", "hard_sigmoid", 4., .25)
    solver = make_solver(energy, "hard_sigmoid", 4., .25, custom)
    samples = settle_with_residuals(energy, solver)
    assert list(samples.values())[-1] < 1e-10, samples
