from types import SimpleNamespace

import pytest
import torch

from labs import mnist_tests
from training.core.batch import Batch


def test_build_minimizer_uses_top_level_config_and_base_energy_amplification(monkeypatch):
    captured = {}

    class FakeMinimizer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(mnist_tests, "CustomQuadraticMinimizer", FakeMinimizer)

    base_energy = SimpleNamespace(_voltage_amp=2.5, _current_amp=0.4)
    parts = SimpleNamespace(
        energy_fn=base_energy,
        energy_minimizer_cfg={
            "adaptive_equilibrium": False,
            "use_polish": False,
            "dynamic_polish": False,
            "exp_clip": 165.0,
        },
        training_cfg={"energy_minimizer": {"adaptive_equilibrium": True}},
        model_cfg={
            "non_linearity": "perfect_diode",
            "quadratic_diode_param": {},
            "exponential_diode_param": {},
            "hard_sigmoid_param": {},
        },
        free_layers=[],
        minimizer_mode="asynchronous",
    )

    augmented_energy = object()
    result = mnist_tests._build_minimizer(parts, 4, fn=augmented_energy)

    assert isinstance(result, FakeMinimizer)
    assert captured["fn"] is augmented_energy
    assert captured["adaptive_equilibrium"] is False
    assert captured["voltage_amp"] == 2.5
    assert captured["current_amp"] == 0.4
    assert captured["minimizer_settings"].use_polish is False
    assert captured["minimizer_settings"].dynamic_polish is False
    assert captured["minimizer_settings"].exp_clip == 165.0


@pytest.mark.parametrize("output", ["states", "residuals"])
def test_pca_diagnostics_share_evaluator_without_targets(monkeypatch, output):
    layer = SimpleNamespace(name="voltage", state=torch.zeros((1, 2)))
    resets, solves = [], []

    def set_input(inputs, reset):
        resets.append(reset)
        layer.state = inputs.clone()

    def settle():
        solves.append(True)
        layer.state.add_(1)

    def reject_target(targets):
        raise AssertionError("PCA diagnostics must not assign cost targets")

    solver = SimpleNamespace(
        compute_equilibrium=settle,
        residual_currents_inf=lambda: {"projected": 99.0},
    )
    monkeypatch.setattr(mnist_tests, "_build_minimizer", lambda parts, count: solver)
    energy = SimpleNamespace(
        layers=lambda: [layer], grad_layer_fn=lambda _: lambda: layer.state * 2,
    )
    inputs = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    parts = SimpleNamespace(
        network=SimpleNamespace(set_input=set_input, layers=lambda: [layer], _function=energy),
        cost_fn=SimpleNamespace(set_target=reject_target),
        test_loader=[inputs, (inputs,), (inputs, [0, 1], [5, 6]),
                     Batch(inputs, None), (), None],
    )

    result = mnist_tests.sweep_pca(parts, 3, output=output)

    assert resets == [True] * 4
    assert len(solves) == 4
    if output == "states":
        torch.testing.assert_close(result["voltage"], torch.cat([inputs + 1] * 4))
    else:
        assert result == {"voltage": [10.0] * 4}
