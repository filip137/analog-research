from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from model.resistive.builders import ParameterCatalog, build_deep_resistive_energy
from model.resistive.digital_low_rank import (
    DigitalLowRankReadout,
    DigitalLowRankWeight,
)
from model.resistive.digital_low_rank_config import Wan2022ProgrammingConfig
from training.device_programming import program_wan2022_base_conductance
from training.core.sgd import DirectReadoutGradient


def _bundle():
    return build_deep_resistive_energy(
        layer_shapes=[(4,), (4,)],
        weight_gains=[0.2],
        input_gain=2.0,
        non_linearity="linear",
        exponential_diode_param={},
        quadratic_diode_param={},
        hard_sigmoid_param={},
        voltage_amp=1.0,
        current_amp=1.0,
        weight_min=0.0,
        weight_max=1.0,
        digital_low_rank_adapter={
            "rank": 2,
            "alpha": 4.0,
            "input_factor_gain": 0.5,
            "device_noise": {
                "type": "aihwkit_reram_wan2022",
                "programming_seed": 13,
                "g_max_us": 40.0,
                "drn_conductance_at_g_max": 0.5,
                "noise_scale": 1.0,
                "t_inference_seconds": 1.0,
            },
        },
    )


def _readout(bundle):
    energy = bundle.energy
    input_factor, output_factor = energy.adapter_params()
    return DigitalLowRankReadout(
        energy.layers()[0],
        energy.layers()[-1],
        input_factor=input_factor,
        output_factor=output_factor,
        logical_input_dim=2,
        num_classes=2,
        input_gain=2.0,
        alpha=4.0,
    )


def test_digital_adapter_catalog_topology_and_neutral_initialization() -> None:
    bundle = _bundle()
    energy = bundle.energy
    input_factor, output_factor = energy.adapter_params()

    assert [layer.shape for layer in energy.layers()] == [(4,), (4,)]
    assert energy.params() == []
    assert energy.trainable_params() == [input_factor, output_factor]
    assert isinstance(input_factor, DigitalLowRankWeight)
    assert isinstance(output_factor, DigitalLowRankWeight)
    assert input_factor.shape == (2, 2)
    assert output_factor.shape == (2, 2)
    assert torch.count_nonzero(output_factor.state) == 0
    assert [binding.key for binding in bundle.catalog.all] == [
        "base.dense_weight.0",
        "adapter.input_factor.0",
        "adapter.output_factor.0",
    ]
    assert [binding.key for binding in bundle.catalog.trainable] == [
        "adapter.input_factor.0",
        "adapter.output_factor.0",
    ]

    energy.layers()[0].state = torch.tensor(
        [[0.4, -0.8, -0.4, 0.8]],
        dtype=torch.float32,
    )
    energy.layers()[-1].state = torch.tensor(
        [[0.7, 0.2, -0.1, -0.4]],
        dtype=torch.float32,
    )
    readout = _readout(bundle)
    torch.testing.assert_close(
        readout.scores(),
        torch.tensor([[0.5, 0.3]]),
    )


def test_digital_readout_formula_and_direct_gradients() -> None:
    bundle = _bundle()
    energy = bundle.energy
    input_factor, output_factor = energy.adapter_params()
    energy.layers()[0].state = torch.tensor(
        [[0.4, -0.8, -0.4, 0.8]],
        dtype=torch.float32,
    )
    energy.layers()[-1].state = torch.tensor(
        [[0.7, 0.2, -0.1, -0.4]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        input_factor.state.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        output_factor.state.copy_(torch.tensor([[0.5, -0.5], [0.25, 0.75]]))

    readout = _readout(bundle)
    readout.set_target(torch.tensor([1]))
    logical_input = torch.tensor([[0.2, -0.4]])
    expected = torch.tensor([[0.5, 0.3]]) + 2.0 * (
        logical_input @ input_factor.state @ output_factor.state
    )
    torch.testing.assert_close(readout.scores(), expected)

    gradients = DirectReadoutGradient(readout).compute_gradient()
    assert len(gradients) == 2
    assert gradients[0].shape == input_factor.shape
    assert gradients[1].shape == output_factor.shape
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.count_nonzero(gradients[0]) > 0
    assert torch.count_nonzero(gradients[1]) > 0


def test_wan_programming_uses_physical_scale_seed_and_one_base_matrix(
    monkeypatch,
) -> None:
    bundle = _bundle()
    base = bundle.catalog.by_key["base.dense_weight.0"]
    with torch.no_grad():
        base.state.copy_(
            torch.tensor(
                [
                    [0.05, 0.10, 0.15, 0.20],
                    [0.25, 0.30, 0.35, 0.40],
                    [0.10, 0.20, 0.30, 0.40],
                    [0.05, 0.15, 0.25, 0.35],
                ]
            )
        )
    clean = base.state.clone()

    class FakeWan:
        def __init__(self, *, g_max, noise_scale):
            self.g_max = g_max
            self.noise_scale = noise_scale

        def apply_drift_noise_to_conductance(self, first, second, age):
            torch.testing.assert_close(first, second)
            assert age == 86400.0
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

    config = Wan2022ProgrammingConfig(
        type="aihwkit_reram_wan2022",
        programming_seed=23,
        g_max_us=40.0,
        drn_conductance_at_g_max=0.5,
        noise_scale=0.25,
        t_inference_seconds=86400.0,
    )
    first = program_wan2022_base_conductance(bundle.catalog, config)
    first_state = base.state.clone()
    with torch.no_grad():
        base.state.copy_(clean)
    second = program_wan2022_base_conductance(bundle.catalog, config)

    torch.testing.assert_close(base.state, first_state, rtol=0.0, atol=0.0)
    assert first == second
    assert first["aihwkit_version"] == "test"
    assert first["parameter_key"] == "base.dense_weight.0"
    assert first["error_rmse"] > 0.0
    assert not torch.equal(clean, first_state)


def test_wan_programming_rejects_checkpoint_above_calibration() -> None:
    bundle = _bundle()
    base = bundle.catalog.by_key["base.dense_weight.0"]
    base.state.fill_(0.6)
    config = Wan2022ProgrammingConfig(
        type="aihwkit_reram_wan2022",
        programming_seed=0,
        g_max_us=40.0,
        drn_conductance_at_g_max=0.5,
        noise_scale=1.0,
        t_inference_seconds=1.0,
    )
    with pytest.raises(ValueError, match="no greater than"):
        program_wan2022_base_conductance(bundle.catalog, config)


def test_real_aihwkit_zero_noise_programming_is_numerically_neutral() -> None:
    pytest.importorskip("aihwkit")
    bundle = _bundle()
    base = bundle.catalog.by_key["base.dense_weight.0"]
    with torch.no_grad():
        base.state.uniform_(0.01, 0.4)
    clean = base.state.clone()
    report = program_wan2022_base_conductance(
        bundle.catalog,
        Wan2022ProgrammingConfig(
            type="aihwkit_reram_wan2022",
            programming_seed=31,
            g_max_us=40.0,
            drn_conductance_at_g_max=0.5,
            noise_scale=0.0,
            t_inference_seconds=1.0,
        ),
    )
    torch.testing.assert_close(base.state, clean, rtol=1e-6, atol=1e-7)
    assert report["error_abs_max"] <= 1e-7
    assert report["clipped_low_fraction"] == 0.0
    assert report["clipped_high_fraction"] == 0.0
