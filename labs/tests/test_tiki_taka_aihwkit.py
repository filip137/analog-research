import copy

import pytest
import torch

pytest.importorskip("aihwkit")

from aihwkit.simulator.rpu_base import cuda as aihwkit_cuda
from networks.variable.parameter import DenseWeight
from methods.tiki_taka import TikiTakaConfig, TikiTakaOptimizer


class _ParameterFunction:
    def __init__(self, *parameters):
        self._parameters = list(parameters)

    def params(self):
        return self._parameters


def _build_optimizer(
    *,
    preset="TikiTakaIdealizedPreset",
    device="cpu",
    clamp_max=1.0,
    conductance_min=0.0,
    conductance_max=1.0,
    post_size=2,
):
    parameter = DenseWeight(
        layer_pre_shape=(3,),
        layer_post_shape=(post_size,),
        gain=0.0,
        device=device,
        clamp=True,
        clamp_min=0.0,
        clamp_max=clamp_max,
    )
    parameter.state.fill_(0.5)
    config = TikiTakaConfig(
        fast_lr=0.1,
        transfer_every=2,
        n_reads_per_transfer=3,
        transfer_lr=1.0,
        scale_transfer_lr=False,
        aihwkit_preset=preset,
        aihwkit_conductance_min=conductance_min,
        aihwkit_conductance_max=conductance_max,
        aihwkit_construction_seed=11,
    )
    optimizer = TikiTakaOptimizer(
        _ParameterFunction(parameter),
        _ParameterFunction(),
        [0.1],
        config,
    )
    return parameter, optimizer


def _set_gradient(parameter):
    parameter.state.grad = torch.tensor(
        [
            [0.2, -0.1],
            [0.3, -0.4],
            [0.1, 0.2],
        ],
        dtype=parameter.state.dtype,
        device=parameter.state.device,
    )


def test_native_aihwkit_accumulates_then_transfers_realized_device_state():
    parameter, optimizer = _build_optimizer()
    initial = parameter.state.clone()

    _set_gradient(parameter)
    optimizer.step()

    torch.testing.assert_close(parameter.state, initial, rtol=0.0, atol=0.0)
    device_state = optimizer.aihwkit_device_state(parameter.state)
    assert device_state is not None
    assert "hidden_weights_0" in device_state
    assert "hidden_weights_1" in device_state
    assert device_state["hidden_weights_0"].abs().sum() > 0.0
    auxiliary = optimizer.auxiliary_state(parameter.state)
    assert torch.sum(auxiliary * parameter.state.grad) < 0.0
    assert optimizer.state[parameter.state]["step"] == 1
    assert optimizer.state[parameter.state]["transfer_count"] == 0

    _set_gradient(parameter)
    optimizer.step()

    assert not torch.equal(parameter.state, initial)
    assert torch.isfinite(parameter.state).all()
    assert parameter.state.min() >= 0.0
    assert parameter.state.max() <= 1.0
    before_clamp = parameter.state.clone()
    parameter.clamp_()
    torch.testing.assert_close(parameter.state, before_clamp, rtol=0.0, atol=0.0)
    assert optimizer.state[parameter.state]["step"] == 2
    assert optimizer.state[parameter.state]["transfer_count"] == 1
    assert optimizer.uses_aihwkit
    assert optimizer.aihwkit_version is not None


def test_construction_seed_reproduces_device_to_device_parameters():
    parameter_a, optimizer_a = _build_optimizer(
        preset="TikiTakaReRamESPreset"
    )
    parameter_b, optimizer_b = _build_optimizer(
        preset="TikiTakaReRamESPreset"
    )

    state_a = optimizer_a.aihwkit_device_state(parameter_a.state)
    state_b = optimizer_b.aihwkit_device_state(parameter_b.state)
    construction_fields = [
        name
        for name in state_a
        if name.startswith(("min_bound_", "max_bound_", "dwmin_"))
    ]
    assert construction_fields
    for name in construction_fields:
        torch.testing.assert_close(
            state_a[name],
            state_b[name],
            rtol=0.0,
            atol=0.0,
        )


def test_reram_persistent_slow_conductance_matches_programming_target():
    parameter, optimizer = _build_optimizer(
        preset="TikiTakaReRamESPreset"
    )
    target = torch.full_like(parameter.state, 0.5)

    persistent = optimizer.aihwkit_slow_conductance(
        parameter.state,
        persistent=True,
    )
    apparent = optimizer.aihwkit_slow_conductance(
        parameter.state,
        persistent=False,
    )

    assert persistent is not None
    assert apparent is not None
    torch.testing.assert_close(persistent, target, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        apparent,
        parameter.state,
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        optimizer.auxiliary_state(parameter.state),
        torch.zeros_like(parameter.state),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "preset",
    [
        "TikiTakaIdealizedPreset",
        "TikiTakaCapacitorPreset",
        "TikiTakaEcRamPreset",
        "TikiTakaEcRamMOPreset",
    ],
)
def test_native_aihwkit_checkpoint_restores_fast_slow_and_transfer_state(preset):
    parameter, optimizer = _build_optimizer(preset=preset)
    for _ in range(2):
        _set_gradient(parameter)
        optimizer.step()

    saved_state = copy.deepcopy(optimizer.state_dict())
    assert "aihwkit_tile_state" not in optimizer.state[parameter.state]
    saved_device_state = optimizer.aihwkit_device_state(parameter.state)
    saved_visible = parameter.state.clone()
    saved_counters = {
        name: optimizer.state[parameter.state][name]
        for name in ("step", "transfer_index", "transfer_count")
    }

    restored_parameter, restored_optimizer = _build_optimizer(preset=preset)
    restored_optimizer.load_state_dict(saved_state)

    torch.testing.assert_close(
        restored_parameter.state,
        saved_visible,
        rtol=0.0,
        atol=0.0,
    )
    restored_device_state = restored_optimizer.aihwkit_device_state(
        restored_parameter.state
    )
    assert list(restored_device_state) == list(saved_device_state)
    for name in saved_device_state:
        torch.testing.assert_close(
            restored_device_state[name],
            saved_device_state[name],
            rtol=0.0,
            atol=0.0,
        )
    assert {
        name: restored_optimizer.state[restored_parameter.state][name]
        for name in ("step", "transfer_index", "transfer_count")
    } == saved_counters
    assert (
        "aihwkit_tile_state"
        not in restored_optimizer.state[restored_parameter.state]
    )
    assert "aihwkit_tile_state" in next(iter(saved_state["state"].values()))


@pytest.mark.parametrize(
    "preset",
    ["TikiTakaReRamESPreset", "TikiTakaReRamSBPreset"],
)
def test_reram_checkpoint_restore_is_rejected_before_mutation(preset):
    parameter, optimizer = _build_optimizer(preset=preset)
    for _ in range(2):
        _set_gradient(parameter)
        optimizer.step()

    saved_state = copy.deepcopy(optimizer.state_dict())
    restored_parameter, restored_optimizer = _build_optimizer(preset=preset)
    parameter_before = restored_parameter.state.clone()
    state_before = dict(restored_optimizer.state[restored_parameter.state])

    with pytest.raises(
        RuntimeError,
        match="loading was rejected before mutating the optimizer",
    ):
        restored_optimizer.load_state_dict(saved_state)

    torch.testing.assert_close(
        restored_parameter.state,
        parameter_before,
        rtol=0.0,
        atol=0.0,
    )
    assert restored_optimizer.state[restored_parameter.state] == state_before


def test_checkpoint_version_mismatch_is_rejected_before_mutation():
    parameter, optimizer = _build_optimizer()
    _set_gradient(parameter)
    optimizer.step()
    saved_state = copy.deepcopy(optimizer.state_dict())
    for parameter_state in saved_state["state"].values():
        parameter_state["aihwkit_tile_state"]["aihwkit_version"] = "mismatch"

    restored_parameter, restored_optimizer = _build_optimizer()
    parameter_before = restored_parameter.state.clone()
    state_before = dict(restored_optimizer.state[restored_parameter.state])

    with pytest.raises(ValueError, match="same AIHWKit version"):
        restored_optimizer.load_state_dict(saved_state)

    torch.testing.assert_close(
        restored_parameter.state,
        parameter_before,
        rtol=0.0,
        atol=0.0,
    )
    assert restored_optimizer.state[restored_parameter.state] == state_before


def test_checkpoint_conductance_bounds_mismatch_is_rejected_before_mutation():
    parameter, optimizer = _build_optimizer(
        conductance_min=None,
        conductance_max=None,
    )
    _set_gradient(parameter)
    optimizer.step()
    saved_state = copy.deepcopy(optimizer.state_dict())

    restored_parameter, restored_optimizer = _build_optimizer(
        clamp_max=0.5,
        conductance_min=None,
        conductance_max=None,
    )
    parameter_before = restored_parameter.state.clone()
    state_before = dict(restored_optimizer.state[restored_parameter.state])

    with pytest.raises(ValueError, match="same parameter-group structure"):
        restored_optimizer.load_state_dict(saved_state)

    torch.testing.assert_close(
        restored_parameter.state,
        parameter_before,
        rtol=0.0,
        atol=0.0,
    )
    assert restored_optimizer.state[restored_parameter.state] == state_before


def test_checkpoint_parameter_shape_mismatch_is_rejected_before_mutation():
    _, optimizer = _build_optimizer()
    saved_state = copy.deepcopy(optimizer.state_dict())

    restored_parameter, restored_optimizer = _build_optimizer(post_size=4)
    parameter_before = restored_parameter.state.clone()
    state_before = dict(restored_optimizer.state[restored_parameter.state])

    with pytest.raises(ValueError, match="same parameter, visible-tile"):
        restored_optimizer.load_state_dict(saved_state)

    torch.testing.assert_close(
        restored_parameter.state,
        parameter_before,
        rtol=0.0,
        atol=0.0,
    )
    assert restored_optimizer.state[restored_parameter.state] == state_before


@pytest.mark.skipif(
    not torch.cuda.is_available() or not aihwkit_cuda.is_compiled(),
    reason="requires CUDA and a CUDA-enabled AIHWKit build",
)
def test_native_aihwkit_cuda_initialization_and_update():
    parameter, optimizer = _build_optimizer(device="cuda")

    _set_gradient(parameter)
    optimizer.step()

    assert parameter.state.is_cuda
    assert torch.isfinite(parameter.state).all()
    assert parameter.state.min() >= 0.0
    assert parameter.state.max() <= 1.0
