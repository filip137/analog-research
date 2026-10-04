import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.parameters import (
    ParameterBinding,
    ParameterCatalog,
)
import training.checkpoint as checkpoint_module
from training.checkpoint import (
    CheckpointError,
    EPOCH_BOUNDARY_SCHEMA,
    EPOCH_BOUNDARY_SCHEMA_VERSION,
    NAMED_WEIGHTS_SCHEMA,
    encode_named_weights,
    load_epoch_boundary_checkpoint,
    load_named_weights,
    save_encoded_named_weights,
    save_epoch_boundary_checkpoint,
    save_named_weights,
)


class _StatefulComponent:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class _CountingModifier:
    def __init__(self, updates=0):
        self.updates = updates

    def state_dict(self):
        return {"updates": self.updates}

    def load_state_dict(self, state):
        self.updates = int(state["updates"])


class _ProgressComponent:
    def __init__(self, state):
        self.state = copy.deepcopy(state)

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(dict(state))


def _parameter(value, *, lower=None, upper=None, requires_grad=False):
    state = torch.tensor(
        value,
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    if requires_grad:
        state = torch.nn.Parameter(state)
    return SimpleNamespace(
        state=state,
        min_cond=lower,
        max_cond=upper,
    )


def _catalog(*, trainable_tensor=False):
    base = _parameter(
        [[0.2, 0.4], [0.6, 0.8]],
        lower=0.0,
        upper=1.0,
        requires_grad=trainable_tensor,
    )
    adapter = _parameter(
        [[0.1], [0.3]],
        lower=0.0,
        upper=1.0,
        requires_grad=trainable_tensor,
    )
    catalog = ParameterCatalog(
        [
            ParameterBinding(
                "base.weight.0",
                base,
                group="base",
                role="weight",
                trainable=trainable_tensor,
            ),
            ParameterBinding(
                "adapter.factor.0",
                adapter,
                group="adapter",
                role="factor",
                trainable=trainable_tensor,
            ),
        ]
    )
    return catalog, base, adapter


def test_named_weights_round_trip_preserves_tensor_identity_and_metadata(
    tmp_path,
):
    catalog, base, adapter = _catalog()
    expected_base = base.state.clone()
    expected_adapter = adapter.state.clone()
    base_identity = id(base.state)
    adapter_identity = id(adapter.state)
    path = tmp_path / "weights.pt"

    save_named_weights(path, catalog, metadata={"seed": 7})
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == NAMED_WEIGHTS_SCHEMA
    assert list(payload["weights"]) == [
        "base.weight.0",
        "adapter.factor.0",
    ]

    base.state.zero_()
    adapter.state.fill_(0.9)
    result = load_named_weights(path, catalog)

    assert result.metadata == {"seed": 7}
    assert result.restored_keys == (
        "base.weight.0",
        "adapter.factor.0",
    )
    assert id(base.state) == base_identity
    assert id(adapter.state) == adapter_identity
    torch.testing.assert_close(base.state, expected_base)
    torch.testing.assert_close(adapter.state, expected_adapter)


def test_named_weights_validate_every_tensor_before_copying(tmp_path):
    catalog, base, adapter = _catalog()
    path = tmp_path / "weights.pt"
    save_named_weights(path, catalog)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["weights"]["base.weight.0"] = torch.full_like(base.state, 0.7)
    payload["weights"]["adapter.factor.0"] = torch.ones(3)
    torch.save(payload, path)
    base_before = base.state.clone()
    adapter_before = adapter.state.clone()

    with pytest.raises(CheckpointError, match="shape"):
        load_named_weights(path, catalog)

    torch.testing.assert_close(base.state, base_before)
    torch.testing.assert_close(adapter.state, adapter_before)


def test_atomic_named_write_keeps_existing_target_on_serialization_failure(
    tmp_path,
    monkeypatch,
):
    catalog, _base, _adapter = _catalog()
    path = tmp_path / "weights.pt"
    path.write_bytes(b"existing checkpoint")

    def fail_after_partial_write(_payload, temporary):
        temporary.write_bytes(b"partial checkpoint")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(
        checkpoint_module.torch,
        "save",
        fail_after_partial_write,
    )
    with pytest.raises(RuntimeError, match="simulated write failure"):
        save_named_weights(path, catalog)

    assert path.read_bytes() == b"existing checkpoint"
    assert list(tmp_path.glob(".weights.pt.*.tmp")) == []


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (torch.tensor([[float("nan")], [0.2]]), "finite"),
        (torch.tensor([[0.2], [1.2]]), "<= 1.0"),
        (torch.ones(2, 1, dtype=torch.float64), "dtype"),
    ],
)
def test_named_weights_reject_nonfinite_bounds_and_dtype(
    tmp_path,
    replacement,
    message,
):
    catalog, _base, adapter = _catalog()
    path = tmp_path / "weights.pt"
    save_named_weights(path, catalog)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["weights"]["adapter.factor.0"] = replacement
    torch.save(payload, path)
    before = adapter.state.clone()

    with pytest.raises(CheckpointError, match=message):
        load_named_weights(path, catalog)

    torch.testing.assert_close(adapter.state, before)


def test_encoded_weights_accept_float32_representable_bound_endpoint(
    tmp_path,
):
    parameter = _parameter([[0.00011]], lower=1e-7, upper=0.00011)
    parameter.state.clamp_(max=0.00011)
    catalog = ParameterCatalog(
        [
            ParameterBinding(
                "base.conductance.0",
                parameter,
                group="base",
                role="conductance",
            )
        ]
    )
    payload = encode_named_weights(catalog)

    save_encoded_named_weights(
        tmp_path / "weights.pt",
        payload,
        catalog=catalog,
    )


def test_epoch_boundary_restores_weights_optimizer_scheduler_and_rng(tmp_path):
    catalog, base, _adapter = _catalog(trainable_tensor=True)
    optimizer = torch.optim.SGD(
        [binding.state for binding in catalog.trainable],
        lr=0.2,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )
    for binding in catalog.trainable:
        binding.state.grad = torch.full_like(binding.state, 0.05)
    optimizer.step()
    scheduler.step()
    saved_weights = [binding.state.detach().clone() for binding in catalog]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    runtime_state = _StatefulComponent(
        torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    )
    saved_runtime_state = runtime_state.value.clone()

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    path = tmp_path / "resume.pt"
    save_epoch_boundary_checkpoint(
        path,
        catalog=catalog,
        epoch=4,
        global_step=37,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime_state=runtime_state,
        progress_state={
            "checkpoint_selection": {
                "best_epoch": 3,
                "best_metric": 0.125,
            },
            "early_stop": {"bad_epochs": 1, "stopped": False},
        },
        metadata={"case_id": "smoke"},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == EPOCH_BOUNDARY_SCHEMA
    assert payload["schema_version"] == EPOCH_BOUNDARY_SCHEMA_VERSION
    assert payload["metadata"] == {"case_id": "smoke"}
    assert payload["progress_state"]["early_stop"]["bad_epochs"] == 1
    expected_random = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(3)

    with torch.no_grad():
        for binding in catalog:
            binding.state.fill_(0.99)
    optimizer.param_groups[0]["lr"] = 9.0
    optimizer.state.clear()
    scheduler.last_epoch = 99
    runtime_state.value.fill_(9.0)
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)

    result = load_epoch_boundary_checkpoint(
        path,
        catalog=catalog,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime_state=runtime_state,
    )

    assert result.epoch == 4
    assert result.global_step == 37
    assert result.metadata == {"case_id": "smoke"}
    assert result.progress_state["checkpoint_selection"]["best_epoch"] == 3
    assert result.resume_capability == "exact"
    assert result.optimizer_restored
    assert not result.modifier_restored
    assert result.scheduler_restored
    assert result.runtime_state_restored
    assert not result.progress_restored
    assert result.rng_restored
    assert result.dataloader_generators_restored == ()
    for binding, expected in zip(catalog, saved_weights):
        torch.testing.assert_close(binding.state, expected)
    assert optimizer.param_groups[0]["lr"] == saved_optimizer["param_groups"][0][
        "lr"
    ]
    assert scheduler.state_dict() == saved_scheduler
    torch.testing.assert_close(runtime_state.value, saved_runtime_state)
    assert random.random() == expected_random
    assert float(np.random.rand()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)


def test_epoch_boundary_rolls_back_model_and_optimizer_on_restore_failure(
    tmp_path,
):
    catalog, base, adapter = _catalog()

    class ControlledOptimizer:
        def __init__(self):
            self.value = "bad"

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state):
            if state["value"] == "bad":
                raise RuntimeError("rejected optimizer state")
            self.value = state["value"]

    optimizer = ControlledOptimizer()
    path = tmp_path / "resume.pt"
    save_epoch_boundary_checkpoint(
        path,
        catalog=catalog,
        epoch=2,
        optimizer=optimizer,
        include_rng=False,
    )

    base.state.fill_(0.7)
    adapter.state.fill_(0.8)
    base_before = base.state.clone()
    adapter_before = adapter.state.clone()
    optimizer.value = "good"

    with pytest.raises(CheckpointError, match="restore atomically"):
        load_epoch_boundary_checkpoint(
            path,
            catalog=catalog,
            optimizer=optimizer,
            restore_rng=False,
        )

    torch.testing.assert_close(base.state, base_before)
    torch.testing.assert_close(adapter.state, adapter_before)
    assert optimizer.value == "good"


def _advance_split_run(
    catalog,
    optimizer,
    modifier,
    generator,
    *,
    global_step,
    steps,
):
    for step in range(global_step, global_step + steps):
        sample = float(torch.rand((), generator=generator))
        gradient = -0.002 * (
            1.0 + sample + 0.1 * modifier.updates + 0.01 * step
        )
        for binding in catalog.trainable:
            binding.state.grad = torch.full_like(binding.state, gradient)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        modifier.updates += 1
    return global_step + steps


def test_epoch_boundary_split_run_restores_modifier_and_named_generator(
    tmp_path,
):
    uninterrupted_catalog, _base, _adapter = _catalog(trainable_tensor=True)
    uninterrupted_optimizer = torch.optim.SGD(
        [binding.state for binding in uninterrupted_catalog.trainable],
        lr=0.1,
        momentum=0.9,
    )
    uninterrupted_modifier = _CountingModifier()
    uninterrupted_generator = torch.Generator().manual_seed(1234)
    uninterrupted_step = _advance_split_run(
        uninterrupted_catalog,
        uninterrupted_optimizer,
        uninterrupted_modifier,
        uninterrupted_generator,
        global_step=0,
        steps=7,
    )

    first_catalog, _base, _adapter = _catalog(trainable_tensor=True)
    first_optimizer = torch.optim.SGD(
        [binding.state for binding in first_catalog.trainable],
        lr=0.1,
        momentum=0.9,
    )
    first_modifier = _CountingModifier()
    first_generator = torch.Generator().manual_seed(1234)
    split_step = _advance_split_run(
        first_catalog,
        first_optimizer,
        first_modifier,
        first_generator,
        global_step=0,
        steps=3,
    )
    path = tmp_path / "resume.pt"
    progress_state = {
        "checkpoint_selection": {
            "best_epoch": 1,
            "best_metric": 0.25,
        },
        "early_stop": {"bad_epochs": 2, "stopped": False},
    }
    save_epoch_boundary_checkpoint(
        path,
        catalog=first_catalog,
        epoch=2,
        global_step=split_step,
        optimizer=first_optimizer,
        modifier=first_modifier,
        progress_state=progress_state,
        dataloader_generators={"train": first_generator},
        metadata={"purpose": "split-run"},
        include_rng=False,
    )

    resumed_catalog, _base, _adapter = _catalog(trainable_tensor=True)
    resumed_optimizer = torch.optim.SGD(
        [binding.state for binding in resumed_catalog.trainable],
        lr=9.0,
        momentum=0.9,
    )
    resumed_modifier = _CountingModifier(updates=99)
    resumed_generator = torch.Generator().manual_seed(9876)
    resumed_progress = _ProgressComponent({"stale": True})
    resume = load_epoch_boundary_checkpoint(
        path,
        catalog=resumed_catalog,
        optimizer=resumed_optimizer,
        modifier=resumed_modifier,
        progress=resumed_progress,
        dataloader_generators={"train": resumed_generator},
        restore_rng=False,
    )

    assert resume.global_step == 3
    assert resume.modifier_restored
    assert resume.progress_restored
    assert resume.dataloader_generators_restored == ("train",)
    assert resume.generators_restored
    assert resume.metadata == {"purpose": "split-run"}
    assert resume.progress_state == progress_state
    assert resumed_progress.state == progress_state

    resumed_step = _advance_split_run(
        resumed_catalog,
        resumed_optimizer,
        resumed_modifier,
        resumed_generator,
        global_step=resume.global_step,
        steps=4,
    )

    assert resumed_step == uninterrupted_step == 7
    assert resumed_modifier.updates == uninterrupted_modifier.updates
    for resumed, uninterrupted in zip(
        (binding.state for binding in resumed_catalog.trainable),
        (binding.state for binding in uninterrupted_catalog.trainable),
    ):
        torch.testing.assert_close(
            resumed,
            uninterrupted,
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        torch.rand(4, generator=resumed_generator),
        torch.rand(4, generator=uninterrupted_generator),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("resume_capability", ["unsupported", "invalid", None])
def test_resume_capability_rejected_before_creating_checkpoint(
    tmp_path,
    resume_capability,
):
    catalog, _base, _adapter = _catalog()
    path = tmp_path / "resume.pt"

    with pytest.raises(CheckpointError, match="resume_capability"):
        save_epoch_boundary_checkpoint(
            path,
            catalog=catalog,
            epoch=0,
            resume_capability=resume_capability,
        )

    assert not path.exists()


def test_epoch_boundary_rolls_back_every_mutable_component(tmp_path, monkeypatch):
    catalog, base, adapter = _catalog()
    optimizer = _StatefulComponent("saved optimizer")
    modifier = _CountingModifier(updates=4)
    scheduler = _StatefulComponent("saved scheduler")
    runtime_state = _StatefulComponent(
        torch.tensor([1.0, 2.0], dtype=torch.float32)
    )
    generator = torch.Generator().manual_seed(23)
    torch.rand(2, generator=generator)

    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    path = tmp_path / "resume.pt"
    save_epoch_boundary_checkpoint(
        path,
        catalog=catalog,
        epoch=1,
        global_step=8,
        optimizer=optimizer,
        modifier=modifier,
        scheduler=scheduler,
        runtime_state=runtime_state,
        progress_state={"early_stop": {"bad_epochs": 2}},
        dataloader_generators={"train": generator},
    )

    base.state.fill_(0.7)
    adapter.state.fill_(0.8)
    base_before = base.state.clone()
    adapter_before = adapter.state.clone()
    optimizer.value = "runtime optimizer"
    modifier.updates = 99
    scheduler.value = "runtime scheduler"
    runtime_state.value = torch.tensor([8.0, 9.0], dtype=torch.float32)
    runtime_state_before = runtime_state.value.clone()
    progress = _ProgressComponent({"runtime": "progress"})
    runtime_generator = torch.Generator().manual_seed(71)
    torch.rand(3, generator=runtime_generator)
    generator_before = runtime_generator.get_state().clone()

    random.seed(81)
    np.random.seed(82)
    torch.manual_seed(83)
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(3)
    random.seed(81)
    np.random.seed(82)
    torch.manual_seed(83)

    real_apply_generator_states = checkpoint_module._apply_generator_states

    def fail_after_generator_restore(generators, states):
        real_apply_generator_states(generators, states)
        raise RuntimeError("failure after final mutable restore")

    monkeypatch.setattr(
        checkpoint_module,
        "_apply_generator_states",
        fail_after_generator_restore,
    )

    with pytest.raises(CheckpointError, match="restore atomically"):
        load_epoch_boundary_checkpoint(
            path,
            catalog=catalog,
            optimizer=optimizer,
            modifier=modifier,
            scheduler=scheduler,
            runtime_state=runtime_state,
            progress=progress,
            dataloader_generators={"train": runtime_generator},
        )

    torch.testing.assert_close(base.state, base_before)
    torch.testing.assert_close(adapter.state, adapter_before)
    assert optimizer.value == "runtime optimizer"
    assert modifier.updates == 99
    assert scheduler.value == "runtime scheduler"
    torch.testing.assert_close(runtime_state.value, runtime_state_before)
    assert progress.state == {"runtime": "progress"}
    torch.testing.assert_close(runtime_generator.get_state(), generator_before)
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)


def test_epoch_boundary_rejects_runtime_state_presence_mismatch_before_restore(
    tmp_path,
):
    catalog, base, adapter = _catalog()
    path = tmp_path / "resume.pt"
    save_epoch_boundary_checkpoint(
        path,
        catalog=catalog,
        epoch=1,
        runtime_state=_StatefulComponent(
            {"layer.0": torch.tensor([[0.2]], dtype=torch.float32)}
        ),
        include_rng=False,
    )

    base.state.fill_(0.7)
    adapter.state.fill_(0.8)
    base_before = base.state.clone()
    adapter_before = adapter.state.clone()

    with pytest.raises(CheckpointError, match="runtime_state presence"):
        load_epoch_boundary_checkpoint(
            path,
            catalog=catalog,
            restore_rng=False,
        )

    torch.testing.assert_close(base.state, base_before)
    torch.testing.assert_close(adapter.state, adapter_before)


def test_epoch_resume_preserves_distinct_latest_and_selected_weights(tmp_path):
    catalog, base, adapter = _catalog()
    selected_metadata = {
        "epoch": 2,
        "metric": 0.125,
        "metric_name": "validation_error",
    }
    selected_weights = encode_named_weights(
        catalog,
        metadata=selected_metadata,
    )
    selected_states = {
        key: value.clone()
        for key, value in selected_weights["weights"].items()
    }

    base.state.fill_(0.55)
    adapter.state.fill_(0.65)
    latest_states = {
        binding.key: binding.state.clone() for binding in catalog
    }
    assert any(
        not torch.equal(latest_states[key], selected_states[key])
        for key in latest_states
    )

    resume_path = tmp_path / "checkpoints" / "resume.pt"
    save_epoch_boundary_checkpoint(
        resume_path,
        catalog=catalog,
        epoch=4,
        global_step=20,
        selected_weights=selected_weights,
        include_rng=False,
    )

    base.state.fill_(0.05)
    adapter.state.fill_(0.15)
    resume = load_epoch_boundary_checkpoint(
        resume_path,
        catalog=catalog,
        restore_rng=False,
    )

    for binding in catalog:
        torch.testing.assert_close(
            binding.state,
            latest_states[binding.key],
            rtol=0.0,
            atol=0.0,
        )
    assert resume.selected_weights is not None
    assert resume.selected_weights["metadata"] == selected_metadata
    for key, expected in selected_states.items():
        torch.testing.assert_close(
            resume.selected_weights["weights"][key],
            expected,
            rtol=0.0,
            atol=0.0,
        )

    latest_before_materializing = {
        binding.key: binding.state.clone() for binding in catalog
    }
    weights_path = tmp_path / "checkpoints" / "weights.pt"
    save_encoded_named_weights(
        weights_path,
        resume.selected_weights,
        catalog=catalog,
    )
    for binding in catalog:
        torch.testing.assert_close(
            binding.state,
            latest_before_materializing[binding.key],
            rtol=0.0,
            atol=0.0,
        )
    materialized = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    assert materialized["metadata"] == selected_metadata
    for key, expected in selected_states.items():
        torch.testing.assert_close(
            materialized["weights"][key],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_invalid_selected_weights_fail_before_resume_or_weights_write(tmp_path):
    catalog, base, adapter = _catalog()
    invalid = encode_named_weights(catalog, metadata={"epoch": 1})
    invalid["weights"]["adapter.factor.0"] = torch.ones(3)
    states_before = {
        binding.key: binding.state.clone() for binding in catalog
    }

    resume_path = tmp_path / "resume.pt"
    weights_path = tmp_path / "weights.pt"
    resume_path.write_bytes(b"existing resume")
    weights_path.write_bytes(b"existing weights")

    with pytest.raises(CheckpointError, match="shape"):
        save_epoch_boundary_checkpoint(
            resume_path,
            catalog=catalog,
            epoch=2,
            selected_weights=invalid,
            include_rng=False,
        )
    with pytest.raises(CheckpointError, match="shape"):
        save_encoded_named_weights(
            weights_path,
            invalid,
            catalog=catalog,
        )

    assert resume_path.read_bytes() == b"existing resume"
    assert weights_path.read_bytes() == b"existing weights"
    torch.testing.assert_close(base.state, states_before["base.weight.0"])
    torch.testing.assert_close(
        adapter.state,
        states_before["adapter.factor.0"],
    )


def test_invalid_selected_weights_fail_before_resume_load_mutation(tmp_path):
    catalog, base, adapter = _catalog()
    selected = encode_named_weights(catalog, metadata={"epoch": 1})
    path = tmp_path / "resume.pt"
    save_epoch_boundary_checkpoint(
        path,
        catalog=catalog,
        epoch=2,
        selected_weights=selected,
        include_rng=False,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["selected_weights"]["weights"]["adapter.factor.0"] = torch.ones(3)
    torch.save(payload, path)

    base.state.fill_(0.45)
    adapter.state.fill_(0.35)
    base_before = base.state.clone()
    adapter_before = adapter.state.clone()

    with pytest.raises(CheckpointError, match="shape"):
        load_epoch_boundary_checkpoint(
            path,
            catalog=catalog,
            restore_rng=False,
        )

    torch.testing.assert_close(base.state, base_before)
    torch.testing.assert_close(adapter.state, adapter_before)


def test_resume_v2_without_selected_weights_remains_loadable(tmp_path):
    catalog, _base, _adapter = _catalog()
    path = tmp_path / "resume.pt"
    save_epoch_boundary_checkpoint(
        path,
        catalog=catalog,
        epoch=1,
        include_rng=False,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.pop("selected_weights")
    torch.save(payload, path)

    result = load_epoch_boundary_checkpoint(
        path,
        catalog=catalog,
        restore_rng=False,
    )

    assert result.selected_weights is None
