import copy
from dataclasses import FrozenInstanceError

import pytest
import torch

from model.variable.parameter import Bias, ConvWeight, DenseWeight, PoolWeight
from training.add_normal import (
    AddNormalConfig,
    AddNormalParameterModifier,
    build_add_normal_modifier,
)
from training.core.modifier import ParameterModifier


def _dense(
    shape=(2, 2),
    *,
    clamp=False,
    clamp_min=None,
    clamp_max=None,
):
    return DenseWeight(
        layer_pre_shape=(shape[0],),
        layer_post_shape=(shape[1],),
        gain=0.0,
        device="cpu",
        clamp=clamp,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def _conv(*, clamp=True, clamp_min=0.0, clamp_max=1.0):
    return ConvWeight(
        shape=(1, 1, 2, 2),
        gain=0.0,
        device="cpu",
        clamp=clamp,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def _pool():
    return PoolWeight(
        shape=(1, 1, 2, 2),
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=1.0,
    )


def _capture(context_factory, parameter):
    clean = parameter.state.detach().clone()
    state_identity = id(parameter.state)
    with context_factory():
        first = parameter.state.detach().clone()
        second = parameter.state.detach().clone()
        assert id(parameter.state) == state_identity
    torch.testing.assert_close(parameter.state, clean, rtol=0.0, atol=0.0)
    assert id(parameter.state) == state_identity
    return first, second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"std_dev": -0.1},
        {"std_dev": float("nan")},
        {"std_dev": float("inf")},
        {"std_dev": True},
        {"std_dev": 0.1, "seed": -1},
        {"std_dev": 0.1, "seed": 2**64},
        {"std_dev": 0.1, "seed": True},
        {"std_dev": 0.1, "noisy_evaluation": 1},
        {"std_dev": 0.1, "scale_mode": "per_tensor"},
    ],
)
def test_config_rejects_invalid_values_with_context(kwargs):
    with pytest.raises(ValueError) as exc_info:
        AddNormalConfig(**kwargs)

    message = str(exc_info.value)
    assert message.startswith("Expected")
    assert "Provided value:" in message


def test_config_is_frozen_normalized_and_zero_noise_builds_no_modifier():
    config = AddNormalConfig(
        std_dev=1,
        seed=7,
        noisy_evaluation=True,
    )

    assert config.std_dev == 1.0
    assert isinstance(config.std_dev, float)
    assert config.seed == 7
    assert config.scale_mode == "tensor_abs_max"
    with pytest.raises(FrozenInstanceError):
        config.std_dev = 0.5

    parameter = _dense()
    assert (
        build_add_normal_modifier(
            [parameter],
            AddNormalConfig(std_dev=0.0),
            run_seed=11,
        )
        is None
    )
    with pytest.raises(ValueError, match="run_seed"):
        build_add_normal_modifier(
            [parameter],
            AddNormalConfig(std_dev=0.1),
            run_seed=-1,
        )


def test_add_normal_scales_from_clean_abs_max_and_preserves_identity():
    parameter = _dense()
    clean = torch.tensor([[-2.0, -0.5], [0.25, 1.0]])
    parameter.state.copy_(clean)
    modifier = AddNormalParameterModifier(
        [parameter],
        AddNormalConfig(std_dev=0.25, seed=13),
    )

    expected_generator = torch.Generator(device="cpu").manual_seed(13)
    expected = clean + 0.5 * torch.randn(
        clean.shape,
        generator=expected_generator,
    )
    actual, repeated = _capture(modifier.training_context, parameter)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(repeated, actual, rtol=0.0, atol=0.0)
    assert isinstance(modifier, ParameterModifier)


def test_add_normal_scales_from_each_clean_output_channel():
    parameter = _dense()
    clean = torch.tensor([[-2.0, -0.5], [0.25, 1.0]])
    parameter.state.copy_(clean)
    modifier = AddNormalParameterModifier(
        [parameter],
        AddNormalConfig(
            std_dev=0.25,
            seed=13,
            scale_mode="output_channel_abs_max",
        ),
    )

    expected_generator = torch.Generator(device="cpu").manual_seed(13)
    expected = clean + torch.randn(
        clean.shape,
        generator=expected_generator,
    ) * torch.tensor([[0.5, 0.25]])
    actual, repeated = _capture(modifier.training_context, parameter)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(repeated, actual, rtol=0.0, atol=0.0)


def test_only_unique_dense_and_conv_weights_are_modified():
    duplicate_parameter = _dense()
    reference_parameter = _dense()
    clean = torch.tensor([[0.25, -0.5], [0.75, -1.0]])
    duplicate_parameter.state.copy_(clean)
    reference_parameter.state.copy_(clean)
    bias = Bias(shape=(2,), gain=0.0, device="cpu")
    pool = _pool()
    bias_clean = bias.state.detach().clone()
    pool_clean = pool.state.detach().clone()
    config = AddNormalConfig(std_dev=0.3, seed=17)
    duplicate_modifier = AddNormalParameterModifier(
        [duplicate_parameter, bias, duplicate_parameter, pool],
        config,
    )
    reference_modifier = AddNormalParameterModifier(
        [reference_parameter],
        config,
    )

    duplicate_first, _ = _capture(
        duplicate_modifier.training_context,
        duplicate_parameter,
    )
    reference_first, _ = _capture(
        reference_modifier.training_context,
        reference_parameter,
    )
    duplicate_second, _ = _capture(
        duplicate_modifier.training_context,
        duplicate_parameter,
    )
    reference_second, _ = _capture(
        reference_modifier.training_context,
        reference_parameter,
    )

    torch.testing.assert_close(
        duplicate_first,
        reference_first,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        duplicate_second,
        reference_second,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(bias.state, bias_clean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pool.state, pool_clean, rtol=0.0, atol=0.0)


def test_dense_and_conv_perturbations_obey_physical_bounds():
    dense = _dense(clamp=True, clamp_min=0.2, clamp_max=0.8)
    conv = _conv(clamp=True, clamp_min=0.1, clamp_max=0.9)
    dense.state.copy_(torch.tensor([[0.2, 0.8], [0.5, 0.5]]))
    conv.state.copy_(torch.tensor([[[[0.1, 0.9], [0.4, 0.6]]]]))
    dense_clean = dense.state.detach().clone()
    conv_clean = conv.state.detach().clone()
    dense_identity = id(dense.state)
    conv_identity = id(conv.state)
    modifier = AddNormalParameterModifier(
        [dense, conv],
        AddNormalConfig(std_dev=10.0, seed=3),
    )

    with modifier.training_context():
        assert torch.all(dense.state >= dense.state.new_tensor(0.2))
        assert torch.all(dense.state <= dense.state.new_tensor(0.8))
        assert torch.all(conv.state >= conv.state.new_tensor(0.1))
        assert torch.all(conv.state <= conv.state.new_tensor(0.9))
        assert id(dense.state) == dense_identity
        assert id(conv.state) == conv_identity

    torch.testing.assert_close(dense.state, dense_clean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(conv.state, conv_clean, rtol=0.0, atol=0.0)


def test_exception_restores_clean_weights_and_scope_is_reusable():
    parameter = _dense()
    clean = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    parameter.state.copy_(clean)
    modifier = AddNormalParameterModifier(
        [parameter],
        AddNormalConfig(std_dev=0.2, seed=19),
    )

    with pytest.raises(RuntimeError, match="phase failed"):
        with modifier.training_context():
            assert not torch.equal(parameter.state, clean)
            raise RuntimeError("phase failed")

    torch.testing.assert_close(parameter.state, clean, rtol=0.0, atol=0.0)
    with modifier.training_context():
        assert not torch.equal(parameter.state, clean)
    torch.testing.assert_close(parameter.state, clean, rtol=0.0, atol=0.0)


def test_overlapping_scopes_and_state_operations_in_scope_are_rejected():
    parameter = _dense()
    parameter.state.copy_(torch.tensor([[0.2, -0.4], [0.6, -0.8]]))
    modifier = AddNormalParameterModifier(
        [parameter],
        AddNormalConfig(
            std_dev=0.3,
            seed=23,
            noisy_evaluation=False,
        ),
    )
    saved = modifier.state_dict()

    with modifier.training_context():
        active = parameter.state.detach().clone()
        with pytest.raises(RuntimeError, match="not to overlap"):
            with modifier.evaluation_context():
                pass
        torch.testing.assert_close(
            parameter.state,
            active,
            rtol=0.0,
            atol=0.0,
        )
        with pytest.raises(RuntimeError, match="state_dict outside"):
            modifier.state_dict()
        with pytest.raises(RuntimeError, match="load_state_dict outside"):
            modifier.load_state_dict(saved)


def test_clean_evaluation_mode_does_not_draw_from_evaluation_stream():
    parameter = _dense()
    clean = torch.tensor([[0.2, -0.4], [0.6, -0.8]])
    parameter.state.copy_(clean)
    modifier = AddNormalParameterModifier(
        [parameter],
        AddNormalConfig(
            std_dev=0.3,
            seed=31,
            noisy_evaluation=False,
        ),
    )

    with modifier.evaluation_context():
        torch.testing.assert_close(
            parameter.state,
            clean,
            rtol=0.0,
            atol=0.0,
        )

    assert modifier.state_dict()["generator_states"]["evaluation"] == {}


def test_evaluation_sampling_does_not_advance_training_stream():
    clean = torch.tensor([[0.25, -0.5], [0.75, -1.0]])
    baseline_parameter = _dense()
    interleaved_parameter = _dense()
    baseline_parameter.state.copy_(clean)
    interleaved_parameter.state.copy_(clean)
    config = AddNormalConfig(
        std_dev=0.3,
        seed=37,
        noisy_evaluation=True,
    )
    baseline = AddNormalParameterModifier([baseline_parameter], config)
    interleaved = AddNormalParameterModifier([interleaved_parameter], config)

    baseline_first, _ = _capture(
        baseline.training_context,
        baseline_parameter,
    )
    baseline_second, _ = _capture(
        baseline.training_context,
        baseline_parameter,
    )
    interleaved_first, _ = _capture(
        interleaved.training_context,
        interleaved_parameter,
    )
    _capture(interleaved.evaluation_context, interleaved_parameter)
    interleaved_second, _ = _capture(
        interleaved.training_context,
        interleaved_parameter,
    )

    torch.testing.assert_close(
        interleaved_first,
        baseline_first,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        interleaved_second,
        baseline_second,
        rtol=0.0,
        atol=0.0,
    )


def test_seed_precedence_and_modifier_do_not_advance_global_rng():
    parameter = _dense()
    parameter.state.copy_(torch.tensor([[0.2, -0.4], [0.6, -0.8]]))

    explicit = AddNormalParameterModifier(
        [parameter],
        AddNormalConfig(std_dev=0.2, seed=41),
        run_seed=43,
    )
    assert explicit.resolved_seed == 41

    from_run = AddNormalParameterModifier(
        [parameter],
        AddNormalConfig(std_dev=0.2),
        run_seed=43,
    )
    assert from_run.resolved_seed == 43

    torch.manual_seed(47)
    before = torch.random.get_rng_state().clone()
    from_torch = AddNormalParameterModifier(
        [parameter],
        AddNormalConfig(
            std_dev=0.2,
            noisy_evaluation=True,
        ),
    )
    assert from_torch.resolved_seed == torch.initial_seed() == 47
    with from_torch.training_context():
        pass
    with from_torch.evaluation_context():
        pass
    after = torch.random.get_rng_state()

    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_state_round_trip_restores_train_and_evaluation_continuations():
    clean = torch.tensor([[0.2, -0.4], [0.6, -0.8]])
    first_parameter = _dense()
    first_parameter.state.copy_(clean)
    config = AddNormalConfig(
        std_dev=0.4,
        noisy_evaluation=True,
    )
    first = AddNormalParameterModifier(
        [first_parameter],
        config,
        run_seed=53,
    )

    _capture(first.training_context, first_parameter)
    _capture(first.evaluation_context, first_parameter)
    saved = copy.deepcopy(first.state_dict())
    expected_train, _ = _capture(first.training_context, first_parameter)
    expected_evaluation, _ = _capture(
        first.evaluation_context,
        first_parameter,
    )

    assert saved["version"] == 1
    assert saved["config"] == {
        "std_dev": 0.4,
        "seed": None,
        "noisy_evaluation": True,
        "scale_mode": "tensor_abs_max",
    }
    assert saved["resolved_seed"] == 53
    assert set(saved["generator_states"]) == {"train", "evaluation"}
    for stream in ("train", "evaluation"):
        assert set(saved["generator_states"][stream]) == {"cpu"}
        state = saved["generator_states"][stream]["cpu"]
        assert state.dtype == torch.uint8
        assert state.ndim == 1
        assert state.device.type == "cpu"

    second_parameter = _dense()
    second_parameter.state.copy_(clean)
    second = AddNormalParameterModifier(
        [second_parameter],
        config,
        run_seed=999,
    )
    second.load_state_dict(saved)

    assert second.resolved_seed == 53
    actual_train, _ = _capture(second.training_context, second_parameter)
    actual_evaluation, _ = _capture(
        second.evaluation_context,
        second_parameter,
    )
    torch.testing.assert_close(
        actual_train,
        expected_train,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual_evaluation,
        expected_evaluation,
        rtol=0.0,
        atol=0.0,
    )


def _invalid_states(valid):
    cases = []

    extra_key = copy.deepcopy(valid)
    extra_key["extra"] = None
    cases.append(extra_key)

    wrong_version = copy.deepcopy(valid)
    wrong_version["version"] = 2
    cases.append(wrong_version)

    unnormalized_config = copy.deepcopy(valid)
    unnormalized_config["config"]["std_dev"] = 1
    cases.append(unnormalized_config)

    mismatched_config = copy.deepcopy(valid)
    mismatched_config["config"]["noisy_evaluation"] = False
    cases.append(mismatched_config)

    missing_stream = copy.deepcopy(valid)
    del missing_stream["generator_states"]["evaluation"]
    cases.append(missing_stream)

    non_mapping_stream = copy.deepcopy(valid)
    non_mapping_stream["generator_states"]["train"] = []
    cases.append(non_mapping_stream)

    non_canonical_device = copy.deepcopy(valid)
    state = non_canonical_device["generator_states"]["train"].pop("cpu")
    non_canonical_device["generator_states"]["train"]["cuda"] = state
    cases.append(non_canonical_device)

    wrong_dtype = copy.deepcopy(valid)
    wrong_dtype["generator_states"]["train"]["cpu"] = torch.zeros(
        4,
        dtype=torch.float32,
    )
    cases.append(wrong_dtype)

    corrupt_bytes = copy.deepcopy(valid)
    corrupt_bytes["generator_states"]["train"]["cpu"] = torch.zeros(
        4,
        dtype=torch.uint8,
    )

    return cases, corrupt_bytes


def test_load_state_dict_rejects_malformed_state_without_partial_mutation():
    clean = torch.tensor([[0.2, -0.4], [0.6, -0.8]])
    config = AddNormalConfig(
        std_dev=0.3,
        seed=59,
        noisy_evaluation=True,
    )
    baseline_parameter = _dense()
    candidate_parameter = _dense()
    baseline_parameter.state.copy_(clean)
    candidate_parameter.state.copy_(clean)
    baseline = AddNormalParameterModifier([baseline_parameter], config)
    candidate = AddNormalParameterModifier([candidate_parameter], config)

    _capture(baseline.training_context, baseline_parameter)
    _capture(baseline.evaluation_context, baseline_parameter)
    _capture(candidate.training_context, candidate_parameter)
    _capture(candidate.evaluation_context, candidate_parameter)
    valid = candidate.state_dict()

    with pytest.raises(ValueError, match="Expected"):
        candidate.load_state_dict(None)

    restored_seed = copy.deepcopy(valid)
    restored_seed["resolved_seed"] = 61
    candidate.load_state_dict(restored_seed)
    assert candidate.resolved_seed == 61

    invalid_states, corrupt_state = _invalid_states(valid)
    for invalid in invalid_states:
        with pytest.raises(ValueError, match="Expected"):
            candidate.load_state_dict(invalid)
    with pytest.raises(RuntimeError):
        candidate.load_state_dict(corrupt_state)

    expected_train, _ = _capture(
        baseline.training_context,
        baseline_parameter,
    )
    expected_evaluation, _ = _capture(
        baseline.evaluation_context,
        baseline_parameter,
    )
    actual_train, _ = _capture(
        candidate.training_context,
        candidate_parameter,
    )
    actual_evaluation, _ = _capture(
        candidate.evaluation_context,
        candidate_parameter,
    )
    torch.testing.assert_close(
        actual_train,
        expected_train,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual_evaluation,
        expected_evaluation,
        rtol=0.0,
        atol=0.0,
    )
