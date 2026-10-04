from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from training.ibm_om_standard_crossbar import (
    CROSSBAR_TRAJECTORY_RNG_BACKEND,
    CROSSBAR_TRAJECTORY_REPRODUCIBILITY_SCOPE,
    CROSSBAR_TRAJECTORY_SEED_DERIVATION,
    CROSSBAR_TRAJECTORY_STATISTICAL_CONTRACT,
    IbmOmEffectiveCrossbarPlant,
    PulseAdam,
    PulseSGD,
    aihwkit_analog_linear_default_logical_weights,
    apply_population_bound_policy,
    build_crossbar_layout,
    build_deterministic_effective_codebook,
    crossbar_state_bundle,
    crossbar_trajectory_seeds,
    effective_state_to_logical_weights,
    flatten_logical_crossbar_matrices,
    local_star_crossbar_step,
    map_logical_weights,
    project_to_nearest_effective_code,
    project_to_nearest_effective_code_device,
    restore_crossbar_state_bundle,
    standard_crossbar_logits,
    standard_crossbar_forward_states,
    tensor_sha256,
    _counter_keyed_hash_pair,
    _counter_keyed_standard_normal,
)
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    om_array_population_fingerprint,
)


def _population(
    layout,
    *,
    minimum: float = -1.0,
    maximum: float = 1.0,
    reference: float = 0.0,
    noise: float = 0.0,
) -> IbmReramArrayPopulation:
    size = sum(tile.cells for tile in layout)
    return IbmReramArrayPopulation(
        assignment_seed=87004,
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(tile.key for tile in layout),
        binding_shapes=tuple(tile.shape for tile in layout),
        binding_sampling_seeds=tuple(range(11, 11 + len(layout))),
        donor_sampling_seeds=tuple(range(21, 21 + len(layout))),
        nominal_dw_min=0.1,
        dw_min_std=noise,
        write_noise_std=noise,
        max_bound=torch.full((size,), maximum),
        min_bound=torch.full((size,), minimum),
        dwmin_up=torch.full((size,), 0.1),
        dwmin_down=torch.full((size,), 0.1),
        reference=torch.full((size,), reference),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint="crossbar-fixture",
        aihwkit_version="1.1.0",
    )


def _published_population(
    layout,
    *,
    noise: float = 0.0,
) -> IbmReramArrayPopulation:
    population = _population(layout, noise=noise)
    corrupt = torch.zeros(population.size, dtype=torch.bool)
    corrupt[0] = True
    minimum = population.min_bound.clone()
    maximum = population.max_bound.clone()
    upward = population.dwmin_up.clone()
    downward = population.dwmin_down.clone()
    minimum[corrupt] = 0.005
    maximum[corrupt] = 0.005
    upward[corrupt] = 0.0
    downward[corrupt] = 0.0
    return replace(
        population,
        corruption_policy="published",
        min_bound=minimum,
        max_bound=maximum,
        dwmin_up=upward,
        dwmin_down=downward,
        corrupt=corrupt,
        published_corrupt=corrupt.clone(),
        fingerprint="published-crossbar-fixture",
    )


def test_layout_matches_aihwkit_balanced_512_input_tiling() -> None:
    layout = build_crossbar_layout((784, 50, 10), maximum_input_size=512)

    assert [(tile.key, tile.shape) for tile in layout] == [
        ("crossbar.layer0.tile0", (392, 50)),
        ("crossbar.layer0.tile1", (392, 50)),
        ("crossbar.layer1.tile0", (50, 10)),
    ]
    assert sum(tile.cells for tile in layout) == 39_700


def test_from_scratch_weights_match_bias_free_torch_linear_defaults() -> None:
    actual = aihwkit_analog_linear_default_logical_weights(
        (8, 3, 2),
        seed=42,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected_layers = (
            torch.nn.Linear(8, 3, bias=False),
            torch.nn.Linear(3, 2, bias=False),
        )
    expected = tuple(
        layer.weight.detach().transpose(0, 1).contiguous()
        for layer in expected_layers
    )

    assert tuple(value.shape for value in actual) == ((8, 3), (3, 2))
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    replay = aihwkit_analog_linear_default_logical_weights((8, 3, 2), seed=42)
    torch.testing.assert_close(actual[0], replay[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], replay[1], rtol=0.0, atol=0.0)


def test_mapping_round_trip_and_forward_match_digital_relu() -> None:
    generator = torch.Generator().manual_seed(7)
    layout = build_crossbar_layout((8, 3, 2), maximum_input_size=5)
    weights = (
        torch.randn((8, 3), generator=generator),
        torch.randn((3, 2), generator=generator),
    )
    inputs = torch.randn((4, 8), generator=generator)

    state, scales = map_logical_weights(
        weights,
        layout,
        weight_scaling_omega=(0.8, 0.9),
    )
    reconstructed = effective_state_to_logical_weights(
        state,
        layout,
        digital_scales=scales,
    )

    torch.testing.assert_close(reconstructed[0], weights[0])
    torch.testing.assert_close(reconstructed[1], weights[1])
    torch.testing.assert_close(
        standard_crossbar_logits(inputs, state, layout, digital_scales=scales),
        torch.relu(inputs @ weights[0]) @ weights[1],
    )
    states = standard_crossbar_forward_states(
        inputs, state, layout, digital_scales=scales
    )
    torch.testing.assert_close(states.hidden_preactivation, inputs @ weights[0])
    torch.testing.assert_close(states.hidden_post_relu, torch.relu(inputs @ weights[0]))
    torch.testing.assert_close(states.output_logits, torch.relu(inputs @ weights[0]) @ weights[1])


def test_zero_omega_keeps_direct_signed_weights_without_mapping() -> None:
    layout = build_crossbar_layout((4, 3, 2), maximum_input_size=2)
    weights = (
        torch.tensor(
            [
                [-1.0, -0.5, 0.0],
                [0.25, 0.5, 1.0],
                [-0.75, 0.125, 0.875],
                [0.625, -0.25, -0.125],
            ]
        ),
        torch.tensor(
            [
                [-1.0, 1.0],
                [-0.25, 0.5],
                [0.75, -0.5],
            ]
        ),
    )

    state, scales = map_logical_weights(
        weights,
        layout,
        weight_scaling_omega=(0.0, 0.0),
    )

    assert scales == (1.0, 1.0)
    torch.testing.assert_close(
        state,
        flatten_logical_crossbar_matrices(weights, layout),
    )
    reconstructed = effective_state_to_logical_weights(
        state,
        layout,
        digital_scales=scales,
    )
    torch.testing.assert_close(reconstructed[0], weights[0])
    torch.testing.assert_close(reconstructed[1], weights[1])


def test_local_star_gradients_match_two_stopped_local_objectives() -> None:
    generator = torch.Generator().manual_seed(91)
    layout = build_crossbar_layout((3, 2, 2), maximum_input_size=2)
    logical = (
        torch.randn((3, 2), generator=generator),
        torch.randn((2, 2), generator=generator),
    )
    state, scales = map_logical_weights(
        logical, layout, weight_scaling_omega=(0.8, 0.9)
    )
    inputs = torch.randn((4, 3), generator=generator)
    labels = torch.tensor([0, 1, 0, 1])
    hidden_targets = torch.randn((2, 2), generator=generator)
    output_targets = torch.randn((2, 2), generator=generator)
    step = local_star_crossbar_step(
        inputs,
        labels,
        state,
        layout,
        digital_scales=scales,
        hidden_targets=hidden_targets,
        output_targets=output_targets,
        hidden_gain=0.3,
        output_gain=0.7,
    )

    weight_0 = logical[0].detach().clone().requires_grad_(True)
    hidden = torch.relu(inputs @ weight_0)
    hidden_loss = 0.5 * 0.3 * (hidden - hidden_targets[labels]).square().sum(dim=1).mean()
    gradient_0 = torch.autograd.grad(hidden_loss, weight_0)[0]
    weight_1 = logical[1].detach().clone().requires_grad_(True)
    output = hidden.detach() @ weight_1
    output_loss = 0.5 * 0.7 * (output - output_targets[labels]).square().sum(dim=1).mean()
    gradient_1 = torch.autograd.grad(output_loss, weight_1)[0]
    expected = flatten_logical_crossbar_matrices(
        (scales[0] * gradient_0, scales[1] * gradient_1), layout
    )

    torch.testing.assert_close(step.gradient_q, expected)
    slices = (
        slice(0, logical[0].numel()),
        slice(logical[0].numel(), state.numel()),
    )
    changed_output_state = state.clone()
    changed_output_state[slices[1]] *= -3.0
    changed = local_star_crossbar_step(
        inputs,
        labels,
        changed_output_state,
        layout,
        digital_scales=scales,
        hidden_targets=hidden_targets,
        output_targets=output_targets,
        hidden_gain=0.3,
        output_gain=0.7,
    )
    torch.testing.assert_close(step.gradient_q[slices[0]], changed.gradient_q[slices[0]])


def test_bound_policy_retains_identity_and_matches_drn_winsorization() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout)
    population = replace(
        population,
        min_bound=torch.tensor([-1.4, -0.8, -1.2, -0.5, -2.0, -0.1]),
        max_bound=torch.tensor([1.3, 0.7, 1.2, 0.5, 2.0, 0.1]),
        reference=torch.tensor([0.1, -0.1, 0.0, 0.0, 0.2, 0.0]),
    )

    native, native_report = apply_population_bound_policy(
        population,
        policy="native_sampled_bounds",
    )
    matched, matched_report = apply_population_bound_policy(
        population,
        policy="winsorize_raw_active_a_to_unit_interval",
    )

    assert native.fingerprint == population.fingerprint
    assert torch.equal(native.min_bound, population.min_bound)
    assert torch.equal(native.max_bound, population.max_bound)
    assert native_report["any_bound_changed"] == 0
    torch.testing.assert_close(
        matched.min_bound,
        torch.maximum(population.min_bound, torch.full_like(population.min_bound, -1.0)),
    )
    torch.testing.assert_close(
        matched.max_bound,
        torch.minimum(population.max_bound, torch.full_like(population.max_bound, 1.0)),
    )
    assert matched.binding_sampling_seeds == population.binding_sampling_seeds
    assert torch.equal(matched.reference, population.reference)
    assert matched.fingerprint == om_array_population_fingerprint(
        assignment_seed=matched.assignment_seed,
        corruption_policy=matched.corruption_policy,
        keys=matched.binding_keys,
        shapes=matched.binding_shapes,
        binding_sampling_seeds=matched.binding_sampling_seeds,
        donor_sampling_seeds=matched.donor_sampling_seeds,
        scalar_parameters={
            "aihwkit_version": matched.aihwkit_version,
            "nominal_dw_min": matched.nominal_dw_min,
            "dw_min_std": matched.dw_min_std,
            "write_noise_std": matched.write_noise_std,
        },
        tensors=matched.tensor_state(),
    )
    assert matched_report["identity_resampled"] is False
    assert matched_report["any_bound_changed"] == 3


def test_deterministic_codebook_uses_effective_q_and_low_pulse_ties() -> None:
    layout = build_crossbar_layout((1, 1, 1), maximum_input_size=1)
    population = _population(layout, minimum=-1.0, maximum=1.0, reference=0.2)
    codebook = build_deterministic_effective_codebook(population, maximum_pulses=2)

    torch.testing.assert_close(codebook.values[0], torch.full((2,), -1.2))
    torch.testing.assert_close(codebook.values[1], torch.full((2,), -1.0))
    midpoint = (codebook.values[0] + codebook.values[1]) / 2.0
    realized, indices = project_to_nearest_effective_code(codebook, midpoint)

    assert torch.equal(indices, torch.zeros(2, dtype=torch.int64))
    torch.testing.assert_close(realized, codebook.values[0])

    device_realized, device_indices = project_to_nearest_effective_code_device(
        codebook.values,
        midpoint,
    )
    assert torch.equal(device_indices, indices)
    torch.testing.assert_close(device_realized, realized)
    cached_realized, cached_indices = project_to_nearest_effective_code_device(
        codebook.values.transpose(0, 1).contiguous(),
        midpoint,
        rowwise=True,
        validate=False,
    )
    assert torch.equal(cached_indices, indices)
    torch.testing.assert_close(cached_realized, realized)


def test_published_corrupt_cell_has_one_immutable_code_and_persistent_state() -> None:
    layout = build_crossbar_layout((1, 1, 1), maximum_input_size=1)
    population = _published_population(layout)
    codebook = build_deterministic_effective_codebook(
        population,
        maximum_pulses=4,
    )

    assert codebook.effective_level_counts.tolist() == [1, 5]
    torch.testing.assert_close(
        codebook.values[:, 0],
        codebook.values[0, 0].expand(5),
    )
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(17),
        device="cpu",
    )
    initial_corrupt = plant.persistent[0].clone()
    for _ in range(3):
        plant.pulse(torch.ones(population.size, dtype=torch.int8))

    torch.testing.assert_close(plant.persistent[0], initial_corrupt)
    assert plant.persistent[1] > population.logical_min[1]
    statistics = plant.pulse_statistics()
    assert statistics["final_corrupt_cells"] == 1
    assert statistics["published_corrupt_cells"] == 1
    assert statistics["commanded_final_corrupt_cells"] == 1
    assert statistics["pulses_to_final_corrupt_cells"] == 3
    assert statistics["persistent_moved_from_reset_cells"] == 1


def test_plant_checkpoint_replays_cycle_and_write_noise_exactly() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    generator = torch.Generator().manual_seed(123)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=generator,
        device="cpu",
    )
    plant.pulse(torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int8))
    checkpoint = plant.state_dict()
    commands = torch.tensor([-1, 1, 0, 1, 0, -1], dtype=torch.int8)
    plant.pulse(commands)
    expected = plant.state_dict()

    replay_generator = torch.Generator().manual_seed(999)
    replay = IbmOmEffectiveCrossbarPlant(
        population,
        generator=replay_generator,
        device="cpu",
    )
    replay.load_state_dict(checkpoint)
    replay.pulse(commands)
    actual = replay.state_dict()

    for name in (
        "persistent",
        "apparent",
        "upward_pulses",
        "downward_pulses",
        "trajectory_seeds",
        "trajectory_draw_indices",
    ):
        assert torch.equal(actual[name], expected[name])


def test_explicit_cell_trajectory_seeds_are_unique_and_bound_to_endpoint() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    first = crossbar_trajectory_seeds(
        population,
        endpoint_seed=2090601,
        stream_role="healthy_program_verify",
    )
    replay = crossbar_trajectory_seeds(
        population,
        endpoint_seed=2090601,
        stream_role="healthy_program_verify",
    )
    other_endpoint = crossbar_trajectory_seeds(
        population,
        endpoint_seed=2090602,
        stream_role="healthy_program_verify",
    )
    other_assignment = crossbar_trajectory_seeds(
        replace(population, assignment_seed=population.assignment_seed + 1),
        endpoint_seed=2090601,
        stream_role="healthy_program_verify",
    )
    other_role = crossbar_trajectory_seeds(
        population,
        endpoint_seed=2090601,
        stream_role="same_array_recovery_control",
    )

    assert first.dtype == torch.int64
    assert first.shape == (population.size,)
    assert int(torch.unique(first).numel()) == population.size
    assert bool(torch.all(first > 0))
    assert torch.equal(first, replay)
    assert not torch.equal(first, other_endpoint)
    assert not torch.equal(first, other_assignment)
    assert not torch.equal(first, other_role)


def test_per_cell_rng_advances_only_commanded_cells_and_is_batching_invariant() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    seeds = crossbar_trajectory_seeds(
        population,
        endpoint_seed=123,
        stream_role="per-cell-test",
    )

    batched = IbmOmEffectiveCrossbarPlant(
        population,
        trajectory_seeds=seeds,
        trajectory_seed_derivation=CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        device="cpu",
    )
    separated = IbmOmEffectiveCrossbarPlant(
        population,
        trajectory_seeds=seeds,
        trajectory_seed_derivation=CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        device="cpu",
    )
    initial = batched.state_dict()
    # Noisy lower-state conditioning consumes exactly one draw for every cell.
    assert initial["trajectory_draw_indices"].tolist() == [1] * population.size
    assert initial["rng_backend"] == CROSSBAR_TRAJECTORY_RNG_BACKEND
    assert initial["rng_reproducibility_scope"] == (
        CROSSBAR_TRAJECTORY_REPRODUCIBILITY_SCOPE
    )
    assert initial["rng_statistical_contract"] == (
        CROSSBAR_TRAJECTORY_STATISTICAL_CONTRACT
    )
    assert initial["trajectory_seed_derivation"] == (
        CROSSBAR_TRAJECTORY_SEED_DERIVATION
    )
    assert not any(
        isinstance(value, torch.Tensor) and value.ndim > 1
        for key, value in initial.items()
        if "trajectory" in key or "rng" in key
    )

    combined = torch.tensor([1, 0, -1, 0, 0, 0], dtype=torch.int8)
    batched.pulse(combined)
    separated.pulse(torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.int8))
    separated.pulse(torch.tensor([0, 0, -1, 0, 0, 0], dtype=torch.int8))

    expected_indices = torch.tensor([3, 1, 3, 1, 1, 1], dtype=torch.int64)
    assert torch.equal(
        batched.state_dict()["trajectory_draw_indices"],
        expected_indices,
    )
    assert torch.equal(
        separated.state_dict()["trajectory_draw_indices"],
        expected_indices,
    )
    torch.testing.assert_close(batched.persistent, separated.persistent)
    torch.testing.assert_close(batched.apparent, separated.apparent)


def test_counter_rng_has_expected_normal_moments_and_declared_backend_boundary() -> None:
    # One wide first-layer tile provides independent seed-zero-counter draws;
    # the small second layer simply preserves the required two-layer layout.
    layout = build_crossbar_layout((100_000, 1, 1), maximum_input_size=100_000)
    population = _population(layout, noise=0.2)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        trajectory_seeds=crossbar_trajectory_seeds(
            population,
            endpoint_seed=2090601,
            stream_role="normal-moment-test",
        ),
        trajectory_seed_derivation=CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        device="cpu",
    )
    samples = (plant.apparent - plant.persistent) / plant._write_scale
    mean = float(samples.mean().item())
    standard_deviation = float(samples.std(unbiased=True).item())
    lag_one_correlation = float(
        torch.corrcoef(torch.stack((samples[:-1], samples[1:])))[0, 1].item()
    )

    assert abs(mean) < 0.015
    assert 0.985 < standard_deviation < 1.015
    assert abs(lag_one_correlation) < 0.015
    state = plant.state_dict()
    assert "cpu_cuda_cross_backend_values_are_not_claimed_bit_exact" in state[
        "rng_reproducibility_scope"
    ]
    assert "62_bit_lane_pair_key_collision_space" in state[
        "rng_statistical_contract"
    ]


def test_counter_rng_pair_projection_is_unique_and_cross_streams_are_uncorrelated() -> None:
    # Exercise a two-dimensional cell/counter cohort.  This catches the subtle
    # error where both Box--Muller lanes are only salted transforms of one
    # shared 31-bit projection: such an implementation has at most 2^31 pair
    # states even though it serializes two output words.
    cell_seeds = torch.arange(1, 513, dtype=torch.int64)
    draw_counters = torch.arange(512, dtype=torch.int64)
    seeds = cell_seeds.repeat_interleave(draw_counters.numel())
    counters = draw_counters.repeat(cell_seeds.numel())
    radius_word, angle_word = _counter_keyed_hash_pair(seeds, counters)
    pairs = torch.stack((radius_word, angle_word), dim=1)
    assert int(torch.unique(pairs, dim=0).shape[0]) == pairs.shape[0]

    # Construct two keys with the exact same lane-0 preimage.  Under the
    # rejected shared-preimage-plus-two-salts design their complete output
    # pairs also collide.  Independent full-key projections intentionally
    # keep lane 1 distinct.
    lane_zero_counter_coefficient = 0xC2B2AE35 & 0x7FFFFFFF
    crafted_seeds = torch.tensor(
        [1, 1 ^ lane_zero_counter_coefficient],
        dtype=torch.int64,
    )
    crafted_counters = torch.tensor([0, 1], dtype=torch.int64)
    crafted_radius, crafted_angle = _counter_keyed_hash_pair(
        crafted_seeds,
        crafted_counters,
    )
    assert crafted_radius[0] == crafted_radius[1]
    assert crafted_angle[0] != crafted_angle[1]

    first = _counter_keyed_standard_normal(
        cell_seeds.repeat(128),
        torch.arange(cell_seeds.numel() * 128, dtype=torch.int64)
        // cell_seeds.numel(),
    )
    second = _counter_keyed_standard_normal(
        cell_seeds.repeat(128),
        torch.arange(cell_seeds.numel() * 128, dtype=torch.int64)
        // cell_seeds.numel()
        + 1,
    )
    cross_stream_correlation = float(
        torch.corrcoef(torch.stack((first, second)))[0, 1].item()
    )
    assert abs(cross_stream_correlation) < 0.015


def test_fault_transition_and_stuck_write_advance_only_touched_cell_streams() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    published = _published_population(layout, noise=0.2)
    healthy = replace(
        _population(layout, noise=0.2),
        published_corrupt=published.corrupt.clone(),
    )
    seeds = crossbar_trajectory_seeds(
        healthy,
        endpoint_seed=321,
        stream_role="fault-stream-test",
    )
    plant = IbmOmEffectiveCrossbarPlant(
        healthy,
        trajectory_seeds=seeds,
        trajectory_seed_derivation=CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        device="cpu",
    )
    before = plant.state_dict()["trajectory_draw_indices"]
    plant.apply_stuck_at_fault_transition(
        mask=published.corrupt,
        stuck_persistent_q=published.logical_min,
        transition_id="fault-rng-test",
        source_population_fingerprint=published.fingerprint,
    )
    after_transition = plant.state_dict()["trajectory_draw_indices"]
    assert torch.equal(
        after_transition,
        before + published.corrupt.to(torch.int64),
    )
    stuck = plant.persistent.clone()
    apparent_before_attempt = plant.apparent.clone()
    plant.pulse(published.corrupt.to(torch.int8))
    after_attempt = plant.state_dict()["trajectory_draw_indices"]
    assert torch.equal(
        after_attempt,
        after_transition + 2 * published.corrupt.to(torch.int64),
    )
    assert torch.equal(
        plant.persistent[published.corrupt],
        stuck[published.corrupt],
    )
    assert torch.equal(
        plant.apparent[~published.corrupt],
        apparent_before_attempt[~published.corrupt],
    )


def test_plant_rejects_forged_per_cell_rng_identity_and_chronology() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        trajectory_seeds=crossbar_trajectory_seeds(
            population,
            endpoint_seed=123,
            stream_role="rng-validation-test",
        ),
        device="cpu",
    )
    checkpoint = plant.state_dict()

    duplicate_seed = deepcopy(checkpoint)
    duplicate_seed["trajectory_seeds"][1] = duplicate_seed["trajectory_seeds"][0]
    with pytest.raises(ValueError, match="unique"):
        plant.load_state_dict(duplicate_seed)

    skipped_draw = deepcopy(checkpoint)
    skipped_draw["trajectory_draw_indices"][1] += 1
    with pytest.raises(ValueError, match="chronology"):
        plant.load_state_dict(skipped_draw)

    impossible_apparent = deepcopy(checkpoint)
    impossible_apparent["apparent"][1] += 0.25
    with pytest.raises(ValueError, match="deterministic last write-noise draw"):
        plant.load_state_dict(impossible_apparent)


def test_noiseless_plant_restore_rejects_apparent_persistent_divergence() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.0)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(123),
        device="cpu",
    )
    checkpoint = plant.state_dict()
    checkpoint["apparent"][0] += 0.25

    with pytest.raises(ValueError, match="deterministic last write-noise draw"):
        plant.load_state_dict(checkpoint)


def test_healthy_recovery_ports_use_apparent_q_without_exposing_plant_authority() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(123),
        device="cpu",
    )
    assert plant.state_kind == "healthy"
    local_port = plant.restricted_recovery_update_port()
    recovery_port = plant.on_chip_recovery_port(
        layout=layout,
        digital_scales=(1.0, 1.0),
    )

    assert local_port.state_hash_receipt()["plant_state_kind"] == "healthy"
    assert recovery_port.hardware_contract()["plant_state_kind"] == "healthy"
    assert recovery_port.hardware_contract()["fault_mask_exposed"] is False
    for forbidden in (
        "persistent",
        "population",
        "post_deployment_fault_mask",
        "post_deployment_stuck_persistent_q",
    ):
        assert not hasattr(local_port, forbidden)
        assert not hasattr(recovery_port, forbidden)
    inputs = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    expected = standard_crossbar_forward_states(
        inputs,
        plant.apparent,
        layout,
        digital_scales=(1.0, 1.0),
    )
    actual = recovery_port.forward(inputs)
    torch.testing.assert_close(actual.output_logits, expected.output_logits)

    persistent_before = plant.persistent.clone()
    apparent_before = plant.apparent.clone()
    local_port.pulse(torch.ones(population.size, dtype=torch.int8))
    assert torch.any(plant.persistent != persistent_before)
    assert torch.any(plant.apparent != apparent_before)


def test_healthy_crossbar_bundle_round_trips_and_replays_exactly() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(123),
        device="cpu",
    )
    plant.pulse(torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int8))
    bundle = crossbar_state_bundle(
        plant,
        layout=layout,
        digital_scales=(0.75, 1.25),
        metadata={
            "stage": "healthy_p0",
            "source": {"sha256": "a" * 64},
            "endpoint_seeds": (2090601, 2090602),
        },
    )
    serialized = bundle.state_dict()
    replay, parsed = restore_crossbar_state_bundle(
        serialized,
        population=population,
        generator=torch.Generator().manual_seed(999),
        device="cpu",
        expected_layout=layout,
        expected_digital_scales=(0.75, 1.25),
        expected_metadata_sha256=bundle.metadata_sha256,
    )

    assert parsed.state_kind == "healthy"
    assert parsed.metadata["endpoint_seeds"] == [2090601, 2090602]
    for name in (
        "persistent",
        "apparent",
        "upward_pulses",
        "downward_pulses",
        "trajectory_seeds",
        "trajectory_draw_indices",
    ):
        assert torch.equal(replay.state_dict()[name], plant.state_dict()[name])
    command = torch.tensor([-1, 1, 0, 1, 0, -1], dtype=torch.int8)
    plant.pulse(command)
    replay.pulse(command)
    assert torch.equal(replay.persistent, plant.persistent)
    assert torch.equal(replay.apparent, plant.apparent)


def test_crossbar_bundle_rejects_tampered_state_layout_scales_and_provenance() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(123),
        device="cpu",
    )
    bundle = crossbar_state_bundle(
        plant,
        layout=layout,
        digital_scales=(1.0, 1.0),
        metadata={"stage": "healthy_p0"},
    )

    tampered_state = bundle.state_dict()
    tampered_state["plant_state"]["persistent"][0] += 0.1
    with pytest.raises(ValueError, match="plant-state digest"):
        restore_crossbar_state_bundle(
            tampered_state,
            population=population,
            generator=torch.Generator().manual_seed(1),
            device="cpu",
            expected_layout=layout,
            expected_digital_scales=(1.0, 1.0),
        )

    with pytest.raises(ValueError, match="layout, scales"):
        restore_crossbar_state_bundle(
            bundle,
            population=population,
            generator=torch.Generator().manual_seed(1),
            device="cpu",
            expected_layout=layout,
            expected_digital_scales=(0.5, 1.0),
        )
    wrong_layout = build_crossbar_layout((2, 2, 1), maximum_input_size=1)
    with pytest.raises(ValueError, match="layout, scales"):
        restore_crossbar_state_bundle(
            bundle,
            population=_population(wrong_layout, noise=0.2),
            generator=torch.Generator().manual_seed(1),
            device="cpu",
            expected_layout=wrong_layout,
            expected_digital_scales=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="provenance digest"):
        restore_crossbar_state_bundle(
            bundle,
            population=population,
            generator=torch.Generator().manual_seed(1),
            device="cpu",
            expected_layout=layout,
            expected_digital_scales=(1.0, 1.0),
            expected_metadata_sha256="0" * 64,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unindexed_cuda_device_is_canonicalized_for_recovery_port() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator(device="cuda").manual_seed(123),
        device="cuda",
    )
    mask = torch.zeros(population.size, dtype=torch.bool)
    mask[0] = True
    plant.apply_stuck_at_fault_transition(
        mask=mask,
        stuck_persistent_q=population.logical_min,
        transition_id="cuda-device-canonicalization",
        source_population_fingerprint="cuda-test-population",
    )
    port = plant.on_chip_recovery_port(
        layout=layout,
        digital_scales=(1.0, 1.0),
    )

    assert plant.device == torch.device("cuda:0")
    assert port.hardware_contract()["device"] == "cuda:0"
    states = port.forward(torch.zeros((1, 2), device="cuda"))
    assert states.output_logits.device == torch.device("cuda:0")


def test_aihwkit_corrupt_transition_keeps_persistent_stuck_but_resamples_apparent() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    published = _published_population(layout, noise=0.2)
    population = replace(
        _population(layout, noise=0.2),
        published_corrupt=published.corrupt.clone(),
    )
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(1234),
        device="cpu",
    )
    plant.pulse(torch.ones(population.size, dtype=torch.int8))
    pre_fault = plant.state_dict()
    healthy_persistent = plant.persistent.clone()
    mask = published.corrupt
    stuck = published.logical_min
    report = plant.apply_stuck_at_fault_transition(
        mask=mask,
        stuck_persistent_q=stuck,
        transition_id="fault-1",
        source_population_fingerprint=published.fingerprint,
    )
    post = plant.state_dict()
    post_fault_apparent = plant.apparent[mask].clone()
    plant.pulse(torch.ones(population.size, dtype=torch.int8))

    torch.testing.assert_close(plant.persistent[mask], stuck[mask])
    assert not torch.equal(plant.apparent[mask], stuck[mask])
    assert not torch.equal(plant.apparent[mask], post_fault_apparent)
    assert torch.any(plant.persistent[~mask] != healthy_persistent[~mask])
    assert report["faulted_cells"] == 1
    assert report["non_fault_persistent_unchanged"] is True
    assert report["non_fault_apparent_unchanged"] is True
    assert report["persistent_fault_state"] == (
        "sampled_collapsed_bound_with_zero_pulse_increments"
    )
    assert report["apparent_fault_state"] == (
        "write_noise_resampled_at_transition_and_each_write_attempt"
    )
    assert plant.pulse_statistics()["pulses_to_post_deployment_fault_cells"] == 1
    with pytest.raises(RuntimeError, match="exactly one"):
        plant.apply_stuck_at_fault_transition(
            mask=mask,
            stuck_persistent_q=stuck,
            transition_id="fault-2",
            source_population_fingerprint=published.fingerprint,
        )

    replay = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(99),
        device="cpu",
    )
    with pytest.raises(ValueError, match="published companion population"):
        replay.load_state_dict(post)
    replay.load_state_dict(
        post,
        expected_fault_source_population=published,
        expected_pre_fault_state=pre_fault,
    )
    replay.pulse(torch.ones(population.size, dtype=torch.int8))
    torch.testing.assert_close(replay.persistent, plant.persistent)
    torch.testing.assert_close(replay.apparent, plant.apparent)


def test_faulted_crossbar_bundle_requires_p0_authority_and_keeps_faults_immutable() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    published = _published_population(layout, noise=0.2)
    healthy = replace(
        _population(layout, noise=0.2),
        published_corrupt=published.corrupt.clone(),
    )
    plant = IbmOmEffectiveCrossbarPlant(
        healthy,
        generator=torch.Generator().manual_seed(1234),
        device="cpu",
    )
    plant.pulse(torch.ones(healthy.size, dtype=torch.int8))
    healthy_p0 = plant.state_dict()
    plant.apply_stuck_at_fault_transition(
        mask=published.corrupt,
        stuck_persistent_q=published.logical_min,
        transition_id="faulted-bundle",
        source_population_fingerprint=published.fingerprint,
    )
    bundle = crossbar_state_bundle(
        plant,
        layout=layout,
        digital_scales=(1.0, 1.0),
        metadata={"stage": "post_deployment_corruption"},
    )
    assert bundle.state_kind == "faulted"
    with pytest.raises(ValueError, match="companion population"):
        restore_crossbar_state_bundle(
            bundle,
            population=healthy,
            generator=torch.Generator().manual_seed(99),
            device="cpu",
            expected_layout=layout,
            expected_digital_scales=(1.0, 1.0),
        )
    replay, _ = restore_crossbar_state_bundle(
        bundle,
        population=healthy,
        generator=torch.Generator().manual_seed(99),
        device="cpu",
        expected_layout=layout,
        expected_digital_scales=(1.0, 1.0),
        expected_fault_source_population=published,
        expected_pre_fault_state=healthy_p0,
    )
    port = replay.restricted_recovery_update_port()
    assert port.state_hash_receipt()["plant_state_kind"] == "faulted"
    for forbidden in (
        "persistent",
        "post_deployment_fault_mask",
        "post_deployment_stuck_persistent_q",
    ):
        assert not hasattr(port, forbidden)
    stuck_before = replay.persistent[published.corrupt].clone()
    apparent_before = replay.apparent[published.corrupt].clone()
    port.pulse(torch.ones(healthy.size, dtype=torch.int8))
    assert torch.equal(replay.persistent[published.corrupt], stuck_before)
    assert not torch.equal(replay.apparent[published.corrupt], apparent_before)


def test_faulted_plant_checkpoint_rejects_corrupt_transition_and_state() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    published = _published_population(layout, noise=0.2)
    population = replace(
        _population(layout, noise=0.2),
        published_corrupt=published.corrupt.clone(),
    )
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(1234),
        device="cpu",
    )
    plant.pulse(torch.ones(population.size, dtype=torch.int8))
    pre_fault = plant.state_dict()
    mask = published.corrupt
    stuck = published.logical_min
    plant.apply_stuck_at_fault_transition(
        mask=mask,
        stuck_persistent_q=stuck,
        transition_id="fault-1",
        source_population_fingerprint=published.fingerprint,
    )
    checkpoint = plant.state_dict()
    corrupted = []

    bad_transition = deepcopy(checkpoint)
    bad_transition["fault_transition"]["fault_mask_sha256"] = "0" * 64
    corrupted.append(bad_transition)

    missing_transition_field = deepcopy(checkpoint)
    del missing_transition_field["fault_transition"]["fault_fraction"]
    corrupted.append(missing_transition_field)

    boolean_transition_version = deepcopy(checkpoint)
    boolean_transition_version["fault_transition"]["schema_version"] = True
    corrupted.append(boolean_transition_version)

    floating_plant_version = deepcopy(checkpoint)
    floating_plant_version["schema_version"] = 2.0
    corrupted.append(floating_plant_version)

    nonfinite = deepcopy(checkpoint)
    nonfinite["persistent"][1] = torch.nan
    corrupted.append(nonfinite)

    negative_pulses = deepcopy(checkpoint)
    negative_pulses["upward_pulses"][1] = -1
    corrupted.append(negative_pulses)

    impossible_baseline = deepcopy(checkpoint)
    impossible_baseline["post_deployment_fault_pulse_baseline"].fill_(10**9)
    corrupted.append(impossible_baseline)

    coherent_wrong_stuck = deepcopy(checkpoint)
    coherent_wrong_stuck["post_deployment_stuck_persistent_q"][mask] += 100.0
    coherent_wrong_stuck["persistent"][mask] += 100.0
    coherent_wrong_stuck["fault_transition"]["stuck_persistent_q_sha256"] = (
        tensor_sha256(
            coherent_wrong_stuck["post_deployment_stuck_persistent_q"][mask]
        )
    )
    coherent_wrong_stuck["fault_transition"]["post_fault_persistent_sha256"] = (
        tensor_sha256(coherent_wrong_stuck["persistent"])
    )
    corrupted.append(coherent_wrong_stuck)

    wrong_pre_hash = deepcopy(checkpoint)
    wrong_pre_hash["fault_transition"]["pre_fault_persistent_sha256"] = "0" * 64
    corrupted.append(wrong_pre_hash)

    wrong_source = deepcopy(checkpoint)
    wrong_source["fault_transition"]["source_population_fingerprint"] = (
        "forged-published-population"
    )
    corrupted.append(wrong_source)

    for state in corrupted:
        replay = IbmOmEffectiveCrossbarPlant(
            population,
            generator=torch.Generator().manual_seed(99),
            device="cpu",
        )
        with pytest.raises(ValueError):
            replay.load_state_dict(
                state,
                expected_fault_source_population=published,
                expected_pre_fault_state=pre_fault,
            )

    outside_aihwkit_range = replace(
        published,
        min_bound=published.min_bound.clone(),
        max_bound=published.max_bound.clone(),
    )
    outside_aihwkit_range.min_bound[mask] = 0.05
    outside_aihwkit_range.max_bound[mask] = 0.05
    replay = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(99),
        device="cpu",
    )
    with pytest.raises(ValueError, match="configured corrupt range"):
        replay.load_state_dict(
            checkpoint,
            expected_fault_source_population=outside_aihwkit_range,
            expected_pre_fault_state=pre_fault,
        )


def test_pulse_adam_layer_scope_and_recovery_cap_are_physical() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(3),
        device="cpu",
    )
    optimizer = PulseAdam(
        size=population.size,
        layout=layout,
        learning_rates=(1.0, 1.0),
        betas=(0.0, 0.0),
        epsilon=1e-8,
        layer_scope="input_only",
        nominal_dw_min=0.1,
        pulse_cap_per_cell=2,
        generator=torch.Generator().manual_seed(4),
        device="cpu",
    )

    for _ in range(3):
        optimizer.step(-torch.ones(population.size), plant)
    report = optimizer.report()

    assert report["applied_pulses_by_layer"] == [8, 0]
    assert report["applied_pulses"] == 8
    assert report["cells_at_cap"] == 4
    assert report["maximum_pulses_per_cell"] == 2
    assert report["blocked_at_cap"] == 4
    assert report["enabled_cells"] == 4


def test_pulse_adam_can_disable_the_cumulative_recovery_cap() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(3),
        device="cpu",
    )
    optimizer = PulseAdam(
        size=population.size,
        layout=layout,
        learning_rates=(1.0, 1.0),
        betas=(0.0, 0.0),
        epsilon=1e-8,
        layer_scope="input_only",
        nominal_dw_min=0.1,
        pulse_cap_per_cell=None,
        generator=torch.Generator().manual_seed(4),
        device="cpu",
    )

    for _ in range(3):
        optimizer.step(-torch.ones(population.size), plant)
    report = optimizer.report()

    assert report["applied_pulses_by_layer"] == [12, 0]
    assert report["applied_pulses"] == 12
    assert report["pulse_cap_per_cell"] is None
    assert report["cells_at_cap"] == 0
    assert report["maximum_pulses_per_cell"] == 3
    assert report["blocked_at_cap"] == 0
    assert report["enabled_cells"] == 4


def test_zero_learning_rate_adam_is_an_exact_no_write_control() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(3),
        device="cpu",
    )
    before = plant.state_dict()
    optimizer = PulseAdam(
        size=population.size,
        layout=layout,
        learning_rates=(0.0, 0.0),
        betas=(0.9, 0.999),
        epsilon=1e-8,
        layer_scope="all",
        nominal_dw_min=0.1,
        pulse_cap_per_cell=640,
        generator=torch.Generator().manual_seed(4),
        device="cpu",
    )

    update = optimizer.step(
        torch.linspace(-1.0, 1.0, population.size),
        plant.restricted_recovery_update_port(),
    )
    after = plant.state_dict()

    assert update["commanded_pulses"] == 0
    assert update["maximum_probability_before_clip"] == 0.0
    assert optimizer.report()["optimizer_steps"] == 1
    assert optimizer.report()["applied_pulses"] == 0
    for name in (
        "persistent",
        "apparent",
        "upward_pulses",
        "downward_pulses",
        "trajectory_draw_indices",
    ):
        assert torch.equal(after[name], before[name])


def test_pulse_adam_checkpoint_and_plant_bundle_resume_bit_exactly() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(31),
        device="cpu",
    )
    optimizer = PulseAdam(
        size=population.size,
        layout=layout,
        learning_rates=(0.04, 0.03),
        betas=(0.9, 0.999),
        epsilon=1e-8,
        layer_scope="all",
        nominal_dw_min=0.1,
        pulse_cap_per_cell=8,
        generator=torch.Generator().manual_seed(41),
        device="cpu",
    )
    first_gradient = torch.tensor([0.5, -0.4, 0.3, -0.2, 0.1, -0.6])
    second_gradient = torch.tensor([-0.7, 0.6, -0.5, 0.4, -0.3, 0.2])
    optimizer.step(first_gradient, plant.restricted_recovery_update_port())
    bundle = crossbar_state_bundle(
        plant,
        layout=layout,
        digital_scales=(1.0, 1.0),
        metadata={"epoch": 1},
    )
    optimizer_state = optimizer.state_dict()
    optimizer.step(second_gradient, plant.restricted_recovery_update_port())
    expected_plant = plant.state_dict()
    expected_optimizer = optimizer.state_dict()

    replay_plant, _ = restore_crossbar_state_bundle(
        bundle,
        population=population,
        generator=torch.Generator().manual_seed(999),
        device="cpu",
        expected_layout=layout,
        expected_digital_scales=(1.0, 1.0),
    )
    replay_optimizer = PulseAdam(
        size=population.size,
        layout=layout,
        learning_rates=(0.04, 0.03),
        betas=(0.9, 0.999),
        epsilon=1e-8,
        layer_scope="all",
        nominal_dw_min=0.1,
        pulse_cap_per_cell=8,
        generator=torch.Generator().manual_seed(777),
        device="cpu",
    )
    replay_optimizer.load_state_dict(optimizer_state)
    replay_optimizer.step(
        second_gradient,
        replay_plant.restricted_recovery_update_port(),
    )
    actual_plant = replay_plant.state_dict()
    actual_optimizer = replay_optimizer.state_dict()

    for name in (
        "persistent",
        "apparent",
        "upward_pulses",
        "downward_pulses",
        "trajectory_seeds",
        "trajectory_draw_indices",
    ):
        assert torch.equal(actual_plant[name], expected_plant[name])
    assert actual_optimizer["contract"] == expected_optimizer["contract"]
    for name in (
        "first_moment",
        "second_moment",
        "pulse_count",
        "generator_state",
    ):
        assert torch.equal(actual_optimizer[name], expected_optimizer[name])
    for name in (
        "step_index",
        "requested",
        "applied",
        "probability_clipped",
        "blocked_at_cap",
    ):
        assert actual_optimizer[name] == expected_optimizer[name]


def test_pulse_adam_checkpoint_rejects_mismatched_contract_and_bad_state() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout)

    def optimizer_for(
        *,
        rates: tuple[float, float] = (0.04, 0.03),
        cap: int | None = 8,
        selected_layout=layout,
    ) -> PulseAdam:
        return PulseAdam(
            size=population.size,
            layout=selected_layout,
            learning_rates=rates,
            betas=(0.9, 0.999),
            epsilon=1e-8,
            layer_scope="all",
            nominal_dw_min=0.1,
            pulse_cap_per_cell=cap,
            generator=torch.Generator().manual_seed(41),
            device="cpu",
        )

    source = optimizer_for()
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(31),
        device="cpu",
    )
    source.step(torch.ones(population.size), plant.restricted_recovery_update_port())
    state = source.state_dict()
    mismatched = (
        optimizer_for(rates=(0.05, 0.03)),
        optimizer_for(cap=None),
        optimizer_for(
            selected_layout=build_crossbar_layout(
                (1, 2, 2),
                maximum_input_size=2,
            )
        ),
    )
    for optimizer in mismatched:
        with pytest.raises(ValueError, match="matching IBM OM pulse-Adam"):
            optimizer.load_state_dict(state)

    negative_second_moment = deepcopy(state)
    negative_second_moment["second_moment"][0] = -1.0
    with pytest.raises(ValueError, match="self-consistent"):
        optimizer_for().load_state_dict(negative_second_moment)
    bad_applied_count = deepcopy(state)
    bad_applied_count["applied"] += 1
    with pytest.raises(ValueError, match="self-consistent"):
        optimizer_for().load_state_dict(bad_applied_count)


def test_pulse_sgd_is_moment_free_and_obeys_layer_scope_and_cap() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(3),
        device="cpu",
    )
    optimizer = PulseSGD(
        size=population.size,
        layout=layout,
        learning_rates=(0.1, 0.1),
        layer_scope="output_only",
        nominal_dw_min=0.1,
        pulse_cap_per_cell=1,
        pulse_rule="stochastic_pulse_sign_sgd",
        generator=torch.Generator().manual_seed(4),
        device="cpu",
    )

    optimizer.step(torch.ones(population.size), plant)
    optimizer.step(torch.ones(population.size), plant)
    report = optimizer.report()

    assert report["commanded_pulses_by_layer"] == [0, 2]
    assert report["commanded_pulses"] == 2
    assert report["commanded_cells"] == 2
    assert report["blocked_at_cap"] == 2
    assert report["digital_first_moment_values"] == 0
    assert report["digital_second_moment_values"] == 0
    assert report["fp32_shadow_weight_values"] == 0
