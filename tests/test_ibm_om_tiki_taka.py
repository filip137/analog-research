from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from hardware.om_crossbar import (
    IbmOmEffectiveCrossbarPlant,
    build_crossbar_layout,
    effective_state_to_logical_weights,
    flatten_logical_crossbar_matrices,
)
from methods.optimizers.tiki_taka import (
    IbmOmDirectPulseSgd,
    IbmOmTikiTakaV1,
    build_tiki_taka_fast_zero_target,
    manual_label_ce_backprop,
    stochastic_compressed_pulse_counts,
)
from hardware.populations import IbmReramArrayPopulation


def _population(layout, *, noise: float = 0.0) -> IbmReramArrayPopulation:
    size = sum(tile.cells for tile in layout)
    return IbmReramArrayPopulation(
        assignment_seed=991,
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(tile.key for tile in layout),
        binding_shapes=tuple(tile.shape for tile in layout),
        binding_sampling_seeds=tuple(range(101, 101 + len(layout))),
        donor_sampling_seeds=tuple(range(201, 201 + len(layout))),
        nominal_dw_min=0.1,
        dw_min_std=noise,
        write_noise_std=noise,
        max_bound=torch.ones(size),
        min_bound=-torch.ones(size),
        dwmin_up=torch.full((size,), 0.1),
        dwmin_down=torch.full((size,), 0.1),
        reference=torch.zeros(size),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint="tiki-taka-fixture",
        aihwkit_version="1.1.0",
    )


def _set_q(plant: IbmOmEffectiveCrossbarPlant, q: torch.Tensor) -> None:
    value = q.detach().to(device=plant.device, dtype=torch.float32).reshape(-1)
    assert value.shape == plant.persistent.shape
    plant.persistent.copy_(value)
    plant.apparent.copy_(value)


def _faulted_slow_port(
    layout,
    q: torch.Tensor,
    scales=(0.5, 0.4),
    *,
    plant_seed: int = 11,
):
    plant = IbmOmEffectiveCrossbarPlant(
        _population(layout),
        generator=torch.Generator().manual_seed(plant_seed),
        device="cpu",
    )
    _set_q(plant, q)
    mask = torch.zeros(plant.size, dtype=torch.bool)
    mask[-1] = True
    plant.apply_stuck_at_fault_transition(
        mask=mask,
        stuck_persistent_q=q,
        transition_id="unit-test-fault",
        source_population_fingerprint="unit-test-published-companion",
    )
    return plant, plant.on_chip_recovery_port(
        layout=layout,
        digital_scales=scales,
    )


def _fast_port(layout, *, plant_seed: int = 12):
    plant = IbmOmEffectiveCrossbarPlant(
        _population(layout),
        generator=torch.Generator().manual_seed(plant_seed),
        device="cpu",
    )
    _set_q(plant, torch.zeros(plant.size))
    port = plant.tiki_taka_fast_port(layout=layout)
    hashes = port.state_hash_receipt()
    target = torch.zeros(plant.size)
    receipt = {
        "policy": "ideal_direct_strict_q_zero",
        "target_q_sha256": build_tiki_taka_fast_zero_target(
            plant.population
        ).report["target_q_sha256"],
        "post_program_persistent_sha256": hashes["persistent_sha256"],
        "post_program_apparent_sha256": hashes["apparent_sha256"],
        "physical_program_verify": False,
        "direct_state_assignment": True,
    }
    assert torch.equal(target, plant.persistent)
    return plant, port, receipt


def _logical_q(layout):
    q0 = torch.tensor(
        [
            [0.20, -0.10],
            [0.05, 0.30],
            [-0.20, 0.10],
            [0.15, -0.25],
        ],
        dtype=torch.float32,
    )
    q1 = torch.tensor([[0.30, -0.20], [-0.10, 0.25]], dtype=torch.float32)
    return flatten_logical_crossbar_matrices((q0, q1), layout)


def _training_batch():
    return (
        torch.tensor(
            [[0.7, -0.2, 0.4, 0.1], [-0.3, 0.8, 0.2, -0.5]],
            dtype=torch.float32,
        ),
        torch.tensor([0, 1], dtype=torch.int64),
    )


def test_stochastic_compressed_matches_sqrt_update_management_oracle() -> None:
    x = torch.tensor([0.8, -0.4])
    d = torch.tensor([0.3, -0.6])
    seed = 401
    result = stochastic_compressed_pulse_counts(
        x,
        d,
        learning_rate=0.2,
        weight_granularity=0.1,
        generator=torch.Generator().manual_seed(seed),
    )

    k = 0.2 * 0.8 * 0.6 / 0.1
    assert result.uncapped_bit_line_length == math.ceil(k) == 1
    assert result.bit_line_length == 1
    base = math.sqrt(0.2 / (0.1 * result.bit_line_length))
    root = math.sqrt(0.8 / 0.6)
    expected_input_probability = x.abs() * (base / root)
    expected_error_probability = d.abs() * (base * root)
    assert result.input_probability_maximum_before_clip == pytest.approx(
        float(expected_input_probability.max())
    )
    assert result.error_probability_maximum_before_clip == pytest.approx(
        float(expected_error_probability.max())
    )

    replay = torch.Generator().manual_seed(seed)
    input_bits = torch.rand((1, 2), generator=replay) < expected_input_probability
    error_bits = torch.rand((1, 2), generator=replay) < expected_error_probability
    coincidences = input_bits.transpose(0, 1).float() @ error_bits.float()
    update_sign = -(torch.sign(x).unsqueeze(1) * torch.sign(d).unsqueeze(0))
    torch.testing.assert_close(
        result.positive,
        torch.where(update_sign > 0, coincidences, 0).to(torch.int64),
    )
    torch.testing.assert_close(
        result.negative,
        torch.where(update_sign < 0, coincidences, 0).to(torch.int64),
    )


def test_stochastic_compressed_caps_bl_and_rescales_error_maximum() -> None:
    result = stochastic_compressed_pulse_counts(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        learning_rate=2.0,
        weight_granularity=0.1,
        generator=torch.Generator().manual_seed(9),
    )

    assert result.uncapped_bit_line_length == 40
    assert result.bit_line_length == 31
    d_value = 1.0 * 31.0 / 40.0
    base = math.sqrt(2.0 / (0.1 * 31.0))
    root = math.sqrt(2.0 / d_value)
    assert result.input_probability_maximum_before_clip == pytest.approx(
        2.0 * base / root
    )
    assert result.error_probability_maximum_before_clip == pytest.approx(base * root)
    assert result.input_probability_clipped == 0
    assert result.error_probability_clipped == 1
    assert result.negative.item() == 31
    assert result.positive.item() == 0


def test_manual_ce_bp_matches_mean_reduction_autograd_oracle() -> None:
    layout = build_crossbar_layout((4, 2, 2), maximum_input_size=2)
    q = _logical_q(layout)
    scales = (0.5, 0.4)
    _, port = _faulted_slow_port(layout, q, scales)
    inputs, labels = _training_batch()

    factors = manual_label_ce_backprop(port, inputs, labels)
    q0, q1 = effective_state_to_logical_weights(
        q,
        layout,
        digital_scales=(1.0, 1.0),
    )
    q0 = q0.clone().requires_grad_(True)
    q1 = q1.clone().requires_grad_(True)
    logits = torch.relu(inputs @ (scales[0] * q0)) @ (scales[1] * q1)
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="mean")
    gradient_q0, gradient_q1 = torch.autograd.grad(loss, (q0, q1))
    actual_q0 = factors.layer_inputs[0].transpose(0, 1) @ factors.layer_errors_q[0]
    actual_q1 = factors.layer_inputs[1].transpose(0, 1) @ factors.layer_errors_q[1]

    torch.testing.assert_close(factors.mean_cross_entropy, loss)
    torch.testing.assert_close(actual_q0, gradient_q0)
    torch.testing.assert_close(actual_q1, gradient_q1)
    expected_output_error = torch.softmax(logits.detach(), dim=1)
    expected_output_error[torch.arange(2), labels] -= 1.0
    expected_output_error /= 2
    torch.testing.assert_close(factors.output_error, expected_output_error)


def test_ports_hide_weights_faults_and_only_fast_selected_column_is_readable() -> None:
    layout = build_crossbar_layout((4, 2, 2), maximum_input_size=2)
    q = _logical_q(layout)
    _, slow_port = _faulted_slow_port(layout, q)
    fast_plant, fast_port, _ = _fast_port(layout)

    forbidden = (
        "apparent",
        "persistent",
        "population",
        "fault_mask",
        "post_deployment_fault_mask",
        "plant",
        "generator",
    )
    for port in (slow_port, fast_port):
        for name in forbidden:
            assert not hasattr(port, name)
        assert port.hardware_contract()["full_weight_tensor_exposed"] is False
        assert port.hardware_contract()["fault_mask_exposed"] is False
    assert not hasattr(fast_port, "forward")
    assert not hasattr(fast_port, "output_transpose_mvm")

    fast_plant.apparent.copy_(torch.arange(fast_plant.size, dtype=torch.float32))
    torch.testing.assert_close(
        fast_port.read_apparent_column(tile_index=0, column_index=1),
        torch.tensor([2.0, 3.0]),
    )
    with pytest.raises(ValueError, match="column"):
        fast_port.read_apparent_column(tile_index=0, column_index=2)


def test_grouped_counts_preserve_both_signs_and_stuck_cells_are_immutable() -> None:
    layout = build_crossbar_layout((1, 1, 1), maximum_input_size=1)
    q = torch.zeros(2)
    plant, port = _faulted_slow_port(layout, q, scales=(1.0, 1.0))
    stuck_before = plant.persistent[-1].clone()

    port.apply_grouped_pulse_counts(
        torch.tensor([2, 3], dtype=torch.int64),
        torch.tensor([1, 2], dtype=torch.int64),
    )

    assert plant.upward_pulses.tolist() == [2, 3]
    assert plant.downward_pulses.tolist() == [1, 2]
    torch.testing.assert_close(plant.persistent[-1], stuck_before)
    assert plant.pulse_statistics()["pulses_to_post_deployment_fault_cells"] == 5


def test_fast_zero_target_is_strict_zero_and_support_is_posthoc_only() -> None:
    layout = build_crossbar_layout((1, 1, 1), maximum_input_size=1)
    population = _population(layout)
    corrupt = torch.tensor([True, False])
    minimum = torch.tensor([0.005, 0.2])
    maximum = torch.tensor([0.005, 1.0])
    population = replace(
        population,
        corruption_policy="published",
        min_bound=minimum,
        max_bound=maximum,
        dwmin_up=torch.tensor([0.0, 0.1]),
        dwmin_down=torch.tensor([0.0, 0.1]),
        corrupt=corrupt,
        published_corrupt=corrupt.clone(),
        fingerprint="published-fast-fixture",
    )
    result = build_tiki_taka_fast_zero_target(population)

    torch.testing.assert_close(result.target_q, torch.zeros(2))
    assert result.report["policy"] == "strict_q_zero_target_mask_blind"
    assert result.report["target_all_zero"] is True
    assert result.report["target_construction"] == {
        "requested_q": "strict_zero_for_every_physical_cell",
        "uses_per_cell_support": False,
        "uses_corrupt_mask": False,
        "uses_sampled_stuck_values": False,
        "mutates_array": False,
    }
    diagnostic = result.report["posthoc_support_diagnostic"]
    assert diagnostic["analysis_only_not_used_by_target_or_controller"] is True
    assert diagnostic["target_in_sampled_support_cells"] == 0
    assert diagnostic["target_outside_sampled_support_cells"] == 2
    assert diagnostic["published_or_preexisting_corrupt_cells"] == 1
    assert diagnostic["corrupt_cells_with_target_outside_support"] == 1


def test_direct_pulse_sgd_is_exactly_replayable_and_has_no_hidden_optimizer() -> None:
    layout = build_crossbar_layout((4, 2, 2), maximum_input_size=2)
    q = _logical_q(layout)
    plant_a, port_a = _faulted_slow_port(layout, q, plant_seed=31)
    plant_b, port_b = _faulted_slow_port(layout, q, plant_seed=31)
    updater_a = IbmOmDirectPulseSgd(
        port=port_a,
        layout=layout,
        learning_rates_q=(0.5, 0.5),
        generator=torch.Generator().manual_seed(88),
        device="cpu",
    )
    updater_b = IbmOmDirectPulseSgd(
        port=port_b,
        layout=layout,
        learning_rates_q=(0.5, 0.5),
        generator=torch.Generator().manual_seed(88),
        device="cpu",
    )
    inputs, labels = _training_batch()
    first_a = updater_a.step(inputs, labels)
    updater_b.step(inputs, labels)
    checkpoint = updater_a.state_dict()
    port_hash_before = port_b.state_hash_receipt()
    updater_b.load_state_dict(checkpoint)
    assert port_b.state_hash_receipt() == port_hash_before

    second_a = updater_a.step(inputs.flip(0), labels.flip(0))
    second_b = updater_b.step(inputs.flip(0), labels.flip(0))
    assert second_a == second_b
    for name in ("persistent", "apparent", "upward_pulses", "downward_pulses"):
        assert torch.equal(getattr(plant_a, name), getattr(plant_b, name))
    state_a = updater_a.state_dict()
    state_b = updater_b.state_dict()
    for name in ("positive_counts", "negative_counts"):
        assert torch.equal(state_a[name], state_b[name])
    assert torch.equal(
        state_a["bit_line_engine"]["generator_state"],
        state_b["bit_line_engine"]["generator_state"],
    )
    report = updater_a.report()
    assert first_a["cross_entropy_sum"] == pytest.approx(
        first_a["mean_label_cross_entropy"] * first_a["examples"]
    )
    assert report["contract"]["teacher_queries"] == 0
    assert report["contract"]["autograd_calls"] == 0
    assert report["contract"]["adam_moment_values"] == 0
    assert report["contract"]["fp32_shadow_weight_values"] == 0
    assert report["contract"]["cumulative_pulse_cap"] is None
    assert report["fast_commanded_pulses"] == 0


def _tiki_taka_fixture(layout, q, *, seed_offset=0):
    slow_plant, slow_port = _faulted_slow_port(
        layout, q, plant_seed=51 + seed_offset
    )
    fast_plant, fast_port, receipt = _fast_port(
        layout, plant_seed=61 + seed_offset
    )
    updater = IbmOmTikiTakaV1(
        slow_port=slow_port,
        fast_port=fast_port,
        layout=layout,
        learning_rates_q=(0.5, 0.25),
        fast_initialization_receipt=receipt,
        generator=torch.Generator().manual_seed(71),
        device="cpu",
        transfer_every=1,
        n_reads_per_transfer=1,
        transfer_lr=1.0,
        scale_transfer_lr=True,
    )
    return slow_plant, fast_plant, slow_port, fast_port, updater


def test_tiki_taka_uses_per_physical_tile_cursors_and_separate_pulse_accounts() -> None:
    layout = build_crossbar_layout((4, 2, 2), maximum_input_size=2)
    assert [tile.shape for tile in layout] == [(2, 2), (2, 2), (2, 2)]
    q = _logical_q(layout)
    _, _, _, _, updater = _tiki_taka_fixture(layout, q)
    inputs, labels = _training_batch()

    steps = [updater.step(inputs, labels) for _ in range(3)]
    report = updater.report()

    assert all(step["transfer_due"] for step in steps)
    assert report["transfer_events"] == 3
    assert [item["cursor"] for item in report["physical_tile_progress"]] == [1, 1, 1]
    assert [item["column_reads"] for item in report["physical_tile_progress"]] == [3, 3, 3]
    assert [item["completed_cursor_passes"] for item in report["physical_tile_progress"]] == [1, 1, 1]
    assert [item["cursor_passes"] for item in report["physical_tile_progress"]] == [1.5, 1.5, 1.5]
    assert report["fast_commanded_pulses"] > 0
    assert report["slow_commanded_pulses"] > 0
    assert report["fast_commanded_cells"] > 0
    assert report["slow_commanded_cells"] > 0
    assert report["contract"]["fast_lr"] == 1.0
    assert report["contract"]["gamma"] == 0.0
    assert report["contract"]["fast_reset"] is False
    assert report["contract"]["transfer_selection"] == (
        "sequential_physical_tile_columns"
    )
    assert report["bit_line_engine"]["roles"]["fast_gradient"]["events"] == 18
    assert report["bit_line_engine"]["roles"]["slow_transfer"]["events"] == 9


def test_tiki_taka_state_replays_rng_cursors_fast_and_slow_updates_exactly() -> None:
    layout = build_crossbar_layout((4, 2, 2), maximum_input_size=2)
    q = _logical_q(layout)
    slow_a, fast_a, _, _, updater_a = _tiki_taka_fixture(layout, q)
    slow_b, fast_b, slow_port_b, fast_port_b, updater_b = _tiki_taka_fixture(
        layout, q
    )
    inputs, labels = _training_batch()
    updater_a.step(inputs, labels)
    updater_b.step(inputs, labels)
    checkpoint = updater_a.state_dict()
    hashes_before = (
        slow_port_b.state_hash_receipt(),
        fast_port_b.state_hash_receipt(),
    )
    updater_b.load_state_dict(checkpoint)
    assert hashes_before == (
        slow_port_b.state_hash_receipt(),
        fast_port_b.state_hash_receipt(),
    )

    result_a = updater_a.step(inputs.flip(0), labels.flip(0))
    result_b = updater_b.step(inputs.flip(0), labels.flip(0))
    assert result_a == result_b
    for plant_a, plant_b in ((slow_a, slow_b), (fast_a, fast_b)):
        for name in ("persistent", "apparent", "upward_pulses", "downward_pulses"):
            assert torch.equal(getattr(plant_a, name), getattr(plant_b, name))
    state_a = updater_a.state_dict()
    state_b = updater_b.state_dict()
    for name in (
        "tile_cursors",
        "tile_read_events",
        "fast_positive_counts",
        "fast_negative_counts",
        "slow_positive_counts",
        "slow_negative_counts",
    ):
        assert torch.equal(state_a[name], state_b[name])
    assert torch.equal(
        state_a["bit_line_engine"]["generator_state"],
        state_b["bit_line_engine"]["generator_state"],
    )

    corrupted = deepcopy(state_a)
    corrupted["tile_cursors"][0] = 1
    with pytest.raises(ValueError, match="cursor"):
        updater_b.load_state_dict(corrupted)


def test_tiki_taka_rejects_uncommissioned_fast_receipt_and_wrong_bl_contract() -> None:
    layout = build_crossbar_layout((4, 2, 2), maximum_input_size=2)
    q = _logical_q(layout)
    _, slow_port = _faulted_slow_port(layout, q)
    _, fast_port, receipt = _fast_port(layout)
    bad_receipt = dict(receipt)
    bad_receipt["post_program_apparent_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="commissioned"):
        IbmOmTikiTakaV1(
            slow_port=slow_port,
            fast_port=fast_port,
            layout=layout,
            learning_rates_q=(0.1, 0.1),
            fast_initialization_receipt=bad_receipt,
            generator=torch.Generator().manual_seed(3),
            device="cpu",
        )
    bad_target_receipt = dict(receipt)
    bad_target_receipt["target_q_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="literal target"):
        IbmOmTikiTakaV1(
            slow_port=slow_port,
            fast_port=fast_port,
            layout=layout,
            learning_rates_q=(0.1, 0.1),
            fast_initialization_receipt=bad_target_receipt,
            generator=torch.Generator().manual_seed(3),
            device="cpu",
        )
    with pytest.raises(ValueError, match="desired_bl=31"):
        IbmOmDirectPulseSgd(
            port=slow_port,
            layout=layout,
            learning_rates_q=(0.1, 0.1),
            generator=torch.Generator().manual_seed(3),
            device="cpu",
            desired_bl=30,
        )
