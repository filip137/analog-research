from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from ebl.runtime.store import sha256_file
from ebl.runtime.mnist_relu_drn.ibm_om_deployed_recovery import (
    _fixed_direct_gradient_thresholds,
)
from networks.resistive.builders import ParameterBinding
from networks.variable.parameter import DenseWeight
from hardware.ibm_om.population import IbmReramArrayPopulation
from methods.ibm_reram_recovery import IbmOmDeployedRecovery
from methods.ttv2_buffered_reference import buffered_transfer_update


_ROOT = Path(__file__).resolve().parents[1]
_DEVICE_MODEL = _ROOT / "data" / "ibm_reram_om_pv128_hwa_v1.json"


def _assert_nested_equal(left, right, *, path: str = "root") -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert isinstance(right, dict), path
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_nested_equal(left[key], right[key], path=f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_equal(
                left_item,
                right_item,
                path=f"{path}[{index}]",
            )
    else:
        assert left == right, path


def _binding(
    shape: tuple[int, int] = (2, 2),
    *,
    key: str = "base.dense_weight.0",
) -> ParameterBinding:
    parameter = DenseWeight(
        (shape[0],),
        (shape[1],),
        1.0,
        "cpu",
        clamp=True,
        clamp_min=0.1,
        clamp_max=1.0,
    )
    return ParameterBinding(key, parameter)


def _population(
    binding: ParameterBinding,
    *,
    fast: bool = False,
) -> IbmReramArrayPopulation:
    if fast:
        keys = (f"fast_pair.{binding.key}",)
        shapes = ((2 * binding.state.shape[0], binding.state.shape[1]),)
        size = 2 * binding.state.numel()
        fingerprint = "fixture-fast"
    else:
        keys = (binding.key,)
        shapes = (tuple(binding.state.shape),)
        size = binding.state.numel()
        fingerprint = "fixture-slow"
    zeros = torch.zeros(size)
    false = torch.zeros(size, dtype=torch.bool)
    return IbmReramArrayPopulation(
        assignment_seed=9001 if fast else 9000,
        corruption_policy="counterfactual_repaired",
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=(11,),
        donor_sampling_seeds=(12,),
        nominal_dw_min=0.0949,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=torch.ones(size),
        min_bound=-torch.ones(size),
        dwmin_up=torch.full((size,), 0.2),
        dwmin_down=torch.full((size,), 0.2),
        reference=zeros,
        corrupt=false,
        published_corrupt=false.clone(),
        fingerprint=fingerprint,
        aihwkit_version="1.1.0",
    )


def _combined_slow_population(
    bindings: tuple[ParameterBinding, ...],
) -> IbmReramArrayPopulation:
    keys = tuple(binding.key for binding in bindings)
    shapes = tuple(tuple(binding.state.shape) for binding in bindings)
    size = sum(binding.state.numel() for binding in bindings)
    zeros = torch.zeros(size)
    false = torch.zeros(size, dtype=torch.bool)
    return IbmReramArrayPopulation(
        assignment_seed=9000,
        corruption_policy="counterfactual_repaired",
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=tuple(11 + index for index in range(len(bindings))),
        donor_sampling_seeds=tuple(21 + index for index in range(len(bindings))),
        nominal_dw_min=0.0949,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=torch.ones(size),
        min_bound=-torch.ones(size),
        dwmin_up=torch.full((size,), 0.2),
        dwmin_down=torch.full((size,), 0.2),
        reference=zeros,
        corrupt=false,
        published_corrupt=false.clone(),
        fingerprint="fixture-slow-combined",
        aihwkit_version="1.1.0",
    )


def _aihwkit_multi_recovery(
    bindings: tuple[ParameterBinding, ...],
    *,
    total_steps: int,
    learning_rates: tuple[float, ...],
    slow_pulse_budget_per_cell: float,
) -> IbmOmDeployedRecovery:
    population = _combined_slow_population(bindings)
    size = population.size
    source = torch.full((size,), 0.5)
    generator = torch.Generator().manual_seed(44)
    keys = tuple(binding.key for binding in bindings)
    return IbmOmDeployedRecovery(
        bindings,
        slow_population=population,
        source_raw_apparent=source,
        source_apparent=source,
        source_persistent=source,
        requested_target=source.clone(),
        reset_baseline=source.clone(),
        generator_state_after_programming=generator.get_state(),
        parameters=_parameters(
            "ttv2_aihwkit_1p1_minibatch_equation",
            fast_backend="ideal",
            binding_keys=keys,
            slow_pulse_budget_per_cell=slow_pulse_budget_per_cell,
        ),
        learning_rates=learning_rates,
        conductance_min=0.1,
        conductance_max=1.0,
        total_steps=total_steps,
        source_deployment_sha256="d" * 64,
        device_model_path=_DEVICE_MODEL,
    )


def _parameters(
    method: str,
    *,
    fast_backend: str | None = None,
    gradient_percentile: float | None = None,
    binding_keys: tuple[str, ...] = ("base.dense_weight.0",),
    slow_pulse_budget_per_cell: float = 1.0,
    ttv2_transfer_every: int = 1,
) -> dict:
    result = {
        "method": method,
        "slow_pulse_budget_per_cell": slow_pulse_budget_per_cell,
        "pulse_step_ratio": 0.04745,
        "update_seed": 123,
        "direct_probability_scale": None,
        "fast_backend": fast_backend,
        "fast_endpoint_seed": 456 if fast_backend == "physical_om" else None,
        "fast_reset_read_samples": 2 if fast_backend == "physical_om" else None,
        "fast_reset_guard_standard_errors": (
            0.0 if fast_backend == "physical_om" else None
        ),
        "expected_device_model_sha256": sha256_file(_DEVICE_MODEL),
        "source_dual_rail_layout_by_parameter": {
            key: "halves" for key in binding_keys
        },
    }
    if gradient_percentile is not None:
        result["direct_gradient_magnitude_percentile"] = gradient_percentile
    if method == "ttv2":
        result.update(
            {
                "ttv2_transfer_every": ttv2_transfer_every,
                "ttv2_gamma0": 10000.0,
                "ttv2_fast_update_model": "symmetric_soft_bounds",
                "ttv2_fast_weight_limit": 1.0,
                "ttv2_scan_mode": "dual_rail_input_pair",
                "ttv2_buffer_threshold": 1.0,
                "ttv2_buffer_residual_mode": "subtract_dispatched",
            }
        )
    elif method == "ttv2_aihwkit_1p1_minibatch_equation":
        result.update(
            {
                "ttv2_transfer_every": ttv2_transfer_every,
                "ttv2_fast_weight_limit": 1.0,
                "ttv2_fast_lr_by_parameter": {
                    key: 0.5 for key in binding_keys
                },
                "ttv2_transfer_lr": 1.0,
                "ttv2_scale_transfer_lr": True,
                "ttv2_units_in_mbatch": True,
                "ttv2_auto_scale": False,
                "ttv2_fast_granularity_by_parameter": {
                    key: 0.2 for key in binding_keys
                },
                "ttv2_buffer_granularity": 0.5,
                "ttv2_auto_granularity": 4.0,
                "ttv2_correct_gradient_magnitudes": True,
                "ttv2_desired_bl": 1,
                "ttv2_momentum": 0.0,
                "ttv2_forget_buffer": True,
                "ttv2_cap_scope": "per_parameter_proportional",
                "ttv2_cursor_policy": "zero",
                "ttv2_in_chop_probability": 0.0,
            }
        )
    return result


def _recovery(
    method: str,
    *,
    total_steps: int,
    fast_backend: str | None = None,
    gradient_percentile: float | None = None,
    shape: tuple[int, int] = (2, 2),
    learning_rate: float = 1.0,
    slow_pulse_budget_per_cell: float = 1.0,
    ttv2_transfer_every: int = 1,
) -> tuple[ParameterBinding, IbmOmDeployedRecovery]:
    binding = _binding(shape)
    generator = torch.Generator().manual_seed(44)
    source = torch.full((binding.state.numel(),), 0.5)
    requested_target = (
        torch.tensor([0.6, 0.5, 0.5, 0.6])
        if shape == (2, 2)
        else source.clone()
    )
    recovery = IbmOmDeployedRecovery(
        (binding,),
        slow_population=_population(binding),
        source_raw_apparent=source,
        source_apparent=source,
        source_persistent=source,
        requested_target=requested_target,
        reset_baseline=source.clone(),
        generator_state_after_programming=generator.get_state(),
        parameters=_parameters(
            method,
            fast_backend=fast_backend,
            gradient_percentile=gradient_percentile,
            slow_pulse_budget_per_cell=slow_pulse_budget_per_cell,
            ttv2_transfer_every=ttv2_transfer_every,
        ),
        learning_rates=(learning_rate,),
        conductance_min=0.1,
        conductance_max=1.0,
        total_steps=total_steps,
        source_deployment_sha256="d" * 64,
        device_model_path=_DEVICE_MODEL,
        fast_population=(
            _population(binding, fast=True)
            if fast_backend == "physical_om"
            else None
        ),
    )
    return binding, recovery


def test_rail_refresh_spreads_exact_cap_and_uses_target_rail_direction() -> None:
    _binding_value, recovery = _recovery("rail_refresh", total_steps=4)

    assert not recovery.requires_gradients
    for _step in range(4):
        recovery.step()

    report = recovery.report()
    assert report["slow_pulse_cap"] == 4
    assert report["slow"]["total_requested"] == 4
    assert report["slow"]["set_requested"] == 2
    assert report["slow"]["reset_requested"] == 2
    assert report["slow"]["unique_cells_touched"] == 4


def test_direct_pulses_follow_individual_gradient_sign_and_cap() -> None:
    binding, recovery = _recovery("direct_pulse", total_steps=1)
    binding.state.grad = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
    recovery.set_direct_probability_scale(1e6, report={"fixture": True})

    recovery.step()

    report = recovery.report()
    assert report["slow"]["set_requested"] == 2
    assert report["slow"]["reset_requested"] == 2
    assert report["slow"]["total_requested"] == report["slow_pulse_cap"] == 4
    persistent = recovery.current_persistent_endpoint()
    assert torch.all(persistent[[0, 2]] > 0.5)
    assert torch.all(persistent[[1, 3]] < 0.5)


def test_direct_gradient_gate_is_absolute_strict_and_resume_exact() -> None:
    binding, recovery = _recovery(
        "direct_pulse",
        total_steps=1,
        gradient_percentile=90.0,
    )
    binding.state.grad = torch.tensor([[-1.0, 0.5], [0.5001, 0.1]])
    recovery.set_direct_gradient_thresholds(
        {binding.key: 0.5},
        report={"percentile": 90.0, "calibration_sha256": "a" * 64},
    )
    recovery.set_direct_probability_scale(1e6, report={"fixture": True})

    recovery.step()

    report = recovery.report()
    assert report["slow"]["set_requested"] == 1
    assert report["slow"]["reset_requested"] == 1
    gate = report["direct_gradient_gate"]
    assert gate["comparison"] == "strictly_greater_than"
    assert gate["threshold_by_parameter"] == {binding.key: 0.5}
    assert gate["cumulative_by_parameter"][binding.key] == {
        "observed": 4,
        "nonzero": 4,
        "eligible": 2,
        "suppressed_nonzero": 2,
        "eligible_fraction_of_observed": 0.5,
        "eligible_fraction_of_nonzero": 0.5,
        "suppressed_fraction_of_nonzero": 0.5,
    }

    saved = deepcopy(recovery.state_dict())
    _binding_value, restored = _recovery(
        "direct_pulse",
        total_steps=1,
        gradient_percentile=90.0,
    )
    restored.load_state_dict(saved)
    assert restored.report() == report


def test_gated_direct_recovery_requires_threshold_calibration() -> None:
    binding, recovery = _recovery(
        "direct_pulse",
        total_steps=1,
        gradient_percentile=99.0,
    )
    binding.state.grad = torch.ones_like(binding.state)
    recovery.set_direct_probability_scale(1.0, report={"fixture": True})
    with pytest.raises(RuntimeError, match="thresholds to be calibrated"):
        recovery.step()


def test_fixed_direct_gradient_threshold_uses_nonzero_linear_quantile() -> None:
    thresholds, report = _fixed_direct_gradient_thresholds(
        {
            "base.dense_weight.0": [
                torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            ]
        },
        percentile=90.0,
        calibration_batches=1,
    )

    assert thresholds == {"base.dense_weight.0": pytest.approx(4.6)}
    parameter = report["parameters"]["base.dense_weight.0"]
    assert parameter["observations"] == 6
    assert parameter["nonzero"] == 5
    assert parameter["eligible"] == 1
    assert parameter["suppressed_nonzero"] == 4
    assert len(parameter["raw_gradient_magnitude_sha256"]) == 64
    assert len(report["calibration_sha256"]) == 64


def test_ideal_tiki_taka_transfers_each_column_and_clears_fast_state() -> None:
    binding, recovery = _recovery(
        "tiki_taka",
        total_steps=2,
        fast_backend="ideal",
    )
    for _step in range(2):
        binding.state.grad = -torch.ones_like(binding.state)
        recovery.step()

    report = recovery.report()
    assert report["transfer_count_by_parameter"] == {
        binding.key: 2
    }
    assert report["slow"]["total_requested"] == 4
    assert recovery.ideal_fast is not None
    # The just-transferred column is cleared; the other column already holds
    # the second minibatch's new gradient for its next cyclic visit.
    assert torch.count_nonzero(recovery.ideal_fast[0]) == 2


def test_ttv2_scans_complete_dual_rail_input_pair_and_preserves_fast_state() -> None:
    binding, recovery = _recovery(
        "ttv2",
        total_steps=2,
        fast_backend="ideal",
        shape=(4, 6),
    )
    assert recovery.ideal_fast is not None
    assert recovery.ttv2_hidden is not None
    recovery.transfer_start[0] = 0
    recovery.ttv2_transfer_scale[0] = 1.0
    recovery.ideal_fast[0][[0, 2], :] = 1.0
    recovery.ideal_fast[0][[1, 3], :] = -1.0
    before = recovery.ideal_fast[0].clone()
    binding.state.grad = torch.zeros_like(binding.state)

    recovery.step()

    report = recovery.report()
    assert report["algorithm_class"] == "ttv2_persistent_fast_digital_residual"
    assert report["transfer_count_by_parameter"] == {binding.key: 1}
    assert report["transfer_opportunities"] == 12
    assert report["slow"]["total_requested"] == 12
    assert report["slow"]["set_requested"] == 12
    assert torch.equal(recovery.ideal_fast[0], before)
    persistent = recovery.current_persistent_endpoint().reshape(4, 6)
    assert torch.all(persistent[[0, 2], :] > 0.5)
    assert torch.all(persistent[[1, 3], :] == 0.5)


def test_ttv2_accumulates_subpulse_evidence_and_subtracts_only_dispatch() -> None:
    binding, recovery = _recovery(
        "ttv2",
        total_steps=5,
        fast_backend="ideal",
        shape=(4, 2),
    )
    assert recovery.ideal_fast is not None
    assert recovery.ttv2_hidden is not None
    recovery.transfer_start[0] = 0
    recovery.ttv2_transfer_scale[0] = 1.0
    recovery.ideal_fast[0][[0, 2], :] = 0.4
    binding.state.grad = torch.zeros_like(binding.state)

    for _ in range(4):
        recovery.step()
        binding.state.grad = torch.zeros_like(binding.state)
    assert recovery.report()["slow"]["total_requested"] == 0
    assert torch.allclose(
        recovery.ttv2_hidden[0][[0, 2], :],
        torch.full((2, 2), 0.8),
    )

    recovery.step()

    assert recovery.report()["slow"]["total_requested"] == 4
    assert torch.allclose(
        recovery.ttv2_hidden[0][[0, 2], :],
        torch.full((2, 2), 0.2),
        atol=1e-6,
    )


def test_ttv2_cap_truncation_keeps_each_retained_cells_own_sign() -> None:
    binding, recovery = _recovery(
        "ttv2",
        total_steps=1,
        fast_backend="ideal",
        shape=(4, 2),
    )
    assert recovery.ideal_fast is not None
    recovery.transfer_start[0] = 0
    recovery.ttv2_transfer_scale[0] = 1.0
    recovery.slow_pulse_cap = 2
    recovery.ideal_fast[0][0, :] = torch.tensor([1.0, -1.0])
    recovery.ideal_fast[0][2, :] = torch.tensor([-1.0, 1.0])
    binding.state.grad = torch.zeros_like(binding.state)

    # Force a non-identity cap selection: global indices [5, 4] retain a
    # positive and a negative local H value in the opposite order from the
    # row-major crossing list.
    recovery._trim_to_remaining_cap = lambda indices: indices.flip(0)[:2]
    recovery.step()

    ledger = recovery.state_dict()["slow_ledger"]
    assert torch.equal(
        ledger["set_count"],
        torch.tensor([0, 0, 0, 0, 0, 1, 0, 0]),
    )
    assert torch.equal(
        ledger["reset_count"],
        torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]),
    )
    assert recovery.ttv2_hidden is not None
    # The retained order is [5, 4], opposite row-major order.  Each retained
    # cell must subtract its own signed pulse from H; deferred cells keep the
    # complete crossing.
    hidden = recovery.ttv2_hidden[0].reshape(-1)
    assert torch.equal(hidden[[4, 5]], torch.zeros(2))
    assert torch.equal(hidden[[0, 1]], torch.tensor([1.0, -1.0]))


def test_ttv2_resume_preserves_fast_buffer_and_scan_cursor_bit_exact() -> None:
    binding_a, recovery_a = _recovery(
        "ttv2",
        total_steps=3,
        fast_backend="ideal",
        shape=(4, 2),
    )
    recovery_a.transfer_start[0] = 0
    recovery_a.ttv2_transfer_scale[0] = 1.0
    binding_a.state.grad = -torch.full_like(binding_a.state, 0.3)
    recovery_a.step()
    saved = deepcopy(recovery_a.state_dict())

    binding_b, recovery_b = _recovery(
        "ttv2",
        total_steps=3,
        fast_backend="ideal",
        shape=(4, 2),
    )
    recovery_b.ttv2_transfer_scale[0] = 1.0
    recovery_b.load_state_dict(saved)

    for binding, recovery in ((binding_a, recovery_a), (binding_b, recovery_b)):
        binding.state.grad = torch.full_like(binding.state, 0.2)
        recovery.step()
    assert recovery_a.report() == recovery_b.report()
    for key, value in recovery_a.state_dict().items():
        counterpart = recovery_b.state_dict()[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, counterpart), key
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            assert all(torch.equal(a, b) for a, b in zip(value, counterpart)), key


def test_ttv2_aihwkit_places_optimizer_lr_only_in_lambda_h() -> None:
    binding_a, recovery_a = _recovery(
        "ttv2_aihwkit_1p1_minibatch_equation",
        total_steps=1,
        fast_backend="ideal",
        shape=(4, 2),
        learning_rate=1.0,
        ttv2_transfer_every=2,
    )
    binding_b, recovery_b = _recovery(
        "ttv2_aihwkit_1p1_minibatch_equation",
        total_steps=1,
        fast_backend="ideal",
        shape=(4, 2),
        learning_rate=0.25,
        ttv2_transfer_every=2,
    )
    gradient = torch.tensor(
        [[0.09, -0.18], [0.27, -0.36], [-0.09, 0.18], [-0.27, 0.36]]
    )
    for binding, recovery in ((binding_a, recovery_a), (binding_b, recovery_b)):
        binding.state.grad = gradient.clone()
        recovery.step()

    assert recovery_a.ideal_fast is not None
    assert recovery_b.ideal_fast is not None
    expected_fast = -0.5 * gradient / 0.9
    assert torch.allclose(recovery_a.ideal_fast[0], expected_fast)
    assert torch.equal(recovery_a.ideal_fast[0], recovery_b.ideal_fast[0])

    # b0 = buffer_granularity * fast_dw * auto_granularity / (n * period)
    # and corrected b = b0 * slow_delta / fast_dw.
    expected_b0 = 0.5 * 0.2 * 4.0 / (2 * 2)
    expected_b = expected_b0 * 0.04745 / 0.2
    report_a = recovery_a.report()["ttv2"]
    report_b = recovery_b.report()["ttv2"]
    assert report_a["buffer_scale_uncorrected_by_parameter"][binding_a.key] == (
        pytest.approx(expected_b0)
    )
    assert report_a["buffer_scale_corrected_by_parameter"][binding_a.key] == (
        pytest.approx(expected_b)
    )
    assert report_a["effective_transfer_lr_by_parameter"][binding_a.key] == 1.0
    assert report_b["effective_transfer_lr_by_parameter"][binding_b.key] == 0.25
    assert report_a["lambda_h_by_parameter"][binding_a.key] == pytest.approx(
        1.0 / (0.5 * expected_b)
    )
    assert report_b["lambda_h_by_parameter"][binding_b.key] == pytest.approx(
        0.25 / (0.5 * expected_b)
    )


def test_ttv2_aihwkit_forget_buffer_clears_only_dispatched_omega() -> None:
    binding, recovery = _recovery(
        "ttv2_aihwkit_1p1_minibatch_equation",
        total_steps=1,
        fast_backend="ideal",
        shape=(4, 2),
    )
    assert recovery.ideal_fast is not None
    assert recovery.ttv2_hidden is not None
    lambda_h = recovery.ttv2_transfer_scale[0]
    omega = torch.tensor([[1.7, -1.7], [-1.7, 1.7]])
    recovery.ideal_fast[0][[0, 2], :] = omega / lambda_h
    binding.state.grad = torch.zeros_like(binding.state)

    recovery.step()

    assert torch.equal(
        recovery.ttv2_hidden[0][[0, 2], :],
        torch.zeros((2, 2)),
    )
    report = recovery.report()
    assert report["slow"]["set_requested"] == 2
    assert report["slow"]["reset_requested"] == 2
    assert report["ttv2"]["threshold_crossing_observations"] == 4
    assert report["ttv2"]["dispatched_pulses"] == 4
    assert report["ttv2"]["cap_deferred_crossing_observations"] == 0
    assert all(report["ttv2"]["invariants"].values())


def test_ttv2_aihwkit_engine_matches_native_buffer_reference_equation() -> None:
    binding, recovery = _recovery(
        "ttv2_aihwkit_1p1_minibatch_equation",
        total_steps=1,
        fast_backend="ideal",
        shape=(4, 2),
    )
    assert recovery.ideal_fast is not None
    assert recovery.ttv2_hidden is not None
    lambda_h = recovery.ttv2_transfer_scale[0]
    initial_h = torch.tensor([0.8, 0.79, -0.8, 0.2])
    read_a = torch.tensor([0.4, 0.2, -0.3, -0.1]) / lambda_h
    recovery.ttv2_hidden[0][[0, 2], :] = initial_h.reshape(2, 2)
    recovery.ideal_fast[0][[0, 2], :] = read_a.reshape(2, 2)
    binding.state.grad = torch.zeros_like(binding.state)

    expected = buffered_transfer_update(
        initial_h,
        read_a,
        transfer_learning_rate=lambda_h,
        threshold=1.0,
        desired_bl=1,
        momentum=0.0,
        forget_buffer=True,
    )
    recovery.step()

    assert torch.equal(
        recovery.ttv2_hidden[0][[0, 2], :].reshape(-1),
        expected.hidden,
    )
    assert torch.equal(expected.pulse_count, torch.tensor([1, 0, -1, 0]))
    report = recovery.report()
    assert report["slow"]["set_requested"] == 1
    assert report["slow"]["reset_requested"] == 1
    assert all(report["ttv2"]["invariants"].values())


def test_ttv2_aihwkit_uses_fair_per_parameter_caps_for_non_square_w2() -> None:
    binding_w1 = _binding((4, 6), key="base.dense_weight.0")
    binding_w2 = _binding((6, 4), key="base.dense_weight.1")
    recovery = _aihwkit_multi_recovery(
        (binding_w1, binding_w2),
        total_steps=2,
        learning_rates=(1.0, 0.25),
        slow_pulse_budget_per_cell=0.25,
    )
    assert recovery.ttv2_hidden is not None
    recovery.ttv2_hidden[0][[0, 2], :] = 1.2
    recovery.ttv2_hidden[1][[0, 3], :] = -1.2
    binding_w1.state.grad = torch.zeros_like(binding_w1.state)
    binding_w2.state.grad = torch.zeros_like(binding_w2.state)

    recovery.step()

    report = recovery.report()
    assert report["slow_by_parameter"][binding_w1.key]["total_requested"] == 6
    assert report["slow_by_parameter"][binding_w2.key]["total_requested"] == 6
    assert report["transfer_count_by_parameter"] == {
        binding_w1.key: 1,
        binding_w2.key: 1,
    }
    assert report["transfer_opportunities"] == 20
    for key in (binding_w1.key, binding_w2.key):
        cap = report["ttv2"]["cap_by_parameter"][key]
        assert cap["pulse_cap"] == 6
        assert cap["requested"] == 6
        assert cap["exhausted"] is True
    assert report["ttv2"]["threshold_crossing_observations"] == 20
    assert report["ttv2"]["dispatched_pulses"] == 12
    assert report["ttv2"]["cap_deferred_crossing_observations"] == 8
    assert all(report["ttv2"]["invariants"].values())


def test_ttv2_aihwkit_two_layer_partial_cap_stability_smoke() -> None:
    binding_w1 = _binding((6, 10), key="base.dense_weight.0")
    binding_w2 = _binding((4, 10), key="base.dense_weight.1")
    recovery_a = _aihwkit_multi_recovery(
        (binding_w1, binding_w2),
        total_steps=4,
        learning_rates=(1.0, 0.25),
        slow_pulse_budget_per_cell=0.1,
    )
    assert recovery_a.ideal_fast is not None
    assert recovery_a.ttv2_hidden is not None

    alternating = torch.tensor([1.2, -1.2] * 5)
    recovery_a.ttv2_hidden[0][0, :] = alternating
    recovery_a.ttv2_hidden[0][3, :] = -alternating
    recovery_a.ttv2_hidden[1][0, :] = alternating
    recovery_a.ttv2_hidden[1][2, :] = -alternating
    binding_w1.state.grad = torch.zeros_like(binding_w1.state)
    binding_w2.state.grad = torch.zeros_like(binding_w2.state)
    recovery_a.step()

    first_report = recovery_a.report()
    assert first_report["slow_by_parameter"][binding_w1.key]["total_requested"] == 6
    assert first_report["slow_by_parameter"][binding_w2.key]["total_requested"] == 4
    assert first_report["ttv2"]["cap_by_parameter"][binding_w1.key]["pulse_cap"] == 6
    assert first_report["ttv2"]["cap_by_parameter"][binding_w2.key]["pulse_cap"] == 4
    assert first_report["ttv2"]["threshold_crossing_observations"] == 40
    assert first_report["ttv2"]["dispatched_pulses"] == 10
    assert first_report["ttv2"]["cap_deferred_crossing_observations"] == 30
    assert first_report["ttv2"]["cap_by_parameter"][binding_w1.key][
        "deferred_crossing_observations"
    ] == 14
    assert first_report["ttv2"]["cap_by_parameter"][binding_w2.key][
        "deferred_crossing_observations"
    ] == 16
    assert first_report["ttv2"]["cursor_by_parameter"][binding_w1.key][
        "next_logical_input"
    ] == 1
    assert first_report["ttv2"]["cursor_by_parameter"][binding_w2.key][
        "next_logical_input"
    ] == 1
    assert all(first_report["ttv2"]["invariants"].values())

    scanned_w1 = recovery_a.ttv2_hidden[0][[0, 3], :]
    scanned_w2 = recovery_a.ttv2_hidden[1][[0, 2], :]
    assert int((scanned_w1 != 0.0).sum().item()) == 14
    assert int((scanned_w2 != 0.0).sum().item()) == 16
    assert torch.all(scanned_w1[scanned_w1 != 0.0].abs() == 1.2)
    assert torch.all(scanned_w2[scanned_w2 != 0.0].abs() == 1.2)

    saved = deepcopy(recovery_a.state_dict())
    binding_w1_b = _binding((6, 10), key="base.dense_weight.0")
    binding_w2_b = _binding((4, 10), key="base.dense_weight.1")
    recovery_b = _aihwkit_multi_recovery(
        (binding_w1_b, binding_w2_b),
        total_steps=4,
        learning_rates=(1.0, 0.25),
        slow_pulse_budget_per_cell=0.1,
    )
    recovery_b.load_state_dict(saved)
    frozen_slow = recovery_a.current_persistent_endpoint().clone()

    for _ in range(3):
        binding_w1.state.grad = torch.ones_like(binding_w1.state)
        binding_w2.state.grad = torch.ones_like(binding_w2.state)
        binding_w1_b.state.grad = torch.ones_like(binding_w1_b.state)
        binding_w2_b.state.grad = torch.ones_like(binding_w2_b.state)
        recovery_a.step()
        recovery_b.step()

    assert torch.equal(recovery_a.current_persistent_endpoint(), frozen_slow)
    assert recovery_a.report() == recovery_b.report()
    _assert_nested_equal(recovery_a.state_dict(), recovery_b.state_dict())
    terminal = recovery_a.report()["ttv2"]
    for key in (binding_w1.key, binding_w2.key):
        assert terminal["cap_by_parameter"][key]["subsequent_steps_skipped"] == 3
        assert terminal["cursor_by_parameter"][key]["executed_scans"] == 1
    assert all(terminal["invariants"].values())
    assert all(
        bool(torch.all(torch.isfinite(value)))
        for value in (*recovery_a.ideal_fast, *recovery_a.ttv2_hidden)
    )


def test_ttv2_aihwkit_freezes_all_parameter_state_after_cap() -> None:
    binding, recovery = _recovery(
        "ttv2_aihwkit_1p1_minibatch_equation",
        total_steps=2,
        fast_backend="ideal",
        shape=(4, 2),
        slow_pulse_budget_per_cell=0.5,
    )
    assert recovery.ideal_fast is not None
    assert recovery.ttv2_hidden is not None
    recovery.ttv2_hidden[0][[0, 2], :] = 1.2
    binding.state.grad = torch.zeros_like(binding.state)
    recovery.step()
    frozen_fast = recovery.ideal_fast[0].clone()
    frozen_hidden = recovery.ttv2_hidden[0].clone()
    frozen_transfer_count = list(recovery.transfer_count)
    frozen_opportunities = recovery.transfer_opportunities

    binding.state.grad = torch.ones_like(binding.state)
    recovery.step()

    assert torch.equal(recovery.ideal_fast[0], frozen_fast)
    assert torch.equal(recovery.ttv2_hidden[0], frozen_hidden)
    assert recovery.transfer_count == frozen_transfer_count
    assert recovery.transfer_opportunities == frozen_opportunities
    cap = recovery.report()["ttv2"]["cap_by_parameter"][binding.key]
    assert cap["pulse_cap"] == 4
    assert cap["requested"] == 4
    assert cap["cap_reached_after_step"] == 1
    assert cap["subsequent_steps_skipped"] == 1


def test_ttv2_aihwkit_resume_is_bit_exact_for_a_h_caps_and_cursor() -> None:
    binding_a, recovery_a = _recovery(
        "ttv2_aihwkit_1p1_minibatch_equation",
        total_steps=3,
        fast_backend="ideal",
        shape=(4, 2),
    )
    assert recovery_a.ttv2_hidden is not None
    recovery_a.ttv2_hidden[0][[0, 2], :] = torch.tensor(
        [[1.2, -1.2], [1.2, -1.2]]
    )
    binding_a.state.grad = torch.full_like(binding_a.state, 0.03)
    recovery_a.step()
    saved = deepcopy(recovery_a.state_dict())

    binding_b, recovery_b = _recovery(
        "ttv2_aihwkit_1p1_minibatch_equation",
        total_steps=3,
        fast_backend="ideal",
        shape=(4, 2),
    )
    recovery_b.load_state_dict(saved)

    for gradient in (0.02, -0.01):
        binding_a.state.grad = torch.full_like(binding_a.state, gradient)
        binding_b.state.grad = torch.full_like(binding_b.state, gradient)
        recovery_a.step()
        recovery_b.step()

    assert recovery_a.report() == recovery_b.report()
    _assert_nested_equal(recovery_a.state_dict(), recovery_b.state_dict())


def test_direct_recovery_resume_is_bit_exact() -> None:
    binding_a, recovery_a = _recovery("direct_pulse", total_steps=2)
    binding_a.state.grad = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    recovery_a.set_direct_probability_scale(1e6, report={"fixture": True})
    recovery_a.step()
    saved = deepcopy(recovery_a.state_dict())

    binding_b, recovery_b = _recovery("direct_pulse", total_steps=2)
    recovery_b.load_state_dict(saved)
    assert torch.equal(
        recovery_a.current_apparent_endpoint(clipped=False),
        recovery_b.current_apparent_endpoint(clipped=False),
    )
    assert torch.equal(
        recovery_a.current_persistent_endpoint(),
        recovery_b.current_persistent_endpoint(),
    )

    for binding, recovery in (
        (binding_a, recovery_a),
        (binding_b, recovery_b),
    ):
        binding.state.grad = torch.tensor([[0.0, -1.0], [0.0, 1.0]])
        recovery.step()
    assert recovery_a.report() == recovery_b.report()
    for key, value in recovery_a.state_dict().items():
        counterpart = recovery_b.state_dict()[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, counterpart), key


def test_physical_fast_tiki_taka_commissions_paired_rails_and_records_cost() -> None:
    binding, recovery = _recovery(
        "tiki_taka",
        total_steps=2,
        fast_backend="physical_om",
    )
    assert recovery.fast_commissioning is not None
    assert recovery.fast_commissioning["hidden_device_bounds_consumed"] is False
    baseline = recovery.fast_commissioning["baseline"]
    assert torch.equal(baseline[:4], baseline[4:])

    binding.state.grad = -torch.ones_like(binding.state)
    recovery.step()
    report = recovery.report()
    assert report["fast"] is not None
    assert report["fast"]["set_requested"] > 0
    assert report["transfer_count_by_parameter"][binding.key] == 1


def test_recovery_rejects_a_checkpoint_for_another_deployment() -> None:
    _binding_value, recovery = _recovery("rail_refresh", total_steps=4)
    state = recovery.state_dict()
    state["source_deployment_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="exact protocol and deployment"):
        recovery.load_state_dict(state)


def test_recovery_checkpoint_detaches_nested_read_only_config() -> None:
    binding = _binding()
    generator = torch.Generator().manual_seed(44)
    parameters = _parameters("rail_refresh")
    parameters["source_dual_rail_layout_by_parameter"] = MappingProxyType(
        parameters["source_dual_rail_layout_by_parameter"]
    )
    recovery = IbmOmDeployedRecovery(
        (binding,),
        slow_population=_population(binding),
        source_raw_apparent=torch.full((4,), 0.5),
        source_apparent=torch.full((4,), 0.5),
        source_persistent=torch.full((4,), 0.5),
        requested_target=torch.full((4,), 0.5),
        reset_baseline=torch.full((4,), 0.5),
        generator_state_after_programming=generator.get_state(),
        parameters=MappingProxyType(parameters),
        learning_rates=(1.0,),
        conductance_min=0.1,
        conductance_max=1.0,
        total_steps=4,
        source_deployment_sha256="d" * 64,
        device_model_path=_DEVICE_MODEL,
    )

    state = deepcopy(recovery.state_dict())
    assert state["parameters"]["source_dual_rail_layout_by_parameter"] == {
        "base.dense_weight.0": "halves"
    }
