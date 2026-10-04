from __future__ import annotations

from dataclasses import make_dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_analog_relu import crossbar_common as runtime
from experiments.mnist_relu.model import BiasFreeReluTeacher
from training.ibm_om_standard_crossbar import build_crossbar_layout
from training.ibm_reram_endpoint_model import (
    build_ibm_reram_accepted_endpoint_model,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


def test_staged_settings_execute_a_real_hwa_minibatch():
    from experiments.mnist_analog_relu.staged_config import parse_staged_crossbar_config
    from experiments.mnist_analog_relu.staged_runtime import _offchip_adaptation_spec

    path = Path(__file__).resolve().parents[1] / (
        "examples/mnist_analog_relu/ibm_om_crossbar_staged_v2/offchip_hwa.json"
    )
    stage = replace(
        parse_staged_crossbar_config(json.loads(path.read_text())).stage,
        epochs=1, maximum_batches=1,
    )
    spec = _offchip_adaptation_spec(stage, evaluate_test=False)
    layout = build_crossbar_layout((4, 3, 2), maximum_input_size=2)
    size = sum(tile.cells for tile in layout)
    population = _population(size, layout)
    source = torch.linspace(-0.2, 0.2, size)
    torch.manual_seed(23)
    teacher = BiasFreeReluTeacher(device=torch.device("cpu"), dims=(4, 3, 2))
    loader = [(torch.rand(2, 4), torch.tensor([0, 1]))]

    def run():
        return runtime._offchip_adapt(
            source_requested=source, population=population,
            codebook_values=torch.stack((-torch.ones(size), torch.ones(size))),
            spec=spec, layout=layout, digital_scales=(1.0, 1.0), teacher=teacher,
            validation_loader=loader, train_loader=loader, device=torch.device("cpu"),
        )

    deployment, report, state = run()
    repeated, repeated_report, repeated_state = run()
    assert report["optimizer_steps"] == 1
    assert report["primary_evaluation_state"] == "held_apparent_q"
    assert report["programming_error"] is None
    assert torch.isfinite(deployment).all()
    assert not torch.equal(state["fixed_final_master_q"], source)
    assert torch.equal(repeated, deployment)
    assert repeated_report == report
    assert runtime._state_tree_equal(repeated_state, state)


def _settings(kind="stochastic"):
    """Fixed numerical fixtures for the shared adaptation algorithm."""
    def freeze(value):
        if isinstance(value, dict):
            cls = make_dataclass("Settings", [(name, object) for name in value], frozen=True)
            return cls(**{name: freeze(item) for name, item in value.items()})
        return value
    settings = {'stochastic': {'offchip': {'policy': 'stochastic_apparent_hwa',
                                'training_protocol': 'legacy_five_epoch_fixed_final',
                                'epochs': 5,
                                'logical_learning_rates': (0.0001, 0.0001),
                                'beta_1': 0.9,
                                'beta_2': 0.999,
                                'epsilon': 1e-08,
                                'objective': 'teacher_kl',
                                'master_q_bounds': (-1.0, 1.0),
                                'maximum_batches': 256,
                                'checkpoint_policy': 'fixed_final_epoch_no_selection',
                                'deployment_target': 'policy_realized_state',
                                'epoch_evaluation': 'validation_only',
                                'forward_noise': {'model': 'aihwkit_1_1_0_softbounds_reference_additive_write_noise',
                                                  'base_state': 'support_clamped_persistent_q',
                                                  'equation': 'q_apparent_equals_q_persistent_plus_write_noise_std_times_nominal_dw_min_times_standard_normal',
                                                  'resampling': 'independent_full_array_draw_per_minibatch',
                                                  'samples_per_minibatch': 1,
                                                  'relative_scale': 1.0,
                                                  'seed': 88042,
                                                  'gradient_estimator': 'identity_straight_through'},
                                'programming_error': None},
                    'evaluation': {'evaluate_test': True,
                                   'sample_limit': 1000,
                                   'maximum_validation_batches': 2,
                                   'selection_metric': 'fixed_final_epoch_no_selection'}},
     'population': {'offchip': {'policy': 'population_programming_error_hwa',
                                'training_protocol': 'full_mnist_ten_epoch_fixed_final',
                                'epochs': 10,
                                'logical_learning_rates': (0.0001, 0.0001),
                                'beta_1': 0.9,
                                'beta_2': 0.999,
                                'epsilon': 1e-08,
                                'objective': 'teacher_kl',
                                'master_q_bounds': (-1.0, 1.0),
                                'maximum_batches': 3438,
                                'checkpoint_policy': 'fixed_final_epoch_no_selection',
                                'deployment_target': 'fixed_final_master_fault_blind_pv',
                                'epoch_evaluation': 'validation_and_test',
                                'forward_noise': None,
                                'programming_error': {'model': 'ibm_reram_om_pv128_healthy_accepted_endpoint_residual_v1',
                                                      'artifact_path': 'data/ibm_reram_om_pv128_hwa_v1.json',
                                                      'artifact_sha256': '3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3',
                                                      'condition_key': 'adaptive__lower_to_target__tau_step_0.5',
                                                      'controller': 'adaptive',
                                                      'start_protocol': 'lower_to_target',
                                                      'maximum_programming_pulses': 128,
                                                      'verify_tolerance_x': 0.023725,
                                                      'target_coordinate': 'global_x_equals_q_plus_one_over_two',
                                                      'training_population': 'healthy_noncorrupt_accepted_endpoints_no_fixed_array',
                                                      'endpoint_state': 'apparent_q',
                                                      'resampling': 'independent_full_array_draw_per_minibatch',
                                                      'samples_per_minibatch': 1,
                                                      'strength_schedule': 'linear_epoch_ramp',
                                                      'initial_strength': 0.0,
                                                      'final_strength': 1.0,
                                                      'ramp_epochs': 10,
                                                      'seed': 88042,
                                                      'gradient_estimator': 'identity_straight_through'}},
                    'evaluation': {'evaluate_test': True,
                                   'sample_limit': 1000,
                                   'maximum_validation_batches': 63,
                                   'selection_metric': 'fixed_final_epoch_no_selection'}}}
    return freeze(settings[kind])


def _population(
    size: int,
    layout,
    *,
    assignment_seed: int = 87004,
    support: float = 0.25,
) -> IbmReramArrayPopulation:
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(tile.key for tile in layout),
        binding_shapes=tuple(tile.shape for tile in layout),
        binding_sampling_seeds=(11, 12, 13),
        donor_sampling_seeds=(21, 22, 23),
        nominal_dw_min=0.0949,
        dw_min_std=0.009,
        write_noise_std=1.4113,
        max_bound=torch.full((size,), support),
        min_bound=torch.full((size,), -support),
        dwmin_up=torch.full((size,), 0.0949),
        dwmin_down=torch.full((size,), 0.0949),
        reference=torch.zeros(size),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint=(
            f"hwa-noise-runtime-fixture-{assignment_seed}-{support}"
        ),
        aihwkit_version="1.1.0",
    )


def _accepted_endpoint_artifact() -> dict[str, object]:
    records = []
    for target, probabilities in (
        (0.0, [0.8, 0.2]),
        (1.0, [0.2, 0.8]),
    ):
        records.append(
            {
                "target": target,
                "accepted_noncorrupt_residual": {
                    "fit_count": 100,
                    "bin_edges": [-0.02, 0.0, 0.02],
                    "bin_probabilities": probabilities,
                },
            }
        )
    return {
        "schema": "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model",
        "schema_version": 2,
        "metadata": {
            "preset": "reram_array_om",
            "execution_profile": "hwa_production_cap128",
            "enable_published_corruption": False,
        },
        "conditions": {
            "adaptive__lower_to_target__tau_step_0.5": {
                "fit_status": "fit",
                "reachability_fit_status": "fit",
                "adequate": True,
                "validation": {"per_target": records},
            }
        },
    }




def test_aihwkit_om_apparent_write_noise_is_replayable_and_unclamped() -> None:
    persistent = torch.ones(1_000, dtype=torch.float32)
    first_generator = torch.Generator(device="cpu").manual_seed(123)
    second_generator = torch.Generator(device="cpu").manual_seed(123)

    first, first_noise = runtime._sample_aihwkit_om_apparent_write_noise(
        persistent_q=persistent,
        nominal_dw_min=0.0949,
        write_noise_std=1.4113,
        relative_scale=1.0,
        generator=first_generator,
    )
    second, second_noise = runtime._sample_aihwkit_om_apparent_write_noise(
        persistent_q=persistent,
        nominal_dw_min=0.0949,
        write_noise_std=1.4113,
        relative_scale=1.0,
        generator=second_generator,
    )

    assert torch.equal(first, second)
    assert torch.equal(first_noise, second_noise)
    assert torch.allclose(first - persistent, first_noise, atol=1e-7, rtol=0.0)
    assert bool(torch.any(first > 1.0))


def test_aihwkit_om_apparent_write_noise_has_preset_sigma() -> None:
    persistent = torch.zeros(250_000, dtype=torch.float32)
    apparent, noise = runtime._sample_aihwkit_om_apparent_write_noise(
        persistent_q=persistent,
        nominal_dw_min=0.0949,
        write_noise_std=1.4113,
        relative_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(456),
    )

    expected_sigma = 1.4113 * 0.0949
    assert torch.equal(apparent, noise)
    assert float(noise.mean()) == pytest.approx(0.0, abs=1e-3)
    assert float(noise.std(unbiased=False)) == pytest.approx(
        expected_sigma, rel=0.01
    )


def test_stochastic_hwa_replays_noise_and_deploys_persistent_support_state() -> None:
    spec = _settings()
    spec = replace(
        spec,
        offchip=replace(spec.offchip, maximum_batches=1),
        evaluation=replace(
            spec.evaluation,
            sample_limit=2,
            maximum_validation_batches=1,
        ),
    )
    layout = build_crossbar_layout((784, 50, 10), maximum_input_size=512)
    size = sum(tile.cells for tile in layout)
    population = _population(size, layout)
    source = torch.linspace(-0.5, 0.5, size, dtype=torch.float32)
    codebook_values = torch.stack((-torch.ones(size), torch.ones(size)))
    torch.manual_seed(9)
    teacher = BiasFreeReluTeacher(device=torch.device("cpu"))
    inputs = torch.rand(2, 784, generator=torch.Generator().manual_seed(10))
    labels = torch.tensor([1, 2], dtype=torch.int64)
    loader = [(inputs, labels)]

    def run(settings=spec.offchip):
        return runtime._offchip_adapt(
            source_requested=source,
            population=population,
            codebook_values=codebook_values,
            spec=replace(spec, offchip=settings),
            layout=layout,
            digital_scales=(1.0, 1.0),
            teacher=teacher,
            validation_loader=loader,
            train_loader=loader,
            device=torch.device("cpu"),
        )

    first_requested, first_report, first_state = run()
    second_requested, second_report, second_state = run()

    assert torch.equal(first_requested, second_requested)
    assert torch.equal(
        first_state["fixed_final_master_q"],
        second_state["fixed_final_master_q"],
    )
    assert torch.equal(
        first_state["forward_noise_generator_final_state"],
        second_state["forward_noise_generator_final_state"],
    )
    assert first_report["epochs"] == second_report["epochs"]
    assert first_report["stochastic_programming_during_training"] is False
    assert first_report["stochastic_apparent_forward_noise_during_training"] is True
    assert first_report["persistent_state_updates_during_training"] is False
    assert first_report["forward_noise"]["sigma_q"] == pytest.approx(
        1.4113 * 0.0949
    )
    assert first_report["forward_noise"]["native_rng_stream_parity"] is False
    assert (
        first_report["forward_noise"]["program_verify_conditioned_endpoint_sampling"]
        is False
    )
    for epoch in first_report["epochs"]:
        assert epoch["forward_noise"]["full_array_draws"] == 1
        assert epoch["forward_noise"]["scalar_values"] == size
        assert epoch["forward_noise"]["observed_std_q"] == pytest.approx(
            1.4113 * 0.0949, rel=0.02
        )
    expected_deployment = torch.maximum(
        torch.minimum(first_state["fixed_final_master_q"], population.logical_max),
        population.logical_min,
    )
    assert torch.equal(first_requested, expected_deployment)

    exact_requested, exact_report, exact_state = run(
        replace(
            spec.offchip,
            deployment_target="fixed_final_master_fault_blind_pv",
        )
    )
    assert torch.equal(exact_requested, exact_state["fixed_final_master_q"])
    assert torch.equal(
        exact_state["fixed_final_deployment_q"],
        exact_state["fixed_final_master_q"],
    )
    assert not torch.equal(
        exact_requested,
        exact_state["fixed_final_realized_q"],
    )
    assert exact_report["deployment_target"] == (
        "fixed_final_master_fault_blind_pv"
    )

    changed_noise = replace(spec.offchip.forward_noise, seed=88043)
    changed_requested, changed_report, _changed_state = run(
        replace(spec.offchip, forward_noise=changed_noise)
    )
    assert not torch.equal(first_requested, changed_requested)
    assert (
        first_report["epochs"][0]["forward_noise"][
            "noise_tensor_sequence_sha256"
        ]
        != changed_report["epochs"][0]["forward_noise"][
            "noise_tensor_sequence_sha256"
        ]
    )




def test_population_hwa_is_replayable_and_independent_of_fixed_array_bounds() -> None:
    spec = _settings("population")
    spec = replace(
        spec,
        offchip=replace(spec.offchip, maximum_batches=1),
        evaluation=replace(
            spec.evaluation,
            sample_limit=2,
            maximum_validation_batches=1,
        ),
    )
    model = build_ibm_reram_accepted_endpoint_model(
        _accepted_endpoint_artifact(),
        condition_key="adaptive__lower_to_target__tau_step_0.5",
    )
    layout = build_crossbar_layout((784, 50, 10), maximum_input_size=512)
    size = sum(tile.cells for tile in layout)
    narrow_population = _population(
        size,
        layout,
        assignment_seed=87004,
        support=0.25,
    )
    wide_population = _population(
        size,
        layout,
        assignment_seed=99991,
        support=0.9,
    )
    source = torch.linspace(-0.5, 0.5, size, dtype=torch.float32)
    codebook_values = torch.stack((-torch.ones(size), torch.ones(size)))
    torch.manual_seed(19)
    teacher = BiasFreeReluTeacher(device=torch.device("cpu"))
    inputs = torch.rand(2, 784, generator=torch.Generator().manual_seed(20))
    labels = torch.tensor([1, 2], dtype=torch.int64)
    loader = [(inputs, labels)]

    def run(population: IbmReramArrayPopulation):
        return runtime._offchip_adapt(
            source_requested=source,
            population=population,
            codebook_values=codebook_values,
            spec=spec,
            layout=layout,
            digital_scales=(1.0, 1.0),
            teacher=teacher,
            validation_loader=loader,
            train_loader=loader,
            device=torch.device("cpu"),
            test_loader=loader,
            programming_error_model=model,
        )

    narrow_requested, narrow_report, narrow_state = run(narrow_population)
    wide_requested, wide_report, wide_state = run(wide_population)

    assert narrow_state["schema_version"] == 3
    assert torch.equal(narrow_state["initial_master_q"], source)
    assert torch.equal(narrow_state["fixed_final_master_q"], wide_state["fixed_final_master_q"])
    assert torch.equal(narrow_requested, narrow_state["fixed_final_master_q"])
    assert torch.equal(narrow_requested, wide_requested)
    assert runtime._state_tree_equal(
        narrow_state["optimizer_state_dict"],
        wide_state["optimizer_state_dict"],
    )
    assert torch.equal(
        narrow_state["programming_error_generator_final_state"],
        wide_state["programming_error_generator_final_state"],
    )
    assert not torch.equal(
        source,
        torch.maximum(
            torch.minimum(source, narrow_population.logical_max),
            narrow_population.logical_min,
        ),
    )
    assert narrow_report["initial_realized_sha256"] == wide_report[
        "initial_realized_sha256"
    ]
    assert narrow_report["stochastic_programming_during_training"] is False
    assert (
        narrow_report["stochastic_programming_error_model_during_training"]
        is True
    )
    assert narrow_report["stochastic_apparent_forward_noise_during_training"] is False
    programming = narrow_report["programming_error"]
    assert programming["fixed_array_identity_during_training"] is False
    assert programming["fixed_per_cell_bounds_during_training"] is False
    assert programming["corrupt_device_mask_during_training"] is False
    assert programming["accepted_noncorrupt_residuals_only"] is True
    assert [
        epoch["programming_error"]["strength"]
        for epoch in narrow_report["epochs"]
    ] == pytest.approx([index / 10.0 for index in range(1, 11)])
    for narrow_epoch, wide_epoch in zip(
        narrow_report["epochs"],
        wide_report["epochs"],
        strict=True,
    ):
        narrow_error = narrow_epoch["programming_error"]
        wide_error = wide_epoch["programming_error"]
        assert narrow_error["full_array_draws"] == 1
        assert narrow_error["scalar_values"] == size
        assert (
            narrow_error["unscaled_residual_tensor_sequence_sha256"]
            == wide_error["unscaled_residual_tensor_sequence_sha256"]
        )


def test_held_apparent_hwa_validation_is_replayable_and_stream_independent() -> None:
    spec = _settings()
    layout = build_crossbar_layout((784, 50, 10), maximum_input_size=512)
    size = sum(tile.cells for tile in layout)
    population = _population(size, layout)
    source = torch.linspace(-0.5, 0.5, size, dtype=torch.float32)
    codebook_values = torch.stack((-torch.ones(size), torch.ones(size)))
    torch.manual_seed(91)
    teacher = BiasFreeReluTeacher(device=torch.device("cpu"))
    inputs = torch.rand(2, 784, generator=torch.Generator().manual_seed(92))
    loader = [(inputs, torch.tensor([1, 2], dtype=torch.int64))]

    def run(*, include_test: bool):
        settings = SimpleNamespace(
            **{
                **vars(spec.offchip),
                "epochs": 1,
                "maximum_batches": 1,
                "epoch_evaluation": (
                    "validation_and_test" if include_test else "validation_only"
                ),
                "evaluation_forward_policy": (
                    "one_sampled_held_apparent_q_per_evaluation"
                ),
            }
        )
        compatibility = SimpleNamespace(
            offchip=settings,
            evaluation=SimpleNamespace(
                maximum_validation_batches=1,
                sample_limit=2,
            ),
        )
        return runtime._offchip_adapt(
            source_requested=source,
            population=population,
            codebook_values=codebook_values,
            spec=compatibility,
            layout=layout,
            digital_scales=(1.0, 1.0),
            teacher=teacher,
            validation_loader=loader,
            train_loader=loader,
            device=torch.device("cpu"),
            test_loader=loader if include_test else None,
        )

    _no_test_q, no_test_report, no_test_state = run(include_test=False)
    _with_test_q, with_test_report, with_test_state = run(include_test=True)

    assert torch.equal(
        no_test_state["forward_noise_generator_final_state"],
        with_test_state["forward_noise_generator_final_state"],
    )
    assert no_test_report["initial_validation"] == with_test_report["initial_validation"]
    assert no_test_report["epochs"][0]["validation"] == (
        with_test_report["epochs"][0]["validation"]
    )
    evaluation = no_test_report["fixed_final_validation"]
    assert evaluation["network_forward_state"] == "held_apparent_q"
    assert evaluation["primary_state"] == "held_apparent_q"
    assert evaluation["persistent_device_state_present"] is False
    assert evaluation["diagnostic_state_role"] == (
        "nonpersistent_support_clamped_digital_master_q"
    )
    receipt = evaluation["held_apparent_state_receipt"]
    assert receipt["evaluation_id"] == "epoch_001.validation"
    assert receipt["support_clamped_digital_master_q_sha256"] == (
        no_test_report["fixed_final_realized_sha256"]
    )
    assert with_test_report["fixed_final_test"]["held_apparent_state_receipt"][
        "evaluation_id"
    ] == "epoch_001.test"
