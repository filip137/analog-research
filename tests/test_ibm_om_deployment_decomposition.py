from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

import ebl.runtime.mnist_relu_drn.ibm_om_deployment_decomposition as decomposition
from networks.resistive.builders import ParameterBinding
from networks.variable.parameter import DenseWeight
from methods.ibm_om.api import (
    IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
    IbmReramArrayPopulation,
    IbmReramHwaConfig,
    _population_fingerprint,
    commission_ibm_reram_reset_relative_baselines,
    map_ibm_reram_array_targets,
)


def _dense_binding(
    key: str,
    shape: tuple[int, int],
    *,
    role: str = "dense_weight",
) -> ParameterBinding:
    parameter = DenseWeight(
        (shape[0],),
        (shape[1],),
        1.0,
        "cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=1.0,
    )
    return ParameterBinding(key, parameter, role=role)


def test_logical_contrast_matches_quad_and_differential_definitions() -> None:
    shapes = ((4, 4), (4, 4))
    single_slices = (slice(0, 16), slice(16, 32))
    single_layers = decomposition.canonical_layer_geometries(
        ("base.dense_weight.0", "base.dense_weight.1"),
        shapes,
        single_slices,
        encoding="single",
        dual_rail_layout_by_parameter={
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        },
    )
    first = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16.0
    second = torch.flip(first, dims=(0,))
    single = torch.cat((first.reshape(-1), second.reshape(-1)))

    expected_first = (
        first[:2, :2] - first[:2, 2:] - first[2:, :2] + first[2:, 2:]
    ).to(torch.float64)
    expected_second = (
        second[:2][:, [0, 2]]
        - second[:2][:, [1, 3]]
        - second[2:][:, [0, 2]]
        + second[2:][:, [1, 3]]
    ).to(torch.float64)
    torch.testing.assert_close(
        decomposition.logical_differential_contrast(
            single,
            single_layers[0],
            conductance_min=0.0,
            conductance_max=1.0,
        ),
        expected_first,
    )
    torch.testing.assert_close(
        decomposition.logical_differential_contrast(
            single,
            single_layers[1],
            conductance_min=0.0,
            conductance_max=1.0,
        ),
        expected_second,
    )

    differential_layers = decomposition.canonical_layer_geometries(
        (
            "base.conductance_plus.0",
            "base.conductance_minus.0",
            "base.conductance_plus.1",
            "base.conductance_minus.1",
        ),
        shapes * 2,
        (
            slice(0, 16),
            slice(16, 32),
            slice(32, 48),
            slice(48, 64),
        ),
        encoding="differential",
        dual_rail_layout_by_parameter=None,
    )
    differential = torch.cat(
        (
            first.reshape(-1),
            torch.zeros(16),
            second.reshape(-1),
            torch.zeros(16),
        )
    )
    torch.testing.assert_close(
        decomposition.logical_differential_contrast(
            differential,
            differential_layers[0],
            conductance_min=0.0,
            conductance_max=1.0,
        ),
        expected_first,
    )
    torch.testing.assert_close(
        decomposition.logical_differential_contrast(
            differential,
            differential_layers[1],
            conductance_min=0.0,
            conductance_max=1.0,
        ),
        expected_second,
    )


def test_state_partitions_and_reversible_application_restore_exactly() -> None:
    bindings = (
        _dense_binding("base.dense_weight.0", (4, 4)),
        _dense_binding("base.dense_weight.1", (4, 4)),
    )
    keys, shapes, sections = decomposition._binding_layout(bindings)
    layers = decomposition.canonical_layer_geometries(
        keys,
        shapes,
        sections,
        encoding="single",
        dual_rail_layout_by_parameter={
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        },
    )
    size = 32
    clean = torch.full((size,), 0.4)
    mapped = torch.full((size,), 0.5)
    residual = torch.linspace(-0.1, 0.1, size)
    apparent = mapped + residual
    accepted = torch.arange(size) % 2 == 0
    saturation = torch.zeros(size, dtype=torch.bool)
    saturation[[0, 17]] = True
    states, report = decomposition.build_decomposition_states(
        clean_selected=clean,
        global_requested=clean,
        mapped_target=mapped,
        apparent_endpoint=apparent,
        persistent_endpoint=apparent - 0.01,
        accepted=accepted,
        layers=layers,
        optional_masks={
            "saturation": saturation,
            "clipping": torch.zeros(size, dtype=torch.bool),
            "fallback": None,
        },
    )

    torch.testing.assert_close(
        (states["w1_endpoint_error_only"] - mapped)
        + (states["w2_endpoint_error_only"] - mapped),
        residual,
    )
    torch.testing.assert_close(
        (states["accepted_endpoint_error_only"] - mapped)
        + (states["failure_endpoint_error_only"] - mapped),
        residual,
    )
    corrected = states["apparent_saturation_corrected"]
    torch.testing.assert_close(corrected[saturation], mapped[saturation])
    torch.testing.assert_close(corrected[~saturation], apparent[~saturation])
    assert report["partitions"] == {
        "layer_error_partition_max_abs": pytest.approx(0.0),
        "acceptance_error_partition_max_abs": pytest.approx(0.0),
    }

    originals = tuple(binding.state.clone() for binding in bindings)
    seen = []
    state_contexts = []

    @contextmanager
    def state_context(name: str):
        state_contexts.append(name)
        yield

    outputs = decomposition.evaluate_reversible_states(
        bindings,
        {"mapped": mapped, "apparent": apparent},
        conductance_min=0.0,
        conductance_max=1.0,
        evaluator=lambda: seen.append(
            torch.cat(tuple(binding.state.reshape(-1) for binding in bindings))
            .clone()
        ),
        state_context=state_context,
    )
    assert outputs == {"mapped": None, "apparent": None}
    torch.testing.assert_close(seen[0], mapped)
    torch.testing.assert_close(seen[1], apparent)
    assert state_contexts == ["mapped", "apparent"]
    assert all(
        torch.equal(binding.state, original)
        for binding, original in zip(bindings, originals)
    )

    def fail() -> None:
        raise RuntimeError("fixture evaluation failure")

    with pytest.raises(RuntimeError, match="fixture evaluation failure"):
        decomposition.evaluate_reversible_states(
            bindings,
            {"mapped": mapped},
            conductance_min=0.0,
            conductance_max=1.0,
            evaluator=fail,
        )
    assert all(
        torch.equal(binding.state, original)
        for binding, original in zip(bindings, originals)
    )


def _pair_deployment_fixture() -> tuple[dict, IbmReramHwaConfig, str, str]:
    keys = (
        "base.conductance_plus.0",
        "base.conductance_minus.0",
    )
    shapes = ((1, 2), (1, 2))
    size = 4
    tensors = {
        "max_bound": torch.ones(size, dtype=torch.float32),
        "min_bound": -torch.ones(size, dtype=torch.float32),
        "dwmin_up": torch.full((size,), 0.2, dtype=torch.float32),
        "dwmin_down": torch.full((size,), 0.2, dtype=torch.float32),
        "reference": torch.zeros(size, dtype=torch.float32),
        "corrupt": torch.zeros(size, dtype=torch.bool),
        "published_corrupt": torch.zeros(size, dtype=torch.bool),
    }
    scalar_parameters = {
        "aihwkit_version": "1.1.0",
        "nominal_dw_min": 0.0949,
        "dw_min_std": 0.0,
        "write_noise_std": 0.0,
    }
    config = IbmReramHwaConfig(
        execution="pulse_resolved",
        assignment_seed=84001,
        endpoint_seed=84003,
        corruption_policy="counterfactual_repaired",
        noisy_evaluation=True,
        target_mapping="differential_pair_common_window",
        common_window_margin_fraction=0.25,
    )
    binding_seeds = (11, 12)
    donor_seeds = (21, 22)
    fingerprint = _population_fingerprint(
        assignment_seed=config.assignment_seed,
        corruption_policy=config.corruption_policy,
        keys=keys,
        shapes=shapes,
        binding_sampling_seeds=binding_seeds,
        donor_sampling_seeds=donor_seeds,
        scalar_parameters=scalar_parameters,
        tensors=tensors,
    )
    population = IbmReramArrayPopulation(
        assignment_seed=config.assignment_seed,
        corruption_policy=config.corruption_policy,
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=binding_seeds,
        donor_sampling_seeds=donor_seeds,
        nominal_dw_min=0.0949,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=tensors["max_bound"],
        min_bound=tensors["min_bound"],
        dwmin_up=tensors["dwmin_up"],
        dwmin_down=tensors["dwmin_down"],
        reference=tensors["reference"],
        corrupt=tensors["corrupt"],
        published_corrupt=tensors["published_corrupt"],
        fingerprint=fingerprint,
        aihwkit_version="1.1.0",
    )
    global_target = torch.tensor([0.1, 0.3, 0.7, 0.9], dtype=torch.float32)
    mapped, mapping_report = map_ibm_reram_array_targets(
        global_target,
        population,
        target_mapping=config.target_mapping,
        dual_rail_layout_by_parameter=None,
        common_window_margin_fraction=config.common_window_margin_fraction,
    )
    selected_sha256 = "b" * 64
    device_model_sha256 = "a" * 64
    programming_report = {
        "execution": "pulse_resolved",
        "devices": size,
        "accepted": size,
        "budget_exhausted": 0,
        "corrupt": 0,
        "saturated": 0,
        "endpoint_clipped": 0,
        "pulse_count": {"maximum": 0},
        "population_fingerprint": fingerprint,
        "endpoint_application_policy": IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
        "target_mapping": config.target_mapping,
        "target_mapping_report": mapping_report,
    }
    deployment = {
        "schema": "ebl.ibm_reram.om_pulse_resolved_deployment",
        "schema_version": 1,
        "device_model_sha256": device_model_sha256,
        "population_fingerprint": fingerprint,
        "population_sampling_receipt": None,
        "config": asdict(config),
        "binding_keys": keys,
        "binding_shapes": shapes,
        "binding_sampling_seeds": binding_seeds,
        "donor_sampling_seeds": donor_seeds,
        "population_scalars": {"preset": "reram_array_om", **scalar_parameters},
        "population": tensors,
        "global_requested_target": global_target,
        "requested_target": mapped,
        "raw_apparent_endpoint": mapped.clone(),
        "apparent_endpoint": mapped.clone(),
        "persistent_endpoint": mapped.clone(),
        "accepted": torch.ones(size, dtype=torch.bool),
        "budget_exhausted": torch.zeros(size, dtype=torch.bool),
        "corrupt": torch.zeros(size, dtype=torch.bool),
        "saturated": torch.zeros(size, dtype=torch.bool),
        "set_count": torch.zeros(size, dtype=torch.int64),
        "reset_count": torch.zeros(size, dtype=torch.int64),
        "total_pulses": torch.zeros(size, dtype=torch.int64),
        "verify_count": torch.ones(size, dtype=torch.int64),
        "reversals": torch.zeros(size, dtype=torch.int64),
        "target_mapping_report": mapping_report,
        "endpoint_application_policy": IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
        "report": programming_report,
        "selected_weights_sha256": selected_sha256,
        "selected_epoch": 3,
    }
    return deployment, config, selected_sha256, device_model_sha256


def _reset_relative_deployment_fixture() -> tuple[
    dict,
    IbmReramHwaConfig,
    str,
    str,
]:
    key = "base.dense_weight.0"
    shape = (2, 40)
    size = 80
    lower = torch.full((size,), -0.6, dtype=torch.float32)
    upper = torch.full((size,), 0.8, dtype=torch.float32)
    population_tensors = {
        "max_bound": upper,
        "min_bound": lower,
        "dwmin_up": torch.full((size,), 0.2, dtype=torch.float32),
        "dwmin_down": torch.full((size,), 0.2, dtype=torch.float32),
        "reference": torch.zeros(size, dtype=torch.float32),
        "corrupt": torch.zeros(size, dtype=torch.bool),
        "published_corrupt": torch.zeros(size, dtype=torch.bool),
    }
    scalar_parameters = {
        "aihwkit_version": "1.1.0",
        "nominal_dw_min": 0.0949,
        "dw_min_std": 0.0,
        "write_noise_std": 0.0,
    }
    config = IbmReramHwaConfig(
        execution="pulse_resolved",
        assignment_seed=84001,
        endpoint_seed=84003,
        corruption_policy="counterfactual_repaired",
        noisy_evaluation=True,
        target_mapping="shared_reset_relative_quad",
        dual_rail_layout_by_parameter={key: "halves"},
        reset_relative_mode="quantized_9_level",
        reset_relative_contrast_step=0.095849,
        reset_read_samples=8,
        reset_guard_standard_errors=3.0,
        forward_logit_gain=562.0,
    )
    fingerprint = _population_fingerprint(
        assignment_seed=config.assignment_seed,
        corruption_policy=config.corruption_policy,
        keys=(key,),
        shapes=(shape,),
        binding_sampling_seeds=(11,),
        donor_sampling_seeds=(21,),
        scalar_parameters=scalar_parameters,
        tensors=population_tensors,
    )
    population = IbmReramArrayPopulation(
        assignment_seed=config.assignment_seed,
        corruption_policy=config.corruption_policy,
        binding_keys=(key,),
        binding_shapes=(shape,),
        binding_sampling_seeds=(11,),
        donor_sampling_seeds=(21,),
        nominal_dw_min=0.0949,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=upper,
        min_bound=lower,
        dwmin_up=population_tensors["dwmin_up"],
        dwmin_down=population_tensors["dwmin_down"],
        reference=population_tensors["reference"],
        corrupt=population_tensors["corrupt"],
        published_corrupt=population_tensors["published_corrupt"],
        fingerprint=fingerprint,
        aihwkit_version="1.1.0",
    )
    commissioning = commission_ibm_reram_reset_relative_baselines(
        population,
        dual_rail_layout_by_parameter={key: "halves"},
        read_samples=8,
        guard_standard_errors=3.0,
    )
    global_target = torch.zeros(size, dtype=torch.float32)
    global_target[0] = 0.125
    global_target[60] = 0.125
    mapped, mapping_report = map_ibm_reram_array_targets(
        global_target,
        population,
        target_mapping=config.target_mapping,
        dual_rail_layout_by_parameter={key: "halves"},
        common_window_margin_fraction=0.0,
        reset_commissioning=commissioning,
        reset_relative_mode=config.reset_relative_mode,
        reset_relative_contrast_step=config.reset_relative_contrast_step,
    )
    selected_sha256 = "c" * 64
    device_model_sha256 = "d" * 64
    programming_report = {
        "execution": "pulse_resolved",
        "devices": size,
        "accepted": size,
        "budget_exhausted": 0,
        "corrupt": 0,
        "saturated": 0,
        "endpoint_clipped": 0,
        "pulse_count": {"maximum": 0},
        "population_fingerprint": fingerprint,
        "endpoint_application_policy": IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
        "target_mapping": config.target_mapping,
        "target_mapping_report": mapping_report,
    }
    deployment = {
        "schema": "ebl.ibm_reram.om_pulse_resolved_deployment",
        "schema_version": 1,
        "device_model_sha256": device_model_sha256,
        "population_fingerprint": fingerprint,
        "population_sampling_receipt": None,
        "config": asdict(config),
        "binding_keys": (key,),
        "binding_shapes": (shape,),
        "binding_sampling_seeds": (11,),
        "donor_sampling_seeds": (21,),
        "population_scalars": {"preset": "reram_array_om", **scalar_parameters},
        "population": population_tensors,
        "reset_commissioning": commissioning.bundle(),
        "global_requested_target": global_target,
        "requested_target": mapped,
        "raw_apparent_endpoint": mapped.clone(),
        "apparent_endpoint": mapped.clone(),
        "persistent_endpoint": mapped.clone(),
        "accepted": torch.ones(size, dtype=torch.bool),
        "budget_exhausted": torch.zeros(size, dtype=torch.bool),
        "corrupt": torch.zeros(size, dtype=torch.bool),
        "saturated": torch.zeros(size, dtype=torch.bool),
        "set_count": torch.zeros(size, dtype=torch.int64),
        "reset_count": torch.zeros(size, dtype=torch.int64),
        "total_pulses": torch.zeros(size, dtype=torch.int64),
        "verify_count": torch.ones(size, dtype=torch.int64),
        "reversals": torch.zeros(size, dtype=torch.int64),
        "target_mapping_report": mapping_report,
        "endpoint_application_policy": IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
        "report": programming_report,
        "selected_weights_sha256": selected_sha256,
        "selected_epoch": 2,
    }
    return deployment, config, selected_sha256, device_model_sha256


def test_deployment_contract_replays_population_and_fails_closed() -> None:
    deployment, config, selected_sha256, device_model_sha256 = (
        _pair_deployment_fixture()
    )
    tensors, loaded_config, contract = decomposition.validate_deployment_contract(
        deployment,
        expected_selected_weights_sha256=selected_sha256,
        expected_binding_keys=deployment["binding_keys"],
        expected_binding_shapes=deployment["binding_shapes"],
        configured_modifiers=(asdict(config),),
        checkpoint_metadata={
            "selection_epoch": 3,
            "device_model_sha256": device_model_sha256,
        },
    )
    assert loaded_config == config
    assert contract["population_fingerprint"] == deployment[
        "population_fingerprint"
    ]
    torch.testing.assert_close(
        tensors["requested_target"],
        torch.tensor([0.3, 0.4, 0.6, 0.7]),
    )

    wrong_policy = dict(deployment)
    wrong_policy["endpoint_application_policy"] = "persistent_forward"
    with pytest.raises(ValueError, match="apparent-forward"):
        decomposition.validate_deployment_contract(
            wrong_policy,
            expected_selected_weights_sha256=selected_sha256,
            expected_binding_keys=deployment["binding_keys"],
            expected_binding_shapes=deployment["binding_shapes"],
            configured_modifiers=(asdict(config),),
            checkpoint_metadata={
                "selection_epoch": 3,
                "device_model_sha256": device_model_sha256,
            },
        )

    wrong_shapes = dict(deployment)
    wrong_shapes["binding_shapes"] = ((2, 1), (1, 2))
    with pytest.raises(ValueError, match="catalog and shapes"):
        decomposition.validate_deployment_contract(
            wrong_shapes,
            expected_selected_weights_sha256=selected_sha256,
            expected_binding_keys=deployment["binding_keys"],
            expected_binding_shapes=deployment["binding_shapes"],
            configured_modifiers=(asdict(config),),
            checkpoint_metadata={
                "selection_epoch": 3,
                "device_model_sha256": device_model_sha256,
            },
        )


def test_deployment_contract_replays_observed_reset_relative_commissioning() -> None:
    deployment, config, selected_sha256, device_model_sha256 = (
        _reset_relative_deployment_fixture()
    )
    tensors, loaded_config, contract = decomposition.validate_deployment_contract(
        deployment,
        expected_selected_weights_sha256=selected_sha256,
        expected_binding_keys=deployment["binding_keys"],
        expected_binding_shapes=deployment["binding_shapes"],
        configured_modifiers=(asdict(config),),
        checkpoint_metadata={
            "selection_epoch": 2,
            "device_model_sha256": device_model_sha256,
        },
    )
    assert loaded_config == config
    assert contract["mapping_report"]["reset_relative_signed_levels"] == 9
    torch.testing.assert_close(
        tensors["requested_target"],
        deployment["requested_target"],
    )

    validation_deployment = dict(deployment)
    validation_deployment["source_weights_sha256"] = validation_deployment.pop(
        "selected_weights_sha256"
    )
    validation_deployment.pop("selected_epoch")
    _tensors, _loaded_config, validation_contract = (
        decomposition.validate_deployment_contract(
            validation_deployment,
            expected_selected_weights_sha256=selected_sha256,
            expected_binding_keys=deployment["binding_keys"],
            expected_binding_shapes=deployment["binding_shapes"],
            configured_modifiers=(asdict(config),),
            checkpoint_metadata={
                "selection_epoch": 2,
                "device_model_sha256": device_model_sha256,
            },
        )
    )
    assert validation_contract["selected_epoch"] == 2
    assert validation_contract["weights_provenance_field"] == (
        "source_weights_sha256"
    )

    tampered = dict(deployment)
    commissioning = dict(deployment["reset_commissioning"])
    commissioning["hidden_min_bound"] = torch.zeros(80)
    tampered["reset_commissioning"] = commissioning
    with pytest.raises(ValueError, match="strict observed RESET commissioning"):
        decomposition.validate_deployment_contract(
            tampered,
            expected_selected_weights_sha256=selected_sha256,
            expected_binding_keys=deployment["binding_keys"],
            expected_binding_shapes=deployment["binding_shapes"],
            configured_modifiers=(asdict(config),),
            checkpoint_metadata={
                "selection_epoch": 2,
                "device_model_sha256": device_model_sha256,
            },
        )

def test_cli_requires_all_inputs_writes_once_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        decomposition.main([])

    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"schema": decomposition.SCHEMA, "schema_version": 1}

    monkeypatch.setattr(decomposition, "run_decomposition", fake_run)
    output = tmp_path / "decomposition.json"
    arguments = [
        "--config",
        str(tmp_path / "config.json"),
        "--weights",
        str(tmp_path / "weights.pt"),
        "--teacher-weights",
        str(tmp_path / "teacher.pt"),
        "--deployment",
        str(tmp_path / "deployment.pt"),
        "--output",
        str(output),
    ]
    assert decomposition.main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema": decomposition.SCHEMA,
        "schema_version": 1,
    }
    assert observed == {
        "config_path": Path(arguments[1]),
        "deployment_config_path": None,
        "weights_path": Path(arguments[3]),
        "teacher_weights_path": Path(arguments[5]),
        "deployment_path": Path(arguments[7]),
    }
    with pytest.raises(FileExistsError, match="not to overwrite"):
        decomposition.main(arguments)
