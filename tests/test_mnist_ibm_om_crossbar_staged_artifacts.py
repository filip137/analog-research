from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from experiments.artifacts import sha256_file
from experiments.mnist_analog_relu.runtime import _matched_published_fault_overlay
from experiments.mnist_analog_relu.staged_artifacts import (
    STAGED_PROGRAM_STREAM_ROLE,
    StagedDeviceState,
    load_device_state,
    population_from_state,
    population_state,
    save_device_state,
)
from training.ibm_om_standard_crossbar import (
    CROSSBAR_TRAJECTORY_SEED_DERIVATION,
    IbmOmEffectiveCrossbarPlant,
    build_crossbar_layout,
    crossbar_state_bundle,
    crossbar_trajectory_seeds,
    tensor_sha256,
)
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    om_array_population_fingerprint,
)


def _with_fingerprint(population: IbmReramArrayPopulation) -> IbmReramArrayPopulation:
    tensors = population.tensor_state()
    fingerprint = om_array_population_fingerprint(
        assignment_seed=population.assignment_seed,
        corruption_policy=population.corruption_policy,
        keys=population.binding_keys,
        shapes=population.binding_shapes,
        binding_sampling_seeds=population.binding_sampling_seeds,
        donor_sampling_seeds=population.donor_sampling_seeds,
        scalar_parameters={
            "aihwkit_version": population.aihwkit_version,
            "nominal_dw_min": population.nominal_dw_min,
            "dw_min_std": population.dw_min_std,
            "write_noise_std": population.write_noise_std,
        },
        tensors=tensors,
    )
    return replace(population, fingerprint=fingerprint)


def _companions():
    layout = build_crossbar_layout((784, 256, 10), maximum_input_size=512)
    size = sum(tile.cells for tile in layout)
    mask = torch.zeros(size, dtype=torch.bool)
    mask[0] = True
    common = dict(
        assignment_seed=1234,
        binding_keys=tuple(tile.key for tile in layout),
        binding_shapes=tuple(tile.shape for tile in layout),
        binding_sampling_seeds=(11, 12, 13),
        donor_sampling_seeds=(21, 22, 23),
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=0.01,
        reference=torch.zeros(size, dtype=torch.float32),
        published_corrupt=mask,
        aihwkit_version="1.1.0",
    )
    healthy = _with_fingerprint(
        IbmReramArrayPopulation(
            corruption_policy="counterfactual_repaired",
            max_bound=torch.ones(size),
            min_bound=-torch.ones(size),
            dwmin_up=torch.full((size,), 0.1),
            dwmin_down=torch.full((size,), 0.1),
            corrupt=torch.zeros(size, dtype=torch.bool),
            fingerprint="pending",
            **common,
        )
    )
    maximum = torch.ones(size)
    minimum = -torch.ones(size)
    upward = torch.full((size,), 0.1)
    downward = torch.full((size,), 0.1)
    maximum[mask] = 0.005
    minimum[mask] = 0.005
    upward[mask] = 0.0
    downward[mask] = 0.0
    published = _with_fingerprint(
        IbmReramArrayPopulation(
            corruption_policy="published",
            max_bound=maximum,
            min_bound=minimum,
            dwmin_up=upward,
            dwmin_down=downward,
            corrupt=mask,
            fingerprint="pending",
            **common,
        )
    )
    return layout, healthy, published


def _healthy_state(tmp_path):
    layout, healthy, published = _companions()
    endpoint_seed = 5678
    trajectory_seeds = crossbar_trajectory_seeds(
        healthy,
        endpoint_seed=endpoint_seed,
        stream_role=STAGED_PROGRAM_STREAM_ROLE,
        random_stream_population_fingerprint=healthy.fingerprint,
    )
    seed_origin = {
        "schema": "ebl.ibm_om_crossbar_trajectory_seed_origin",
        "schema_version": 1,
        "derivation": CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        "stream_role": STAGED_PROGRAM_STREAM_ROLE,
        "assignment_seed": healthy.assignment_seed,
        "endpoint_seed": endpoint_seed,
        "random_stream_population_fingerprint": healthy.fingerprint,
        "trajectory_seeds_sha256": tensor_sha256(trajectory_seeds),
    }
    plant = IbmOmEffectiveCrossbarPlant(
        healthy,
        trajectory_seeds=trajectory_seeds,
        trajectory_seed_derivation=CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        device="cpu",
    )
    bundle = crossbar_state_bundle(
        plant,
        layout=layout,
        digital_scales=(1.0, 1.0),
        metadata={
            "assignment_seed": 1234,
            "endpoint_seed": endpoint_seed,
            "trajectory_seed_origin": seed_origin,
        },
    )
    state = StagedDeviceState(
        role="healthy_p0",
        source_kind="scratch",
        dims=(784, 256, 10),
        assignment_seed=1234,
        endpoint_seed=endpoint_seed,
        teacher_sha256="a" * 64,
        source_artifact_sha256=None,
        parent_device_state_sha256=None,
        healthy_population=healthy,
        published_population=published,
        healthy_p0=bundle,
        current=bundle,
        recovery=None,
    )
    path = tmp_path / "healthy.pt"
    save_device_state(path, state)
    return path, plant, state


def test_population_embedding_recomputes_fingerprint() -> None:
    _layout, healthy, _published = _companions()
    encoded = population_state(healthy)
    restored = population_from_state(encoded)
    assert restored.fingerprint == healthy.fingerprint
    encoded["max_bound"][1] -= 0.1
    with pytest.raises(ValueError, match="fingerprint"):
        population_from_state(encoded)


def test_healthy_device_state_round_trip_is_self_contained(tmp_path) -> None:
    path, _plant, original = _healthy_state(tmp_path)
    restored = load_device_state(path)
    assert restored.role == "healthy_p0"
    assert restored.current.plant_state_sha256 == original.current.plant_state_sha256
    assert restored.healthy_population.fingerprint == original.healthy_population.fingerprint


def test_staged_device_state_rejects_coherently_replaced_trajectory_seeds(
    tmp_path,
) -> None:
    _path, plant, original = _healthy_state(tmp_path)
    forged_plant = IbmOmEffectiveCrossbarPlant(
        original.healthy_population,
        trajectory_seeds=plant.trajectory_seeds.detach().cpu().flip(0),
        trajectory_seed_derivation=CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        device="cpu",
    )
    forged_bundle = crossbar_state_bundle(
        forged_plant,
        layout=original.healthy_p0.layout,
        digital_scales=original.healthy_p0.digital_scales,
        metadata=original.healthy_p0.metadata,
    )
    forged = replace(
        original,
        healthy_p0=forged_bundle,
        current=forged_bundle,
    )
    forged_path = tmp_path / "forged-seeds.pt"
    save_device_state(forged_path, forged)

    with pytest.raises(ValueError, match="exact declared staged trajectory seeds"):
        load_device_state(forged_path)


def test_faulted_state_authenticates_exact_healthy_parent(tmp_path) -> None:
    healthy_path, plant, healthy_state = _healthy_state(tmp_path)
    mask, stuck, report = _matched_published_fault_overlay(
        healthy=healthy_state.healthy_population,
        published=healthy_state.published_population,
        preset_default_corrupt_devices_prob=0.0,
        enabled_corrupt_devices_prob=0.1348,
        corrupt_devices_range=0.01,
    )
    plant.apply_stuck_at_fault_transition(
        mask=mask,
        stuck_persistent_q=stuck,
        transition_id="test-fault",
        source_population_fingerprint=healthy_state.published_population.fingerprint,
    )
    faulted_bundle = crossbar_state_bundle(
        plant,
        layout=healthy_state.healthy_p0.layout,
        digital_scales=healthy_state.healthy_p0.digital_scales,
        metadata={"fault_report": report},
    )
    faulted = replace(
        healthy_state,
        role="faulted_p0",
        parent_device_state_sha256=sha256_file(healthy_path),
        current=faulted_bundle,
    )
    path = tmp_path / "faulted.pt"
    save_device_state(path, faulted)
    assert load_device_state(path).current.state_kind == "faulted"

    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["healthy_p0"]["plant_state"]["persistent"][1] += 0.25
    # The nested plant digest detects the altered P0 before fault replay.
    tampered = tmp_path / "tampered.pt"
    torch.save(raw, tampered)
    with pytest.raises(ValueError):
        load_device_state(tampered)
