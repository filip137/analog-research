from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.reram_program_verify.hwa_population_sampler import _request as parse_hwa_population_request
from training.parameters import ParameterBinding, DenseWeight
from training.ibm_reram_hwa import sample_om_array_population, save_om_array_population, load_om_array_population


def _population_request() -> dict[str, object]:
    return {
        "preset": "reram_array_om",
        "assignment_seed": 87004,
        "corruption_policy": "published",
        "preset_default_corrupt_devices_prob": 0.0,
        "published_corrupt_devices_prob": 0.1348,
        "corrupt_devices_range": 0.01,
        "binding_keys": ["crossbar.layer0.tile0"],
        "binding_shapes": [[2, 3]],
        "required_aihwkit_version": "1.1.0",
    }

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("preset_default_corrupt_devices_prob", 0.1348),
        ("published_corrupt_devices_prob", 0.0),
        ("corrupt_devices_range", 0.1),
        ("corrupt_devices_range", True),
    ],
)
def test_external_population_request_pins_aihwkit_corruption_contract(
    field: str,
    value: object,
) -> None:
    request = _population_request()
    assert parse_hwa_population_request(json.dumps(request)) == request
    request[field] = value
    with pytest.raises(ValueError, match="canonical OM HWA population request"):
        parse_hwa_population_request(json.dumps(request))

def _binding(shape: tuple[int, int] = (2, 2)) -> ParameterBinding:
    parameter = DenseWeight(
        (shape[0],),
        (shape[1],),
        1.0,
        "cpu",
        clamp=True,
        clamp_min=0.1,
        clamp_max=1.0,
    )
    return ParameterBinding("base.dense_weight.0", parameter)

def test_repaired_assignment_changes_only_published_corrupt_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding((1, 4))

    def sample_hidden(shape, *, construction_seed, published_corruption):
        size = shape[0] * shape[1]
        values = {
            "max_bound": torch.full((size,), 1.0 if published_corruption else 0.8),
            "min_bound": torch.full((size,), -1.0 if published_corruption else -0.8),
            "dwmin_up": torch.full((size,), 0.2 if published_corruption else 0.3),
            "dwmin_down": torch.full((size,), 0.2 if published_corruption else 0.3),
            "reference": torch.zeros(size),
        }
        if published_corruption:
            for name in ("max_bound", "min_bound"):
                values[name][1] = 0.1
            values["dwmin_up"][1] = 0.0
            values["dwmin_down"][1] = 0.0
        return values, {
            "aihwkit_version": "1.1.0",
            "nominal_dw_min": 0.0949,
            "dw_min_std": 0.0,
            "write_noise_std": 0.0,
        }

    monkeypatch.setattr(
        "training.ibm_reram_hwa._sample_tile_hidden",
        sample_hidden,
    )
    published = sample_om_array_population(
        (binding,), assignment_seed=83001, corruption_policy="published"
    )
    repaired = sample_om_array_population(
        (binding,),
        assignment_seed=83001,
        corruption_policy="counterfactual_repaired",
    )

    healthy = ~published.published_corrupt
    assert torch.equal(published.max_bound[healthy], repaired.max_bound[healthy])
    assert torch.equal(published.dwmin_up[healthy], repaired.dwmin_up[healthy])
    assert published.published_corrupt.tolist() == [False, True, False, False]
    assert torch.equal(repaired.published_corrupt, published.published_corrupt)
    assert not bool(torch.any(repaired.corrupt))
    assert repaired.max_bound[1].item() == pytest.approx(0.8)

def test_om_array_population_npz_round_trip_and_fingerprint_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding((2, 3))

    def sample_hidden(shape, *, construction_seed, published_corruption):
        size = shape[0] * shape[1]
        offset = float(construction_seed % 17) / 1000.0
        return {
            "max_bound": torch.full((size,), 1.0 - offset),
            "min_bound": torch.full((size,), -1.0 + offset),
            "dwmin_up": torch.full((size,), 0.2 + offset),
            "dwmin_down": torch.full((size,), 0.21 + offset),
            "reference": torch.full((size,), offset),
        }, {
            "aihwkit_version": "1.1.0",
            "nominal_dw_min": 0.0949,
            "dw_min_std": 0.3,
            "write_noise_std": 0.1,
        }

    monkeypatch.setattr(
        "training.ibm_reram_hwa._sample_tile_hidden",
        sample_hidden,
    )
    population = sample_om_array_population(
        (binding,), assignment_seed=83001, corruption_policy="published"
    )
    path = tmp_path / "population.npz"
    save_om_array_population(path, population)
    loaded = load_om_array_population(path)
    assert loaded.fingerprint == population.fingerprint
    assert loaded.binding_keys == population.binding_keys
    assert loaded.binding_shapes == population.binding_shapes
    assert loaded.binding_sampling_seeds == population.binding_sampling_seeds
    assert loaded.donor_sampling_seeds == population.donor_sampling_seeds
    for name, expected in population.tensor_state().items():
        assert torch.equal(loaded.tensor_state()[name], expected)

    with np.load(path, allow_pickle=False) as raw:
        tampered = {name: raw[name].copy() for name in raw.files}
    tampered["fingerprint"] = np.asarray("0" * 64)
    bad = tmp_path / "tampered.npz"
    with bad.open("wb") as handle:
        np.savez_compressed(handle, **tampered)
    with pytest.raises(ValueError, match="recomputed OM population fingerprint"):
        load_om_array_population(bad)
