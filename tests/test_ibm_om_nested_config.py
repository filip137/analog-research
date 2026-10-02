"""Nested IBM OM config sections flatten to the canonical flat config."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn.config import _parse_weight_modifier
from experiments.schema import ConfigError, to_plain_data


ROOT = Path(__file__).resolve().parents[1]
NESTED = ROOT / "examples" / "mnist_relu_drn" / "nested"
FLAT_SOURCES = {
    "literal_global": "ibm_om_hwa_pilot/om_published.json",
    "dual_rail_quad_common_window": "ibm_om_common_window_hwa_pilot/exact_bounds_hwa.json",
    "differential_pair_common_window": "ibm_om_differential_pair_hwa_pilot/exact_bounds_hwa.json",
    "shared_reset_relative_quad": "ibm_om_reset_relative_quantized_hwa/continuous_hwa.json",
    "raw_active_p90_quad": "ibm_om_raw_active_p90_qat/continuous_hwa.json",
}


@pytest.mark.parametrize("mapping", sorted(FLAT_SOURCES))
@pytest.mark.parametrize("mode", ["train", "validate"])
def test_nested_example_resolves_like_its_flat_source(mapping: str, mode: str) -> None:
    flat_path = ROOT / "examples" / "mnist_relu_drn" / FLAT_SOURCES[mapping]
    nested_path = NESTED / f"{mapping}.json"
    _definition, flat = resolve_experiment_config(flat_path, mode)
    _definition, nested = resolve_experiment_config(nested_path, mode)
    assert to_plain_data(nested) == to_plain_data(flat)


def _nested_parameters() -> dict:
    document = json.loads((NESTED / "shared_reset_relative_quad.json").read_text())
    return document["modes"]["train"]["weight_modifier"]["parameters"]


def _parse(parameters: dict):
    return _parse_weight_modifier(
        {"type": "ibm_reram_om_program_verify", "parameters": parameters},
        "config.modes.train.weight_modifier",
    )


def test_nested_sections_must_all_be_declared() -> None:
    parameters = _nested_parameters()
    for section in ("assignment", "topology", "characterization", "mapping", "programming", "readback", "streams"):
        incomplete = copy.deepcopy(parameters)
        del incomplete[section]
        with pytest.raises(ConfigError, match=f"weight_modifier.parameters.{section} to be present"):
            _parse(incomplete)
    without_coordinate = copy.deepcopy(parameters)
    del without_coordinate["device_coordinate"]
    with pytest.raises(ConfigError, match="device_coordinate to be present"):
        _parse(without_coordinate)


def test_nested_sections_reject_missing_and_misplaced_keys() -> None:
    parameters = _nested_parameters()
    missing = copy.deepcopy(parameters)
    del missing["programming"]["initial_state"]
    with pytest.raises(ConfigError, match="parameters.programming to contain"):
        _parse(missing)
    misplaced = copy.deepcopy(parameters)
    misplaced["readback"]["endpoint_policy"] = "clip_0_1"
    with pytest.raises(ConfigError, match="parameters.readback to contain"):
        _parse(misplaced)
    mixed = copy.deepcopy(parameters)
    mixed["execution"] = "compact_endpoint"
    with pytest.raises(ConfigError, match="only the sections"):
        _parse(mixed)


def test_nested_explicit_fields_are_still_validated() -> None:
    parameters = copy.deepcopy(_nested_parameters())
    parameters["device_coordinate"] = "raw_active_v1"
    with pytest.raises(ConfigError, match="device_coordinate to equal"):
        _parse(parameters)
