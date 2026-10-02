"""Explicit, config-selected IBM OM layers and their legacy back-fill."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from experiments.mnist_relu_drn.config import _parse_weight_modifier
from experiments.schema import to_plain_data
from training.ibm_om.constants import IBM_OM_RAW_ACTIVE_TOLERANCE
from training.ibm_om.mappings import map_ibm_reram_array_targets
from training.ibm_om.spec import (
    EXPLICIT_LAYER_FIELDS,
    IbmReramHwaConfig,
    backfill_config_dict,
    backfill_weight_modifier_metadata,
    legacy_explicit_defaults,
)


RAW_ACTIVE = {
    "execution": "pulse_resolved",
    "assignment_seed": 84001,
    "endpoint_seed": 84003,
    "corruption_policy": "counterfactual_repaired",
    "noisy_evaluation": True,
    "endpoint_policy": "preserve",
    "controller": "one_pulse",
    "target_mapping": "raw_active_p90_quad",
    "dual_rail_layout_by_parameter": {"base.dense_weight.0": "halves"},
    "raw_active_mode": "quantized_7_level",
    "raw_active_unsupported_quad_policy": "structural_failure",
}
LITERAL = {
    "execution": "compact_endpoint",
    "assignment_seed": 83001,
    "endpoint_seed": 83002,
    "corruption_policy": "published",
    "noisy_evaluation": False,
}


def test_legacy_defaults_name_each_layer_choice() -> None:
    assert legacy_explicit_defaults("raw_active_p90_quad") == {
        "device_coordinate": "raw_active_v1",
        "initial_state": "conditioned_lower_boundary",
        "verify_tolerance_absolute": IBM_OM_RAW_ACTIVE_TOLERANCE,
        "clamp_to_parameter_bounds": False,
        "characterization": "none",
    }
    assert legacy_explicit_defaults("shared_reset_relative_quad")[
        "characterization"
    ] == "reset_reads"
    for mapping in (
        "literal_global",
        "dual_rail_quad_common_window",
        "differential_pair_common_window",
    ):
        assert legacy_explicit_defaults(mapping) == {
            "device_coordinate": "logical_reference_subtracted",
            "initial_state": "exact_reset_bound",
            "verify_tolerance_absolute": None,
            "clamp_to_parameter_bounds": True,
            "characterization": "none",
        }


@pytest.mark.parametrize("parameters", [RAW_ACTIVE, LITERAL])
def test_config_stores_explicit_fields_and_accepts_them_declared(parameters) -> None:
    implicit = IbmReramHwaConfig(**parameters)
    stored = asdict(implicit)
    expected = legacy_explicit_defaults(parameters.get("target_mapping", "literal_global"))
    assert {name: stored[name] for name in EXPLICIT_LAYER_FIELDS} == expected
    declared = IbmReramHwaConfig(
        **parameters, **{name: stored[name] for name in EXPLICIT_LAYER_FIELDS}
    )
    assert declared == implicit
    # Old stored configs lack the fields; back-fill reproduces them exactly.
    legacy = {key: value for key, value in stored.items() if key not in EXPLICIT_LAYER_FIELDS}
    assert backfill_config_dict(legacy) == stored
    assert asdict(IbmReramHwaConfig(**legacy)) == stored


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device_coordinate", "raw_active_v1"),
        ("initial_state", "conditioned_lower_boundary"),
        ("verify_tolerance_absolute", 0.01),
        ("clamp_to_parameter_bounds", False),
        ("characterization", "reset_reads"),
    ],
)
def test_config_rejects_unsupported_layer_combinations(field, value) -> None:
    with pytest.raises(ValueError, match="supported combination"):
        IbmReramHwaConfig(**LITERAL, **{field: value})


def test_config_rejects_unknown_layer_values() -> None:
    with pytest.raises(ValueError, match="device_coordinate to be one of"):
        IbmReramHwaConfig(**LITERAL, device_coordinate="volts")
    with pytest.raises(ValueError, match="clamp_to_parameter_bounds"):
        IbmReramHwaConfig(**LITERAL, clamp_to_parameter_bounds=1)


def _parser_parameters(base: dict) -> dict:
    parameters = {
        "endpoint_policy": "clip_0_1",
        "target_out_of_support": "error",
        "preset": "reram_array_om",
        "controller": "adaptive",
        "start_protocol": "lower_to_target",
        "tolerance_step_ratio": 0.5,
        "maximum_program_pulses": 128,
        **base,
    }
    if parameters.get("target_mapping") == "raw_active_p90_quad":
        parameters["dual_rail_layout_by_parameter"] = {
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        }
        parameters["common_window_margin_fraction"] = 0.0
    return parameters


def _parse(parameters: dict) -> dict:
    settings = _parse_weight_modifier(
        {"type": "ibm_reram_om_program_verify", "parameters": parameters},
        "config.modes.train.weight_modifier",
    )
    return to_plain_data(settings.parameters)


@pytest.mark.parametrize("base", [RAW_ACTIVE, LITERAL])
def test_parser_always_emits_the_explicit_fields(base) -> None:
    parameters = _parser_parameters(base)
    normalized = _parse(parameters)
    implied = legacy_explicit_defaults(normalized["target_mapping"])
    assert {name: normalized[name] for name in EXPLICIT_LAYER_FIELDS} == implied
    assert _parse({**parameters, **implied}) == normalized
    with pytest.raises(Exception, match="device_coordinate"):
        _parse(
            {
                **parameters,
                "device_coordinate": (
                    "logical_reference_subtracted"
                    if implied["device_coordinate"] == "raw_active_v1"
                    else "raw_active_v1"
                ),
            }
        )


def test_old_checkpoint_modifier_metadata_back_fills_to_current() -> None:
    parameters = _parser_parameters(RAW_ACTIVE)
    current = {"type": "ibm_reram_om_program_verify", "parameters": _parse(parameters)}
    old = {
        "type": "ibm_reram_om_program_verify",
        "parameters": {
            key: value
            for key, value in current["parameters"].items()
            if key not in EXPLICIT_LAYER_FIELDS
        },
    }
    assert old != current
    assert backfill_weight_modifier_metadata(old) == current
    assert backfill_weight_modifier_metadata({"type": "none", "parameters": {}}) == {
        "type": "none",
        "parameters": {},
    }


def test_mapper_rejects_a_coordinate_the_mapping_cannot_use() -> None:
    import torch

    from training.ibm_om.population import IbmReramArrayPopulation

    size = 4
    population = IbmReramArrayPopulation(
        assignment_seed=1,
        corruption_policy="counterfactual_repaired",
        binding_keys=("base.dense_weight.0",),
        binding_shapes=((2, 2),),
        binding_sampling_seeds=(11,),
        donor_sampling_seeds=(12,),
        nominal_dw_min=0.0949,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=torch.ones(size),
        min_bound=-torch.ones(size),
        dwmin_up=torch.full((size,), 0.2),
        dwmin_down=torch.full((size,), 0.2),
        reference=torch.zeros(size),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint="coordinate-fixture",
        aihwkit_version="1.1.0",
    )
    fractions = torch.full((size,), 0.5)
    implicit, _report = map_ibm_reram_array_targets(
        fractions,
        population,
        target_mapping="literal_global",
        dual_rail_layout_by_parameter=None,
        common_window_margin_fraction=0.0,
    )
    declared, _report = map_ibm_reram_array_targets(
        fractions,
        population,
        target_mapping="literal_global",
        dual_rail_layout_by_parameter=None,
        common_window_margin_fraction=0.0,
        device_coordinate="logical_reference_subtracted",
    )
    assert torch.equal(implicit, declared)
    with pytest.raises(ValueError, match="device coordinate"):
        map_ibm_reram_array_targets(
            fractions,
            population,
            target_mapping="literal_global",
            dual_rail_layout_by_parameter=None,
            common_window_margin_fraction=0.0,
            device_coordinate="raw_active_v1",
        )
