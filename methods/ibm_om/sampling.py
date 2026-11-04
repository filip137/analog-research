"""Validate fitted IBM OM model artifacts and expose device/network sampling helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch

from core.artifacts import sha256_file
from hardware.ibm_om.model_schema import (
    CONDITION_KEY,
    SCHEMA as DEVICE_MODEL_SCHEMA,
    SCHEMA_VERSION as DEVICE_MODEL_SCHEMA_VERSION,
)
from networks.resistive.builders import ParameterBinding
from networks.variable.parameter import DenseWeight
from hardware.ibm_om.constants import (
    _ARRAY_SAMPLING_RECEIPT_SCHEMA,
    _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION,
    _CORRUPTION_POLICIES,
    _REQUIRED_AIHWKIT_VERSION,
)
from hardware.ibm_om.population import (
    IbmReramArrayPopulation,
    _native_seed,
    _population_fingerprint,
    load_om_array_population,
)
from methods.ibm_reram_program_verify import (
    OM_PRESET,
    PUBLISHED_CORRUPT_PROBABILITY,
    PopulationStepEstimator,
    derive_seed,
)


# Repository root (this module lives in methods/ibm_om/).
_ROOT = Path(__file__).resolve().parents[2]



def _artifact(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Expected --device-model to name a readable IBM OM HWA JSON "
            f"artifact. Provided value: {str(resolved)!r}."
        ) from error
    if not isinstance(value, dict) or (
        value.get("schema") != DEVICE_MODEL_SCHEMA
        or value.get("schema_version") != DEVICE_MODEL_SCHEMA_VERSION
    ):
        raise ValueError(
            "Expected --device-model to use the IBM OM HWA schema version 1."
        )
    programming = value.get("programming")
    expected = {
        "controller": "adaptive",
        "start_protocol": "lower_to_target",
        "condition_key": CONDITION_KEY,
        "tolerance_step_ratio": 0.5,
        "maximum_program_pulses": 128,
    }
    if not isinstance(programming, Mapping) or any(
        programming.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError(
            "Expected --device-model to contain the declared OM cap-128 "
            "adaptive lower-from-RESET protocol."
        )
    endpoint_models = value.get("endpoint_models")
    estimators = value.get("step_estimators")
    if (
        not isinstance(endpoint_models, Mapping)
        or set(endpoint_models) != {"continuous", "published_corruption"}
        or not isinstance(estimators, Mapping)
        or set(estimators) != {"continuous", "published_corruption"}
    ):
        raise ValueError(
            "Expected --device-model to contain matched continuous and "
            "published-corruption endpoint models and step estimators."
        )
    for branch, published_corruption in (
        ("continuous", False),
        ("published_corruption", True),
    ):
        model = endpoint_models[branch]
        if (
            not isinstance(model, Mapping)
            or model.get("schema")
            != "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model"
            or model.get("schema_version") != 2
        ):
            raise ValueError(
                f"Expected {branch} to contain endpoint-model schema version 2."
            )
        condition = model.get("conditions", {}).get(CONDITION_KEY)
        if not isinstance(condition, Mapping) or condition.get("adequate") is not True:
            raise ValueError(
                f"Expected embedded {branch} endpoint condition to pass its "
                "held-out adequacy gate."
            )
        metadata = model.get("metadata")
        expected_metadata = {
            "preset": OM_PRESET,
            "execution_profile": "hwa_production_cap128",
            "enable_published_corruption": published_corruption,
        }
        if not isinstance(metadata, Mapping) or any(
            metadata.get(key) != expected_value
            for key, expected_value in expected_metadata.items()
        ):
            raise ValueError(
                f"Expected embedded {branch} endpoint metadata to match the "
                "declared OM cap-128 corruption arm."
            )
        PopulationStepEstimator.from_mapping(estimators[branch])
    return value, sha256_file(resolved)



from hardware.ibm_om.sampling import _sample_tile_hidden



from hardware.ibm_om.sampling import _sample_om_array_population_layout



from networks.resistive.ibm_om.population import _binding_layout



from networks.resistive.ibm_om.population import sample_om_array_population



from hardware.ibm_om.sampling import sample_om_array_population_layout
