"""Build the immutable IBM OM device-model bundle used for crossbar programming-error HWA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.artifacts import atomic_write_json, sha256_file
from training.ibm_reram_program_verify import PopulationStepEstimator


SCHEMA = "ebl.ibm_reram.om_hwa_device_model"
SCHEMA_VERSION = 1
CONDITION_KEY = "adaptive__lower_to_target__tau_step_0.5"
ESTIMATOR_KEY = "lower_to_target__tau_step_0.5"


def _read_object(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Expected {role} to be a readable JSON object. Provided value: "
            f"{str(resolved)!r}."
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected {role} to be a JSON object. Provided value: {value!r}."
        )
    return value


def _endpoint_model(
    path: Path,
    *,
    role: str,
    published_corruption: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    value = _read_object(path, role=role)
    if (
        value.get("schema")
        != "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model"
        or value.get("schema_version") != 2
    ):
        raise ValueError(
            f"Expected {role} to use the bounded endpoint-model schema version 2."
        )
    condition = value.get("conditions", {}).get(CONDITION_KEY)
    if not isinstance(condition, Mapping) or condition.get("adequate") is not True:
        raise ValueError(
            f"Expected {role} condition {CONDITION_KEY!r} to pass its held-out "
            "adequacy gate."
        )
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Expected {role} to contain characterization metadata.")
    for digest_key in (
        "trajectory_artifact_sha256",
        "device_population_artifact_sha256",
    ):
        digest = metadata.get(digest_key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"Expected {role} metadata {digest_key!r} to be a lowercase "
                "SHA-256 digest."
            )
    expected = {
        "preset": "reram_array_om",
        "execution_profile": "hwa_production_cap128",
        "enable_published_corruption": published_corruption,
    }
    mismatches = {
        key: {"expected": expected_value, "provided": metadata.get(key)}
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    controller = metadata.get("controller")
    if not isinstance(controller, Mapping) or controller.get(
        "maximum_program_pulses"
    ) != 128:
        mismatches["controller.maximum_program_pulses"] = {
            "expected": 128,
            "provided": (
                controller.get("maximum_program_pulses")
                if isinstance(controller, Mapping)
                else None
            ),
        }
    if mismatches:
        raise ValueError(
            f"Expected {role} to be the declared OM cap-128 characterization. "
            f"Provided value: {mismatches!r}."
        )
    resolved = path.expanduser().resolve()
    return value, {"path": str(resolved), "sha256": sha256_file(resolved)}


def _step_estimator(
    path: Path,
    *,
    role: str,
    published_corruption: bool,
    endpoint_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    value = _read_object(path, role=role)
    if (
        value.get("schema") != "ebl.ibm_reram.step_estimators"
        or value.get("schema_version") != 1
        or value.get("calibration_partition_only") is not True
    ):
        raise ValueError(
            f"Expected {role} to be a calibration-only step-estimator artifact."
        )
    estimator = value.get("estimators", {}).get(ESTIMATOR_KEY)
    if not isinstance(estimator, Mapping):
        raise ValueError(
            f"Expected {role} to contain estimator {ESTIMATOR_KEY!r}."
        )
    metadata = value.get("metadata")
    expected_metadata = {
        "execution_profile": "hwa_production_cap128",
        "preset": "reram_array_om",
        "enable_published_corruption": published_corruption,
        "nominal_dw_min": endpoint_metadata.get("nominal_dw_min"),
        "trajectory_artifact_sha256": endpoint_metadata.get(
            "trajectory_artifact_sha256"
        ),
        "device_population_artifact_sha256": endpoint_metadata.get(
            "device_population_artifact_sha256"
        ),
    }
    mismatches = {
        key: {
            "expected": expected,
            "provided": (
                metadata.get(key) if isinstance(metadata, Mapping) else None
            ),
        }
        for key, expected in expected_metadata.items()
        if not isinstance(metadata, Mapping) or metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Expected {role} provenance to match its endpoint artifact. "
            f"Provided value: {mismatches!r}."
        )
    restored = PopulationStepEstimator.from_mapping(estimator)
    if restored.fallback_step != float(endpoint_metadata["nominal_dw_min"]) / 2.0:
        raise ValueError(
            f"Expected {role} fallback step to equal the normalized OM nominal "
            "step."
        )
    resolved = path.expanduser().resolve()
    return dict(estimator), {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def build_om_hwa_device_model(
    *,
    continuous_endpoint: Path,
    continuous_estimators: Path,
    published_endpoint: Path,
    published_estimators: Path,
    output: Path,
) -> Path:
    """Combine matched cap-128 characterization artifacts without refitting."""

    continuous, continuous_source = _endpoint_model(
        continuous_endpoint,
        role="continuous endpoint model",
        published_corruption=False,
    )
    published, published_source = _endpoint_model(
        published_endpoint,
        role="published-corruption endpoint model",
        published_corruption=True,
    )
    continuous_estimator, continuous_estimator_source = _step_estimator(
        continuous_estimators,
        role="continuous step estimators",
        published_corruption=False,
        endpoint_metadata=continuous["metadata"],
    )
    published_estimator, published_estimator_source = _step_estimator(
        published_estimators,
        role="published-corruption step estimators",
        published_corruption=True,
        endpoint_metadata=published["metadata"],
    )
    continuous_step = float(continuous["metadata"]["nominal_dw_min"])
    published_step = float(published["metadata"]["nominal_dw_min"])
    if continuous_step != published_step:
        raise ValueError(
            "Expected continuous and published-corruption OM artifacts to use "
            "the same nominal pulse step."
        )
    continuous_controller = continuous["metadata"]["controller"]
    published_controller = published["metadata"]["controller"]
    if continuous_controller.get("adaptive") != published_controller.get("adaptive"):
        raise ValueError(
            "Expected continuous and published-corruption OM artifacts to use "
            "the same adaptive-controller settings."
        )
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "preset": "reram_array_om",
        "evidence_class": "model_based_aihwkit_preset",
        "coordinate": "x=(w+1)/2",
        "programming": {
            "controller": "adaptive",
            "start_protocol": "lower_to_target",
            "condition_key": CONDITION_KEY,
            "estimator_key": ESTIMATOR_KEY,
            "tolerance_step_ratio": 0.5,
            "maximum_program_pulses": 128,
            "adaptive": continuous_controller["adaptive"],
            "nominal_dw_min": continuous_step,
            "nominal_step_fraction": continuous_step / 2.0,
        },
        "endpoint_models": {
            "continuous": continuous,
            "published_corruption": published,
        },
        "step_estimators": {
            "continuous": continuous_estimator,
            "published_corruption": published_estimator,
        },
        "sources": {
            "continuous_endpoint": continuous_source,
            "continuous_estimators": continuous_estimator_source,
            "published_endpoint": published_source,
            "published_estimators": published_estimator_source,
        },
        "claim_boundary": (
            "Hardware-derived AIHWKit OM preset model; not raw independent "
            "device measurements. Compact endpoints may be used only after "
            "the embedded held-out adequacy gate passes."
        ),
    }
    destination = output.expanduser().resolve()
    atomic_write_json(destination, payload)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the strict IBM OM cap-128 HWA device-model bundle."
    )
    parser.add_argument("--continuous-endpoint", type=Path, required=True)
    parser.add_argument("--continuous-estimators", type=Path, required=True)
    parser.add_argument("--published-endpoint", type=Path, required=True)
    parser.add_argument("--published-estimators", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = build_om_hwa_device_model(
        continuous_endpoint=args.continuous_endpoint,
        continuous_estimators=args.continuous_estimators,
        published_endpoint=args.published_endpoint,
        published_estimators=args.published_estimators,
        output=args.output,
    )
    print(destination)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "CONDITION_KEY",
    "ESTIMATOR_KEY",
    "SCHEMA",
    "SCHEMA_VERSION",
    "build_om_hwa_device_model",
    "main",
]
