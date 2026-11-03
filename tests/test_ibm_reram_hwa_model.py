from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.artifacts import sha256_file
from experiments.reram_program_verify.hwa_model import (
    CONDITION_KEY,
    ESTIMATOR_KEY,
    build_om_hwa_device_model,
)
from methods.programming.om import PopulationStepEstimator


def _endpoint(path: Path, *, corrupt: bool, adequate: bool = True) -> Path:
    trajectory_digest = ("b" if corrupt else "a") * 64
    population_digest = ("d" if corrupt else "c") * 64
    payload = {
        "schema": "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model",
        "schema_version": 2,
        "metadata": {
            "preset": "reram_array_om",
            "execution_profile": "hwa_production_cap128",
            "enable_published_corruption": corrupt,
            "nominal_dw_min": 0.0949,
            "trajectory_artifact_sha256": trajectory_digest,
            "device_population_artifact_sha256": population_digest,
            "controller": {
                "maximum_program_pulses": 128,
                "adaptive": {
                    "eta": 0.75,
                    "maximum_batch": 32,
                    "epsilon": 1e-8,
                    "force_one_within_steps": 2.0,
                },
            },
        },
        "conditions": {CONDITION_KEY: {"adequate": adequate}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _estimators(path: Path, *, corrupt: bool = False) -> Path:
    payload = {
        "schema": "ebl.ibm_reram.step_estimators",
        "schema_version": 1,
        "calibration_partition_only": True,
        "metadata": {
            "execution_profile": "hwa_production_cap128",
            "preset": "reram_array_om",
            "enable_published_corruption": corrupt,
            "nominal_dw_min": 0.0949,
            "trajectory_artifact_sha256": ("b" if corrupt else "a") * 64,
            "device_population_artifact_sha256": ("d" if corrupt else "c") * 64,
        },
        "estimators": {
            ESTIMATOR_KEY: PopulationStepEstimator(
                bins=4,
                fallback_step=0.0949 / 2.0,
            ).to_mapping()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_hwa_bundle_embeds_matched_sources_without_refitting(tmp_path: Path) -> None:
    continuous = _endpoint(tmp_path / "continuous.json", corrupt=False)
    published = _endpoint(tmp_path / "published.json", corrupt=True)
    continuous_estimator = _estimators(
        tmp_path / "continuous-estimator.json", corrupt=False
    )
    published_estimator = _estimators(
        tmp_path / "published-estimator.json", corrupt=True
    )
    output = tmp_path / "device-model.json"

    assert build_om_hwa_device_model(
        continuous_endpoint=continuous,
        continuous_estimators=continuous_estimator,
        published_endpoint=published,
        published_estimators=published_estimator,
        output=output,
    ) == output.resolve()
    payload = json.loads(output.read_text())
    assert payload["schema"] == "ebl.ibm_reram.om_hwa_device_model"
    assert payload["programming"]["maximum_program_pulses"] == 128
    assert payload["sources"]["continuous_endpoint"]["sha256"] == sha256_file(
        continuous
    )
    assert set(payload["endpoint_models"]) == {
        "continuous",
        "published_corruption",
    }


def test_hwa_bundle_fails_closed_when_endpoint_gate_fails(tmp_path: Path) -> None:
    continuous = _endpoint(
        tmp_path / "continuous.json",
        corrupt=False,
        adequate=False,
    )
    published = _endpoint(tmp_path / "published.json", corrupt=True)
    estimator = _estimators(tmp_path / "estimator.json")
    with pytest.raises(ValueError, match="adequacy gate"):
        build_om_hwa_device_model(
            continuous_endpoint=continuous,
            continuous_estimators=estimator,
            published_endpoint=published,
            published_estimators=estimator,
            output=tmp_path / "device-model.json",
        )


def test_hwa_bundle_fails_closed_on_mismatched_estimator_provenance(
    tmp_path: Path,
) -> None:
    continuous = _endpoint(tmp_path / "continuous.json", corrupt=False)
    published = _endpoint(tmp_path / "published.json", corrupt=True)
    continuous_estimator = _estimators(
        tmp_path / "continuous-estimator.json", corrupt=False
    )
    wrong_published_estimator = _estimators(
        tmp_path / "published-estimator.json", corrupt=False
    )

    with pytest.raises(ValueError, match="provenance"):
        build_om_hwa_device_model(
            continuous_endpoint=continuous,
            continuous_estimators=continuous_estimator,
            published_endpoint=published,
            published_estimators=wrong_published_estimator,
            output=tmp_path / "device-model.json",
        )
