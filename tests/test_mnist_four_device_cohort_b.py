from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from ebl.execution.schema import load_campaign_manifest
from ebl.registry import parse_experiment_config
from ebl.runtime.mnist_relu_drn.components import build_student_stack
from ebl.runtime.mnist_relu_drn.runtime import (
    _amplification_index_report,
    _single_topology_signature,
    _validate_cohort_b_source_metadata,
)
from ebl.schema.base import RunMode


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_single_quad_common_window_finetune_10ep.json"
)
DEPLOYMENT = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_single_quad_common_window_cohort_b_deployment_control.json"
)
FINETUNE = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_single_quad_common_window_cohort_b_finetune_10ep.json"
)
CAMPAIGN = (
    ROOT
    / "campaigns/manifests/"
    "mnist_relu_drn_single_quad_common_window_cohort_b_10ep.json"
)


def _train(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition, document = parse_experiment_config(payload)
    return definition.resolve(document, RunMode.TRAIN)


def test_four_device_cohort_b_configs_freeze_matched_protocol() -> None:
    source = _train(SOURCE)
    deployment = _train(DEPLOYMENT)
    finetune = _train(FINETUNE)

    for target in (deployment, finetune):
        assert target.model == source.model
        assert target.mapping == source.mapping
        assert target.runtime == source.runtime
        assert target.data == source.data
        assert target.solver == source.solver
        assert target.settings.update_backend.type == "measured_cohort_b"
        parameters = target.settings.update_backend.parameters
        assert parameters["cohort"] == "B"
        assert (
            parameters["initial_target_mapping"]
            == "dual_rail_quad_common_window"
        )
        assert dict(parameters["dual_rail_layout_by_parameter"]) == {
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        }
    assert deployment.settings.num_epochs == 1
    assert deployment.settings.max_batches == 1
    assert deployment.settings.learning_rates == (0.0, 0.0)
    assert finetune.settings.num_epochs == 10
    assert finetune.settings.max_batches is None
    assert finetune.settings.learning_rates == source.settings.learning_rates


def test_four_device_cohort_b_source_provenance_fails_closed() -> None:
    spec = _train(FINETUNE)
    metadata = {
        "encoding": "single",
        "initialization": "teacher_mapped",
        "update_backend": "measured_cohort_a",
        "initial_target_mapping": "dual_rail_quad_common_window",
        "dual_rail_layout_by_parameter": {
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        },
        "device_data_sha256": "device-sha",
    }
    _validate_cohort_b_source_metadata(
        metadata,
        spec=spec,
        device_data_sha256="device-sha",
    )

    for key in tuple(metadata):
        mutated = deepcopy(metadata)
        mutated[key] = "wrong"
        with pytest.raises(ValueError, match=key):
            _validate_cohort_b_source_metadata(
                mutated,
                spec=spec,
                device_data_sha256="device-sha",
            )


def test_four_device_cohort_b_topology_is_model_local_across_builds() -> None:
    spec = _train(FINETUNE)
    spec = replace(spec, runtime=replace(spec.runtime, device="cpu"))
    first = build_student_stack(spec, enable_measured=False)
    second = build_student_stack(spec, enable_measured=False)
    expected = ((0, 1, 1.0, 1.0), (1, 2, 4.0, 0.0625))

    for stack in (first, second):
        assert _single_topology_signature(
            _amplification_index_report(stack),
            voltage_amp=4.0,
            current_amp=0.25,
        ) == expected
    assert [
        item["pre_layer"]
        for item in _amplification_index_report(first)
    ] != [
        item["pre_layer"]
        for item in _amplification_index_report(second)
    ]


def test_four_device_cohort_b_campaign_has_control_and_adaptation() -> None:
    manifest = load_campaign_manifest(CAMPAIGN)

    assert manifest.campaign_id == (
        "mnist-four-device-reram-cohort-b-quad-common-window-10ep-"
        "20260819-v1"
    )
    assert [stage.stage_id for stage in manifest.stages] == [
        "cohort_b_quad_deployment_control",
        "cohort_b_quad_deployment_test",
        "cohort_b_quad_train",
        "cohort_b_quad_test",
    ]
    assert manifest.stages[1].depends_on == (
        "cohort_b_quad_deployment_control",
    )
    assert manifest.stages[3].depends_on == ("cohort_b_quad_train",)
