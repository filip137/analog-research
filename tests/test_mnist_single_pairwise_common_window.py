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
    _model_checkpoint_metadata,
    _validate_checkpoint_metadata,
)
from ebl.schema.base import RunMode
from methods.measured_trace import MeasuredTraceOptimizer


ROOT = Path(__file__).resolve().parents[1]
MEASURED = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_single_pairwise_common_window_finetune_10ep.json"
)
FP32 = (
    ROOT
    / "examples/mnist_relu_drn/"
    "ideal_single_centered_70_90us_finetune_10ep.json"
)
CAMPAIGN = (
    ROOT
    / "campaigns/manifests/"
    "mnist_relu_drn_single_pairwise_common_window_vs_fp32_10ep.json"
)


def _train(path: Path):
    payload = json.loads(path.read_text())
    definition, document = parse_experiment_config(payload)
    return definition.resolve(document, RunMode.TRAIN)


def test_measured_pairwise_protocol_is_explicit_and_range_screened() -> None:
    spec = _train(MEASURED)

    assert spec.model.encoding == "single"
    assert spec.mapping.range_placement == "centered"
    assert spec.mapping.scale_fraction_pairs == (
        (1.0, 0.125),
        (1.0, 0.25),
        (1.0, 0.5),
        (1.0, 0.75),
        (1.0, 1.0),
    )
    assert spec.settings.num_epochs == 10
    assert spec.settings.update_backend.type == "measured_cohort_a"
    parameters = spec.settings.update_backend.parameters
    assert (
        parameters["initial_target_mapping"]
        == "dual_rail_pairwise_common_window"
    )
    assert dict(parameters["dual_rail_layout_by_parameter"]) == {
        "base.dense_weight.0": "halves",
        "base.dense_weight.1": "paired",
    }


def test_fp32_control_uses_matched_uniform_bounds_without_devices() -> None:
    spec = _train(FP32)

    assert spec.model.encoding == "single"
    assert spec.model.conductance_min == 70.0e-6
    assert spec.model.conductance_max == 90.0e-6
    assert spec.mapping.range_placement == "centered"
    assert spec.settings.update_backend.type == "ideal"
    assert dict(spec.settings.update_backend.parameters) == {}


def test_campaign_has_two_train_test_arms() -> None:
    manifest = load_campaign_manifest(CAMPAIGN)

    assert [stage.stage_id for stage in manifest.stages] == [
        "measured_pairwise_train",
        "measured_pairwise_test",
        "fp32_bounded_train",
        "fp32_bounded_test",
    ]
    assert manifest.stages[1].depends_on == ("measured_pairwise_train",)
    assert manifest.stages[3].depends_on == ("fp32_bounded_train",)


def test_checkpoint_metadata_serializes_frozen_layout_mapping(
    monkeypatch,
) -> None:
    spec = _train(MEASURED)
    spec = replace(spec, runtime=replace(spec.runtime, device="cpu"))
    stack = build_student_stack(spec, enable_measured=False)
    measured = object.__new__(MeasuredTraceOptimizer)
    monkeypatch.setattr(
        MeasuredTraceOptimizer,
        "data_report",
        property(
            lambda _self: {
                "source_sha256": "device-sha",
                "assignment_sha256_by_parameter": {},
            }
        ),
    )
    stack = replace(stack, optimizer=measured)

    metadata = _model_checkpoint_metadata(
        stack,
        spec=spec,
        teacher_sha256="teacher-sha",
        mapping=None,
    )

    assert metadata["dual_rail_layout_by_parameter"] == {
        "base.dense_weight.0": "halves",
        "base.dense_weight.1": "paired",
    }
    json.dumps(metadata)

    validation_arguments = {
        "spec": spec,
        "teacher_sha256": "teacher-sha",
        "expected_amplification_indices": metadata[
            "amplification_indices"
        ],
    }
    _validate_checkpoint_metadata(metadata, **validation_arguments)
    mutations = {
        "conductance_bounds_s": [0.0, 1.0e-4],
        "mapping_range_placement": "lower",
        "mapping_scale_fraction_pairs": [[1.0, 1.0]],
        "initial_target_mapping": "per_device_affine",
        "dual_rail_layout_by_parameter": {
            "base.dense_weight.0": "paired",
            "base.dense_weight.1": "halves",
        },
    }
    for key, value in mutations.items():
        mutated = deepcopy(metadata)
        mutated[key] = value
        with pytest.raises(ValueError, match=key):
            _validate_checkpoint_metadata(mutated, **validation_arguments)
