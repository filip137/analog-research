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
    _model_checkpoint_metadata,
    _single_topology_signature,
    _validate_checkpoint_metadata,
)
from ebl.schema.base import ConfigError, RunMode
from methods.measured_trace import MeasuredTraceOptimizer


ROOT = Path(__file__).resolve().parents[1]
FOUR_DEVICE = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_single_quad_common_window_finetune_10ep.json"
)
EIGHT_DEVICE = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_differential_finetune_10ep.json"
)
CAMPAIGN = (
    ROOT
    / "campaigns/manifests/"
    "mnist_relu_drn_four_vs_eight_common_window_10ep.json"
)


def _train(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition, document = parse_experiment_config(payload)
    return definition.resolve(document, RunMode.TRAIN)


def test_four_device_protocol_uses_one_window_per_logical_synapse() -> None:
    spec = _train(FOUR_DEVICE)

    assert spec.experiment_id == "mnist_relu_drn_kd.v1"
    assert spec.model.encoding == "single"
    assert spec.model.include_biases is False
    assert spec.model.amplification_indexing == "logical"
    assert spec.model.voltage_amp == 4.0
    assert spec.model.current_amp == 0.25
    assert spec.settings.num_epochs == 10
    assert spec.settings.learning_rates == (2.1e-10, 5.7e-13)
    parameters = spec.settings.update_backend.parameters
    assert (
        parameters["initial_target_mapping"]
        == "dual_rail_quad_common_window"
    )
    assert dict(parameters["dual_rail_layout_by_parameter"]) == {
        "base.dense_weight.0": "halves",
        "base.dense_weight.1": "paired",
    }


def test_four_and_eight_device_arms_freeze_shared_comparison_controls() -> None:
    four = _train(FOUR_DEVICE)
    eight = _train(EIGHT_DEVICE)

    assert four.runtime == eight.runtime
    assert four.data == eight.data
    assert four.solver == eight.solver
    assert four.settings.num_epochs == eight.settings.num_epochs == 10
    assert four.settings.temperature == eight.settings.temperature == 1.0
    assert four.settings.update_backend.type == (
        eight.settings.update_backend.type
    ) == "measured_cohort_a"
    for name in (
        "curve_preprocessing",
        "split_seed",
        "assignment_seed",
        "formed_resistance_max_ohm",
        "cohort_fraction",
        "cohort",
        "source_traces_per_cell",
        "initial_pulse_index",
        "projection",
        "expected_trace_length",
        "programming_deadband_mode",
        "probabilistic_write_mode",
    ):
        assert four.settings.update_backend.parameters[name] == (
            eight.settings.update_backend.parameters[name]
        )


def test_campaign_runs_four_and_eight_device_train_test_arms() -> None:
    manifest = load_campaign_manifest(CAMPAIGN)

    assert manifest.campaign_id == (
        "mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1"
    )
    assert [stage.stage_id for stage in manifest.stages] == [
        "four_device_quad_train",
        "four_device_quad_test",
        "eight_device_differential_train",
        "eight_device_differential_test",
    ]
    assert manifest.stages[1].depends_on == ("four_device_quad_train",)
    assert manifest.stages[3].depends_on == (
        "eight_device_differential_train",
    )


def test_four_device_topology_is_model_local_across_repeated_builds() -> None:
    spec = _train(FOUR_DEVICE)
    spec = replace(spec, runtime=replace(spec.runtime, device="cpu"))
    first = build_student_stack(spec, enable_measured=False)
    second = build_student_stack(spec, enable_measured=False)
    first_report = _amplification_index_report(first)
    second_report = _amplification_index_report(second)
    expected = ((0, 1, 1.0, 1.0), (1, 2, 4.0, 0.0625))

    assert _single_topology_signature(
        first_report,
        voltage_amp=4.0,
        current_amp=0.25,
    ) == expected
    assert _single_topology_signature(
        second_report,
        voltage_amp=4.0,
        current_amp=0.25,
    ) == expected
    assert [item["pre_layer"] for item in first_report] != [
        item["pre_layer"] for item in second_report
    ]


def test_checkpoint_metadata_freezes_quad_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _train(FOUR_DEVICE)
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

    assert (
        metadata["initial_target_mapping"]
        == "dual_rail_quad_common_window"
    )
    assert metadata["dual_rail_layout_by_parameter"] == {
        "base.dense_weight.0": "halves",
        "base.dense_weight.1": "paired",
    }
    _validate_checkpoint_metadata(
        metadata,
        spec=spec,
        teacher_sha256="teacher-sha",
        expected_amplification_indices=metadata["amplification_indices"],
    )

    mutated = deepcopy(metadata)
    mutated["initial_target_mapping"] = (
        "dual_rail_pairwise_common_window"
    )
    with pytest.raises(ValueError, match="initial_target_mapping"):
        _validate_checkpoint_metadata(
            mutated,
            spec=spec,
            teacher_sha256="teacher-sha",
            expected_amplification_indices=metadata[
                "amplification_indices"
            ],
        )


def test_quad_mapping_fails_closed_outside_the_single_model_layout() -> None:
    payload = json.loads(FOUR_DEVICE.read_text(encoding="utf-8"))
    payload["model"]["encoding"] = "differential"
    with pytest.raises(ConfigError, match="single model encoding"):
        parse_experiment_config(payload)

    payload = json.loads(FOUR_DEVICE.read_text(encoding="utf-8"))
    layouts = payload["modes"]["train"]["update_backend"]["parameters"][
        "dual_rail_layout_by_parameter"
    ]
    layouts["base.dense_weight.0"] = "paired"
    with pytest.raises(ConfigError, match="model-local dual-rail layouts"):
        parse_experiment_config(payload)
