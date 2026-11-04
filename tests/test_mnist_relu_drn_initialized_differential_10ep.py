from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from ebl.registry import parse_experiment_config
from ebl.runtime.mnist_relu_drn.components import build_student_stack
from ebl.runtime.mnist_relu_drn.runtime import (
    _amplification_index_report,
    _differential_topology_signature,
    _validate_checkpoint_metadata,
)
from ebl.schema.base import RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/measured_raw_differential_finetune_10ep.json"
)
CAMPAIGN = (
    ROOT
    / "campaigns/manifests/mnist_relu_drn_initialized_differential_10ep.json"
)


def _specs():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    definition, document = parse_experiment_config(payload)
    return (
        definition.resolve(document, RunMode.TRAIN),
        definition.resolve(document, RunMode.VALIDATE),
    )


def test_initialized_differential_config_freezes_the_ten_epoch_protocol() -> None:
    train, validate = _specs()

    assert train.experiment_id == "mnist_relu_drn_kd.v1"
    assert train.model.encoding == "differential"
    assert train.model.include_biases is False
    assert train.model.amplification_indexing == "logical"
    assert train.model.voltage_amp == 4.0
    assert train.model.current_amp == 0.25
    assert train.settings.num_epochs == 10
    assert train.settings.learning_rates == (4.2e-10, 1.14e-12)
    assert train.settings.update_backend.type == "measured_cohort_a"
    assert (
        train.settings.update_backend.parameters["initial_target_mapping"]
        == "paired_affine_common_window"
    )
    assert validate.settings.split == "test"
    assert validate.settings.sample_limit is None


def test_initialized_differential_campaign_has_train_then_explicit_test() -> None:
    payload = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    assert payload["campaign_id"] == (
        "mnist-differential-reram-initialized-finetune-10ep-20260817-v1"
    )
    assert [stage["id"] for stage in payload["stages"]] == [
        "initialized_differential_train",
        "initialized_differential_test",
    ]
    train, test = payload["stages"]
    assert train["command"] == "train"
    assert test["command"] == "validate"
    assert test["depends_on"] == ["initialized_differential_train"]
    assert test["inputs"]["weights"] == {
        "stage": "initialized_differential_train",
        "artifact_kind": "selected_named_weights",
    }


def test_differential_topology_is_model_local_across_repeated_construction() -> None:
    train, _validate = _specs()
    train = replace(train, runtime=replace(train.runtime, device="cpu"))
    first = build_student_stack(train, enable_measured=False)
    second = build_student_stack(train, enable_measured=False)
    first_report = _amplification_index_report(first)
    second_report = _amplification_index_report(second)

    expected = ((0, 1, 1.0, 1.0), (1, 2, 4.0, 0.0625))
    assert _differential_topology_signature(first_report) == expected
    assert _differential_topology_signature(second_report) == expected
    assert [item["pre_layer"] for item in first_report] != [
        item["pre_layer"] for item in second_report
    ]


def test_checkpoint_topology_validation_ignores_names_but_rejects_semantics() -> None:
    train, _validate = _specs()
    train = replace(train, runtime=replace(train.runtime, device="cpu"))
    stack = build_student_stack(train, enable_measured=False)
    report = _amplification_index_report(stack)
    metadata = {
        "experiment_id": train.experiment_id,
        "encoding": "differential",
        "include_biases": False,
        "amplification_indexing": "logical",
        "amplification_indices": report,
        "objective": "teacher_kl",
        "initialization": "teacher_mapped",
        "temperature": 1.0,
        "teacher_sha256": "teacher-sha",
    }
    _validate_checkpoint_metadata(
        metadata,
        spec=train,
        teacher_sha256="teacher-sha",
        expected_amplification_indices=report,
    )

    mutated = copy.deepcopy(metadata)
    mutated["amplification_indices"][1]["post_layer_metric"] = 1.0
    with pytest.raises(ValueError, match="amplification_indices"):
        _validate_checkpoint_metadata(
            mutated,
            spec=train,
            teacher_sha256="teacher-sha",
            expected_amplification_indices=report,
        )
