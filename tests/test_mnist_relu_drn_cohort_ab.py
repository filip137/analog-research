from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ebl.registry import parse_experiment_config
from ebl.runtime.mnist_relu_drn.runtime import (
    _adaptation_summary,
    _validate_cohort_b_source_metadata,
    _validate_train_request,
)
from ebl.schema.base import RunMode


ROOT = Path(__file__).resolve().parents[1]
LITERAL_CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_differential_cohort_b_literal_finetune_10ep.json"
)
COMMON_WINDOW_CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_differential_cohort_b_common_window_finetune_10ep.json"
)
CAMPAIGN = (
    ROOT
    / "campaigns/manifests/"
    "mnist_relu_drn_differential_cohort_ab_10ep.json"
)
DEPLOYMENT_CONTROLS = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_differential_cohort_b_literal_deployment_control.json",
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_differential_cohort_b_common_window_deployment_control.json",
)


def _train_spec(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition, document = parse_experiment_config(payload)
    return definition.resolve(document, RunMode.TRAIN)


@pytest.mark.parametrize(
    ("path", "mapping"),
    [
        (LITERAL_CONFIG, "literal"),
        (COMMON_WINDOW_CONFIG, "paired_affine_common_window"),
    ],
)
def test_cohort_b_configs_freeze_matched_ten_epoch_protocol(
    path: Path,
    mapping: str,
) -> None:
    train = _train_spec(path)

    assert train.model.encoding == "differential"
    assert train.model.voltage_amp == 4.0
    assert train.model.current_amp == 0.25
    assert train.settings.num_epochs == 10
    assert train.settings.learning_rates == (4.2e-10, 1.14e-12)
    assert train.settings.update_backend.type == "measured_cohort_b"
    assert train.settings.update_backend.parameters["cohort"] == "B"
    assert (
        train.settings.update_backend.parameters["initial_target_mapping"]
        == mapping
    )


def test_cohort_b_config_fails_closed_on_backend_cohort_mismatch() -> None:
    payload = json.loads(LITERAL_CONFIG.read_text(encoding="utf-8"))
    payload["modes"]["train"]["update_backend"]["parameters"][
        "cohort"
    ] = "A"

    with pytest.raises(ValueError, match="cohort"):
        parse_experiment_config(payload)


@pytest.mark.parametrize("path", DEPLOYMENT_CONTROLS)
def test_deployment_controls_make_no_conductance_update(path: Path) -> None:
    train = _train_spec(path)

    assert train.settings.num_epochs == 1
    assert train.settings.max_batches == 1
    assert train.settings.learning_rates == (0.0, 0.0)


def test_cohort_b_single_encoding_rejects_non_quad_transfer() -> None:
    payload = json.loads(LITERAL_CONFIG.read_text(encoding="utf-8"))
    payload["model"]["encoding"] = "single"

    with pytest.raises(ValueError, match="dual_rail_quad_common_window"):
        parse_experiment_config(payload)


def test_cohort_b_training_requires_a_source_checkpoint() -> None:
    spec = _train_spec(LITERAL_CONFIG)
    request = SimpleNamespace(
        teacher_weights=Path("teacher.pt"),
        base_weights=None,
        weights=Path("cohort-a.pt"),
        resume=None,
        device_data=Path("devices.hdf5"),
    )
    _validate_train_request(request, spec)

    request.weights = None
    with pytest.raises(ValueError, match="cohort-A --weights"):
        _validate_train_request(request, spec)


def test_cohort_b_source_provenance_fails_closed() -> None:
    spec = _train_spec(COMMON_WINDOW_CONFIG)
    metadata = {
        "encoding": "differential",
        "initialization": "teacher_mapped",
        "update_backend": "measured_cohort_a",
        "initial_target_mapping": "paired_affine_common_window",
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


def test_adaptation_summary_uses_post_deployment_calibrated_baseline() -> None:
    baseline = {
        "student_accuracy": 0.90,
        "teacher_agreement": 0.91,
        "kl_teacher_student": 0.20,
    }
    selected = {
        "student_accuracy": 0.95,
        "teacher_agreement": 0.96,
        "kl_teacher_student": 0.10,
    }
    last = {
        "student_accuracy": 0.94,
        "teacher_agreement": 0.95,
        "kl_teacher_student": 0.12,
    }

    report = _adaptation_summary(
        baseline,
        selected,
        last,
        selected_epoch=7,
    )

    assert report["baseline"] == "post_deployment_calibrated_validation"
    assert report["selected_epoch"] == 7
    assert report["selected"]["student_accuracy_change"] == pytest.approx(
        0.05
    )
    assert report["selected"]["relative_kl_reduction"] == pytest.approx(
        0.5
    )
    assert report["after_ten_epochs"]["kl_reduction"] == pytest.approx(
        0.08
    )


def test_cohort_ab_campaign_has_a_and_two_b_transfer_arms() -> None:
    payload = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    stages = payload["stages"]

    assert [stage["id"] for stage in stages] == [
        "cohort_a_train",
        "cohort_a_test",
        "cohort_b_literal_deployment_control",
        "cohort_b_literal_deployment_test",
        "cohort_b_literal_train",
        "cohort_b_literal_test",
        "cohort_b_common_window_deployment_control",
        "cohort_b_common_window_deployment_test",
        "cohort_b_common_window_train",
        "cohort_b_common_window_test",
    ]
    for index in (2, 4, 6, 8):
        stage = stages[index]
        assert stage["depends_on"] == ["cohort_a_train"]
        assert stage["inputs"]["weights"] == {
            "stage": "cohort_a_train",
            "artifact_kind": "selected_named_weights",
        }
