from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ebl.registry import resolve_experiment_config
from ebl.runtime.mnist_relu_drn.runtime import _validate_train_request
from ebl.schema.base import ConfigError, RunMode
from ebl.lifecycle.study import load_study_plan


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_reset_relative_on_chip_recovery"
)
_STUDY = (
    _ROOT
    / "studies"
    / "mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1.json"
)
_THRESHOLD_CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_reset_relative_direct_gradient_threshold"
)
_THRESHOLD_STUDY = (
    _ROOT
    / "studies"
    / "mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1.json"
)
_TTV2_CONFIG = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_reset_relative_ttv2_repair"
    / "ttv2_ideal_budget_0p1.json"
)
_TTV2_STUDY = (
    _ROOT
    / "studies"
    / "mnist-ibm-om-reset-relative-ttv2-repair-20260825-v1.json"
)


def _request(**overrides):
    values = {
        "teacher_weights": Path("teacher.pt"),
        "base_weights": None,
        "weights": Path("source.pt"),
        "resume": None,
        "device_data": None,
        "device_model": Path("device.json"),
        "deployment": Path("deployment.pt"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _aihwkit_ttv2_payload() -> dict:
    payload = json.loads(_TTV2_CONFIG.read_text())
    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    parameters["method"] = "ttv2_aihwkit_1p1_minibatch_equation"
    parameters.pop("ttv2_gamma0")
    parameters.pop("ttv2_buffer_residual_mode")
    parameters.update(
        {
            "ttv2_fast_lr_by_parameter": {
                "base.dense_weight.0": 0.5,
                "base.dense_weight.1": 0.25,
            },
            "ttv2_transfer_lr": 1.0,
            "ttv2_scale_transfer_lr": True,
            "ttv2_units_in_mbatch": True,
            "ttv2_auto_scale": False,
            "ttv2_fast_granularity_by_parameter": {
                "base.dense_weight.0": 0.02,
                "base.dense_weight.1": 0.03,
            },
            "ttv2_buffer_granularity": 1.0,
            "ttv2_auto_granularity": 10000.0,
            "ttv2_correct_gradient_magnitudes": True,
            "ttv2_desired_bl": 1,
            "ttv2_momentum": 0.0,
            "ttv2_forget_buffer": True,
            "ttv2_cap_scope": "per_parameter_proportional",
            "ttv2_cursor_policy": "zero",
            "ttv2_in_chop_probability": 0.0,
        }
    )
    return payload


def _write_payload(tmp_path: Path, payload: dict, name: str = "config.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_predeclared_recovery_grid_is_exact_and_uses_one_source() -> None:
    expected = {
        (method, backend, budget)
        for method, backend in (
            ("rail_refresh", None),
            ("direct_pulse", None),
            ("tiki_taka", "ideal"),
            ("tiki_taka", "physical_om"),
        )
        for budget in (0.1, 1.0, 4.0)
    }
    observed = set()
    for path in sorted(_CONFIG_ROOT.glob("*.json")):
        if path.name == "development_deployment.json":
            continue
        _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
        parameters = spec.settings.update_backend.parameters
        observed.add(
            (
                parameters["method"],
                parameters["fast_backend"],
                parameters["slow_pulse_budget_per_cell"],
            )
        )
        assert parameters["expected_assignment_seed"] == 84501
        assert parameters["expected_endpoint_seed"] == 84601
        assert parameters["expected_source_selection_epoch"] == 4
        assert parameters["expected_source_weights_sha256"] == (
            "537681598b887d4d63c6429c7f7bee3a7082041215c09d6cb3c25947452f311a"
        )
        assert dict(parameters["source_dual_rail_layout_by_parameter"]) == {
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        }
        assert spec.settings.num_epochs == 10
        assert spec.settings.selection_evaluation == "clean"
        assert spec.settings.selection_noise_repeats == 1
    assert observed == expected


def test_recovery_study_declares_source_plus_twelve_one_seed_arms() -> None:
    study = load_study_plan(_STUDY)
    assert len(study["arms"]) == 13
    assert sum(len(arm["configs"]) for arm in study["arms"]) == 13
    assert [arm["mode"] for arm in study["arms"]].count("validate") == 1
    assert [arm["mode"] for arm in study["arms"]].count("train") == 12


def test_threshold_grid_is_exact_and_freezes_control_scales() -> None:
    expected_scales = {
        0.1: (
            0.005644706616388747,
            "f3ded58c814858f3bb1a6c8da282521e04ee8f3fdaebf9cba8ffacb1cc02a1c2",
        ),
        1.0: (
            0.05644706616388745,
            "e9944e8ce1494e95369bc632be02a8ca60588be20d22ba59471d836290b2dbc1",
        ),
        4.0: (
            0.2257882646555498,
            "b36e5d09f648ad47b12c12c629587cd826587fc5460d3dac4f8aea55c3cc23b1",
        ),
    }
    observed = set()
    for path in sorted(_THRESHOLD_CONFIG_ROOT.glob("*.json")):
        _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
        parameters = spec.settings.update_backend.parameters
        percentile = parameters["direct_gradient_magnitude_percentile"]
        budget = parameters["slow_pulse_budget_per_cell"]
        observed.add((percentile, budget))
        scale, digest = expected_scales[budget]
        assert parameters["direct_probability_scale"] == scale
        assert (
            parameters["direct_probability_scale_source_report_sha256"]
            == digest
        )
        assert parameters["method"] == "direct_pulse"
        assert parameters["direct_probability_calibration_batches"] == 64
        assert spec.settings.num_epochs == 10
    assert observed == {
        (percentile, budget)
        for percentile in (90.0, 95.0, 99.0)
        for budget in (0.1, 1.0, 4.0)
    }


def test_threshold_study_declares_nine_one_seed_arms() -> None:
    study = load_study_plan(_THRESHOLD_STUDY)
    assert len(study["arms"]) == 9
    assert all(arm["mode"] == "train" for arm in study["arms"])
    assert sum(len(arm["configs"]) for arm in study["arms"]) == 9


def test_ttv2_repair_is_one_frozen_ideal_fast_arm() -> None:
    _definition, spec = resolve_experiment_config(_TTV2_CONFIG, RunMode.TRAIN)
    parameters = spec.settings.update_backend.parameters
    assert parameters["method"] == "ttv2"
    assert parameters["fast_backend"] == "ideal"
    assert parameters["slow_pulse_budget_per_cell"] == 0.1
    assert parameters["update_seed"] == 85901
    assert parameters["ttv2_transfer_every"] == 1
    assert parameters["ttv2_gamma0"] == 10000.0
    assert parameters["ttv2_fast_update_model"] == "symmetric_soft_bounds"
    assert parameters["ttv2_scan_mode"] == "dual_rail_input_pair"
    assert parameters["ttv2_buffer_threshold"] == 1.0
    assert parameters["ttv2_buffer_residual_mode"] == "subtract_dispatched"
    assert spec.settings.num_epochs == 10

    study = load_study_plan(_TTV2_STUDY)
    assert [arm["arm_id"] for arm in study["arms"]] == [
        "ttv2-ideal-budget-0p1"
    ]


def test_ttv2_rejects_legacy_fast_slice_reset_semantics(tmp_path: Path) -> None:
    payload = json.loads(_TTV2_CONFIG.read_text())
    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    parameters["ttv2_buffer_residual_mode"] = "clear_fast_slice"
    path = tmp_path / "invalid_ttv2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="subtract_dispatched"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_aihwkit_ttv2_minibatch_equation_config_is_strict_and_normalized(
    tmp_path: Path,
) -> None:
    path = _write_payload(tmp_path, _aihwkit_ttv2_payload())
    _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    parameters = spec.settings.update_backend.parameters

    assert parameters["method"] == "ttv2_aihwkit_1p1_minibatch_equation"
    assert parameters["fast_backend"] == "ideal"
    assert parameters["ttv2_transfer_every"] == 1
    assert parameters["ttv2_fast_update_model"] == "symmetric_soft_bounds"
    assert parameters["ttv2_fast_weight_limit"] == 1.0
    assert parameters["ttv2_scan_mode"] == "dual_rail_input_pair"
    assert parameters["ttv2_buffer_threshold"] == 1.0
    assert dict(parameters["ttv2_fast_lr_by_parameter"]) == {
        "base.dense_weight.0": 0.5,
        "base.dense_weight.1": 0.25,
    }
    assert dict(parameters["ttv2_fast_granularity_by_parameter"]) == {
        "base.dense_weight.0": 0.02,
        "base.dense_weight.1": 0.03,
    }
    assert parameters["ttv2_transfer_lr"] == 1.0
    assert parameters["ttv2_scale_transfer_lr"] is True
    assert parameters["ttv2_units_in_mbatch"] is True
    assert parameters["ttv2_auto_scale"] is False
    assert parameters["ttv2_buffer_granularity"] == 1.0
    assert parameters["ttv2_auto_granularity"] == 10000.0
    assert parameters["ttv2_correct_gradient_magnitudes"] is True
    assert parameters["ttv2_desired_bl"] == 1
    assert parameters["ttv2_momentum"] == 0.0
    assert parameters["ttv2_forget_buffer"] is True
    assert parameters["ttv2_cap_scope"] == "per_parameter_proportional"
    assert parameters["ttv2_cursor_policy"] == "zero"
    assert parameters["ttv2_in_chop_probability"] == 0.0
    assert "ttv2_gamma0" not in parameters
    assert "ttv2_buffer_residual_mode" not in parameters


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        (
            "ttv2_fast_lr_by_parameter",
            {"base.dense_weight.0": 0.5},
            "map exactly the source layout keys",
        ),
        (
            "ttv2_fast_lr_by_parameter",
            {"base.dense_weight.0": 0.5, "base.dense_weight.1": 0.0},
            "positive finite numbers",
        ),
        ("ttv2_transfer_lr", 0.0, "to be positive"),
        ("ttv2_scale_transfer_lr", 1, "to be a boolean"),
        ("ttv2_units_in_mbatch", False, "to equal True"),
        ("ttv2_auto_scale", True, "to equal False"),
        (
            "ttv2_fast_granularity_by_parameter",
            {"base.dense_weight.0": 0.02, "unexpected": 0.03},
            "map exactly the source layout keys",
        ),
        ("ttv2_buffer_granularity", 0.0, "to be positive"),
        ("ttv2_auto_granularity", 0.0, "to be positive"),
        ("ttv2_correct_gradient_magnitudes", False, "to equal True"),
        ("ttv2_desired_bl", 2, "to equal 1"),
        ("ttv2_momentum", 0.1, "to equal 0"),
        ("ttv2_forget_buffer", False, "to equal True"),
        ("ttv2_cap_scope", "global", "per_parameter_proportional"),
        ("ttv2_cursor_policy", "random", "zero"),
        ("ttv2_in_chop_probability", 0.1, "to equal 0"),
    ],
)
def test_aihwkit_ttv2_rejects_invalid_explicit_fields(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    payload = _aihwkit_ttv2_payload()
    payload["modes"]["train"]["update_backend"]["parameters"][field] = invalid
    path = _write_payload(tmp_path, payload, f"invalid_{field}.json")
    with pytest.raises(ConfigError, match=message):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_aihwkit_ttv2_requires_every_explicit_field(tmp_path: Path) -> None:
    payload = _aihwkit_ttv2_payload()
    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    del parameters["ttv2_transfer_lr"]
    path = _write_payload(tmp_path, payload, "missing_aihwkit_field.json")
    with pytest.raises(ConfigError, match="every required AIHWKit-1.1 TTv2 field"):
        resolve_experiment_config(path, RunMode.TRAIN)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ttv2_gamma0", 10000.0),
        ("ttv2_buffer_residual_mode", "subtract_dispatched"),
    ],
)
def test_aihwkit_ttv2_rejects_legacy_scale_and_residual_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _aihwkit_ttv2_payload()
    payload["modes"]["train"]["update_backend"]["parameters"][field] = value
    path = _write_payload(tmp_path, payload, f"forbidden_{field}.json")
    with pytest.raises(ConfigError, match="omit ttv2_gamma0"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_recovery_requires_exact_deployment_and_source_or_resume() -> None:
    _definition, spec = resolve_experiment_config(
        _CONFIG_ROOT / "direct_pulse_budget_1.json",
        RunMode.TRAIN,
    )
    _validate_train_request(_request(), spec)
    _validate_train_request(
        _request(weights=None, resume=Path("resume.pt")),
        spec,
    )
    with pytest.raises(ValueError, match="exact --deployment"):
        _validate_train_request(_request(deployment=None), spec)
    with pytest.raises(ValueError, match="exactly one"):
        _validate_train_request(_request(weights=None, resume=None), spec)


def test_recovery_config_rejects_hidden_layout_drift(tmp_path: Path) -> None:
    payload = json.loads(
        (_CONFIG_ROOT / "direct_pulse_budget_1.json").read_text()
    )
    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    parameters["source_dual_rail_layout_by_parameter"][
        "base.dense_weight.1"
    ] = "halves"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="source_dual_rail_layout"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_threshold_config_rejects_unregistered_percentile(tmp_path: Path) -> None:
    payload = json.loads(
        (_THRESHOLD_CONFIG_ROOT / "p95_budget_0p1.json").read_text()
    )
    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    parameters["direct_gradient_magnitude_percentile"] = 97
    path = tmp_path / "invalid_threshold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="equal 90, 95, or 99"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_non_recovery_training_rejects_deployment_argument() -> None:
    _definition, spec = resolve_experiment_config(
        _ROOT / "examples" / "mnist_relu_drn" / "ideal_single.json",
        RunMode.TRAIN,
    )
    with pytest.raises(ValueError, match="only for ibm_om_deployed_recovery"):
        _validate_train_request(
            _request(device_model=None),
            spec,
        )
