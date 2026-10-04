from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ebl.cli import CommandHandlers, TrainRequest, ValidateRequest, main
from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.runtime import (
    _evaluate,
    _ibm_target_mapping_preflights,
    _modifier_forward_gain,
    _selection_improved,
    _selected_payload,
    _validate_array_specific_modifier_population_parity,
)
from experiments.schema import ConfigError, RunMode
from experiments.study_workflow import load_study_plan
from training.ibm_reram_hwa import IbmReramHwaParameterModifier
from training.core.modifier import SplitParameterModifier


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _ROOT / "examples" / "mnist_relu_drn" / "ibm_om_hwa_pilot"
_COMMON_WINDOW_CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_common_window_hwa_pilot"
)
_RESET_RELATIVE_CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_reset_relative_quantized_hwa"
)
_RAW_ACTIVE_CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_raw_active_p90_qat"
)
_STUDY = (
    _ROOT
    / "studies"
    / "mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json"
)
_RESET_RELATIVE_STUDY = (
    _ROOT
    / "studies"
    / "mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1.json"
)


def test_raw_active_production_configs_freeze_matched_protocol() -> None:
    expected = {
        "zero_update_quantized.json": (
            "none",
            (0.0, 0.0),
            "quantized_7_level",
            223.87211385683378,
        ),
        "clean_bptt_quantized_deploy.json": (
            "none",
            (0.1388429752066116, 0.00037685950413223146),
            "quantized_7_level",
            223.87211385683378,
        ),
        "continuous_hwa.json": (
            "ibm_reram_om_program_verify",
            (0.00701276595744681, 0.000019031148936170215),
            "continuous",
            251.18864315095797,
        ),
        "quantized_qat.json": (
            "ibm_reram_om_program_verify",
            (0.017647659574468084, 0.00004789923404255319),
            "quantized_7_level",
            223.87211385683378,
        ),
    }
    for name, (training_type, rates, mode, gain) in expected.items():
        _definition, spec = resolve_experiment_config(
            _RAW_ACTIVE_CONFIG_ROOT / name,
            RunMode.TRAIN,
        )
        assert spec.model.conductance_min == 0.0
        assert spec.model.conductance_max == 1.0
        assert spec.settings.learning_rates == rates
        assert spec.settings.weight_modifier.type == training_type
        selection = spec.settings.selection_weight_modifier.parameters
        assert selection["target_mapping"] == "raw_active_p90_quad"
        assert selection["raw_active_mode"] == mode
        assert selection["forward_logit_gain"] == gain
        assert selection["execution"] == "pulse_resolved"
        assert selection["controller"] == "one_pulse"
        assert selection["endpoint_policy"] == "preserve"
        assert spec.settings.selection_noise_repeats == 3
        if training_type == "ibm_reram_om_program_verify":
            training = spec.settings.weight_modifier.parameters
            assert training["execution"] == "mapped_target"
            assert training["raw_active_mode"] == mode
            assert training["forward_logit_gain"] == gain

    for mode, gain in (
        ("continuous", 251.18864315095797),
        ("quantized_7_level", 223.87211385683378),
    ):
        label = "continuous" if mode == "continuous" else "quantized"
        for endpoint_seed in range(85101, 85106):
            _definition, spec = resolve_experiment_config(
                _RAW_ACTIVE_CONFIG_ROOT
                / f"heldout_{label}_seed_{endpoint_seed}.json",
                RunMode.VALIDATE,
            )
            parameters = spec.settings.weight_modifier.parameters
            assert parameters["assignment_seed"] == 85001
            assert parameters["endpoint_seed"] == endpoint_seed
            assert parameters["raw_active_mode"] == mode
            assert parameters["forward_logit_gain"] == gain


def test_four_training_arms_share_common_physical_selection() -> None:
    expected_training = {
        "clean.json": "none",
        "gaussian_3pct.json": "add_normal",
        "om_repaired.json": "ibm_reram_om_program_verify",
        "om_published.json": "ibm_reram_om_program_verify",
    }
    selections = []
    for name, modifier_type in expected_training.items():
        _definition, spec = resolve_experiment_config(
            _CONFIG_ROOT / name,
            RunMode.TRAIN,
        )
        assert spec.settings.weight_modifier.type == modifier_type
        selection = spec.settings.selection_weight_modifier
        assert selection.type == "ibm_reram_om_program_verify"
        assert selection.parameters["execution"] == "pulse_resolved"
        assert selection.parameters["corruption_policy"] == "published"
        assert selection.parameters["assignment_seed"] == 83003
        assert selection.parameters["endpoint_seed"] == 83003
        assert selection.parameters["target_mapping"] == "literal_global"
        assert selection.parameters["dual_rail_layout_by_parameter"] is None
        assert selection.parameters["common_window_margin_fraction"] == 0.0
        selections.append(dict(selection.parameters))
    assert selections[1:] == selections[:-1]


def test_heldout_assignments_and_endpoint_seeds_are_predeclared() -> None:
    for offset in range(4):
        assignment = 83101 + offset
        _definition, spec = resolve_experiment_config(
            _CONFIG_ROOT / f"heldout_assignment_{assignment}.json",
            RunMode.VALIDATE,
        )
        modifier = spec.settings.weight_modifier
        assert modifier.parameters["assignment_seed"] == assignment
        assert modifier.parameters["endpoint_seed"] == 83201 + offset
        assert modifier.parameters["execution"] == "pulse_resolved"
        assert modifier.parameters["corruption_policy"] == "published"


def test_study_declares_each_checkpoint_assignment_validation_as_one_run() -> None:
    study = load_study_plan(_STUDY)
    training_arms = [arm for arm in study["arms"] if arm["mode"] == "train"]
    validation_arms = [
        arm for arm in study["arms"] if arm["mode"] == "validate"
    ]

    assert len(training_arms) == 4
    assert len(validation_arms) == 16
    assert {
        arm["arm_id"] for arm in validation_arms
    } == {
        f"test-{training}-{assignment}"
        for training in (
            "clean",
            "gaussian-3pct",
            "om-repaired",
            "om-published",
        )
        for assignment in range(83101, 83105)
    }


def test_training_ibm_modifier_must_be_compact(tmp_path: Path) -> None:
    payload = json.loads((_CONFIG_ROOT / "om_repaired.json").read_text())
    payload["modes"]["train"]["weight_modifier"]["parameters"][
        "execution"
    ] = "pulse_resolved"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="compact_endpoint"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_selection_ibm_modifier_must_keep_published_corruption(
    tmp_path: Path,
) -> None:
    payload = json.loads((_CONFIG_ROOT / "clean.json").read_text())
    payload["modes"]["train"]["selection_weight_modifier"]["parameters"][
        "corruption_policy"
    ] = "counterfactual_repaired"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="published corruption"):
        resolve_experiment_config(path, RunMode.TRAIN)


def _set_quad_mapping(parameters: dict, *, margin: float = 0.25) -> None:
    parameters.update(
        {
            "target_mapping": "dual_rail_quad_common_window",
            "dual_rail_layout_by_parameter": {
                "base.dense_weight.0": "halves",
                "base.dense_weight.1": "paired",
            },
            "common_window_margin_fraction": margin,
        }
    )


def _set_differential_pair_mapping(
    parameters: dict,
    *,
    margin: float = 0.25,
) -> None:
    parameters.update(
        {
            "target_mapping": "differential_pair_common_window",
            "dual_rail_layout_by_parameter": None,
            "common_window_margin_fraction": margin,
        }
    )


def _set_shared_reset_relative_mapping(
    parameters: dict,
    *,
    mode: str = "quantized_9_level",
) -> None:
    parameters.update(
        {
            "target_mapping": "shared_reset_relative_quad",
            "dual_rail_layout_by_parameter": {
                "base.dense_weight.0": "halves",
                "base.dense_weight.1": "paired",
            },
            "common_window_margin_fraction": 0.0,
            "reset_relative_mode": mode,
            "reset_relative_contrast_step": 0.095849,
            "reset_read_samples": 8,
            "reset_guard_standard_errors": 3.0,
            "forward_logit_gain": 20.0,
        }
    )


def _set_raw_active_p90_mapping(
    parameters: dict,
    *,
    mode: str = "quantized_7_level",
) -> None:
    parameters.update(
        {
            "target_mapping": "raw_active_p90_quad",
            "dual_rail_layout_by_parameter": {
                "base.dense_weight.0": "halves",
                "base.dense_weight.1": "paired",
            },
            "common_window_margin_fraction": 0.0,
            "raw_active_mode": mode,
            "raw_active_unsupported_quad_policy": "structural_failure",
            "controller": "one_pulse",
            "endpoint_policy": "preserve",
            "forward_logit_gain": 20.0,
        }
    )


def _differential_pair_hwa_payload() -> dict:
    payload = json.loads(
        (_ROOT / "examples/mnist_relu_drn/ideal_differential.json").read_text()
    )
    source = json.loads((_CONFIG_ROOT / "om_repaired.json").read_text())
    training = json.loads(
        json.dumps(
            source["modes"]["train"]["weight_modifier"]
        )
    )
    selection = json.loads(
        json.dumps(
            source["modes"]["train"]["selection_weight_modifier"]
        )
    )
    train_parameters = training["parameters"]
    selection_parameters = selection["parameters"]
    _set_differential_pair_mapping(train_parameters)
    _set_differential_pair_mapping(selection_parameters)
    train_parameters.update(
        {
            "assignment_seed": 84001,
            "endpoint_seed": 84002,
            "corruption_policy": "counterfactual_repaired",
        }
    )
    selection_parameters.update(
        {
            "assignment_seed": 84001,
            "endpoint_seed": 84003,
            "corruption_policy": "counterfactual_repaired",
        }
    )
    payload["modes"]["train"].update(
        {
            "weight_modifier": training,
            "selection_evaluation": "modifier",
            "selection_noise_repeats": 1,
            "selection_weight_modifier": selection,
        }
    )
    return payload


def test_quad_selection_requires_repaired_corruption_and_exact_layout(
    tmp_path: Path,
) -> None:
    payload = json.loads((_CONFIG_ROOT / "clean.json").read_text())
    parameters = payload["modes"]["train"]["selection_weight_modifier"][
        "parameters"
    ]
    _set_quad_mapping(parameters)
    parameters["corruption_policy"] = "counterfactual_repaired"
    path = tmp_path / "quad-clean.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    parsed = spec.settings.selection_weight_modifier.parameters
    assert parsed["target_mapping"] == "dual_rail_quad_common_window"
    assert parsed["corruption_policy"] == "counterfactual_repaired"
    assert parsed["common_window_margin_fraction"] == 0.25
    assert dict(parsed["dual_rail_layout_by_parameter"]) == {
        "base.dense_weight.0": "halves",
        "base.dense_weight.1": "paired",
    }

    payload["modes"]["train"]["selection_weight_modifier"]["parameters"][
        "dual_rail_layout_by_parameter"
    ]["base.dense_weight.1"] = "halves"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="canonical W1/W2 layout"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_quad_hwa_training_and_selection_must_share_fixed_array_mapping(
    tmp_path: Path,
) -> None:
    payload = json.loads((_CONFIG_ROOT / "om_repaired.json").read_text())
    train_parameters = payload["modes"]["train"]["weight_modifier"][
        "parameters"
    ]
    selection_parameters = payload["modes"]["train"][
        "selection_weight_modifier"
    ]["parameters"]
    _set_quad_mapping(train_parameters)
    _set_quad_mapping(selection_parameters)
    selection_parameters["assignment_seed"] = train_parameters["assignment_seed"]
    selection_parameters["corruption_policy"] = "counterfactual_repaired"
    path = tmp_path / "quad-hwa.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    assert (
        spec.settings.weight_modifier.parameters["assignment_seed"]
        == spec.settings.selection_weight_modifier.parameters["assignment_seed"]
    )

    selection_parameters["assignment_seed"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="fixed array"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_differential_pair_mapping_requires_differential_encoding_and_null_layout(
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    path = tmp_path / "pair-hwa.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    training = spec.settings.weight_modifier.parameters
    selection = spec.settings.selection_weight_modifier.parameters
    assert spec.model.encoding == "differential"
    assert training["target_mapping"] == "differential_pair_common_window"
    assert training["dual_rail_layout_by_parameter"] is None
    assert training["common_window_margin_fraction"] == 0.25
    assert selection["corruption_policy"] == "counterfactual_repaired"
    assert training["endpoint_seed"] == 84002
    assert selection["endpoint_seed"] == 84003
    assert training["execution"] == "compact_endpoint"
    assert selection["execution"] == "pulse_resolved"

    payload["model"]["encoding"] = "single"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="model.encoding='differential'"):
        resolve_experiment_config(path, RunMode.TRAIN)

    payload = _differential_pair_hwa_payload()
    payload["modes"]["train"]["weight_modifier"]["parameters"][
        "dual_rail_layout_by_parameter"
    ] = {"base.dense_weight.0": "halves"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="to be null"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_quad_mapping_rejects_differential_model_encoding(
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    for role in ("weight_modifier", "selection_weight_modifier"):
        parameters = payload["modes"]["train"][role]["parameters"]
        _set_quad_mapping(parameters)
    path = tmp_path / "differential-quad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="model.encoding='single'"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_shared_reset_relative_config_selects_accuracy_and_matches_commissioning(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (_COMMON_WINDOW_CONFIG_ROOT / "exact_bounds_hwa.json").read_text()
    )
    training = payload["modes"]["train"]["weight_modifier"]["parameters"]
    selection = payload["modes"]["train"]["selection_weight_modifier"][
        "parameters"
    ]
    _set_shared_reset_relative_mapping(training)
    _set_shared_reset_relative_mapping(selection)
    payload["modes"]["train"]["selection_metric"] = "student_accuracy"
    payload["modes"]["train"]["selection_noise_repeats"] = 3
    path = tmp_path / "shared-reset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    assert spec.model.encoding == "single"
    assert spec.settings.selection_metric == "student_accuracy"
    assert spec.settings.selection_noise_repeats == 3
    assert training["reset_read_samples"] == 8
    assert (
        spec.settings.weight_modifier.parameters["reset_relative_mode"]
        == "quantized_9_level"
    )

    for role in ("weight_modifier", "selection_weight_modifier"):
        payload["modes"]["train"][role]["parameters"].pop(
            "forward_logit_gain"
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    _definition, spec_without_gain = resolve_experiment_config(
        path,
        RunMode.TRAIN,
    )
    assert (
        spec_without_gain.settings.weight_modifier.parameters[
            "forward_logit_gain"
        ]
        is None
    )

    payload["modes"]["train"]["selection_weight_modifier"]["parameters"][
        "reset_read_samples"
    ] = 16
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="fixed array"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_raw_active_p90_config_uses_direct_g_mapping_and_one_pulse_selection(
    tmp_path: Path,
) -> None:
    payload = json.loads((_CONFIG_ROOT / "om_repaired.json").read_text())
    payload["model"]["conductance_min"] = 0.0
    payload["model"]["conductance_max"] = 1.0
    training = payload["modes"]["train"]["weight_modifier"]["parameters"]
    selection = payload["modes"]["train"]["selection_weight_modifier"][
        "parameters"
    ]
    _set_raw_active_p90_mapping(training)
    _set_raw_active_p90_mapping(selection)
    training.update(
        {
            "execution": "mapped_target",
            "assignment_seed": 84001,
            "endpoint_seed": 84002,
            "corruption_policy": "counterfactual_repaired",
            "noisy_evaluation": False,
        }
    )
    selection.update(
        {
            "execution": "pulse_resolved",
            "assignment_seed": 84001,
            "endpoint_seed": 84003,
            "corruption_policy": "counterfactual_repaired",
            "noisy_evaluation": True,
        }
    )
    payload["modes"]["train"]["selection_evaluation"] = "modifier"
    payload["modes"]["train"]["selection_noise_repeats"] = 1
    path = tmp_path / "raw-active.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.TRAIN)
    assert spec.model.conductance_min == 0.0
    assert spec.model.conductance_max == 1.0
    assert spec.settings.weight_modifier.parameters["execution"] == "mapped_target"
    assert spec.settings.selection_weight_modifier.parameters["controller"] == "one_pulse"
    assert spec.settings.selection_weight_modifier.parameters["endpoint_policy"] == "preserve"

    payload["model"]["conductance_min"] = 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="not rescaled a second time"):
        resolve_experiment_config(path, RunMode.TRAIN)

    payload["model"]["conductance_min"] = 0.0
    payload["modes"]["train"]["weight_modifier"]["parameters"][
        "execution"
    ] = "compact_endpoint"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="historical q-coordinate"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_frozen_reset_relative_training_configs_define_four_matched_pipelines() -> None:
    expected = {
        "zero_update_quantized.json": (0, "none", None, 562.341325190349),
        "clean_bptt_quantized_deploy.json": (
            10,
            "none",
            None,
            562.341325190349,
        ),
        "continuous_hwa.json": (
            10,
            "ibm_reram_om_program_verify",
            "continuous",
            707.945784384138,
        ),
        "quantized_qat.json": (
            10,
            "ibm_reram_om_program_verify",
            "quantized_9_level",
            562.341325190349,
        ),
    }
    for name, (epochs, training_type, training_mode, selection_gain) in expected.items():
        _definition, spec = resolve_experiment_config(
            _RESET_RELATIVE_CONFIG_ROOT / name,
            RunMode.TRAIN,
        )
        assert spec.settings.num_epochs == epochs
        assert spec.settings.weight_modifier.type == training_type
        assert spec.settings.selection_metric == "student_accuracy"
        assert spec.settings.selection_noise_repeats == 3
        selection = spec.settings.selection_weight_modifier.parameters
        assert selection["assignment_seed"] == 84001
        assert selection["endpoint_seed"] == 84003
        assert selection["target_mapping"] == "shared_reset_relative_quad"
        assert selection["reset_read_samples"] == 8
        assert selection["reset_guard_standard_errors"] == 3.0
        assert selection["reset_relative_contrast_step"] == 0.095849
        assert selection["forward_logit_gain"] == selection_gain
        if training_type == "ibm_reram_om_program_verify":
            training = spec.settings.weight_modifier.parameters
            assert training["assignment_seed"] == 84001
            assert training["endpoint_seed"] == 84002
            assert training["reset_relative_mode"] == training_mode


def test_frozen_reset_relative_heldout_configs_use_one_untouched_assignment() -> None:
    paths = sorted(_RESET_RELATIVE_CONFIG_ROOT.glob("heldout_*.json"))
    assert len(paths) == 10
    observed: dict[str, set[int]] = {"continuous": set(), "quantized_9_level": set()}
    for path in paths:
        _definition, spec = resolve_experiment_config(path, RunMode.VALIDATE)
        parameters = spec.settings.weight_modifier.parameters
        assert parameters["execution"] == "pulse_resolved"
        assert parameters["assignment_seed"] == 85001
        assert parameters["target_mapping"] == "shared_reset_relative_quad"
        assert parameters["reset_read_samples"] == 8
        assert parameters["reset_guard_standard_errors"] == 3.0
        observed[parameters["reset_relative_mode"]].add(
            parameters["endpoint_seed"]
        )
    assert observed == {
        "continuous": set(range(85101, 85106)),
        "quantized_9_level": set(range(85101, 85106)),
    }


def test_reset_relative_study_declares_development_and_heldout_coverage() -> None:
    study = load_study_plan(_RESET_RELATIVE_STUDY)
    assert study["study_id"] == (
        "mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1"
    )
    assert [arm["mode"] for arm in study["arms"]] == [
        "train",
        "train",
        "train",
        "train",
        "validate",
        "validate",
        "validate",
        "validate",
    ]
    assert [len(arm["configs"]) for arm in study["arms"]] == [
        1,
        1,
        1,
        1,
        5,
        5,
        5,
        5,
    ]


def test_accuracy_selection_is_strict_and_keeps_earliest_tie() -> None:
    incumbent = {"student_accuracy": 0.75, "kl_teacher_student": 0.2}
    assert not _selection_improved(
        {"student_accuracy": 0.75},
        incumbent,
        metric="student_accuracy",
    )
    assert _selection_improved(
        {"student_accuracy": 0.751},
        incumbent,
        metric="student_accuracy",
    )
    assert _selection_improved(
        {"kl_teacher_student": 0.19},
        incumbent,
        metric="kl_teacher_student",
    )


def test_modifier_evaluation_reports_applied_forward_gain_and_restores_master() -> None:
    class Cost:
        def __init__(self) -> None:
            self.gain = 4.5

        def set_teacher(self, teacher_logits, labels) -> None:
            del teacher_logits, labels

        def student_logits(self) -> torch.Tensor:
            raw = torch.zeros((2, 10), dtype=torch.float32)
            raw[0, 0] = 1.0
            raw[1, 1] = 1.0
            return raw * self.gain

    modifier = IbmReramHwaParameterModifier.__new__(
        IbmReramHwaParameterModifier
    )
    modifier._config = SimpleNamespace(forward_logit_gain=562.0)
    modifier.evaluation_context = lambda: nullcontext()
    stack = SimpleNamespace(
        device=torch.device("cpu"),
        cost=Cost(),
        network=SimpleNamespace(set_input=lambda *args, **kwargs: None),
        minimizer=SimpleNamespace(compute_equilibrium=lambda: None),
    )
    teacher = SimpleNamespace(
        logits=lambda inputs: torch.zeros((inputs.shape[0], 10))
    )
    report = _evaluate(
        stack,
        teacher,
        [(torch.zeros((2, 784)), torch.tensor([0, 1]))],
        modifier=modifier,
    )
    assert report["fixed_logit_gain"] == 562.0
    assert stack.cost.gain == 4.5


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("common_window_margin_fraction", 0.1, "equal 0.0"),
        ("reset_relative_mode", "round", "continuous"),
        ("reset_relative_contrast_step", 0.0, r"\(0, 0.5\]"),
        ("reset_read_samples", 1, ">= 2"),
        ("reset_guard_standard_errors", -1.0, ">= 0.0"),
    ],
)
def test_shared_reset_relative_config_rejects_invalid_contract(
    field: str,
    replacement,
    match: str,
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (_COMMON_WINDOW_CONFIG_ROOT / "exact_bounds_hwa.json").read_text()
    )
    for role in ("weight_modifier", "selection_weight_modifier"):
        parameters = payload["modes"]["train"][role]["parameters"]
        _set_shared_reset_relative_mapping(parameters)
        parameters[field] = replacement
    path = tmp_path / f"invalid-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        resolve_experiment_config(path, RunMode.TRAIN)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("assignment_seed", 84009),
        ("corruption_policy", "published"),
        ("target_mapping", "literal_global"),
        ("common_window_margin_fraction", 0.2),
    ],
)
def test_differential_pair_hwa_requires_matched_fixed_array_fields(
    field: str,
    replacement,
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    selection = payload["modes"]["train"]["selection_weight_modifier"][
        "parameters"
    ]
    selection[field] = replacement
    if field == "target_mapping":
        selection["common_window_margin_fraction"] = 0.0
        selection["corruption_policy"] = "published"
    path = tmp_path / f"mismatch-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    expected = "fixed array" if field != "corruption_policy" else "counterfactual_repaired"
    with pytest.raises(ConfigError, match=expected):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_differential_pair_margin_rejects_negative_value(
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    for role in ("weight_modifier", "selection_weight_modifier"):
        payload["modes"]["train"][role]["parameters"][
            "common_window_margin_fraction"
        ] = -0.01
    path = tmp_path / "negative-margin.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="finite number"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_differential_pair_validation_requires_repaired_pulse_deployment(
    tmp_path: Path,
) -> None:
    payload = _differential_pair_hwa_payload()
    selection = payload["modes"]["train"]["selection_weight_modifier"]
    payload["modes"] = {
        "validate": {
            "split": "test",
            "sample_limit": 1600,
            "weight_modifier": selection,
            "noise_repeats": 1,
        }
    }
    path = tmp_path / "pair-validate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    _definition, spec = resolve_experiment_config(path, RunMode.VALIDATE)
    modifier = spec.settings.weight_modifier
    assert modifier.parameters["execution"] == "pulse_resolved"
    assert modifier.parameters["corruption_policy"] == "counterfactual_repaired"
    assert (
        modifier.parameters["target_mapping"]
        == "differential_pair_common_window"
    )

    selection["parameters"]["corruption_policy"] = "published"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="counterfactual_repaired"):
        resolve_experiment_config(path, RunMode.VALIDATE)


def _fake_ibm_modifier(
    *,
    target_mapping: str,
    fingerprint: str,
    report: dict,
) -> IbmReramHwaParameterModifier:
    modifier = IbmReramHwaParameterModifier.__new__(
        IbmReramHwaParameterModifier
    )
    modifier._config = SimpleNamespace(target_mapping=target_mapping)
    modifier._population = SimpleNamespace(fingerprint=fingerprint)
    modifier.preflight_target_mapping = lambda: (None, report)
    return modifier


def test_runtime_pair_mapping_requires_population_parity_and_persists_preflights() -> None:
    training_report = {"target_mapping": "differential_pair_common_window"}
    selection_report = {
        "target_mapping": "differential_pair_common_window",
        "role": "selection",
    }
    training = _fake_ibm_modifier(
        target_mapping="differential_pair_common_window",
        fingerprint="fixed-pair-array",
        report=training_report,
    )
    selection = _fake_ibm_modifier(
        target_mapping="differential_pair_common_window",
        fingerprint="fixed-pair-array",
        report=selection_report,
    )
    split = SplitParameterModifier(
        training=training,
        evaluation=selection,
    )

    _validate_array_specific_modifier_population_parity(split)
    assert _ibm_target_mapping_preflights(split) == {
        "training": training_report,
        "selection": selection_report,
    }

    selection._population = SimpleNamespace(fingerprint="different-array")
    with pytest.raises(RuntimeError, match="same fixed population fingerprint"):
        _validate_array_specific_modifier_population_parity(split)


def test_runtime_literal_mapping_remains_outside_array_specific_preflights() -> None:
    literal = _fake_ibm_modifier(
        target_mapping="literal_global",
        fingerprint="literal-array",
        report={"unexpected": True},
    )
    _validate_array_specific_modifier_population_parity(literal)
    assert _ibm_target_mapping_preflights(literal) == {}


def test_runtime_applies_and_restores_declared_device_forward_gain() -> None:
    modifier = IbmReramHwaParameterModifier.__new__(
        IbmReramHwaParameterModifier
    )
    modifier._config = SimpleNamespace(forward_logit_gain=562.0)
    stack = SimpleNamespace(cost=SimpleNamespace(gain=4.5))

    with _modifier_forward_gain(stack, modifier, evaluation=True):
        assert stack.cost.gain == 562.0
    assert stack.cost.gain == 4.5


@pytest.mark.parametrize("config_name", ["clean.json", "exact_bounds_hwa.json"])
def test_common_window_selected_checkpoint_thaws_frozen_modifier_metadata(
    config_name: str,
    tmp_path: Path,
) -> None:
    _definition, spec = resolve_experiment_config(
        _COMMON_WINDOW_CONFIG_ROOT / config_name,
        RunMode.TRAIN,
    )
    frozen_layout = spec.settings.selection_weight_modifier.parameters[
        "dual_rail_layout_by_parameter"
    ]
    with pytest.raises(TypeError):
        frozen_layout["base.dense_weight.0"] = "paired"

    cpu_spec = replace(spec, runtime=replace(spec.runtime, device="cpu"))
    stack = build_student_stack(cpu_spec, enable_measured=False)
    selected = _selected_payload(
        stack,
        spec=cpu_spec,
        teacher_path=tmp_path / "teacher.pt",
        teacher_sha256="teacher-sha",
        mapping=None,
        deployment_source=None,
        device_model_sha256="device-model-sha",
        epoch=-1,
        validation={
            "kl_teacher_student": 0.5,
            "student_accuracy": 0.25,
            "teacher_agreement": 0.3,
        },
    )

    metadata = selected["metadata"]
    json.dumps(metadata, allow_nan=False)
    serialized_layout = metadata["selection_weight_modifier"]["parameters"][
        "dual_rail_layout_by_parameter"
    ]
    assert type(serialized_layout) is dict
    assert serialized_layout == dict(frozen_layout)
    if config_name == "exact_bounds_hwa.json":
        assert type(
            metadata["weight_modifier"]["parameters"][
                "dual_rail_layout_by_parameter"
            ]
        ) is dict


def test_quad_mapping_fields_are_all_or_none_and_margin_is_strict(
    tmp_path: Path,
) -> None:
    payload = json.loads((_CONFIG_ROOT / "clean.json").read_text())
    parameters = payload["modes"]["train"]["selection_weight_modifier"][
        "parameters"
    ]
    parameters["target_mapping"] = "dual_rail_quad_common_window"
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="all three target-mapping fields"):
        resolve_experiment_config(path, RunMode.TRAIN)

    _set_quad_mapping(parameters, margin=0.5)
    parameters["corruption_policy"] = "counterfactual_repaired"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\[0, 0.5\)"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_cli_forwards_explicit_device_model_to_train_and_validate(
    tmp_path: Path,
) -> None:
    train_seen: list[TrainRequest] = []
    validate_seen: list[ValidateRequest] = []
    teacher = tmp_path / "teacher.pt"
    weights = tmp_path / "weights.pt"
    device_model = tmp_path / "device.json"
    for path in (teacher, weights, device_model):
        path.write_bytes(b"fixture")

    assert main(
        [
            "train",
            "--config",
            str(_CONFIG_ROOT / "clean.json"),
            "--output-dir",
            str(tmp_path / "train"),
            "--teacher-weights",
            str(teacher),
            "--device-model",
            str(device_model),
        ],
        handlers=CommandHandlers(train=lambda request: train_seen.append(request) or 0),
        stdout=io.StringIO(),
    ) == 0
    assert train_seen[0].device_model == device_model

    assert main(
        [
            "validate",
            "--config",
            str(_CONFIG_ROOT / "heldout_assignment_83101.json"),
            "--output-dir",
            str(tmp_path / "validate"),
            "--weights",
            str(weights),
            "--teacher-weights",
            str(teacher),
            "--device-model",
            str(device_model),
        ],
        handlers=CommandHandlers(
            validate=lambda request: validate_seen.append(request) or 0
        ),
        stdout=io.StringIO(),
    ) == 0
    assert validate_seen[0].device_model == device_model
