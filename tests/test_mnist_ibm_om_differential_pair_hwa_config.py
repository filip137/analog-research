from __future__ import annotations

import json
from pathlib import Path

import pytest

from ebl.registry import resolve_experiment_config
from ebl.schema.base import ConfigError, RunMode
from ebl.lifecycle.study import load_study_plan


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_differential_pair_hwa_pilot"
)
_STUDY = (
    _ROOT
    / "studies"
    / "mnist-ibm-om-differential-pair-hwa-program-verify-pilot-"
    "20260824-v1.json"
)
_SOURCE_SHA = (
    "5a9dece30a938f00d2022de3e0aa57755e17f754c2df249507dd709bf9394fbc"
)
_TEACHER_SHA = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
_DEVICE_SHA = (
    "3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3"
)
_FINGERPRINT = (
    "d0dfae135fa7b741c46f1178e52f645da4593f7313b4e6a0a71a9b3793d1b2f0"
)


def _modifier(parameters, *, execution: str, endpoint_seed: int) -> None:
    assert parameters["execution"] == execution
    assert parameters["assignment_seed"] == 84001
    assert parameters["endpoint_seed"] == endpoint_seed
    assert parameters["corruption_policy"] == "counterfactual_repaired"
    assert parameters["noisy_evaluation"] is (
        execution == "pulse_resolved"
    )
    assert parameters["target_mapping"] == (
        "differential_pair_common_window"
    )
    assert parameters["dual_rail_layout_by_parameter"] is None
    assert parameters["common_window_margin_fraction"] == 0.25
    assert parameters["start_protocol"] == "lower_to_target"
    assert parameters["controller"] == "adaptive"
    assert parameters["maximum_program_pulses"] == 128
    assert parameters["target_out_of_support"] == "error"


def test_two_configs_are_internally_matched_canonical_differential_arms() -> None:
    specs = {}
    for name in ("clean.json", "exact_bounds_hwa.json"):
        definition, spec = resolve_experiment_config(
            _CONFIG_ROOT / name,
            RunMode.TRAIN,
        )
        assert definition.experiment_id == "mnist_relu_drn_kd.v1"
        specs[name] = spec

    clean = specs["clean.json"]
    hwa = specs["exact_bounds_hwa.json"]
    for field in ("runtime", "data", "teacher", "model", "solver", "mapping"):
        assert getattr(clean, field) == getattr(hwa, field)
    assert clean.runtime.seed == clean.runtime.data_seed == 42
    assert clean.model.encoding == "differential"
    assert clean.model.conductance_min == 0.0
    assert clean.model.conductance_max == 0.00011
    assert clean.teacher.type == "bias_free_relu"
    assert clean.teacher.initialization == "signed_weight_mapping"
    assert clean.settings.num_epochs == hwa.settings.num_epochs == 10
    assert clean.settings.learning_rates == hwa.settings.learning_rates == (
        4.2e-10,
        1.14e-12,
    )
    assert clean.settings.update_backend == hwa.settings.update_backend
    assert clean.settings.weight_modifier.type == "none"
    assert hwa.settings.weight_modifier.type == (
        "ibm_reram_om_program_verify"
    )
    _modifier(
        hwa.settings.weight_modifier.parameters,
        execution="compact_endpoint",
        endpoint_seed=84002,
    )
    for spec in (clean, hwa):
        assert spec.settings.selection_evaluation == "modifier"
        assert spec.settings.selection_noise_repeats == 1
        assert spec.settings.selection_weight_modifier.type == (
            "ibm_reram_om_program_verify"
        )
        _modifier(
            spec.settings.selection_weight_modifier.parameters,
            execution="pulse_resolved",
            endpoint_seed=84003,
        )


def test_pair_config_rejects_noncanonical_encoding_or_layout(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        (_CONFIG_ROOT / "exact_bounds_hwa.json").read_text(encoding="utf-8")
    )
    payload["model"]["encoding"] = "single"
    path = tmp_path / "wrong-encoding.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="model.encoding='differential'"):
        resolve_experiment_config(path, RunMode.TRAIN)

    payload = json.loads(
        (_CONFIG_ROOT / "exact_bounds_hwa.json").read_text(encoding="utf-8")
    )
    payload["modes"]["train"]["weight_modifier"]["parameters"][
        "dual_rail_layout_by_parameter"
    ] = {"base.dense_weight.0": "halves"}
    path = tmp_path / "wrong-layout.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="to be null"):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_study_freezes_inputs_pair_fixture_and_within_eight_claim() -> None:
    study = load_study_plan(_STUDY)
    assert study["study_id"].endswith("differential-pair-hwa-program-verify-pilot-20260824-v1")
    assert study["evidence_class"] == "model_based_aihwkit_preset"
    assert [arm["arm_id"] for arm in study["arms"]] == [
        "train-clean-differential-pair",
        "train-exact-bounds-pair-hwa",
    ]
    joined = "\n".join(
        [study["motivation"], *study["completion_criteria"], *study["analysis_plan"]]
    )
    for digest in (_SOURCE_SHA, _TEACHER_SHA, _DEVICE_SHA, _FINGERPRINT):
        assert digest in joined
    assert "317,600 cells" in joined
    assert "158,800" in joined
    assert "exactly eight empty" in joined
    assert "317,584 compact" in joined
    assert "exactly 16" in joined
    assert "pulse_resolved_noncorrupt_out_of_bound_empty_pair_only" in joined
    assert "--weights" in joined
    assert "--teacher-weights" in joined
    assert "unmatched mechanistic reference" in joined
    assert "Do not calculate a causal four-versus-eight effect" in joined
    assert "contains no on-chip recovery arm" in joined


def test_readme_and_protocol_record_current_partial_evidence() -> None:
    readme = (_CONFIG_ROOT / "README.md").read_text(encoding="utf-8")
    protocol = (
        _ROOT
        / "campaigns/ibm-om-deployment-recovery/series/001-common-window"
        / "experiments/exp-003-eight-device.md"
    ).read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    normalized_protocol = " ".join(protocol.split())
    assert "internally matched canonical eight-device pilot" in normalized_readme
    assert "do not treat their results as a matched comparison" in normalized_readme
    assert (
        "Status: interrupted partial evidence; the clean arm completed"
        in normalized_protocol
    )
    assert (
        "not a matched comparison with the completed four-device v2"
        in normalized_protocol
    )
    assert _FINGERPRINT in protocol
