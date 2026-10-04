from __future__ import annotations

import json
from pathlib import Path

from experiments.mnist_analog_relu.generate_staged_campaign import (
    CONFIG_ROOT,
    DEVELOPMENT_PLAN_ID,
    PRODUCTION_ENDPOINTS,
    PRODUCTION_PLAN_ID,
    ROOT,
    TEACHER_PLAN_ID,
    generate,
)
from experiments.mnist_analog_relu.staged_config import (
    StagedCrossbarTrainSpec,
    parse_staged_crossbar_config,
    resolve_staged_crossbar_spec,
)
from experiments.mnist_analog_relu.staged_analysis import (
    PRODUCTION_ENDPOINTS as ANALYSIS_PRODUCTION_ENDPOINTS,
)
from experiments.schema import RunMode
from experiments.study_workflow import load_study_plan


def test_generated_campaign_is_byte_stable_and_all_configs_are_strict() -> None:
    before = {
        path: path.read_bytes()
        for path in CONFIG_ROOT.rglob("*.json")
    }
    plans = generate()
    after = {
        path: path.read_bytes()
        for path in CONFIG_ROOT.rglob("*.json")
    }
    assert before == after
    assert set(plans) == {"teacher", "development", "production"}
    # One config per scientific run prevents source/start-state miswiring:
    # 29 tuning/HWA configs plus 160 production configs.
    assert len(after) == 189
    for path, payload in after.items():
        raw = json.loads(payload)
        assert not any(
            token in json.dumps(raw)
            for token in ("teacher_weights_path", "device_state_path", "output_dir")
        )
        spec = resolve_staged_crossbar_spec(
            parse_staged_crossbar_config(raw),
            RunMode.TRAIN,
        )
        assert isinstance(spec, StagedCrossbarTrainSpec), path


def test_three_formal_plans_declare_exact_native_run_matrix() -> None:
    teacher = load_study_plan(ROOT / "studies" / f"{TEACHER_PLAN_ID}.json")
    development = load_study_plan(ROOT / "studies" / f"{DEVELOPMENT_PLAN_ID}.json")
    production = load_study_plan(ROOT / "studies" / f"{PRODUCTION_PLAN_ID}.json")
    assert [(arm["arm_id"], arm["mode"]) for arm in teacher["arms"]] == [
        ("teacher", "train"),
        ("teacher-test", "validate"),
    ]
    assert sum(len(arm["configs"]) for arm in development["arms"]) == 29
    assert sum(len(arm["configs"]) for arm in production["arms"]) == 160
    assert len(production["arms"]) == 10


def test_hwa_generation_is_validation_only_and_apparent_primary() -> None:
    payload = json.loads((CONFIG_ROOT / "offchip_hwa.json").read_text())
    assert payload["evaluation"] == {"profile": "tuning_validation_only"}
    assert payload["stage"]["evaluation_forward_policy"] == (
        "one_sampled_held_apparent_q_per_evaluation"
    )
    assert payload["stage"]["forward_noise"]["base_state"] == (
        "support_clamped_digital_master_q_surrogate"
    )
    plan = load_study_plan(ROOT / "studies" / f"{DEVELOPMENT_PLAN_ID}.json")
    serialized = json.dumps(plan)
    assert "never evaluates the official test set" in serialized
    assert "nonpersistent diagnostic" in serialized


def test_production_seed_matrix_is_four_assignments_by_four_writes() -> None:
    assert PRODUCTION_ENDPOINTS == ANALYSIS_PRODUCTION_ENDPOINTS
    assert len(PRODUCTION_ENDPOINTS) == 4
    assert all(len(set(endpoints)) == 4 for endpoints in PRODUCTION_ENDPOINTS.values())
    assert len({seed for values in PRODUCTION_ENDPOINTS.values() for seed in values}) == 16
    deploy_configs = sorted((CONFIG_ROOT / "production").glob("deploy_teacher_*.json"))
    observed = {
        (
            json.loads(path.read_text())["device"]["assignment_seed"],
            json.loads(path.read_text())["device"]["endpoint_seed"],
        )
        for path in deploy_configs
    }
    expected = {
        (assignment, endpoint)
        for assignment, endpoints in PRODUCTION_ENDPOINTS.items()
        for endpoint in endpoints
    }
    assert observed == expected
