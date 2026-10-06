"""End-to-end lifecycle stages through the native runtime on synthetic inputs.

Teacher loading, CIFAR caching and AIHWKit sampling are replaced by synthetic
equivalents; every stage still creates a native bundle, checks upstream
provenance and is validated with the campaign runner's result contract.
"""

import json
from pathlib import Path

import pytest
import torch

from campaigns.runner import _validate_run_result
from ebl.cli import TrainRequest
from experiments.definitions import get_definition
from workflow import data as data_stage
from workflow import deployment, runtime
from workflow import devices as device_stage
from workflow.__main__ import main as workflow_main
from workflow.lifecycle import (
    EXPERIMENT_ID,
    Execution,
    StageConfig,
    StageSelection,
    parse_lifecycle,
    stage_inputs,
    stage_selections,
)
from workflow_helpers import lifecycle_document, network_for, om_raw, synthetic_cache


def smoke(name, **changes):
    doc = lifecycle_document(name)
    doc["network"]["analog_convolutions"] = 0
    doc["data"]["max_examples"] = 16
    doc["hwa"]["batch_size"] = 8
    doc["onchip"]["batch_size"] = 8
    if doc["devices"]["technology"] == "om":
        doc["devices"]["characterization"].update(bins=5, samples=16)
    for dotted, value in changes.items():
        *parents, key = dotted.split("__")
        target = doc
        for part in parents:
            target = target[part]
        target[key] = value
    return parse_lifecycle(doc)


@pytest.fixture
def teacher_weights(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    teacher, _ = network_for(0)
    monkeypatch.setattr(deployment, "load_teacher", lambda path, network, device: (teacher, "f" * 64))
    monkeypatch.setattr(
        data_stage,
        "build_cache",
        lambda teacher_, network, lc, device: synthetic_cache(teacher_, network, n=lc.data.max_examples),
    )
    monkeypatch.setattr(
        device_stage,
        "native_sample",
        lambda *, kind, size, seed, policy, variation, output: om_raw(size, seed),
    )
    path = tmp_path / "teacher.pt"
    path.write_bytes(b"synthetic teacher")
    return path


def request_for(lc, selection, inputs, output_dir):
    return TrainRequest(
        definition=get_definition(EXPERIMENT_ID),
        spec=StageConfig(stage=selection, execution=Execution("cpu", 1, False), lifecycle=lc),
        config_path=Path("stage.json"),
        output_dir=output_dir,
        weights=inputs.get("weights"),
        resume=None,
        command=("pytest",),
        device_data=inputs.get("device_data"),
        device_model=inputs.get("device_model"),
        teacher_weights=inputs.get("teacher_weights"),
        device_state=inputs.get("device_state"),
    )


def run_stage(lc, selection, results, teacher, root):
    inputs = {}
    for role, upstream in stage_inputs(selection).items():
        if upstream is None:
            inputs[role] = teacher
        else:
            run_dir, result = results[upstream[0]]
            [artifact] = [a for a in result["artifacts"] if a["kind"] == upstream[1]]
            inputs[role] = run_dir / artifact["path"]
    output = root / selection.stage_id
    assert runtime.run_train(request_for(lc, selection, inputs, output)) == 0
    [run_dir] = sorted(output.iterdir())
    result = dict(_validate_run_result(run_dir / "result.json"))
    return run_dir, result


def run_lifecycle(lc, teacher, root):
    results = {}
    for selection in stage_selections(lc):
        results[selection.stage_id] = run_stage(lc, selection, results, teacher, root)
    return results


@pytest.mark.parametrize("name", ["cifar10-pcm-conv4-smoke", "cifar10-pcm-conv4-drift-smoke", "cifar10-om-conv4-smoke"])
def test_lifecycle_runs_end_to_end_with_matched_controls(teacher_weights, tmp_path, name):
    lc = smoke(name)
    results = run_lifecycle(lc, teacher_weights, tmp_path / "runs")
    arms = [arm.arm_id for arm in lc.onchip.arms]
    for hwa_arm in lc.hwa.arms:
        _, source = results[f"hwa__{hwa_arm.arm_id}"]
        assert source["metrics"]["fit"]["status"] == "complete"
        run_dir, result = results[f"onchip__{hwa_arm.arm_id}__271001"]
        rows = json.loads((run_dir / "measurements.json").read_text())
        assert [(r["case"], r["onchip_arm"]) for r in rows] == [
            (case.label, arm) for case in lc.defects.cases for arm in arms
        ]
        assert all(row["replay_exact"] for row in rows)
        for row in rows:
            if row["onchip_arm"] == "none":
                assert row["final"] == row["initial"] and row["curve"] == []
            if row["weights"] == "hold":
                writes = row["cost"].get("physical_pulses", row["cost"].get("array_reprogram_calls", 0))
                assert writes == 0
            else:
                assert row["curve"] and row["image_presentations"] == lc.data.max_examples
        kinds = {a["kind"] for a in result["artifacts"]}
        assert {"mapping", "measurements", "onchip_states"} <= kinds
        assert len(kinds) == len(result["artifacts"]) == 3 + len(rows)
    # The deployment-only and calibration arms share one P0 per assignment and case.
    deploy_dir, deploy = results["deploy__cdt_gmax__271001"]
    assert set(deploy["metrics"]["cases"]) == {case.label for case in lc.defects.cases}
    collected = tmp_path / "collected"
    assert workflow_main(["collect", str(tmp_path / "runs"), "--output", str(collected)]) == 0
    rows = json.loads((collected / "rows.json").read_text())
    assert len(rows) == len(lc.hwa.arms) * len(lc.defects.cases) * len(arms)
    assert not any(row["duplicate_coverage"] for row in rows)


def test_upstream_artifacts_are_reused_only_when_their_sections_match(teacher_weights, tmp_path):
    lc = smoke("cifar10-pcm-conv4-smoke")
    results = run_lifecycle(lc, teacher_weights, tmp_path / "runs")
    selection = StageSelection("onchip", "standard_hwa", 271001)
    # A later lifecycle may change on-chip settings and reuse the same deployment.
    later = smoke("cifar10-pcm-conv4-smoke", onchip__epochs=2, question__decision="A later decision.")
    run_dir, result = run_stage(later, selection, results, teacher_weights, tmp_path / "later")
    assert result["metrics"]["lifecycle_sha256"] != lc.digest()
    # Changing the defects invalidates the deployment's physical identities.
    changed = smoke(
        "cifar10-pcm-conv4-smoke", defects__cases=[{"kind": "none", "rate_ppm": 0}, {"kind": "gmax", "rate_ppm": 30000}]
    )
    with pytest.raises(ValueError, match="defects"):
        run_stage(changed, selection, results, teacher_weights, tmp_path / "changed")
    [failed] = sorted((tmp_path / "changed" / selection.stage_id).iterdir())
    assert json.loads((failed / "status.json").read_text())["status"] == "failed"


def test_stage_inputs_are_exact(teacher_weights, tmp_path):
    lc = smoke("cifar10-pcm-conv4-smoke")
    selection = StageSelection("deploy", "digital", 271001)
    with pytest.raises(ValueError, match="deploy stage inputs"):
        runtime.run_train(request_for(lc, selection, {"teacher_weights": teacher_weights}, tmp_path / "x"))
    assert not (tmp_path / "x").exists()
