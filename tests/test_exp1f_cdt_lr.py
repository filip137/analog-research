"""Frozen selection and restart-safe conditional deployment for the CDT sweep."""
import importlib.util
import json
from pathlib import Path

import pytest

from workflow.lifecycle import provenance, verify_provenance

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp1f_cdt_lr", ROOT / "campaigns/opt-cmo-corruption/launch/exp1f_cdt_lr.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def grid(zero=.08, small=.07, large=.071):
    return [{"label": label, "digital_lr": rate, "mean_kl": metric}
            for label, rate, metric in [("0", 0, zero), ("1em5", 1e-5, small), ("1em4", 1e-4, large)]]


def test_selection_honors_one_percent_tie_and_never_uses_test_scores():
    assert study.choose(grid())["label"] == "1em5"
    assert study.choose(grid(zero=.0706))["label"] == "0"
    assert study.choose(grid(large=.05))["label"] == "1em4"
    rows = grid()
    rows[0]["test_kl"] = 0
    rows[1]["test_kl"] = 100
    assert study.choose(rows)["label"] == "1em5"


@pytest.mark.parametrize("rows", [grid()[:2], grid() + [grid()[0]], grid(small=float("nan")), grid(large=-1)])
def test_incomplete_or_invalid_grid_cannot_select(rows):
    with pytest.raises(ValueError):
        study.choose(rows)


def test_sources_keep_fixed_protocol_and_transfer_adds_no_training_change():
    for label in study.RATES:
        source = study.lifecycle(label)
        transfer = study.lifecycle(label, "transfer")
        assert source.hwa.lr_schedule.total_steps == 25600
        assert source.hwa.lr_schedule.warmup_steps == 205
        assert source.hwa.learning_rate == .005
        assert source.hwa.lr_schedule.factor(25600) == .2
        assert source.hwa.seed == 41
        assert source.transfer_tests is None
        arm = source.to_dict()["hwa"]["arms"][0]
        assert arm["corruption"] == {"kind": "random", "rates_ppm": [30000]}
        assert arm["scope"] == "model" and arm["digital_learning_rate"] == study.RATES[label]
        verify_provenance(provenance(source, "hwa_source", hwa_arm=study.ARM), transfer, "hwa_source", hwa_arm=study.ARM)
    assert study.lifecycle("1em5", "smoke").hwa.lr_schedule.total_steps == 512


@pytest.mark.parametrize("label", ["0", "1em5", "1em4"])
def test_completion_replay_releases_only_winner_once(tmp_path, monkeypatch, label):
    evidence = tmp_path / "evidence"
    (evidence / "prepared").mkdir(parents=True)
    (evidence / "collected").mkdir()
    for host in study.HOSTS.values():
        (evidence / "prepared" / (host + ".json")).write_text(json.dumps({"source": "/remote/frozen"}))
    cfg = tmp_path / "queue.json"
    cfg.write_text(json.dumps({"targets": {host: {"host": host} for host in study.HOSTS.values()}}))
    selected = {"coverage_complete": True, "selected_label": label,
                "selected_source": {"source_sha256": "a" * 64}, "test_data_used": False}
    monkeypatch.setattr(study, "selection", lambda _: selected)
    monkeypatch.setattr(study, "transfer_review", lambda *_: {"pending": label != "0"})
    calls, gates = [], []
    states = {study.lifecycle(l, "transfer").lifecycle_id: "held" for l in study.RATES}
    def transition(_cfg, action, job):
        if states[job] == "held":
            calls.append((action, job))
            states[job] = "queued" if action == "release" else "cancelled"
    monkeypatch.setattr(study, "transition", transition)
    monkeypatch.setattr(study, "gate", lambda cfg, evidence, name, payload: gates.append(name))
    monkeypatch.setattr(study.subprocess, "run", lambda *a, **kw: None)
    for _ in range(2):
        study.advance(cfg, evidence, study.lifecycle("1em4").lifecycle_id, tmp_path)
    assert len(calls) == 2
    assert [job for action, job in calls if action == "release"] == ([] if label == "0" else [study.lifecycle(label, "transfer").lifecycle_id])
    assert gates == ([] if label == "0" else ["selection_" + label] * 2)
    assert json.loads((evidence / "selection.json").read_text()) == selected


def test_partial_collection_leaves_deployments_held(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    (evidence / "collected").mkdir(parents=True)
    monkeypatch.setattr(study, "selection", lambda _: {"coverage_complete": False})
    monkeypatch.setattr(study, "transition", lambda *a: pytest.fail("partial coverage advanced dispatch"))
    assert not study.advance(tmp_path / "queue.json", evidence, study.lifecycle("1em5").lifecycle_id, tmp_path)["coverage_complete"]
    assert not (evidence / "selection.json").exists()


def test_deployment_refuses_unselected_checkpoint_before_gpu_work(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected_label": "1em4", "test_data_used": False}))
    with pytest.raises(ValueError, match="does not match"):
        study.execute_deployment("1em5", path)
