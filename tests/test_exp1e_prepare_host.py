from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "campaigns/opt-cmo-corruption/launch/exp1e_prepare_host.py"
SPEC = importlib.util.spec_from_file_location("exp1e_prepare_host", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_host_preparation_emits_current_pinned_grid_and_only_dry_runs(tmp_path, monkeypatch):
    source, base = tmp_path / "checkout", tmp_path / "host"
    config_root = source / "campaigns/opt-cmo-corruption/lifecycles"
    config_root.mkdir(parents=True)
    cells = [prepare.development.lifecycle(rate, schedule)
             for rate in prepare.development.RATES for schedule in prepare.development.SCHEDULES]
    cells += [prepare.development.lifecycle(1e-4, schedule, smoke=True) for schedule in prepare.development.SCHEDULES]
    for lifecycle in cells:
        (config_root / (lifecycle.lifecycle_id + ".json")).write_text(json.dumps(lifecycle.to_dict()))
    teacher = base / "inputs/opt-125m-27dcfa74/pytorch_model.bin"
    corpus = base / "inputs/corpora/opt125m-synthetic-50m.pt"
    for path in (teacher, corpus):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture")
    original_sha = prepare.sha256

    def fixture_sha(path):
        if path == teacher:
            return prepare.PINNED["facebook/opt-125m"]["sha256"]
        if path == corpus:
            return cells[0].data.corpus_sha256
        return original_sha(path)

    monkeypatch.setattr(prepare, "sha256", fixture_sha)
    commands = []

    def fake_command(argv, *, cwd, env=None, logs=None):
        commands.append(argv)
        if argv[:2] == ["git", "status"]:
            return ""
        if argv[:2] == ["git", "rev-parse"]:
            return "a" * 40
        assert env["EBL_CORPUS_ROOT"] == str(corpus.parent)
        assert env["EBL_CAMPAIGN_STREAM"] == "1"
        if argv[1:4] == ["-m", "workflow", "plan"]:
            target = Path(argv[argv.index("--output") + 1])
            target.mkdir(parents=True, exist_ok=True)
            config = json.loads(Path(argv[4]).read_text())
            stages = []
            for name, dependencies in (("prepare", []), ("devices", ["prepare"]),
                                       ("hwa__standard_hwa_model", ["prepare", "devices"])):
                stage_path = target / f"{name}.json"
                stage_path.write_text(json.dumps({"stage": name, "lifecycle": config}))
                stages.append({"id": name, "case_id": name, "target": "local", "command": "train",
                               "config": stage_path.name, "depends_on": dependencies,
                               "inputs": {"teacher_weights": {"path": str(teacher)}}})
            manifest = {"schema_version": 1, "campaign_id": config["lifecycle_id"],
                        "targets": [{"id": "local", "worktree": str(source), "python": argv[0]}],
                        "stages": stages}
            (target / "campaign.json").write_text(json.dumps(manifest))
        else:
            assert argv[1:5] == ["-m", "ebl", "campaign", "run"]
            assert "--dry-run" in argv
            assert "--allow-dirty" not in argv
        return "{}"

    monkeypatch.setattr(prepare, "command", fake_command)
    record = prepare.prepare_host(base, source=source)
    count = len(prepare.development.RATES) * len(prepare.development.SCHEDULES) + 2
    assert len(record["cases"]) == count
    assert record["launched"] is False
    assert len([c for c in commands if "--dry-run" in c]) == count
    assert len([c for c in commands if c[1:4] == ["-m", "workflow", "plan"]]) == count
    assert sum(case["smoke"] for case in record["cases"]) == 2
    for case in record["cases"]:
        spec = json.loads(Path(case["campaign_worker_spec"]).read_text())
        assert spec["timeout_seconds"] == (1800 if case["smoke"] else 5400)
        assert spec["lease_seconds"] == 900
        assert spec["env"]["EBL_CAMPAIGN_STREAM"] == "1"
        assert spec["env"]["EBL_CORPUS_ROOT"] == str(corpus.parent)
        assert case["queue_run_dir"] != case["campaign_result_root"]
        assert Path(case["campaign_result_root"]).parent == Path(case["native_run_root"])
        assert not Path(case["queue_run_dir"]).exists(), "preparation must not create a worker"
        assert not Path(case["native_run_root"]).exists(), "preparation must not start native execution"
        assert len(case["prepared_files"]) == 6
    assert prepare.prepare_host(base, source=source) == record


def test_host_preparation_rejects_dirty_checkout_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "command", lambda *args, **kwargs: " M workflow/hwa.py\n")
    with pytest.raises(ValueError, match="clean committed"):
        prepare.prepare_host(tmp_path, source=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_input_verification_rejects_missing_and_changed_bytes(tmp_path):
    path = tmp_path / "input"
    with pytest.raises(ValueError, match="Missing required"):
        prepare.checked_input(path, "0" * 64)
    path.write_bytes(b"original")
    digest = prepare.sha256(path)
    assert prepare.checked_input(path, digest)["sha256"] == digest
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare.checked_input(path, digest)
