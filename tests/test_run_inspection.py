from io import StringIO
import json
from pathlib import Path

import pytest

from ebl.cli import main
from experiments.artifacts import RunStore
from experiments.study_workflow import inspect_run


def make_run(tmp_path, state="complete"):
    store = RunStore.create(output_root=tmp_path / "runs", experiment_id="small_drn.v1",
        resolved_config={"seed": 7}, command=["ebl", "train", "--config", "example.json"],
        repo_root=tmp_path, run_id="case-001")
    if state == "complete":
        store.append_metric({"loss": 0.25})
        artifact = store.run_dir / "artifacts" / "value.bin"
        artifact.write_bytes(b"real artifact")
        store.complete(metrics={"loss": 0.25}, artifacts=[store.artifact_record(artifact, kind="test")])
    elif state == "failed":
        store.fail(RuntimeError("recorded incident"))
    return store.run_dir


def call(*args):
    out, err = StringIO(), StringIO()
    code = main(["runs", "inspect", *map(str, args), "--json"], stdout=out, stderr=err)
    return code, json.loads(out.getvalue())


def test_standalone_inspection_is_read_only(tmp_path):
    run = make_run(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    code, records = call(run, "--verify-artifacts", "--require-complete")
    assert code == 0
    assert records[0]["valid"] and records[0]["process_complete"]
    assert records[0]["metrics"] == {"loss": 0.25}
    assert records[0]["validation_mode"] == "full_artifact_hashes"
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("state", ["running", "failed"])
def test_valid_attempt_is_not_completion(tmp_path, state):
    run = make_run(tmp_path, state)
    assert call(run)[0] == 0
    code, records = call(run, "--require-complete")
    assert code == 1 and records[0]["valid"]
    assert records[0]["status"] == state
    assert not records[0]["process_complete"]


def test_artifact_tampering_requires_hash_audit(tmp_path):
    run = make_run(tmp_path)
    (run / "artifacts/value.bin").write_bytes(b"fake artifact")
    assert inspect_run(run)["valid"]
    record = inspect_run(run, verify_artifacts=True)
    assert not record["valid"]
    assert "SHA-256" in " ".join(record["errors"])


@pytest.mark.parametrize("damage", ["config", "result", "missing", "escape", "status"])
def test_invalid_evidence_is_detected(tmp_path, damage):
    run = make_run(tmp_path)
    if damage == "config":
        (run / "config.resolved.json").write_text('{"seed":8}')
    elif damage == "missing":
        (run / "artifacts/value.bin").unlink()
    else:
        path = run / ("status.json" if damage == "status" else "result.json")
        value = json.loads(path.read_text())
        if damage == "result":
            value["experiment_id"] = "wrong.experiment"
        elif damage == "escape":
            value["artifacts"][0]["path"] = "../outside.bin"
        else:
            value["status"] = {}
        path.write_text(json.dumps(value))
    assert call(run, "--require-complete")[0] == 1


def test_missing_case_does_not_hide_valid_case(tmp_path):
    run = make_run(tmp_path)
    code, records = call(run, tmp_path / "missing", "--require-complete")
    assert code == 1 and len(records) == 2
    assert records[0]["valid"] and not records[1]["valid"]
