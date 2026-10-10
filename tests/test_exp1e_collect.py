"""Real native stages plus collector identity, integrity and concurrency guards."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from experiments.artifacts import sha256_file
from test_workflow_cmo import cmo_inputs, small, model_arm
from workflow.lifecycle import stage_selections
from workflow_helpers import run_stage

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp1e_collect", ROOT / "campaigns/opt-cmo-corruption/launch/exp1e_collect.py")
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


@pytest.fixture
def native(cmo_inputs, tmp_path):
    weights, corpus = cmo_inputs
    lc = small(**{"data.corpus": {"name": "synthetic", "sha256": corpus}, "data.sequence_length": 9,
                  "data.evaluation": "development", "data.test_sequences": 0, "hwa.arms": [model_arm()],
                  "hwa.batch_size": 4, "hwa.lr_schedule": {"kind": "linear_warmup_decay", "total_steps": 2, "warmup_steps": 1},
                  "onchip": None})
    root, results, records = tmp_path / "campaign", {}, []
    for selection in stage_selections(lc):
        if selection.kind not in {"prepare", "devices", "hwa"}:
            continue
        results[selection.stage_id] = run_stage(lc, selection, results, weights, root)
        run, _ = results[selection.stage_id]
        records.append({"stage_id": selection.stage_id, "status": "complete", "run_result": str(run / "result.json"),
                        "run_result_sha256": sha256_file(run / "result.json")})
    (root / "aggregate").mkdir()
    (root / "aggregate/results.json").write_text(json.dumps({"stages": records}))
    return root


def test_real_three_stage_native_export_and_compact_validate(native, capsys):
    payload = collector.export(native)
    assert set(payload["stages"]) == set(collector.STAGES)
    assert collector.checked_payload(payload, native) == payload
    collector.main(["validate", str(native)])
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result == {"valid": True, "campaign_root": str(native), "stages": sorted(collector.STAGES)}
    assert len(json.dumps(result)) < 500


@pytest.fixture
def exported(native):
    return native, collector.export(native)


@pytest.mark.parametrize("failure", ["duplicate_stage", "missing_stage", "digest", "teacher_kl", "missing_teacher_kl",
                                      "source_steps", "source_history", "source_selected", "device", "upstream", "inspection"])
def test_export_gate_rejects_bad_native_evidence(exported, failure):
    root, original = exported
    payload = copy.deepcopy(original)
    aggregate = json.loads(payload["aggregate"])
    if failure == "duplicate_stage":
        aggregate["stages"].append(aggregate["stages"][0])
    elif failure == "missing_stage":
        aggregate["stages"].pop()
    elif failure == "digest":
        aggregate["stages"][0]["run_result_sha256"] = "0" * 64
    elif failure in {"upstream", "inspection"}:
        files = payload["stages"]["hwa__standard_hwa_model"]["files"]
        if failure == "upstream":
            manifest = json.loads(files["manifest.json"])
            for item in manifest["inputs"]:
                if item["role"] == "device_data":
                    item["sha256"] = "0" * 64
            files["manifest.json"] = json.dumps(manifest)
        else:
            inspected = json.loads(files["inspection.json"])
            inspected[0]["validation_mode"] = "metadata_only"
            files["inspection.json"] = json.dumps(inspected)
    else:
        stage = "prepare" if "teacher_kl" in failure else "devices" if failure == "device" else "hwa__standard_hwa_model"
        files = payload["stages"][stage]["files"]
        result = json.loads(files["result.json"])
        metrics = result["metrics"]
        if failure == "teacher_kl":
            metrics["digital"]["teacher_kl"] = 1e-3
        elif failure == "missing_teacher_kl":
            del metrics["digital"]["teacher_kl"]
        elif failure == "source_steps":
            metrics["fit"]["lr_schedule"]["steps_run"] = 1
        elif failure == "source_history":
            metrics["fit"]["history"].pop()
        elif failure == "source_selected":
            metrics["fit"]["development_kl"] = 100
        else:
            metrics["technology"] = "pcm"
        files["result.json"] = json.dumps(result)
        digest = collector.text_sha(files["result.json"])
        next(row for row in aggregate["stages"] if row["stage_id"] == stage)["run_result_sha256"] = digest
        receipt = json.loads(files["artifact-inspection.json"])
        receipt["result_sha256"] = digest
        files["artifact-inspection.json"] = json.dumps(receipt)
    payload["aggregate"] = json.dumps(aggregate)
    with pytest.raises(ValueError):
        collector.checked_payload(payload, root)


def test_native_artifact_corruption_is_detected_by_real_inspector(native):
    records = json.loads((native / "aggregate/results.json").read_text())["stages"]
    record = next(item for item in records if item["stage_id"].startswith("hwa__"))
    run = Path(record["run_result"]).parent
    source = next(item for item in json.loads((run / "result.json").read_text())["artifacts"] if item["kind"] == "hwa_source")
    (run / source["path"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        collector.export(native)


def queue_db(path):
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE jobs(id TEXT, spec TEXT, state TEXT); CREATE TABLE attempts(token TEXT,job_id TEXT,state TEXT,target TEXT);")
        db.execute("INSERT INTO jobs VALUES(?,?,?)", ("case", "{}", "collecting"))
        db.execute("INSERT INTO attempts VALUES(?,?,?,?)", ("one", "case", "complete", "fifi"))


def test_queue_uses_complete_and_refuses_duplicate_or_uncertain_attempts(tmp_path):
    path = tmp_path / "queue.sqlite3"
    queue_db(path)
    assert collector.exact_attempt(path, "case")[1]["token"] == "one"
    for state in ("complete", "running", "unknown"):
        with sqlite3.connect(path) as db:
            db.execute("INSERT INTO attempts VALUES(?,?,?,?)", ("two", "case", state, "loulou"))
        with pytest.raises(ValueError, match="one exact"):
            collector.exact_attempt(path, "case")
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM attempts WHERE token='two'")
    with sqlite3.connect(path) as db:
        db.execute("UPDATE attempts SET state='succeeded'")
    with pytest.raises(ValueError):
        collector.exact_attempt(path, "case")


def test_collection_lock_serializes_publishers_and_frozen_files_reject_replacement(tmp_path):
    entered, finished = threading.Event(), threading.Event()
    def other():
        entered.set()
        with collector.collection_lock(tmp_path):
            finished.set()
    with collector.collection_lock(tmp_path):
        worker = threading.Thread(target=other)
        worker.start()
        assert entered.wait(1)
        assert not finished.wait(0.05)
    worker.join(1)
    assert finished.is_set()
    path = tmp_path / "job/identity.json"
    collector.write_frozen(path, "one")
    collector.write_frozen(path, "one")
    with pytest.raises(ValueError, match="overwrite"):
        collector.write_frozen(path, "two")


@pytest.mark.parametrize("transport", ["ssh", "local"])
def test_collection_publishes_one_immutable_attempt_and_repeats_idempotently(exported, tmp_path, monkeypatch, transport):
    root, payload = exported
    state = tmp_path / "state"
    state.mkdir()
    database = state / "queue.sqlite3"
    queue_db(database)
    spec = {"placements": {"fifi": {"workdir": str(ROOT), "campaign_python": sys.executable,
                                     "campaign_result_root": str(root)}}}
    with sqlite3.connect(database) as db:
        db.execute("UPDATE jobs SET spec=?", (json.dumps(spec),))
    config = tmp_path / "queue.json"
    target = {"transport": transport, **({"host": "fifi", "ssh_options": ["-F", "/tmp/ssh config"]} if transport == "ssh" else {})}
    config.write_text(json.dumps({"state_dir": str(state), "targets": {"fifi": target}}))
    monkeypatch.setenv("RUN_WATCH_JOB_ID", "case")
    calls = []
    def transport_call(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
    monkeypatch.setattr(collector.subprocess, "run", transport_call)
    reviewed = []
    monkeypatch.setattr(collector, "summarize", lambda index, output: reviewed.append(json.loads(index.read_text())))
    output = tmp_path / "collected"
    first = collector.collect(config, output)
    assert collector.collect(config, output) == first
    receipts = list(output.glob("*/*/collection.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["job"] == "case" and receipt["attempt"] == "one"
    assert len(reviewed) == 2 and len(reviewed[0]) == 1
    for command, kwargs in calls:
        assert kwargs["check"] is True and kwargs.get("shell", False) is False
        if transport == "local":
            assert command == [sys.executable, str(ROOT / "campaigns/opt-cmo-corruption/launch/exp1e_collect.py"),
                               "export", str(root)]
            assert kwargs["cwd"] == str(ROOT)
        else:
            assert command[0] == "ssh" and command[-2] == "fifi"
            assert command[5:7] == ["-F", "/tmp/ssh config"]
            assert kwargs["cwd"] is None
    for filename, digest in receipt["files"].items():
        assert sha256_file(receipts[0].parent / filename) == digest
    with sqlite3.connect(database) as db:
        db.execute("UPDATE attempts SET token='replacement'")
    with pytest.raises(ValueError, match="overwrite"):
        collector.collect(config, output)


@pytest.mark.parametrize("corrupt", [False, True])
def test_local_transport_runs_actual_native_export_and_preserves_integrity(native, tmp_path, monkeypatch, corrupt):
    state = tmp_path / "state"
    state.mkdir()
    database = state / "queue.sqlite3"
    queue_db(database)
    placement = {"workdir": str(ROOT), "campaign_python": sys.executable, "campaign_result_root": str(native)}
    with sqlite3.connect(database) as db:
        db.execute("UPDATE attempts SET target='local'")
        db.execute("UPDATE jobs SET spec=?", (json.dumps({"placements": {"local": placement}}),))
    config = tmp_path / "queue.json"
    config.write_text(json.dumps({"state_dir": str(state), "targets": {"local": {"transport": "local"}}}))
    monkeypatch.setenv("RUN_WATCH_JOB_ID", "case")
    monkeypatch.setattr(collector, "summarize", lambda *args: None)
    output = tmp_path / "collected"
    if corrupt:
        records = json.loads((native / "aggregate/results.json").read_text())["stages"]
        source_run = Path(next(row["run_result"] for row in records if row["stage_id"].startswith("hwa__"))).parent
        (source_run / "checkpoints/source.pt").write_bytes(b"corrupt source")
        with pytest.raises(subprocess.CalledProcessError):
            collector.collect(config, output)
        assert not output.exists()
    else:
        receipt = collector.collect(config, output)
        folder = Path(receipt["collected"])
        identity = json.loads((folder / "collection.json").read_text())
        assert identity["target"] == "local" and identity["attempt"] == "one"
        assert len(identity["stages"]) == 3
        assert all(sha256_file(folder / path) == digest for path, digest in identity["files"].items())


@pytest.mark.parametrize("transport", [None, "slurm"])
def test_unsupported_transport_fails_before_execution_or_collection(tmp_path, monkeypatch, transport):
    state = tmp_path / "state"
    state.mkdir()
    database = state / "queue.sqlite3"
    queue_db(database)
    placement = {"workdir": str(ROOT), "campaign_python": sys.executable, "campaign_result_root": "/tmp/example"}
    with sqlite3.connect(database) as db:
        db.execute("UPDATE jobs SET spec=?", (json.dumps({"placements": {"fifi": placement}}),))
    config = tmp_path / "queue.json"
    config.write_text(json.dumps({"state_dir": str(state), "targets": {"fifi": {"transport": transport}}}))
    monkeypatch.setenv("RUN_WATCH_JOB_ID", "case")
    monkeypatch.setattr(collector.subprocess, "run", lambda *args, **kwargs: pytest.fail("Transport must reject before executing"))
    with pytest.raises(ValueError, match="Unsupported collector transport"):
        collector.collect(config, tmp_path / "collected")
    assert not (tmp_path / "collected").exists()
