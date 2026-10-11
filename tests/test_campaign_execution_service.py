"""Durability and admission tests against the actual installed run-watch store."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sqlite3

import pytest

from campaigns.execution_service import (
    CampaignLedger, artifact_probe, digest, investigator_command, load_queue,
    prepare_job, scheduler_type, storage_probe, validate_manifest,
    service_dropin,
    translate_remote_deadlines, CampaignTransport,
    merge_terminal_worker_receipt,
)


@pytest.fixture
def module():
    scripts = Path.home() / ".codex/skills/run-watch/scripts"
    if not (scripts / "gpu_queue.py").exists():
        pytest.skip("installed run-watch required for integration tests")
    return load_queue(scripts)


@pytest.fixture
def setup(tmp_path, module):
    handoff = tmp_path / "handoff.md"
    handoff.write_text("Frozen experiment handoff")
    cfg = {"state_dir": str(tmp_path / "state"), "lock_dir": str(tmp_path / "lock"),
           "targets": {"lab": {"canonical_host": "lab", "transport": "local", "python": "python3",
                               "our_users": ["filip"], "sharing_users": [], "enabled": True, "quota_observer": False,
                               "worker_root": str(tmp_path / "workers")}}, "recovery": {}, "_path": str(tmp_path / "queue.json")}
    now = [1000.0]
    class Transport:
        def collect(self, attempts):
            return {"lab": {"gpus": [], "attempts": {}}}
    scheduler = scheduler_type(module)(cfg, transport=Transport(), clock=lambda: now[0])
    def job(name="one", allowance=900):
        raw = {"id": name, "project_dir": str(tmp_path), "handoff": str(handoff), "authorization": "test",
               "peak_memory_mib": 1, "expected_seconds": 600, "budget_seconds": allowance,
               "recovery": False, "retry_safe": True, "placements": {"lab": {
                   "argv": ["true"], "validate_argv": ["true"], "workdir": str(tmp_path),
                   "run_dir": str(tmp_path / name), "campaign_worker_spec": str(tmp_path / f"{name}.json"),
                   "campaign_result_root": str(tmp_path / name)}}}
        prepare_job(module, scheduler, raw)
        return scheduler.store.job(name)
    def manifest(names=("one",), **kwargs):
        value = {"version": 1, "id": "study", "poll_seconds": 300, "required_adapters": ["tmux"],
                 "budgets": {"gpu_hours": 1, "storage_bytes": 1000, "window_seconds": 43200},
                 "cases": [{"id": name, "stage": "training", "dependencies": [], "required_gates": ["handoff"],
                            "expected_gpu_seconds": 600, "expected_storage_bytes": 100,
                            "progress": {"startup_seconds": 600, "stale_seconds": 600, "expected_seconds": 600}}
                           for name in names]}
        value.update(kwargs)
        return value
    return scheduler, now, job, manifest


def ready(setup):
    scheduler, now, job, manifest = setup
    value = job()
    sha = scheduler.campaigns.register(manifest())
    scheduler.campaigns.gate("study", "handoff", {"manifest_sha256": sha, "passed": True, "evidence": ["tested"]})
    scheduler.campaigns.storage("study", 10)
    scheduler.store.update_job("one", "queued")
    return scheduler, now, scheduler.store.job("one")


def test_prepared_job_is_atomically_held_and_registration_idempotent(setup):
    scheduler, now, job, manifest = setup
    assert job()["state"] == "held"
    sha = scheduler.campaigns.register(manifest())
    scheduler.store.update_job("one", "running")
    assert scheduler.campaigns.register(manifest()) == sha
    assert scheduler.campaigns.contract("study")["started"] is None


def relocation(setup, tmp_path):
    scheduler, _, job, manifest = setup
    value = job()
    sha = scheduler.campaigns.register(manifest())
    placement = value["spec"]["placements"]["lab"]
    proof = tmp_path / "prepared.json"
    proof.write_text(json.dumps({"host": "spare", "cases": [{"id": "one", "dry_run_passed": True, "placement": placement}]}))
    return {"version": 1, "id": "move-one", "manifest_sha256": sha,
        "authorization": "existing GPU allocation", "reason": "original GPU occupied", "evidence": [str(proof)],
        "relocations": [{"job_id": "one", "from_target": "lab", "to_target": "spare", "to_canonical_host": "spare",
            "placement_sha256": digest(placement), "prepared_receipt": {"path": str(proof), "sha256": hashlib.sha256(proof.read_bytes()).hexdigest()}}],
        "storage_roots": [{"target": "spare", "path": str(tmp_path / "one")}]}


def test_queued_relocation_preserves_identity_budget_and_is_idempotent(setup, tmp_path):
    amendment = relocation(setup, tmp_path)
    scheduler = setup[0]
    before = scheduler.store.job("one")
    scheduler.campaigns.storage("study", 10)
    sha = scheduler.campaigns.amend("study", amendment)
    assert scheduler.campaigns.amend("study", amendment) == sha
    after = scheduler.store.job("one")
    assert after["state"] == "held" and after["used_seconds"] == 0
    assert set(after["spec"]["placements"]) == {"lab", "spare"}
    assert after["spec"]["placements"]["spare"] == before["spec"]["placements"]["lab"]
    assert {k:v for k,v in after["spec"].items() if k != "placements"} == {k:v for k,v in before["spec"].items() if k != "placements"}
    assert scheduler.campaigns.remaining("study")["storage_bytes_observed"] is None
    assert scheduler.campaigns.contract("study")["effective_budgets"]["gpu_hours"] == 1
    assert scheduler.store.db.execute("SELECT job_id FROM destinations WHERE host='spare'").fetchone()[0] == "one"


@pytest.mark.parametrize("failure", ["runnable", "history", "changed_placement", "changed_receipt", "destination"])
def test_queued_relocation_fails_closed(setup, tmp_path, failure):
    amendment = relocation(setup, tmp_path)
    scheduler = setup[0]
    if failure == "runnable":
        scheduler.store.update_job("one", "queued")
    elif failure == "history":
        scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at) VALUES(?,?,?,?,?,?,?,?,?)",
            ("old", "one", "lab", "gpu", "5090", "complete", 900, 945, 900))
    elif failure == "changed_placement":
        proof = tmp_path / "prepared.json"
        doc = json.loads(proof.read_text()); doc["cases"][0]["placement"]["argv"] = ["different-science"]
        proof.write_text(json.dumps(doc))
        amendment["relocations"][0]["prepared_receipt"]["sha256"] = hashlib.sha256(proof.read_bytes()).hexdigest()
    elif failure == "changed_receipt":
        (tmp_path / "prepared.json").write_text("{}")
    else:
        setup[2]("other")
        scheduler.store.db.execute("INSERT INTO destinations VALUES(?,?,?)", ("spare", str(tmp_path / "one"), "other"))
    with pytest.raises(ValueError):
        scheduler.campaigns.amend("study", amendment)
    assert set(scheduler.store.job("one")["spec"]["placements"]) == {"lab"}
    assert not scheduler.campaigns.contract("study")["amendments"]


def test_registration_rejects_runnable_unowned_job(setup):
    scheduler, _, job, manifest = setup
    job()
    scheduler.store.update_job("one", "queued")
    with pytest.raises(ValueError, match="held"):
        scheduler.campaigns.register(manifest())


def test_frozen_gate_and_contract_cannot_silently_change(setup):
    scheduler, _, job, manifest = setup
    job()
    sha = scheduler.campaigns.register(manifest())
    receipt = {"manifest_sha256": sha, "passed": False, "evidence": ["canary failed"]}
    scheduler.campaigns.gate("study", "handoff", receipt)
    assert not scheduler.campaigns.eligible(scheduler.store.job("one"))[0]
    with pytest.raises(ValueError, match="immutable"):
        scheduler.campaigns.gate("study", "handoff", dict(receipt, passed=True))
    with pytest.raises(ValueError, match="frozen"):
        scheduler.campaigns.register(manifest(poll_seconds=300, review="different"))


def test_missing_gate_storage_and_budget_block_dispatch(setup):
    scheduler, _, job, manifest = setup
    value = job()
    sha = scheduler.campaigns.register(manifest())
    assert "gate" in scheduler.campaigns.eligible(value)[1]
    scheduler.campaigns.gate("study", "handoff", {"manifest_sha256": sha, "passed": True, "evidence": ["done"]})
    assert "storage" in scheduler.campaigns.eligible(value)[1]
    scheduler.campaigns.storage("study", 950)
    assert "storage budget" in scheduler.campaigns.eligible(value)[1]
    scheduler.campaigns.storage("study", 10)
    assert scheduler.campaigns.eligible(value)[0]


def test_unknown_attempt_keeps_reservation_across_restart(setup):
    scheduler, now, value = ready(setup)
    scheduler.campaigns.reserve(value)
    scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at) VALUES(?,?,?,?,?,?,?,?,?)",
                               ("token", "one", "lab", "gpu", "5090", "unknown", now[0], now[0]+45, now[0]))
    scheduler.campaigns.reconcile_reservations()
    recreated = CampaignLedger(scheduler.store, lambda: now[0])
    recreated.reconcile_reservations()
    assert recreated.remaining("study")["gpu_seconds_reserved"] == 900
    assert not recreated.eligible(value)[0]


def test_submission_intent_without_claim_is_safe_to_release(setup):
    scheduler, _, value = ready(setup)
    scheduler.campaigns.reserve(value)
    scheduler.campaigns.reconcile_reservations()
    assert scheduler.campaigns.remaining("study")["gpu_seconds_reserved"] == 0


def test_actual_charge_replaces_reservation_once(setup):
    scheduler, now, value = ready(setup)
    scheduler.campaigns.reserve(value)
    scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at,charged) VALUES(?,?,?,?,?,?,?,?,?,?)",
                               ("token", "one", "lab", "gpu", "5090", "complete", now[0], now[0]+45, now[0], 1))
    scheduler.store.db.execute("UPDATE jobs SET used_seconds=650 WHERE id='one'")
    scheduler.campaigns.reconcile_reservations()
    scheduler.campaigns.reconcile_reservations()
    usage = scheduler.campaigns.remaining("study")
    assert usage["gpu_seconds_charged"] == 650
    assert usage["gpu_seconds_reserved"] == 0
    assert usage["gpu_seconds_available"] == 2950


def test_hard_gpu_and_window_allowances_are_reserved(setup):
    scheduler, now, value = ready(setup)
    scheduler.store.db.execute("UPDATE jobs SET used_seconds=3000 WHERE id='one'")
    assert "GPU-hour" in scheduler.campaigns.eligible(value)[1]
    scheduler.store.db.execute("UPDATE jobs SET used_seconds=0 WHERE id='one'")
    scheduler.campaigns.db.execute("UPDATE campaign_contracts SET started=?", (now[0] - 43100,))
    assert "window" in scheduler.campaigns.eligible(value)[1]


def test_storage_freshness_and_deadline_survive_restart(setup):
    scheduler, now, value = ready(setup)
    now[0] += 601
    assert "stale" in scheduler.campaigns.eligible(value)[1]


def test_dependencies_wait_for_validated_completion(setup):
    scheduler, _, job, manifest = setup
    job("one"); job("two")
    config = manifest(("one", "two"))
    config["cases"][1]["dependencies"] = ["one"]
    scheduler.campaigns.register(config)
    assert "dependency" in scheduler.campaigns.eligible(scheduler.store.job("two"))[1]
    scheduler.store.update_job("one", "collecting")
    assert "dependency" in scheduler.campaigns.eligible(scheduler.store.job("two"))[1]


def test_unrelated_job_policy_is_unchanged(setup):
    scheduler, _, job, _ = setup
    value = job("unrelated")
    assert scheduler.campaigns.eligible(value) == (True, "")
    scheduler.campaigns.reserve(value)
    assert scheduler.campaigns.db.execute("SELECT COUNT(*) FROM campaign_reservations").fetchone()[0] == 0


def test_incident_dedup_and_remaining_budget_persist(setup):
    scheduler, now, value = ready(setup)
    first = scheduler.campaigns.incident("one", "stalled", {"log": "old"})
    second = scheduler.campaigns.incident("one", "stalled", {"log": "new"})
    assert first == second
    ledger = CampaignLedger(scheduler.store, lambda: now[0])
    snapshot = ledger.snapshot(ledger.db.execute("SELECT * FROM campaign_incidents").fetchone())
    assert snapshot["case_id"] == "one"
    assert snapshot["manifest_sha256"] == digest(ledger.contract("study")["manifest"])
    assert snapshot["remaining_budget"]["gpu_seconds_available"] == 3600


def test_incident_snapshot_preserves_collection_failure_log(setup):
    scheduler, _, _ = ready(setup)
    payload = {"receipt": {"state": "failed", "run_exit_code": 1}, "log_tail": "ValueError: gate calibration rejected"}
    scheduler.store.db.execute("INSERT INTO collections VALUES(?,?,?,?,?)", ("collect-one", "one", 1000, "failed", json.dumps(payload)))
    scheduler.campaigns.incident("one", "queue_anomaly", {"reason": "collection failed"})
    incident = scheduler.store.db.execute("SELECT * FROM campaign_incidents").fetchone()
    history = scheduler.campaigns.snapshot(incident)["collection_history"]
    assert len(history) == 1 and history[0]["payload"] == payload
    assert history[0]["job_id"] == "one" and history[0]["state"] == "failed"


def test_read_only_invocation_has_no_approval_or_inherited_tools(tmp_path):
    command = investigator_command({"command": ["codex"]}, tmp_path)
    assert "--ignore-user-config" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert 'approval_policy="never"' in command
    assert "--approve-for-me" not in command
    assert "--output-last-message" not in command
    assert "--ignore-rules" in command


def test_retry_requires_known_class_terminal_workers_and_only_once(setup):
    scheduler, _, value = ready(setup)
    scheduler.store.update_job("one", "attention")
    evidence = {"classified_by": "deterministic_adapter", "workers_confirmed_terminal": True, "failure_class": "node_failure"}
    with pytest.raises(ValueError, match="operational"):
        scheduler.campaigns.authorize_retry("one", "oom", evidence)
    with pytest.raises(ValueError, match="terminal"):
        scheduler.campaigns.authorize_retry("one", "node_failure", dict(evidence, workers_confirmed_terminal=False))
    scheduler.campaigns.authorize_retry("one", "node_failure", evidence)
    assert scheduler.store.job("one")["state"] == "queued"
    with pytest.raises(ValueError, match="already consumed"):
        scheduler.campaigns.authorize_retry("one", "node_failure", evidence)


def driver_crash_receipt(setup, tmp_path):
    scheduler, _, _ = ready(setup)
    job = scheduler.store.job("one")
    spec = job["spec"]
    spec["retry_safe"] = False
    spec["placements"]["spare"] = {**spec["placements"]["lab"], "run_dir": str(tmp_path / "spare")}
    scheduler.store.db.execute("UPDATE jobs SET spec=?,used_seconds=45 WHERE id='one'", (json.dumps(spec),))
    scheduler.store.update_job("one", "attention")
    terminal = {"alive": False, "reservation_held": False,
        "campaign_worker_terminal": {"worker_alive": False, "supervisor_alive": False}}
    scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,receipt,progress_at,charged) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        ("failed-one", "one", "lab", "gpu", "5090", "failed", 900, 945, json.dumps(terminal), 945, 1))
    evidence = {"failure_class": "node_failure", "failed_attempt": "failed-one", "failed_target": "lab", "native_pid": 123,
        "stage_result": {"returncode": -11, "status": "failed"}, "kernel_log": ["python[123]: segfault in libcuda.so.610"]}
    path = tmp_path / "driver-evidence.json"; path.write_text(json.dumps(evidence))
    return {"version": 1, "manifest_sha256": scheduler.campaigns.contract("study")["sha256"],
        "job_spec_sha256": digest(spec), "failed_attempt": "failed-one", "authorization": "User permits classified operational retries within the original budget",
        "evidence": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, "restart_from_original_inputs": True}


def test_driver_recovery_is_one_audited_replay_without_changing_science_or_budget(setup, tmp_path):
    receipt = driver_crash_receipt(setup, tmp_path)
    scheduler = setup[0]; before = scheduler.store.job("one")
    identity = scheduler.campaigns.recover_driver_crash("one", receipt)
    after = scheduler.store.job("one")
    assert after["state"] == "queued" and after["used_seconds"] == 45
    assert after["spec"]["retry_safe"] is False
    assert set(after["spec"]["placements"]) == {"spare"}
    assert after["spec"]["placements"]["spare"] == before["spec"]["placements"]["spare"]
    assert {k:v for k,v in before["spec"].items() if k != "placements"} == {k:v for k,v in after["spec"].items() if k != "placements"}
    assert scheduler.campaigns.recover_driver_crash("one", receipt) == identity
    assert len(scheduler.store.attempts()) == 1
    assert scheduler.store.db.execute("SELECT COUNT(*) FROM campaign_retries").fetchone()[0] == 1
    with pytest.raises(ValueError, match="allowance already consumed"):
        scheduler.campaigns.recover_driver_crash("one", {**receipt, "authorization": "different retry"})


@pytest.mark.parametrize("failure", ["hash", "not_driver", "nonfinite", "active", "complete", "worker_alive", "budget", "no_spare", "contract"])
def test_driver_recovery_fails_closed(setup, tmp_path, failure):
    receipt = driver_crash_receipt(setup, tmp_path)
    scheduler = setup[0]; db = scheduler.store.db
    if failure == "hash":
        receipt["evidence"]["sha256"] = "0" * 64
    elif failure in {"not_driver", "nonfinite"}:
        path = Path(receipt["evidence"]["path"]); data = json.loads(path.read_text())
        if failure == "not_driver": data["kernel_log"] = ["python[123]: segfault in unrelated.so"]
        else: data["stage_result"]["returncode"] = 1
        path.write_text(json.dumps(data)); receipt["evidence"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif failure in {"active", "complete"}:
        db.execute("UPDATE attempts SET state=?", ("running" if failure == "active" else "complete",))
    elif failure == "worker_alive":
        data = json.loads(db.execute("SELECT receipt FROM attempts").fetchone()[0]); data["campaign_worker_terminal"]["worker_alive"] = True
        db.execute("UPDATE attempts SET receipt=?", (json.dumps(data),))
    elif failure == "budget":
        db.execute("UPDATE jobs SET used_seconds=890 WHERE id='one'")
    elif failure == "no_spare":
        spec = scheduler.store.job("one")["spec"]; del spec["placements"]["spare"]
        db.execute("UPDATE jobs SET spec=? WHERE id='one'", (json.dumps(spec),)); receipt["job_spec_sha256"] = digest(spec)
    else:
        receipt["job_spec_sha256"] = "0" * 64
    before = scheduler.store.job("one")
    with pytest.raises(ValueError): scheduler.campaigns.recover_driver_crash("one", receipt)
    assert scheduler.store.job("one") == before
    assert db.execute("SELECT COUNT(*) FROM campaign_retries").fetchone()[0] == 0


@pytest.mark.parametrize("repaired", [True, False])
def test_missing_tmux_recovery_requires_repair_and_keeps_the_prepared_command(setup, tmp_path, repaired):
    receipt = driver_crash_receipt(setup, tmp_path)
    scheduler = setup[0]; db = scheduler.store.db
    path = Path(receipt["evidence"]["path"])
    evidence = {"failure_class": "transport_before_start", "failed_attempt": "failed-one", "failed_target": "lab",
        "launcher_log": 'if tmux(spec, "has-session", check=False)\nFileNotFoundError: [Errno 2] No such file or directory: \'tmux\'',
        "tmux_probe": {"passed": repaired}}
    path.write_text(json.dumps(evidence)); receipt["evidence"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    terminal = {"alive": False, "reservation_held": False, "campaign_worker_unstarted": {"reason": "tmux_not_found"}}
    db.execute("UPDATE attempts SET receipt=?", (json.dumps(terminal),))
    before = scheduler.store.job("one")
    if not repaired:
        with pytest.raises(ValueError): scheduler.campaigns.recover_missing_tmux("one", receipt)
        assert scheduler.store.job("one") == before
    else:
        scheduler.campaigns.recover_missing_tmux("one", receipt)
        after = scheduler.store.job("one")
        expected = before["spec"]; expected["placements"]["lab"]["allow_existing_run_dir"] = True
        assert after["spec"] == expected and after["used_seconds"] == before["used_seconds"]
        assert after["state"] == "queued" and not after["spec"]["retry_safe"]
        # Simulate an older audited recovery that was rejected before execution
        # because its same-job receipt directory already existed.
        spec = after["spec"]; del spec["placements"]["lab"]["allow_existing_run_dir"]
        db.execute("UPDATE jobs SET spec=? WHERE id='one'", (json.dumps(spec),))
        db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at) VALUES(?,?,?,?,?,?,?,?,?)",
            ("refused", "one", "lab", "gpu", "5090", "rejected", 1001, 1046, 1001))
        scheduler.store.update_job("one", "attention", "result_path_exists; explicitly declare reuse if intentional")
        scheduler.campaigns.recover_missing_tmux("one", receipt)
        repaired_job = scheduler.store.job("one")
        assert repaired_job["state"] == "queued" and repaired_job["used_seconds"] == 45
        assert repaired_job["spec"]["placements"]["lab"]["allow_existing_run_dir"] is True
        assert db.execute("SELECT COUNT(*) FROM campaign_retries").fetchone()[0] == 1


def test_missing_tmux_before_creation_is_terminal_without_inventing_worker_receipt():
    observed = {"receipt": {"state": "failed", "started_at": 1000, "elapsed_seconds": .2},
        "alive": False, "reservation_held": False,
        "log_tail": 'if tmux(spec, "has-session", check=False)\nFileNotFoundError: [Errno 2] No such file or directory: \'tmux\''}
    value = merge_terminal_worker_receipt(observed, {"missing": True})
    assert value["receipt"]["state"] == "failed" and value["receipt"]["elapsed_seconds"] == .2
    assert value["campaign_worker_unstarted"]["reason"] == "tmux_not_found"
    assert "campaign_worker_terminal" not in value and "probe_error" not in value
    unknown = {**observed, "log_tail": "FileNotFoundError: unrelated file"}
    assert "probe_error" in merge_terminal_worker_receipt(unknown, {"missing": True})


def test_cycles_and_unsupported_slurm_are_rejected(setup):
    _, _, _, manifest = setup
    bad = manifest()
    bad["cases"][0]["dependencies"] = ["one"]
    with pytest.raises(ValueError, match="cycle"):
        validate_manifest(bad)
    with pytest.raises(ValueError, match="Slurm"):
        validate_manifest(manifest(required_adapters=["slurm"]))


def test_nested_metrics_finite_validation_and_partial_write(tmp_path):
    stage = tmp_path / "campaign/stages/hwa/bundle"
    stage.mkdir(parents=True)
    metric = stage / "metrics.jsonl"
    metric.write_text('{"step":1,"loss":0.5}\n{"step":2')
    value = artifact_probe(tmp_path)
    assert value["metrics_valid"]
    assert next(iter(value["progress"].values()))["records"] == 1
    metric.write_text('{"step":1,"loss":NaN}\n')
    assert not artifact_probe(tmp_path)["metrics_valid"]


def native_stage(root, name, kind, *, status="running", stamp=100, metric=None, seed=None):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    stage = {"kind": kind, "hwa_arm": "standard_hwa", "array_seed": seed}
    (path / "config.resolved.json").write_text(json.dumps({"stage": stage}))
    (path / "status.json").write_text(json.dumps({"status": status}))
    if metric is not None:
        (path / "metrics.jsonl").write_text(json.dumps(metric) + "\n")
    for file in path.iterdir():
        os.utime(file, (stamp, stamp))
    return path


def test_artifact_probe_identifies_native_stage_and_its_latest_complete_metric(tmp_path):
    old = native_stage(tmp_path, "hwa/run", "hwa", status="complete", stamp=200,
                       metric={"stage": "hwa_selection", "global_step": 25600})
    current = native_stage(tmp_path, "deploy/run", "deploy", stamp=300, seed=271001,
                           metric={"stage": "deploy_case", "case": "nominal", "teacher_kl": 0.1})
    # A completed stage's newer file timestamp must not mask the active run.
    os.utime(old / "metrics.jsonl", (400, 400))
    with (current / "metrics.jsonl").open("a") as stream:
        stream.write('{"stage":"deploy_case","case":')
    observed = artifact_probe(tmp_path)
    assert observed["active_stage"] == {"kind": "deploy", "hwa_arm": "standard_hwa", "array_seed": 271001}
    assert observed["stage_key"] == "deploy/run" and observed["stage_status"] == "running"
    assert observed["latest_metric"]["case"] == "nominal"
    assert observed["latest_metric"]["stage"] == "deploy_case"
    assert observed["stage_progress"]["deploy/run/metrics.jsonl"]["records"] == 1
    assert all(name.startswith("deploy/run/") for name in observed["stage_progress"])
    assert "hwa/run/metrics.jsonl" in observed["progress"]
    # A new stage is visible before its first metric is available.
    (current / "status.json").write_text('{"status":"complete"}')
    native_stage(tmp_path, "deploy2/run", "deploy", stamp=500, seed=271002)
    observed = artifact_probe(tmp_path)
    assert observed["active_stage"]["array_seed"] == 271002
    assert observed["latest_metric"] is None


def test_new_native_config_and_running_status_identify_phase_before_first_metric(tmp_path):
    native_stage(tmp_path, "earlier/run", "hwa", stamp=100, metric={"global_step": 25600})
    native_stage(tmp_path, "new/run", "deploy", stamp=200, seed=271001)
    observed = artifact_probe(tmp_path)
    assert observed["active_stage"]["kind"] == "deploy"
    assert observed["stage_key"] == "new/run" and observed["latest_metric"] is None


@pytest.mark.parametrize("phases", [None, {}, [], {"training": {}},
    {"deploy": {"startup_seconds": 600, "stale_seconds": 600}},
    {"hwa": {"startup_seconds": True, "stale_seconds": 600, "expected_seconds": 600}},
    {"validation": {"startup_seconds": 600, "stale_seconds": 0, "expected_seconds": 600}},
    {"deploy": {"startup_seconds": 600, "stale_seconds": 600, "expected_seconds": 600, "other": 1}},
])
def test_phase_contract_rejects_unknown_or_incomplete_thresholds(setup, phases):
    _, _, _, manifest = setup
    doc = manifest()
    doc["cases"][0]["progress"]["phases"] = phases
    with pytest.raises(ValueError):
        validate_manifest(doc)


def phase_thresholds():
    return {"hwa": {"startup_seconds": 6000, "stale_seconds": 6000, "expected_seconds": 12000},
            "deploy": {"startup_seconds": 900, "stale_seconds": 600, "expected_seconds": 1200},
            "validation": {"startup_seconds": 600, "stale_seconds": 600, "expected_seconds": 900}}


def test_explicit_phase_contract_wins_over_old_whole_case_calibration(setup):
    scheduler, _, job, manifest = setup
    job()
    doc = manifest()
    doc["cases"][0]["progress"]["phases"] = phase_thresholds()
    sha = scheduler.campaigns.register(doc)
    scheduler.campaigns.gate("study", "handoff", {"manifest_sha256": sha, "passed": True,
        "evidence": ["old calibration"], "progress_calibration": {"training": {
            "startup_seconds": 10000, "stale_seconds": 10000, "expected_seconds": 10000}}})
    for kind, thresholds in phase_thresholds().items():
        assert scheduler.campaigns.progress_expectation("one", "lab", {"kind": kind}) == thresholds
    assert scheduler.campaigns.progress_expectation("one", "lab", "prepare") == {
        "startup_seconds": 600, "stale_seconds": 600, "expected_seconds": 600}


def test_handoff_phase_calibration_is_frozen_without_overwriting_other_phase_or_manifest(setup):
    scheduler, _, job, manifest = setup
    job()
    doc = manifest()
    doc["cases"][0]["progress"]["phases"] = phase_thresholds()
    sha = scheduler.campaigns.register(doc)
    measured = {"startup_seconds": 600, "stale_seconds": 600, "expected_seconds": 300}
    receipt = {"manifest_sha256": sha, "passed": True, "evidence": ["native canary"],
               "phase_progress_calibration": {"deploy": measured}}
    scheduler.campaigns.gate("study", "handoff", receipt)
    assert scheduler.campaigns.progress_expectation("one", "lab", "deploy") == measured
    assert scheduler.campaigns.progress_expectation("one", "lab", "hwa") == phase_thresholds()["hwa"]
    assert scheduler.campaigns.contract("study")["manifest"] == doc
    assert scheduler.campaigns.contract("study")["sha256"] == sha
    scheduler.campaigns.gate("study", "handoff", receipt)
    with pytest.raises(ValueError, match="immutable"):
        scheduler.campaigns.gate("study", "handoff", {**receipt, "phase_progress_calibration": {
            "deploy": {**measured, "expected_seconds": 500}}})
    with pytest.raises(ValueError, match="handoff"):
        scheduler.campaigns.gate("study", "other", receipt)


@pytest.mark.parametrize("phases", [False, True])
def test_required_gate_calibration_is_immutable_and_scoped_to_its_case(setup, phases):
    scheduler, _, job, manifest = setup
    job("one"); job("two")
    doc = manifest(("one", "two"))
    doc["cases"][0]["required_gates"] = ["selected-r64"]
    if phases:
        for case in doc["cases"]:
            case["progress"]["phases"] = phase_thresholds()
    sha = scheduler.campaigns.register(doc)
    measured = {"startup_seconds": 600, "stale_seconds": 900, "expected_seconds": 1961}
    key = "phase_progress_calibration" if phases else "progress_calibration"
    stage = "deploy" if phases else "training"
    receipt = {"manifest_sha256": sha, "passed": True, "evidence": ["validated-r64-screen"], key: {stage: measured}}
    scheduler.campaigns.gate("study", "selected-r64", receipt)
    scheduler.campaigns.gate("study", "selected-r64", receipt)
    assert scheduler.campaigns.progress_expectation("one", "lab", stage) == measured
    expected = phase_thresholds()[stage] if phases else doc["cases"][1]["progress"]
    assert scheduler.campaigns.progress_expectation("two", "lab", stage) == expected
    assert scheduler.campaigns.contract("study")["manifest"] == doc
    with pytest.raises(ValueError, match="immutable"):
        scheduler.campaigns.gate("study", "selected-r64", {**receipt, key: {stage: {**measured, "expected_seconds": 3000}}})
    with pytest.raises(ValueError, match="declared required gate"):
        scheduler.campaigns.gate("study", "undeclared-selection", receipt)


def test_failed_required_gate_cannot_supply_calibration(setup):
    scheduler, _, job, manifest = setup
    job()
    doc = manifest(); doc["cases"][0]["required_gates"] = ["selection"]
    sha = scheduler.campaigns.register(doc)
    scheduler.campaigns.gate("study", "selection", {"manifest_sha256": sha, "passed": False,
        "evidence": ["failed canary"], "progress_calibration": {"training": {
            "startup_seconds": 900, "stale_seconds": 900, "expected_seconds": 2000}}})
    assert scheduler.campaigns.progress_expectation("one") == doc["cases"][0]["progress"]


@pytest.mark.parametrize("calibration", [None, [], {"wrong": {}},
    {"deploy": {"startup_seconds": 599, "stale_seconds": 600, "expected_seconds": 300}},
    {"deploy": {"startup_seconds": 600, "stale_seconds": True, "expected_seconds": 300}},
    {"deploy": {"startup_seconds": 600, "stale_seconds": 600}},
])
def test_handoff_phase_calibration_rejects_invalid_or_short_poll_thresholds(setup, calibration):
    scheduler, _, job, manifest = setup
    job()
    sha = scheduler.campaigns.register(manifest())
    with pytest.raises(ValueError):
        scheduler.campaigns.gate("study", "handoff", {"manifest_sha256": sha, "passed": True,
            "evidence": ["native canary"], "phase_progress_calibration": calibration})


def phase_monitor(setup, monkeypatch):
    scheduler, now, job, manifest = setup
    job()
    doc = manifest()
    doc["cases"][0]["progress"]["phases"] = phase_thresholds()
    scheduler.campaigns.register(doc)
    scheduler.store.update_job("one", "running")
    scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at) VALUES(?,?,?,?,?,?,?,?,?)",
                               ("token", "one", "lab", "gpu", "5090", "running", now[0] - 10000, now[0] - 9955, now[0]))
    evidence = {"worker": {"worker_alive": True, "supervisor_alive": True, "receipt": {"state": "running"}},
        "artifacts": {"metrics_valid": True, "errors": [], "progress": {"old": {"records": 1}},
                      "active_stage": {"kind": "hwa"}, "stage_key": "hwa/run",
                      "latest_metric": {"global_step": 100}, "stage_progress": {"metrics": {"records": 1}}}}
    def remote(cfg, target, argv, **kwargs):
        if "status" in argv:
            return json.loads(json.dumps(evidence["worker"]))
        if "probe-artifacts" in argv:
            return json.loads(json.dumps(evidence["artifacts"]))
        assert "renew" in argv
        return {}
    monkeypatch.setattr("campaigns.execution_service.remote_command", remote)
    return scheduler, now, evidence


def observed_phase(scheduler):
    row = scheduler.store.db.execute("SELECT payload FROM campaign_observations WHERE token='token'").fetchone()
    return json.loads(row["payload"])["monitor_phase"]


def test_phase_clocks_follow_each_native_array_and_survive_ledger_restart(setup, monkeypatch):
    scheduler, now, evidence = phase_monitor(setup, monkeypatch)
    scheduler.probe_campaigns(now[0])
    assert observed_phase(scheduler)["kind"] == "hwa"
    now[0] += 2000
    evidence["artifacts"].update(active_stage={"kind": "deploy", "array_seed": 271001}, stage_key="deploy1/run")
    scheduler.probe_campaigns(now[0])
    first = observed_phase(scheduler)
    assert first["started_at"] == now[0] and first["expected"] == phase_thresholds()["deploy"]
    assert not scheduler.store.db.execute("SELECT 1 FROM campaign_incidents WHERE kind='slow_progress'").fetchone()
    now[0] += 300
    scheduler.campaigns = CampaignLedger(scheduler.store, lambda: now[0])
    scheduler.probe_campaigns(now[0])
    assert observed_phase(scheduler)["started_at"] == first["started_at"]
    now[0] += 301
    evidence["artifacts"].update(active_stage={"kind": "deploy", "array_seed": 271002}, stage_key="deploy2/run")
    scheduler.probe_campaigns(now[0])
    assert observed_phase(scheduler)["started_at"] == now[0]
    assert scheduler.store.job("one")["state"] == "running"


def test_deployment_stall_uses_its_threshold_and_ignores_old_stage_activity(setup, monkeypatch):
    scheduler, now, evidence = phase_monitor(setup, monkeypatch)
    evidence["artifacts"].update(active_stage={"kind": "deploy"}, stage_key="deploy/run")
    scheduler.probe_campaigns(now[0])
    now[0] += 601
    # Fresh data in a completed stage cannot hide a stalled current deployment.
    evidence["artifacts"]["progress"]["old"]["records"] += 1
    scheduler.probe_campaigns(now[0])
    assert scheduler.store.job("one")["state"] == "attention"
    assert "stage-specific" in scheduler.store.job("one")["reason"]


def test_validation_worker_gets_its_own_clock_liveness_and_incident_context(setup, monkeypatch):
    scheduler, now, evidence = phase_monitor(setup, monkeypatch)
    scheduler.probe_campaigns(now[0])
    now[0] += 3000
    evidence["worker"] = {"worker_alive": False, "supervisor_alive": False, "receipt": {"state": "complete"}}
    scheduler.observations = {"lab": {"attempts": {"token": {"alive": True, "receipt": {"stage": "validate"}}}}}
    scheduler.probe_campaigns(now[0])
    phase = observed_phase(scheduler)
    assert phase["kind"] == "validation" and phase["started_at"] == now[0]
    assert scheduler.store.job("one")["state"] == "running"
    now[0] += 601
    scheduler.probe_campaigns(now[0])
    assert scheduler.store.job("one")["state"] == "attention"
    incident_id = scheduler.campaigns.incident("one", "validation_test", {})
    incident = scheduler.store.db.execute("SELECT * FROM campaign_incidents WHERE id=?", (incident_id,)).fetchone()
    snapshot = scheduler.campaigns.snapshot(incident)
    assert snapshot["expected_progress"] == phase_thresholds()["validation"]
    assert snapshot["latest_observation"]["monitor_phase"]["kind"] == "validation"


def test_storage_counts_staged_copies_but_not_hardlinks_twice(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "one").write_bytes(b"12345")
    (a / "two").hardlink_to(a / "one")
    (a / "copy").write_bytes(b"12345")
    assert storage_probe([a])["storage_bytes"] == 10


def test_all_completed_means_review_not_automatic_scientific_release(setup):
    scheduler, _, _ = ready(setup)
    scheduler.store.update_job("one", "complete")
    assert scheduler.campaigns.status()["campaigns"][0]["state"] == "awaiting_scientific_review"


def test_storage_roots_cannot_double_count_overlap(setup):
    _, _, _, manifest = setup
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(manifest(storage_roots=[{"target": "lab", "path": "/runs"}, {"target": "lab", "path": "/runs/one"}]))


@pytest.mark.parametrize('allowed', [True, False])
def test_remote_only_campaign_accounts_its_prepared_local_controller(setup, tmp_path, allowed):
    scheduler, _, job, manifest = setup
    value = job()
    value['spec']['completion'] = {'argv':['true'],'workdir':str(tmp_path/'evidence/controller'),
                                 'timeout_seconds':60,'retry_safe':True}
    scheduler.store.db.execute('UPDATE jobs SET spec=? WHERE id=?', (json.dumps(value['spec']),'one'))
    original = scheduler.campaigns.register(manifest())
    amendment = {'version':1,'id':'local-storage','manifest_sha256':original,
        'authorization':'existing experiment','reason':'include collection artifacts','evidence':['prepared controller'],
        'storage_roots':[{'target':'local','path':str(tmp_path/('evidence' if allowed else 'unrelated'))}]}
    if not allowed:
        with pytest.raises(ValueError,match='prepared local controller'):
            scheduler.campaigns.amend('study',amendment)
        return
    scheduler.campaigns.amend('study',amendment)
    assert scheduler.campaigns.contract('study')['sha256'] == original
    assert scheduler.campaigns.contract('study')['effective_storage_roots'] == amendment['storage_roots']
    assert set(scheduler.store.job('one')['spec']['placements']) == {'lab'}


def test_shared_dispatcher_lock_prevents_second_owner(module, tmp_path):
    with module.owner_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with module.owner_lock(tmp_path):
                pass


def test_frozen_smoke_calibration_controls_full_training_thresholds(setup):
    scheduler, _, job, manifest = setup
    job()
    sha = scheduler.campaigns.register(manifest())
    calibration = {"startup_seconds": 800, "stale_seconds": 900, "expected_seconds": 3000}
    scheduler.campaigns.gate("study", "handoff", {"manifest_sha256": sha, "passed": True,
        "evidence": ["smoke/receipt.json"], "progress_calibration": {"training": calibration}})
    assert scheduler.campaigns.progress_expectation("one") == calibration
    with pytest.raises(ValueError, match="two five-minute"):
        scheduler.campaigns.gate("study", "handoff", {"manifest_sha256": sha, "passed": True,
            "evidence": ["changed"], "progress_calibration": {"training": dict(calibration, stale_seconds=30)}})


def test_failed_campaign_gate_blocks_other_nonreadiness_case(setup):
    scheduler, _, job, manifest = setup
    job()
    config = manifest()
    config["cases"][0]["required_gates"] = []
    sha = scheduler.campaigns.register(config)
    scheduler.campaigns.storage("study", 0)
    scheduler.campaigns.gate("study", "mandatory", {"manifest_sha256": sha, "passed": False, "evidence": ["failed"]})
    assert "mandatory" in scheduler.campaigns.eligible(scheduler.store.job("one"))[1]


def test_service_dropin_replaces_one_service_and_supervises_heartbeat(tmp_path):
    unit = service_dropin("/usr/bin/python3", tmp_path, tmp_path / "queue.json")
    assert "WatchdogSec=660" in unit
    assert "Type=notify" in unit
    assert "KillMode=process" in unit
    assert "ExecStart=\nExecStart=" in unit
    assert "campaigns.execution_service" in unit
    assert f"WorkingDirectory={tmp_path}\n" in unit
    assert 'WorkingDirectory="' not in unit


@pytest.mark.parametrize("offset", [-7243.0, -73.0, -56.0, 0.0, 125.0])
def test_clock_translation_bounds_admission_and_stop_for_skewed_hosts(offset):
    spec = {"launch_before": 1045.0, "stop_at": 4000.0, "allowed_seconds": 900}
    translated, mapping = translate_remote_deadlines(spec, 1000.25 + offset, 1000.0, 1000.5, .5)
    assert mapping["offset_seconds"] == pytest.approx(offset)
    assert translated["launch_before"] - offset <= spec["launch_before"]
    assert translated["stop_at"] - offset <= spec["stop_at"]
    assert translated["allowed_seconds"] == spec["allowed_seconds"]
    assert spec["launch_before"] == 1045.0


@pytest.mark.parametrize("remote,before,after,elapsed", [
    (1000, 1000, 1003, 3), (1000, 1000, 2000, .2), (float("nan"), 1000, 1000.1, .1),
])
def test_uncertain_clock_probe_is_rejected(remote, before, after, elapsed):
    with pytest.raises(ValueError):
        translate_remote_deadlines({"launch_before": 1045}, remote, before, after, elapsed)


def test_campaign_clock_probe_failure_never_issues_launch(setup, monkeypatch):
    scheduler, _, value = ready(setup)
    class Base:
        calls = []
        def call(self, target, request):
            self.calls.append(request)
            return {"accepted": True}
    base = Base()
    transport = CampaignTransport(base, scheduler.cfg, scheduler.campaigns)
    def fail(*args, **kwargs):
        raise TimeoutError("clock probe unavailable")
    monkeypatch.setattr("campaigns.execution_service.remote_command", fail)
    result = transport.call("lab", {"action": "launch", "token": "token", "spec": {"job_id": "one", "launch_before": 1045}})
    assert not result["accepted"]
    assert base.calls == []
    assert transport.call("lab", {"action": "launch", "spec": {"job_id": "unrelated"}})["accepted"]
    assert len(base.calls) == 1


def test_active_cost_uses_controller_time_not_skewed_worker_epoch(setup):
    scheduler, now, value = ready(setup)
    scheduler.campaigns.reserve(value)
    observed = {"receipt": {"state": "running", "started_at": now[0] - 7243}}
    scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at,receipt) VALUES(?,?,?,?,?,?,?,?,?,?)",
                               ("token", "one", "lab", "gpu", "5090", "running", now[0], now[0]+45, now[0], json.dumps(observed)))
    scheduler.campaigns.reconcile_reservations()
    now[0] += 20
    assert scheduler.campaigns.remaining("study")["gpu_seconds_active_estimate"] == 20


def test_base_flat_directory_stall_does_not_override_calibrated_campaign_monitor(setup):
    scheduler, now, value = ready(setup)
    scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at,progress) VALUES(?,?,?,?,?,?,?,?,?,?)",
                               ("token", "one", "lab", "gpu", "5090", "running", now[0]-5000, now[0]-4900, now[0]-4000, "unchanged"))
    scheduler.reconcile({"lab": {"attempts": {"token": {"receipt": {"state": "running", "started_at": now[0]-5000},
                        "alive": True, "progress_token": "unchanged"}}}}, now[0])
    assert scheduler.store.job("one")["state"] == "running"
    assert scheduler.store.job("one")["spec"]["stale_seconds"] != float("inf")


def test_bridge_failure_charges_continuing_tmux_runtime_without_claiming_validity():
    observed = {"receipt": {"state": "failed", "started_at": 1000, "elapsed_seconds": 10}, "alive": False}
    worker = {"worker_alive": False, "supervisor_alive": False,
              "receipt": {"state": "complete", "started_at": 1002, "elapsed_seconds": 600}}
    merged = merge_terminal_worker_receipt(observed, worker)
    assert merged["receipt"]["state"] == "failed"
    assert merged["receipt"]["elapsed_seconds"] == 602


def test_uncertain_or_live_tmux_prevents_terminal_charge():
    observed = {"receipt": {"state": "failed", "started_at": 1000, "elapsed_seconds": 10}, "alive": False}
    assert merge_terminal_worker_receipt(dict(observed), {"worker_alive": True})["alive"]
    assert "probe_error" in merge_terminal_worker_receipt(dict(observed), {"worker_alive": False, "receipt": {"state": "starting"}})


def test_actual_launch_retains_reservation_with_advancing_clock(setup):
    scheduler, now, job = ready(setup)
    launch_now = now[0]
    # Reproduce real service timing: caller sampled time before reserve samples
    # its own clock. Previously the claim's earlier issued time lost the match.
    now[0] += .05
    class AcceptingTransport:
        def call(self, target, request):
            return {"accepted": True}
    scheduler.transport = AcceptingTransport()
    scheduler.launch(job, {"target": "lab", "uuid": "gpu", "name": "RTX 5090"}, launch_now)
    attempts = scheduler.store.attempts(active=True)
    row = scheduler.store.db.execute("SELECT * FROM campaign_reservations").fetchone()
    assert len(attempts) == 1
    assert row["attempt_token"] == attempts[0]["token"]
    assert row["issued"] == attempts[0]["issued"] == launch_now
    assert scheduler.campaigns.remaining("study")["gpu_seconds_reserved"] == 900


def test_restart_restores_dropped_reservation_for_live_campaign_attempt(setup):
    scheduler, now, job = ready(setup)
    scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at) VALUES(?,?,?,?,?,?,?,?,?)",
                               ("live", "one", "lab", "gpu", "5090", "unknown", now[0] - 20, now[0]+45, now[0]))
    recreated = CampaignLedger(scheduler.store, lambda: now[0])
    recreated.reconcile_reservations()
    recreated.reconcile_reservations()
    row = scheduler.store.db.execute("SELECT * FROM campaign_reservations").fetchone()
    assert row["attempt_token"] == "live"
    assert row["gpu_seconds"] == 900
    assert recreated.contract("study")["started"] == now[0] - 20
    assert scheduler.store.db.execute("SELECT COUNT(*) FROM events WHERE kind='campaign_reservation_restored'").fetchone()[0] == 1


def test_duplicate_active_campaign_attempts_fail_closed(setup):
    scheduler, now, _ = ready(setup)
    for token in ("one", "two"):
        scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at) VALUES(?,?,?,?,?,?,?,?,?)",
                                   (token, "one", "lab", "gpu", "5090", "running", now[0], now[0]+45, now[0]))
    with pytest.raises(RuntimeError, match="multiple active"):
        scheduler.campaigns.reconcile_reservations()


def amendment_ready(setup):
    scheduler, now, job, manifest = setup
    job("one"); job("readiness-new")
    for name in ("one", "readiness-new"):
        value = scheduler.store.job(name)
        value["spec"]["placements"]["lab"]["prepared_files"] = [{"path": "/frozen/lifecycles/smoke.json", "sha256": "a" * 64}]
        scheduler.store.db.execute("UPDATE jobs SET spec=? WHERE id=?", (json.dumps(value["spec"]), name))
    config = manifest()
    config["cases"][0]["stage"] = "readiness"
    sha = scheduler.campaigns.register(config)
    scheduler.campaigns.gate("study", "handoff", {"manifest_sha256": sha, "passed": True, "evidence": ["original handoff"]})
    scheduler.campaigns.storage("study", 50)
    scheduler.store.db.execute("UPDATE campaign_contracts SET started=? WHERE id='study'", (now[0] - 100,))
    new_case = {**config["cases"][0], "id": "readiness-new", "required_gates": []}
    amendment = {"version": 1, "id": "add-readiness-host", "manifest_sha256": sha,
        "authorization": "User permits all available GPUs", "reason": "Measure new host memory and throughput",
        "evidence": ["matching-runtime.json"], "template_case_id": "one",
        "scientific_inputs": [{"kind": "lifecycle", "sha256": "a" * 64}], "cases": [new_case],
        "storage_roots": [{"target": "lab", "path": "/new/runtime"}]}
    return scheduler, now, config, amendment


def test_additive_readiness_preserves_science_gates_window_and_aggregates_cost(setup):
    scheduler, now, config, amendment = amendment_ready(setup)
    before = scheduler.campaigns.contract("study")
    receipt = scheduler.store.db.execute("SELECT receipt FROM campaign_gates").fetchone()[0]
    sha = scheduler.campaigns.amend("study", amendment)
    after = scheduler.campaigns.contract("study")
    assert after["manifest"] == config == before["manifest"]
    assert after["sha256"] == before["sha256"]
    assert after["started"] == before["started"] == now[0] - 100
    assert scheduler.store.db.execute("SELECT receipt FROM campaign_gates").fetchone()[0] == receipt
    assert {case["id"] for case in after["effective_cases"]} == {"one", "readiness-new"}
    assert after["effective_storage_roots"] == amendment["storage_roots"]
    assert after["storage_observed"] is None
    assert scheduler.store.job("readiness-new")["state"] == "held"
    scheduler.store.db.execute("UPDATE jobs SET used_seconds=123 WHERE id='readiness-new'")
    assert scheduler.campaigns.remaining("study")["gpu_seconds_charged"] == 123
    assert scheduler.campaigns.amend("study", amendment) == sha
    assert len(scheduler.campaigns.contract("study")["amendments"]) == 1


@pytest.mark.parametrize("change", ["active", "different_hash", "scientific_stage", "duplicate", "wrong_manifest", "budget_field"])
def test_readiness_amendment_fails_closed_without_partial_edits(setup, change):
    scheduler, now, _, amendment = amendment_ready(setup)
    if change == "active":
        scheduler.store.update_job("readiness-new", "running")
    elif change == "different_hash":
        spec = scheduler.store.job("readiness-new")["spec"]
        spec["placements"]["lab"]["prepared_files"][0]["sha256"] = "b" * 64
        scheduler.store.db.execute("UPDATE jobs SET spec=? WHERE id='readiness-new'", (json.dumps(spec),))
    elif change == "scientific_stage":
        amendment["cases"][0]["stage"] = "training"
    elif change == "duplicate":
        amendment["cases"][0]["id"] = "one"
    elif change == "wrong_manifest":
        amendment["manifest_sha256"] = "b" * 64
    elif change == "budget_field":
        amendment["budgets"] = {"gpu_hours": 100}
    with pytest.raises(ValueError):
        scheduler.campaigns.amend("study", amendment)
    assert len(scheduler.campaigns.contract("study")["effective_cases"]) == 1
    assert scheduler.store.db.execute("SELECT COUNT(*) FROM campaign_amendments").fetchone()[0] == 0
    assert scheduler.campaigns.contract("study")["storage_bytes"] == 50


def test_target_progress_calibration_is_additive_and_does_not_replace_global_gate(setup):
    scheduler, _, _, amendment = amendment_ready(setup)
    calibration = {"startup_seconds": 600, "stale_seconds": 900, "expected_seconds": 7000}
    amendment["target_progress_calibration"] = {"lab": {"readiness": calibration}}
    scheduler.campaigns.amend("study", amendment)
    assert scheduler.campaigns.progress_expectation("readiness-new", "lab") == calibration
    assert scheduler.campaigns.progress_expectation("readiness-new", "other")["expected_seconds"] == 600
    second = {key: value for key, value in amendment.items() if key not in {"cases", "storage_roots"}}
    second["id"] = "replace-target-calibration"
    with pytest.raises(ValueError, match="already frozen"):
        scheduler.campaigns.amend("study", second)


def test_storage_only_amendment_rejects_unknown_target_and_overlap(setup):
    scheduler, _, _, amendment = amendment_ready(setup)
    scheduler.campaigns.amend("study", amendment)
    second = {key: value for key, value in amendment.items() if key not in {"cases", "scientific_inputs", "template_case_id"}}
    second["id"] = "extra-storage"
    with pytest.raises(ValueError, match="overlap"):
        scheduler.campaigns.amend("study", second)
    second["storage_roots"] = [{"target": "undeclared", "path": "/another/runtime"}]
    with pytest.raises(ValueError, match="target"):
        scheduler.campaigns.amend("study", second)


def matching_waiver_fixture(setup, required=False):
    scheduler, _, job, manifest = setup
    value = job()
    config = manifest()
    config["cases"][0]["required_gates"] = ["rtx3090_matched_randomness"] if required else []
    sha = scheduler.campaigns.register(config)
    failed = {"manifest_sha256": sha, "passed": False, "evidence": ["3090 randomness differs"]}
    scheduler.campaigns.gate("study", "rtx3090_matched_randomness", failed)
    scheduler.campaigns.storage("study", 0)
    waiver = {"author": "user", "authorization": "i dont care about the matching check",
              "manifest_sha256": sha, "failed_gate_receipt_sha256": digest(failed),
              "reason": "User explicitly waived this matching requirement", "evidence": ["user message"]}
    return scheduler, value, failed, waiver


@pytest.mark.parametrize("required", [False, True])
def test_explicit_user_waiver_preserves_failure_and_only_releases_matching_gate(setup, required):
    scheduler, job, failed, waiver = matching_waiver_fixture(setup, required)
    assert not scheduler.campaigns.eligible(job)[0]
    expected = scheduler.campaigns.waive_gate("study", "rtx3090_matched_randomness", waiver)
    assert scheduler.campaigns.eligible(job)[0]
    assert scheduler.campaigns.waive_gate("study", "rtx3090_matched_randomness", waiver) == expected
    persisted = scheduler.store.db.execute("SELECT receipt FROM campaign_gates WHERE name='rtx3090_matched_randomness'").fetchone()[0]
    assert json.loads(persisted) == failed
    assert scheduler.store.db.execute("SELECT COUNT(*) FROM campaign_gate_waivers").fetchone()[0] == 1
    scheduler.campaigns.gate("study", "numerical_validity", dict(failed, evidence=["nonfinite metrics"]))
    assert not scheduler.campaigns.eligible(job)[0]


@pytest.mark.parametrize("field,value", [("author", "assistant"), ("authorization", "probably okay"),
    ("manifest_sha256", "stale"), ("failed_gate_receipt_sha256", "stale")])
def test_gate_waiver_rejects_false_authority_or_stale_evidence(setup, field, value):
    scheduler, _, _, waiver = matching_waiver_fixture(setup)
    with pytest.raises(ValueError):
        scheduler.campaigns.waive_gate("study", "rtx3090_matched_randomness", dict(waiver, **{field: value}))
    assert scheduler.store.db.execute("SELECT COUNT(*) FROM campaign_gate_waivers").fetchone()[0] == 0


def test_matching_waiver_cannot_waive_other_checks_or_replace_prior_waiver(setup):
    scheduler, _, _, waiver = matching_waiver_fixture(setup)
    with pytest.raises(ValueError, match="only waives"):
        scheduler.campaigns.waive_gate("study", "numerical_validity", waiver)
    scheduler.campaigns.waive_gate("study", "rtx3090_matched_randomness", waiver)
    with pytest.raises(ValueError, match="immutable"):
        scheduler.campaigns.waive_gate("study", "rtx3090_matched_randomness", dict(waiver, reason="replace previous rationale"))


def storage_estimate_receipt(scheduler, **change):
    return {"version": 1, "id": "measured-case-storage", "manifest_sha256": scheduler.campaigns.contract("study")["sha256"],
            "authorization": "Calibrate operational reservation within original resource caps",
            "reason": "Completed-case artifact measurement replaces preliminary storage estimate",
            "evidence": ["completed-case-storage-measurement.json"],
            "storage_estimates": [{"case_id": "one", "previous_bytes": 100, "expected_bytes": 80, "measured_bytes": 70, **change}]}


def test_measured_storage_amendment_updates_live_reservation_without_changing_other_budgets(setup):
    scheduler, now, job = ready(setup)
    scheduler.campaigns.reserve(job)
    original = scheduler.campaigns.contract("study")
    reserved = dict(scheduler.store.db.execute("SELECT * FROM campaign_reservations").fetchone())
    job_before = scheduler.store.job("one")
    receipt = storage_estimate_receipt(scheduler)
    identity = scheduler.campaigns.amend("study", receipt)
    updated = scheduler.campaigns.contract("study")
    current = dict(scheduler.store.db.execute("SELECT * FROM campaign_reservations").fetchone())
    assert current == {**reserved, "storage_bytes": 80}
    assert updated["manifest"] == original["manifest"]
    assert updated["sha256"] == original["sha256"]
    assert updated["started"] == original["started"]
    assert updated["storage_observed"] == original["storage_observed"]
    assert updated["effective_cases"][0] == {**original["effective_cases"][0], "expected_storage_bytes": 80}
    assert scheduler.store.job("one") == job_before
    assert scheduler.campaigns.remaining("study")["storage_bytes_reserved"] == 80
    assert scheduler.campaigns.amend("study", receipt) == identity
    assert len(scheduler.campaigns.contract("study")["amendments"]) == 1


def test_measured_storage_estimate_survives_reopen_and_missing_reservation_recovery(setup, module):
    scheduler, now, job = ready(setup)
    scheduler.campaigns.amend("study", storage_estimate_receipt(scheduler))
    scheduler.store.db.execute("INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at) VALUES(?,?,?,?,?,?,?,?,?)",
                               ("running", "one", "lab", "gpu", "5090", "running", now[0], now[0]+45, now[0]))
    reopened_store = module.Store(scheduler.store.root)
    try:
        reopened = CampaignLedger(reopened_store, lambda: now[0])
        reopened.reconcile_reservations()
        reservation = dict(reopened_store.db.execute("SELECT * FROM campaign_reservations").fetchone())
        assert reservation["storage_bytes"] == 80
        assert reservation["gpu_seconds"] == 900
        assert reopened.contract("study")["manifest"]["cases"][0]["expected_storage_bytes"] == 100
        assert reopened.case("one")["contract"]["expected_storage_bytes"] == 80
    finally:
        reopened_store.close()


@pytest.mark.parametrize("change", [{"previous_bytes": 99}, {"case_id": "unknown"},
    {"expected_bytes": 69}, {"expected_bytes": 0}, {"measured_bytes": -1}, {"expected_bytes": 80.5}])
def test_invalid_measured_storage_estimates_fail_transactionally(setup, change):
    scheduler, _, job = ready(setup)
    scheduler.campaigns.reserve(job)
    before = dict(scheduler.store.db.execute("SELECT * FROM campaign_reservations").fetchone())
    with pytest.raises(ValueError):
        scheduler.campaigns.amend("study", storage_estimate_receipt(scheduler, **change))
    assert scheduler.campaigns.case("one")["contract"]["expected_storage_bytes"] == 100
    assert dict(scheduler.store.db.execute("SELECT * FROM campaign_reservations").fetchone()) == before
    assert scheduler.store.db.execute("SELECT COUNT(*) FROM campaign_amendments").fetchone()[0] == 0


def test_storage_estimates_need_measurement_evidence_and_unique_cases(setup):
    scheduler, _, _ = ready(setup)
    receipt = storage_estimate_receipt(scheduler)
    with pytest.raises(ValueError, match="evidence"):
        scheduler.campaigns.amend("study", {**receipt, "evidence": []})
    with pytest.raises(ValueError, match="unique"):
        scheduler.campaigns.amend("study", {**receipt, "storage_estimates": receipt["storage_estimates"] * 2})


def scientific_extension_fixture(setup, calibration=None):
    scheduler, now, job, manifest = setup
    job("one"); job("upper-lr")
    config = manifest()
    sha = scheduler.campaigns.register(config)
    gate = {"manifest_sha256": sha, "passed": True, "evidence": ["validated"]}
    if calibration:
        gate["progress_calibration"] = calibration
    scheduler.campaigns.gate("study", "handoff", gate)
    scheduler.campaigns.storage("study", 100)
    scheduler.store.db.execute("UPDATE campaign_contracts SET started=? WHERE id='study'", (now[0]-100,))
    spec = scheduler.store.job("upper-lr")["spec"]
    spec["placements"]["lab"]["gpu_models"] = ["RTX 5090"]
    spec["placements"]["lab"]["prepared_files"] = [{"path": "/experiment/lifecycles/upper-lr.json", "sha256": "c"*64}]
    scheduler.store.db.execute("UPDATE jobs SET spec=? WHERE id='upper-lr'", (json.dumps(spec),))
    case = {**config["cases"][0], "id": "upper-lr", "dependencies": ["one"]}
    receipt = {"version": 1, "id": "explicit-upper-lr", "author": "user", "authorization":
        "please schedule also 5e-3 on the 5090s -- so that we dont finish on the upper bound of good performance",
        "manifest_sha256": sha, "reason": "Test the explicitly requested higher learning rate",
        "evidence": ["user-message", "lifecycle-config.json"], "template_case_id": "one", "cases": [case],
        "scientific_inputs": {"upper-lr": [{"kind": "lifecycle", "sha256": "c"*64}]}, "storage_bytes": 1200}
    return scheduler, now, receipt


def test_scientific_extension_preserves_old_science_and_only_adds_storage_budget(setup, module):
    scheduler, now, receipt = scientific_extension_fixture(setup)
    old = scheduler.campaigns.contract("study")
    old_case = scheduler.campaigns.case("one")
    old_job = scheduler.store.job("one")
    gate = scheduler.store.db.execute("SELECT receipt FROM campaign_gates").fetchone()[0]
    sha = scheduler.campaigns.extend("study", receipt)
    after = scheduler.campaigns.contract("study")
    assert after["manifest"] == old["manifest"]
    assert after["sha256"] == old["sha256"]
    assert after["started"] == old["started"]
    assert after["effective_budgets"] == {**old["manifest"]["budgets"], "storage_bytes": 1200}
    assert scheduler.campaigns.case("one") == old_case
    assert scheduler.store.job("one") == old_job
    assert scheduler.store.db.execute("SELECT receipt FROM campaign_gates").fetchone()[0] == gate
    assert scheduler.store.job("upper-lr")["state"] == "held"
    assert scheduler.campaigns.extend("study", receipt) == sha
    assert len(scheduler.campaigns.contract("study")["scientific_extensions"]) == 1
    reopened_store = module.Store(scheduler.store.root)
    try:
        reopened = CampaignLedger(reopened_store, lambda: now[0])
        assert reopened.case("upper-lr")["contract"] == receipt["cases"][0]
        assert reopened.contract("study")["effective_budgets"]["storage_bytes"] == 1200
        assert reopened.remaining("study")["gpu_seconds_available"] == 3600
        assert reopened.remaining("study")["window_seconds_remaining"] == 43100
    finally:
        reopened_store.close()


@pytest.mark.parametrize("bad", ["authority", "authorization", "hash", "active", "old_case", "gate", "dependency", "gpu", "cap", "window_budget"])
def test_scientific_extension_rejects_invalid_authority_inputs_gates_and_resources(setup, bad):
    scheduler, _, receipt = scientific_extension_fixture(setup)
    if bad == "authority":
        receipt["author"] = "assistant"
    elif bad == "authorization":
        receipt["authorization"] = ""
    elif bad == "hash":
        receipt["scientific_inputs"]["upper-lr"][0]["sha256"] = "d"*64
    elif bad == "active":
        scheduler.store.update_job("upper-lr", "running")
    elif bad == "old_case":
        receipt["cases"][0]["id"] = "one"
    elif bad == "gate":
        receipt["cases"][0]["required_gates"] = []
    elif bad == "dependency":
        receipt["cases"][0]["dependencies"] = ["not-declared"]
    elif bad == "gpu":
        spec = scheduler.store.job("upper-lr")["spec"]
        spec["placements"]["lab"]["gpu_models"] = ["RTX 3090"]
        scheduler.store.db.execute("UPDATE jobs SET spec=? WHERE id='upper-lr'", (json.dumps(spec),))
    elif bad == "cap":
        receipt["storage_bytes"] = 999
    elif bad == "window_budget":
        receipt["window_seconds"] = 100000
    with pytest.raises(ValueError):
        scheduler.campaigns.extend("study", receipt)
    assert scheduler.campaigns.case("upper-lr") is None
    assert scheduler.store.db.execute("SELECT COUNT(*) FROM campaign_extensions").fetchone()[0] == 0
    assert scheduler.campaigns.contract("study")["effective_budgets"]["storage_bytes"] == 1000


def test_storage_admission_uses_extended_cap_without_changing_original_manifest(setup):
    scheduler, _, receipt = scientific_extension_fixture(setup)
    scheduler.campaigns.extend("study", receipt)
    scheduler.store.update_job("one", "complete")
    scheduler.campaigns.storage("study", 1050)
    assert scheduler.campaigns.eligible(scheduler.store.job("upper-lr"))[0]
    scheduler.campaigns.storage("study", 1150)
    assert "storage budget" in scheduler.campaigns.eligible(scheduler.store.job("upper-lr"))[1]


def test_user_gpu_hour_extension_preserves_charges_window_and_survives_restart(setup, module):
    scheduler, now, receipt = scientific_extension_fixture(setup)
    scheduler.store.db.execute("UPDATE jobs SET used_seconds=321 WHERE id='one'")
    original = scheduler.campaigns.contract("study")
    receipt["gpu_hours"] = 20
    sha = scheduler.campaigns.extend("study", receipt)
    after = scheduler.campaigns.contract("study")
    assert after["manifest"] == original["manifest"]
    assert after["sha256"] == original["sha256"]
    assert after["started"] == original["started"]
    assert after["effective_budgets"]["window_seconds"] == original["effective_budgets"]["window_seconds"]
    assert after["effective_budgets"]["gpu_hours"] == 20
    assert scheduler.campaigns.extend("study", receipt) == sha
    reopened_store = module.Store(scheduler.store.root)
    try:
        reopened = CampaignLedger(reopened_store, lambda: now[0])
        usage = reopened.remaining("study")
        assert usage["gpu_seconds_charged"] == 321
        assert usage["gpu_seconds_available"] == 20 * 3600 - 321
        assert usage["window_seconds_remaining"] == 43100
        assert reopened.contract("study")["manifest"]["budgets"]["gpu_hours"] == 1
        event = json.loads(reopened_store.db.execute("SELECT detail FROM events WHERE kind='campaign_scientific_extension_by_user'").fetchone()[0])
        assert event["previous_gpu_hours"] == 1
        assert event["gpu_hours"] == 20
        assert len(reopened.contract("study")["scientific_extensions"]) == 1
    finally:
        reopened_store.close()


@pytest.mark.parametrize("bad_hours", [float("nan"), float("inf"), 1e308, True, False, 0, -1, .5, "20"])
def test_user_gpu_hour_extension_rejects_nonfinite_boolean_or_decreased_caps(setup, bad_hours):
    scheduler, _, receipt = scientific_extension_fixture(setup)
    receipt["gpu_hours"] = bad_hours
    with pytest.raises(ValueError):
        scheduler.campaigns.extend("study", receipt)
    assert scheduler.campaigns.contract("study")["effective_budgets"]["gpu_hours"] == 1
    assert scheduler.store.db.execute("SELECT COUNT(*) FROM campaign_extensions").fetchone()[0] == 0


def test_gpu_hour_increase_still_requires_user_authority(setup):
    scheduler, _, receipt = scientific_extension_fixture(setup)
    receipt.update(gpu_hours=20, author="assistant")
    with pytest.raises(ValueError, match="user authority"):
        scheduler.campaigns.extend("study", receipt)


def test_long_case_waits_for_both_predecessors_and_retains_mandatory_gates(setup):
    scheduler, now, receipt = scientific_extension_fixture(setup,
        calibration={"training": {"startup_seconds": 600, "stale_seconds": 600, "expected_seconds": 3600}})
    scheduler.campaigns.extend("study", receipt)
    _, _, make_job, _ = setup
    make_job("longer")
    spec = scheduler.store.job("longer")["spec"]
    spec["placements"]["lab"]["gpu_models"] = ["RTX 5090"]
    spec["placements"]["lab"]["prepared_files"] = [{"path": "/experiment/lifecycles/longer.json", "sha256": "d"*64}]
    scheduler.store.db.execute("UPDATE jobs SET spec=? WHERE id='longer'", (json.dumps(spec),))
    long = {**receipt, "id": "long-duration", "gpu_hours": 20,
        "cases": [{**receipt["cases"][0], "id": "longer", "dependencies": ["one", "upper-lr"],
                   "progress": {"startup_seconds": 600, "stale_seconds": 600, "expected_seconds": 7200}}],
        "scientific_inputs": {"longer": [{"kind": "lifecycle", "sha256": "d"*64}]}}
    scheduler.campaigns.extend("study", long)
    assert scheduler.campaigns.progress_expectation("one", "lab")["expected_seconds"] == 3600
    assert scheduler.campaigns.progress_expectation("longer", "lab")["expected_seconds"] == 7200
    job = scheduler.store.job("longer")
    assert "dependency" in scheduler.campaigns.eligible(job)[1]
    scheduler.store.update_job("one", "complete")
    scheduler.store.update_job("upper-lr", "collecting")
    assert "upper-lr" in scheduler.campaigns.eligible(job)[1]
    scheduler.store.update_job("upper-lr", "complete")
    assert scheduler.campaigns.eligible(job)[0]
    # Replaying the extension after predecessor completion cannot add a second
    # case or attempt; failed numerical gates still prevent any admission.
    scheduler.campaigns.extend("study", long)
    assert len([case for case in scheduler.campaigns.contract("study")["effective_cases"] if case["id"] == "longer"]) == 1
    assert not scheduler.store.attempts()
    scheduler.campaigns.gate("study", "mandatory-numerical", {"manifest_sha256": receipt["manifest_sha256"], "passed": False, "evidence": ["nonfinite validation"]})
    assert "mandatory" in scheduler.campaigns.eligible(job)[1]


def prepared_recovery_placement(setup, tmp_path):
    scheduler, _, job, manifest = setup
    value = job()
    placement = value['spec']['placements']['lab']
    placement['recovery_input_contract'] = {'artifacts': {role: {'path':str(tmp_path/role),'sha256':'1'*64}
        for role in ('teacher_weights','device_data','device_model','device_state')},
        'array_seed':271001,'apparent_sha256':{'faulty':'2'*64}}
    placement['prepared_files'] = [{'path':str(tmp_path/'campaigns/c/lifecycles/frozen.json'),'sha256':'3'*64}]
    value['spec']['placements']['second'] = json.loads(json.dumps(placement))
    scheduler.store.db.execute('UPDATE jobs SET spec=? WHERE id=?',(json.dumps(value['spec']),'one'))
    sha = scheduler.campaigns.register(manifest())
    staged = json.loads(json.dumps(placement)); staged['workdir']=str(tmp_path/'new-host')
    staged['argv']=['python','-m','campaigns.tmux_worker','bridge','--spec',str(tmp_path/'new-host/worker.json')]
    staged['run_dir']=str(tmp_path/'new-host/receipts')
    staged['recovery_input_contract']['artifacts']['device_state']['path']=str(tmp_path/'new-host/p0.pt')
    proof=tmp_path/'new-prepared.json'
    proof.write_text(json.dumps({'host':'spare','cases':[{'id':'one','dry_run_passed':True,'placement':staged}]}))
    return {'version':1,'id':'add-spare','manifest_sha256':sha,'authorization':'same approved GPU scope',
        'reason':'original hosts occupied','evidence':[str(proof)],'prepared_placements':[
        {'job_id':'one','to_target':'spare','to_canonical_host':'spare',
         'prepared_receipt':{'path':str(proof),'sha256':hashlib.sha256(proof.read_bytes()).hexdigest()}}]}


def test_add_prepared_recovery_host_preserves_job_identity_and_budget(setup,tmp_path):
    amendment=prepared_recovery_placement(setup,tmp_path); scheduler=setup[0]
    before=scheduler.store.job('one')
    sha=scheduler.campaigns.amend('study',amendment)
    assert scheduler.campaigns.amend('study',amendment)==sha
    after=scheduler.store.job('one')
    assert set(after['spec']['placements'])=={'lab','second','spare'}
    assert after['used_seconds']==0 and after['state']=='held'
    assert {k:v for k,v in before['spec'].items() if k!='placements'}=={k:v for k,v in after['spec'].items() if k!='placements'}
    assert after['spec']['placements']['lab']==before['spec']['placements']['lab']


@pytest.mark.parametrize('failure',['input','lifecycle','receipt','runnable','history'])
def test_add_prepared_recovery_host_rejects_scientific_drift_or_started_jobs(setup,tmp_path,failure):
    amendment=prepared_recovery_placement(setup,tmp_path); scheduler=setup[0]
    proof=tmp_path/'new-prepared.json'; value=json.loads(proof.read_text())
    if failure in ('input','lifecycle'):
        p=value['cases'][0]['placement']
        if failure=='input':p['recovery_input_contract']['artifacts']['device_state']['sha256']='4'*64
        else:p['prepared_files'][0]['sha256']='4'*64
        proof.write_text(json.dumps(value))
        amendment['prepared_placements'][0]['prepared_receipt']['sha256']=hashlib.sha256(proof.read_bytes()).hexdigest()
    elif failure=='receipt':proof.write_text('{}')
    elif failure=='runnable':scheduler.store.update_job('one','queued')
    else:
        scheduler.store.db.execute('INSERT INTO attempts(token,job_id,target,gpu_uuid,gpu_name,state,issued,launch_before,progress_at) VALUES(?,?,?,?,?,?,?,?,?)',
            ('old','one','lab','gpu','3090','failed',900,945,900))
    with pytest.raises(ValueError):scheduler.campaigns.amend('study',amendment)
    assert set(scheduler.store.job('one')['spec']['placements'])=={'lab','second'}
