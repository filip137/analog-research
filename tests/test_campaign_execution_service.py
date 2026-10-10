"""Durability and admission tests against the actual installed run-watch store."""
from __future__ import annotations

import json
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
