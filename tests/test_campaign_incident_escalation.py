"""Failure/restart tests for opt-in escalation; no model or GPU calls."""
import json
import hashlib
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ebl.execution.incident_escalation import IncidentEscalator, repair_command, validate_settings


@pytest.fixture
def harness(tmp_path):
    db = sqlite3.connect(tmp_path / "queue.sqlite3", isolation_level=None)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE campaign_incidents (id INTEGER PRIMARY KEY, campaign_id TEXT,
            job_id TEXT, created REAL, state TEXT, kind TEXT);
        CREATE TABLE campaign_investigations (id TEXT, incident_id INTEGER, state TEXT);
        CREATE TABLE recovery (id TEXT, state TEXT);
        CREATE TABLE campaign_retries (job_id TEXT, at REAL);
    """)
    now = [1000.0]
    living, commands, kills = set(), [], []
    settings = {"campaigns": {"prep": "authorized preparation", "main": "authorized main study"},
                "not_before": 900, "timeout_seconds": 300, "max_per_incident": 2,
                "max_per_day": 6, "cooldown_seconds": 1800}
    recovery = {"command": ["codex"], "model": "configured-model", "effort": "medium"}
    ledger = SimpleNamespace(db=db, store=SimpleNamespace(root=tmp_path), clock=lambda: now[0],
        snapshot=lambda incident: {"incident": dict(incident), "job": {"spec": {"project_dir": str(tmp_path)}},
                                   "remaining_budget": {"gpu_seconds_available": 100}})

    def popen(command, **kwargs):
        pid = 100 + len(commands)
        commands.append(command)
        living.add(pid)
        return SimpleNamespace(pid=pid, wait=lambda timeout: 0)

    def make(config=settings):
        return IncidentEscalator(ledger, config, recovery, tmp_path / "queue.json", popen=popen,
            alive=lambda p: p["pid"] in living, identity=lambda pid: {"pid": pid, "start_ticks": "1", "boot_id": "boot"},
            killpg=lambda pid, sig: kills.append((pid, sig)))

    def incident(i=1, campaign="main", created=950, diagnosis="finished", kind="failure"):
        db.execute("INSERT INTO campaign_incidents VALUES(?,?,?,?,?,?)", (i, campaign, f"job-{i}", created, "open", kind))
        if diagnosis:
            db.execute("INSERT INTO campaign_investigations VALUES(?,?,?)", (f"diag-{i}", i, diagnosis))

    def finish(owner, outcome="no_action", receipt=False):
        row = owner.rows()[-1]
        living.discard(row["payload"]["process"]["pid"])
        Path(row["payload"]["folder"], "report.json").write_text(json.dumps({"outcome": outcome, "summary": "evidence checked"}))
        Path(row["payload"]["folder"], "terminal.json").write_text(json.dumps({"group_cleanup": True, "returncode": 0, "reason": "exited"}))
        if receipt:
            db.execute("INSERT INTO campaign_retries VALUES(?,?)", (row["job_id"], now[0]))

    yield SimpleNamespace(**locals())
    db.close()


@pytest.mark.parametrize("bad", [
    {"campaigns": {}}, {"campaigns": {"main": ""}}, {"not_before": float("nan")},
    {"max_per_day": 7}, {"max_per_incident": True}, {"timeout_seconds": 301},
    {"timeout_seconds": 1.5}, {"unexpected": True},
    {"allow_code_repairs": "true"}, {"allow_code_repairs": 1},
])
def test_configuration_fails_closed(harness, bad):
    with pytest.raises(ValueError):
        validate_settings({**harness.settings, **bad})


def test_disabled_and_campaign_filter_and_historical_incidents(harness):
    h = harness
    h.incident(campaign="unrelated")
    h.incident(2, created=800)
    h.incident(3, campaign="prep")
    h.make(None).cycle()
    assert not h.commands
    owner = h.make()
    owner.cycle()
    assert [r["incident_id"] for r in owner.rows()] == [3]


def test_waits_for_diagnosis_and_serializes_with_existing_owners(harness):
    h = harness
    h.incident(diagnosis="running")
    owner = h.make()
    owner.cycle()
    assert not h.commands
    h.db.execute("UPDATE campaign_investigations SET state='failed'")
    h.db.execute("INSERT INTO recovery VALUES('other','running')")
    owner.cycle()
    assert not h.commands
    h.db.execute("UPDATE recovery SET state='finished'")
    owner.cycle()
    assert len(h.commands) == 1


def test_restart_keeps_claim_and_report_never_reexecutes(harness):
    h = harness
    h.incident()
    original = h.make()
    original.cycle()
    restarted = h.make()
    restarted.cycle()
    assert restarted.busy()
    assert len(h.commands) == 1
    h.finish(restarted)
    restarted.cycle()
    h.now[0] += 4000
    h.make().cycle()
    assert restarted.rows()[0]["state"] == "reported"
    assert len(h.commands) == 1
    # Neither diagnosis nor incident/worker state was modified by report parsing.
    assert h.db.execute("SELECT state FROM campaign_incidents").fetchone()[0] == "open"
    assert h.db.execute("SELECT state FROM campaign_investigations").fetchone()[0] == "finished"


def test_spawn_receipt_gap_is_uncertain_and_never_retried(harness):
    h = harness
    h.incident()
    owner = h.make()
    owner.cycle()
    row = owner.rows()[0]
    row["payload"]["process"] = None
    owner.save(row)
    restarted = h.make()
    restarted.cycle()
    h.now[0] += 100000
    restarted.cycle()
    assert restarted.rows()[0]["state"] == "uncertain"
    assert len(h.commands) == 1


def test_timeout_signals_only_verified_owner_and_survives_disable(harness):
    h = harness
    h.incident()
    owner = h.make()
    owner.cycle()
    disabled = h.make(None)
    h.now[0] += 331
    disabled.cycle()
    assert h.kills == [(100, signal.SIGTERM)]
    h.now[0] += 11
    disabled.cycle()
    assert h.kills[-1] == (100, signal.SIGKILL)
    h.living.clear()
    disabled.cycle()
    assert disabled.rows()[0]["state"] == "uncertain"
    assert len(h.commands) == 1


@pytest.mark.parametrize("receipt,expected", [(False, "failed"), (True, "reported")])
def test_recovery_report_requires_real_classified_receipt(harness, receipt, expected):
    h = harness
    h.incident()
    owner = h.make()
    owner.cycle()
    h.finish(owner, "recovery_requested", receipt=receipt)
    owner.cycle()
    assert owner.rows()[0]["state"] == expected


def test_snapshot_tampering_cannot_be_reported_as_success(harness):
    h = harness
    h.incident()
    owner = h.make()
    owner.cycle()
    h.finish(owner)
    path = Path(owner.rows()[0]["payload"]["folder"], "snapshot.json")
    path.chmod(0o600)
    path.write_text("{}")
    owner.cycle()
    assert owner.rows()[0]["state"] == "failed"
    assert "snapshot changed" in owner.rows()[0]["payload"]["error"]


def test_failed_spawn_budgets_and_cooldown_survive_restart(harness):
    h = harness
    h.incident()
    owner = h.make()
    def fail(*args, **kwargs):
        raise OSError("launcher unavailable")
    owner.popen = fail
    owner.cycle()
    assert owner.rows()[0]["state"] == "failed"
    restarted = h.make()
    restarted.cycle()
    assert not h.commands
    h.now[0] += 1800
    restarted.cycle()
    assert len(restarted.rows()) == 2
    h.finish(restarted, "invalid")
    restarted.cycle()
    h.now[0] += 1800
    h.make().cycle()
    assert len(restarted.rows()) == 2


def test_daily_limit_applies_across_both_phases(harness):
    h = harness
    settings = {**h.settings, "max_per_day": 1}
    h.incident(campaign="prep")
    h.incident(2, campaign="main")
    owner = h.make(settings)
    owner.cycle()
    h.finish(owner)
    h.make(settings).cycle()
    assert len(h.commands) == 1


def test_blocked_does_not_retry_and_prompt_restricts_authority(harness):
    h = harness
    h.incident()
    owner = h.make()
    owner.cycle()
    folder = Path(owner.rows()[0]["payload"]["folder"])
    prompt = (folder / "prompt.txt").read_text()
    for text in ("Do not change code", "recover-driver-crash", "recover-missing-tmux", "never bypass their guards"):
        assert text in prompt
    command = json.loads((folder / "launch.json").read_text())["argv"]
    assert command[:4] == ["timeout", "--signal=TERM", "--kill-after=10s", "300s"]
    assert command[command.index("-m") + 1] == "configured-model"
    assert "--approve-for-me" in command
    assert "--sandbox" not in command
    h.finish(owner, "blocked")
    owner.cycle()
    h.now[0] += 100000
    h.make().cycle()
    assert len(h.commands) == 1


def repair_evidence(owner):
    row = owner.rows()[-1]
    folder = Path(row["payload"]["folder"])
    def artifact(name, content):
        (folder / name).write_text(content)
        return {"path": name, "sha256": hashlib.sha256(content.encode()).hexdigest()}
    receipt = {"version": 1, **{k: row[k] for k in ("incident_id", "campaign_id", "job_id")},
               "source_before": "original-source", "source_after": "tested-source",
               "patch": artifact("repair.patch", "retained operational patch\n"),
               "tests": [{"argv": ["python", "-m", "pytest"], "returncode": 0,
                          "log": artifact("tests.log", "1 passed\n")}],
               "verification": {"argv": ["python", "probe.py"], "returncode": 0,
                                "log": artifact("probe.log", "fresh observation succeeded\n")}}
    (folder / "repair.json").write_text(json.dumps(receipt))
    return folder, receipt


def test_code_repairs_are_opt_in_and_report_requires_authorization(harness):
    h = harness
    h.incident()
    owner = h.make()
    owner.cycle()
    repair_evidence(owner)
    h.finish(owner, "repair_applied")
    owner.cycle()
    assert owner.rows()[0]["state"] == "failed"
    assert "not authorized" in owner.rows()[0]["payload"]["error"]


@pytest.mark.parametrize("fault", [None, "structured_source", "same_structured_revision", "missing", "wrong_incident", "failed_test", "failed_probe",
                                  "tamper", "escape", "same_source", "no_tests", "malformed"])
def test_code_repair_requires_bound_evidence_and_successful_verification(harness, fault):
    h = harness
    h.incident()
    settings = {**h.settings, "allow_code_repairs": True}
    owner = h.make(settings)
    owner.cycle()
    folder, receipt = repair_evidence(owner)
    prompt = (folder / "prompt.txt").read_text()
    assert "Do not change code" not in prompt
    for text in ("isolated checkout", "Preserve dirty user work", "scientific configs",
                 "remaining budget", "recover-resource-preemption", "post-deployment probe"):
        assert text in prompt
    command = json.loads((folder / "launch.json").read_text())["argv"]
    assert command[command.index("--add-dir") + 1] == str(folder)
    if fault == "wrong_incident": receipt["incident_id"] = 999
    if fault == "failed_test": receipt["tests"][0]["returncode"] = 1
    if fault == "failed_probe": receipt["verification"]["returncode"] = 1
    if fault == "tamper": (folder / "tests.log").write_text("changed")
    if fault == "escape": receipt["patch"]["path"] = "../outside.patch"
    if fault == "same_source": receipt["source_after"] = receipt["source_before"]
    if fault in {"structured_source", "same_structured_revision"}:
        receipt["source_before"] = {"path": "/source", "revision": "original"}
        receipt["source_after"] = {"path": "/isolated", "revision": "repaired" if fault == "structured_source" else "original"}
    if fault == "no_tests": receipt["tests"] = []
    if fault == "malformed": receipt = []
    (folder / "repair.json").write_text(json.dumps(receipt))
    if fault == "missing": (folder / "repair.json").unlink()
    h.finish(owner, "repair_applied")
    # The launch's authority survives a controller restart/config disable.
    restarted = h.make(None)
    restarted.cycle()
    row = restarted.rows()[0]
    valid = fault in {None, "structured_source"}
    assert row["state"] == ("reported" if valid else "failed")
    if valid:
        assert row["payload"]["repair"] == receipt
    assert h.db.execute("SELECT COUNT(*) FROM campaign_retries").fetchone()[0] == 0
    assert h.db.execute("SELECT state FROM campaign_incidents").fetchone()[0] == "open"


def test_dead_supervisor_without_terminal_receipt_retains_ownership(harness):
    h = harness
    h.incident()
    owner = h.make()
    owner.cycle()
    h.living.clear()
    restarted = h.make()
    restarted.cycle()
    assert restarted.rows()[0]["state"] == "uncertain"
    h.now[0] += 100000
    restarted.cycle()
    assert len(h.commands) == 1


def test_failure_after_report_cannot_claim_success(harness):
    h = harness
    h.incident()
    owner = h.make()
    owner.cycle()
    h.finish(owner)
    Path(owner.rows()[0]["payload"]["folder"], "terminal.json").write_text(
        json.dumps({"group_cleanup": True, "returncode": 1, "reason": "exited"}))
    owner.cycle()
    assert owner.rows()[0]["state"] == "failed"


@pytest.mark.parametrize("hang", [False, True])
def test_real_supervisor_publishes_exit_and_timeout_receipts(tmp_path, hang):
    """Exercise actual detached subprocess supervision without invoking Codex."""
    import ebl.execution.incident_escalation_worker as incident_escalation_worker
    script = "import time; time.sleep(30)" if hang else "print('completed bounded owner')"
    request = {"folder": str(tmp_path), "project_dir": str(tmp_path), "timeout_seconds": 0.2,
               "argv": [sys.executable, "-c", script]}
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(request))
    (tmp_path / "prompt.txt").write_text("test")
    completed = subprocess.run([sys.executable, incident_escalation_worker.__file__, str(path)], timeout=15)
    assert completed.returncode == 0
    receipt = json.loads((tmp_path / "terminal.json").read_text())
    assert receipt["group_cleanup"] is True
    assert receipt["reason"] == ("timeout" if hang else "exited")
    assert (receipt["returncode"] == 0) is (not hang)
    assert receipt["elapsed_seconds"] < 12


def test_reopening_database_retains_daily_claims(harness):
    h = harness
    h.incident()
    owner = h.make()
    owner.cycle()
    # A second connection observes the reservation before worker completion.
    db = sqlite3.connect(h.tmp_path / "queue.sqlite3")
    try:
        assert db.execute("SELECT incident_id,state FROM campaign_escalations").fetchall() == [(1, "running")]
    finally:
        db.close()


def test_installed_cli_accepts_owner_flags_before_schema_loading(tmp_path):
    """--help alone skips conflict checks, so also fail at a missing schema.

    Both paths terminate before starting a model or admitting any workload.
    The negative control reproduces the exact live canary's argument error.
    """
    codex = Path.home() / ".local/bin/codex"
    if not codex.exists():
        pytest.skip("installed Codex CLI is required for parser regression")
    recovery = {"command": [str(codex)], "model": "configured-model", "effort": "medium"}
    command = repair_command(recovery, {"timeout_seconds": 10}, tmp_path, tmp_path)[4:]
    help_result = subprocess.run([*command, "--help"], stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, timeout=10)
    assert help_result.returncode == 0, help_result.stderr
    assert not (tmp_path / "schema.json").exists()
    result = subprocess.run(command, input="Parser regression only.", capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    assert "Failed to read output schema file" in result.stderr
    assert "cannot be used with" not in result.stderr
    incompatible = [*command[:-1], "--sandbox", "workspace-write", command[-1]]
    negative = subprocess.run(incompatible, input="Parser regression only.", capture_output=True, text=True, timeout=10)
    assert negative.returncode == 2
    assert "cannot be used with" in negative.stderr
