"""Completion delivery, receipt checks and restart behavior without a live chat."""
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest

from ebl.execution.completion_notifications import CompletionNotifier, validate_settings


THREAD = "01a139d5-3f18-7ef1-9e6e-5c611900c843"
OTHER_THREAD = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
def h(tmp_path):
    db = sqlite3.connect(tmp_path / "queue.sqlite3", isolation_level=None)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE jobs (id TEXT PRIMARY KEY, spec TEXT, state TEXT, reason TEXT, used_seconds REAL);
        CREATE TABLE attempts (token TEXT, job_id TEXT, state TEXT, issued REAL, receipt TEXT);
        CREATE TABLE collections (token TEXT, job_id TEXT, state TEXT, issued REAL, payload TEXT);
    """)
    report = tmp_path / "summary.json"
    report.write_text(json.dumps({"metric": 0.25, "coverage": 2}))
    spec = {"handoff": str(tmp_path / "experiment.md"), "project_dir": str(tmp_path),
            "placements": {"lab": {"run_dir": str(tmp_path / "runs")}},
            "completion": {"workdir": str(tmp_path), "argv": ["collect"]}}
    for name in ("one", "two"):
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?)", (name, json.dumps(spec), "running", "", 10))
    now, calls = [1000.0], []

    def job(name):
        row = db.execute("SELECT * FROM jobs WHERE id=?", (name,)).fetchone()
        return {**dict(row), "spec": json.loads(row["spec"])} if row else None

    def contract(campaign_id):
        if campaign_id != "study":
            raise ValueError("unknown campaign")
        return {"effective_cases": [{"id": "one"}, {"id": "two"}]}

    ledger = SimpleNamespace(db=db, clock=lambda: now[0], contract=contract,
                             store=SimpleNamespace(root=tmp_path, job=job))
    settings = {"enabled": True, "command": ["codex"], "timeout_seconds": 10,
                "retry_seconds": 60, "max_attempts": 2}

    def send(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="queued", stderr="")

    def make(config=settings, run=send):
        return CompletionNotifier(ledger, config, tmp_path / "queue.json", run=run)

    def complete(name, *, collected=True, receipt_ok=True):
        observed = {"alive": False, "reservation_held": False,
                    "receipt": {"state": "complete", "run_exit_code": 0,
                                "validation_exit_code": 0 if receipt_ok else 1}}
        db.execute("UPDATE jobs SET state='complete' WHERE id=?", (name,))
        db.execute("INSERT INTO attempts VALUES(?,?,?,?,?)", (name, name, "complete", now[0], json.dumps(observed)))
        if collected:
            db.execute("INSERT INTO collections VALUES(?,?,?,?,?)", (name, name, "complete", now[0], json.dumps(observed)))

    def delivery():
        row = db.execute("SELECT * FROM campaign_completion_deliveries").fetchone()
        return dict(row) if row else None

    yield SimpleNamespace(**locals())
    db.close()


@pytest.mark.parametrize("settings", [{"enabled": "yes"}, {"command": "codex"}, {"timeout_seconds": 31},
                                      {"max_attempts": True}, {"unexpected": True}])
def test_invalid_delivery_settings(settings):
    with pytest.raises(ValueError):
        validate_settings(settings)


def test_registration_and_results_paths_are_idempotent(h):
    owner = h.make()
    watch = owner.register("study", THREAD)
    assert owner.register("study", THREAD, report_paths=[h.report]) == watch
    assert owner.register("study", THREAD) == watch
    assert h.db.execute("SELECT COUNT(*) FROM campaign_completion_watches").fetchone()[0] == 1
    assert json.loads(h.db.execute("SELECT report_paths FROM campaign_completion_watches").fetchone()[0]) == [str(h.report)]
    with pytest.raises(ValueError): owner.register("study", "not-a-uuid")
    with pytest.raises(ValueError): owner.register("study", THREAD, ["other-job"])
    with pytest.raises(ValueError): owner.register("unknown-campaign", THREAD)


def test_waits_for_every_selected_case_and_never_forwards_intermediate_collection(h):
    owner = h.make()
    owner.register("study", THREAD, report_paths=[h.report])
    h.complete("one")
    owner.cycle()
    assert not h.calls
    h.db.execute("UPDATE jobs SET state='collecting' WHERE id='two'")
    owner.cycle()
    assert not h.calls
    h.complete("two")
    owner.cycle()
    assert len(h.calls) == 1
    argv = h.calls[0]
    assert argv[:4] == ["codex", "queue", "--thread", THREAD]
    assert "--model" not in argv and "resume" not in argv
    bundle = json.loads(Path(h.delivery()["bundle"]).read_text())
    assert bundle["outcome"] == "ready_for_review"
    assert bundle["coverage"] == {"complete": 2, "expected": 2}
    assert bundle["reports"][0]["path"] == str(h.report)
    h.make().cycle()
    assert len(h.calls) == 1
    assert h.delivery()["state"] == "queued"
    assert [h.job(j)["state"] for j in ("one", "two")] == ["complete", "complete"]


@pytest.mark.parametrize("fault", ["collection", "validation", "missing_report", "bad_receipt"])
def test_bad_completion_evidence_wakes_review_without_claiming_valid_success(h, fault):
    owner = h.make()
    owner.register("study", THREAD, report_paths=[h.report])
    h.complete("one", collected=fault != "collection", receipt_ok=fault != "validation")
    h.complete("two")
    if fault == "missing_report": h.report.unlink()
    if fault == "bad_receipt": h.db.execute("UPDATE attempts SET receipt='[]' WHERE job_id='one'")
    owner.cycle()
    bundle = json.loads(Path(h.delivery()["bundle"]).read_text())
    assert bundle["outcome"] == "needs_verification" and bundle["problems"]


def test_cancelled_coverage_is_partial_and_attention_is_not_terminal(h):
    owner = h.make()
    owner.register("study", THREAD)
    h.complete("one")
    h.db.execute("UPDATE jobs SET state='attention' WHERE id='two'")
    owner.cycle()
    assert not h.calls
    h.db.execute("UPDATE jobs SET state='cancelled' WHERE id='two'")
    owner.cycle()
    assert json.loads(Path(h.delivery()["bundle"]).read_text())["outcome"] == "finished_incomplete"


def test_already_finished_campaign_needs_explicit_backfill(h):
    h.complete("one"); h.complete("two")
    owner = h.make()
    with pytest.raises(ValueError, match="include-existing"):
        owner.register("study", THREAD)
    owner.register("study", THREAD, include_existing=True)
    owner.cycle()
    assert len(h.calls) == 1


def test_failed_delivery_retries_with_persistent_limit_and_cooldown(h):
    def fail(argv, **kwargs):
        h.calls.append(argv)
        return SimpleNamespace(returncode=1, stdout="", stderr="app-server unavailable")
    owner = h.make(run=fail)
    owner.register("study", THREAD)
    h.complete("one"); h.complete("two")
    owner.cycle()
    assert h.delivery()["state"] == "failed"
    h.make(run=fail).cycle()
    assert len(h.calls) == 1
    h.now[0] += 61
    h.make(run=fail).cycle()
    h.now[0] += 1000
    h.make(run=fail).cycle()
    assert len(h.calls) == 2 and h.delivery()["attempts"] == 2


@pytest.mark.parametrize("crash", [True, False])
def test_ambiguous_acceptance_is_not_resent_and_late_ack_resolves_it(h, crash):
    def send(argv, **kwargs):
        if crash:
            raise KeyboardInterrupt("simulate service death after reservation")
        raise subprocess.TimeoutExpired(argv, 10)
    owner = h.make(run=send)
    event = owner.register("study", THREAD)
    h.complete("one"); h.complete("two")
    if crash:
        with pytest.raises(KeyboardInterrupt): owner.cycle()
    else:
        owner.cycle()
    restarted = h.make()
    restarted.cycle()
    assert h.delivery()["state"] == "uncertain" and not h.calls
    assert not restarted.acknowledge(event, THREAD, "received")["already_acknowledged"]
    assert restarted.acknowledge(event, THREAD, "duplicate")["already_acknowledged"]
    with pytest.raises(ValueError): restarted.acknowledge(event, OTHER_THREAD, "wrong chat")


def test_acknowledgement_during_send_is_not_overwritten(h):
    owner = h.make()
    event = owner.register("study", THREAD)
    h.complete("one"); h.complete("two")
    def send(argv, **kwargs):
        owner.acknowledge(event, THREAD, "fast receiver")
        return SimpleNamespace(returncode=0, stdout="queued", stderr="")
    owner.run = send
    owner.cycle()
    assert h.delivery()["state"] == "acknowledged"


def test_disabled_delivery_never_sends(h):
    owner = h.make()
    owner.register("study", THREAD)
    h.complete("one"); h.complete("two")
    h.make(None).cycle()
    assert not h.calls
