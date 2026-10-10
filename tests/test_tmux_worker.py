from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from campaigns import tmux_worker as worker


@pytest.fixture
def spec_factory(tmp_path):
    specifications = []

    def make(program, **overrides):
        spec_path = tmp_path / f"spec-{len(specifications)}.json"
        raw = {"case_id": "hwa-readiness", "state_root": str(tmp_path / "workers"),
               "cwd": str(tmp_path), "argv": [sys.executable, "-u", "-c", program],
               "timeout_seconds": 10, "lease_seconds": 10, "poll_seconds": .03,
               "termination_grace_seconds": .1, "tmux_socket": str(tmp_path / "tmux.sock"),
               **overrides}
        spec_path.write_text(json.dumps(raw))
        spec, directory = worker.load_spec(spec_path, f"attempt{len(specifications)}")
        specifications.append((spec, directory))
        return spec_path, spec, directory

    yield make
    for spec, directory in specifications:
        if (directory / "receipt.json").exists():
            worker.cancel(spec, directory)
    deadline = time.monotonic() + 5
    for spec, directory in specifications:
        while time.monotonic() < deadline:
            status = worker.observe(spec, directory)
            if status.get("missing") or not status["worker_alive"]:
                break
            time.sleep(.05)
        worker.tmux(spec, "kill-session", "-t", "=" + spec["session"], check=False)


def wait_for(spec, directory, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = worker.observe(spec, directory)
        if predicate(status):
            return status
        time.sleep(.03)
    pytest.fail(f"Worker did not meet expected state: {status}")


def terminal(spec, directory):
    return wait_for(spec, directory,
                    lambda x: x["receipt"]["state"] in worker.TERMINAL and not x["worker_alive"])


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required")
def test_detached_completion_retains_pane_and_is_idempotent(spec_factory):
    _, spec, directory = spec_factory("import time; print('progress'); time.sleep(.3)")
    worker.launch(spec, directory)
    status = terminal(spec, directory)
    assert status["receipt"]["state"] == "complete"
    assert status["receipt"]["worker"]["start_ticks"] > 0
    assert "progress" in (directory / "run.log").read_text()
    assert worker.tmux(spec, "has-session", "-t", "=" + spec["session"]).returncode == 0
    original = status["receipt"]
    assert worker.launch(spec, directory)["existing"] is True
    assert worker.observe(spec, directory)["receipt"] == original
    conflicting = dict(spec, spec_sha256="changed")
    with pytest.raises(ValueError, match="different immutable"):
        worker.launch(conflicting, directory)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required")
def test_failed_exit_does_not_look_complete_because_tmux_remains(spec_factory):
    _, spec, directory = spec_factory("raise SystemExit(7)")
    worker.launch(spec, directory)
    status = terminal(spec, directory)
    assert status["receipt"]["state"] == "failed"
    assert status["receipt"]["exit_code"] == 7
    assert worker.tmux(spec, "has-session", "-t", "=" + spec["session"]).returncode == 0


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required")
def test_simultaneous_duplicate_launches_create_one_worker(spec_factory, tmp_path):
    path, spec, directory = spec_factory(
        "from pathlib import Path; import time; "
        "p=Path('launch-count'); p.write_text(p.read_text()+'x' if p.exists() else 'x'); time.sleep(.2)")
    command = [sys.executable, "-m", "campaigns.tmux_worker", "launch", "--spec", str(path),
               "--attempt", spec["attempt_id"]]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE) for _ in range(2)]
    outputs = [json.loads(process.communicate(timeout=5)[0]) for process in processes]
    assert all(process.returncode == 0 for process in processes)
    assert sorted(output["existing"] for output in outputs) == [False, True]
    assert terminal(spec, directory)["receipt"]["state"] == "complete"
    assert (tmp_path / "launch-count").read_text() == "x"


@pytest.mark.parametrize("reason,limits", [
    ("supervision_lost", {"lease_seconds": .5}),
    ("timeout", {"timeout_seconds": .5}),
])
@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required")
def test_limits_kill_owned_descendants_but_leave_unrelated_worker(spec_factory, reason, limits):
    _, spec, directory = spec_factory(
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)",
        **limits)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        worker.launch(spec, directory)
        running = wait_for(spec, directory, lambda x: x["receipt"]["state"] == "running")
        status = terminal(spec, directory)
        assert status["receipt"]["state"] == reason
        assert not worker.group_members(running["receipt"]["worker"]["pgid"])
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required")
def test_bridge_restart_preserves_gpu_reservation_and_renews_remote_lease(spec_factory, tmp_path):
    path, spec, directory = spec_factory("import time; print('running'); time.sleep(1.2)", lease_seconds=.8)
    command = [sys.executable, "-m", "campaigns.tmux_worker", "bridge", "--spec", str(path),
               "--attempt", spec["attempt_id"]]
    reservation = (tmp_path / "gpu.lock").open("a+")
    fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = subprocess.Popen(command, pass_fds=(reservation.fileno(),), stdout=subprocess.PIPE)
    try:
        running = wait_for(spec, directory, lambda x: not x.get("missing") and x["receipt"]["state"] == "running")
        assert running["receipt"]["reservation_fds"] == 1
        reservation.close()
        process.kill()
        process.wait(timeout=3)
        with (tmp_path / "gpu.lock").open("a+") as challenger:
            with pytest.raises(BlockingIOError):
                fcntl.flock(challenger, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert worker.renew(spec, directory)["renewed"]
        renewed = worker.read(directory / "lease.json")
        assert renewed["boot_id"] == worker.boot_id()
        assert 0 <= time.monotonic() - renewed["renewed_monotonic"] < .5
        replacement = subprocess.Popen(command, stdout=subprocess.PIPE)
        time.sleep(.5)
        worker.renew(spec, directory)
        assert replacement.wait(timeout=5) == 0
        assert worker.observe(spec, directory)["receipt"]["worker"] == running["receipt"]["worker"]
        with (tmp_path / "gpu.lock").open("a+") as released:
            fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if not reservation.closed:
            reservation.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required")
def test_bridge_term_propagates_cancel_and_waits_for_worker(spec_factory):
    path, spec, directory = spec_factory("import time; time.sleep(30)")
    process = subprocess.Popen([sys.executable, "-m", "campaigns.tmux_worker", "bridge", "--spec", str(path),
                                "--attempt", spec["attempt_id"]], stdout=subprocess.PIPE)
    try:
        wait_for(spec, directory, lambda x: not x.get("missing") and x["receipt"]["state"] == "running")
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) != 0
        assert terminal(spec, directory)["receipt"]["state"] == "cancelled"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required")
def test_controller_cancels_verified_orphan_after_supervisor_sigkill(spec_factory):
    _, spec, directory = spec_factory(
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']); "
        "print('descendant started',flush=True); time.sleep(30)")
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        worker.launch(spec, directory)
        running = wait_for(spec, directory, lambda x: x["receipt"]["state"] == "running" and
                           len(worker.group_members(x["receipt"]["worker"]["pgid"])) == 2)
        os.kill(running["receipt"]["supervisor"]["pid"], signal.SIGKILL)
        wait_for(spec, directory, lambda x: x["worker_alive"] and not x["supervisor_alive"])
        result = worker.cancel(spec, directory)
        assert result["orphan_stopped"] is True
        status = terminal(spec, directory)
        assert status["receipt"]["state"] == "supervision_lost"
        assert status["receipt"]["orphan_cancellation"]["complete"] is True
        assert status["receipt"]["elapsed_seconds"] > 0
        assert not worker.group_members(running["receipt"]["worker"]["pgid"])
        assert unrelated.poll() is None
        assert worker.launch(spec, directory)["existing"] is True
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_orphan_cancellation_rejects_changed_identity_without_signalling(monkeypatch):
    current = worker.identity(os.getpid())
    stale = dict(current, start_ticks=current["start_ticks"] - 1)
    monkeypatch.setattr(worker, "_pidfd_send_signal", lambda *args: pytest.fail("stale identity was signalled"))
    with pytest.raises(RuntimeError, match="identity changed"):
        worker.stop_orphan_group(stale, .1)
    with pytest.raises(RuntimeError, match="identity changed"):
        worker.stop_orphan_group(dict(current, boot_id="another boot"), .1)


def test_lease_fails_closed_on_other_boot_future_timestamp_and_malformed_receipt(spec_factory):
    _, spec, directory = spec_factory("pass")
    directory.mkdir(parents=True)
    for value in ({}, {"attempt_id": spec["attempt_id"], "boot_id": "other", "renewed_monotonic": time.monotonic()},
                  {"attempt_id": spec["attempt_id"], "boot_id": worker.boot_id(), "renewed_monotonic": time.monotonic()+100}):
        worker.atomic(directory / "lease.json", value)
        assert not worker.lease_valid(spec, directory)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_limits_rejected(spec_factory, value):
    with pytest.raises(ValueError, match="finite and positive"):
        spec_factory("pass", timeout_seconds=value)
