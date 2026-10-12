"""Bound one repair-owner process group and publish a terminal receipt.

Internal entrypoint of incident_escalation, not a GPU or campaign launcher.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(request_path):
    request = json.loads(Path(request_path).read_text())
    folder = Path(request["folder"])
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    child = None
    started = time.monotonic()
    receipt = {"returncode": None, "reason": "launch_failed", "group_cleanup": False}
    try:
        with (folder / "prompt.txt").open() as stdin, (folder / "agent.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
            child = subprocess.Popen(request["argv"], stdin=stdin, stdout=stdout, stderr=stderr,
                                     cwd=request["project_dir"], start_new_session=True)
            receipt["child_pid"] = child.pid
            while child.poll() is None:
                over_time = time.monotonic() - started >= request["timeout_seconds"]
                over_output = sum((folder / name).stat().st_size for name in ("agent.log", "stderr.log")) > 4 * 1024 * 1024
                if stopping[0] or over_time or over_output:
                    receipt["reason"] = "stopped" if stopping[0] else "timeout" if over_time else "output_limit"
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
                    break
                time.sleep(0.1)
            else:
                receipt["reason"] = "exited"
            receipt["returncode"] = child.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        receipt["error"] = str(exc)[:1000]
    finally:
        # Cleanup the exclusively owned group even if its leader exited first.
        # A receipt is terminal only when this succeeds (or the group is gone).
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
                receipt["group_cleanup"] = True
            except ProcessLookupError:
                receipt["group_cleanup"] = True
            except OSError as exc:
                receipt["cleanup_error"] = str(exc)[:1000]
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                receipt["group_cleanup"] = False
        else:
            receipt["group_cleanup"] = True
        receipt["elapsed_seconds"] = time.monotonic() - started
        path = folder / "terminal.json"
        temporary = folder / "terminal.json.tmp"
        with temporary.open("w") as stream:
            json.dump(receipt, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


if __name__ == "__main__":
    run(sys.argv[1])
