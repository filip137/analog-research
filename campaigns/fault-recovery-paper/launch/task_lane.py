"""One budgeted native XB lab lane. Every unit uses the same reviewed prepare/run/inspect worker."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from worker import digest, read, write


def queue(depth):
    """Front-load a real Child pair on each depth, then its nominal control and other fractions."""
    tasks = ["hellaswag", "boolq", "pubmed"] if depth == 6 else ["boolq", "hellaswag", "pubmed"]
    primary = "child01" if depth == 6 else "child10"
    secondary = "child10" if depth == 6 else "child01"
    cells = [(task, method) for method in (primary, secondary, "lora", "full") for task in tasks]
    return [(task, method, rate, phase) for task, method in cells for rate in (3, 0)
            for phase in ("development", "evaluation")]


def main(args):
    root, source = args.root.resolve(), args.source.resolve()
    ledger_path = root / "lane-status.json"
    lock = (root / "lane.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "lane.pid").write_text(f"{os.getpid()}\n")
    base = source / "campaigns/fault-recovery-paper/launch"
    unit = f"xb{args.depth}-hellaswag-r03-canary-{args.canary}"
    epoch = int((root / (unit + ".epoch")).read_text())
    initial = 0.
    for failed in sorted((root / "runs").glob(f"xb{args.depth}-hellaswag-r03-canary-*/failed-*.json")):
        failed_epoch = root / (failed.parent.name + ".epoch")
        initial += read(failed)["time"] - int(failed_epoch.read_text())
    # A single monotonic deadline includes all pre-run setup, transfer, validation and worker time.
    start = time.monotonic()
    precharged = initial + max(0, time.time() - epoch)
    deadline = start + max(0, args.seconds - precharged)
    cells = queue(args.depth)
    def name(cell):
        task, method, rate, phase = cell
        return f"xb{args.depth}-{task}-r{rate:02d}-{phase}-{args.tag}-{method}"
    status = {"state": "running", "active_unit": unit, "budget_seconds": args.seconds,
              "prior_failure_seconds": initial, "frozen_source": source.name,
              "queue": [name(cell) for cell in cells], "verified_endpoint_count": 0, "completed_units": [],
              "ledger": [], "reason": None}
    def publish():
        status["spent_seconds"] = min(args.seconds, precharged + time.monotonic() - start)
        status["remaining_coverage"] = [u for u in status["queue"] if u not in status["completed_units"]]
        status["updated_epoch"] = time.time()
        write(ledger_path, status)
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Lane ceiling reached; unverified units remain in remaining_coverage.")
        return value
    def command(argv, log):
        proc = subprocess.Popen(argv, cwd=source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    proc.wait(timeout=min(15, remaining()))
                    break
                except subprocess.TimeoutExpired:
                    publish()
            if proc.returncode:
                raise RuntimeError(f"Exit {proc.returncode} from {argv!r}.")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    def receipt(unit):
        proof = read(root / "runs" / unit / "complete.json")
        if proof["schema"] != "fault_tuning.supervised.unit.v1":
            raise ValueError("Invalid native completion receipt.")
        if digest(Path(proof["run"]) / "measurements.json") != proof["measurements_sha256"]:
            raise ValueError("Completed result hash changed.")
        return proof
    publish()
    try:
        canary_dir = root / "runs" / unit
        with (root / (unit + ".pid")).open() as stream:
            canary_pid = int(stream.read())
        while not (canary_dir / "complete.json").exists():
            if list(canary_dir.glob("failed-*.json")):
                raise RuntimeError(f"Canary failed at {canary_dir}.")
            if not Path(f"/proc/{canary_pid}").is_dir():
                raise RuntimeError(f"Canary process exited before native receipt at {canary_dir}.")
            publish()
            time.sleep(min(5, remaining()))
        proof = receipt(unit)
        status["ledger"].append({"unit": unit, "seconds": proof["completed"] - epoch, "phase": "canary"})
        for cell in cells:
            task, method, rate, phase = cell
            unit = name(cell)
            status["active_unit"] = unit
            status["phase"] = phase
            now = time.monotonic()
            publish()
            tag = f"{args.tag}-{method}"
            routes = ["offline_full"] if method == "full" else ["offline_" + method, "onchip_" + method]
            argv = [sys.executable, str(base / "task_manifest.py"), "--root", str(root), "--depth", str(args.depth),
                    "--task", task, "--phase", phase, "--rate", str(rate), "--routes", *routes, "--tag", tag]
            if phase == "evaluation":
                dev = name((task, method, rate, "development"))
                weights = Path(receipt(dev)["run"]) / "checkpoints/selection.pt"
                pin = root / "inputs" / (dev + "-selection.pt")
                if not pin.exists():
                    os.link(weights, pin)
                if digest(pin) != digest(weights):
                    raise ValueError("Development selection identity changed.")
                argv += ["--selection", pin.stem]
            with (root / (unit + ".log")).open("x") as log:
                command(argv, log)
                manifest = root / "manifests" / unit / "manifest.json"
                worker = [sys.executable, str(base / "worker.py"), str(manifest), unit]
                command(worker + ["--prepare"], log)
                command(worker, log)
            proof = receipt(unit)
            status["completed_units"].append(unit)
            status["ledger"].append({"unit": unit, "phase": phase, "seconds": time.monotonic() - now,
                                     "receipt_sha256": digest(root / "runs" / unit / "complete.json")})
            if phase == "evaluation":
                status["verified_endpoint_count"] += len(routes)
            publish()
        status.update(state="complete", active_unit=None, reason="All declared units verified.")
    except TimeoutError as error:
        status.update(state="budget_exhausted", reason=str(error))
    except BaseException as error:
        status.update(state="failed", reason=repr(error))
        raise
    finally:
        publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=(6, 12), required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--canary", default="v2")
    parser.add_argument("--tag", default="v2")
    main(parser.parse_args())
