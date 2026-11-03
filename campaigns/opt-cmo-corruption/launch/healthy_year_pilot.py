"""Subprocess-only native lifecycle orchestration for the healthy one-year pilot."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/lifecycles/opt125m-healthy-one-year"
CONFIGS = ROOT / "campaigns/opt-cmo-corruption/lifecycles"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe", "main"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1200:
        raise ValueError("Expected at most twenty minutes, within the remaining pilot allowance.")
    started = time.monotonic()
    deadline = started + args.seconds
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.phase}.pid").write_text(str(os.getpid()) + "\n")
    status_path = RESULTS / f"{args.phase}-status.json"

    def status(state, **extra):
        write(status_path, dict(state=state, heartbeat_at=datetime.now(timezone.utc).isoformat(),
                                elapsed_seconds=time.monotonic() - started, **extra))

    inputs = json.loads(args.inputs.read_text())
    env = dict(os.environ, EBL_CORPUS_ROOT=inputs["corpus_root"])
    py = sys.executable

    def run(argv):
        argv = list(map(str, argv))
        print(json.dumps({"command": argv}), flush=True)
        process = subprocess.Popen(argv, cwd=ROOT, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, argv)

    try:
        status("running", stage="verify_inputs")
        for record in [inputs["teacher"]] + inputs["corpora"] + list(inputs["sources"].values()) + list(inputs["retention"].values()):
            if digest(record["path"]) != record["sha256"]:
                raise ValueError(f"Changed frozen input: {record['path']}")
        labels = ["probe"] if args.phase == "probe" else ["std3", "std9", "cdt3"]
        parents, roots = {}, {"retention": [], "compensation": []}
        audits = []
        for kind in roots:
            for label in labels:
                status("running", stage=f"{kind}/{label}")
                root = RESULTS / args.phase / kind / label
                plan = root / "plan"
                src = inputs["sources"][label]
                command = [py, "-m", "workflow", "plan", CONFIGS / f"opt125m-healthy-year-{kind}-{label}.json",
                           "--teacher-weights", inputs["teacher"]["path"], "--output", plan,
                           "--device", "cuda:0", "--cpu-threads", "1", "--python", py,
                           "--hwa-source", f"{src['arm']}={src['path']}"]
                if kind == "compensation":
                    command += ["--retention-source", f"{src['arm']}:271002={parents[label]}"]
                run(command)
                command = [py, "-m", "ebl", "campaign", "run", "--manifest", plan / "campaign.json",
                           "--output-dir", root / "runs", "--resume"]
                run(command + ["--dry-run"])
                active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
                if active.strip():
                    raise RuntimeError("GPU is not idle; refusing to share.")
                run(command)
                bundles = [p.parent for p in sorted((root / "runs").rglob("result.json"))
                           if json.loads(p.read_text()).get("schema") == "ebl.run"]
                if not bundles:
                    raise RuntimeError("Missing native bundles.")
                run([py, "-m", "ebl", "runs", "inspect", *bundles, "--verify-artifacts", "--require-complete"])
                roots[kind].append(root / "runs")
                if kind == "retention":
                    matches = list((root / "runs").rglob("retention_states.json"))
                    if len(matches) != 1:
                        raise RuntimeError("Expected one native retention index.")
                    parents[label] = matches[0]
                    old = Path(inputs["retention"][label]["path"]).parent
                    previous = json.loads((old / "retention_measurements.json").read_text())
                    current = json.loads((matches[0].parent / "retention_measurements.json").read_text())
                    for row in current:
                        if row["seconds"] is None:
                            prior = next(r for r in previous if r["case"] == row["case"] and r["model"] == row["model"] and r["seconds"] is None)
                            if row["apparent_sha256"] != prior["apparent_sha256"] or row["metrics"] != prior["metrics"]:
                                raise RuntimeError("Original programmed array did not reproduce exactly.")
                            audits.append({"source": label, "model": row["model"], "baseline_exact": True,
                                           "original_parent": str(old), "new_parent": str(matches[0].parent)})
            collected = RESULTS / f"collected-{args.phase}-{kind}"
            run([py, "-m", "workflow", "collect", *roots[kind], "--output", collected])
            rows = json.loads((collected / f"{kind}.json").read_text())
            expected = len(labels) * (4 if kind == "retention" else 14)
            if len(rows) != expected or any(r["duplicate_coverage"] or not r["replay_exact"] or r["failed_devices"] or r["case"] != "nominal" for r in rows):
                raise RuntimeError("Unexpected scientific coverage or failed replay.")
        receipt = dict(phase=args.phase, elapsed_seconds=time.monotonic() - started,
                       allowance_seconds=args.seconds, retention_rows=len(labels)*4,
                       compensation_rows=len(labels)*14, baseline_audit=audits, native_validation="passed")
        write(RESULTS / f"{args.phase}-receipt.json", receipt)
        status("complete", stage="collected")
        print(json.dumps(receipt), flush=True)
    except BaseException as exc:
        status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
