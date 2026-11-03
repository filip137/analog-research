"""Subprocess-only native lifecycle orchestration for the Wan noise-scale pilot.

`all` runs the readiness probe and then the main sweep inside one wall-time
allowance, writing heartbeat status, receipts, `result.json` and `exit-code`.
"""

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
RESULTS = ROOT / "results/lifecycles/opt125m-wan-noise-scale"
CONFIGS = ROOT / "campaigns/opt-cmo-corruption/lifecycles"
ALLOWANCE = 2700  # 45 GPU-minutes, including readiness and understood retries.
SCALES = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
TWO_DAYS = 172800.0


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def status(path, started, state, **extra):
    write(path, dict(state=state, heartbeat_at=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.monotonic() - started, **extra))


def phase(name, inputs, deadline):
    started = time.monotonic()
    status_path = RESULTS / f"{name}-status.json"
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

    status(status_path, started, "running", stage="verify_inputs")
    for record in [inputs["teacher"]] + inputs["corpora"] + list(inputs["sources"].values()) + list(inputs["prior"].values()):
        if digest(record["path"]) != record["sha256"]:
            raise ValueError(f"Changed frozen input: {record['path']}")
    labels = ["probe"] if name == "probe" else ["std3", "std9", "cdt3"]
    parents, roots, audits = {}, {"retention": [], "compensation": []}, []
    for kind in roots:
        for label in labels:
            status(status_path, started, "running", stage=f"{kind}/{label}")
            root = RESULTS / name / kind / label
            plan = root / "plan"
            src = inputs["sources"][label]
            command = [py, "-m", "workflow", "plan", CONFIGS / f"opt125m-wan-scale-{kind}-{label}.json",
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
                previous = json.loads(Path(inputs["prior"][label]["path"]).read_text())
                current = json.loads((matches[0].parent / "retention_measurements.json").read_text())
                # Programmed array and unscaled two-day snapshot must equal the original Wan arrays.
                for seconds in (None, TWO_DAYS):
                    row = next(r for r in current if r["seconds"] == seconds and r["noise_scale"] == 1)
                    prior = next(r for r in previous if r["case"] == "nominal" and r["model"] == "reram_wan2022"
                                 and r["seconds"] == seconds)
                    if row["apparent_sha256"] != prior["apparent_sha256"] or row["metrics"] != prior["metrics"]:
                        raise RuntimeError(f"Original Wan {row['snapshot']} array did not reproduce exactly.")
                    audits.append({"source": label, "snapshot": row["snapshot"], "exact": True,
                                   "original": inputs["prior"][label]["path"], "new_parent": str(matches[0].parent)})
        collected = RESULTS / f"collected-{name}-{kind}"
        run([py, "-m", "workflow", "collect", *roots[kind], "--output", collected])
        rows = json.loads((collected / f"{kind}.json").read_text())
        per_label = len(SCALES) + 1 if kind == "retention" else 2 * len(SCALES)
        aged = [r for r in rows if r["seconds"] is not None]
        if (len(rows) != len(labels) * per_label
                or any(r["duplicate_coverage"] or not r["replay_exact"] or r["failed_devices"]
                       or r["case"] != "nominal" or r["model"] != "reram_wan2022" for r in rows)
                or sorted({r["noise_scale"] for r in aged}) != SCALES
                or any(r["seconds"] != TWO_DAYS for r in aged)):
            raise RuntimeError("Unexpected scientific coverage or failed replay.")
    receipt = dict(phase=name, elapsed_seconds=time.monotonic() - started,
                   retention_rows=len(labels) * (len(SCALES) + 1), compensation_rows=len(labels) * 2 * len(SCALES),
                   replay_audit=audits, native_validation="passed")
    write(RESULTS / f"{name}-receipt.json", receipt)
    status(status_path, started, "complete", stage="collected")
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe", "main", "all"))
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + ALLOWANCE
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "worker.pid").write_text(str(os.getpid()) + "\n")
    inputs = json.loads(args.inputs.read_text())
    phases = ["probe", "main"] if args.phase == "all" else [args.phase]
    try:
        status(RESULTS / "status.json", started, "running", phases=phases)
        receipts = [phase(name, inputs, deadline) for name in phases]
        write(RESULTS / "result.json", dict(state="complete", elapsed_seconds=time.monotonic() - started,
                                            allowance_seconds=ALLOWANCE, phases=receipts))
        status(RESULTS / "status.json", started, "complete", phases=phases)
        (RESULTS / "exit-code").write_text("0\n")
    except BaseException as exc:
        status(RESULTS / "status.json", started, "failed", error=repr(exc))
        (RESULTS / "exit-code").write_text("1\n")
        raise


if __name__ == "__main__":
    main()
