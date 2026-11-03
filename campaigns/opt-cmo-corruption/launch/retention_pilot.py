"""Subprocess-only launcher for the bounded 2026-10-13 retention pilot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results/lifecycles/opt125m-aihwkit-retention-pilot"
CONFIGS = ROOT / "campaigns/opt-cmo-corruption/lifecycles"


def run(argv, *, deadline, env):
    print(json.dumps({"command": list(map(str, argv))}), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, env=env, check=True,
                   timeout=max(1, deadline - time.monotonic()))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe", "main"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600:
        raise ValueError("Expected a positive execution allowance no greater than one hour.")
    started = time.monotonic()
    deadline = started + args.seconds
    inputs = json.loads(args.inputs.read_text())
    for record in [inputs["teacher"]] + inputs["corpora"] + list(inputs["sources"].values()):
        if digest(record["path"]) != record["sha256"]:
            raise ValueError(f"Changed frozen input: {record['path']}")
    env = dict(os.environ, EBL_CORPUS_ROOT=inputs["corpus_root"])
    py = sys.executable
    labels = ["probe"] if args.phase == "probe" else ["std3", "std9", "cdt3"]
    runs = []
    for label in labels:
        root = RESULTS / label
        plan = root / "plan"
        command = [py, "-m", "workflow", "plan", CONFIGS / f"opt125m-aihwkit-retention-{label}.json",
                   "--teacher-weights", inputs["teacher"]["path"], "--output", plan,
                   "--device", "cuda:0", "--cpu-threads", "1", "--python", py]
        if label != "probe":
            src = inputs["sources"][label]
            command += ["--hwa-source", f"{src['arm']}={src['path']}"]
        run(command, deadline=deadline, env=env)
        command = [py, "-m", "ebl", "campaign", "run", "--manifest", plan / "campaign.json",
                   "--output-dir", root / "runs", "--resume"]
        run(command + ["--dry-run"], deadline=deadline, env=env)
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if processes.strip():
            raise RuntimeError("GPU is not idle; refusing to share with another process.")
        run(command, deadline=deadline, env=env)
        run_dirs = []
        for path in sorted((root / "runs").rglob("result.json")):
            result = json.loads(path.read_text())
            if result.get("schema") == "ebl.run":
                run_dirs.append(path.parent)
        if not run_dirs:
            raise RuntimeError("No native result bundles were produced.")
        run([py, "-m", "ebl", "runs", "inspect", *run_dirs, "--verify-artifacts", "--require-complete"],
            deadline=deadline, env=env)
        runs.append(root / "runs")
    run([py, "-m", "workflow", "collect", *runs, "--output", RESULTS / f"collected-{args.phase}"],
        deadline=deadline, env=env)
    rows = json.loads((RESULTS / f"collected-{args.phase}" / "retention.json").read_text())
    expected = 24 if args.phase == "probe" else 72
    if len(rows) != expected or any(row["duplicate_coverage"] or not row["replay_exact"] for row in rows):
        raise RuntimeError("Incomplete or duplicated retention coverage.")
    receipt = {"phase": args.phase, "coverage": len(rows), "elapsed_seconds": time.monotonic() - started,
               "allowance_seconds": args.seconds, "native_validation": "passed", "collected": str(RESULTS / f"collected-{args.phase}")}
    (RESULTS / f"{args.phase}-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
