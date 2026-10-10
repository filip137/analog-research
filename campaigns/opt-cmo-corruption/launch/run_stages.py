#!/usr/bin/env python3
"""Run selected stages of a planned lifecycle through `python -m ebl train`.

usage: run_stages.py --source SRC --plan PLAN_DIR --runs RUNS_DIR [--input STAGE:ROLE=PATH ...] STAGE [STAGE ...]

Each stage runs from the clean source checkout SRC with the plan's stage config, into
RUNS_DIR/<stage>. Inputs come from the plan: a fixed path, or the declared artifact of a
completed upstream stage under RUNS_DIR, unless --input overrides it (for an artifact
produced on another host). A stage that already has a complete run is skipped.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def complete_run(root: Path):
    for path in sorted(root.glob("*/result.json"), reverse=True) if root.is_dir() else []:
        result = json.loads(path.read_text())
        if result.get("status") == "complete":
            return result, path.parent
    return None


def artifact(run, kind: str) -> Path:
    result, directory = run
    [item] = [a for a in result["artifacts"] if a["kind"] == kind]
    return directory / item["path"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--input", action="append", default=[], help="STAGE:ROLE=PATH")
    parser.add_argument("stages", nargs="+")
    args = parser.parse_args()
    if subprocess.run(["git", "status", "--porcelain"], cwd=args.source, capture_output=True, text=True).stdout:
        sys.exit("Expected a clean source checkout.")
    overrides = dict(item.split("=", 1) for item in args.input)
    plan = json.loads((args.plan / "campaign.json").read_text())
    entries = {stage["id"]: stage for stage in plan["stages"]}
    for stage in args.stages:
        out = args.runs / stage
        if complete_run(out):
            print(f"skip {stage}: complete run exists", flush=True)
            continue
        entry = entries[stage]
        command = [sys.executable, "-m", "ebl", "train", "--config", str(args.plan / entry["config"]),
                   "--output-dir", str(out)]
        for role, spec in sorted(entry["inputs"].items()):
            key = f"{stage}:{role}"
            if key in overrides:
                path = Path(overrides[key])
            elif "path" in spec:
                path = Path(spec["path"])
            else:
                upstream = complete_run(args.runs / spec["stage"])
                if upstream is None:
                    sys.exit(f"Expected a complete {spec['stage']} run for {stage}.")
                path = artifact(upstream, spec["artifact_kind"])
            command += ["--" + role.replace("_", "-"), str(path)]
        print("run " + " ".join(command), flush=True)
        code = subprocess.run(command, cwd=args.source).returncode
        if code or complete_run(out) is None:
            sys.exit(f"Stage {stage} failed with exit code {code}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
