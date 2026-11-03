"""Run exp-019 on one worker: per task, train the LoRA arms, select one on development data, deploy it.

The arm with the highest selected development score in its native fit is chosen (ties: lower LR).
Only its deploy and on-chip stages run on the evaluation array. Test data never enter the choice.

    EBL_CORPUS_ROOT=... EBL_SOURCE_RECEIPT=... python campaigns/opt-cmo-corruption/launch/exp19_drive.py \\
        --base SOURCE.pt hellaswag [sst2 piqa]
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
TEACHER = Path("/home/filip/analog-research/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin")


def log(path: Path, text: str) -> None:
    with path.open("a") as stream:
        stream.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {text}\n")


def run(manifest: dict, base: Path, name: str, keep: str, output) -> int:
    trimmed = dict(manifest, stages=[s for s in manifest["stages"] if re.fullmatch(keep, s["id"])])
    path = base / "plan" / f"campaign-{name}.json"
    path.write_text(json.dumps(trimmed, indent=2, sort_keys=True) + "\n")
    return subprocess.run([PYTHON, "-m", "ebl", "campaign", "run", "--manifest", str(path), "--output-dir",
                           str(base / "runs"), "--resume"], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT).returncode


def selected_arm(base: Path, lifecycle: dict) -> tuple[str, dict]:
    rates = {arm["id"]: arm["learning_rate"] for arm in lifecycle["hwa"]["arms"]}
    scores = {}
    for path in (base / "runs").rglob("result.json"):
        result = json.loads(path.read_text())
        metrics = result.get("metrics", {})
        if result.get("status") == "complete" and metrics.get("stage", {}).get("kind") == "hwa":
            fit = metrics["fit"]
            scores[metrics["stage"]["hwa_arm"]] = {"score": fit["development_score"], "epoch": fit["selected_epoch"],
                                                  "learning_rate": rates[metrics["stage"]["hwa_arm"]]}
    if set(scores) != set(rates):
        raise RuntimeError(f"Expected every LoRA arm to complete. Provided value: {sorted(scores)}.")
    arm = max(scores, key=lambda a: (scores[a]["score"], -scores[a]["learning_rate"]))
    return arm, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args()
    logs = ROOT / "results/lifecycles/logs"
    logs.mkdir(parents=True, exist_ok=True)
    driver = logs / f"driver-exp19-{'-'.join(args.tasks)}.log"
    failed = 0
    for task in args.tasks:
        lifecycle_id = f"opt125m-cmo-l6-exp19-analog-lora-{task}"
        source = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / f"{lifecycle_id}.json"
        lifecycle = json.loads(source.read_text())
        base = ROOT / "results/lifecycles" / lifecycle_id
        try:
            command = [PYTHON, "-m", "workflow", "plan", str(source), "--teacher-weights", str(TEACHER), "--output",
                       str(base / "plan"), "--runs", str(base / "runs"), "--device", "cuda", "--cpu-threads", "4",
                       "--python", PYTHON]
            for arm in lifecycle["hwa"]["arms"]:
                command += ["--lora-base", f"{arm['id']}={args.base}"]
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            manifest = json.loads((base / "plan/campaign.json").read_text())
            with (logs / f"{lifecycle_id}.log").open("a") as output:
                log(driver, f"start {lifecycle_id} lora")
                code = run(manifest, base, "lora", r"prepare|devices|hwa__.*", output)
                log(driver, f"exit {code} {lifecycle_id} lora")
                if code:
                    failed += 1
                    continue
                arm, scores = selected_arm(base, lifecycle)
                (base / "arm-selection.json").write_text(json.dumps({"selected": arm, "arms": scores,
                    "rule": "highest selected development score; ties: lower LR"}, indent=2, sort_keys=True) + "\n")
                log(driver, f"selected {lifecycle_id} {arm} {json.dumps(scores, sort_keys=True)}")
                keep = rf"prepare|devices|hwa__.*|deploy__{arm}__271002|onchip__{arm}__271002"
                log(driver, f"start {lifecycle_id} deploy {arm}")
                code = run(manifest, base, "deploy", keep, output)
                log(driver, f"exit {code} {lifecycle_id} deploy {arm}")
                failed += code != 0
        except Exception as error:  # noqa: BLE001 - recorded, then the next task runs
            failed += 1
            log(driver, f"error {lifecycle_id}: {type(error).__name__}: {error}")
    log(driver, f"done failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
