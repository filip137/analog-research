"""Run exp-018 lifecycles sequentially on one worker: plan, trim, run.

Each job names a lifecycle, its teacher checkpoint, an optional frozen HWA
source and the stages to keep. A source is given as a path, or as
``find:<lifecycle_id>`` for the completed source of an earlier job on this
worker. Trimming only drops stages, so every kept stage is the native stage of
the full plan. Runs use ``python -m ebl campaign run --resume``.

    EBL_CORPUS_ROOT=... EBL_SOURCE_RECEIPT=... python campaigns/opt-cmo-corruption/launch/exp18_drive.py jobs.json
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable


def log(path: Path, text: str) -> None:
    with path.open("a") as stream:
        stream.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {text}\n")


def find_source(lifecycle_id: str) -> Path:
    found = sorted((ROOT / "results/lifecycles" / lifecycle_id / "runs").rglob("checkpoints/source.pt"))
    complete = [p for p in found if json.loads((p.parents[1] / "status.json").read_text())["status"] == "complete"]
    if len(complete) != 1:
        raise RuntimeError(f"Expected one completed HWA source of {lifecycle_id}. Provided value: {complete}.")
    return complete[0]


def run(job: dict, driver_log: Path) -> int:
    lifecycle_id = job["lifecycle"]
    base = ROOT / "results/lifecycles" / lifecycle_id
    command = [PYTHON, "-m", "ebl", "lifecycle", "plan", f"campaigns/opt-cmo-corruption/lifecycles/{lifecycle_id}.json",
               "--teacher-weights", job["teacher_weights"], "--output", str(base / "plan"), "--runs",
               str(base / "runs"), "--device", "cuda", "--cpu-threads", "4", "--python", PYTHON]
    source = job.get("hwa_source")
    if source:
        path = find_source(source[5:]) if source.startswith("find:") else Path(source)
        command += ["--hwa-source", f"{job.get('hwa_arm', 'standard_hwa_model')}={path}"]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    manifest = json.loads((base / "plan/campaign.json").read_text())
    keep = re.compile(job["keep"])
    manifest["stages"] = [stage for stage in manifest["stages"] if keep.fullmatch(stage["id"])]
    trimmed = base / "plan/campaign-trimmed.json"
    trimmed.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(driver_log, f"start {lifecycle_id} stages={[s['id'] for s in manifest['stages']]}")
    with (base.parent / "logs" / f"{lifecycle_id}.log").open("a") as output:
        code = subprocess.run([PYTHON, "-m", "ebl", "campaign", "run", "--manifest", str(trimmed), "--output-dir",
                               str(base / "runs"), "--resume"], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT).returncode
    log(driver_log, f"exit {code} {lifecycle_id}")
    return code


def main() -> int:
    spec = Path(sys.argv[1])
    jobs = json.loads(spec.read_text())
    (ROOT / "results/lifecycles/logs").mkdir(parents=True, exist_ok=True)
    driver_log = ROOT / "results/lifecycles/logs" / f"driver-{spec.stem}.log"
    failed = 0
    for job in jobs:
        try:
            failed += run(job, driver_log) != 0
        except Exception as error:  # noqa: BLE001 - recorded, then the next independent job runs
            failed += 1
            log(driver_log, f"error {job['lifecycle']}: {type(error).__name__}: {error}")
    log(driver_log, f"done failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
