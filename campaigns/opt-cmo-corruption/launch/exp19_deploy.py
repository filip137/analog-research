"""Deploy exp-019's development-selected LoRA arm as its own campaign, with explicit upstream artifacts.

The runner keeps one resolved contract per campaign ID and output root, so the deploy and
on-chip stages of the selected arm run as campaign ``<lifecycle_id>-deploy``. Their inputs
are the completed prepare, devices and LoRA-source artifacts of the training campaign, given
as explicit paths. Stage configs and the worktree (staged code) are the plan's own.

    EBL_CORPUS_ROOT=... EBL_SOURCE_RECEIPT=... python exp19_deploy.py STAGED_ROOT hellaswag [sst2 piqa]
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def artifacts(runs: Path) -> dict:
    """stage_id -> {artifact kind: absolute path} of complete native runs."""

    found = {}
    for path in runs.rglob("result.json"):
        result = json.loads(path.read_text())
        if result.get("status") != "complete" or "stage_id" not in result.get("metrics", {}):
            continue
        stage = result["metrics"]["stage_id"]
        if stage in found:
            raise RuntimeError(f"Expected one complete run of {stage}.")
        found[stage] = {a["kind"]: str((path.parent / a["path"]).resolve()) for a in result["artifacts"]}
    return found


def main() -> int:
    root, tasks = Path(sys.argv[1]), sys.argv[2:]
    failed = 0
    for task in tasks:
        lifecycle_id = f"opt125m-cmo-l6-exp19-analog-lora-{task}"
        base = root / "results/lifecycles" / lifecycle_id
        arm = json.loads((base / "arm-selection.json").read_text())["selected"]
        manifest = json.loads((base / "plan/campaign.json").read_text())
        stages = {stage["id"]: stage for stage in manifest["stages"]}
        done = artifacts(base / "runs")
        chosen = []
        for stage_id in (f"deploy__{arm}__271002", f"onchip__{arm}__271002"):
            stage = json.loads(json.dumps(stages[stage_id]))
            for role, spec in stage["inputs"].items():
                if "stage" in spec and spec["stage"] in done:
                    stage["inputs"][role] = {"path": done[spec["stage"]][spec["artifact_kind"]]}
            stage["depends_on"] = [d for d in stage["depends_on"] if d not in done]
            chosen.append(stage)
        deploy = dict(manifest, campaign_id=f"{lifecycle_id}-deploy", stages=chosen)
        path = base / "plan/campaign-deploy-explicit.json"
        path.write_text(json.dumps(deploy, indent=2, sort_keys=True) + "\n")
        with (root / "results/lifecycles/logs" / f"{lifecycle_id}.log").open("a") as output:
            code = subprocess.run([sys.executable, "-m", "ebl", "campaign", "run", "--manifest", str(path),
                                   "--output-dir", str(base / "runs"), "--resume"], cwd=root, stdout=output,
                                  stderr=subprocess.STDOUT).returncode
        print(json.dumps({"lifecycle": lifecycle_id, "arm": arm, "exit": code}), flush=True)
        failed += code != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
