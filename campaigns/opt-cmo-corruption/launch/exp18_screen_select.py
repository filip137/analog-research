"""Select exp-018's frozen on-chip learning rates from the development screens.

Rules, per (task, teacher):
- **Weight LR:** minimum final ``task_development`` choice KL of rewrite+learn
  (``pair_rewrite_verify_recovery``).
- **Calibration LR:** minimum final ``task_development`` choice KL of
  rewrite+calibration (``pair_rewrite_verify_calibration``).
- **Ties:** the lower LR.

Only development data enter the choice; screen test scores are never read.

    python campaigns/opt-cmo-corruption/launch/exp18_screen_select.py RESULTS_ROOT LIFECYCLES_DIR task-teacher [...]
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def screen_rows(root: Path, lifecycle_id: str) -> list[dict]:
    rows = []
    for path in sorted((root / lifecycle_id / "runs").rglob("result.json")):
        result = json.loads(path.read_text())
        metrics = result.get("metrics", {})
        if result.get("status") != "complete" or metrics.get("stage", {}).get("kind") != "onchip":
            continue
        artifact = next(a for a in result["artifacts"] if a["kind"] == "measurements")
        rows += json.loads((path.parent / artifact["path"]).read_text())
    return rows


def select(root: Path, lifecycles: Path, pair: str) -> dict:
    task, teacher = pair.split("-")
    candidates = {"learning_rate": [], "calibration_learning_rate": []}
    for tag in "abc":
        lifecycle_id = f"opt125m-cmo-l6-exp18-{task}-{teacher}-screen-{tag}"
        lifecycle = json.loads((lifecycles / f"{lifecycle_id}.json").read_text())
        rows = {r["onchip_arm"]: r for r in screen_rows(root, lifecycle_id) if r["case"] == "random_30000ppm"}
        if {"pair_rewrite_verify_recovery", "pair_rewrite_verify_calibration"} - set(rows):
            raise RuntimeError(f"Incomplete screen {lifecycle_id}.")
        for key, arm, rate in (("learning_rate", "pair_rewrite_verify_recovery", lifecycle["onchip"]["learning_rate"]),
                               ("calibration_learning_rate", "pair_rewrite_verify_calibration",
                                lifecycle["onchip"]["calibration_learning_rate"])):
            candidates[key].append({"rate": rate, "screen": tag,
                                    "development_choice_kl": rows[arm]["final"]["task_development"]["choice_kl"],
                                    "hold_development_choice_kl":
                                        rows["pair_rewrite_verify"]["final"]["task_development"]["choice_kl"]})
    chosen = {key: min(values, key=lambda v: (v["development_choice_kl"], v["rate"]))["rate"]
              for key, values in candidates.items()}
    return {**chosen, "candidates": candidates}


if __name__ == "__main__":
    root, lifecycles = Path(sys.argv[1]), Path(sys.argv[2])
    print(json.dumps({pair: select(root, lifecycles, pair) for pair in sys.argv[3:]}, indent=1))
