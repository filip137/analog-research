"""Select exp-020's frozen on-chip learning rates from its development screens.

Rule (frozen in the exp-020 note), per task:
- **Adapter-only weight LR:** the screen whose rewrite + adapter-only learning has the highest final
  ``task_development`` agreement with the clean adapted model.
- **All-weight LR:** the same rule, for rewrite + all-weight learning.
- **Calibration LR:** the same rule, for rewrite + calibration.
- **Ties:** the lower LR.

Only development data enter the choice. Writes ``launch/exp20_frozen.json`` (rates and candidates).

    python campaigns/opt-cmo-corruption/launch/exp20_select.py RESULTS_ROOT LIFECYCLES_DIR OUTPUT.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ARMS = {"adapter": ("pair_rewrite_verify_adapter_recovery", "learning_rate"),
        "all": ("pair_rewrite_verify_recovery", "learning_rate"),
        "calibration": ("pair_rewrite_verify_calibration", "calibration_learning_rate")}


def agreement(a: str, b: str) -> float:
    return 100 * sum(x == y for x, y in zip(a, b)) / len(a)


def screen_rows(root: Path, lifecycle_id: str) -> dict:
    rows = {}
    for path in (root / lifecycle_id / "runs").rglob("result.json"):
        result = json.loads(path.read_text())
        if result.get("status") == "complete" and result["metrics"].get("stage", {}).get("kind") == "onchip":
            for row in json.loads((path.parent / "measurements.json").read_text()):
                if row["case"] == "gmax_60000ppm":
                    rows[row["onchip_arm"]] = row
    return rows


def main() -> None:
    root, lifecycles, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    frozen = {}
    for task in ("hellaswag", "sst2", "piqa"):
        candidates = {key: [] for key in ARMS}
        for tag in "abc":
            lifecycle_id = f"opt125m-cmo-l6-exp20-{task}-screen-{tag}"
            onchip = json.loads((lifecycles / f"{lifecycle_id}.json").read_text())["onchip"]
            rows = screen_rows(root, lifecycle_id)
            for key, (arm, field) in ARMS.items():
                if arm not in rows:
                    raise RuntimeError(f"Incomplete screen {lifecycle_id}: {arm}.")
                row = rows[arm]
                clean = row["clean"]["task_development"]["predictions"]
                candidates[key].append({
                    "screen": tag, "rate": onchip[field],
                    "development_agreement": agreement(row["final"]["task_development"]["predictions"], clean),
                    "development_accuracy_norm_percent": row["final"]["task_development"]["accuracy_norm_percent"],
                    "hold_development_agreement": agreement(
                        rows["pair_rewrite_verify"]["final"]["task_development"]["predictions"], clean),
                    "issued_write_attempts": row["cost"].get("issued_write_attempts", 0),
                    "adapter_device_writes": row["cost"].get("adapter_device_writes"),
                    "base_device_writes": row["cost"].get("base_device_writes"),
                })
        frozen[task] = {key: max(values, key=lambda v: (v["development_agreement"], -v["rate"]))["rate"]
                        for key, values in candidates.items()}
        frozen[task]["candidates"] = candidates
    output.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    print(json.dumps({task: {k: v for k, v in value.items() if k != "candidates"} for task, value in frozen.items()}))


if __name__ == "__main__":
    main()
