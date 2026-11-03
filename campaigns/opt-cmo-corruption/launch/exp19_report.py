"""Apply exp-019's frozen rule: are adapter arrays more fragile than base arrays at the same stuck rate?

For every task, fault kind/rate and arm (raw ``none`` deployment or ``pair_rewrite_verify``) on array
271002, it reports:
- the paired test acc_norm loss of base-only, adapter-only and all-array faults against the
  healthy (nominal) deployment of the same arm;
- the adapter − base loss difference with its discordant-pair SE, and the frozen verdict;
- the loss per 1,000 failed devices;
- the additivity check;
- agreement with the clean digital adapted model.

    python campaigns/opt-cmo-corruption/launch/exp19_report.py --runs ROOT [...] --tasks-dir data/corpora --output OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

ARMS = ("none", "pair_rewrite_verify")
FAULTS = ("random_10000ppm", "random_30000ppm", "gmax_30000ppm")


def rows(roots) -> dict:
    found = {}
    for root in roots:
        for path in Path(root).rglob("measurements.json"):
            result = json.loads((path.parent / "result.json").read_text())
            if result.get("status") != "complete":
                continue
            for row in json.loads(path.read_text()):
                found.setdefault(row["lifecycle_id"], []).append(row)
    return found


def correct(predictions: str, labels: torch.Tensor) -> torch.Tensor:
    return torch.tensor([int(p) for p in predictions]) == labels


def paired(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Accuracy difference a − b (points) and its discordant-pair SE."""

    n, only_a, only_b = len(a), int((a & ~b).sum()), int((b & ~a).sum())
    return 100 * (only_a - only_b) / n, 100 * math.sqrt(max(only_a + only_b - (only_a - only_b) ** 2 / n, 0)) / n


def agreement(a: str, b: str) -> float:
    return 100 * sum(x == y for x, y in zip(a, b)) / len(a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    found = rows(args.runs)
    report = {"schema": "exp19.review.v1", "tasks": {}}
    for task in ("hellaswag", "sst2", "piqa"):
        measured = found.get(f"opt125m-cmo-l6-exp19-analog-lora-{task}")
        if not measured:
            report["tasks"][task] = {"coverage": "missing"}
            continue
        labels = torch.load(args.tasks_dir / f"{task}-opt125m-v1.pt", weights_only=True)["splits"]["test"]["labels"]
        by = {(r["case"], r["onchip_arm"]): r for r in measured}
        clean = measured[0]["clean"]["task_test"]
        out = {"hwa_arm": measured[0]["hwa_arm"], "clean_accuracy_norm_percent": clean["accuracy_norm_percent"],
               "arms": {}}
        for arm in ARMS:
            healthy = by["nominal", arm]["final"]["task_test"]
            healthy_right = correct(healthy["predictions"], labels)
            table = {"healthy_accuracy_norm_percent": healthy["accuracy_norm_percent"],
                     "healthy_agreement_with_clean": agreement(healthy["predictions"], clean["predictions"])}
            for fault in FAULTS:
                entry = {}
                right = {}
                for target, label in (("all", fault), ("base", f"{fault}_base"), ("adapter", f"{fault}_adapter")):
                    row = by[label, arm]
                    test = row["final"]["task_test"]
                    right[target] = correct(test["predictions"], labels)
                    loss, se = paired(healthy_right, right[target])
                    failed = row["identity"]["failed_devices"]
                    entry[target] = {
                        "accuracy_norm_percent": test["accuracy_norm_percent"], "loss_points": loss, "loss_se": se,
                        "agreement_with_clean": agreement(test["predictions"], clean["predictions"]),
                        "choice_kl_vs_teacher": test["choice_kl"], "failed_devices": failed,
                        "failed_adapter_devices": row["identity"].get("failed_adapter_devices", 0),
                        "loss_per_1000_failed": 1000 * loss / failed if failed else None,
                        **({"rewrite_programs": row["fault_rewrite"]["cost"].get("device_programs"),
                            "rewrite_flagged": row["fault_rewrite"].get("flagged_devices")}
                           if row.get("fault_rewrite") else {}),
                    }
                difference, se = paired(right["base"], right["adapter"])  # = loss(adapter) − loss(base)
                verdict = ("adapters more fragile" if difference >= 0.5 and difference >= 2 * se else
                           "base more fragile" if difference <= -0.5 and -difference >= 2 * se else
                           "comparable / inconclusive")
                entry["adapter_minus_base_loss"] = {"points": difference, "se": se, "verdict": verdict}
                entry["additivity_gap_points"] = entry["all"]["loss_points"] - (entry["base"]["loss_points"]
                                                                                + entry["adapter"]["loss_points"])
                table[fault] = entry
            out["arms"][arm] = table
        report["tasks"][task] = out
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for task, value in report["tasks"].items():
        if "arms" not in value:
            print(task, value)
            continue
        print(f"{task}: clean {value['clean_accuracy_norm_percent']:.2f} ({value['hwa_arm']})")
        for arm, table in value["arms"].items():
            print(f"  {arm}: healthy {table['healthy_accuracy_norm_percent']:.2f}")
            for fault in FAULTS:
                e = table[fault]
                print(f"    {fault:16s} loss all {e['all']['loss_points']:+.2f} base {e['base']['loss_points']:+.2f} "
                      f"adapter {e['adapter']['loss_points']:+.2f} | adapter−base {e['adapter_minus_base_loss']['points']:+.2f}"
                      f"±{e['adapter_minus_base_loss']['se']:.2f} {e['adapter_minus_base_loss']['verdict']}")


if __name__ == "__main__":
    main()
