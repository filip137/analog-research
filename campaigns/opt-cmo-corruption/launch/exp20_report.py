"""Apply exp-020's frozen rule: does on-chip recovery after the verify rewrite beat rewrite alone?

For each task and fault case on array 271002, the paired test acc_norm difference of
rewrite + calibration (R2), rewrite + adapter-only learning (RA) and rewrite + all-weight
learning (RW) against rewrite hold (R1) is reported with its discordant-pair SE and verdict.
So are RA − R2, RA − RW, agreement with the clean adapted model and write costs.

    python campaigns/opt-cmo-corruption/launch/exp20_report.py --runs ROOT --tasks-dir data/corpora --output OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

ARMS = {"R1": "pair_rewrite_verify", "R2": "pair_rewrite_verify_calibration",
        "RA": "pair_rewrite_verify_adapter_recovery", "RW": "pair_rewrite_verify_recovery", "raw": "none"}
CASES = ["nominal"] + [f"{k}_{p}ppm{t}" for k, p in (("gmax", 30000), ("gmax", 60000), ("random", 60000))
                       for t in ("", "_base", "_adapter")]


def paired(a, b):
    n, only_a, only_b = len(a), int((a & ~b).sum()), int((b & ~a).sum())
    return 100 * (only_a - only_b) / n, 100 * math.sqrt(max(only_a + only_b - (only_a - only_b) ** 2 / n, 0)) / n


def verdict(delta, se):
    return "helps" if delta >= 0.5 and delta >= 2 * se else "hurts" if delta <= -2 * se else "no detectable effect"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for task in ("hellaswag", "sst2", "piqa"):
        rows = {}
        for path in (args.runs / f"opt125m-cmo-l6-exp20-{task}" / "runs").rglob("measurements.json"):
            if json.loads((path.parent / "result.json").read_text()).get("status") == "complete":
                rows.update({(r["case"], r["onchip_arm"]): r for r in json.loads(path.read_text())})
        labels = torch.load(args.tasks_dir / f"{task}-opt125m-v1.pt", weights_only=True)["splits"]["test"]["labels"]
        clean = next(iter(rows.values()))["clean"]["task_test"]
        task_report = {"clean_accuracy_norm_percent": clean["accuracy_norm_percent"], "cases": {}}
        for case in CASES:
            test = {key: rows[case, arm]["final"]["task_test"] for key, arm in ARMS.items()}
            right = {key: torch.tensor([int(p) for p in value["predictions"]]) == labels for key, value in test.items()}
            entry = {"accuracy_norm_percent": {k: v["accuracy_norm_percent"] for k, v in test.items()},
                     "agreement_with_clean": {k: 100 * sum(a == b for a, b in zip(v["predictions"], clean["predictions"]))
                                              / len(labels) for k, v in test.items()}}
            for name, (a, b) in {"R2-R1": ("R2", "R1"), "RA-R1": ("RA", "R1"), "RW-R1": ("RW", "R1"),
                                 "RA-R2": ("RA", "R2"), "RA-RW": ("RA", "RW")}.items():
                delta, se = paired(right[a], right[b])
                entry[name] = {"points": delta, "se": se, "verdict": verdict(delta, se)}
            for key in ("RA", "RW"):
                cost = rows[case, ARMS[key]]["cost"]
                entry[f"{key}_cost"] = {k: cost.get(k) for k in ("issued_write_attempts", "device_writes", "touched_devices",
                                                                 "adapter_device_writes", "base_device_writes",
                                                                 "max_onchip_device_write_attempts")}
            task_report["cases"][case] = entry
        report[task] = task_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for task, value in report.items():
        print(f"{task}: clean {value['clean_accuracy_norm_percent']:.2f}")
        for case, e in value["cases"].items():
            acc = e["accuracy_norm_percent"]
            print(f"  {case:24s} raw {acc['raw']:.2f} R1 {acc['R1']:.2f} R2 {acc['R2']:.2f} RA {acc['RA']:.2f} RW {acc['RW']:.2f} | "
                  + " ".join(f"{n} {e[n]['points']:+.2f}±{e[n]['se']:.2f}{'*' if e[n]['verdict'] != 'no detectable effect' else ''}"
                             for n in ("R2-R1", "RA-R1", "RW-R1", "RA-R2"))
                  + f" | RA writes {e['RA_cost']['issued_write_attempts']} base {e['RA_cost']['base_device_writes']} RW {e['RW_cost']['issued_write_attempts']}")


if __name__ == "__main__":
    main()
