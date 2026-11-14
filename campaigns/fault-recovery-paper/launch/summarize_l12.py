"""exp-002: tabulate the l12 screen units (final dev KL, write cost, identity checks) and apply the selection rule.

usage: summarize_l12.py ROOT [UNIT ...]   (default units s1 s2 s3; ROOT/runs/<unit>/...)
Prints JSON: per-arm rows, per-method selection (lowest final dev KL; ties within 1 % -> fewer
healthy writes) with grid-edge flags, and the identity hashes seen in every unit.
"""
import json
from pathlib import Path
import sys

EXPECTED_SAMPLES = "df73ce6615be3cdd3005d2ca0b97c03131f9236ba481bc7250edde5ff66fb01a"
EXPECTED_APPARENT = {"naive": "0c6140f96d48134faf3c546613ee65c1a69c8ebb2a859f19203d8600a2abffa1",
                     "fault_aware": "06bd5454ba85ea83e9beab4204189d60f1c115830fbf8178c58a642d0473f5f2",
                     "adapter": "263bbf76548fc8d922f18dbc218e37f00eb9f85356b5b5b46b7651c171310c43"}


def main(root, units):
    rows, identity = [], {}
    for unit in units:
        runs = sorted((root / "runs" / unit).glob("*/stages/*/local/fault_tuning__*/attempt-*/runs/*/summary.json"))
        if not runs:
            raise SystemExit(f"{unit}: no completed run")
        run = runs[-1].parent
        summary = json.loads((run / "summary.json").read_text())
        if summary["samples_sha256"] != EXPECTED_SAMPLES or summary["apparent_sha256"] != EXPECTED_APPARENT:
            raise SystemExit(f"{unit}: unexpected samples or apparent-state hashes")
        stage = json.loads((run / "result.json").read_text())
        identity[unit] = {"run": str(run), "samples_sha256": summary["samples_sha256"],
                          "apparent_sha256": summary["apparent_sha256"], "clean": summary["clean"]["teacher_kl"],
                          "gpu": summary["gpu"], "fisher_seconds": summary["fisher_seconds"],
                          "peak_gpu_memory_bytes": _peak(stage)}
        for row in json.loads((run / "measurements.json").read_text()):
            cost = row["cost"]
            rows.append({"unit": unit, "label": row["label"], "method": row["method"], "rank": row["rank"],
                         "learning_rate": row["learning_rate"], "initial_kl": row["initial"]["teacher_kl"],
                         "final_kl": row["final"]["teacher_kl"],
                         "min_curve_kl": min((p["development"]["teacher_kl"] for p in row["curve"]), default=None),
                         "healthy_device_writes": cost.get("healthy_device_writes", 0),
                         "stuck_write_attempts": cost.get("stuck_write_attempts", 0),
                         "touched_devices": cost.get("touched_devices", 0),
                         "max_writes_per_device": cost.get("max_writes_per_device", 0),
                         "out_of_range_devices": cost.get("out_of_range_devices"),
                         "writes_outside_mask": cost.get("writes_outside_mask"), "mask_weights": cost.get("mask_weights"),
                         "seconds": row["seconds"],
                         "initial_apparent_sha256": row["initial_apparent_sha256"]})
    selection = {}
    for row in rows:
        if row["learning_rate"] is None:
            continue
        key = row["label"].rsplit("-", 1)[0]
        selection.setdefault(key, []).append(row)
    picked = {}
    for key, arms in selection.items():
        best = min(a["final_kl"] for a in arms)
        tied = [a for a in arms if a["final_kl"] <= best * 1.01]
        choice = min(tied, key=lambda a: (a["healthy_device_writes"], a["final_kl"]))
        grid = sorted(a["learning_rate"] for a in arms)
        picked[key] = {"learning_rate": choice["learning_rate"], "final_kl": choice["final_kl"],
                       "best_kl": best, "tied": [a["learning_rate"] for a in tied],
                       "edge": "low" if choice["learning_rate"] == grid[0] else
                       "high" if choice["learning_rate"] == grid[-1] else None,
                       "cost": {k: choice[k] for k in ("healthy_device_writes", "stuck_write_attempts",
                                                       "touched_devices", "max_writes_per_device",
                                                       "out_of_range_devices", "writes_outside_mask")}}
    print(json.dumps({"identity": identity, "rows": rows, "selection": picked}, indent=1))


def _peak(result):
    found = []

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k == "peak_gpu_memory_bytes" and isinstance(v, int):
                    found.append(v)
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
    walk(result)
    return max(found, default=None)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:] or ["s1", "s2", "s3"])
