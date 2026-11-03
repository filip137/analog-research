"""Native preparation and collection for the user-narrowed array rewrite pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pstdev

import exp21_placement as shared

PREFIX = "opt125m-cmo-l6-exp21-array-rewrite"


def lifecycle():
    return shared.parse_lifecycle(json.loads((shared.ROOT / "campaigns/opt-cmo-corruption/lifecycles" / f"{PREFIX}.json").read_text()))


def collect(queue_config, evidence):
    from campaigns.lifecycle_collect import collect as native_collect
    from experiments.artifacts import atomic_write_json
    result = native_collect(queue_config, evidence / "collected", recovery=True, recovery_assignments=shared.ARRAYS)
    folder = Path(result["collected"])
    lc = lifecycle()
    cases = {case.label: case for case in lc.defects.cases}
    rows = [row for path in folder.glob("onchip__*/measurements.json") for row in json.loads(path.read_text())]
    expected = {(seed, case, arm.arm_id) for seed in shared.ARRAYS for case in cases for arm in lc.onchip.arms}
    if len(rows) != 42 or {(r["array_seed"], r["case"], r["onchip_arm"]) for r in rows} != expected:
        raise ValueError("Incomplete rewrite-only coverage")
    historical = json.loads((evidence / "reference/exp016-uniform.json").read_text())
    reference = {(r["array_seed"], r["case"]): r for r in historical["rows"] if r["onchip_arm"] == shared.HOLD}
    report = []
    for row in rows:
        case = cases[row["case"]]
        if row["identity"]["failed_devices"] != 6 * round(case.rate * 9437184) or row["replay_exact"] is not True:
            raise ValueError("Changed fault total or inexact replay")
        if row["onchip_arm"] != shared.HOLD:
            continue
        dies, matrices = [], {}
        for group in row["identity"].get("groups", []):
            if group["failed_devices"] != round(case.rate * group["devices"]):
                raise ValueError("Changed per-MLP fault total")
            for die in group["dies"]:
                size = sum(a["devices"] for a in group["arrays"] if a["die"] == die["die"])
                dies.append(die["failed_devices"] / size)
            for array in group["arrays"]:
                matrices[array["matrix"]] = matrices.get(array["matrix"], 0) + array["failed_devices"]
        old = reference.get((row["array_seed"], f"{case.kind}_{case.rate_ppm}ppm"))
        comparison = None
        if old is not None:
            comparison = {domain: {metric: row["final"][domain][metric] - old["final"][domain][metric]
                for metric in ("teacher_kl", "perplexity")} for domain in ("transfer_wikitext", "transfer_generated")}
        report.append(dict(array_seed=row["array_seed"], case=row["case"], initial=row["initial"], final=row["final"],
            cost=row["cost"], fault_rewrite=row["fault_rewrite"], historical_uniform_delta=comparison,
            array_fault_fraction=row["identity"].get("array_fault_fraction"),
            die_fault_fraction=dict(std=pstdev(dies), min=min(dies), max=max(dies)) if dies else None,
            failed_by_matrix=matrices))
    summary = dict(coverage_complete=True, assignments=list(shared.ARRAYS), rewrite_endpoints=21,
        baseline_endpoints=21, learning_performed=False, fresh_uniform_performed=False,
        historical_uniform_assignments=sorted({key[0] for key in reference}),
        three_assignment_placement_claim=False, results=report)
    atomic_write_json(evidence / "reviews/summary.json", summary)
    return {"collected": str(folder), "rewrite_endpoints": len(report), "summary": str(evidence / "reviews/summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--host", required=True)
    p.add_argument("--base", type=Path, default=shared.DEFAULT_BASE)
    p = sub.add_parser("collect")
    p.add_argument("--queue-config", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = shared.prepare(args.host, args.base, "array", config=lifecycle(), timeout_seconds=2700)
    else:
        result = collect(args.queue_config, args.evidence)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
