"""Apply exp-018's frozen decision rule to collected main-comparison measurements.

For every task, teacher (FT, PT) and defect case on array 271002:
- **Selected rows.** L is the better learn row (rewrite+learn, with or without
  calibration) and A the better no-weight-learning row (rewrite hold, with or
  without calibration), both by final ``task_development`` choice KL.
- **Δ.** The paired test acc_norm difference L − A, with the discordant-pair SE.
- **D.** Δ_FT − Δ_PT, with a paired bootstrap SE (10,000 resamples, seed 18).
- **Companion.** The choice-KL residual closed by L beyond A, (KL_A − KL_L) / KL_A.

    python campaigns/opt-cmo-corruption/launch/exp18_report.py \\
        --runs ROOT [ROOT ...] --tasks-dir data/corpora --output reviews/exp18.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

LEARN = ("pair_rewrite_verify_recovery", "pair_rewrite_verify_recovery_calibration")
HOLD = ("pair_rewrite_verify", "pair_rewrite_verify_calibration")
CASES = ("nominal", "random_30000ppm", "gmax_30000ppm")
TASK_CHOICES = {"hellaswag": 4, "sst2": 2, "piqa": 2}


def measurements(roots) -> dict:
    """lifecycle_id -> {"rows": [...], "prepare": metrics} from complete native runs."""

    found = {}
    for root in roots:
        for path in sorted(Path(root).rglob("result.json")):
            result = json.loads(path.read_text())
            if result.get("status") != "complete" or result.get("experiment_id") != "crossbar_lifecycle.v1":
                continue
            metrics = result["metrics"]
            entry = found.setdefault(metrics["lifecycle_id"], {"rows": [], "prepare": None, "runs": []})
            kind = metrics["stage"]["kind"]
            if kind == "prepare":
                entry["prepare"] = metrics
            elif kind == "onchip":
                artifact = next(a for a in result["artifacts"] if a["kind"] == "measurements")
                entry["rows"] += json.loads((path.parent / artifact["path"]).read_text())
                entry["runs"].append(str(path.parent))
    return found


def correct(predictions: str, labels: torch.Tensor) -> torch.Tensor:
    return torch.tensor([int(p) for p in predictions]) == labels


def paired(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    n = len(a)
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    delta = 100 * (only_a - only_b) / n
    se = 100 * math.sqrt(max(only_a + only_b - (only_a - only_b) ** 2 / n, 0.0)) / n
    return delta, se


def select(rows: dict, arms) -> str:
    """Development-only choice: minimum final task_development choice KL (ties: first listed)."""

    return min(arms, key=lambda arm: (rows[arm]["final"]["task_development"]["choice_kl"], arms.index(arm)))


def summarize(task: str, teacher: str, entry: dict, labels: torch.Tensor) -> dict:
    by_case = {}
    for case in CASES:
        rows = {r["onchip_arm"]: r for r in entry["rows"] if r["case"] == case}
        learn, hold = select(rows, LEARN), select(rows, HOLD)
        test = {arm: row["final"]["task_test"] for arm, row in rows.items()}
        right = {arm: correct(value["predictions"], labels) for arm, value in test.items()}
        delta, se = paired(right[learn], right[hold])
        kl_a, kl_l = test[hold]["choice_kl"], test[learn]["choice_kl"]
        by_case[case] = {
            "L": learn, "A": hold,
            "delta_points": delta, "se_points": se,
            "verdict": ("helps" if delta >= 0.5 and delta >= 2 * se else
                        "harmful" if delta <= -2 * se else "inconclusive"),
            "choice_kl_residual_closed": (kl_a - kl_l) / kl_a if kl_a > 0 else None,
            "accuracy_norm_percent": {arm: value["accuracy_norm_percent"] for arm, value in test.items()},
            "choice_kl": {arm: value["choice_kl"] for arm, value in test.items()},
            "development_choice_kl": {arm: row["final"]["task_development"]["choice_kl"] for arm, row in rows.items()},
            "clean_accuracy_norm_percent": rows["none"]["clean"]["task_test"]["accuracy_norm_percent"],
            "deployed_accuracy_norm_percent": rows["none"]["initial"]["task_test"]["accuracy_norm_percent"],
            "stream_teacher_kl": {arm: row["final"]["test"]["teacher_kl"] for arm, row in rows.items()},
            "wikitext_perplexity": {arm: row["final"].get("transfer_wikitext", {}).get("perplexity")
                                    for arm, row in rows.items()},
            "learn_cost": rows[learn]["cost"],
            "rewrite": {k: v for k, v in (rows[learn].get("fault_rewrite") or {}).items()
                        if k in ("cost", "flagged_devices", "controller")},
            "correct": {arm: value for arm, value in right.items()},
        }
    prepare = entry["prepare"] or {}
    teacher_task = prepare.get("task", {}).get("task_test", {}).get("teacher", {})
    return {"task": task, "teacher": teacher, "cases": by_case,
            "teacher_accuracy_norm_percent": teacher_task.get("accuracy_norm_percent"),
            "runs": entry["runs"]}


def bootstrap_difference(ft: dict, pt: dict, samples=10000, seed=18) -> tuple[float, float]:
    """Paired bootstrap SE of (L-A)_FT - (L-A)_PT over the shared test examples."""

    a = ft["correct"][ft["L"]].double() - ft["correct"][ft["A"]].double()
    b = pt["correct"][pt["L"]].double() - pt["correct"][pt["A"]].double()
    difference = 100 * (a - b)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, len(difference), (samples, len(difference)), generator=generator)
    draws = difference[index].mean(1)
    return float(difference.mean()), float(draws.std())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    found = measurements(args.runs)
    report = {"schema": "exp18.review.v1", "tasks": {}}
    for task, k in TASK_CHOICES.items():
        ids = {teacher: f"opt125m-cmo-l6-exp18-{task}-{teacher}" for teacher in ("ft", "pt")}
        if not all(i in found and found[i]["rows"] for i in ids.values()):
            report["tasks"][task] = {"coverage": {t: i in found for t, i in ids.items()}}
            continue
        document = torch.load(args.tasks_dir / f"{task}-opt125m-v1.pt", map_location="cpu", weights_only=True)
        labels = document["splits"]["test"]["labels"]
        sides = {teacher: summarize(task, teacher, found[i], labels) for teacher, i in ids.items()}
        chance = 100 / k
        pt_teacher = sides["pt"]["teacher_accuracy_norm_percent"]
        near_chance = pt_teacher is not None and abs(pt_teacher - chance) <= 3
        main_question = {}
        for case in CASES[1:]:
            ft, pt = sides["ft"]["cases"][case], sides["pt"]["cases"][case]
            d, se = bootstrap_difference(ft, pt)
            closed = (ft["choice_kl_residual_closed"] or 0) > (pt["choice_kl_residual_closed"] or 0)
            verdict = ("larger" if d >= 0.5 and d >= 2 * se and closed else "not larger" if d <= 0 else "inconclusive")
            if near_chance:
                verdict = "inconclusive on accuracy (PT near chance); see choice KL"
            main_question[case] = {"D_points": d, "se_D_points": se, "ft_kl_closed_larger": closed,
                                   "verdict": verdict}
        for side in sides.values():
            for value in side["cases"].values():
                value.pop("correct")
        report["tasks"][task] = {"chance_percent": chance, "pt_teacher_near_chance": near_chance,
                                 "sides": sides, "main_question": main_question}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({task: value.get("main_question", value) for task, value in report["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
