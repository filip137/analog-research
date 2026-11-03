"""Write the exp-020 lifecycles: on-chip recovery of the exp-019 analog-LoRA deployments.

Each lifecycle copies its exp-019 lifecycle and keeps only the development-selected LoRA arm,
so the frozen exp-019 source is reused unchanged (``workflow plan --hwa-source``). Only defects,
on-chip settings and the question change. On-chip arms follow the verify rewrite and minimise
KL to the clean deployed source (``source_kl``), with digital calibration, adapter-only
learning and all-weight learning.

    python campaigns/opt-cmo-corruption/launch/exp20_lifecycles.py screens
    python campaigns/opt-cmo-corruption/launch/exp20_lifecycles.py main --frozen launch/exp20_frozen.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LIFECYCLES = ROOT / "campaigns/opt-cmo-corruption/lifecycles"
NOTE = "campaigns/opt-cmo-corruption/series/003-finetuned-tasks/experiments/exp-020-adapter-onchip-recovery.md"
SELECTED = {"hellaswag": "lora_lr1em3", "sst2": "lora_lr1em3", "piqa": "lora_lr3em4"}
SCREEN = (("a", 0.001, 0.0001), ("b", 0.003, 0.0003), ("c", 0.01, 0.001))
REWRITE = {"detection": "verify", "roles": "pair", "seed": 241016, "threshold_sigma": 5.0}


def case(kind="none", ppm=0, target=None):
    value = {"array_concentration": None, "die_concentration": None, "kind": kind, "rate_ppm": ppm}
    return value if target is None else {**value, "target": target}


MAIN_CASES = [case()] + [case(kind, ppm, target) for kind, ppm in (("gmax", 30000), ("gmax", 60000), ("random", 60000))
                         for target in (None, "base", "adapter")]


def arms(adapter_rate=None, all_rate=None):
    def learn(arm_id, scope, rate):
        value = {"calibration": False, "fault_rewrite": dict(REWRITE), "id": arm_id, "weights": "learn"}
        if scope == "adapter":
            value["write_scope"] = "adapter"
        if rate is not None:
            value["learning_rate"] = rate
        return value

    return [
        {"calibration": False, "id": "none", "weights": "hold"},
        {"calibration": True, "id": "calibration", "weights": "hold"},
        {"calibration": False, "fault_rewrite": dict(REWRITE), "id": "pair_rewrite_verify", "weights": "hold"},
        {"calibration": True, "fault_rewrite": dict(REWRITE), "id": "pair_rewrite_verify_calibration", "weights": "hold"},
        learn("pair_rewrite_verify_adapter_recovery", "adapter", adapter_rate),
        learn("pair_rewrite_verify_recovery", "all", all_rate),
    ]


def base(task: str) -> dict:
    doc = json.loads((LIFECYCLES / f"opt125m-cmo-l6-exp19-analog-lora-{task}.json").read_text())
    doc["hwa"]["arms"] = [arm for arm in doc["hwa"]["arms"] if arm["id"] == SELECTED[task]]
    doc["onchip"] = {
        "accumulate_updates": True, "batch_size": 32, "epochs": 4, "epoch_evaluation": "development",
        "max_write_attempts_per_device": 4, "objective": "source_kl", "seed": 53, "threshold_sigma": 2.0,
        "training_cohort": "recovery", "update_law": "cmo_thresholded_write_adam",
    }
    return doc


def write(doc, lifecycle_id):
    doc = {**doc, "lifecycle_id": lifecycle_id}
    path = LIFECYCLES / f"{lifecycle_id}.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def screens():
    paths = []
    for task in SELECTED:
        for tag, rate, calibration in SCREEN:
            doc = base(task)
            doc["defects"]["cases"] = [case(), case("gmax", 60000)]
            doc["onchip"].update(learning_rate=rate, calibration_learning_rate=calibration, arms=arms(),
                                 case_labels=["gmax_60000ppm"])
            doc["question"]["text"] = (f"exp-020 development screen {tag}: {task}, weight LR {rate}, calibration LR "
                                       f"{calibration}, gmax 6%, array 271002, development data only.")
            doc["question"]["decision"] = ("Per learn arm, select the weight LR, and for rewrite+calibration the "
                                           "calibration LR, by the highest final task_development agreement with the "
                                           f"clean adapted model (ties: lower LR). {NOTE}")
            paths.append(write(doc, f"opt125m-cmo-l6-exp20-{task}-screen-{tag}"))
    return paths


def main_lifecycles(frozen: dict):
    paths = []
    for task in SELECTED:
        rates = frozen[task]
        doc = base(task)
        doc["defects"]["cases"] = copy.deepcopy(MAIN_CASES)
        doc["onchip"].update(learning_rate=rates["all"], calibration_learning_rate=rates["calibration"],
                             arms=arms(rates["adapter"], rates["all"]))
        doc["question"]["text"] = (f"exp-020: after the verify rewrite, does on-chip learning (adapters only or all "
                                   f"weights) recover more {task} accuracy than digital calibration, and what does it write?")
        doc["question"]["decision"] = f"Frozen paired acc_norm rule of the exp-020 note. {NOTE}"
        paths.append(write(doc, f"opt125m-cmo-l6-exp20-{task}"))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("screens", "main"))
    parser.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    written = screens() if args.kind == "screens" else main_lifecycles(json.loads(args.frozen.read_text()))
    print("\n".join(str(p.relative_to(ROOT)) for p in written))
