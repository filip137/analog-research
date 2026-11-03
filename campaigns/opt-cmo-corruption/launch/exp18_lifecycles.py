"""Write the exp-018 lifecycles: one HWA source, three learning-rate screens and the main comparison.

Every lifecycle of a (task, teacher) pair shares the network, data, devices,
deployment and program-and-verify sections and the HWA settings. The screens
and the main comparison therefore reuse the source lifecycle's HWA artifact
(``workflow plan --hwa-source``).
- The screens run assignment 271001 with smaller test cohorts.
- The main comparison runs 271002.
- Each plan is trimmed to its assignment's stages.
The FT and PT lifecycles of a task differ only in ``network.teacher``.

    python campaigns/opt-cmo-corruption/launch/exp18_lifecycles.py sources
    python campaigns/opt-cmo-corruption/launch/exp18_lifecycles.py screens
    python campaigns/opt-cmo-corruption/launch/exp18_lifecycles.py main --frozen frozen.json

``frozen.json`` maps ``<task>-<ft|pt>`` to ``{"learning_rate": .., "calibration_learning_rate": ..}``.
Teacher digests are read from ``campaigns/opt-cmo-corruption/teachers/selected.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "campaigns/opt-cmo-corruption"
NOTE = "campaigns/opt-cmo-corruption/series/003-finetuned-tasks/experiments/exp-018-finetuned-task-recovery.md"
TASKS = {
    # stream sha256, task-file sha256, HWA cohort, development examples, test examples
    "hellaswag": ("bfa4e8dd656b0aa645c35009665d773405dd6160037ae8ea9c600dab232f8deb",
                  "a4e4e9773bab1b56f36856fc8b2e95e700e8e8263137257de939f81799f04f9b", 10240, 2000, 10042),
    "sst2": ("ace4201b30b6936be6c0f8acaae8d8a2aa648b3611a9c3e1cde45979e2f40e6c",
             "d0e4e957aab38dd2da79b7ab3b54bbf77cd8b646dbb593e610070be8718c66bb", 6144, 3000, 872),
    "piqa": ("25670635606c8308d93b8570e82928d89e4313bf4052d8afed49619106af2a05",
             "767bfbf120e57370296b3db4ff53132da7c005ffba151e9b070c179219d849a0", 2048, 1500, 1838),
}
VERIFY_PAIR = {"detection": "verify", "roles": "pair", "seed": 241016, "threshold_sigma": 5.0}
SCREEN = (("a", 0.001, 0.0001), ("b", 0.003, 0.0003), ("c", 0.01, 0.001))
CASES = [{"array_concentration": None, "die_concentration": None, "kind": kind, "rate_ppm": ppm}
         for kind, ppm in (("none", 0), ("random", 30000), ("gmax", 30000))]
LABELS = ["nominal", "random_30000ppm", "gmax_30000ppm"]
WIKITEXT = {"corpus": {"name": "wikitext-2-raw-v1",
                       "sha256": "33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01"},
            "label": "wikitext", "sequences": 256, "split": "test"}


def arm(arm_id, weights, calibration, rewrite):
    value = {"calibration": calibration, "id": arm_id, "weights": weights}
    if rewrite:
        value["fault_rewrite"] = dict(VERIFY_PAIR)
    return value


ARMS = [
    arm("none", "hold", False, False),
    arm("calibration", "hold", True, False),
    arm("pair_rewrite_verify", "hold", False, True),
    arm("pair_rewrite_verify_calibration", "hold", True, True),
    arm("pair_rewrite_verify_recovery", "learn", False, True),
    arm("pair_rewrite_verify_recovery_calibration", "learn", True, True),
]


def base(task: str, teacher: str) -> dict:
    stream, choices, hwa_sequences, development, test = TASKS[task]
    network = {"analog_decoder_layers": 6, "family": "opt_mlp_suffix", "model": "facebook/opt-125m"}
    if teacher == "ft":
        selected = json.loads((CAMPAIGN / "teachers/selected.json").read_text())[task]
        network["teacher"] = {"name": selected["teacher_id"], "sha256": selected["sha256"]}
    return {
        "schema": "crossbar_lifecycle",
        "schema_version": 1,
        "network": network,
        "data": {
            "corpus": {"name": f"{task}-opt125m-v1-stream", "sha256": stream},
            "data_seed": 20261008, "development_sequences": 128, "evaluation": "test",
            "hwa_sequences": hwa_sequences, "max_examples": 0, "recovery_sequences": 2048,
            "sequence_length": 256, "test_sequences": 64,
        },
        "task": {"name": task, "choices": {"name": f"{task}-opt125m-v1", "sha256": choices},
                 "decision": "acc_norm", "development_examples": development, "evaluation_examples": test},
        "defects": {"arrays_per_die": 48, "cases": copy.deepcopy(CASES), "persistence": "permanent",
                    "placement": "hierarchical_die_array"},
        "deployment": {"calibration": "output_gain_mlp_layer_norm_mlp_bias", "encoding": "active_reference_pair",
                       "read_noise": 0.0, "scaling": "per_output_channel_absmax", "tile_size": 256},
        "devices": {"assignment_seeds": [271001, 271002], "endpoint_seed_offset": 10000,
                    "model": "aihwkit_1.1.0_reram_cmo", "noise_scale": 1.0, "selection_seeds": [211001, 211002],
                    "technology": "cmo"},
        "hwa": {
            "arms": [{"corruption": None, "digital_learning_rate": 1e-05, "id": "standard_hwa_model",
                      "method": "hwa", "noise_strength": 1.0, "scope": "model",
                      "selection_cases": [copy.deepcopy(CASES[0]), copy.deepcopy(CASES[1])]}],
            "augment": False, "batch_size": 32, "epochs": 50, "learning_rate": 0.005, "lr_decay": 1.0,
            "lr_milestones": [],
            "lr_schedule": {"kind": "cosine_warmup_decay", "min_factor": 0.2, "total_steps": 3200,
                            "warmup_steps": 26},
            "noise_model": "additive_gaussian_weight", "noise_scaling": "output_channel_abs_max",
            "noise_std": 0.03, "objective": "teacher_kl", "optimizer": "adam", "seed": 41,
            "selection": {"every_epochs": 8, "extension_epochs": 0}, "sequences_per_epoch": 2048,
        },
        "program_verify": {"acceptance_percent": 2.0, "method": "gaussian_endpoint",
                           "relaxation": {"model": "none"}},
        "transfer_tests": [copy.deepcopy(WIKITEXT)],
    }


def onchip(learning_rate, calibration_rate, arms, labels) -> dict:
    return {
        "accumulate_updates": True, "arms": copy.deepcopy(arms), "batch_size": 32,
        "calibration_learning_rate": calibration_rate, "case_labels": list(labels), "epochs": 4,
        "learning_rate": learning_rate, "max_write_attempts_per_device": 4, "objective": "teacher_kl",
        "seed": 53, "threshold_sigma": 2.0, "training_cohort": "recovery", "update_law": "cmo_thresholded_write_adam",
    }


def question(text, decision, metric="task_accuracy_norm_percent") -> dict:
    return {"decision": decision, "evidence_class": "exploratory_model_based", "primary_metric": metric,
            "text": text}


def write(doc: dict, lifecycle_id: str) -> Path:
    doc = {"lifecycle_id": lifecycle_id, **doc}
    path = CAMPAIGN / "lifecycles" / f"{lifecycle_id}.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def sources(teachers, tasks=TASKS) -> list[Path]:
    paths = []
    for task in tasks:
        for teacher in teachers:
            doc = base(task, teacher)
            doc["onchip"] = None
            doc["question"] = question(
                f"exp-018 HWA source: std3 HWA of the {teacher.upper()} {task} teacher on its task stream.",
                f"Source only; the screens and main comparison reuse it. {NOTE}")
            paths.append(write(doc, f"opt125m-cmo-l6-exp18-{task}-{teacher}-source"))
    return paths


def screens(teachers, tasks=TASKS) -> list[Path]:
    paths = []
    for task in tasks:
        for teacher in teachers:
            for tag, rate, calibration in SCREEN:
                doc = base(task, teacher)
                # Screens never read test data for selection; a small task test cohort and no transfer test
                # keep stages cheap. ``data`` stays identical, because the HWA source depends on it.
                doc["task"]["evaluation_examples"] = 100
                del doc["transfer_tests"]
                doc["onchip"] = onchip(rate, calibration, ARMS[:5], ["random_30000ppm"])
                doc["question"] = question(
                    f"exp-018 development screen {tag}: weight LR {rate}, calibration LR {calibration}, "
                    f"{teacher.upper()} {task}, array 271001, random 3%.",
                    "Select the weight LR on rewrite+learn and the calibration LR on rewrite+calibration by "
                    f"minimum final task_development choice_kl (ties: lower LR). {NOTE}")
                paths.append(write(doc, f"opt125m-cmo-l6-exp18-{task}-{teacher}-screen-{tag}"))
    return paths


def main_lifecycles(teachers, frozen: dict, tasks=TASKS) -> list[Path]:
    paths = []
    for task in tasks:
        for teacher in teachers:
            rates = frozen[f"{task}-{teacher}"]
            doc = base(task, teacher)
            doc["onchip"] = onchip(rates["learning_rate"], rates["calibration_learning_rate"], ARMS, LABELS)
            doc["question"] = question(
                f"exp-018: does on-chip training after the verify partner rewrite gain more task accuracy for the "
                f"{'fine-tuned' if teacher == 'ft' else 'pretrained'} {task} teacher than the best "
                "no-weight-learning row, on array 271002?",
                f"Frozen paired acc_norm rule of the exp-018 note (L vs A chosen on development choice_kl). {NOTE}")
            paths.append(write(doc, f"opt125m-cmo-l6-exp18-{task}-{teacher}"))
    return paths


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("sources", "screens", "main"))
    parser.add_argument("--teachers", nargs="+", default=["ft", "pt"], choices=("ft", "pt"))
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=tuple(TASKS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    if args.kind == "sources":
        written = sources(args.teachers, args.tasks)
    elif args.kind == "screens":
        written = screens(args.teachers, args.tasks)
    else:
        written = main_lifecycles(args.teachers, json.loads(args.frozen.read_text()), args.tasks)
    print("\n".join(str(p.relative_to(ROOT)) for p in written))
