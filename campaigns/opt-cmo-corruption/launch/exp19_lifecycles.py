"""Write the exp-019 lifecycles: analog LoRA adapters on the frozen std3 HWA base, per task.

Each lifecycle declares rank-16 analog adapters on fc1/fc2 of layers 6–11 and three LoRA arms (Adam
LR 1e-4, 3e-4, 1e-3) on the pinned std3 source. Its defect cases put stuck devices in all arrays,
only the base arrays or only the adapter arrays, at the same per-device rates. Only the
development-selected arm is deployed; that choice and the trimmed manifests are the driver's.

    python campaigns/opt-cmo-corruption/launch/exp19_lifecycles.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTE = "campaigns/opt-cmo-corruption/series/003-finetuned-tasks/experiments/exp-019-analog-lora.md"
BASE = {"name": "exp1g-std3", "sha256": "86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028"}
TASKS = {
    # stream sha256, task-file sha256, development, test, training examples, epochs
    "hellaswag": ("bfa4e8dd656b0aa645c35009665d773405dd6160037ae8ea9c600dab232f8deb",
                  "a4e4e9773bab1b56f36856fc8b2e95e700e8e8263137257de939f81799f04f9b", 2000, 10042, 37905, 2),
    "sst2": ("ace4201b30b6936be6c0f8acaae8d8a2aa648b3611a9c3e1cde45979e2f40e6c",
             "d0e4e957aab38dd2da79b7ab3b54bbf77cd8b646dbb593e610070be8718c66bb", 3000, 872, 64349, 2),
    "piqa": ("25670635606c8308d93b8570e82928d89e4313bf4052d8afed49619106af2a05",
             "767bfbf120e57370296b3db4ff53132da7c005ffba151e9b070c179219d849a0", 1500, 1838, 14612, 4),
}
RATES = ((1e-4, "lr1em4"), (3e-4, "lr3em4"), (1e-3, "lr1em3"))


def case(kind="none", ppm=0, target=None):
    value = {"array_concentration": None, "die_concentration": None, "kind": kind, "rate_ppm": ppm}
    return value if target is None else {**value, "target": target}


CASES = [case()] + [case(kind, ppm, target) for kind, ppm in (("random", 10000), ("random", 30000), ("gmax", 30000))
                    for target in (None, "base", "adapter")]
REWRITE = {"detection": "verify", "roles": "pair", "seed": 241016, "threshold_sigma": 5.0}


def lifecycle(task: str) -> dict:
    stream, choices, development, test, training, epochs = TASKS[task]
    return {
        "lifecycle_id": f"opt125m-cmo-l6-exp19-analog-lora-{task}",
        "schema": "crossbar_lifecycle",
        "schema_version": 1,
        "question": {
            "text": f"exp-019: do stuck devices in analog LoRA adapter arrays hurt {task} task accuracy more than "
                    "the same per-device rate in the base MLP arrays of the frozen std3 HWA OPT-125M?",
            "decision": f"Per fault kind and rate on array 271002, compare paired test acc_norm loss of "
                        f"adapter-only vs base-only faults against the healthy deployment. {NOTE}",
            "primary_metric": "task_accuracy_norm_percent",
            "evidence_class": "exploratory_model_based",
        },
        "network": {"family": "opt_mlp_suffix", "model": "facebook/opt-125m", "analog_decoder_layers": 6,
                    "analog_lora": {"rank": 16, "alpha": 32.0}},
        "data": {"corpus": {"name": f"{task}-opt125m-v1-stream", "sha256": stream}, "data_seed": 20261008,
                 "development_sequences": 128, "evaluation": "test", "hwa_sequences": 2048, "max_examples": 0,
                 "recovery_sequences": 2048, "sequence_length": 256, "test_sequences": 64},
        "task": {"name": task, "choices": {"name": f"{task}-opt125m-v1", "sha256": choices}, "decision": "acc_norm",
                 "development_examples": development, "evaluation_examples": test, "training_examples": training},
        "devices": {"assignment_seeds": [271002], "endpoint_seed_offset": 10000, "model": "aihwkit_1.1.0_reram_cmo",
                    "noise_scale": 1.0, "selection_seeds": [211001, 211002], "technology": "cmo"},
        "defects": {"arrays_per_die": 48, "cases": copy.deepcopy(CASES), "persistence": "permanent",
                    "placement": "hierarchical_die_array"},
        "deployment": {"calibration": "output_gain_mlp_layer_norm_mlp_bias", "encoding": "active_reference_pair",
                       "read_noise": 0.0, "scaling": "per_output_channel_absmax", "tile_size": 256},
        "hwa": {
            # LoRA arms use their own Adam rates; the shared rate is a required field they do not read.
            "arms": [{"id": f"lora_{tag}", "method": "lora", "noise_strength": 1.0, "selection_cases": [case()],
                      "base_source": dict(BASE), "learning_rate": rate, "warmup_fraction": 0.06}
                     for rate, tag in RATES],
            "augment": False, "batch_size": 32, "epochs": epochs, "learning_rate": 0.0003, "lr_decay": 1.0,
            "lr_milestones": [], "noise_model": "additive_gaussian_weight", "noise_scaling": "output_channel_abs_max",
            "noise_std": 0.03, "objective": "task_choice_token_mean", "optimizer": "adam", "seed": 41,
            "selection": {"every_epochs": 1, "extension_epochs": 0},
        },
        "program_verify": {"acceptance_percent": 2.0, "method": "gaussian_endpoint", "relaxation": {"model": "none"}},
        "onchip": {
            "accumulate_updates": True, "batch_size": 32, "calibration_learning_rate": 0.0, "epochs": 1,
            "learning_rate": 0.003, "max_write_attempts_per_device": 4, "objective": "teacher_kl", "seed": 53,
            "threshold_sigma": 2.0, "training_cohort": "recovery", "update_law": "cmo_thresholded_write_adam",
            "arms": [{"calibration": False, "id": "none", "weights": "hold"},
                     {"calibration": False, "fault_rewrite": dict(REWRITE), "id": "pair_rewrite_verify",
                      "weights": "hold"}],
        },
    }


if __name__ == "__main__":
    for task in TASKS:
        doc = lifecycle(task)
        path = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / f"{doc['lifecycle_id']}.json"
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        print(path.relative_to(ROOT))
