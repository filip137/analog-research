"""Explicit dependency manifests for many-pass recovery; reuse the EBL executor."""

import argparse
import json
from pathlib import Path

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.epoch_config import EXPERIMENT_ID, default_config


def prepare(root, dataset, prior_execution, smoke=False):
    root = Path(root).resolve()
    if (root / "campaign.json").exists():
        raise ValueError("Cannot overwrite a frozen epoch campaign.")
    prior = json.loads(Path(prior_execution).read_text())
    completed = prior["completed"]
    cache = completed[f"{dataset}_cache"]["artifacts"]["checkpoints/cache.pt"]
    source = completed[f"{dataset}_tune"]["artifacts"]["checkpoints/recovery.pt"]
    common = dict(dataset=dataset, device="cuda:0", cpu_threads=4,
                  calibration_lr=0.0003 if dataset == "cifar10" else 0.001)
    if smoke:
        common.update(max_examples=128, epochs=2, milestones=(1, 2), tuning_arrays=1,
                      fault_kinds=("gmax",), decay_after=(1,))
    nodes = []
    for stage in ("tune", "screen1", "screen2", "screen3"):
        if smoke and stage in ("screen2", "screen3"):
            continue
        index = 1 if stage == "tune" else int(stage[-1])
        name = f"{dataset}_{stage}"
        config = default_config(**common, stage="tune" if stage == "tune" else "screen",
                                array_seed=101000 + index, endpoint_seed=111000 + index)
        path = root / "configs" / f"{name}.json"
        atomic_write_json(path, config)
        inputs = {"device-data": cache, "weights": source if stage == "tune" else {"node": f"{dataset}_tune", "artifact": "checkpoints/selection.pt"}}
        nodes.append(dict(id=name, dataset=dataset, config=str(path.relative_to(root)),
                          sha256=sha256_file(path), inputs=inputs,
                          depends=[] if stage == "tune" else [f"{dataset}_tune"]))
    study_id = f"{dataset}-pcm-recovery-epochs-{'canary' if smoke else '20'}-v1"
    plan = {
        "schema_version": 1, "study_id": study_id,
        "title": f"{dataset}: recovery convergence after digital/HWA deployment",
        "evidence_class": "exploratory",
        "hypothesis": "Additional passes over the same adaptation images can improve teacher fidelity beyond one-pass recovery and equally trained calibration/rewrite controls.",
        "motivation": "User-requested extension of the completed one-pass PCM failure experiment to both CIFAR datasets.",
        "arms": [dict(arm_id=n["id"], configs=[n["config"]], description=n["id"], experiment_id=EXPERIMENT_ID, mode="train") for n in nodes],
        "completion_criteria": [
            "One valid completed native run per declared arm, preserving failed attempts.",
            "Full: 48 development trajectories and 3 fresh arrays with 90 controls each; all learning trajectories run 20 passes over 5000 images.",
            "Metrics at 1,2,5,10,20 passes on 10000 test images; held-out development evaluated every pass.",
            "Exact paired P0/RNG, permanent fault identities, unchanged prefix and matched rewrite counts.",
        ],
        "analysis_plan": [
            "Primary mean teacher KL at T=1; also accuracy, paired improvement over calibration/rewrite, image and programming budgets.",
            "Schedules and optional reporting epoch selected only on independent development arrays; show every predeclared test milestone without test-based selection.",
            "Digital/standard/noise HWA: nominal and 1% open/Gmax/random; each CDT: nominal and 1% of its own fault type.",
            "Preserve earlier HWA convergence limitations; three arrays are descriptive spread, not a significance test.",
        ],
    }
    atomic_write_json(root / "study-plan.json", plan)
    campaign = dict(schema="cifar_crossbar.campaign.v1", study_id=study_id,
                    phase="recovery_epochs", plan="study-plan.json",
                    plan_sha256=sha256_file(root / "study-plan.json"), selection_inputs={}, nodes=nodes)
    atomic_write_json(root / "campaign.json", campaign)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--prior-execution", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    value = prepare(args.output, args.dataset, args.prior_execution, args.smoke)
    print(json.dumps({"study": value["study_id"], "nodes": len(value["nodes"])}))


if __name__ == "__main__":
    main()
