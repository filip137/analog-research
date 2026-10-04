"""Prepare the explicit five-full-epoch dependency graph for the EBL executor."""

import argparse
from pathlib import Path
import json

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.full_epoch_config import EXPERIMENT_ID, default_config


def prepare(root, dataset, backend, inputs, *, smoke=False):
    root, inputs = Path(root), Path(inputs)
    if root.exists():
        raise ValueError("Cannot overwrite a frozen full-epoch campaign.")
    remote_inputs = Path("/home/filip/cifar_full_epoch_inputs")

    def external(name):
        return {"path": str(remote_inputs / name), "sha256": sha256_file(inputs / name)}

    common = dict(dataset=dataset, backend=backend, device="cuda:0", cpu_threads=4,
                  recovery_model="gaussian_endpoint_reprogramming" if backend == "pcm" else "om_closed_loop_pulse_adam",
                  calibration_lr=.0003 if dataset == "cifar10" else .001)
    if smoke:
        common.update(max_examples=128, hwa_epochs=1)
    nodes = []
    for stage in ("cache", "sources", "screen1", "screen2", "screen3"):
        if smoke and stage in ("screen2", "screen3"):
            continue
        actual = "screen" if stage.startswith("screen") else stage
        index = int(stage[-1]) if actual == "screen" else 1
        name = f"{dataset}_{backend}_{stage}"
        config = default_config(**common, stage=actual, array_seed=151000 + index, endpoint_seed=161000 + index)
        path = root / "configs" / f"{name}.json"
        atomic_write_json(path, config)
        bindings = {}
        if stage != "cache":
            bindings["device-data"] = {"node": f"{dataset}_{backend}_cache", "artifact": "checkpoints/cache.pt"}
            if backend == "om":
                bindings["device-model"] = external(f"{dataset}_om.pt")
        if stage == "sources" and backend == "pcm":
            bindings["weights"] = external(f"{dataset}_pcm_source.pt")
        if actual == "screen":
            bindings["weights"] = {"node": f"{dataset}_{backend}_sources", "artifact": "checkpoints/sources.pt"}
        nodes.append({"id": name, "dataset": dataset, "config": str(path.relative_to(root)),
                      "sha256": sha256_file(path), "inputs": bindings,
                      "depends": sorted({v["node"] for v in bindings.values() if "node" in v})})
    study_id = f"{dataset}-{backend}-five-full-epochs{'-canary' if smoke else ''}-v1"
    plan = {"schema_version": 1, "study_id": study_id,
            "title": f"{dataset} {backend}: five full epochs after digital/HWA deployment",
            "evidence_class": "exploratory", "hypothesis": "Five full-cohort recovery epochs can reduce teacher KL beyond matched calibration and unchanged-target rewriting on some fixed faulty arrays.",
            "motivation": "User requested a five-epoch repeat and OM devices; see docs/cifar_full_epoch_recovery.md.",
            "arms": [{"arm_id": n["id"], "configs": [n["config"]], "description": n["id"], "experiment_id": EXPERIMENT_ID, "mode": "train"} for n in nodes],
            "completion_criteria": ["All native stages and paired controls complete with verified artifacts.",
                                    "Full: 3 independent arrays, each 2 sources x 4 cases x 5 controls; 5 full 45000-image epochs for every adapting control and 10000 test images each epoch.",
                                    "Preserve P0, fixed faults, frozen prefix, optimizer/RNG states and device-specific write costs. Canary is explicitly truncated and non-scientific."],
            "analysis_plan": ["Mean teacher KL at T=1, accuracy, array spread and paired differences against equal-epoch calibration/rewrite.",
                              "Report every epoch without test selection, apparent versus persistent OM metrics, pulse-cap saturation and limited HWA tuning.",
                              "PCM standard HWA is reused; OM HWA is independently fitted using its own endpoint kernel. Distinguish full programming from incremental pulses."]}
    atomic_write_json(root / "study-plan.json", plan)
    value = {"schema": "cifar_crossbar.campaign.v1", "study_id": study_id, "phase": "full_epochs",
             "plan": "study-plan.json", "plan_sha256": sha256_file(root / "study-plan.json"),
             "selection_inputs": {}, "nodes": nodes}
    atomic_write_json(root / "campaign.json", value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    p.add_argument("--backend", choices=("pcm", "om"), required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    print(json.dumps(prepare(a.output, a.dataset, a.backend, a.inputs, smoke=a.smoke)))


if __name__ == "__main__":
    main()
