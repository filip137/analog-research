"""Materialize explicit studies and execute their native EBL dependency graph.

This module orchestrates subprocesses only. Numerical stages live in runtime.py.
Every generated config is written before study preparation; inputs are named by
an exact upstream receipt or a path and digest, never by a latest-file search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.config import EXPERIMENT_ID, default_config
from experiments.cifar_crossbar.prepare import SOURCES

METHODS = ("none", "calibration", "rewrite", "open_loop", "closed_loop")
HWA_SEEDS = (41, 43, 47)


def default_backends(phase):
    """PCM means statistical endpoint programming unless pulses are explicit."""
    if phase == "reference":
        return ["pcm_inference"]
    if phase == "hwa":
        return ["om", "pcm_inference"]
    return ["om"]


def cases(backend):
    values = {"nominal": {}}
    if backend == "om":
        values["native"] = {"policy": "native"}
    values.update(
        {
            "noise05": {"noise_scale": 0.5},
            "noise2": {"noise_scale": 2.0},
            "variation2": {"variation_scale": 2.0},
        }
    )
    for kind in ("low", "high", "middle"):
        for rate in (0.001, 0.01, 0.05):
            values[f"stuck_{kind}_{int(rate*1000)}permil"] = {
                "fault_kind": kind,
                "fault_rate": rate,
            }
    for noise in (0.01, 0.05, 0.1):
        values[f"read{int(noise*1000)}"] = {"read_noise": noise}
    return values


def read_receipt(path, schema):
    value = json.loads(Path(path).read_text())
    if value.get("schema") != schema:
        raise ValueError(f"Expected {schema}: {path}")
    return value


def external(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def upstream(node, artifact):
    return {"node": node, "artifact": artifact}


@dataclass
class Builder:
    root: Path
    study_id: str
    phase: str
    common: dict
    nodes: list = field(default_factory=list)

    def add(self, name, stage, *, inputs=None, depends=(), **changes):
        if any(n["id"] == name for n in self.nodes):
            raise ValueError(f"Duplicate campaign node: {name}")
        config = default_config(**{**self.common, **changes, "stage": stage})
        path = self.root / "configs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, config)
        self.nodes.append(
            {
                "id": name,
                "config": str(path.relative_to(self.root)),
                "sha256": sha256_file(path),
                "dataset": config["dataset"],
                "inputs": inputs or {},
                "depends": list(depends),
            }
        )
        return name

    def finish(self, selection_inputs):
        plan = {
            "schema_version": 1,
            "study_id": self.study_id,
            "title": f"CIFAR ResNet suffix recovery: {self.phase}",
            "evidence_class": "exploratory" if self.phase == "pilot" else "model_based",
            "hypothesis": (
                "Observable pulse recovery can improve fresh-array accuracy beyond "
                "population HWA, digital calibration and a no-learning rewrite."
            ),
            "motivation": "Locate hardware mismatch regimes requiring same-task on-chip adaptation.",
            "arms": [
                {
                    "arm_id": n["id"],
                    "configs": [n["config"]],
                    "description": n["id"],
                    "experiment_id": EXPERIMENT_ID,
                    "mode": "train",
                }
                for n in self.nodes
            ],
            "completion_criteria": [
                "Exactly one valid completed native run for every declared config.",
                "Exact P0 pairing, unchanged digital prefix and no test-set selection.",
                "A pilot validates implementation only; formal claims require independent arrays.",
            ],
            "analysis_plan": [
                "Compare paired recovery gain against calibration and frozen-target rewrite.",
                "Average endpoint replicas within arrays; retain negative and failed outcomes.",
                "Confirm on fresh arrays using clustered intervals and Holm-adjusted paired tests.",
                "Report accuracy, KL, deployment loss, image presentations, pulses and verify reads.",
            ],
        }
        atomic_write_json(self.root / "study-plan.json", plan)
        campaign = {
            "schema": "cifar_crossbar.campaign.v1",
            "study_id": self.study_id,
            "phase": self.phase,
            "plan": "study-plan.json",
            "plan_sha256": sha256_file(self.root / "study-plan.json"),
            "selection_inputs": selection_inputs,
            "nodes": self.nodes,
        }
        atomic_write_json(self.root / "campaign.json", campaign)
        return campaign


def materialize(args):
    if args.backends is None:
        args.backends = default_backends(args.phase)
    root = args.output.resolve()
    if (root / "campaign.json").exists():
        raise ValueError(
            "Use a new campaign directory; frozen plans are never overwritten."
        )
    common = {
        "device": args.device,
        "suffix": args.suffix,
        "disable_cudnn": args.disable_cudnn,
    }
    if args.phase == "pilot":
        common.update(max_examples=500, epochs=1, ramp_epochs=1, kernel_samples=64)
    b = Builder(root, args.study_id, args.phase, common)
    runtime_path = getattr(args, "runtime_selection", None)
    runtime = (
        read_receipt(runtime_path, "cifar_crossbar.runtime_selection.v1")
        if runtime_path
        else None
    )
    if args.phase != "pilot" and any(
        backend != "pcm_inference" for backend in args.backends
    ):
        if runtime is None or runtime["suffix"] != args.suffix:
            raise ValueError(
                "Scientific pulse campaigns require a matching --runtime-selection from the completed pilot."
            )
        for dataset in args.datasets:
            for backend in args.backends:
                if (
                    backend != "pcm_inference"
                    and not runtime["groups"][f"{dataset}_{backend}"]["passed"]
                ):
                    raise ValueError(
                        "The runtime gate failed: use a smaller suffix and repeat its pilot before selection."
                    )
    hwa = (
        read_receipt(args.hwa_selection, "cifar_crossbar.hwa_selection.v1")
        if args.hwa_selection
        else None
    )
    recovery = (
        read_receipt(args.recovery_selection, "cifar_crossbar.recovery_selection.v1")
        if args.recovery_selection
        else None
    )
    screen = (
        read_receipt(args.screen_selection, "cifar_crossbar.screen_selection.v1")
        if args.screen_selection
        else None
    )
    if args.phase in ("tune", "screen", "confirm", "reference") and hwa is None:
        raise ValueError("This phase requires a frozen --hwa-selection receipt.")
    if args.phase in ("screen", "confirm") and recovery is None:
        raise ValueError("This phase requires a frozen --recovery-selection receipt.")
    if args.phase == "confirm" and screen is None:
        raise ValueError("Confirmation requires development-only --screen-selection.")
    for receipt in (hwa, recovery, screen):
        if receipt and receipt["suffix"] != args.suffix:
            raise ValueError(
                "Selections must use the same suffix; repeat development after changing it."
            )
    selection_inputs = {
        k: external(v)
        for k, v in (
            ("hwa", args.hwa_selection),
            ("recovery", args.recovery_selection),
            ("screen", args.screen_selection),
            ("runtime", runtime_path),
        )
        if v
    }
    for dataset in args.datasets:
        audit = b.add(
            f"{dataset}_audit",
            "audit",
            dataset=dataset,
            evaluation="confirmation",
            max_examples=0,
        )
        for backend in args.backends:
            group = f"{dataset}_{backend}"
            base = {"dataset": dataset, "backend": backend}
            if backend == "pcm_inference" and args.phase not in ("hwa", "reference"):
                raise ValueError(
                    "PCM inference supports only separate HWA/reference campaigns."
                )
            if args.phase == "reference":
                if backend != "pcm_inference":
                    raise ValueError(
                        "Reference campaigns require --backends pcm_inference."
                    )
                weights = hwa["groups"][group]["weights"]
                for endpoint in range(5):
                    for seconds in (1.0, 3600.0, 86400.0, 31536000.0):
                        for compensated in (False, True):
                            for budget in (500, 5000):
                                b.add(
                                    f"{group}_e{endpoint}_t{int(seconds)}_comp{int(compensated)}_n{budget}",
                                    "pcm_reference",
                                    **base,
                                    endpoint_seed=710000 + endpoint,
                                    retention_seconds=seconds,
                                    drift_compensation=compensated,
                                    recovery_images=budget,
                                    evaluation="confirmation",
                                    depends=[audit],
                                    inputs={"weights": weights},
                                )
                continue
            if args.phase == "hwa":
                char = (
                    b.add(f"{group}_kernel", "characterize", **base)
                    if backend != "pcm_inference"
                    else None
                )
                for objective in ("cross_entropy", "teacher_kl"):
                    for lr in (0.0001, 0.0003, 0.001):
                        b.add(
                            f"{group}_{objective}_lr{lr:g}",
                            "hwa",
                            **base,
                            epochs=args.hwa_epochs,
                            objective=objective,
                            learning_rate=lr,
                            depends=[audit],
                            inputs=(
                                {"device-model": upstream(char, "kernel.pt")}
                                if char
                                else {}
                            ),
                        )
                continue
            selected_hwa = hwa["groups"][group] if hwa else {}
            if args.phase == "tune":
                scenarios = {k: cases(backend)[k] for k in ("nominal", "noise2")}
            elif args.phase == "confirm":
                scenarios = {
                    k: cases(backend)[k] for k in screen["groups"][group]["cases"]
                }
            elif args.phase == "screen":
                scenarios = cases(backend)
            else:
                scenarios = {"nominal": {}}
            for scenario, settings in scenarios.items():
                for seed in (HWA_SEEDS if args.phase == "confirm" else HWA_SEEDS[:1]):
                    stem = f"{group}_{scenario}_s{seed}"
                    spec = {**base, **settings, "seed": seed}
                    # Calibration populations and target arrays are independent.
                    kernel = b.add(
                        f"{stem}_kernel", "characterize", **spec, assignment_seed=9001
                    )
                    if args.phase == "tune":
                        weights = selected_hwa["weights"]
                    else:
                        hwa_name = b.add(
                            f"{stem}_hwa",
                            "hwa",
                            **spec,
                            depends=[audit],
                            **(
                                {
                                    k: selected_hwa[k]
                                    for k in ("learning_rate", "objective", "epochs")
                                }
                                if selected_hwa
                                else {}
                            ),
                            inputs={"device-model": upstream(kernel, "kernel.pt")},
                        )
                        weights = upstream(hwa_name, "checkpoints/weights.pt")
                    arrays = (
                        5
                        if args.phase == "confirm"
                        else (1 if args.phase == "pilot" else 2)
                    )
                    endpoints = 2 if args.phase == "confirm" else 1
                    for array in range(arrays):
                        assignment = (
                            (40000 if args.phase == "confirm" else 20000)
                            + seed * 100
                            + array
                        )
                        array_stem = f"{stem}_a{array}"
                        population = b.add(
                            f"{array_stem}_population",
                            "characterize",
                            **spec,
                            assignment_seed=assignment,
                        )
                        for endpoint in range(endpoints):
                            end_seed = (
                                (500000 if args.phase == "confirm" else 300000)
                                + seed * 100
                                + array * 10
                                + endpoint
                            )
                            match = {
                                **spec,
                                "assignment_seed": assignment,
                                "endpoint_seed": end_seed,
                                "evaluation": (
                                    "confirmation"
                                    if args.phase == "confirm"
                                    else "development"
                                ),
                            }
                            pair = f"{array_stem}_e{endpoint}"
                            deploy = b.add(
                                f"{pair}_deploy",
                                "deploy",
                                **match,
                                depends=[audit],
                                inputs={
                                    "weights": weights,
                                    "device-model": upstream(
                                        population, "population.pt"
                                    ),
                                },
                            )
                            for budget in (
                                (500,) if args.phase == "pilot" else (500, 5000)
                            ):
                                for method in METHODS:
                                    if args.phase == "tune" and method in (
                                        "open_loop",
                                        "closed_loop",
                                    ):
                                        trials = [
                                            {
                                                "learning_rate": lr,
                                                "calibration_lr": 0.001,
                                                "epochs": 30,
                                            }
                                            for lr in (0.00001, 0.0001, 0.001)
                                        ]
                                    elif args.phase == "tune" and method in (
                                        "calibration",
                                        "rewrite",
                                    ):
                                        trials = [
                                            {
                                                "learning_rate": 0.0,
                                                "calibration_lr": lr,
                                                "epochs": 30,
                                            }
                                            for lr in (0.0001, 0.001)
                                        ]
                                    elif recovery and method != "none":
                                        trials = [
                                            recovery["groups"][group][
                                                f"{method}_{budget}"
                                            ]
                                        ]
                                    else:
                                        trials = [{}]
                                    for trial, params in enumerate(trials):
                                        b.add(
                                            f"{pair}_{method}_n{budget}_t{trial}",
                                            "recover",
                                            **match,
                                            method=method,
                                            recovery_images=budget,
                                            **params,
                                            inputs={
                                                "device-state": upstream(
                                                    deploy, "checkpoints/p0.pt"
                                                )
                                            },
                                        )
    campaign = b.finish(selection_inputs)
    print(
        json.dumps(
            {"campaign": str(root / "campaign.json"), "nodes": len(campaign["nodes"])}
        )
    )


def execute(args):
    path = args.campaign.resolve()
    root = path.parent
    campaign = read_receipt(path, "cifar_crossbar.campaign.v1")
    if sha256_file(root / campaign["plan"]) != campaign["plan_sha256"]:
        raise ValueError("Frozen study plan changed.")
    for value in campaign["selection_inputs"].values():
        if sha256_file(Path(value["path"])) != value["sha256"]:
            raise ValueError("Frozen selection receipt changed.")
    results = args.results_root.resolve()
    subprocess.run(
        [
            args.python,
            "-m",
            "ebl",
            "study",
            "prepare",
            "--plan",
            str(root / campaign["plan"]),
            "--results-root",
            str(results),
        ],
        check=True,
    )
    study = results / campaign["study_id"]
    receipt_path = study / "campaign-execution.json"
    receipt = (
        json.loads(receipt_path.read_text())
        if receipt_path.exists()
        else {
            "schema": "cifar_crossbar.execution.v1",
            "campaign_sha256": sha256_file(path),
            "campaign": str(path),
            "study": str(study),
            "completed": {},
            "active": None,
        }
    )
    if receipt["campaign_sha256"] != sha256_file(path):
        raise ValueError("Execution receipt belongs to a different frozen campaign.")
    by_id = {n["id"]: n for n in campaign["nodes"]}
    selected = set(by_id if not args.only else args.only)

    def dependencies(name):
        n = by_id[name]
        deps = n["depends"] + [v["node"] for v in n["inputs"].values() if "node" in v]
        for dep in deps:
            if dep not in selected:
                selected.add(dep)
                dependencies(dep)

    for name in list(selected):
        dependencies(name)
    environment = os.environ.copy()
    environment.update(
        EBL_CIFAR_ROOT=str(args.cifar_root.resolve()),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        EBL_DEFER_CURRENT_SIMULATIONS="1",
    )
    if args.native_python:
        environment["EBL_AIHWKIT_PYTHON"] = args.native_python
    for node in campaign["nodes"]:
        name = node["id"]
        if name not in selected:
            continue
        if sha256_file(root / node["config"]) != node["sha256"]:
            raise ValueError(f"Frozen config changed: {name}")
        if name in receipt["completed"]:
            done = receipt["completed"][name]
            run = Path(done["run_dir"])
            if sha256_file(run / "result.json") != done["result_sha256"]:
                raise ValueError(f"Completed result changed: {name}")
            for artifact in done["artifacts"].values():
                if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
                    raise ValueError(f"Completed artifact changed: {name}")
            continue
        teacher = args.inputs.resolve() / SOURCES[node["dataset"]][0]
        arm = study / "runs" / name
        command = [
            args.python,
            "-m",
            "ebl",
            "train",
            "--config",
            str(root / node["config"]),
            "--output-dir",
            str(arm),
            "--teacher-weights",
            str(teacher),
            "--selection-receipt",
            str(path),
        ]
        for role, value in node["inputs"].items():
            if "node" in value:
                value = receipt["completed"][value["node"]]["artifacts"][
                    value["artifact"]
                ]
            if sha256_file(Path(value["path"])) != value["sha256"]:
                raise ValueError(f"Input digest mismatch: {name}/{role}")
            command += [f"--{role}", value["path"]]
        log = study / "analysis" / f"{name}.log"
        receipt["active"] = {
            "node": name,
            "command": command,
            "log": str(log),
            "started_unix": time.time(),
        }
        atomic_write_json(receipt_path, receipt)
        before = set(arm.iterdir()) if arm.exists() else set()
        with log.open("a") as stream:
            process = subprocess.Popen(
                command, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
            receipt["active"]["pid"] = process.pid
            atomic_write_json(receipt_path, receipt)
            code = process.wait()
        created = set(arm.iterdir()) - before
        if code or len(created) != 1:
            raise RuntimeError(
                f"Native stage {name} failed ({code}); retained log: {log}"
            )
        run = created.pop()
        status = json.loads((run / "status.json").read_text())
        if status["status"] != "complete":
            raise RuntimeError(f"Native stage did not complete semantically: {run}")
        artifacts = {str(p.relative_to(run)): external(p) for p in run.rglob("*.pt")}
        receipt["completed"][name] = {
            "run_dir": str(run),
            "result_sha256": sha256_file(run / "result.json"),
            "config_sha256": node["sha256"],
            "artifacts": artifacts,
        }
        receipt["active"] = None
        atomic_write_json(receipt_path, receipt)
        print(json.dumps({"completed": name, "run_dir": str(run)}), flush=True)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "completed": len(receipt["completed"]),
                "declared": len(campaign["nodes"]),
            }
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--study-id", required=True)
    make.add_argument(
        "--phase",
        choices=("pilot", "hwa", "tune", "screen", "confirm", "reference"),
        required=True,
    )
    make.add_argument(
        "--datasets", nargs="+", choices=tuple(SOURCES), default=list(SOURCES)
    )
    make.add_argument(
        "--backends",
        nargs="+",
        choices=("om", "pcm", "pcm_inference"),
        default=None,
        help=(
            "Default: OM for pulse phases, OM + pcm_inference for HWA, "
            "pcm_inference for reference. pcm explicitly selects the legacy "
            "pulse/refresh experiment, not Gaussian programming-endpoint noise."
        ),
    )
    make.add_argument(
        "--suffix",
        choices=("head", "last_block", "last_two_blocks"),
        default="last_two_blocks",
    )
    make.add_argument("--device", default="cuda:0")
    make.add_argument("--disable-cudnn", action="store_true")
    make.add_argument("--hwa-epochs", type=int, default=30)
    for flag in (
        "hwa-selection",
        "recovery-selection",
        "screen-selection",
        "runtime-selection",
    ):
        make.add_argument("--" + flag, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--results-root", type=Path, default=Path("results"))
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--cifar-root", type=Path, required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--native-python")
    run.add_argument("--only", nargs="+")
    args = p.parse_args()
    (materialize if args.command == "prepare" else execute)(args)


if __name__ == "__main__":
    main()
