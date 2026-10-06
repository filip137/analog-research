"""Native ``python -m ebl train`` entry point for one lifecycle stage.

Every stage is its own run bundle with explicit, hashed inputs:

======== ========================================== =========================
stage    inputs                                     artifact kinds
======== ========================================== =========================
prepare  teacher_weights                            feature_cache
devices  teacher_weights                            device_bundle
hwa      + device_data, device_model                hwa_source
deploy   + device_data, device_model, weights       deployment
onchip   + device_data, device_model, device_state  measurements, onchip_states
======== ========================================== =========================

All stages also record ``mapping``. Upstream artifacts are accepted only when
the lifecycle sections they depend on are unchanged (``verify_provenance``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.cifar_crossbar.model import state_hash, tensor_hash
from experiments.cifar_crossbar.prepare import SOURCES
from experiments.cifar_crossbar.runtime import evaluate, load, save
from workflow import data as data_stage
from workflow import deployment, hwa, onchip, program_verify
from workflow import devices as device_stage
from workflow.lifecycle import EXPERIMENT_ID, StageConfig, provenance, verify_provenance

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROLES = (
    "teacher_weights",
    "weights",
    "resume",
    "device_data",
    "device_model",
    "device_state",
    "selection_receipt",
)
REQUIRED_INPUTS = {
    "prepare": {"teacher_weights"},
    "devices": {"teacher_weights"},
    "hwa": {"teacher_weights", "device_data", "device_model"},
    "deploy": {"teacher_weights", "device_data", "device_model", "weights"},
    "onchip": {"teacher_weights", "device_data", "device_model", "device_state"},
}


class Stage:
    """Everything one stage function needs; built once per run."""

    def __init__(self, request, store, device):
        self.request, self.store, self.device = request, store, device
        self.config: StageConfig = request.spec
        self.lifecycle = self.config.lifecycle
        self.selection = self.config.stage
        self.technology = self.lifecycle.devices.technology
        self.teacher, teacher_sha256 = deployment.load_teacher(
            request.teacher_weights, self.lifecycle.network, device
        )
        self.network = deployment.build_network(self.teacher, self.lifecycle)
        self.mapping = deployment.mapping_receipt(self.network, self.lifecycle)
        self.context = {
            "teacher_sha256": teacher_sha256,
            "mapping_sha256": deployment.mapping_digest(self.mapping),
        }

    @property
    def run_dir(self) -> Path:
        return self.store.run_dir

    def checked(self, role: str, kind: str, *, hwa_arm=None) -> dict:
        """Load an upstream bundle and verify its lifecycle and mapping identity."""

        bundle = load(getattr(self.request, role))
        verify_provenance(bundle.get("provenance"), self.lifecycle, kind, hwa_arm=hwa_arm)
        if bundle.get("context") != self.context:
            raise ValueError(f"Expected the {kind} artifact to share the teacher and mapping.")
        return bundle

    def cache(self) -> dict:
        cache = self.checked("device_data", "feature_cache")
        data_stage.check_cache(cache, self.lifecycle)
        return data_stage.to_device(cache, self.device)

    def device_bundle(self) -> dict:
        bundle = self.checked("device_model", "device_bundle")
        if bundle["technology"] != self.technology:
            raise ValueError("Expected a device bundle of the lifecycle technology.")
        return bundle


def prepare(stage: Stage):
    lifecycle = stage.lifecycle
    cache = data_stage.build_cache(stage.teacher, stage.network, lifecycle, stage.device)
    split = data_stage.evaluation_split(lifecycle)
    digital = evaluate(stage.network, cache[split], stage.device)
    teacher_accuracy = float(
        (cache[split]["teacher_logits"].argmax(1) == cache[split]["labels"]).float().mean() * 100
    )
    published = SOURCES[lifecycle.network.dataset][2]
    if split == "test" and not lifecycle.data.smoke and abs(teacher_accuracy - published) > 0.25:
        raise RuntimeError("Digital checkpoint accuracy does not reproduce the published source within 0.25 pp.")
    path = stage.run_dir / "checkpoints/cache.pt"
    save(path, {"provenance": provenance(lifecycle, "feature_cache"), "context": stage.context, **cache})
    cohorts = {name: cache[name]["cohort"] for name in data_stage.FEATURE_SPLITS if name in cache}
    cohorts["hwa"] = cache["hwa_cohort"]
    result = {
        "digital": digital,
        "teacher_accuracy_percent": teacher_accuracy,
        "published_accuracy_percent": published,
        "cohorts": cohorts,
    }
    return result, [("feature_cache", path)]


def devices(stage: Stage):
    lifecycle = stage.lifecycle
    result = {"technology": stage.technology}
    bundle = {
        "provenance": provenance(lifecycle, "device_bundle"),
        "context": stage.context,
        "technology": stage.technology,
        "populations": None,
        "kernels": None,
    }
    path = stage.run_dir / "checkpoints/devices.pt"
    if stage.technology == "om":
        populations = device_stage.sample_populations(
            lifecycle, stage.network.layout, stage.run_dir / "scratch"
        )
        (stage.run_dir / "scratch").rmdir()
        kernels = program_verify.characterize_endpoints(populations, lifecycle, stage.device)
        bundle.update(populations=populations, kernels=kernels)
        result["kernel_gates"] = {
            kind: {k: gate[k] for k in ("adequate", "max_mean_error")} for kind, gate in kernels["gates"].items()
        }
        result["kernel_adequate"] = kernels["adequate"]
        save(path, bundle)
        if not kernels["adequate"]:
            raise RuntimeError("The held-out OM endpoint characterization failed its adequacy gate.")
    else:
        save(path, bundle)
    result["populations"] = device_stage.population_receipt(bundle["populations"])
    return result, [("device_bundle", path)]


def train_hwa(stage: Stage):
    lifecycle = stage.lifecycle
    arm = lifecycle.hwa.arm(stage.selection.hwa_arm)
    cache = stage.cache()
    bundle = stage.device_bundle()
    outcome = hwa.train_source(
        lifecycle=lifecycle,
        arm=arm,
        network=stage.network,
        teacher=stage.teacher,
        cache=cache,
        device_bundle=bundle,
        store=stage.store,
        device=stage.device,
    )
    stage.network.load_state_dict(outcome["model"])
    clean = evaluate(stage.network, cache["development"], stage.device)
    path = stage.run_dir / "checkpoints/source.pt"
    save(
        path,
        {
            "provenance": provenance(lifecycle, "hwa_source", hwa_arm=arm.arm_id),
            "context": stage.context,
            "arm": arm.to_dict(),
            "model": outcome["model"],
            "fit": outcome["fit"],
        },
    )
    result = {
        "arm": arm.to_dict(),
        "fit": outcome["fit"],
        "clean_development": clean,
        "source_model_sha256": state_hash(outcome["model"]),
        "test_used_for_selection": False,
    }
    return result, [("hwa_source", path)]


def deploy(stage: Stage):
    lifecycle, network, device = stage.lifecycle, stage.network, stage.device
    arm, seed = stage.selection.hwa_arm, stage.selection.array_seed
    cache = stage.cache()
    bundle = stage.device_bundle()
    source = stage.checked("weights", "hwa_source", hwa_arm=arm)
    if source["fit"]["status"] != "complete":
        raise ValueError(
            f"Expected a completed HWA source; arm {arm!r} is {source['fit']['status']!r}."
        )
    network.load_state_dict(source["model"])
    target = deployment.targets(network)
    split = data_stage.evaluation_split(lifecycle)
    clean = {"development": evaluate(network, cache["development"], device)}
    if split == "test":
        clean["test"] = evaluate(network, cache["test"], device)
    cases, summary = {}, {}
    for case in lifecycle.defects.cases:
        array = program_verify.fresh(lifecycle, network.layout, seed, case, bundle["populations"], device)
        identity = device_stage.array_identity(array, stage.technology)
        programming = program_verify.program(array, target, lifecycle)
        initial = onchip.evaluations(network, array, cache, lifecycle, device)
        cases[case.label] = {
            "case": case.to_dict(),
            "identity": identity,
            "programming": programming,
            "initial": initial,
            "array": device_stage.array_state(array, stage.technology),
        }
        summary[case.label] = {
            "identity": identity,
            "programming": programming,
            "initial": initial,
            "deployment_drop_pp": clean[split]["accuracy_percent"] - initial[split]["accuracy_percent"],
        }
        print(json.dumps({"case": case.label, "teacher_kl": initial[split]["teacher_kl"]}), flush=True)
    path = stage.run_dir / "checkpoints/deployment.pt"
    save(
        path,
        {
            "provenance": provenance(lifecycle, "deployment", hwa_arm=arm),
            "context": stage.context,
            "array_seed": seed,
            "endpoint_seed": lifecycle.devices.endpoint_seed(seed),
            "model": source["model"],
            "target": target,
            "source_sha256": sha256_file(stage.request.weights),
            "hwa_fit": {k: v for k, v in source["fit"].items() if k != "history"},
            "clean": clean,
            "cases": cases,
        },
    )
    result = {
        "array_seed": seed,
        "endpoint_seed": lifecycle.devices.endpoint_seed(seed),
        "clean": clean,
        "cases": summary,
        "hwa_convergence_review_required": bool(source["fit"].get("convergence_review_required", False)),
    }
    return result, [("deployment", path)]


def train_onchip(stage: Stage):
    lifecycle, network, device = stage.lifecycle, stage.network, stage.device
    arm_id, seed = stage.selection.hwa_arm, stage.selection.array_seed
    split = data_stage.evaluation_split(lifecycle)
    cache = stage.cache()
    bundle = stage.device_bundle()
    p0 = stage.checked("device_state", "deployment", hwa_arm=arm_id)
    if p0["array_seed"] != seed or set(p0["cases"]) != {case.label for case in lifecycle.defects.cases}:
        raise ValueError("Expected the deployment of this assignment and every lifecycle defect case.")
    p0_sha256 = sha256_file(stage.request.device_state)
    network.load_state_dict(p0["model"])
    target = p0["target"].to(device)
    if not torch.equal(target, network.q.detach()):
        raise ValueError("Expected deployment targets mapped from the deployed source model.")
    rows, index, artifacts = [], {}, []
    for case in lifecycle.defects.cases:
        deployed = p0["cases"][case.label]
        index[case.label] = {}
        for arm in lifecycle.onchip.arms:
            started = time.monotonic()
            network.load_state_dict(p0["model"])
            array = program_verify.fresh(lifecycle, network.layout, seed, case, bundle["populations"], device)
            device_stage.load_array_state(array, deployed["array"], stage.technology)
            if tensor_hash(array.read()) != deployed["programming"]["apparent_sha256"]:
                raise RuntimeError("Expected the exact deployed apparent state before on-chip training.")
            outcome = onchip.train_arm(
                lifecycle=lifecycle,
                arm=arm,
                network=network,
                array=array,
                target=target,
                cache=cache,
                store=stage.store,
                label=case.label,
                device=device,
            )
            state = outcome.pop("state")
            final = onchip.evaluations(network, array, cache, lifecycle, device)
            relative = Path("checkpoints/onchip") / case.label / f"{arm.arm_id}.pt"
            save(stage.run_dir / relative, {"p0_sha256": p0_sha256, "case": case.to_dict(), "arm": arm.to_dict(), **state})
            # Exact replay: the saved state alone must reproduce the final metrics.
            replay = program_verify.fresh(lifecycle, network.layout, seed, case, bundle["populations"], device)
            onchip.restore(network, replay, p0["model"], load(stage.run_dir / relative), stage.technology)
            if onchip.evaluations(network, replay, cache, lifecycle, device) != final:
                raise RuntimeError("Saved final on-chip state did not replay its metrics exactly.")
            if outcome["curve"] and outcome["curve"][-1][split] != final[split]:
                raise RuntimeError("Final metrics differ from the last recorded epoch.")
            state_sha256 = sha256_file(stage.run_dir / relative)
            index[case.label][arm.arm_id] = {"path": relative.as_posix(), "sha256": state_sha256}
            artifacts.append((f"onchip_state.{case.label}.{arm.arm_id}", stage.run_dir / relative))
            rows.append(
                {
                    "lifecycle_id": lifecycle.lifecycle_id,
                    "technology": stage.technology,
                    "hwa_arm": arm_id,
                    "array_seed": seed,
                    "endpoint_seed": p0["endpoint_seed"],
                    "case": case.label,
                    "defect": case.to_dict(),
                    "onchip_arm": arm.arm_id,
                    "weights": arm.weights,
                    "calibration": arm.calibration,
                    "identity": deployed["identity"],
                    "programming": deployed["programming"],
                    "p0_sha256": p0_sha256,
                    "clean": p0["clean"],
                    "initial": deployed["initial"],
                    "final": final,
                    "curve": outcome["curve"],
                    "cost": outcome["curve"][-1]["cost"] if outcome["curve"] else {},
                    "image_presentations": outcome["curve"][-1]["image_presentations"] if outcome["curve"] else 0,
                    "initial_apparent_sha256": outcome["initial_apparent_sha256"],
                    "final_apparent_sha256": outcome["final_apparent_sha256"],
                    "state_sha256": state_sha256,
                    "replay_exact": True,
                    "hwa_convergence_review_required": bool(p0["hwa_fit"].get("convergence_review_required", False)),
                    "seconds": time.monotonic() - started,
                }
            )
            atomic_write_json(stage.run_dir / "measurements.json", rows)
    atomic_write_json(stage.run_dir / "onchip_states.json", index)
    result = {
        "array_seed": seed,
        "rows": [
            {
                "case": row["case"],
                "onchip_arm": row["onchip_arm"],
                "initial": {k: row["initial"][split][k] for k in ("teacher_kl", "accuracy_percent")},
                "final": {k: row["final"][split][k] for k in ("teacher_kl", "accuracy_percent")},
                "cost": row["cost"],
            }
            for row in rows
        ],
        "exact_replays": len(rows),
        "selection": "fixed schedule; final epoch; no test-set checkpoint selection",
    }
    artifacts = [
        ("measurements", stage.run_dir / "measurements.json"),
        ("onchip_states", stage.run_dir / "onchip_states.json"),
    ] + artifacts
    return result, artifacts


STAGES = {
    "prepare": prepare,
    "devices": devices,
    "hwa": train_hwa,
    "deploy": deploy,
    "onchip": train_onchip,
}


def configure_runtime(execution, seed) -> dict:
    """Deterministic FP32 execution; this changes execution, not physics."""

    torch.set_num_threads(execution.cpu_threads)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = not execution.disable_cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "tf32": False,
        "deterministic": True,
        "torch_version": str(torch.__version__),
        "device": execution.device,
        "cpu_threads": execution.cpu_threads,
    }


def _inputs(request, kind) -> list[dict]:
    provided = {role for role in INPUT_ROLES if getattr(request, role, None) is not None}
    required = REQUIRED_INPUTS[kind]
    if provided != required:
        raise ValueError(
            f"Expected exactly the {kind} stage inputs {sorted(required)}. "
            f"Provided value: {sorted(provided)}."
        )
    inputs = [
        {"role": role, "path": str(getattr(request, role).resolve()), "sha256": sha256_file(getattr(request, role))}
        for role in INPUT_ROLES
        if role in provided
    ]
    receipt_path = os.environ.get("EBL_SOURCE_RECEIPT")
    if receipt_path:
        receipt = json.loads(Path(receipt_path).read_text())
        for name, digest in receipt["files"].items():
            source = (ROOT / name).resolve()
            if not source.is_relative_to(ROOT) or sha256_file(source) != digest:
                raise ValueError(f"Staged source snapshot mismatch: {name}")
        inputs.append(
            {"role": "staged_source_snapshot", "path": receipt_path, "sha256": sha256_file(Path(receipt_path))}
        )
    return inputs


def run_train(request) -> int:
    config = request.spec
    if not isinstance(config, StageConfig):
        raise TypeError("Expected a crossbar lifecycle stage config.")
    lifecycle, selection = config.lifecycle, config.stage
    inputs = _inputs(request, selection.kind)
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=EXPERIMENT_ID,
        resolved_config=config.to_dict(),
        command=request.command,
        repo_root=ROOT,
        input_artifacts=inputs,
        resume_capability="unsupported",
    )
    started = time.monotonic()
    try:
        seed = lifecycle.onchip.seed if selection.kind == "onchip" else lifecycle.hwa.seed
        runtime_math = configure_runtime(config.execution, seed)
        device = torch.device(config.execution.device)
        stage = Stage(request, store, device)
        atomic_write_json(store.run_dir / "mapping.json", stage.mapping)
        prefix = stage.network.prefix_hash()
        result, artifacts = STAGES[selection.kind](stage)
        if stage.network.prefix_hash() != prefix:
            raise RuntimeError("The frozen digital prefix changed.")
        result.update(
            stage=selection.to_dict(),
            stage_id=selection.stage_id,
            lifecycle_id=lifecycle.lifecycle_id,
            lifecycle_sha256=lifecycle.digest(),
            technology=stage.technology,
            dataset=lifecycle.network.dataset,
            analog_convolutions=lifecycle.network.analog_convolutions,
            evidence_class=lifecycle.question.evidence_class,
            smoke_only=lifecycle.data.smoke,
            prefix_sha256=prefix,
            **stage.context,
            runtime_math=runtime_math,
            elapsed_seconds=time.monotonic() - started,
            peak_gpu_memory_bytes=(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        )
        store.append_metric({"stage": selection.kind, "terminal": True, "elapsed_seconds": result["elapsed_seconds"]})
        records = [store.artifact_record(store.run_dir / "mapping.json", kind="mapping")]
        records += [store.artifact_record(path, kind=kind) for kind, path in artifacts]
        store.complete(metrics=result, artifacts=records)
        print(json.dumps({"run_dir": str(store.run_dir), "stage": selection.stage_id, "status": "complete"}), flush=True)
        return 0
    except BaseException as exc:
        store.fail(exc)
        raise
