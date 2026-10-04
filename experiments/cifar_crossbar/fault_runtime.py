"""Native EBL execution of the Gaussian-PCM permanent-failure screen."""

from dataclasses import asdict
from pathlib import Path
import json
import os
import time

import torch
from torch.nn import functional as F

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.cifar_crossbar.data import cache_features, get_dataset, split_indices
from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.fault_config import FaultSpec
from experiments.cifar_crossbar.model import CrossbarSuffix, tensor_hash
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults
from experiments.cifar_crossbar.runtime import evaluate, load, save, source_model

ROOT = Path(__file__).resolve().parents[2]


def cache_context(spec, teacher_hash, student):
    return {
        "dataset": spec.dataset,
        "data_seed": spec.data_seed,
        "recovery_images": spec.recovery_images,
        "teacher_sha256": teacher_hash,
        "mapping": student.mapping_receipt(),
    }


def build_cache(teacher, student, spec, device):
    training = get_dataset(spec.dataset, True)
    indices, development = split_indices(training.targets, seed=spec.data_seed)
    selected = indices[: spec.recovery_images]
    if set(selected) & set(development):
        raise RuntimeError("Recovery and development images overlap.")
    test = get_dataset(spec.dataset, False)
    return {
        "training": cache_features(teacher, student, training, selected, device=device),
        "test": cache_features(teacher, student, test, range(len(test)), device=device),
    }


def limited_cache(cache, maximum):
    if not maximum:
        return cache
    return {
        k: v[:maximum] if isinstance(v, torch.Tensor) else v for k, v in cache.items()
    }


def adapt(student, array, target, train, test, spec, method, store, case_id, device):
    calibrate = method in ("calibration", "rewrite", "onchip_calibration")
    learn_weights = method in ("onchip_weights", "onchip_calibration")
    student.enable_calibration(calibrate)
    student.q.requires_grad_(False)
    master = torch.nn.Parameter(target.clone(), requires_grad=learn_weights)
    groups = []
    if learn_weights:
        groups.append({"params": [master], "lr": spec.learning_rate})
    if calibrate:
        groups.append(
            {"params": student.calibration_parameters(), "lr": spec.calibration_lr}
        )
    optimizer = torch.optim.Adam(groups) if groups else None
    baseline_cost = array.cost()
    initial_hash = tensor_hash(array.read())
    if method != "none":
        if spec.recovery_model == "ideal_update_upper_bound":
            array.noise_scale = 0.0
        order = torch.randperm(
            len(train["labels"]),
            generator=torch.Generator().manual_seed(spec.data_seed + 1),
        )
        student.train()
        for batch, begin in enumerate(range(0, len(order), spec.batch_size)):
            idx = order[begin : begin + spec.batch_size]
            x = train["features"][idx].to(device)
            t = train["teacher_logits"][idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            observed = array.read()
            # Controller uses apparent weights and digital gradients, never a fault mask.
            q = master + (observed - master).detach() if learn_weights else observed
            logits = student.forward_features(x, q)
            loss = F.kl_div(logits.log_softmax(1), t.softmax(1), reduction="batchmean")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite teacher KL during PCM recovery.")
            loss.backward()
            optimizer.step()
            if learn_weights and spec.learning_rate > 0:
                with torch.no_grad():
                    master.clamp_(-1, 1)
                array.program(master.detach())
            elif method == "rewrite":
                # Equal rewrite schedule, frozen targets: separate learning from rewriting.
                array.program(target)
            if batch % 20 == 0:
                store.append_metric(
                    {
                        "case": case_id,
                        "method": method,
                        "epoch": 1,
                        "batch": batch,
                        "teacher_kl_train": float(loss.detach()),
                        "array_reprogram_calls": array.program_calls,
                    }
                )
    array.verify_permanence()
    metrics = evaluate(student, test, device, q=array.read())
    final_cost = array.cost()
    return {
        "method": method,
        "initial_apparent_sha256": initial_hash,
        "final": metrics,
        "cost": {k: final_cost[k] - baseline_cost[k] for k in baseline_cost},
        "image_presentations": 0 if method == "none" else len(train["labels"]),
        "model": clone_cpu(student.state_dict()),
        "master_target": master.detach().cpu(),
        "array": array.state_dict(),
    }


def run_screen(request, spec, teacher, student, teacher_hash, store, device):
    if request.device_data is None:
        raise ValueError("Expected --device-data naming the declared feature cache.")
    cached = load(request.device_data)
    context = cache_context(spec, teacher_hash, student)
    if cached["context"] != context:
        raise ValueError(
            "Expected cache matching teacher, mapping, dataset and recovery cohort."
        )
    if (
        len(cached["training"]["labels"]) != spec.recovery_images
        or len(cached["test"]["labels"]) != 10000
    ):
        raise ValueError(
            "Expected full recovery cohort and 10000-image official test cache."
        )
    train = limited_cache(cached["training"], spec.max_examples)
    test = limited_cache(cached["test"], spec.max_examples)
    pristine = clone_cpu(student.state_dict())
    target = student.q.detach().clone()
    prefix = student.prefix_hash()
    clean = evaluate(student, test, device)
    scenarios = [("open", 0)] + [
        (kind, ppm) for kind in spec.fault_kinds for ppm in spec.fault_rates_ppm if ppm
    ]
    measurements = []
    for kind, ppm in scenarios:
        case_id = "nominal" if ppm == 0 else f"{kind}_{ppm}ppm"
        student.load_state_dict(pristine)
        faults = PermanentFaults.sample(
            target.numel(), kind, ppm / 1e6, spec.array_seed, device
        )
        array = GaussianEndpointArray(
            target, faults, spec.endpoint_seed, spec.noise_scale
        )
        deployed = evaluate(student, test, device, q=array.read())
        p0 = array.state_dict()
        path = store.run_dir / "checkpoints" / case_id / "p0.pt"
        save(path, {"context": context, "model": pristine, "array": p0})
        p0_hash = sha256_file(path)
        identity = faults.receipt()
        for method in spec.methods:
            started = time.monotonic()
            student.load_state_dict(pristine)
            array.load_state_dict(p0)
            outcome = adapt(
                student,
                array,
                target,
                train,
                test,
                spec,
                method,
                store,
                case_id,
                device,
            )
            if student.prefix_hash() != prefix:
                raise RuntimeError("Frozen prefix changed during recovery.")
            if identity != faults.receipt():
                raise RuntimeError(
                    "Permanent fault identities changed during recovery."
                )
            state = {k: outcome.pop(k) for k in ("model", "master_target", "array")}
            if method != "none":
                save(
                    path.parent / f"{method}.pt",
                    {"context": context, "p0_sha256": p0_hash, **state},
                )
            item = {
                "case": case_id,
                "fault_kind": "none" if ppm == 0 else kind,
                "fault_rate_ppm": ppm,
                "array_seed": spec.array_seed,
                "endpoint_seed": spec.endpoint_seed,
                "faults": identity,
                "p0_sha256": p0_hash,
                "initial": deployed,
                **outcome,
                "teacher_kl_reduction": deployed["teacher_kl"]
                - outcome["final"]["teacher_kl"],
                "elapsed_seconds": time.monotonic() - started,
            }
            measurements.append(item)
            store.append_metric({"terminal_control": True, **item})
            atomic_write_json(store.run_dir / "measurements.json", measurements)
            print(
                json.dumps(
                    {
                        "case": case_id,
                        "method": method,
                        "teacher_kl": item["final"]["teacher_kl"],
                        "seconds": item["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
    return {
        "stage": "screen",
        "clean": clean,
        "measurements": measurements,
        "evaluation_images": len(test["labels"]),
        "recovery_images": len(train["labels"]),
        "cases": len(scenarios),
        "completed_controls": len(measurements),
        "prefix_sha256": prefix,
        "training_cohort": cached["training"]["cohort"],
        "test_cohort": cached["test"]["cohort"],
    }


def run_train(request):
    spec = request.spec
    if not isinstance(spec, FaultSpec) or request.teacher_weights is None:
        raise ValueError("Expected PCM fault settings and explicit --teacher-weights.")
    if any(
        getattr(request, key, None) is not None
        for key in ("weights", "base_weights", "resume", "device_model", "device_state")
    ):
        raise ValueError(
            "Expected pretrained teacher plus cache only; this exploratory screen has no HWA or resume input."
        )
    inputs = [
        {
            "role": name,
            "path": str(getattr(request, name).resolve()),
            "sha256": sha256_file(getattr(request, name)),
        }
        for name in ("teacher_weights", "device_data", "selection_receipt")
        if getattr(request, name, None) is not None
    ]
    source_receipt = os.environ.get("EBL_SOURCE_RECEIPT")
    if source_receipt:
        receipt = json.loads(Path(source_receipt).read_text())
        for name, digest in receipt["files"].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError(f"Source snapshot changed: {name}")
        inputs.append(
            {
                "role": "source_snapshot",
                "path": source_receipt,
                "sha256": sha256_file(Path(source_receipt)),
            }
        )
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=asdict(spec),
        command=request.command,
        repo_root=ROOT,
        input_artifacts=inputs,
        resume_capability="unsupported",
    )
    started = time.monotonic()
    try:
        torch.set_num_threads(spec.cpu_threads)
        torch.manual_seed(spec.seed)
        torch.backends.cudnn.enabled = not spec.disable_cudnn
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(spec.device)
        teacher, teacher_hash = source_model(request.teacher_weights, spec, device)
        student = CrossbarSuffix(teacher, spec.suffix, spec.tile_size)
        atomic_write_json(store.run_dir / "mapping.json", student.mapping_receipt())
        if spec.stage == "cache":
            cache = build_cache(teacher, student, spec, device)
            clean = evaluate(student, cache["test"], device)
            from experiments.cifar_crossbar.prepare import SOURCES

            if (
                clean["examples"] != 10000
                or abs(clean["accuracy_percent"] - SOURCES[spec.dataset][2]) > 0.25
            ):
                raise RuntimeError("Digital CIFAR checkpoint audit failed.")
            save(
                store.run_dir / "checkpoints/cache.pt",
                {"context": cache_context(spec, teacher_hash, student), **cache},
            )
            result = {
                "stage": "cache",
                "clean": clean,
                "training_cohort": cache["training"]["cohort"],
                "test_cohort": cache["test"]["cohort"],
            }
        else:
            result = run_screen(
                request, spec, teacher, student, teacher_hash, store, device
            )
        result.update(
            {
                "dataset": spec.dataset,
                "source": "pretrained_digital_no_hwa",
                "backend": "pcm_gaussian_endpoint",
                "recovery_model": spec.recovery_model,
                "evidence_class": "exploratory_model_based",
                "smoke_only": bool(spec.max_examples),
                "teacher_kl_definition": "mean KL(p_teacher || p_student), nats, temperature 1",
                "retention": "immediate post-programming; no drift or extra read noise",
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        store.append_metric(
            {
                "terminal": True,
                "stage": spec.stage,
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        artifacts = [
            store.artifact_record(
                p, kind="checkpoint" if p.suffix == ".pt" else "mapping"
            )
            for p in store.run_dir.rglob("*")
            if p.is_file()
            and (p.suffix == ".pt" or p.name in ("mapping.json", "measurements.json"))
        ]
        store.complete(metrics=result, artifacts=artifacts)
        print(
            json.dumps({"run_dir": str(store.run_dir), "status": "complete"}),
            flush=True,
        )
        return 0
    except BaseException as exc:
        store.fail(exc)
        raise
