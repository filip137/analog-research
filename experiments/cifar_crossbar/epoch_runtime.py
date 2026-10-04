"""Many-pass recovery with persistent Adam state and development-only selection.

The original one-pass experiments remain unchanged. Numerical execution enters
through EBL; forward weights and the programming law match that experiment.
"""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.epoch_config import EpochSpec
from experiments.cifar_crossbar.fault_runtime import cache_context, limited_cache
from experiments.cifar_crossbar.hwa_fault_runtime import case_name, checked_sources, source_cases
from experiments.cifar_crossbar.model import CrossbarSuffix, tensor_hash
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults
from experiments.cifar_crossbar.runtime import evaluate, load, save, source_model

ROOT = Path(__file__).resolve().parents[2]
LEARNING_METHODS = ("calibration", "onchip_weights", "onchip_calibration")


def lr_factor(spec, schedule, epoch):
    if schedule not in spec.schedules:
        raise ValueError("Unknown recovery schedule.")
    return 1.0 if schedule == "constant" else spec.decay_factor ** sum(epoch > e for e in spec.decay_after)


def compact_state(student, array, master, optimizer, epoch):
    # Fixed source tensors and failure identities are stored once in paired P0.
    return clone_cpu({
        "epoch": epoch,
        "calibration": {n: p for n, p in student.named_parameters() if n != "q" and p.requires_grad},
        "master_target": master,
        "conductance": array.g,
        "rng": array.generator.get_state(),
        "program_calls": array.program_calls,
        "optimizer": optimizer.state_dict(),
    })


def restore_compact(student, array, p0, state):
    """Read-only checkpoint replay from exact source/P0 plus changing tensors."""
    student.load_state_dict(p0["model"])
    with torch.no_grad():
        parameters = dict(student.named_parameters())
        for name, value in state["calibration"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    value = clone_cpu(p0["array"])
    value.update(g=state["conductance"], target=state["master_target"],
                 rng=state["rng"], program_calls=state["program_calls"])
    array.load_state_dict(value)
    array.verify_permanence()


def trajectory(student, array, target, train, development, spec, method, schedule,
               store, case_id, device, *, test=None, checkpoint=None):
    if method not in spec.methods or method == "none":
        raise ValueError("Trajectory requires an adapting paired control.")
    calibrate = method in ("calibration", "rewrite", "onchip_calibration")
    learn_weights = method in ("onchip_weights", "onchip_calibration")
    student.enable_calibration(calibrate)
    student.q.requires_grad_(False)
    master = torch.nn.Parameter(target.clone(), requires_grad=learn_weights)
    groups = []
    if learn_weights:
        groups.append({"params": [master], "lr": spec.learning_rate})
    if calibrate:
        groups.append({"params": student.calibration_parameters(), "lr": spec.calibration_lr})
    optimizer = torch.optim.Adam(groups)
    base_rates = [g["lr"] for g in optimizer.param_groups]
    baseline_cost = array.cost()
    initial_hash = tensor_hash(array.read())
    prefix = student.prefix_hash()
    identity = array.faults.receipt()
    curves = []
    initial_development = evaluate(student, development, device, q=array.read())
    initial_test = evaluate(student, test, device, q=array.read()) if test is not None else None
    snapshots = {}
    for epoch in range(1, spec.epochs + 1):
        factor = lr_factor(spec, schedule, epoch)
        for group, rate in zip(optimizer.param_groups, base_rates):
            group["lr"] = rate * factor
        # First epoch reproduces the old adapter. Subsequent epochs reshuffle.
        order = torch.randperm(len(train["labels"]), generator=torch.Generator().manual_seed(spec.data_seed + epoch))
        student.train()
        loss_sum = 0.0
        for begin in range(0, len(order), spec.batch_size):
            idx = order[begin:begin + spec.batch_size]
            x = train["features"][idx].to(device)
            t = train["teacher_logits"][idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            observed = array.read()
            q = master + (observed - master).detach() if learn_weights else observed
            logits = student.forward_features(x, q)
            loss = F.kl_div(logits.log_softmax(1), t.softmax(1), reduction="batchmean")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite recovery teacher KL.")
            loss.backward()
            optimizer.step()
            if learn_weights and spec.learning_rate > 0:
                with torch.no_grad():
                    master.clamp_(-1, 1)
                array.program(master.detach())
            elif method == "rewrite":
                array.program(target)
            loss_sum += float(loss.detach()) * len(idx)
        array.verify_permanence()
        if student.prefix_hash() != prefix or array.faults.receipt() != identity:
            raise RuntimeError("Frozen prefix or permanent fault identities changed.")
        value = {
            "epoch": epoch, "teacher_kl_train_mean": loss_sum / len(order),
            "development": evaluate(student, development, device, q=array.read()),
            "learning_rate_factor": factor,
            "image_presentations": epoch * len(order),
            "cost": {k: v - baseline_cost[k] for k, v in array.cost().items()},
        }
        if epoch in spec.milestones:
            if test is not None:
                value["test"] = evaluate(student, test, device, q=array.read())
            if checkpoint is not None:
                snapshots[str(epoch)] = compact_state(student, array, master, optimizer, epoch)
                save(checkpoint, {"p0_sha256": sha256_file(checkpoint.parent / "p0.pt"), "milestones": snapshots})
        curves.append(value)
        store.append_metric({"case": case_id, "method": method, "schedule": schedule, **value})
        print(json.dumps({"case": case_id, "method": method, "schedule": schedule,
                          "epoch": epoch, "development_kl": value["development"]["teacher_kl"]}), flush=True)
    return {
        "method": method, "schedule": schedule,
        "initial_apparent_sha256": initial_hash,
        "initial_development": initial_development, "initial_test": initial_test,
        "curve": curves,
    }


def select_policies(rows, spec):
    selected = {}
    for method in LEARNING_METHODS:
        scores = {}
        for schedule in spec.schedules:
            candidates = [r for r in rows if r["method"] == method and r["schedule"] == schedule]
            if not candidates:
                raise ValueError("Missing schedule-selection trajectory.")
            scores[schedule] = sum(r["curve"][-1]["development"]["teacher_kl"] / max(r["initial_development"]["teacher_kl"], 1e-12) for r in candidates) / len(candidates)
        schedule = min(scores, key=lambda s: (scores[s], s))
        candidates = [r for r in rows if r["method"] == method and r["schedule"] == schedule]
        epoch_scores = {e: sum(r["curve"][e-1]["development"]["teacher_kl"] / max(r["initial_development"]["teacher_kl"], 1e-12) for r in candidates) / len(candidates) for e in spec.milestones}
        selected[method] = {"schedule": schedule, "schedule_scores": scores,
                            "selected_epoch": min(epoch_scores, key=lambda e: (epoch_scores[e], e)),
                            "epoch_scores": epoch_scores}
    # Rewriting is a calibration control, with the same calibration schedule.
    selected["rewrite"] = dict(selected["calibration"])
    return selected


def tune(student, cache, bundle, spec, store, device):
    tasks = [("standard_hwa", "open", 0)] + [(f"cdt_{k}", k, 10000) for k in spec.fault_kinds]
    rows = []
    for name, kind, ppm in tasks:
        source = bundle["sources"][name]["model"]
        for draw in range(spec.tuning_arrays):
            student.load_state_dict(source)
            target = student.q.detach().clone()
            faults = PermanentFaults.sample(target.numel(), kind, ppm / 1e6, 121001 + draw, device)
            array = GaussianEndpointArray(target, faults, 131001 + draw)
            p0 = array.state_dict()
            for method in LEARNING_METHODS:
                for schedule in spec.schedules:
                    student.load_state_dict(source)
                    array.load_state_dict(p0)
                    row = trajectory(student, array, target, cache["training"], cache["development"],
                                     spec, method, schedule, store, f"tune/{name}/{kind}/{draw}", device)
                    rows.append({"source": name, "case": case_name(kind, ppm), "array_seed": 121001 + draw, **row})
                    atomic_write_json(store.run_dir / "measurements.json", rows)
    return {"measurements": rows, "selected": select_policies(rows, spec), "completed_trajectories": len(rows)}


def screen(student, cache, bundle, policies, spec, store, device, context):
    rows = []
    prefix = student.prefix_hash()
    for name, source in bundle["sources"].items():
        state = source["model"]
        student.load_state_dict(state)
        target = student.q.detach().clone()
        clean = evaluate(student, cache["test"], device)
        for kind, ppm in source_cases(spec, name):
            case = case_name(kind, ppm)
            student.load_state_dict(state)
            faults = PermanentFaults.sample(target.numel(), kind, ppm / 1e6, spec.array_seed, device)
            array = GaussianEndpointArray(target, faults, spec.endpoint_seed)
            p0 = {"context": context, "model": state, "array": array.state_dict()}
            directory = store.run_dir / "checkpoints" / name / case
            save(directory / "p0.pt", p0)
            p0_hash = sha256_file(directory / "p0.pt")
            initial = evaluate(student, cache["test"], device, q=array.read())
            for method in spec.methods:
                student.load_state_dict(state)
                array.load_state_dict(p0["array"])
                base = {"source": name, "case": case, "fault_kind": kind if ppm else "none",
                        "fault_rate_ppm": ppm, "array_seed": spec.array_seed,
                        "endpoint_seed": spec.endpoint_seed, "faults": faults.receipt(),
                        "p0_sha256": p0_hash, "clean": clean, "initial_test": initial}
                if method == "none":
                    result = {"method": method, "initial_apparent_sha256": tensor_hash(array.read()), "curve": []}
                else:
                    result = trajectory(student, array, target, cache["training"], cache["development"],
                                        spec, method, policies[method]["schedule"], store,
                                        f"{name}/{case}", device, test=cache["test"],
                                        checkpoint=directory / f"{method}.pt")
                    if result["initial_test"] != initial:
                        raise RuntimeError("Paired initial evaluation changed.")
                if student.prefix_hash() != prefix:
                    raise RuntimeError("Frozen prefix changed.")
                rows.append({**base, **result})
                store.append_metric({"terminal_control": True, "source": name, "case": case, "method": method})
                atomic_write_json(store.run_dir / "measurements.json", rows)
    return {"measurements": rows, "completed_controls": len(rows), "prefix_sha256": prefix}


def run_train(request):
    spec = request.spec
    if not isinstance(spec, EpochSpec) or any(getattr(request, k, None) is None for k in ("teacher_weights", "weights", "device_data")):
        raise ValueError("Expected explicit teacher, cached cohorts and source/selection bundle.")
    if any(getattr(request, k, None) is not None for k in ("resume", "base_weights", "device_model", "device_state")):
        raise ValueError("Unexpected epoch-recovery input.")
    inputs = [{"role": k, "path": str(getattr(request, k).resolve()), "sha256": sha256_file(getattr(request, k))}
              for k in ("teacher_weights", "weights", "device_data", "selection_receipt") if getattr(request, k, None) is not None]
    receipt_path = os.environ.get("EBL_SOURCE_RECEIPT")
    if receipt_path:
        receipt = json.loads(Path(receipt_path).read_text())
        for name, digest in receipt["files"].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError(f"Source snapshot changed: {name}")
        inputs.append({"role": "source_snapshot", "path": receipt_path, "sha256": sha256_file(Path(receipt_path))})
    store = RunStore.create(output_root=request.output_dir, experiment_id=spec.experiment_id,
                            resolved_config=asdict(spec), command=request.command, repo_root=ROOT,
                            input_artifacts=inputs, resume_capability="unsupported")
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
        context = cache_context(spec, teacher_hash, student)
        cache = load(request.device_data)
        if cache["context"] != context or any(len(cache[k]["labels"]) != n for k, n in (("training", 5000), ("development", 1000), ("test", 10000))):
            raise ValueError("Teacher/cache cohort or coverage mismatch.")
        for key in ("training", "development", "test"):
            cache[key] = limited_cache(cache[key], spec.max_examples)
        bundle = checked_sources(request, context)
        incoming = load(request.weights)
        # Keep the previously development-selected base learning rates fixed.
        if spec.stage == "tune":
            rates = incoming["selected"]
            setting = replace(spec, learning_rate=rates["onchip_weights"], calibration_lr=rates["calibration"])
        else:
            if incoming.get("epoch_contract") != {"epochs": spec.epochs, "milestones": list(spec.milestones), "schedules": list(spec.schedules), "decay_after": list(spec.decay_after), "decay_factor": spec.decay_factor}:
                raise ValueError("Schedule selection belongs to a different epoch contract.")
            setting = replace(spec, **incoming["base_rates"])
        atomic_write_json(store.run_dir / "mapping.json", student.mapping_receipt())
        if spec.stage == "tune":
            result = tune(student, cache, bundle, setting, store, device)
            save(store.run_dir / "checkpoints/selection.pt", {
                "context": context, "source_bundle": bundle, "source_sha256": sha256_file(request.weights),
                "base_rates": {k: getattr(setting, k) for k in ("learning_rate", "calibration_lr")},
                "epoch_contract": {"epochs": spec.epochs, "milestones": list(spec.milestones), "schedules": list(spec.schedules), "decay_after": list(spec.decay_after), "decay_factor": spec.decay_factor},
                "selected": result["selected"],
            })
        else:
            result = screen(student, cache, bundle, incoming["selected"], setting, store, device, context)
            result["selected"] = incoming["selected"]
        result.update(stage=spec.stage, dataset=spec.dataset, smoke_only=bool(spec.max_examples),
                      evidence_class="exploratory_model_based", epochs=spec.epochs,
                      milestones=list(spec.milestones), base_rates={k: getattr(setting, k) for k in ("learning_rate", "calibration_lr")},
                      evaluation_images=len(cache["test"]["labels"]), recovery_images=len(cache["training"]["labels"]),
                      development_images=len(cache["development"]["labels"]),
                      teacher_kl_definition="mean KL(p_teacher || p_student), nats at T=1",
                      elapsed_seconds=time.monotonic() - started)
        artifacts = [store.artifact_record(p, kind="checkpoint" if p.suffix == ".pt" else "analysis") for p in store.run_dir.rglob("*") if p.is_file() and (p.suffix == ".pt" or p.name in ("mapping.json", "measurements.json"))]
        store.complete(metrics=result, artifacts=artifacts)
        print(json.dumps({"run_dir": str(store.run_dir), "status": "complete"}), flush=True)
        return 0
    except BaseException as exc:
        store.fail(exc)
        raise
