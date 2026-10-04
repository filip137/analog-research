"""Native full-cohort PCM endpoint and IBM OM pulse recovery comparison."""

from dataclasses import asdict
from pathlib import Path
import json
import os
import time

import torch
from torch.nn import functional as F

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.cifar_crossbar.data import cache_features, get_dataset, split_indices
from experiments.cifar_crossbar.devices import (
    EndpointKernel, PulseWriter, clone_cpu, make_plant, population_fingerprint,
    prepare_population, program,
)
from experiments.cifar_crossbar.fault_runtime import cache_context, limited_cache
from experiments.cifar_crossbar.full_epoch_config import FullEpochSpec
from experiments.cifar_crossbar.hwa_fault_runtime import case_name, cases, augment_images
from experiments.cifar_crossbar.model import CrossbarSuffix, state_hash, tensor_hash
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults
from experiments.cifar_crossbar.runtime import evaluate, load, save, source_model

ROOT = Path(__file__).resolve().parents[2]


def om_population(raw, kind, rate, seed):
    """Repair baseline is sampled before this independent Bernoulli fault mask.

    Low/high/random refer to each active cell's own native bound interval;
    the sampled intrinsic reference is retained. Not physical PCM G=0/25 uS.
    """
    if raw["kind"] != "om" or raw["policy"] != "repaired" or raw["aihwkit_version"] != "1.1.0":
        raise ValueError("Expected literal repaired AIHWKit 1.1.0 OM identities.")
    p = prepare_population(raw)
    h = p["hidden"]
    if h["corrupt"].any():
        raise ValueError("Expected a counterfactual repaired OM baseline.")
    rng = torch.Generator().manual_seed(seed)
    mask = torch.rand(h["max_bound"].shape, generator=rng) < rate
    fraction = torch.rand(mask.shape, generator=rng)
    if kind == "open":
        fraction.zero_()
    elif kind == "gmax":
        fraction.fill_(1)
    elif kind != "random":
        raise ValueError("Unknown failure kind.")
    stuck = h["min_bound"] + fraction * (h["max_bound"] - h["min_bound"])
    for key in ("min_bound", "max_bound"):
        h[key][mask] = stuck[mask]
    for key in ("dwmin_up", "dwmin_down"):
        h[key][mask] = 0
    h["corrupt"] |= mask
    h["injected_faults"] = mask
    p.update(fault_kind=kind, fault_rate=rate, fault_seed=seed)
    p["fingerprint"] = population_fingerprint(p)
    return p


def make_array(target, spec, kind, ppm, populations=None):
    if spec.backend == "pcm":
        faults = PermanentFaults.sample(target.numel(), kind, ppm / 1e6, spec.array_seed, target.device)
        return GaussianEndpointArray(target, faults, spec.endpoint_seed)
    raw = populations["arrays"][str(spec.array_seed)]
    if raw["size"] != target.numel() or raw["seed"] != spec.array_seed:
        raise ValueError("OM population layout/seed mismatch.")
    plant = make_plant(om_population(raw, kind, ppm / 1e6, spec.array_seed), spec.endpoint_seed, target.device)
    plant.deployment = program(plant.port(), target, tolerance=plant.nominal_step * spec.verify_step_fraction,
                              maximum_pulses=spec.program_pulses)
    return plant


def array_identity(array, spec):
    if spec.backend == "pcm":
        return array.faults.receipt()
    h = array.source["hidden"]
    return {"identity_sha256": array.source["fingerprint"],
            "failed_devices": int(h["injected_faults"].sum()), "physical_devices": array.size,
            "published_corrupt_repaired": int(h["published_corrupt"].sum()),
            "semantics": "one active OM cell per logical weight; exact intrinsic reference; stuck persistent state with apparent write noise"}


def verify_fixed(array, spec, initial):
    if spec.backend == "pcm":
        array.verify_permanence()
    else:
        mask = array.source["hidden"]["injected_faults"].flatten().to(array.engine.device)
        expected = initial["engine"]["persistent"].to(array.engine.device)
        if not torch.equal(array.engine.persistent[mask], expected[mask]):
            raise RuntimeError("Permanent OM states changed.")


def trajectory(student, array, target, cache, spec, method, store, directory, checkpoint_writer=None, writer_factory=None):
    calibrate = method in ("calibration", "rewrite", "onchip_calibration")
    learn = method in ("onchip_weights", "onchip_calibration")
    student.enable_calibration(calibrate)
    student.q.requires_grad_(False)
    master = torch.nn.Parameter(target.clone(), requires_grad=learn)
    groups = []
    if learn and spec.backend == "pcm":
        groups.append({"params": [master], "lr": spec.learning_rate})
    if calibrate:
        groups.append({"params": student.calibration_parameters(), "lr": spec.calibration_lr})
    optimizer = torch.optim.Adam(groups) if groups else None
    writer = None
    if learn and spec.backend == "om":
        writer = (writer_factory(array.port(), target, learning_rate=spec.learning_rate, seed=spec.seed)
                  if writer_factory is not None else
                  PulseWriter(array.port(), method="closed_loop", learning_rate=spec.learning_rate,
                             tolerance=array.nominal_step * spec.verify_step_fraction,
                             maximum_pulses=spec.update_pulses, cap=spec.recovery_pulses, seed=spec.seed))
    baseline = array.cost()
    initial_state = array.state_dict()
    prefix = student.prefix_hash()
    initial_hash = tensor_hash(array.read())
    curve = []
    rewrite_count = torch.zeros_like(target, dtype=torch.int64)
    rewrite_reads = 0
    for epoch in range(1, spec.epochs + 1):
        started = time.monotonic()
        order = torch.randperm(len(cache["training"]["labels"]), generator=torch.Generator().manual_seed(spec.data_seed + epoch))
        student.train()
        loss_sum = 0.0
        for batch, begin in enumerate(range(0, len(order), spec.batch_size)):
            idx = order[begin:begin + spec.batch_size]
            train = cache["training"]
            x = train["features"][idx].to(target.device)
            t = train["teacher_logits"][idx].to(target.device)
            student.zero_grad(set_to_none=True)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            observed = array.read().detach().clone()
            if spec.backend == "pcm":
                q = master + (observed - master).detach() if learn else observed
            else:
                q = observed.requires_grad_(learn)
            logits = student.forward_features(x, q)
            loss = F.kl_div(logits.log_softmax(1), t.softmax(1), reduction="batchmean")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite full-epoch teacher KL.")
            loss.backward()
            if optimizer:
                optimizer.step()
            if writer:
                writer.step(q.grad)
            elif learn:
                with torch.no_grad():
                    master.clamp_(-1, 1)
                array.program(master.detach())
            elif method == "rewrite":
                if spec.backend == "pcm":
                    array.program(target)
                else:
                    # Same one-pulse opportunity/cell/minibatch and total cap as recovery.
                    error = target - array.port().read()
                    active = (error.abs() > array.nominal_step * spec.verify_step_fraction) & (rewrite_count < spec.recovery_pulses)
                    array.port().pulse(error.sign().to(torch.int8) * active)
                    rewrite_count += active
                    rewrite_reads += array.size
            loss_sum += float(loss.detach()) * len(idx)
            if batch % 100 == 0:
                store.append_metric({"method": method, "epoch": epoch, "batch": batch,
                                     "case": str(directory.relative_to(store.run_dir)), "teacher_kl_train": float(loss.detach())})
        verify_fixed(array, spec, initial_state)
        if student.prefix_hash() != prefix:
            raise RuntimeError("Frozen prefix changed.")
        cost = {k: v - baseline[k] for k, v in array.cost().items()}
        if spec.backend == "om":
            count = writer.count if writer else rewrite_count
            cost.update(verify_reads=writer.verify_reads if writer else rewrite_reads,
                        capped_cells=0 if spec.recovery_pulses is None else int((count >= spec.recovery_pulses).sum()),
                        max_recovery_cell_pulses=int(count.max()))
            if writer_factory is not None:
                cost["probability_clipped"] = writer.probability_clipped if writer else 0
        row = {"epoch": epoch, "teacher_kl_train_mean": loss_sum / len(order),
               "image_presentations": epoch * len(order), "cost": cost,
               "development": evaluate(student, cache["development"], target.device, q=array.read()),
               "test": evaluate(student, cache["test"], target.device, q=array.read()),
               "epoch_seconds": time.monotonic() - started}
        if spec.backend == "om":
            row["persistent_test"] = evaluate(student, cache["test"], target.device, q=array.engine.persistent)
        (checkpoint_writer or save)(directory / f"{method}_epoch{epoch}.pt", {
            "p0_sha256": sha256_file(directory / "p0.pt"), "epoch": epoch,
            "calibration": {n: p for n, p in student.named_parameters() if n != "q" and p.requires_grad},
            "master_target": master, "array": array.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "writer": writer.state_dict() if writer else None,
            "rewrite_count": rewrite_count, "rewrite_reads": rewrite_reads})
        curve.append(row)
        store.append_metric({"terminal_epoch": True, "method": method,
                             "case": str(directory.relative_to(store.run_dir)), **row})
        print(json.dumps({"method": method, "epoch": epoch, "test": row["test"], "seconds": row["epoch_seconds"]}), flush=True)
    return {"method": method, "curve": curve, "initial_apparent_sha256": initial_hash}


def restore_endpoint(student, array, p0, endpoint):
    student.load_state_dict(p0["model"])
    with torch.no_grad():
        parameters = dict(student.named_parameters())
        for name, value in endpoint["calibration"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    array.load_state_dict(endpoint["array"])


def build_cache(teacher, student, spec, device):
    dataset = get_dataset(spec.dataset, True)
    training, development = split_indices(dataset.targets, seed=spec.data_seed)
    test = get_dataset(spec.dataset, False)
    if len(training) != 45000 or len(development) != 5000 or set(training) & set(development):
        raise ValueError("Expected disjoint 45000/5000 CIFAR split.")
    if spec.max_examples:
        training = training[:spec.max_examples]
    return {
        "training": cache_features(teacher, student, dataset, training, device=device),
        "development": cache_features(teacher, student, dataset, development[:spec.max_examples or 1000], device=device),
        "test": cache_features(teacher, student, test, range(spec.max_examples or len(test)), device=device),
        "hwa_raw": torch.from_numpy(dataset.data[training].copy()).permute(0, 3, 1, 2).contiguous(),
    }


def fit_om(teacher, student, cache, spec, populations, store, device):
    grid = torch.linspace(-1, 1, spec.kernel_bins, device=device).repeat_interleave(spec.kernel_samples)
    tables, diagnostics = [], []
    for role, seed in (("fit", 171001), ("heldout", 171002)):
        raw = populations[role]
        if raw["size"] != len(grid) or raw["seed"] != seed:
            raise ValueError("Kernel population mismatch.")
        plant = make_plant(prepare_population(raw), seed + 10000, device)
        endpoint = program(plant.port(), grid, tolerance=plant.nominal_step * spec.verify_step_fraction,
                           maximum_pulses=spec.program_pulses)
        table = (plant.read() - grid).reshape(spec.kernel_bins, spec.kernel_samples).detach().cpu()
        tables.append(table)
        diagnostics.append({"role": role, "endpoint": endpoint, "population": plant.source["fingerprint"], "cost": plant.cost()})
    error = (tables[0].mean(1) - tables[1].mean(1)).abs()
    stderr = (tables[0].var(1) / spec.kernel_samples + tables[1].var(1) / spec.kernel_samples).sqrt()
    adequate = bool((error <= torch.maximum(torch.full_like(error, .02), 4 * stderr)).all())
    save(store.run_dir / "checkpoints/kernel.pt", {"fit": tables[0], "heldout": tables[1], "diagnostics": diagnostics, "adequate": adequate})
    if not adequate:
        raise RuntimeError("OM endpoint kernel failed its existing held-out adequacy gate.")
    student.enable_calibration(True)
    student.q.requires_grad_(True)
    optimizer = torch.optim.Adam([student.q] + student.calibration_parameters(), lr=spec.hwa_learning_rate)
    kernel = EndpointKernel(tables[0], spec.seed + 1000, device)
    raw = cache["hwa_raw"].to(device)
    best, history = None, []
    prefix = student.prefix_hash()
    for epoch in range(spec.hwa_epochs + 1):
        if epoch:
            factor = .1 ** sum(epoch > e for e in (15, 25))
            for group in optimizer.param_groups:
                group["lr"] = spec.hwa_learning_rate * factor
            student.train()
            rng = torch.Generator(device=device).manual_seed(spec.data_seed + epoch)
            order = torch.randperm(len(raw), generator=rng, device=device)
            for batch, begin in enumerate(range(0, len(order), spec.batch_size)):
                x = augment_images(raw[order[begin:begin + spec.batch_size]], spec.dataset, rng)
                with torch.no_grad():
                    t = teacher(x)
                    features = student.features(x)
                optimizer.zero_grad(set_to_none=True)
                logits = student.forward_features(features, kernel.perturb(student.q))
                loss = F.kl_div(logits.log_softmax(1), t.softmax(1), reduction="batchmean")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite OM HWA loss.")
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    student.q.clamp_(-1, 1)
                if batch % 100 == 0:
                    store.append_metric({"stage": "om_hwa", "epoch": epoch, "batch": batch, "loss": float(loss.detach())})
        scores = [evaluate(student, cache["development"], device,
                           q=EndpointKernel(tables[1], 181001 + draw, device).perturb(student.q).detach())
                  for draw in range(2)]
        score = sum(r["teacher_kl"] for r in scores) / len(scores)
        history.append({"epoch": epoch, "development_kl": score})
        if best is None or score < best["score"]:
            best = {"score": score, "epoch": epoch, "model": clone_cpu(student.state_dict())}
        store.append_metric({"stage": "om_hwa", **history[-1]})
        save(store.run_dir / "checkpoints/hwa_progress.pt", {"best": best, "history": history, "model": student.state_dict(), "optimizer": optimizer.state_dict(), "kernel_rng": kernel.generator.get_state()})
        print(json.dumps({"stage": "om_hwa", **history[-1]}), flush=True)
    if student.prefix_hash() != prefix:
        raise RuntimeError("HWA changed frozen prefix.")
    return {"model": best["model"], "fit": {"selected_epoch": best["epoch"], "history": history,
             "recipe": "30 augmented full-data epochs; Adam teacher KL 1e-4; independent OM endpoint kernel",
             "convergence_review_required": best["epoch"] >= .8 * spec.hwa_epochs,
             "limited_recipe_not_exhaustively_tuned": True}}


def screen(student, cache, sources, spec, populations, store, context):
    rows = []
    for name, source in sources.items():
        student.load_state_dict(source["model"])
        target = student.q.detach().clone()
        clean = evaluate(student, cache["test"], target.device)
        for kind, ppm in cases(spec):
            student.load_state_dict(source["model"])
            case = case_name(kind, ppm)
            array = make_array(target, spec, kind, ppm, populations)
            directory = store.run_dir / "checkpoints" / name / case
            p0 = {"context": context, "model": source["model"], "array": array.state_dict(),
                  "target": target, "identity": array_identity(array, spec)}
            save(directory / "p0.pt", p0)
            initial = evaluate(student, cache["test"], target.device, q=array.read())
            for method in spec.methods:
                student.load_state_dict(source["model"])
                array.load_state_dict(p0["array"])
                base = {"source": name, "case": case, "array_seed": spec.array_seed,
                        "backend": spec.backend, "p0_sha256": sha256_file(directory / "p0.pt"),
                        "identity": p0["identity"], "clean": clean, "initial_test": initial}
                if method == "none":
                    result = {"method": method, "curve": [], "initial_apparent_sha256": tensor_hash(array.read())}
                else:
                    result = trajectory(student, array, target, cache, spec, method, store, directory)
                rows.append({**base, **result})
                atomic_write_json(store.run_dir / "measurements.json", rows)
    return {"measurements": rows, "completed_controls": len(rows)}


def run_train(request):
    spec = request.spec
    if not isinstance(spec, FullEpochSpec) or request.teacher_weights is None:
        raise ValueError("Expected full-epoch spec and explicit digital teacher.")
    if any(getattr(request, k, None) is not None for k in ("resume", "base_weights", "device_state")):
        raise ValueError("Unexpected full-epoch input.")
    inputs = [{"role": k, "path": str(getattr(request, k).resolve()), "sha256": sha256_file(getattr(request, k))}
              for k in ("teacher_weights", "weights", "device_data", "device_model", "selection_receipt") if getattr(request, k, None) is not None]
    receipt_path = os.environ.get("EBL_SOURCE_RECEIPT")
    if receipt_path:
        receipt = json.loads(Path(receipt_path).read_text())
        for name, digest in receipt["files"].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError(f"Source changed: {name}")
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
        teacher, digest = source_model(request.teacher_weights, spec, device)
        student = CrossbarSuffix(teacher, spec.suffix, spec.tile_size)
        context = cache_context(spec, digest, student)
        atomic_write_json(store.run_dir / "mapping.json", student.mapping_receipt())
        if spec.stage == "cache":
            cache = build_cache(teacher, student, spec, device)
            save(store.run_dir / "checkpoints/cache.pt", {"context": context, **cache})
            result = {"digital_test": evaluate(student, cache["test"], device),
                      "cohorts": {k: cache[k]["cohort"] for k in ("training", "development", "test")}}
        else:
            if request.device_data is None:
                raise ValueError("Missing full-cohort cache.")
            cache = load(request.device_data)
            if cache["context"] != context:
                raise ValueError("Full-cohort cache context mismatch.")
            for key, n in (("training", 45000), ("development", 1000), ("test", 10000)):
                if len(cache[key]["labels"]) != (spec.max_examples or n):
                    raise ValueError(f"Incomplete {key} cohort.")
                for field in ("features", "teacher_logits"):
                    cache[key][field] = cache[key][field].to(device)
            populations = load(request.device_model) if request.device_model else None
            if spec.backend == "om" and (populations is None or populations["dataset"] != spec.dataset):
                raise ValueError("Missing matching literal OM population bundle.")
            if spec.stage == "sources":
                sources = {"digital": {"model": clone_cpu(student.state_dict()), "fit": {"kind": "original_teacher"}}}
                if spec.backend == "om":
                    if request.weights is not None:
                        raise ValueError("OM HWA must not import PCM HWA weights.")
                    sources["standard_hwa"] = fit_om(teacher, student, cache, spec, populations, store, device)
                else:
                    if request.weights is None:
                        raise ValueError("Missing explicit prior PCM HWA bundle.")
                    incoming = load(request.weights)
                    prior = incoming.get("source_bundle", incoming)
                    old_context = {**context, "recovery_images": 5000}
                    if prior["context"] != old_context:
                        raise ValueError("Prior HWA teacher/mapping/split mismatch.")
                    if state_hash(prior["sources"]["digital"]["model"]) != state_hash(sources["digital"]["model"]):
                        raise ValueError("Prior digital source changed.")
                    sources["standard_hwa"] = prior["sources"]["standard_hwa"]
                save(store.run_dir / "checkpoints/sources.pt", {"context": context, "backend": spec.backend, "sources": sources})
                result = {"sources": {k: {"sha256": state_hash(v["model"]), "fit": v["fit"]} for k, v in sources.items()}}
            else:
                incoming = load(request.weights)
                if incoming["context"] != context or incoming["backend"] != spec.backend or set(incoming["sources"]) != {"digital", "standard_hwa"}:
                    raise ValueError("Unmatched source bundle.")
                result = screen(student, cache, incoming["sources"], spec, populations, store, context)
        result.update(stage=spec.stage, dataset=spec.dataset, backend=spec.backend,
                      smoke_only=bool(spec.max_examples), evidence_class="exploratory_model_based",
                      epochs=spec.epochs, recovery_images=spec.max_examples or 45000,
                      teacher_kl_definition="mean KL(p_teacher || p_student), nats at T=1",
                      elapsed_seconds=time.monotonic() - started)
        store.append_metric({"stage": spec.stage, "terminal": True,
                             "elapsed_seconds": result["elapsed_seconds"]})
        artifacts = [store.artifact_record(p, kind="checkpoint" if p.suffix == ".pt" else "analysis")
                     for p in store.run_dir.rglob("*") if p.is_file() and (p.suffix == ".pt" or p.name in ("mapping.json", "measurements.json"))]
        store.complete(metrics=result, artifacts=artifacts)
        print(json.dumps({"run_dir": str(store.run_dir), "status": "complete"}), flush=True)
        return 0
    except BaseException as exc:
        store.fail(exc)
        raise
