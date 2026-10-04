"""Native, artifact-backed stages for CIFAR suffix experiments."""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import time
import torch
from torch.nn import functional as F
from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.cifar_crossbar.config import Spec
from experiments.cifar_crossbar.data import (
    get_dataset,
    split_indices,
    cache_features,
    cohort_receipt,
)
from experiments.cifar_crossbar.devices import (
    EndpointKernel,
    PulseWriter,
    clone_cpu,
    make_plant,
    prepare_population,
    program,
)
from experiments.cifar_crossbar.model import (
    CifarResNet32,
    CrossbarSuffix,
    tensor_hash,
)
from experiments.cifar_crossbar.prepare import SOURCES

ROOT = Path(__file__).resolve().parents[2]
DEVICE_FIELDS = (
    "backend",
    "policy",
    "noise_scale",
    "variation_scale",
    "fault_rate",
    "fault_kind",
    "program_pulses",
    "verify_step_fraction",
)


def scientific_device(spec):
    return {k: getattr(spec, k) for k in DEVICE_FIELDS}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(clone_cpu(value), temporary)
    temporary.replace(path)


def load(path):
    import gzip
    import io
    with open(path, "rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rb") as handle:
            return torch.load(io.BytesIO(handle.read()), map_location="cpu", weights_only=True)
    return torch.load(path, map_location="cpu", weights_only=True)


def source_model(path, spec, device):
    digest = sha256_file(Path(path))
    if not digest.startswith(SOURCES[spec.dataset][1]):
        raise ValueError(
            "Expected the declared public pretrained ResNet-32 checkpoint digest."
        )
    teacher = CifarResNet32(10 if spec.dataset == "cifar10" else 100).to(device)
    teacher.load_state_dict(load(path), strict=True)
    teacher.eval().requires_grad_(False)
    return teacher, digest


def native_population(spec, size, seed, output):
    interpreter = os.environ.get("EBL_AIHWKIT_PYTHON")
    if not interpreter:
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to name a pinned AIHWKit 1.1.0 CPU interpreter."
        )
    command = [
        interpreter,
        "-m",
        "experiments.cifar_crossbar.native",
        "--kind",
        spec.backend,
        "--size",
        str(size),
        "--seed",
        str(seed),
        "--policy",
        spec.policy,
        "--variation",
        str(spec.variation_scale),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return prepare_population(
        load(output),
        noise_scale=spec.noise_scale,
        fault_rate=spec.fault_rate,
        fault_kind=spec.fault_kind,
        fault_seed=seed + 171,
    )


@torch.no_grad()
def evaluate(student, cache, device, *, q=None, batch_size=256, read_seed=50001):
    old_rng = student.read_generator.get_state()
    student.read_generator.manual_seed(read_seed)
    student.eval()
    total = correct = agreement = 0
    kl = ce = 0.0
    predictions = []
    try:
        for begin in range(0, len(cache["labels"]), batch_size):
            x = cache["features"][begin : begin + batch_size].to(device)
            t = cache["teacher_logits"][begin : begin + batch_size].to(device)
            y = cache["labels"][begin : begin + batch_size].to(device)
            out = student.forward_features(x, q)
            if not bool(torch.isfinite(out).all()):
                raise FloatingPointError("Nonfinite network scores.")
            p = out.argmax(1)
            predictions.append(p.cpu())
            correct += int((p == y).sum())
            agreement += int((p == t.argmax(1)).sum())
            total += len(y)
            kl += float(F.kl_div(out.log_softmax(1), t.softmax(1), reduction="sum"))
            ce += float(F.cross_entropy(out, y, reduction="sum"))
    finally:
        student.read_generator.set_state(old_rng)
    return {
        "examples": total,
        "accuracy_percent": 100 * correct / total,
        "teacher_agreement_percent": 100 * agreement / total,
        "teacher_kl": kl / total,
        "cross_entropy": ce / total,
        "prediction_sha256": tensor_hash(torch.cat(predictions)),
    }


def build_cache(teacher, student, spec, device, *, training=False):
    dataset = get_dataset(spec.dataset, train=True)
    train_indices, validation = split_indices(dataset.targets, seed=spec.data_seed)
    if training:
        indices = train_indices[: spec.recovery_images]
    elif spec.evaluation == "confirmation":
        dataset = get_dataset(spec.dataset, train=False)
        indices = list(range(len(dataset)))
    else:
        indices = validation
    if spec.max_examples:
        indices = indices[: spec.max_examples]
    return cache_features(
        teacher, student, dataset, indices, device=device
    ), cohort_receipt(indices)


def checkpoint_context(spec, teacher_hash, student):
    return {
        "schema": "cifar_crossbar.checkpoint.v1",
        "dataset": spec.dataset,
        "suffix": spec.suffix,
        "teacher_sha256": teacher_hash,
        "mapping": student.mapping_receipt(),
        "device_settings": scientific_device(spec),
        "read_noise": spec.read_noise,
        "assignment_seed": spec.assignment_seed,
        "endpoint_seed": spec.endpoint_seed,
    }


def verify_context(bundle, context, *, devices=True):
    for key in ("schema", "dataset", "suffix", "teacher_sha256", "mapping"):
        if bundle.get(key) != context[key]:
            raise ValueError(f"Expected matching checkpoint {key}.")
    if devices and bundle.get("device_settings") != context["device_settings"]:
        raise ValueError("Expected matching physical device settings.")
    if (
        bundle.get("device_settings", {}).get("backend")
        != context["device_settings"]["backend"]
    ):
        raise ValueError("Expected HWA and deployment to use the same device family.")
    if devices and bundle.get("read_noise") != context["read_noise"]:
        raise ValueError("Expected matching MVM read noise.")


def rng_state(student):
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "reads": student.read_generator.get_state(),
    }


def restore_rng(state, student):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    student.read_generator.set_state(state["reads"].cpu())


def characterize(spec, student, store, device):
    bundle = native_population(
        spec,
        student.q.numel(),
        spec.assignment_seed,
        store.run_dir / "native_population.pt",
    )
    save(
        store.run_dir / "population.pt",
        {
            "population": bundle,
            "device_settings": scientific_device(spec),
            "mapping": student.mapping_receipt(),
        },
    )
    grid = torch.linspace(-1, 1, spec.kernel_bins, device=device).repeat_interleave(
        spec.kernel_samples
    )
    results = []
    tables = []
    for role, seed in (
        ("fit", spec.kernel_seed),
        ("heldout", spec.kernel_seed + 100000),
    ):
        raw = native_population(
            spec, len(grid), seed, store.run_dir / f"{role}_population.pt"
        )
        plant = make_plant(raw, spec.endpoint_seed + seed, device)
        report = program(
            plant.port(),
            grid,
            tolerance=plant.nominal_step * spec.verify_step_fraction,
            maximum_pulses=spec.program_pulses,
        )
        residual = (
            (plant.read() - grid)
            .reshape(spec.kernel_bins, spec.kernel_samples)
            .detach()
            .cpu()
        )
        tables.append(residual)
        results.append(
            {
                "role": role,
                "population_seed": seed,
                "endpoint": report,
                "cost": plant.cost(),
                "residual_sha256": tensor_hash(residual),
            }
        )
        store.append_metric({"stage": "characterize", **results[-1]})
    error = (tables[0].mean(1) - tables[1].mean(1)).abs()
    stderr = (
        tables[0].var(1) / spec.kernel_samples + tables[1].var(1) / spec.kernel_samples
    ).sqrt()
    adequate = bool(
        (error <= torch.maximum(torch.full_like(error, 0.02), 4 * stderr)).all()
    )
    kernel = {
        "schema": "cifar_crossbar.endpoint_kernel.v1",
        "residuals": tables[0],
        "heldout_residuals": tables[1],
        "kernel_seed": spec.kernel_seed,
        "device_settings": scientific_device(spec),
        "includes_failure_terminals": True,
        "characterization": results,
        "adequate": adequate,
        "maximum_bin_mean_discrepancy": float(error.max()),
    }
    save(store.run_dir / "kernel.pt", kernel)
    return {
        "stage": "characterize",
        "adequate": adequate,
        "logical_weights": student.q.numel(),
        "kernel": {k: v for k, v in kernel.items() if not isinstance(v, torch.Tensor)},
    }


def fit_hwa(request, spec, teacher, student, context, cache, store, device):
    from experiments.cifar_crossbar.pcm_reference import PcmInferenceNoise

    def inference(seed):
        return PcmInferenceNoise(
            student,
            seed,
            device,
            time_seconds=spec.retention_seconds,
            noise_scale=spec.noise_scale,
            compensate=spec.drift_compensation,
        )

    if spec.backend == "pcm_inference":
        kernel_bundle = None
        kernel = inference(spec.seed + 1000)
    else:
        if request.device_model is None:
            raise ValueError("Expected --device-model naming a characterized kernel.")
        kernel_bundle = load(request.device_model)
        if (
            kernel_bundle["device_settings"] != scientific_device(spec)
            or not kernel_bundle["adequate"]
        ):
            raise ValueError(
                "Expected an adequate kernel with matching device settings."
            )
        kernel = EndpointKernel(kernel_bundle["residuals"], spec.seed + 1000, device)
    student.enable_calibration(True)
    optimizer = torch.optim.Adam(
        [student.q] + student.calibration_parameters(), lr=spec.learning_rate
    )
    dataset = get_dataset(spec.dataset, True, augment=True)
    indices, _ = split_indices(dataset.targets, seed=spec.data_seed)
    if spec.max_examples:
        indices = indices[: spec.max_examples]
    best = None
    start = 0
    history = []
    if request.resume is not None:
        resumed = load(request.resume)
        verify_context(resumed, context)
        if resumed["run_spec"] != asdict(spec):
            raise ValueError("Expected the same resume run specification.")
        student.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        kernel.generator.set_state(resumed["kernel_rng"].cpu())
        restore_rng(resumed["rng"], student)
        start = resumed["epoch"]
        best = resumed["best"]
        history = resumed["history"]
    for epoch in range(start + int(request.resume is not None), spec.epochs + 1):
        if epoch > 0:
            student.train()
            order = torch.Generator().manual_seed(spec.data_seed + epoch)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, list(map(int, indices))),
                batch_size=spec.batch_size,
                shuffle=True,
                generator=order,
                num_workers=0,
            )
            for batch, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                q = kernel.perturb(
                    student.q, min(1.0, epoch / max(1, spec.ramp_epochs))
                )
                out = student(x, q)
                if spec.objective == "teacher_kl":
                    with torch.no_grad():
                        target = teacher(x).softmax(1)
                    loss = F.kl_div(out.log_softmax(1), target, reduction="batchmean")
                else:
                    loss = F.cross_entropy(out, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite HWA objective.")
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    student.q.clamp_(-1, 1)
                if batch % 50 == 0:
                    store.append_metric(
                        {
                            "stage": "hwa",
                            "epoch": epoch,
                            "batch": batch,
                            "loss": float(loss),
                        }
                    )
        # Selection averages three independent population endpoint realizations.
        selection = []
        for draw in range(3):
            evaluator = (
                inference(60000 + draw)
                if kernel_bundle is None
                else EndpointKernel(
                    kernel_bundle["heldout_residuals"], 60000 + draw, device
                )
            )
            selection.append(
                evaluate(
                    student, cache, device, q=evaluator.perturb(student.q).detach()
                )
            )
        score = sum(x["accuracy_percent"] for x in selection) / 3
        kl = sum(x["teacher_kl"] for x in selection) / 3
        item = {
            "stage": "hwa",
            "epoch": epoch,
            "validation_accuracy_percent": score,
            "teacher_kl": kl,
        }
        history.append(item)
        store.append_metric(item)
        rank = (score, -kl, -epoch)
        if best is None or rank > tuple(best["rank"]):
            best = {
                "rank": rank,
                "epoch": epoch,
                "metrics": item,
                "model": clone_cpu(student.state_dict()),
            }
            save(
                store.run_dir / "checkpoints/weights.pt",
                {
                    **context,
                    "model": best["model"],
                    "selected_epoch": epoch,
                    "selection": item,
                    "hwa_training_epochs": spec.epochs,
                    "convergence_review_required": epoch
                    >= max(1, int(0.8 * spec.epochs)),
                    "kernel_sha256": (
                        None
                        if kernel_bundle is None
                        else sha256_file(request.device_model)
                    ),
                },
            )
        save(
            store.run_dir / "checkpoints/resume.pt",
            {
                **context,
                "run_spec": asdict(spec),
                "epoch": epoch,
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "kernel_rng": kernel.generator.get_state(),
                "rng": rng_state(student),
                "best": best,
                "history": history,
            },
        )
    if not (store.run_dir / "checkpoints/weights.pt").exists():
        # A continuation may never improve on its inherited best checkpoint.
        save(
            store.run_dir / "checkpoints/weights.pt",
            {
                **context,
                "model": best["model"],
                "selected_epoch": best["epoch"],
                "selection": best["metrics"],
                "hwa_training_epochs": spec.epochs,
                "convergence_review_required": best["epoch"]
                >= max(1, int(0.8 * spec.epochs)),
                "kernel_sha256": (
                    None if kernel_bundle is None else sha256_file(request.device_model)
                ),
            },
        )
    return {
        "stage": "hwa",
        "selected": best["metrics"],
        "history": history,
        "convergence_review_required": best["epoch"] >= max(1, int(0.8 * spec.epochs)),
        "training_cohort": cohort_receipt(indices),
        "test_used_for_selection": False,
    }


def recover(request, spec, teacher, student, context, cache, store, device):
    if request.device_state is None:
        raise ValueError("Expected --device-state naming the exact deployment P0.")
    p0 = load(request.device_state)
    verify_context(p0, context)
    for key in ("assignment_seed", "endpoint_seed"):
        if p0.get(key) != getattr(spec, key):
            raise ValueError(f"Expected the same P0 {key}.")
    student.load_state_dict(p0["model"])
    student.q.requires_grad_(True)
    student.enable_calibration(spec.method != "none")
    plant = make_plant(p0["plant"]["population"], spec.endpoint_seed, device)
    plant.load_state_dict(p0["plant"])
    if hasattr(plant, "begin_recovery"):
        plant.begin_recovery(spec.recovery_pulses)
    initial_cost = plant.cost()
    initial = evaluate(student, cache, device, q=plant.read())
    initial_hash = tensor_hash(plant.read())
    training, cohort = build_cache(teacher, student, spec, device, training=True)
    optimizer = (
        torch.optim.Adam(student.calibration_parameters(), lr=spec.calibration_lr)
        if spec.method != "none"
        else None
    )
    writer = (
        PulseWriter(
            plant.port(),
            method=spec.method,
            learning_rate=spec.learning_rate,
            tolerance=plant.nominal_step * spec.verify_step_fraction,
            cap=spec.recovery_pulses,
            maximum_pulses=spec.program_pulses,
            seed=spec.seed,
        )
        if spec.method in ("open_loop", "closed_loop")
        else None
    )
    start = 0
    history = []
    rewrite_report = None
    if request.resume is not None:
        resumed = load(request.resume)
        verify_context(resumed, context)
        if resumed["run_spec"] != asdict(spec) or resumed["p0_sha256"] != sha256_file(
            request.device_state
        ):
            raise ValueError(
                "Expected identical recovery settings and P0 on continuation."
            )
        plant.load_state_dict(resumed["plant"])
        student.load_state_dict(resumed["model"])
        if optimizer:
            optimizer.load_state_dict(resumed["optimizer"])
        if writer:
            writer.load_state_dict(resumed["writer"])
        restore_rng(resumed["rng"], student)
        start = resumed["epoch"]
        history = resumed["history"]
        rewrite_report = resumed["rewrite_report"]
    elif spec.method == "rewrite":
        rewrite_report = program(
            plant.port(),
            p0["target_q"].to(device),
            tolerance=plant.nominal_step * spec.verify_step_fraction,
            maximum_pulses=spec.program_pulses,
        )
    for epoch in range(start + 1, (spec.epochs if spec.method != "none" else 0) + 1):
        epoch_started = time.monotonic()
        student.train()
        order = torch.randperm(
            len(training["labels"]),
            generator=torch.Generator().manual_seed(spec.data_seed + epoch),
        )
        for batch, begin in enumerate(range(0, len(order), spec.batch_size)):
            selected = order[begin : begin + spec.batch_size]
            x = training["features"][selected].to(device)
            t = training["teacher_logits"][selected].to(device)
            student.zero_grad(set_to_none=True)
            with torch.no_grad():
                student.q.copy_(plant.read())
            out = student.forward_features(x)
            loss = F.kl_div(out.log_softmax(1), t.softmax(1), reduction="batchmean")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite recovery objective.")
            loss.backward()
            optimizer.step()
            if writer:
                writer.step(student.q.grad)
            if batch % 50 == 0:
                store.append_metric(
                    {
                        "stage": "recover",
                        "epoch": epoch,
                        "batch": batch,
                        "loss": float(loss),
                    }
                )
        metrics = evaluate(student, cache, device, q=plant.read())
        record = {
            "stage": "recover",
            "epoch": epoch,
            "metrics": metrics,
            "cost": plant.cost(),
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(record)
        store.append_metric(record)
        with torch.no_grad():
            student.q.copy_(plant.read())
        save(
            store.run_dir / "checkpoints/resume.pt",
            {
                **context,
                "run_spec": asdict(spec),
                "p0_sha256": sha256_file(request.device_state),
                "model": student.state_dict(),
                "plant": plant.state_dict(),
                "optimizer": optimizer.state_dict(),
                "writer": None if writer is None else writer.state_dict(),
                "rng": rng_state(student),
                "epoch": epoch,
                "history": history,
                "rewrite_report": rewrite_report,
            },
        )
    final = evaluate(student, cache, device, q=plant.read())
    with torch.no_grad():
        student.q.copy_(plant.read())
    save(
        store.run_dir / "checkpoints/final.pt",
        {
            **context,
            "model": student.state_dict(),
            "plant": plant.state_dict(),
            "p0_sha256": sha256_file(request.device_state),
        },
    )
    final_cost = plant.cost()
    return {
        "stage": "recover",
        "method": spec.method,
        "initial": initial,
        "final": final,
        "history": history,
        "gain_pp": final["accuracy_percent"] - initial["accuracy_percent"],
        "p0_sha256": sha256_file(request.device_state),
        "initial_apparent_sha256": initial_hash,
        "recovery_cohort": cohort,
        "image_presentations": (
            cohort["count"] * spec.epochs if spec.method != "none" else 0
        ),
        "cost": {
            k: final_cost[k] - initial_cost[k]
            for k in final_cost
            if k != "max_cell_pulses"
        },
        "max_total_cell_pulses": final_cost["max_cell_pulses"],
        "controller_verify_reads": 0 if writer is None else writer.verify_reads,
        "rewrite": rewrite_report,
        "selection": "fixed_development_selected_schedule_no_target_checkpoint_selection",
    }


def run_train(request):
    started = time.monotonic()
    spec = request.spec
    if not isinstance(spec, Spec):
        raise TypeError("Expected CIFAR suffix specification.")
    if request.teacher_weights is None:
        raise ValueError(
            "Expected --teacher-weights with the pretrained digital checkpoint."
        )
    if request.resume is not None and spec.stage not in ("hwa", "recover"):
        raise ValueError("Only HWA and recovery support continuation checkpoints.")
    inputs = [
        {
            "role": key,
            "path": str(getattr(request, key).resolve()),
            "sha256": sha256_file(getattr(request, key)),
        }
        for key in (
            "teacher_weights",
            "weights",
            "device_model",
            "device_state",
            "resume",
            "selection_receipt",
        )
        if getattr(request, key, None) is not None
    ]
    source_receipt = os.environ.get("EBL_SOURCE_RECEIPT")
    if source_receipt:
        receipt = json.loads(Path(source_receipt).read_text())
        for name, digest in receipt["files"].items():
            source = (ROOT / name).resolve()
            if not source.is_relative_to(ROOT) or sha256_file(source) != digest:
                raise ValueError(f"Staged source snapshot mismatch: {name}")
        inputs.append(
            {
                "role": "staged_source_snapshot",
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
        resume_capability=(
            "exact" if spec.stage in ("hwa", "recover") else "unsupported"
        ),
    )
    try:
        torch.set_num_threads(1)
        torch.manual_seed(spec.seed)
        torch.backends.cudnn.enabled = not spec.disable_cudnn
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device(spec.device)
        teacher, teacher_hash = source_model(request.teacher_weights, spec, device)
        student = CrossbarSuffix(teacher, spec.suffix, spec.tile_size)
        student.read_noise = spec.read_noise
        context = checkpoint_context(spec, teacher_hash, student)
        prefix = student.prefix_hash()
        atomic_write_json(store.run_dir / "mapping.json", student.mapping_receipt())
        if request.weights:
            weights = load(request.weights)
            verify_context(weights, context, devices=False)
            if (
                not spec.max_examples
                and spec.stage in ("deploy", "pcm_reference")
                and weights.get("convergence_review_required", True)
            ):
                raise ValueError(
                    "Resolve HWA convergence on development data before scientific deployment."
                )
            student.load_state_dict(weights["model"])
        if spec.stage == "characterize":
            result = characterize(spec, student, store, device)
        else:
            cache, cohort = build_cache(teacher, student, spec, device)
            if spec.stage == "audit":
                result = {
                    "stage": "audit",
                    "metrics": evaluate(student, cache, device),
                    "evaluation_cohort": cohort,
                    "teacher_accuracy_percent": float(
                        (cache["teacher_logits"].argmax(1) == cache["labels"])
                        .float()
                        .mean()
                        * 100
                    ),
                    "mapping": student.mapping_receipt(),
                }
                if (
                    spec.evaluation == "confirmation"
                    and abs(
                        result["teacher_accuracy_percent"] - SOURCES[spec.dataset][2]
                    )
                    > 0.25
                ):
                    raise RuntimeError(
                        "Digital checkpoint accuracy does not reproduce the published source within 0.25 pp."
                    )
            elif spec.stage == "hwa":
                result = fit_hwa(
                    request, spec, teacher, student, context, cache, store, device
                )
            elif spec.stage == "deploy":
                if request.device_model is None:
                    raise ValueError(
                        "Expected --device-model with a target population bundle."
                    )
                bundle = load(request.device_model)
                if (
                    bundle["device_settings"] != scientific_device(spec)
                    or bundle["mapping"] != student.mapping_receipt()
                ):
                    raise ValueError(
                        "Expected target population settings and mapping to match."
                    )
                if bundle["population"]["seed"] != spec.assignment_seed:
                    raise ValueError("Expected the declared fresh-array seed.")
                plant = make_plant(bundle["population"], spec.endpoint_seed, device)
                target = student.q.detach().clone()
                clean = evaluate(student, cache, device)
                programming = program(
                    plant.port(),
                    target,
                    tolerance=plant.nominal_step * spec.verify_step_fraction,
                    maximum_pulses=spec.program_pulses,
                )
                apparent = evaluate(student, cache, device, q=plant.read())
                save(
                    store.run_dir / "checkpoints/p0.pt",
                    {
                        **context,
                        "model": student.state_dict(),
                        "target_q": target,
                        "plant": plant.state_dict(),
                        "programming": programming,
                        "hwa_weights_sha256": (
                            None
                            if request.weights is None
                            else sha256_file(request.weights)
                        ),
                    },
                )
                result = {
                    "stage": "deploy",
                    "clean": clean,
                    "apparent": apparent,
                    "programming": programming,
                    "cost": plant.cost(),
                    "deployment_drop_pp": clean["accuracy_percent"]
                    - apparent["accuracy_percent"],
                    "source": "direct_pretrained" if request.weights is None else "hwa",
                    "apparent_sha256": tensor_hash(plant.read()),
                }
            elif spec.stage == "recover":
                result = recover(
                    request, spec, teacher, student, context, cache, store, device
                )
            elif spec.stage == "pcm_reference":
                from experiments.cifar_crossbar.pcm_reference import evaluate_reference

                calibration_cache, _ = build_cache(
                    teacher, student, spec, device, training=True
                )
                result = evaluate_reference(
                    spec, student, cache, device, calibration_cache
                )
            else:
                raise ValueError("Expected a supported CIFAR stage.")
        if student.prefix_hash() != prefix:
            raise RuntimeError("Frozen digital prefix changed.")
        result.update(
            {
                "dataset": spec.dataset,
                "suffix": spec.suffix,
                "backend": spec.backend,
                "assignment_seed": spec.assignment_seed,
                "endpoint_seed": spec.endpoint_seed,
                "hwa_seed": spec.seed,
                "evaluation": spec.evaluation,
                "evidence_class": "model_based_aihwkit_preset",
                "smoke_only": bool(spec.max_examples),
                "prefix_sha256": prefix,
            }
        )
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_gpu_memory_bytes"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        store.append_metric(
            {
                "stage": spec.stage,
                "terminal": True,
                "elapsed_seconds": result["elapsed_seconds"],
                "summary": {
                    k: result[k]
                    for k in (
                        "metrics",
                        "clean",
                        "apparent",
                        "final",
                        "selected",
                        "adequate",
                    )
                    if k in result
                },
            }
        )
        artifacts = [
            store.artifact_record(
                p, kind="checkpoint" if p.suffix == ".pt" else "mapping"
            )
            for p in store.run_dir.rglob("*")
            if p.is_file() and (p.suffix == ".pt" or p.name == "mapping.json")
        ]
        store.complete(metrics=result, artifacts=artifacts)
        print(
            json.dumps(
                {
                    "run_dir": str(store.run_dir),
                    "stage": spec.stage,
                    "status": "complete",
                }
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        store.fail(exc)
        raise
