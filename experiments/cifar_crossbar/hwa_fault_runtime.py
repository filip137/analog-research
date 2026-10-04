"""Generic HWA/CDT fitting, development selection, and fixed-array recovery.

All numerical stages enter through EBL. Existing campaign orchestration and
the tested permanent-array recovery adapter are reused.
"""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.cifar_crossbar.data import (
    NORMALIZATION,
    cache_features,
    cohort_receipt,
    get_dataset,
    split_indices,
)
from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.fault_runtime import adapt, cache_context, limited_cache
from experiments.cifar_crossbar.hwa_fault_config import ComparisonSpec
from experiments.cifar_crossbar.model import CrossbarSuffix, state_hash
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults
from experiments.cifar_crossbar.prepare import SOURCES
from experiments.cifar_crossbar.runtime import evaluate, load, save, source_model

ROOT = Path(__file__).resolve().parents[2]


def cases(spec, kind=None):
    return [("open", 0)] + [
        (k, p)
        for k in (spec.fault_kinds if kind is None else (kind,))
        for p in spec.fault_rates_ppm
        if p
    ]


def case_name(kind, ppm):
    return "nominal" if not ppm else f"{kind}_{ppm}ppm"


def sampled_weights(q, generator, noise_scale=1.0, kind="open", rate=0.0):
    """Resample the physical-bank population every minibatch; preserve master.

    Uses the paper's CDT principle with our physical-device probability
    convention consistently in training and deployment. The temporary mask
    never becomes a deployment fault map. Straight-through gradient follows
    the temporary effective weight, as in the existing HWA implementation.
    """
    with torch.no_grad():
        target = torch.stack((q.detach().clamp_min(0), (-q.detach()).clamp_min(0))) * 25
        r = target / 25
        sigma = 0.26348 + 1.965 * r - 1.1731 * r.square()
        g = (
            target
            + noise_scale
            * sigma
            * torch.randn(target.shape, generator=generator, device=q.device)
        ).clamp_min(0)
        mask = torch.rand(target.shape, generator=generator, device=q.device) < rate
        fixed = torch.rand(target.shape, generator=generator, device=q.device) * 25
        if kind == "open":
            fixed.zero_()
        elif kind == "gmax":
            fixed.fill_(25)
        elif kind != "random":
            raise ValueError("Expected open/gmax/random training corruption.")
        g = torch.where(mask, fixed, g)
        apparent = (g[0] - g[1]) / 25
    return q + (apparent - q).detach(), mask


def augment_images(raw, dataset, generator):
    """Vectorized random crop/flip before normalization, fresh each epoch."""
    x = raw.float() / 255
    padded = F.pad(x, (4, 4, 4, 4))
    n, channels = raw.shape[:2]
    offsets = torch.randint(9, (n, 2), generator=generator, device=raw.device)
    a = torch.arange(32, device=raw.device)
    ys = offsets[:, 0, None, None] + a[None, :, None]
    xs = offsets[:, 1, None, None] + a[None, None, :]
    x = padded[
        torch.arange(n, device=raw.device)[:, None, None, None],
        torch.arange(channels, device=raw.device)[None, :, None, None],
        ys[:, None, :, :],
        xs[:, None, :, :],
    ]
    flip = torch.rand(n, generator=generator, device=raw.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    mean, sd = NORMALIZATION[dataset]
    return (x - x.new_tensor(mean)[None, :, None, None]) / x.new_tensor(sd)[
        None, :, None, None
    ]


def prepare_cache(teacher, student, spec, device):
    training = get_dataset(spec.dataset, True)
    indices, development = split_indices(training.targets, seed=spec.data_seed)
    test = get_dataset(spec.dataset, False)
    raw = (
        torch.from_numpy(training.data[indices].copy()).permute(0, 3, 1, 2).contiguous()
    )
    labels = torch.tensor(training.targets)[torch.from_numpy(indices)]
    return {
        "hwa_raw": raw,
        "hwa_labels": labels,
        "hwa_cohort": cohort_receipt(indices),
        "training": cache_features(
            teacher, student, training, indices[: spec.recovery_images], device=device
        ),
        "development": cache_features(
            teacher,
            student,
            training,
            development[: spec.validation_images],
            device=device,
        ),
        "test": cache_features(teacher, student, test, range(len(test)), device=device),
        "reserved_development_cohort": cohort_receipt(development),
    }


def population_score(student, cache, spec, device, selected_cases):
    measurements = []
    for kind, ppm in selected_cases:
        draws = []
        for draw in range(spec.development_arrays):
            faults = PermanentFaults.sample(
                student.q.numel(), kind, ppm / 1e6, 71001 + draw, device
            )
            array = GaussianEndpointArray(student.q.detach(), faults, 72001 + draw)
            draws.append(evaluate(student, cache, device, q=array.read()))
        measurements.append(
            {
                "case": case_name(kind, ppm),
                "teacher_kl": sum(r["teacher_kl"] for r in draws) / len(draws),
                "accuracy_percent": sum(r["accuracy_percent"] for r in draws)
                / len(draws),
            }
        )
    return {
        "teacher_kl": sum(r["teacher_kl"] for r in measurements) / len(measurements),
        "cases": measurements,
    }


def fit_candidate(
    student, reference, pristine, cache, spec, store, device, candidate, selected_cases
):
    student.load_state_dict(pristine)
    student.q.requires_grad_(True)
    student.enable_calibration(True)
    parameters = [student.q] + student.calibration_parameters()
    optimizer = (
        torch.optim.SGD(parameters, lr=candidate["learning_rate"], momentum=0.9)
        if candidate["optimizer"] == "sgd"
        else torch.optim.Adam(parameters, lr=candidate["learning_rate"])
    )
    images = cache["hwa_raw"].to(device)
    labels = cache["hwa_labels"].to(device)
    train_rng = torch.Generator(device=device).manual_seed(spec.seed + 10000)
    data_rng = torch.Generator(device=device).manual_seed(spec.data_seed + 10000)
    history, best, rejection = [], None, None
    maximum = spec.hwa_epochs
    epoch, started = 0, time.monotonic()
    while epoch <= maximum:
        if epoch:
            student.train()
            order = torch.randperm(len(images), generator=data_rng, device=device)
            total_loss = 0.0
            for batch, start in enumerate(range(0, len(order), spec.hwa_batch_size)):
                idx = order[start : start + spec.hwa_batch_size]
                x = augment_images(images[idx], spec.dataset, data_rng)
                with torch.no_grad():
                    features = reference.features(x)
                    target = reference.forward_features(features).softmax(1)
                optimizer.zero_grad(set_to_none=True)
                q, _ = sampled_weights(
                    student.q,
                    train_rng,
                    candidate["noise_scale"],
                    candidate["fault_kind"],
                    candidate["fault_rate"],
                )
                logits = student.forward_features(features, q)
                loss = (
                    F.kl_div(logits.log_softmax(1), target, reduction="batchmean")
                    if candidate["objective"] == "teacher_kl"
                    else F.cross_entropy(logits, labels[idx])
                )
                if not bool(torch.isfinite(loss)):
                    rejection = {
                        "epoch": epoch,
                        "batch": batch,
                        "completed_epochs": epoch - 1,
                        "reason": "Nonfinite HWA/CDT objective.",
                    }
                    break
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    student.q.clamp_(-1, 1)
                total_loss += float(loss.detach()) * len(idx)
                if batch % 100 == 0:
                    store.append_metric(
                        {
                            "candidate": candidate["id"],
                            "epoch": epoch,
                            "batch": batch,
                            "loss": float(loss.detach()),
                        }
                    )
            if rejection is not None:
                break
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                rejection = {"epoch": epoch, "reason": "Nonfinite HWA/CDT parameters."}
                break
        if epoch == 0 or epoch % spec.validation_every == 0 or epoch == maximum:
            try:
                score = population_score(
                    student, cache["development"], spec, device, selected_cases
                )
                if not math.isfinite(score["teacher_kl"]):
                    raise FloatingPointError("Nonfinite HWA/CDT development KL.")
            except FloatingPointError as exc:
                if best is None:
                    raise
                rejection = {"epoch": epoch, "reason": str(exc)}
                break
            item = {
                "candidate": candidate["id"],
                "epoch": epoch,
                "elapsed_seconds": time.monotonic() - started,
                **score,
            }
            history.append(item)
            store.append_metric(item)
            print(
                json.dumps({k: v for k, v in item.items() if k != "cases"}), flush=True
            )
            if best is None or score["teacher_kl"] < best["score"]["teacher_kl"]:
                best = {
                    "model": clone_cpu(student.state_dict()),
                    "score": score,
                    "epoch": epoch,
                }
                save(
                    store.run_dir / f"checkpoints/candidates/{candidate['id']}.pt", best
                )
        if (
            epoch == maximum
            and maximum == spec.hwa_epochs
            and best["epoch"] >= 0.8 * maximum
        ):
            maximum += spec.extension_epochs
            store.append_metric(
                {
                    "candidate": candidate["id"],
                    "extended_to_epoch": maximum,
                    "reason": "best development checkpoint in final 20%",
                }
            )
        epoch += 1
    student.load_state_dict(best["model"])
    all_scores = population_score(
        student, cache["development"], spec, device, cases(spec)
    )
    outcome = {
        **candidate,
        "status": "rejected_nonfinite" if rejection else "complete",
        "rejection": rejection,
        "selected_epoch": best["epoch"],
        "epochs_run": (
            rejection.get("completed_epochs", epoch) if rejection else maximum
        ),
        "planned_epochs": maximum,
        "convergence_review_required": bool(rejection)
        or best["epoch"] >= 0.8 * maximum,
        "selection": best["score"],
        "all_cases": all_scores,
        "history": history,
    }
    if rejection:
        message = {
            "candidate": candidate["id"],
            "status": outcome["status"],
            **rejection,
        }
        store.append_metric(message)
        print(json.dumps(message), flush=True)
    return {"model": best["model"], "fit": outcome}


def score_for(bundle, selected_cases):
    names = {case_name(k, p) for k, p in selected_cases}
    scores = [
        r["teacher_kl"]
        for r in bundle["fit"]["all_cases"]["cases"]
        if r["case"] in names
    ]
    if len(scores) != len(names):
        raise ValueError("Missing development case in source selection.")
    return sum(scores) / len(scores)


def select_completed(candidates, selected_cases):
    eligible = [x for x in candidates if x["fit"]["status"] == "complete"]
    if not eligible:
        raise RuntimeError("All candidates in the declared HWA family were nonfinite.")
    return min(eligible, key=lambda x: score_for(x, selected_cases))


def fit_sources(student, reference, cache, spec, store, device, context):
    pristine = clone_cpu(student.state_dict())
    fits, nominal = [], []
    grid = [
        ("adam", objective, lr)
        for objective in spec.hwa_objectives
        for lr in spec.hwa_learning_rates
    ]
    grid += [("sgd", "cross_entropy", lr) for lr in spec.hwa_sgd_learning_rates]
    for optimizer_name, objective, lr in grid:
        candidate = dict(
            id=f"standard_{optimizer_name}_{objective}_{lr:g}",
            optimizer=optimizer_name,
            objective=objective,
            learning_rate=lr,
            noise_scale=1.0,
            fault_kind="open",
            fault_rate=0.0,
        )
        item = fit_candidate(
            student,
            reference,
            pristine,
            cache,
            spec,
            store,
            device,
            candidate,
            [("open", 0)],
        )
        fits.append(item["fit"])
        nominal.append(item)
    standard = select_completed(nominal, [("open", 0)])
    strong = [standard]
    for noise in spec.hwa_noise_scales[1:]:
        candidate = {
            k: standard["fit"][k]
            for k in (
                "optimizer",
                "objective",
                "learning_rate",
                "fault_kind",
                "fault_rate",
            )
        }
        candidate.update(id=f"noise_{noise:g}", noise_scale=noise)
        item = fit_candidate(
            student,
            reference,
            pristine,
            cache,
            spec,
            store,
            device,
            candidate,
            cases(spec),
        )
        fits.append(item["fit"])
        strong.append(item)
    noisy = select_completed(strong, cases(spec))
    sources = {
        "digital": {"model": pristine, "fit": None},
        "standard_hwa": standard,
        "noise_hwa": noisy,
    }
    for kind in spec.fault_kinds:
        generic = select_completed(strong, cases(spec, kind))
        candidates = []
        for rate in spec.cdt_rates:
            candidate = {
                k: generic["fit"][k]
                for k in ("optimizer", "objective", "learning_rate", "noise_scale")
            }
            candidate.update(
                id=f"cdt_{kind}_{rate:g}", fault_kind=kind, fault_rate=rate
            )
            item = fit_candidate(
                student,
                reference,
                pristine,
                cache,
                spec,
                store,
                device,
                candidate,
                cases(spec, kind),
            )
            fits.append(item["fit"])
            candidates.append(item)
        sources[f"cdt_{kind}"] = select_completed(candidates, cases(spec, kind))
    bundle = {
        "context": context,
        "sources": sources,
        "fits": fits,
        "selection": "minimum mean development teacher KL; final arrays and test images unused",
    }
    save(store.run_dir / "checkpoints/sources.pt", bundle)
    atomic_write_json(
        store.run_dir / "selection.json",
        {name: item["fit"] for name, item in sources.items()},
    )
    return {
        "stage": "fit",
        "candidates": len(fits),
        "completed_candidates": sum(x["status"] == "complete" for x in fits),
        "rejected_candidates": sum(x["status"] != "complete" for x in fits),
        "sources": {name: item["fit"] for name, item in sources.items()},
        "fits": fits,
        "training_images": len(cache["hwa_labels"]),
        "validation_images": len(cache["development"]["labels"]),
    }


def source_cases(spec, name):
    return cases(spec, name[4:]) if name.startswith("cdt_") else cases(spec)


def checked_sources(request, context):
    if request.weights is None:
        raise ValueError("Expected explicit --weights source-selection bundle.")
    bundle = load(request.weights)
    if "source_bundle" in bundle:
        bundle = bundle["source_bundle"]
    if bundle["context"] != context:
        raise ValueError("HWA source and original teacher/cache context differ.")
    expected = {
        "digital",
        "standard_hwa",
        "noise_hwa",
        "cdt_open",
        "cdt_gmax",
        "cdt_random",
    }
    if set(bundle["sources"]) != expected:
        raise ValueError("Missing HWA/CDT source.")
    for source in bundle["sources"].values():
        if not source["model"]:
            raise ValueError("Missing learned source tensors.")
    return bundle


def tune_recovery(student, cache, sources, spec, store, device, context):
    """Select global learning rates on independent development arrays only."""
    rows = []
    # Cover healthy and all severe faults; each source sees the same four cases.
    tasks = [("standard_hwa", "open", 0)] + [
        (f"cdt_{k}", k, 10000) for k in spec.fault_kinds
    ]
    for source_name, kind, ppm in tasks:
        state = sources[source_name]["model"]
        for draw in range(spec.tuning_arrays):
            student.load_state_dict(state)
            target = student.q.detach().clone()
            faults = PermanentFaults.sample(
                target.numel(), kind, ppm / 1e6, 73001 + draw, device
            )
            array = GaussianEndpointArray(target, faults, 74001 + draw)
            p0 = array.state_dict()
            initial = evaluate(student, cache["development"], device, q=array.read())
            for method, rates in (
                ("calibration", spec.calibration_learning_rates),
                ("onchip_weights", spec.recovery_learning_rates),
            ):
                for lr in rates:
                    student.load_state_dict(state)
                    array.load_state_dict(p0)
                    setting = replace(spec, learning_rate=lr, calibration_lr=lr)
                    value = adapt(
                        student,
                        array,
                        target,
                        cache["training"],
                        cache["development"],
                        setting,
                        method,
                        store,
                        f"tune_{source_name}_{ppm}_{draw}_{lr}",
                        device,
                    )
                    row = {
                        "source": source_name,
                        "kind": kind,
                        "ppm": ppm,
                        "array_seed": 73001 + draw,
                        "method": method,
                        "learning_rate": lr,
                        "initial_teacher_kl": initial["teacher_kl"],
                        "final_teacher_kl": value["final"]["teacher_kl"],
                        "ratio": value["final"]["teacher_kl"]
                        / max(initial["teacher_kl"], 1e-12),
                    }
                    rows.append(row)
                    store.append_metric(row)
                    print(json.dumps(row), flush=True)
    selected = {}
    for method, rates in (
        ("calibration", spec.calibration_learning_rates),
        ("onchip_weights", spec.recovery_learning_rates),
    ):
        scores = {
            lr: sum(
                r["ratio"]
                for r in rows
                if r["method"] == method and r["learning_rate"] == lr
            )
            / len(tasks)
            / spec.tuning_arrays
            for lr in rates
        }
        selected[method] = min(scores, key=lambda x: (scores[x], x))
    result = {
        "stage": "tune",
        "selected": selected,
        "measurements": rows,
        "criterion": "mean final/initial development KL, equally weighted over four cases and independent arrays",
    }
    return result


def screen_sources(request, student, cache, bundle, spec, store, device, context):
    tuning = load(request.weights)
    if tuning["context"] != context or "source_bundle" not in tuning:
        raise ValueError("Recovery selection does not belong to these HWA sources.")
    setting = replace(
        spec,
        learning_rate=tuning["selected"]["onchip_weights"],
        calibration_lr=tuning["selected"]["calibration"],
    )
    measurements = []
    prefix = student.prefix_hash()
    for name, source in bundle["sources"].items():
        state = source["model"]
        student.load_state_dict(state)
        target = student.q.detach().clone()
        clean = evaluate(student, cache["test"], device)
        for kind, ppm in source_cases(spec, name):
            case = case_name(kind, ppm)
            student.load_state_dict(state)
            faults = PermanentFaults.sample(
                target.numel(), kind, ppm / 1e6, spec.array_seed, device
            )
            array = GaussianEndpointArray(target, faults, spec.endpoint_seed)
            p0 = array.state_dict()
            initial = evaluate(student, cache["test"], device, q=array.read())
            directory = store.run_dir / "checkpoints" / name / case
            save(directory / "p0.pt", {"context": context, "model": state, "array": p0})
            p0_hash = sha256_file(directory / "p0.pt")
            identity = faults.receipt()
            for method in spec.methods:
                student.load_state_dict(state)
                array.load_state_dict(p0)
                result = adapt(
                    student,
                    array,
                    target,
                    cache["training"],
                    cache["test"],
                    setting,
                    method,
                    store,
                    f"{name}/{case}",
                    device,
                )
                if student.prefix_hash() != prefix or faults.receipt() != identity:
                    raise RuntimeError("Frozen prefix or physical fault map changed.")
                recovered = {
                    k: result.pop(k) for k in ("model", "master_target", "array")
                }
                if method != "none":
                    save(
                        directory / f"{method}.pt",
                        {"context": context, "p0_sha256": p0_hash, **recovered},
                    )
                row = {
                    "source": name,
                    "case": case,
                    "fault_kind": kind if ppm else "none",
                    "fault_rate_ppm": ppm,
                    "array_seed": spec.array_seed,
                    "endpoint_seed": spec.endpoint_seed,
                    "faults": identity,
                    "p0_sha256": p0_hash,
                    "clean": clean,
                    "initial": initial,
                    **result,
                    "teacher_kl_reduction": initial["teacher_kl"]
                    - result["final"]["teacher_kl"],
                }
                measurements.append(row)
                store.append_metric({"terminal_control": True, **row})
                atomic_write_json(store.run_dir / "measurements.json", measurements)
                print(
                    json.dumps(
                        {
                            "source": name,
                            "case": case,
                            "method": method,
                            "teacher_kl": row["final"]["teacher_kl"],
                        }
                    ),
                    flush=True,
                )
    return {
        "stage": "screen",
        "measurements": measurements,
        "completed_controls": len(measurements),
        "cases": len(measurements) // len(spec.methods),
        "evaluation_images": len(cache["test"]["labels"]),
        "recovery_images": len(cache["training"]["labels"]),
        "selected_recovery": tuning["selected"],
        "prefix_sha256": prefix,
    }


def run_train(request):
    spec = request.spec
    if not isinstance(spec, ComparisonSpec) or request.teacher_weights is None:
        raise ValueError("Expected HWA comparison settings and pinned digital teacher.")
    if any(
        getattr(request, k, None) is not None
        for k in ("resume", "base_weights", "device_model", "device_state")
    ):
        raise ValueError("Unexpected artifact for HWA comparison.")
    if spec.stage in ("cache", "fit") and request.weights is not None:
        raise ValueError("HWA fitting must start from the declared digital teacher.")
    inputs = [
        {
            "role": k,
            "path": str(getattr(request, k).resolve()),
            "sha256": sha256_file(getattr(request, k)),
        }
        for k in (
            "teacher_weights",
            "device_data",
            "weights",
            "base_weights",
            "selection_receipt",
        )
        if getattr(request, k, None) is not None
    ]
    receipt_path = os.environ.get("EBL_SOURCE_RECEIPT")
    if receipt_path:
        receipt = json.loads(Path(receipt_path).read_text())
        for name, digest in receipt["files"].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError(f"Source snapshot changed: {name}")
        inputs.append(
            {
                "role": "source_snapshot",
                "path": receipt_path,
                "sha256": sha256_file(Path(receipt_path)),
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
        reference = CrossbarSuffix(teacher, spec.suffix, spec.tile_size).requires_grad_(
            False
        )
        context = cache_context(spec, teacher_hash, student)
        atomic_write_json(store.run_dir / "mapping.json", student.mapping_receipt())
        if spec.stage == "cache":
            cache = prepare_cache(teacher, student, spec, device)
            clean = evaluate(student, cache["test"], device)
            if (
                clean["examples"] != 10000
                or abs(clean["accuracy_percent"] - SOURCES[spec.dataset][2]) > 0.25
            ):
                raise RuntimeError("Digital CIFAR audit failed.")
            save(store.run_dir / "checkpoints/cache.pt", {"context": context, **cache})
            result = {
                "stage": "cache",
                "clean": clean,
                "hwa_cohort": cache["hwa_cohort"],
                "recovery_cohort": cache["training"]["cohort"],
                "development_cohort": cache["development"]["cohort"],
            }
        else:
            if request.device_data is None:
                raise ValueError("Expected --device-data feature/image cache.")
            cache = load(request.device_data)
            if cache["context"] != context:
                raise ValueError("Teacher, mapping, or cache context mismatch.")
            if (
                len(cache["hwa_labels"]) != 45000
                or len(cache["test"]["labels"]) != 10000
                or len(cache["training"]["labels"]) != spec.recovery_images
                or len(cache["development"]["labels"]) != spec.validation_images
            ):
                raise ValueError(
                    "Cache has incorrect HWA/recovery/development/test coverage."
                )
            if spec.max_examples:
                cache = {
                    **cache,
                    **{
                        k: limited_cache(cache[k], spec.max_examples)
                        for k in ("training", "development", "test")
                    },
                    "hwa_raw": cache["hwa_raw"][: spec.max_examples],
                    "hwa_labels": cache["hwa_labels"][: spec.max_examples],
                }
            if spec.stage == "fit":
                result = fit_sources(
                    student, reference, cache, spec, store, device, context
                )
            else:
                bundle = checked_sources(request, context)
                if spec.stage == "tune":
                    result = tune_recovery(
                        student, cache, bundle["sources"], spec, store, device, context
                    )
                    save(
                        store.run_dir / "checkpoints/recovery.pt",
                        {
                            "context": context,
                            "source_sha256": sha256_file(request.weights),
                            "source_bundle": bundle,
                            **result,
                        },
                    )
                else:
                    result = screen_sources(
                        request, student, cache, bundle, spec, store, device, context
                    )
        result.update(
            dataset=spec.dataset,
            smoke_only=bool(spec.max_examples),
            evidence_class="exploratory_model_based",
            teacher_kl_definition="mean KL(p_teacher || p_student), nats at T=1",
            elapsed_seconds=time.monotonic() - started,
        )
        store.append_metric({"terminal": True, "stage": spec.stage})
        artifacts = [
            store.artifact_record(
                p, kind="checkpoint" if p.suffix == ".pt" else "analysis"
            )
            for p in store.run_dir.rglob("*")
            if p.is_file()
            and (
                p.suffix == ".pt"
                or p.name in ("mapping.json", "measurements.json", "selection.json")
            )
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
