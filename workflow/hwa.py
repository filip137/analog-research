"""(ii) Hardware-aware training of one source arm.

``method: none`` deploys the digital teacher unchanged. ``method: hwa`` trains
a digital FP32 master (normalized suffix weights plus the calibration
parameters) on sampled apparent hardware views -- an off-chip update, not a
persistent-device trajectory:

- noise awareness: ``noise_strength`` scales the declared endpoint model
  (analytic Gaussian PCM programming noise, or OM residual tables of the
  lifecycle's own P&V law) drawn fresh every minibatch with a
  straight-through gradient; 0 trains on the clean digital weights;
- corruption awareness: optional temporary Bernoulli failure masks at a rate
  drawn per minibatch from ``corruption.rates_ppm``. They never become a
  deployed fault map.

Checkpoints are selected by mean development teacher KL over the arm's
``selection_cases`` on independent selection arrays programmed with the
lifecycle's P&V. Final arrays and test images are never used.
"""

from __future__ import annotations

import json
import math
import time

import torch
from torch.nn import functional as F

from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.hwa_fault_runtime import sampled_weights
from experiments.cifar_crossbar.runtime import evaluate
from experiments.cifar_crossbar.sweep_devices import MixedOmSampler
from workflow import program_verify
from workflow.data import training_images


def selection_score(network, cache, lifecycle, populations, cases, device) -> dict:
    """Mean development metrics of the current master after real P&V."""

    rows = []
    target = network.q.detach()
    for case in cases:
        values = []
        for seed in lifecycle.devices.selection_seeds:
            array = program_verify.fresh(lifecycle, network.layout, seed, case, populations, device)
            program_verify.program(array, target, lifecycle)
            values.append(evaluate(network, cache, device, q=array.read()))
        rows.append(
            {
                "case": case.label,
                "teacher_kl": sum(v["teacher_kl"] for v in values) / len(values),
                "accuracy_percent": sum(v["accuracy_percent"] for v in values) / len(values),
            }
        )
    return {"teacher_kl": sum(r["teacher_kl"] for r in rows) / len(rows), "cases": rows}


def endpoint_sampler(lifecycle, arm, device_bundle, device):
    """Per-minibatch apparent-weight sampler of the declared noise model."""

    seed = lifecycle.hwa.seed + 10000
    if lifecycle.hwa.noise_model == "pcm_gaussian_endpoint":
        generator = torch.Generator(device=device).manual_seed(seed)

        def sample(q, kind, rate):
            return sampled_weights(q, generator, arm.noise_strength, kind, rate)[0]

        return sample
    kernels = device_bundle["kernels"]
    if not kernels["adequate"]:
        raise ValueError("Expected an adequate characterized OM endpoint kernel.")
    sampler = MixedOmSampler(kernels["tables"]["fit"], seed, device)

    def sample(q, kind, rate):
        return sampler.perturb(q, kind, rate, arm.noise_strength)

    return sample


def train_source(*, lifecycle, arm, network, teacher, cache, device_bundle, store, device) -> dict:
    """Return the selected source state and its fit record for one HWA arm."""

    pristine = clone_cpu(network.state_dict())
    if not arm.trains:
        return {"model": pristine, "fit": {"kind": "digital_teacher", "method": "none", "status": "complete"}}
    hwa, data, dataset = lifecycle.hwa, lifecycle.data, lifecycle.network.dataset
    populations = device_bundle.get("populations")
    network.enable_calibration(True)
    network.q.requires_grad_(True)
    parameters = [network.q] + network.calibration_parameters()
    optimizer = torch.optim.Adam(parameters, lr=hwa.learning_rate)
    sample = endpoint_sampler(lifecycle, arm, device_bundle, device)
    rate_generator = torch.Generator().manual_seed(hwa.seed + 20000)
    rates = arm.corruption.rates_ppm if arm.corruption else ()
    histogram = {str(rate): 0 for rate in rates}
    raw = cache["hwa_raw"].to(device)
    labels = cache["hwa_labels"].to(device)
    prefix = network.prefix_hash()
    best, history, rejected = None, [], None
    limit, epoch, started = hwa.epochs, 0, time.monotonic()
    while epoch <= limit:
        if epoch:
            for group in optimizer.param_groups:
                group["lr"] = hwa.learning_rate * hwa.lr_factor(epoch)
            data_rng = torch.Generator(device=device).manual_seed(data.data_seed + epoch)
            order = torch.randperm(len(raw), generator=data_rng, device=device)
            network.train()
            for batch, begin in enumerate(range(0, len(raw), hwa.batch_size)):
                index = order[begin : begin + hwa.batch_size]
                x = training_images(raw[index], dataset, data_rng, hwa.augment)
                with torch.no_grad():
                    target = teacher(x).softmax(1) if hwa.objective == "teacher_kl" else None
                    features = network.features(x)
                kind, rate = "open", 0.0
                if arm.corruption:
                    ppm = rates[int(torch.randint(len(rates), (), generator=rate_generator))]
                    histogram[str(ppm)] += 1
                    kind, rate = arm.corruption.kind, ppm / 1e6
                q = sample(network.q, kind, rate)
                optimizer.zero_grad(set_to_none=True)
                logits = network.forward_features(features, q)
                loss = (
                    F.kl_div(logits.log_softmax(1), target, reduction="batchmean")
                    if target is not None
                    else F.cross_entropy(logits, labels[index])
                )
                if not bool(torch.isfinite(loss)):
                    rejected = {"epoch": epoch, "batch": batch, "reason": "nonfinite HWA objective"}
                    break
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    network.q.clamp_(-1, 1)
                if batch % 100 == 0:
                    store.append_metric(
                        {"stage": "hwa", "arm": arm.arm_id, "epoch": epoch, "batch": batch, "loss": float(loss.detach())}
                    )
            if rejected:
                break
            if any(not bool(torch.isfinite(p).all()) for p in network.parameters()):
                rejected = {"epoch": epoch, "reason": "nonfinite parameters"}
                break
        if epoch == 0 or epoch % hwa.selection_every == 0 or epoch == limit:
            score = selection_score(network, cache["development"], lifecycle, populations, arm.selection_cases, device)
            if not math.isfinite(score["teacher_kl"]):
                rejected = {"epoch": epoch, "reason": "nonfinite selection score"}
                break
            item = {"epoch": epoch, "elapsed_seconds": time.monotonic() - started, **score}
            history.append(item)
            store.append_metric({"stage": "hwa_selection", "arm": arm.arm_id, **item})
            print(json.dumps({"arm": arm.arm_id, "epoch": epoch, "teacher_kl": score["teacher_kl"]}), flush=True)
            if best is None or score["teacher_kl"] < best["score"]:
                best = {"epoch": epoch, "score": score["teacher_kl"], "model": clone_cpu(network.state_dict())}
        if (
            epoch == hwa.epochs
            and limit == hwa.epochs
            and hwa.extension_epochs
            and best is not None
            and best["epoch"] >= 0.8 * limit
        ):
            limit += hwa.extension_epochs
            store.append_metric({"stage": "hwa", "arm": arm.arm_id, "extended_to_epoch": limit})
        epoch += 1
    if network.prefix_hash() != prefix:
        raise RuntimeError("HWA changed the frozen digital prefix.")
    if best is None:
        raise FloatingPointError(f"HWA arm {arm.arm_id!r} produced no finite selection checkpoint.")
    fit = {
        "kind": "hwa",
        "method": arm.method,
        "noise_strength": arm.noise_strength,
        "corruption": None if arm.corruption is None else arm.corruption.to_dict(),
        "status": "rejected_nonfinite" if rejected else "complete",
        "rejection": rejected,
        "selected_epoch": best["epoch"],
        "development_kl": best["score"],
        "epochs_run": min(epoch - 1, limit),
        "maximum_epochs": limit,
        "convergence_review_required": bool(rejected) or best["epoch"] >= 0.8 * limit,
        "rate_histogram": histogram,
        "history": history,
        "selection": "minimum mean development teacher KL over selection cases and selection arrays",
    }
    return {"model": best["model"], "fit": fit}
