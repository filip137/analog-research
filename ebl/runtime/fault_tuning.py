"""Native ``fault_tuning`` stage: the fault-recovery LR-tuning screen (see ``methods.fault_tuning``).

Without a pinned ``fault_tuning.deployment`` the stage programs the assignment
(naive P&V, then the fault-aware pair rewrite, plus the analog-LoRA tiles when
declared) and saves every state in portable tensor form. With one, it loads that
file (resolved under ``EBL_CORPUS_ROOT``, pinned by SHA-256) and checks every
apparent-state hash, so arms on any GPU start from the identical deployment.
"""
from __future__ import annotations

import json
import hashlib
import math
from dataclasses import replace
import os
from pathlib import Path
import time

import torch

from core.artifacts import atomic_write_json, sha256_file
from core.serialization import load, save
from core.tensor_identity import tensor_hash
import hardware.arrays as devices
import methods.fault_tuning as ft
import methods.programming.lifecycle as program_verify
from methods.onchip import IdealizedWriteAdam, ThresholdedWriteAdam


def pinned_path(entry) -> Path:
    root = os.environ.get("EBL_CORPUS_ROOT")
    if not root:
        raise ValueError("Expected EBL_CORPUS_ROOT to hold the pinned fault-tuning inputs.")
    path = Path(root) / f"{entry['name']}.pt"
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise ValueError(f"Expected {entry['name']} with SHA-256 {entry['sha256']}. Provided value: {path} {digest}.")
    return path


def pinned_inputs(lifecycle) -> list[dict]:
    cfg = lifecycle.fault_tuning
    entries = [("fault_deployment", cfg.deployment)]
    if cfg.analog_lora is not None:
        entries.append(("analog_lora_b_scale", cfg.analog_lora["b_scale"]))
    inputs = [{"role": role, "path": str(pinned_path(e)), "sha256": e["sha256"]}
              for role, e in entries if e is not None]
    if cfg.evaluation_cohorts:
        import networks.opt as opt
        for cohort in cfg.evaluation_cohorts:
            data = replace(lifecycle.data, corpus=cohort["corpus"], corpus_sha256=cohort["corpus_sha256"])
            inputs.append({"role": f"evaluation_corpus.{cohort['name']}", "path": str(opt.corpus_path(data)),
                           "sha256": cohort["corpus_sha256"]})
    return inputs


class Context:
    def __init__(self, stage):
        self.stage, self.lc, self.cfg = stage, stage.lifecycle, stage.lifecycle.fault_tuning
        self.network, self.device = stage.network, stage.device
        self.seed = stage.selection.array_seed
        self.case = self.lc.defects.cases[0]
        self.layout = list(self.network.layout)
        self.fisher = None
        spec = self.cfg.deployment_mapping()
        if spec is None:
            self.mapping, self.readout = None, lambda array: array.read()
        else:
            import methods.drift_tuning as dt
            self.mapping = dt.Mapping(spec)
            self.readout = self.mapping.read_array

    def fresh(self, layout=None):
        layout = self.layout if layout is None else layout
        if layout is self.layout:
            return program_verify.fresh(self.lc, layout, self.seed, self.case, None, self.device)
        from networks.mapping import fault_domains
        lc = self.lc
        return devices.fresh_array(
            technology="cmo", layout=layout, assignment_seed=self.seed, case=self.case, populations=None,
            device=self.device, namespace=lc.network.binding_namespace, endpoint_seed=lc.devices.endpoint_seed(self.seed),
            noise_scale=lc.devices.noise_scale, tile_size=lc.deployment.tile_size,
            arrays_per_die=lc.defects.arrays_per_die, acceptance_percent=lc.program_verify.acceptance_percent,
            domains=fault_domains(layout))

    def array(self, key):
        entry = self.deployed[key]
        array = self.fresh(self.adapter_layout if key == "adapter" else None)
        devices.load_array_state(array, entry["state"], "cmo")
        if tensor_hash(self.readout(array)) != entry["apparent_sha256"]:
            raise RuntimeError(f"Expected the exact deployed {key} state.")
        return array

    def evaluate(self, q):
        return self.stage.family.evaluate(self.network, {"tokens": self.development}, self.device, q=q)


def read(ctx, array):
    """Normalized weights of the held base array through the deployment mapping's readout.

    Every device-facing forward of the non-drift path reads through this: the standard pair
    reads ``array.read()`` (bit for bit), a two-sided deployment ``(G0 - G1) / 79.2``.
    """
    readout = getattr(ctx, "readout", None)
    return array.read() if readout is None else readout(array)


def prepare_evaluation_cohorts(ctx):
    """Verify every held-out corpus/cohort before targets, deployment or training."""
    ctx.evaluation_cohorts = []
    for spec in getattr(ctx.cfg, "evaluation_cohorts", ()):
        settings = {k: v for k, v in spec.items() if k != "name"}
        tokens = ft.cohort(ctx.lc, {**settings, "seed": None})
        ctx.evaluation_cohorts.append((dict(spec), tokens))
        ctx.stage.store.append_metric({"stage": "fault_tuning_evaluation_cohort_verified", **spec})


def _state_digest(value):
    """Fingerprint nested physical state, including counters and device RNG."""
    digest = hashlib.sha256()
    def update(item):
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor" + tensor_hash(item).encode())
        elif isinstance(item, dict):
            digest.update(b"dict[")
            for key in sorted(item):
                update(key)
                update(item[key])
            digest.update(b"]")
        elif isinstance(item, (list, tuple)):
            digest.update(b"list[")
            for part in item:
                update(part)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, allow_nan=False, sort_keys=True).encode() + b"\0")
    update(value)
    return digest.hexdigest()


def _physical_snapshot(array):
    # Hash one matrix at a time: a full twelve-MLP state_dict would clone every
    # conductance/write-count tensor onto CPU simultaneously.
    children = getattr(array, "children", None)
    state = ([_state_digest(child.state_dict()) for child in children]
             if children is not None else _state_digest(array.state_dict()))
    return {"state_sha256": _state_digest(state), "apparent_sha256": tensor_hash(array.read()),
            "cost": dict(array.cost())}


@torch.no_grad()
def evaluate_held_out(ctx, arm, row, base, adapter=None):
    """Score the final live learner, retaining all corrections until after scoring."""
    cohorts = getattr(ctx, "evaluation_cohorts", ())
    if not cohorts:
        return
    arrays = {"base": base, **({"adapter": adapter} if adapter is not None else {})}
    before = {name: _physical_snapshot(array) for name, array in arrays.items()}
    correction = ctx.network.fault_output_calibration
    modules = {"lora": None if ctx.network.lora is None else ctx.network.lora.state_dict(),
               "calibration": None if correction is None else correction.state_dict()}
    correction_hash = _state_digest(modules)
    scores = []
    for spec, tokens in cohorts:
        started = time.monotonic()
        score = ctx.stage.family.evaluate(ctx.network, {"tokens": tokens}, ctx.device, q=read(ctx, base))
        if any(isinstance(value, (float, int)) and not math.isfinite(value) for value in score.values()):
            raise FloatingPointError("Nonfinite held-out evaluation score.")
        after = {name: _physical_snapshot(array) for name, array in arrays.items()}
        current_correction = ctx.network.fault_output_calibration
        current = {"lora": None if ctx.network.lora is None else ctx.network.lora.state_dict(),
                   "calibration": None if current_correction is None else current_correction.state_dict()}
        if before != after or correction_hash != _state_digest(current):
            raise RuntimeError("Held-out evaluation changed physical state, cost counters or corrections.")
        point = {**spec, "scores": score, "seconds": time.monotonic() - started,
                 "physical_state": after, "correction_sha256": correction_hash, "state_unchanged": True}
        scores.append(point)
        ctx.stage.store.append_metric({"stage": "fault_tuning_held_out", "arm": arm.label, **point})
    row["held_out"] = scores


def deploy_mapped(ctx, target):
    """``deploy`` for a mapping (``mapping`` or ``drift.mapping``) other than the standard pair (two-sided).

    Naive Gaussian-endpoint P&V of the mapped (2, size) conductance targets from each
    matrix's own stream, then the same verify-based pair rewrite with the mapping's expected
    targets and span. Apparent hashes are of the mapping's readout.
    """
    lc, cfg, mapping = ctx.lc, ctx.cfg, ctx.mapping
    rewrite = cfg.rewrite()
    array = ctx.fresh()
    targets_us = mapping.targets(target)
    before = array.cost()
    array.program_conductances(targets_us)
    after = array.cost()
    apparent = mapping.read_array(array)
    naive_report = {"cycles": 0, "verify_reads": 0, **{k: after[k] - before[k] for k in after},
                    "acceptance_percent": lc.program_verify.acceptance_percent,
                    "out_of_range_devices": array.out_of_range(), "method": lc.program_verify.method,
                    "relaxation": lc.program_verify.relaxation.model, "mapping": dict(mapping.spec),
                    "apparent_sha256": tensor_hash(apparent),
                    "apparent_residual_rms": float((target - apparent).square().mean().sqrt())}
    states = {"naive": {"state": devices.array_state(array, "cmo", portable=True), "report": naive_report,
                        "apparent_sha256": naive_report["apparent_sha256"]}}
    identity = devices.array_identity(array, "cmo")
    report = program_verify.fault_rewrite(array, target, lc, rewrite, ctx.seed, expected_us=targets_us,
                                          span_us=mapping.span, readout=mapping.read_array)
    states["fault_aware"] = {"state": array.state_dict(), "report": report,
                             "apparent_sha256": report["rewritten_apparent_sha256"]}
    return {"schema": "fault_tuning.deployment.v1", "assignment_seed": ctx.seed, "case": ctx.case.to_dict(),
            "target_sha256": tensor_hash(target), "identity": identity, "states": states, "adapter": None,
            "mapping": dict(mapping.spec),
            "gpu": torch.cuda.get_device_name(ctx.device) if ctx.device.type == "cuda" else "cpu",
            "torch": str(torch.__version__)}


def deploy(ctx, target):
    """Naive P&V, fault-aware pair rewrite and (if declared) the analog-LoRA tiles; portable states."""
    lc, cfg = ctx.lc, ctx.cfg
    if ctx.mapping is not None and ctx.mapping.kind != "reference":
        if cfg.analog_lora is not None:
            raise ValueError("Analog-LoRA tiles use the standard pair mapping.")
        return deploy_mapped(ctx, target)
    rewrite = cfg.rewrite()
    array = ctx.fresh()
    naive_report = program_verify.program(array, target, lc)
    states = {"naive": {"state": devices.array_state(array, "cmo", portable=True), "report": naive_report,
                        "apparent_sha256": naive_report["apparent_sha256"]}}
    identity = devices.array_identity(array, "cmo")
    report = program_verify.fault_rewrite(array, target, lc, rewrite, ctx.seed)
    states["fault_aware"] = {"state": array.state_dict(), "report": report,
                             "apparent_sha256": report["rewritten_apparent_sha256"]}
    del array
    adapter = None
    if cfg.analog_lora is not None:
        rank = cfg.analog_lora["rank"]
        adapter_target, a_scales = ft.adapter_targets(ctx.layout, rank, cfg.lora_seed)
        adapter_target = adapter_target.to(ctx.device)
        array = ctx.fresh(ctx.adapter_layout)
        programmed = program_verify.program(array, adapter_target, lc)
        rewritten = program_verify.fault_rewrite(array, adapter_target, lc, rewrite, ctx.seed)
        states["adapter"] = {"state": array.state_dict(), "report": {"program": programmed, "rewrite": rewritten},
                             "apparent_sha256": rewritten["rewritten_apparent_sha256"]}
        adapter = {"rank": rank, "lora_seed": cfg.lora_seed, "a_scales": a_scales,
                   "target_sha256": tensor_hash(adapter_target), "identity": devices.array_identity(array, "cmo")}
    return {"schema": "fault_tuning.deployment.v1", "assignment_seed": ctx.seed, "case": ctx.case.to_dict(),
            "target_sha256": tensor_hash(target), "identity": identity, "states": states, "adapter": adapter,
            "gpu": torch.cuda.get_device_name(ctx.device) if ctx.device.type == "cuda" else "cpu",
            "torch": str(torch.__version__)}


@torch.no_grad()
def device_statistics(ctx, target):
    """Physical statistics of the deployed naive and fault-aware states (analysis side).

    Fractions are of all physical devices (two per weight). ``target_*`` describe the mapped
    conductance targets; ``below_floor``/``above_max`` count healthy devices whose unclamped
    programmed conductance lies below 9 uS or above 88.2 uS; ``stuck`` counts permanent faults.
    """
    from hardware.devices.cmo import G_MAX_US, G_MIN_US
    import methods.drift_tuning as dt
    mapping = ctx.mapping or dt.Mapping()
    expected = mapping.targets(target)
    total = expected.numel()
    tolerance = 1e-4
    out = {"devices": total,
           "target_at_floor_fraction": int((expected <= G_MIN_US + tolerance).sum()) / total,
           "target_at_max_fraction": int((expected >= G_MAX_US - tolerance).sum()) / total}
    del expected
    for name in ("naive", "fault_aware"):
        array = ctx.array(name)
        g = torch.cat([child.g for child in array.children], 1)
        mask, _ = devices.hidden_faults(array)
        healthy = ~mask
        q = read(ctx, array)
        out[name] = {"stuck_fraction": int(mask.sum()) / total,
                     "below_floor_fraction": int(((g < G_MIN_US) & healthy).sum()) / total,
                     "above_max_fraction": int(((g > G_MAX_US) & healthy).sum()) / total,
                     "apparent_residual_rms": float((q - target).square().mean().sqrt()),
                     "apparent_max_abs_error": float((q - target).abs().max())}
        del array, g, mask, healthy, q
    report = ctx.deployed["fault_aware"]["report"]
    out["rewrite"] = {**report["controller"], "analysis": report["analysis"], "cost": report["cost"],
                      "out_of_range_devices": report["out_of_range_devices"]}
    return out


def deployment_records(ctx, target, clean):
    """Clean references and device statistics of a declared (non-drift) deployment mapping."""
    import methods.drift_tuning as dt
    mapping = ctx.mapping or dt.Mapping()
    clean_mapped = mapping.clean(target)
    return {"mapping": None if mapping.spec is None else dict(mapping.spec), "span_us_per_unit_q": mapping.span,
            "source_ideal_teacher_kl": clean["teacher_kl"],
            "clean_mapped_readout_teacher_kl": ctx.evaluate(clean_mapped)["teacher_kl"],
            "clean_mapped_readout_max_abs_error": float((clean_mapped - target).abs().max()),
            "device_statistics": device_statistics(ctx, target)}


def check_deployment(ctx, bundle, target):
    if (bundle.get("schema") != "fault_tuning.deployment.v1" or bundle["assignment_seed"] != ctx.seed
            or bundle["case"] != ctx.case.to_dict() or bundle["target_sha256"] != tensor_hash(target)):
        raise ValueError("Expected the pinned deployment of this source, assignment and defect case.")
    mapped = ctx.mapping is not None and ctx.mapping.kind != "reference"
    if bundle.get("mapping") != (dict(ctx.mapping.spec) if mapped else None):
        raise ValueError("Expected a deployment of the declared mapping.")
    if (ctx.cfg.analog_lora is not None) != (bundle["adapter"] is not None):
        raise ValueError("Expected adapter tiles exactly when analog LoRA is declared.")
    if bundle["adapter"] is not None and (bundle["adapter"]["rank"] != ctx.cfg.analog_lora["rank"]
                                          or bundle["adapter"]["lora_seed"] != ctx.cfg.lora_seed):
        raise ValueError("Expected adapter tiles of the declared rank and seed.")


def compute_fisher(ctx):
    """Mean squared batch gradient of the training loss over the whole recovery set, fault-aware state."""
    network, cfg = ctx.network, ctx.cfg
    base = ctx.array("fault_aware")
    q0 = base.read() if getattr(ctx, "mapping", None) is None else ctx.readout(base)
    total, batches = torch.zeros_like(q0), 0
    batch = cfg.fisher["batch"]
    network.train()
    for begin in range(0, len(ctx.recovery), batch):
        tokens = ctx.recovery[begin:begin + batch].to(ctx.device)
        q = q0.clone().requires_grad_(True)
        loss = ft.sample_loss(network.forward_features(network.features(tokens), q), ctx.samples[begin:begin + batch])
        loss.backward()
        total.add_(q.grad.square())
        batches += 1
    return total / batches


def _lora(network, rank, learning_rate, seed):
    from methods.config import DigitalLora
    network.enable_lora(DigitalLora(rank, float(rank), learning_rate), seed)
    parameters = list(network.lora.parameters())
    for p in parameters:
        p.requires_grad_(True)
    return parameters


def _set_adapter(network, q, layout, a_scales, b_scales):
    with torch.no_grad():
        for k, module in enumerate(network.lora):
            a, b = layout[2 * k], layout[2 * k + 1]
            module["a"].copy_(q[a.offset:a.offset + a.size].view(a.shape) * a_scales[k][:, None])
            module["b"].copy_(q[b.offset:b.offset + b.size].view(b.shape) * b_scales[k][:, None])


def _adapter_gradient(network, a_scales, b_scales):
    parts = []
    for k, module in enumerate(network.lora):
        parts += [(module["a"].grad * a_scales[k][:, None]).flatten(), (module["b"].grad * b_scales[k][:, None]).flatten()]
    return torch.cat(parts)


def _cost(array, baseline, baseline_writes):
    now = array.cost()
    cost = {key: now[key] - baseline[key] for key in now}
    writes = array.write_counts() - baseline_writes
    cost.update(healthy_device_writes=cost["device_writes"], touched_devices=int((writes > 0).sum()),
                max_writes_per_device=int(writes.max()), out_of_range_devices=array.out_of_range())
    return cost, writes


def run_arm(ctx, arm):
    network, cfg, device = ctx.network, ctx.cfg, ctx.device
    schedule, law = cfg.schedule, cfg.write_law
    started = time.monotonic()
    network.enable_lora(None)
    network.fault_output_calibration = None
    base = ctx.array("naive" if arm.method == "hold_naive" else "fault_aware")
    held = read(ctx, base)
    initial = ctx.evaluate(held)
    row = {"label": arm.label, "method": arm.method, "learning_rate": arm.learning_rate, "rank": arm.rank,
           "reserve_p": arm.reserve_p, "initial_apparent_sha256": tensor_hash(held), "initial": initial}
    if arm.method not in ft.TRAINS:
        row.update(final=initial, curve=[], cost={}, final_apparent_sha256=row["initial_apparent_sha256"],
                   seconds=time.monotonic() - started)
        evaluate_held_out(ctx, arm, row, base)
        if "held_out" in row:
            row["seconds"] = time.monotonic() - started
        return row, None
    acceptance, noise = ctx.lc.program_verify.acceptance_percent, ctx.lc.devices.noise_scale

    def thresholded(array, mask=None, write_noise_scale=None):
        return ThresholdedWriteAdam(array, learning_rate=arm.learning_rate, threshold_sigma=law["threshold_sigma"],
                                    accumulate_updates=law["accumulate_updates"], acceptance_percent=acceptance,
                                    noise_scale=noise, max_write_attempts_per_device=None, write_mask=mask,
                                    write_noise_scale=write_noise_scale)

    writer = optimizer = scheduler = adapter = mask = None
    tracked = base
    if arm.method == "onchip_noisy":
        writer = thresholded(base)
    elif arm.method == "matched_noiseless":
        writer = thresholded(base, write_noise_scale=0.0)
    elif arm.method == "onchip_noiseless":
        writer = IdealizedWriteAdam(base, learning_rate=arm.learning_rate, write_noise_scale=0.0)
    elif arm.method == "child_tuning_d":
        if ctx.fisher is None:
            t0 = time.monotonic()
            ctx.fisher = compute_fisher(ctx)
            ctx.fisher_seconds = time.monotonic() - t0
        mask = ft.child_mask(ctx.fisher, ctx.layout, arm.reserve_p)
        writer = thresholded(base, mask)
    elif arm.method == "calibration":
        network.requires_grad_(False)
        network.fault_output_calibration = ft.OutputCalibration(network)
        optimizer = torch.optim.Adam(network.fault_output_calibration.parameters(),
                                     lr=arm.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, schedule["steps"])
    elif arm.method == "digital_lora":
        parameters = _lora(network, arm.rank, arm.learning_rate, cfg.lora_seed)
        optimizer = torch.optim.Adam(parameters, lr=arm.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, schedule["steps"])
    else:  # analog_lora
        _lora(network, arm.rank, arm.learning_rate, cfg.lora_seed)
        tracked = adapter = ctx.array("adapter")
        writer = thresholded(adapter)
    baseline, baseline_writes = tracked.cost(), tracked.write_counts().clone()
    base_hash = tensor_hash(held)
    generator = torch.Generator().manual_seed(schedule["order_seed"])
    steps, every = schedule["steps"], schedule["development_every_steps"]
    losses, curve = [], []
    for step in range(steps):
        index = torch.randint(0, len(ctx.recovery), (schedule["batch"],), generator=generator)
        tokens, drawn = ctx.recovery[index].to(device), ctx.samples[index.to(device)]
        network.train()
        if arm.method in ft.WRITES:
            q = read(ctx, base).requires_grad_(True)
        else:
            q = held
            if adapter is not None:
                _set_adapter(network, adapter.read(), ctx.adapter_layout, ctx.a_scales, ctx.b_scales)
            if arm.method == "calibration":
                optimizer.zero_grad(set_to_none=True)
            else:
                for p in network.lora.parameters():
                    p.grad = None
        loss = ft.sample_loss(network.forward_features(network.features(tokens), q), drawn)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite training loss in arm {arm.label}.")
        loss.backward()
        if optimizer is not None:
            optimizer.step()
            scheduler.step()
        else:
            writer.learning_rate = ft.cosine(arm.learning_rate, step, steps)
            writer.step(q.grad if adapter is None else _adapter_gradient(network, ctx.a_scales, ctx.b_scales))
        losses.append(float(loss.detach()))
        if (step + 1) % every == 0 or step + 1 == steps:
            if adapter is not None:
                _set_adapter(network, adapter.read(), ctx.adapter_layout, ctx.a_scales, ctx.b_scales)
            point = {"step": step + 1, "train_loss": losses[-1],
                     "development": ctx.evaluate(read(ctx, base) if arm.method in ft.WRITES else held)}
            if writer is not None:
                point["cost"] = _cost(tracked, baseline, baseline_writes)[0]
            curve.append(point)
            ctx.stage.store.append_metric({"stage": "fault_tuning_development", "arm": arm.label, **point})
            print(json.dumps({"arm": arm.label, "step": step + 1,
                              "teacher_kl": point["development"]["teacher_kl"]}), flush=True)
    base.verify_permanence()
    final = curve[-1]["development"]
    row.update(final=final, curve=curve, train_loss_first10=sum(losses[:10]) / 10,
               train_loss_last10=sum(losses[-10:]) / 10, train_losses=losses)
    state = None
    if writer is not None:
        cost, writes = _cost(tracked, baseline, baseline_writes)
        if isinstance(writer, ThresholdedWriteAdam):
            cost["writer"] = writer.diagnostics()
        if mask is not None:
            cost.update(mask_weights=int(mask.sum()), mask_sha256=tensor_hash(mask),
                        writes_outside_mask=int(writes[~mask].sum()))
            if cost["writes_outside_mask"]:
                raise RuntimeError("Child-Tuning_D wrote outside its fixed mask.")
        row["cost"] = cost
    else:
        row["cost"] = {"healthy_device_writes": 0, "stuck_write_attempts": 0, "touched_devices": 0,
                       "max_writes_per_device": 0, "out_of_range_devices": base.out_of_range()}
    if arm.method == "calibration":
        if tensor_hash(read(ctx, base)) != base_hash or base.cost() != ctx.deployed_cost:
            raise RuntimeError("Calibration changed the base deployment.")
        correction = network.fault_output_calibration
        row["calibration_parameters"] = sum(p.numel() for p in correction.parameters())
        state = {"schema": "fault_tuning.sampled_output_calibration.v1",
                 "placement": "complete_mlp_output_before_residual",
                 "state": {k: v.detach().cpu().clone() for k, v in correction.state_dict().items()}}
    if arm.method in ft.LORA:
        if tensor_hash(read(ctx, base)) != base_hash or base.cost() != ctx.deployed_cost:
            raise RuntimeError("A LoRA arm changed the base deployment.")
        row["lora_parameters"] = sum(p.numel() for p in network.lora.parameters())
        state = {"lora": {k: v.detach().cpu().clone() for k, v in network.lora.state_dict().items()},
                 "rank": arm.rank, "alpha": float(arm.rank), "seed": cfg.lora_seed}
        if adapter is not None:
            adapter.verify_permanence()
            state["adapter_array"] = adapter.state_dict()
            state["final_adapter_apparent_sha256"] = tensor_hash(adapter.read())
            row["final_adapter_apparent_sha256"] = state["final_adapter_apparent_sha256"]
    evaluate_held_out(ctx, arm, row, base, adapter)
    row.update(final_apparent_sha256=tensor_hash(read(ctx, base)), seconds=time.monotonic() - started)
    network.enable_lora(None)
    network.fault_output_calibration = None
    return row, state


def run_drift_arm(ctx, arm, refs):
    """One arm under ``fault_tuning.drift`` (AIHWKit CMO log-time law).

    Every device-facing forward -- gradients, development, grid and projections -- reads the
    held devices through the drift law at its age (``methods.drift_tuning.DriftPlant``);
    ``no_drift`` arms read the held state. ``hold_fault_aware`` is the untreated deployment
    read on the grid (A0; with ``no_drift`` the release-state twin). Learners start at release
    and update k reads at k * update_seconds; equal ages give identical draws in every arm
    (paired). Writing learners (``onchip_noisy``, ``child_tuning_d``) write through the plant
    (``reset_absorb``) or the array (``continues``, the drift token-budget learner, unchanged);
    ``calibration``, ``layernorm`` and ``digital_lora`` learn digital parameters on the drifted
    read, write nothing and are frozen after the window.
    """
    import methods.drift_tuning as dt
    network, cfg, device, d = ctx.network, ctx.cfg, ctx.device, ctx.cfg.drift
    schedule, law = cfg.schedule, cfg.write_law
    started = time.monotonic()
    network.enable_lora(None)
    network.fault_output_calibration = None
    norms = {k: v.detach().clone() for k, v in network.mlp_norms.state_dict().items()}
    norms_hash = _state_digest(norms)
    base = ctx.array("fault_aware")
    plant = dt.DriftPlant(base, d, ctx.mapping or dt.Mapping(), drift=not arm.no_drift)
    legacy = d.write_clock == "continues"
    start_hash = tensor_hash(plant.read(None))

    def evaluate(seconds, step, kind="grid"):
        metrics = ctx.evaluate(plant.read(seconds))
        point = {"arm": arm.label, "kind": kind, "age_seconds": seconds, "step": step,
                 "teacher_kl": metrics["teacher_kl"], "development": metrics,
                 "bounds": None if seconds is None or arm.no_drift else plant.bounds(seconds)}
        ctx.stage.store.append_metric({"stage": "drift_tuning_evaluation", **point})
        print(json.dumps({"arm": arm.label, "kind": kind, "age_seconds": seconds, "step": step,
                          "teacher_kl": metrics["teacher_kl"]}), flush=True)
        return point

    row = {"label": arm.label, "method": arm.method, "learning_rate": arm.learning_rate, "rank": arm.rank,
           "reserve_p": arm.reserve_p, "initial_apparent_sha256": start_hash}
    if arm.no_drift:
        row["no_drift"] = True
    no_writes = {"healthy_device_writes": 0, "device_writes": 0, "stuck_write_attempts": 0, "touched_devices": 0,
                 "max_writes_per_device": 0, "out_of_range_devices": base.out_of_range()}
    if arm.method == "hold_fault_aware":
        grid = [evaluate(t, None) for t in d.evaluation_seconds]
        if tensor_hash(plant.read(None)) != start_hash:
            raise RuntimeError("Evaluation changed the held deployment.")
        row.update(grid=grid, cost=no_writes, digital_parameters=0)
    else:
        writer = optimizer = scheduler = mask = None
        parameters = []
        if arm.method in ("onchip_noisy", "child_tuning_d"):
            if arm.method == "child_tuning_d":
                if ctx.fisher is None:
                    t0 = time.monotonic()
                    ctx.fisher = compute_fisher(ctx)
                    ctx.fisher_seconds = time.monotonic() - t0
                mask = ft.child_mask(ctx.fisher, ctx.layout, arm.reserve_p)
            acceptance, noise = ctx.lc.program_verify.acceptance_percent, ctx.lc.devices.noise_scale
            if legacy:
                writer = ThresholdedWriteAdam(base, learning_rate=arm.learning_rate,
                                              threshold_sigma=law["threshold_sigma"],
                                              accumulate_updates=law["accumulate_updates"],
                                              acceptance_percent=acceptance, noise_scale=noise,
                                              max_write_attempts_per_device=None)
                baseline, baseline_writes = base.cost(), base.write_counts().clone()
            else:
                if not law["accumulate_updates"]:
                    raise ValueError("The reset_absorb learner accumulates its commands.")
                writer = dt.GatedDriftAdam(plant, learning_rate=arm.learning_rate,
                                           threshold_us=dt.gate_us(law["threshold_sigma"], noise, acceptance),
                                           mask=mask)
        else:
            if arm.method == "calibration":
                network.fault_output_calibration = ft.OutputCalibration(network)
                parameters = list(network.fault_output_calibration.parameters())
            elif arm.method == "layernorm":
                parameters = list(network.mlp_norms.parameters())
                for p in parameters:
                    p.requires_grad_(True)
            else:  # digital_lora
                parameters = _lora(network, arm.rank, arm.learning_rate, cfg.lora_seed)
            optimizer = torch.optim.Adam(parameters, lr=arm.learning_rate, betas=(0.9, 0.999), eps=1e-8)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, schedule["steps"])

        def cost_now():
            if writer is None:
                return dict(no_writes)
            return _cost(base, baseline, baseline_writes)[0] if legacy else plant.cost()

        generator = torch.Generator().manual_seed(schedule["order_seed"])
        steps, every = schedule["steps"], schedule["development_every_steps"]
        horizon = d.age(steps)
        grid_steps = {d.step_at(t): t for t in d.evaluation_seconds if 0 < t <= horizon}
        project = writer is not None and not arm.no_drift
        grid, curve, projections, losses = [evaluate(0.0, 0)], [], [], []
        for step in range(steps):
            index = torch.randint(0, len(ctx.recovery), (schedule["batch"],), generator=generator)
            tokens, drawn = ctx.recovery[index].to(device), ctx.samples[index.to(device)]
            network.train()
            age = d.age(step + 1)
            q = plant.read(age, keep=writer is not None and not legacy)
            if writer is not None:
                q.requires_grad_(True)
            else:
                optimizer.zero_grad(set_to_none=True)
            loss = ft.sample_loss(network.forward_features(network.features(tokens), q), drawn)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss in arm {arm.label}.")
            loss.backward()
            if writer is None:
                optimizer.step()
                scheduler.step()
                learning_rate = scheduler.get_last_lr()[0]
            else:
                writer.learning_rate = learning_rate = ft.cosine(arm.learning_rate, step, steps)
                if legacy:
                    writer.step(q.grad)
                else:
                    writer.step(q.grad, age)
            losses.append(float(loss.detach()))
            done = step + 1
            ctx.stage.store.append_metric({"stage": "drift_tuning_update", "arm": arm.label, "step": done,
                                           "age_seconds": d.age(done), "train_loss": losses[-1],
                                           "learning_rate": learning_rate})
            if done in grid_steps:
                point = evaluate(grid_steps[done], done)
                point["cost"] = cost_now()
                grid.append(point)
                curve.append(point)
            elif done % every == 0:
                point = evaluate(d.age(done), done, "development")
                point["cost"] = cost_now()
                curve.append(point)
            if project and done in d.projection_steps:
                projections.append(evaluate(d.projection_seconds, done, "projection"))
        base.verify_permanence()
        for p in parameters:
            p.requires_grad_(False)
        grid += [evaluate(t, steps) for t in d.evaluation_seconds if t > horizon]
        cost = cost_now()
        if writer is not None:
            cost["writer"] = writer.diagnostics()
        if mask is not None:
            writes = plant.weight_writes() if not legacy else base.write_counts() - baseline_writes
            cost.update(mask_weights=int(mask.sum()), mask_sha256=tensor_hash(mask),
                        writes_outside_mask=int(writes[~mask].sum()))
            if cost["writes_outside_mask"]:
                raise RuntimeError("Child-Tuning_D wrote outside its fixed mask.")
        if writer is None and (tensor_hash(plant.read(None)) != start_hash or base.cost() != ctx.deployed_cost):
            raise RuntimeError(f"The digital arm {arm.label} changed the base deployment.")
        digital = sum(p.numel() for p in parameters)
        row.update(grid=grid, curve=curve, projections=projections, final_no_drift=evaluate(None, steps, "no_drift"),
                   train_loss_first10=sum(losses[:10]) / min(10, len(losses)),
                   train_loss_last10=sum(losses[-10:]) / min(10, len(losses)), train_losses=losses, cost=cost,
                   digital_parameters=digital)
        if arm.method == "layernorm":
            row["layernorm_state_sha256"] = _state_digest({k: v.detach() for k, v in network.mlp_norms.state_dict().items()})
        reference = refs.get("twin" if arm.no_drift else "hold")
        if writer is not None and cost["device_writes"] == 0 and reference is not None:
            mismatched = [p["age_seconds"] for p in grid if p["teacher_kl"] != reference[p["age_seconds"]]]
            row["zero_write_replay"] = {"checked": True, "reference": "twin" if arm.no_drift else "hold",
                                        "mismatched_ages": mismatched}
            if mismatched:
                raise RuntimeError(f"A learner with no writes did not replay the hold arm: {mismatched}.")
        else:
            row["zero_write_replay"] = {"checked": False}
    ages = [p["age_seconds"] for p in row["grid"]]
    if ages != list(d.evaluation_seconds):
        raise RuntimeError(f"Arm {arm.label} grid coverage {ages} differs from the declared grid.")
    kl = [p["teacher_kl"] for p in row["grid"]]
    last = row["grid"][-1]["bounds"]
    row.update(initial=row["grid"][0]["development"], final=row["grid"][-1]["development"],
               mean_teacher_kl=dt.trapezoid_mean(ages, kl), final_apparent_sha256=tensor_hash(plant.read(None)),
               at_drift_min_fraction_final=None if last is None else last["at_drift_min_fraction"],
               primary_state="held_apparent" if arm.no_drift else "held_apparent_drifted",
               seconds=time.monotonic() - started)
    network.enable_lora(None)
    network.fault_output_calibration = None
    network.mlp_norms.load_state_dict(norms)
    network.mlp_norms.requires_grad_(False)
    if _state_digest({k: v.detach() for k, v in network.mlp_norms.state_dict().items()}) != norms_hash:
        raise RuntimeError("Expected the source LayerNorms restored after the arm.")
    return row


def run(stage):
    from ebl.schema.source_fit import validate_source_fit
    lc, cfg = stage.lifecycle, stage.lifecycle.fault_tuning
    if sha256_file(stage.request.weights) != cfg.source_sha256:
        raise ValueError("Fault-tuning frozen source SHA-256 mismatch.")
    source = stage.checked("weights", "hwa_source", hwa_arm=stage.selection.hwa_arm)
    validate_source_fit(lc, lc.hwa.arm(stage.selection.hwa_arm), source["fit"])
    stage.network.load_state_dict(source["model"])
    del source
    stage.cache()
    stage.device_bundle()
    network = stage.network
    for p in network.parameters():
        p.requires_grad_(False)
    target = network.q.detach().clone()
    ctx = Context(stage)
    store = stage.store
    started = time.monotonic()
    ctx.recovery = ft.cohort(lc, cfg.recovery)
    ctx.development = ft.cohort(lc, cfg.development)
    prepare_evaluation_cohorts(ctx)
    t = cfg.targets
    if "file" in t:
        # A stored draw of the same procedure (the draw is GPU-specific); accepted by tensor hash only.
        root = os.environ.get("EBL_CORPUS_ROOT")
        if not root:
            raise ValueError("Expected EBL_CORPUS_ROOT to hold the pinned teacher samples.")
        samples_path = Path(root) / f"{t['file']}.pt"
        ctx.samples = torch.load(samples_path, map_location="cpu", weights_only=True)
        if (not isinstance(ctx.samples, torch.Tensor) or ctx.samples.dtype != torch.int64
                or tuple(ctx.samples.shape) != (len(ctx.recovery), lc.data.sequence_length - 1, t["count"])):
            raise ValueError("Expected an int64 (sequences, length - 1, count) teacher-sample tensor.")
        ctx.samples = ctx.samples.to(stage.device)
        store.append_metric({"stage": "fault_tuning_samples_file", "path": str(samples_path.resolve()),
                             "file_sha256": sha256_file(samples_path)})
    else:
        ctx.samples = ft.teacher_samples(network, ctx.recovery, t["count"], t["seed"], t["draw_batch"], stage.device)
    samples_sha256 = ft.raw_hash(ctx.samples)
    store.append_metric({"stage": "fault_tuning_samples", "samples_sha256": samples_sha256, "count": t["count"],
                         "seed": t["seed"]})
    if samples_sha256 != t["sha256"]:
        raise ValueError(f"Teacher-sample draw {samples_sha256} differs from the pinned {t['sha256']}.")
    if cfg.analog_lora is not None:
        rank = cfg.analog_lora["rank"]
        ctx.adapter_layout = ft.adapter_layout(ctx.layout, rank)
    artifacts = []
    if cfg.deployment is None:
        bundle = deploy(ctx, target)
        path = stage.run_dir / "checkpoints/fault_deployment.pt"
        save(path, bundle)
        artifacts.append(("fault_deployment", path))
    else:
        bundle = load(pinned_path(cfg.deployment))
        check_deployment(ctx, bundle, target)
    ctx.deployed = bundle["states"]
    ctx.deployed_cost = ctx.array("fault_aware").cost()
    if cfg.analog_lora is not None:
        ctx.a_scales = [s.to(stage.device) for s in bundle["adapter"]["a_scales"]]
    if cfg.analog_lora is not None and cfg.analog_lora["b_scale"] is not None:
        b_file = load(pinned_path(cfg.analog_lora["b_scale"]))
        if b_file.get("rank") != cfg.analog_lora["rank"] or len(b_file["b_scales"]) != len(ctx.layout):
            raise ValueError("Expected one per-output-channel B scale per analog matrix of the declared rank.")
        ctx.b_scales = [s.to(stage.device).clamp_min(1e-12) for s in b_file["b_scales"]]
    deployment_seconds = time.monotonic() - started
    clean = ctx.evaluate(target)
    store.append_metric({"stage": "fault_tuning_deployed", "clean": clean,
                         "apparent_sha256": {k: v["apparent_sha256"] for k, v in ctx.deployed.items()}})
    records = None
    if cfg.mapping is not None:
        records = deployment_records(ctx, target, clean)
        store.append_metric({"stage": "fault_tuning_deployment_records", **records})
    rows = []
    drift_records = None
    if cfg.drift is not None:
        refs = {}
        # The drifted hold first, then its drift-off twin, then the learners in declared order.
        for arm in sorted(cfg.arms, key=lambda a: 2 if a.method != "hold_fault_aware" else int(a.no_drift)):
            row = run_drift_arm(ctx, arm, refs)
            if arm.method == "hold_fault_aware":
                refs["twin" if arm.no_drift else "hold"] = {p["age_seconds"]: p["teacher_kl"] for p in row["grid"]}
            rows.append(row)
            atomic_write_json(stage.run_dir / "measurements.json", rows)
            store.append_metric({"stage": "drift_tuning_arm", "arm": arm.label,
                                 "mean_teacher_kl": row["mean_teacher_kl"], "seconds": row["seconds"]})
        released = rows[0]["grid"][0]["teacher_kl"]
        # Every arm starts from the one released deployment: equal release reads (A0, the twin, drift-off).
        mismatched = [r["label"] for r in rows if r["grid"][0]["teacher_kl"] != released]
        if mismatched:
            raise RuntimeError(f"Arms {mismatched} do not start from the released deployment of A0.")
        from methods.drift_tuning import Mapping
        mapping = ctx.mapping or Mapping()
        clean_mapped = mapping.clean(target)
        drift_records = {
            "mapping": None if mapping.spec is None else dict(mapping.spec), "span_us_per_unit_q": mapping.span,
            "source_ideal_teacher_kl": clean["teacher_kl"],
            "clean_mapped_readout_teacher_kl": ctx.evaluate(clean_mapped)["teacher_kl"],
            "clean_mapped_readout_max_abs_error": float((clean_mapped - target).abs().max()),
            "deployed_release_teacher_kl": released,
            "per_arm": {r["label"]: {"mean_teacher_kl": r["mean_teacher_kl"],
                                     "final_teacher_kl": r["final"]["teacher_kl"],
                                     "device_writes": r["cost"].get("device_writes", 0),
                                     "writes_per_device": r["cost"].get("writes_per_device", 0.0),
                                     "fraction_devices_written": r["cost"].get("fraction_devices_written", 0.0),
                                     "at_drift_min_fraction_final": r["at_drift_min_fraction_final"],
                                     "digital_parameters": r["digital_parameters"],
                                     "no_drift": bool(r.get("no_drift"))} for r in rows},
        }
        store.append_metric({"stage": "drift_tuning_records", **{k: v for k, v in drift_records.items()
                                                                 if k != "per_arm"}})
    for arm in (cfg.arms if cfg.drift is None else ()):
        row, state = run_arm(ctx, arm)
        if state is not None:
            path = stage.run_dir / "checkpoints" / "arms" / f"{arm.label}.pt"
            save(path, state)
            row["state_sha256"] = sha256_file(path)
            artifacts.append((f"fault_tuning_state.{arm.label}", path))
        rows.append(row)
        atomic_write_json(stage.run_dir / "measurements.json", rows)
        store.append_metric({"stage": "fault_tuning_arm", "arm": arm.label,
                             "final_teacher_kl": row["final"]["teacher_kl"], "seconds": row["seconds"]})
    identity = {
        "assignment_seed": ctx.seed, "case": ctx.case.to_dict(), "source_sha256": cfg.source_sha256,
        "recovery_cohort_sha256": cfg.recovery["cohort_sha256"],
        "development_cohort_sha256": cfg.development["cohort_sha256"], "samples_sha256": samples_sha256,
        "deployment": cfg.deployment, "deployment_gpu": bundle["gpu"], "deployment_torch": bundle["torch"],
        "apparent_sha256": {k: v["apparent_sha256"] for k, v in ctx.deployed.items()},
        "fault_identity_sha256": bundle["identity"].get("identity_sha256"),
        "adapter_fault_identity_sha256": None if bundle["adapter"] is None
        else bundle["adapter"]["identity"].get("identity_sha256"),
        "fisher_seconds": getattr(ctx, "fisher_seconds", None),
        "gpu": torch.cuda.get_device_name(stage.device) if stage.device.type == "cuda" else "cpu",
    }
    summary = {**identity, "clean": clean,
               "deployment_reports": {k: v["report"] for k, v in ctx.deployed.items()},
               "arms": {r["label"]: {"method": r["method"], "learning_rate": r["learning_rate"], "rank": r["rank"],
                                     "initial_teacher_kl": r["initial"]["teacher_kl"],
                                     "final_teacher_kl": r["final"]["teacher_kl"],
                                     "cost": {k: v for k, v in r["cost"].items() if k != "writer"},
                                     "seconds": r["seconds"]} for r in rows},
               "deployment_seconds": deployment_seconds,
               **({"deployment_records": records} if records is not None else {}),
               **({"drift": {"settings": cfg.to_dict()["drift"],
                             "mean_teacher_kl": {r["label"]: r["mean_teacher_kl"] for r in rows},
                             **({"records": drift_records} if cfg.drift.write_clock != "continues" else {})}}
                  if cfg.drift is not None else {}),
               "selection": "final endpoint; development teacher KL; no test cohort"}
    if cfg.evaluation_cohorts:
        summary["evaluation_cohorts"] = [dict(c) for c in cfg.evaluation_cohorts]
        summary["selection"] = "final endpoint; development teacher KL; held-out cohorts are endpoint-only"
    atomic_write_json(stage.run_dir / "summary.json", summary)
    return summary, [("fault_tuning_measurements", stage.run_dir / "measurements.json"),
                     ("fault_tuning_summary", stage.run_dir / "summary.json")] + artifacts
