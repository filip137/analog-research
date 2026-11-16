"""Native ``fault_tuning`` stage: the fault-recovery LR-tuning screen (see ``methods.fault_tuning``).

Without a pinned ``fault_tuning.deployment`` the stage programs the assignment
(naive P&V, then the fault-aware pair rewrite, plus the analog-LoRA tiles when
declared) and saves every state in portable tensor form. With one, it loads that
file (resolved under ``EBL_CORPUS_ROOT``, pinned by SHA-256) and checks every
apparent-state hash, so arms on any GPU start from the identical deployment.
"""
from __future__ import annotations

import json
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
    return [{"role": role, "path": str(pinned_path(e)), "sha256": e["sha256"]} for role, e in entries if e is not None]


class Context:
    def __init__(self, stage):
        self.stage, self.lc, self.cfg = stage, stage.lifecycle, stage.lifecycle.fault_tuning
        self.network, self.device = stage.network, stage.device
        self.seed = stage.selection.array_seed
        self.case = self.lc.defects.cases[0]
        self.layout = list(self.network.layout)
        self.fisher = None

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
        if tensor_hash(array.read()) != entry["apparent_sha256"]:
            raise RuntimeError(f"Expected the exact deployed {key} state.")
        return array

    def evaluate(self, q):
        return self.stage.family.evaluate(self.network, {"tokens": self.development}, self.device, q=q)


def deploy(ctx, target):
    """Naive P&V, fault-aware pair rewrite and (if declared) the analog-LoRA tiles; portable states."""
    lc, cfg = ctx.lc, ctx.cfg
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


def check_deployment(ctx, bundle, target):
    if (bundle.get("schema") != "fault_tuning.deployment.v1" or bundle["assignment_seed"] != ctx.seed
            or bundle["case"] != ctx.case.to_dict() or bundle["target_sha256"] != tensor_hash(target)):
        raise ValueError("Expected the pinned deployment of this source, assignment and defect case.")
    if (ctx.cfg.analog_lora is not None) != (bundle["adapter"] is not None):
        raise ValueError("Expected adapter tiles exactly when analog LoRA is declared.")
    if bundle["adapter"] is not None and (bundle["adapter"]["rank"] != ctx.cfg.analog_lora["rank"]
                                          or bundle["adapter"]["lora_seed"] != ctx.cfg.lora_seed):
        raise ValueError("Expected adapter tiles of the declared rank and seed.")


def compute_fisher(ctx):
    """Mean squared batch gradient of the training loss over the whole recovery set, fault-aware state."""
    network, cfg = ctx.network, ctx.cfg
    q0 = ctx.array("fault_aware").read()
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
    held = base.read()
    initial = ctx.evaluate(held)
    row = {"label": arm.label, "method": arm.method, "learning_rate": arm.learning_rate, "rank": arm.rank,
           "reserve_p": arm.reserve_p, "initial_apparent_sha256": tensor_hash(held), "initial": initial}
    if arm.method not in ft.TRAINS:
        row.update(final=initial, curve=[], cost={}, final_apparent_sha256=row["initial_apparent_sha256"],
                   seconds=time.monotonic() - started)
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
            q = base.read().requires_grad_(True)
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
                     "development": ctx.evaluate(base.read() if arm.method in ft.WRITES else held)}
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
        if tensor_hash(base.read()) != base_hash or base.cost() != ctx.deployed_cost:
            raise RuntimeError("Calibration changed the base deployment.")
        correction = network.fault_output_calibration
        row["calibration_parameters"] = sum(p.numel() for p in correction.parameters())
        state = {"schema": "fault_tuning.sampled_output_calibration.v1",
                 "placement": "complete_mlp_output_before_residual",
                 "state": {k: v.detach().cpu().clone() for k, v in correction.state_dict().items()}}
    if arm.method in ft.LORA:
        if tensor_hash(base.read()) != base_hash or base.cost() != ctx.deployed_cost:
            raise RuntimeError("A LoRA arm changed the base deployment.")
        row["lora_parameters"] = sum(p.numel() for p in network.lora.parameters())
        state = {"lora": {k: v.detach().cpu().clone() for k, v in network.lora.state_dict().items()},
                 "rank": arm.rank, "alpha": float(arm.rank), "seed": cfg.lora_seed}
        if adapter is not None:
            adapter.verify_permanence()
            state["adapter_array"] = adapter.state_dict()
            state["final_adapter_apparent_sha256"] = tensor_hash(adapter.read())
            row["final_adapter_apparent_sha256"] = state["final_adapter_apparent_sha256"]
    row.update(final_apparent_sha256=tensor_hash(base.read()), seconds=time.monotonic() - started)
    network.enable_lora(None)
    network.fault_output_calibration = None
    return row, state


def run_drift_arm(ctx, arm, hold):
    """One arm under ``fault_tuning.drift`` (AIHWKit CMO log-time law, clock continues through writes).

    Every device-facing forward reads the held devices through the drift law at its age;
    writes shift the held programmed conductances. ``hold_fault_aware`` is the untreated
    deployment read on the grid; ``onchip_noisy`` starts at release and update k reads at
    k * update_seconds. Equal ages give identical draws in every arm (paired).
    """
    import methods.drift_tuning as dt
    network, cfg, device, d = ctx.network, ctx.cfg, ctx.device, ctx.cfg.drift
    schedule, law = cfg.schedule, cfg.write_law
    started = time.monotonic()
    network.enable_lora(None)
    network.fault_output_calibration = None
    base = ctx.array("fault_aware")
    start_hash = tensor_hash(base.read())

    def evaluate(seconds, step, kind="grid"):
        q = base.read() if seconds is None else dt.drifted_weights(base, d, seconds)
        metrics = ctx.evaluate(q)
        point = {"arm": arm.label, "kind": kind, "age_seconds": seconds, "step": step,
                 "teacher_kl": metrics["teacher_kl"], "development": metrics,
                 "bounds": None if seconds is None else dt.bound_counts(base, d, seconds)}
        ctx.stage.store.append_metric({"stage": "drift_tuning_evaluation", **point})
        print(json.dumps({"arm": arm.label, "kind": kind, "age_seconds": seconds, "step": step,
                          "teacher_kl": metrics["teacher_kl"]}), flush=True)
        return point

    row = {"label": arm.label, "method": arm.method, "learning_rate": arm.learning_rate, "rank": None,
           "reserve_p": None, "initial_apparent_sha256": start_hash}
    if arm.method == "hold_fault_aware":
        grid = [evaluate(t, None) for t in d.evaluation_seconds]
        if tensor_hash(base.read()) != start_hash:
            raise RuntimeError("Evaluation changed the held deployment.")
        row.update(grid=grid, cost={"healthy_device_writes": 0, "stuck_write_attempts": 0, "touched_devices": 0,
                                    "max_writes_per_device": 0, "out_of_range_devices": base.out_of_range()})
    else:
        acceptance, noise = ctx.lc.program_verify.acceptance_percent, ctx.lc.devices.noise_scale
        writer = ThresholdedWriteAdam(base, learning_rate=arm.learning_rate, threshold_sigma=law["threshold_sigma"],
                                      accumulate_updates=law["accumulate_updates"], acceptance_percent=acceptance,
                                      noise_scale=noise, max_write_attempts_per_device=None)
        baseline, baseline_writes = base.cost(), base.write_counts().clone()
        generator = torch.Generator().manual_seed(schedule["order_seed"])
        steps, every = schedule["steps"], schedule["development_every_steps"]
        horizon = d.age(steps)
        grid_steps = {d.step_at(t): t for t in d.evaluation_seconds if 0 < t <= horizon}
        grid, curve, projections, losses = [evaluate(0.0, 0)], [], [], []
        for step in range(steps):
            index = torch.randint(0, len(ctx.recovery), (schedule["batch"],), generator=generator)
            tokens, drawn = ctx.recovery[index].to(device), ctx.samples[index.to(device)]
            network.train()
            q = dt.drifted_weights(base, d, d.age(step + 1)).requires_grad_(True)
            loss = ft.sample_loss(network.forward_features(network.features(tokens), q), drawn)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss in arm {arm.label}.")
            loss.backward()
            writer.learning_rate = ft.cosine(arm.learning_rate, step, steps)
            writer.step(q.grad)
            losses.append(float(loss.detach()))
            done = step + 1
            ctx.stage.store.append_metric({"stage": "drift_tuning_update", "arm": arm.label, "step": done,
                                           "age_seconds": d.age(done), "train_loss": losses[-1],
                                           "learning_rate": writer.learning_rate})
            if done in grid_steps:
                point = evaluate(grid_steps[done], done)
                point["cost"] = _cost(base, baseline, baseline_writes)[0]
                grid.append(point)
                curve.append(point)
            elif done % every == 0:
                point = evaluate(d.age(done), done, "development")
                point["cost"] = _cost(base, baseline, baseline_writes)[0]
                curve.append(point)
            if done in d.projection_steps:
                projections.append(evaluate(d.projection_seconds, done, "projection"))
        base.verify_permanence()
        grid += [evaluate(t, steps) for t in d.evaluation_seconds if t > horizon]
        cost = _cost(base, baseline, baseline_writes)[0]
        cost["writer"] = writer.diagnostics()
        row.update(grid=grid, curve=curve, projections=projections, final_no_drift=evaluate(None, steps, "no_drift"),
                   train_loss_first10=sum(losses[:10]) / min(10, len(losses)),
                   train_loss_last10=sum(losses[-10:]) / min(10, len(losses)), train_losses=losses, cost=cost)
        if cost["device_writes"] == 0 and hold is not None:
            mismatched = [p["age_seconds"] for p in grid if p["teacher_kl"] != hold[p["age_seconds"]]]
            row["zero_write_replay"] = {"checked": True, "mismatched_ages": mismatched}
            if mismatched:
                raise RuntimeError(f"A learner with no writes did not replay the hold arm: {mismatched}.")
        else:
            row["zero_write_replay"] = {"checked": False}
    ages = [p["age_seconds"] for p in row["grid"]]
    if ages != list(d.evaluation_seconds):
        raise RuntimeError(f"Arm {arm.label} grid coverage {ages} differs from the declared grid.")
    kl = [p["teacher_kl"] for p in row["grid"]]
    row.update(initial=row["grid"][0]["development"], final=row["grid"][-1]["development"],
               mean_teacher_kl=dt.trapezoid_mean(ages, kl), final_apparent_sha256=tensor_hash(base.read()),
               primary_state="held_apparent_drifted", seconds=time.monotonic() - started)
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
    rows = []
    if cfg.drift is not None:
        hold = None
        for arm in sorted(cfg.arms, key=lambda a: a.method != "hold_fault_aware"):
            row = run_drift_arm(ctx, arm, hold)
            if arm.method == "hold_fault_aware":
                hold = {p["age_seconds"]: p["teacher_kl"] for p in row["grid"]}
            rows.append(row)
            atomic_write_json(stage.run_dir / "measurements.json", rows)
            store.append_metric({"stage": "drift_tuning_arm", "arm": arm.label,
                                 "mean_teacher_kl": row["mean_teacher_kl"], "seconds": row["seconds"]})
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
               **({"drift": {"settings": cfg.to_dict()["drift"],
                             "mean_teacher_kl": {r["label"]: r["mean_teacher_kl"] for r in rows}}}
                  if cfg.drift is not None else {}),
               "selection": "final endpoint; development teacher KL; no test cohort"}
    atomic_write_json(stage.run_dir / "summary.json", summary)
    return summary, [("fault_tuning_measurements", stage.run_dir / "measurements.json"),
                     ("fault_tuning_summary", stage.run_dir / "summary.json")] + artifacts
