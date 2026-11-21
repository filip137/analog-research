"""Native downstream task adaptation within the stationary CMO fault_tuning stage."""
from __future__ import annotations

import copy
import math
import time
import torch

from core.artifacts import atomic_write_json
from core.serialization import clone_cpu, load, save
from core.tensor_identity import state_hash, tensor_hash
from hardware.devices.cmo import G_MIN_US, G_MAX_US
from methods.fault_tuning import OutputCalibration
from methods.physical_child import PhysicalWriter, held_conductances, nested_masks
from methods.programming.lifecycle import detect_stuck_devices
from methods.task_adaptation import backward_batch, evaluate
from ebl.runtime import fault_tuning as fault


def archived_v100_readout(children, span):
    """V100 scalar division used a pre-rounded FP32 reciprocal. Require the archive's exact q hash."""
    def read(child):
        g = child["g"]
        return (g[0] - g[1]) * g.new_tensor(span).reciprocal()
    return torch.cat([read(child) for child in children])


def calibrate(network, target_q, measured_q, tokens, device):
    """Layerwise least-squares correction of block outputs on the same local input."""
    result = OutputCalibration(network).requires_grad_(False)
    with torch.no_grad():
        hidden = [network.analog_input(batch.long().to(device)) for batch in tokens.split(4)]
        for k in range(len(result.gain)):
            packets = []
            stats = None
            n = 0
            for h in hidden:
                residual, measured, bias = network.mlp_branch(k, h, measured_q)
                _, ref, _ = network.mlp_branch(k, h, target_q)
                measured, ref = measured + bias, ref + bias
                packets.append((residual, measured))
                x, y = measured.flatten(0, 1).double(), ref.flatten(0, 1).double()
                batch_stats = torch.stack([x.sum(0), y.sum(0), x.square().sum(0), (x * y).sum(0)])
                stats = batch_stats if stats is None else stats + batch_stats
                n += len(x)
            mx, my, xx, xy = stats / n
            var = xx - mx.square()
            gain = torch.where(var > 1e-12, (xy - mx * my) / var.clamp_min(1e-12), 1.)
            result.gain[k].copy_(gain)
            result.shift[k].copy_(my - gain * mx)
            hidden = [r + m * result.gain[k] + result.shift[k] for r, m in packets]
    return result


class Snapshot:
    def __init__(self, ctx, target, cal_tokens, seed):
        self.ctx = copy.copy(ctx)
        self.ctx.seed = seed
        states = fault.deploy_mapped(self.ctx, target)["states"]
        self.ctx.deployed = states
        # Reuse the already acquired naive verify readings without another acquisition.
        naive = self.ctx.array("naive")
        self.flags = detect_stuck_devices(held_conductances(naive), target, expected=ctx.mapping.targets(target),
                                          acceptance=ctx.lc.program_verify.acceptance_percent,
                                          noise_scale=ctx.lc.devices.noise_scale, threshold_sigma=5.)
        del naive
        self.q = self.ctx.readout(self.array())
        self.correction = calibrate(ctx.network, target, self.q, cal_tokens, ctx.device)
        self.receipt = {"assignment": seed, "q_sha256": tensor_hash(self.q), "flags_sha256": tensor_hash(self.flags),
                        "calibration_sha256": state_hash(self.correction.state_dict()), "observed_flags": int(self.flags.sum()),
                        "pv": states["fault_aware"]["report"], "common_deployment_writes": states["naive"]["report"]}
        del states["naive"]

    def array(self):
        return self.ctx.array("fault_aware")


class Cell:
    def __init__(self, ctx, target):
        self.ctx, self.net, self.target, self.device = ctx, ctx.network, target, ctx.device
        self.spec = ctx.cfg.task_adaptation
        self.task = load(fault.pinned_path(self.spec["task"]))
        self.retention = load(fault.pinned_path(self.spec["retention"]))
        if self.task["schema"] != "s7-task.v1" or self.retention["task"] != "wikitext":
            raise ValueError("Invalid pinned task/retention bundle.")
        self.cal_tokens = self.retention["splits"]["train"]["tokens"]
        self.source = clone_cpu(self.net.state_dict())
        self.chip = Snapshot(ctx, target, self.cal_tokens, ctx.seed)
        self.proxy = None
        if self.spec["phase"] != "evaluation":
            self.proxy = Snapshot(ctx, target, self.cal_tokens, self.spec["proxy_seed"])
        self.cached_masks = {}
        self.mask_files = []

    def reset(self, view):
        self.net.enable_lora(None)
        self.net.fault_output_calibration = None
        self.net.load_state_dict(self.source)
        self.net.requires_grad_(False)
        if view == "ideal":
            return self.target
        if view == "clean":
            return self.ctx.mapping.clean(self.target)
        state = self.chip if view == "chip" else self.proxy
        self.net.fault_output_calibration = state.correction
        return state.q

    def batch_gradient(self, q, indices):
        q = q.detach().requires_grad_()
        loss = backward_batch(lambda t: self.net(t, q), self.task["splits"]["train"], indices,
                              self.device, self.spec["microbatch"])
        if q.grad is None or not torch.isfinite(q.grad).all():
            raise FloatingPointError("Invalid analog gradient on task objective.")
        return loss, q.grad

    def masks(self, view):
        if view not in self.cached_masks:
            q = self.reset(view)
            scores = torch.zeros_like(q)
            for ids in torch.arange(self.spec["fisher_items"]).split(8):
                _, grad = self.batch_gradient(q, ids)
                scores += grad.square() * (len(ids) / self.spec["fisher_items"])
            eligible = (torch.ones(2, q.numel(), device=self.device, dtype=torch.bool) if view == "clean"
                        else ~(self.chip if view == "chip" else self.proxy).flags)
            masks = nested_masks(scores, eligible, self.net.layout, (.01, .10), self.ctx.mapping.span, self.ctx.lc.deployment.tile_size)
            self.cached_masks[view] = {k: v.cpu() for k, v in masks.items()}
            mask_path = self.ctx.stage.run_dir / "checkpoints" / f"physical-masks-{view}.pt"
            # Bit-packing keeps both full masks, their physical index convention and nesting auditable.
            import numpy as np
            payload = {"schema": "physical-child.masks.v1", "view": view, "shape": list(eligible.shape),
                       "eligible": int(eligible.sum()), "fisher_items": self.spec["fisher_items"], "batch": 8,
                       "index_order": "physical_leg_then_global_logical_weight", "bitorder": "little",
                       "masks": {str(k): {"sha256": tensor_hash(v), "selected": int(v.sum()),
                                 "bits": torch.from_numpy(np.packbits(v.numpy(), bitorder="little"))}
                                 for k, v in self.cached_masks[view].items()}}
            save(mask_path, payload)
            self.mask_files.append((f"physical_masks_{view}", mask_path))
        return self.cached_masks[view]

    @torch.no_grad()
    def score(self, q, split, retention=False):
        self.net.eval()
        target = evaluate(lambda t: self.net(t, q), self.task["splits"][split], self.device,
                          self.spec["microbatch"], limit=self.spec["eval_limit"])
        score = {"target": target}
        if retention:
            score["retention"] = evaluate(lambda t: self.net(t, q), self.retention["splits"][split],
                                          self.device, self.spec["microbatch"],
                                          teacher_fn=lambda t: self.net.teacher(t)[:, :-1],
                                          limit=self.spec["eval_limit"])
        return score

    def lr_for_step(self, rate, step):
        steps = self.spec["steps"]
        warm = max(1, round(.06 * steps))
        if step < warm:
            return rate * (step + 1) / warm
        return rate * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm - 1))) / 2

    def event(self, **fields):
        self.ctx.stage.store.append_metric({"stage": "task_adaptation", **fields})

    def train(self, route, rate, view):
        mask = None
        writer = None
        full, child = route.endswith("full"), "child" in route
        online = route.startswith("onchip")
        if child:
            fraction = .01 if route.endswith("01") else .1
            mask = self.masks(view)[fraction].to(self.device)
        q_train = self.reset(view).detach()
        self.net.train()
        span = self.ctx.mapping.span
        src_g = self.ctx.mapping.targets(self.target)
        q = q_train
        optimizer_params = {}
        if child:
            role, weight = torch.where(mask)
            sign = (1 - 2 * role).to(q.dtype)
            # Coordinates and Adam steps are normalized by the common conductance span.
            delta = torch.nn.Parameter(torch.zeros_like(weight, dtype=q.dtype))
            params = [delta]
            if online:
                array = self.chip.array()
                writer = PhysicalWriter(array, mask, "online-child", self.spec["seed"])
                current = held_conductances(array)
                pending = torch.zeros_like(delta)
        elif full:
            if self.net.digital is None:
                raise ValueError("Full-model task fine-tuning requires a scope=model source.")
            self.net.enable_digital(True)
            self.net.enable_calibration(True)
            optimizer_params = {n: p for n, p in self.net.named_parameters()
                                if p.requires_grad and not n.startswith("fault_output_calibration.")}
            delta = torch.nn.Parameter(torch.zeros_like(q))
            params = list(optimizer_params.values()) + [delta]
        else:
            params = fault._lora(self.net, 16, rate, self.spec["seed"])
        optimizer = torch.optim.Adam(params, lr=rate, foreach=False) if child else torch.optim.AdamW(
            params, lr=rate, weight_decay=0, foreach=False)
        losses = []
        for step, ids in enumerate(self.task["order"][:self.spec["steps"]]):
            optimizer.param_groups[0]["lr"] = (rate * (1 + math.cos(math.pi * step / self.spec["steps"])) / 2
                                                if child else self.lr_for_step(rate, step))
            optimizer.zero_grad(set_to_none=True)
            if child:
                if online:
                    q = self.ctx.mapping.readout(current)
                else:
                    q = q_train.index_add(0, weight, delta.detach() * sign)
                loss, q_grad = self.batch_gradient(q, ids)
                delta.grad = q_grad[weight] * sign
                prev = delta.detach().clone() if online else None
            else:
                # Recompute this graph for each microbatch of a full-model update.
                forward = (lambda t: self.net(t, q_train + delta)) if full else (lambda t: self.net(t, q_train))
                loss = backward_batch(forward, self.task["splits"]["train"], ids, self.device, self.spec["microbatch"])
            if not child:
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                if child and online:
                    pending += (delta - prev) * span
                    fire = pending.abs() >= 1.12
                    if bool(fire.any()):
                        selected = torch.zeros_like(mask)
                        selected[role[fire], weight[fire]] = True
                        targets = current.clone()
                        targets[role[fire], weight[fire]] += pending[fire]
                        writer.program(targets, selected)
                        current = held_conductances(array)
                        pending[fire] = 0
                elif child:
                    baseline = src_g[role, weight]
                    delta.clamp_((G_MIN_US - baseline) / span, (G_MAX_US - baseline) / span)
            losses.append(loss)
            if (step + 1) % 16 == 0:
                self.event(route=route, view=view, lr=rate, step=step + 1, loss=loss)
        if child:
            result = {"kind": "child", "indices": torch.where(mask.flatten())[0].cpu(),
                      "delta_us": (delta.detach() * span).cpu(), "mask_sha256": tensor_hash(mask)}
        elif full:
            result = {"kind": "full", "delta_q": delta.detach().cpu(),
                      "digital": {n: p.detach().cpu().clone() for n, p in optimizer_params.items()}}
        else:
            result = {"kind": "lora", "state": clone_cpu(self.net.lora.state_dict())}
        if writer:
            result.update(chip_q=self.ctx.mapping.readout(current), cost=writer.cost())
        result["training"] = {"steps": len(losses), "first": losses[0], "last": losses[-1],
                        "examples": len(losses) * self.task["order"].shape[1],
                        "optimizer_moment_bytes": sum(2 * p.numel() * p.element_size() for p in params),
                        "fisher_items": self.spec["fisher_items"] if child else 0, "fisher_batch": 8 if child else 0,
                        "selected_devices": int(mask.sum()) if child else 0,
                        "mask_bytes": mask.numel() * mask.element_size() if child else 0,
                        "accumulator_bytes": delta.numel() * delta.element_size() if child and online else 0}

        return result, result["training"]

    def deploy(self, trained, policy="clip"):
        """Apply an independently trained model on the calibrated physical recipient."""
        q = self.reset("chip")
        kind = trained["kind"]
        if kind == "lora":
            fault._lora(self.net, 16, 1., self.spec["seed"])
            self.net.lora.load_state_dict(trained["state"])
            return q, {"attempted_device_writes": 0, "successful_device_writes": 0,
                       "adapter_bytes": sum(v.numel() * v.element_size() for v in trained["state"].values()),
                       "digital_adapter_macs_per_token": sum(v.numel() for v in trained["state"].values())}
        if "chip_q" in trained:
            return trained["chip_q"], trained["cost"]
        mapping = self.ctx.mapping
        nominal = mapping.targets(self.target)
        if kind == "child":
            mask = torch.zeros_like(nominal, dtype=torch.bool)
            indices = trained["indices"].to(self.device)
            mask.flatten()[indices] = True
            if tensor_hash(mask) != trained["mask_sha256"]:
                raise RuntimeError("Uploaded Child mask differs from the frozen one.")
            targets = nominal.clone()
            targets.flatten()[indices] += trained["delta_us"].to(self.device)
            targets.clamp_(G_MIN_US, G_MAX_US)
            desired = mapping.readout(targets)
            upload_bytes = indices.numel() * (8 + 4)
        else:
            desired = self.target + trained["delta_q"].to(self.device)
            with torch.no_grad():
                for name, value in trained["digital"].items():
                    self.net.get_parameter(name).copy_(value)
                if policy == "rescale":
                    for i, spec in enumerate(self.net.layout):
                        matrix = desired[spec.offset:spec.offset + spec.size].view(spec.shape)
                        factor = matrix.abs().amax(1).clamp_min_(1.)
                        self.net.channel_scales[self.net.channel_offsets[i]:self.net.channel_offsets[i+1]].mul_(factor)
                        matrix.div_(factor[:, None])
            desired = desired.clamp(-1, 1)
            targets = mapping.targets(desired)
            mask = torch.ones_like(targets, dtype=torch.bool)
            upload_bytes = trained["delta_q"].numel() * 4 + sum(v.numel() * v.element_size() for v in trained["digital"].values())
        array = self.chip.array()
        writer = PhysicalWriter(array, mask, "offline-sparse-or-full", self.spec["seed"])
        writer.upload(targets, desired, nominal, self.chip.flags, mapping.span)
        return mapping.read_array(array), {**writer.cost(), "upload_bytes": upload_bytes}


def quality(metrics):
    row = metrics["target"]
    return row["accuracy_percent"] if "accuracy_percent" in row else -row["nll"]


def without_predictions(metrics):
    return {name: {k: v for k, v in row.items() if k != "predictions"} for name, row in metrics.items()}


def train_candidates(cell, route, *, canary):
    """Freeze LR, offline view and full-programming policy by final development score."""
    category = "full" if route.endswith("full") else ("lora" if route.endswith("lora") else "child")
    center = cell.spec["rates"][category]
    rates = [center] if canary else [center / 3, center, center * 3]
    views = ["chip"] if route.startswith("onchip") else ["clean", "proxy"]
    trials = []
    best = None
    for rate in rates:
        for view in views:
            trained, training = cell.train(route, rate, view)
            for policy in (["clip", "rescale"] if category == "full" else ["clip"]):
                q, costs = cell.deploy(trained, policy)
                measured = cell.score(q, "development")
                row = {"route": route, "rate": rate, "view": view, "policy": policy, "train": training,
                       "scores": without_predictions(measured), "cost": costs}
                trials.append(row)
                score = quality(measured)
                if best is None or score > best["score"]:
                    best = {"score": score, "rate": rate, "view": view, "policy": policy}
                    best["transfer" if route.startswith("offline") else "replay"] = trained
                cell.event(route=route, lr=rate, view=view, policy=policy, development=score)
    return best, trials


def run_supervised(ctx, target):
    start = time.monotonic()
    cell = Cell(ctx, target)
    spec = cell.spec
    phase = spec["phase"]
    split = "test" if phase == "evaluation" else "development"
    stage = ctx.stage
    identity = {"source": ctx.cfg.source_sha256, "mapping": ctx.mapping.spec, "task": spec["task"],
                "retention": spec["retention"], "case": ctx.case.to_dict(), "seed": spec["seed"],
                "proxy_seed": spec["proxy_seed"], "routes": spec["routes"], "rates": spec["rates"],
                "steps": spec["steps"], "fisher_items": spec["fisher_items"]}
    out = {"schema": "fault_tuning.supervised.v1", "phase": phase, "identity": identity,
           "deployment": cell.chip.receipt, "proxy": None if cell.proxy is None else cell.proxy.receipt,
           "scores": {}, "trials": [], "selections": {}}
    if phase == "evaluation":
        package = load(fault.pinned_path(spec["selection"]))
        if package["schema"] != "supervised.selection.v1" or package["identity"] != identity:
            raise ValueError("Frozen task selection is incompatible with the evidence run.")
        if package["development_assignment"] in (ctx.seed, spec["proxy_seed"]):
            raise ValueError("Development and confirmation chips are not disjoint.")
        selections = package["selections"]
        del package
    else:
        selections = {}
    if spec.get("historical"):
        # Replay the archived V100 physical tensors. A CUDA seed recipe generates a different
        # endpoint draw across accelerator models, even with the same assignment and noise law.
        cell.reset("ideal")
        archive = load(fault.pinned_path(spec["historical"]))
        fault.check_deployment(ctx, archive, target)
        state = archive["states"]["fault_aware"]
        q = archived_v100_readout(state["state"]["children"], ctx.mapping.span).to(ctx.device)
        q_sha = tensor_hash(q)
        if q_sha != state["apparent_sha256"]:
            raise ValueError("The historical apparent state differs from its archive.")
        historical = ctx.stage.family.evaluate(cell.net, cell.retention["splits"]["test"], ctx.device, q=q)
        out["historical_wikitext_hold"] = {**historical, "examples": historical["sequences"], "q_sha256": q_sha,
                                          "archive": spec["historical"], "original_gpu": archive["gpu"]}
        del archive, state, q
    for key in ("ideal", "mapped", "hold", "calibration"):
        q = cell.reset("ideal" if key == "ideal" else "clean" if key == "mapped" else "chip")
        if key == "hold":
            cell.net.fault_output_calibration = None
        scores = cell.score(q, split, retention=True)
        out["scores"][key] = {"scores": scores, "cost": {"attempted_device_writes": 0}}
        cell.event(route=key, score=quality(scores))
    results_path = stage.run_dir / "measurements.json"
    atomic_write_json(results_path, out)
    for route in spec["routes"]:
        before = time.monotonic()
        if phase != "evaluation":
            selections[route], trials = train_candidates(cell, route, canary=phase == "canary")
            out["trials"].extend(trials)
        winner = selections[route]
        if "replay" in winner:
            model = winner.pop("replay")
        elif "transfer" in winner:
            model = winner["transfer"]
        else:
            model, _ = cell.train(route, winner["rate"], winner["view"])
        q, cost = cell.deploy(model, winner["policy"])
        scores = cell.score(q, split, retention=True)
        out["scores"][route] = {"scores": scores, "cost": {**cost, "training": model["training"]}, "seconds": time.monotonic()-before}
        out["selections"][route] = {k: v for k, v in winner.items() if k != "transfer"}
        atomic_write_json(results_path, out)
        cell.event(route=route, complete=True, final=quality(scores), seconds=out["scores"][route]["seconds"])
        del model, q
    paths = [("fault_tuning_measurements", results_path), *cell.mask_files]
    if phase == "development":
        path = stage.run_dir / "checkpoints" / "selection.pt"
        save(path, {"schema": "supervised.selection.v1", "identity": identity, "development_assignment": ctx.seed,
                    "selections": selections})
        paths.append(("task_adaptation_selection", path))
    summary = {"schema": out["schema"], "phase": phase, "identity": identity, "assignment": ctx.seed,
               "selections": out["selections"], "scores": {k: without_predictions(v["scores"])
                                                         for k, v in out["scores"].items()},
               "seconds": time.monotonic() - start}
    path = stage.run_dir / "summary.json"
    atomic_write_json(path, summary)
    return summary, paths + [("fault_tuning_summary", path)]
