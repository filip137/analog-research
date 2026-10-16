"""(iv) On-chip training: matched arms from one programmed deployment.

An arm combines what happens to the array weights with digital calibration:

- ``weights: hold`` never writes (with ``calibration`` it is the matched
  no-weight-write control; without it, the deployment-only control);
- ``weights: rewrite`` reprograms the frozen deployment targets on the same
  write schedule, separating rewriting from learning;
- ``weights: learn`` writes digital gradient updates into the same array.

Gradients and optimizer state are digital. Every forward -- gradients,
curves and the final evaluation -- uses the current held post-write apparent
state. Update laws: PCM endpoint reprogramming of a digital master
(straight-through on the apparent state); OM closed-loop pulse Adam (an
accumulated digital target verified with up to ``pulses_per_update`` pulses);
OM open-loop pulse Adam (stochastic one-pulse rounding, no verify reads);
CMO idealized-write Adam (one noisy write per active device and step on the
held state, no master, no verify and no retry; thresholded learning can explicitly
route writes through the verified partner). ``max_pulses_per_cell`` caps OM
on-chip writes per cell; ``null`` is uncapped.
"""

from __future__ import annotations

import json
import time

import torch

from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.model import tensor_hash
from workflow import data as data_stage
from workflow import devices as device_stage
from workflow import context_views
from workflow import program_verify
from workflow.networks import family


def _adam_command(writer, gradient):
    if gradient.shape != writer.first.shape or not bool(torch.isfinite(gradient).all()):
        raise ValueError("Expected a finite normalized gradient for every crosspoint.")
    writer.step_index += 1
    writer.first.mul_(0.9).add_(gradient, alpha=0.1)
    writer.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
    # A declared ``onchip.lr_schedule`` sets ``lr_factor`` before each update; constant laws never do.
    factor = getattr(writer, "lr_factor", None)
    rate = writer.learning_rate if factor is None else writer.learning_rate * factor
    return (
        -rate
        * (writer.first / (1 - 0.9**writer.step_index))
        / ((writer.second / (1 - 0.999**writer.step_index)).sqrt() + 1e-8)
    )


class ClosedLoopPulseAdam:
    """Accumulated-target Adam with verified pulses, optionally capped per cell.

    Equals the historical capped ``PulseWriter`` closed-loop law and, with
    one pulse per update and no cap, ``UncappedClosedLoopAdam``.
    """

    contract = "crossbar_lifecycle.om_closed_loop_pulse_adam.v1"

    def __init__(self, port, *, learning_rate, tolerance, pulses_per_update, cap):
        self.port, self.learning_rate = port, float(learning_rate)
        self.tolerance, self.pulses_per_update, self.cap = tolerance, pulses_per_update, cap
        self.target = port.read()
        self.first = torch.zeros_like(self.target)
        self.second = torch.zeros_like(self.target)
        self.count = torch.zeros_like(self.target, dtype=torch.int64)
        self.step_index, self.verify_reads, self.probability_clipped = 0, 0, 0

    @torch.no_grad()
    def step(self, gradient):
        command = _adam_command(self, gradient)
        # A zero-learning-rate control must not trigger verify writes.
        if self.learning_rate == 0:
            return
        self.target.add_(command)
        for _ in range(self.pulses_per_update):
            error = self.target - self.port.read()
            self.verify_reads += self.port.size
            active = error.abs() > self.tolerance
            if self.cap is not None:
                active &= self.count < self.cap
            if not bool(active.any()):
                break
            self.port.pulse(error.sign().to(torch.int8) * active)
            self.count += active
        self.port.finish_step()

    def state_dict(self):
        return clone_cpu(
            {
                "contract": [self.contract, self.learning_rate, self.tolerance, self.pulses_per_update, self.cap],
                "target": self.target,
                "first": self.first,
                "second": self.second,
                "count": self.count,
                "step_index": self.step_index,
                "verify_reads": self.verify_reads,
            }
        )


class OpenLoopPulseAdam:
    """Stochastic one-pulse rounding of digital Adam, optionally capped.

    Equals the historical capped ``PulseWriter`` open-loop law and, with no
    cap, ``UncappedOpenLoopAdam``. It never reads the device to decide pulses.
    """

    contract = "crossbar_lifecycle.om_open_loop_pulse_adam.v1"

    def __init__(self, port, *, learning_rate, cap, seed):
        self.port, self.learning_rate, self.cap = port, float(learning_rate), cap
        template = port.read()
        self.first = torch.zeros_like(template)
        self.second = torch.zeros_like(template)
        self.count = torch.zeros_like(template, dtype=torch.int64)
        self.step_index, self.verify_reads, self.probability_clipped = 0, 0, 0
        self.generator = torch.Generator(device=template.device).manual_seed(seed)

    @torch.no_grad()
    def step(self, gradient):
        command = _adam_command(self, gradient)
        if self.learning_rate == 0:
            return
        probability = command.abs() / self.port.nominal_step
        self.probability_clipped += int((probability > 1).sum())
        selected = (
            torch.rand(probability.shape, device=probability.device, generator=self.generator)
            < probability.clamp_max(1)
        )
        if self.cap is not None:
            selected &= self.count < self.cap
        self.port.pulse(command.sign().to(torch.int8) * selected)
        self.count += selected
        self.port.finish_step()

    def state_dict(self):
        return clone_cpu(
            {
                "contract": [self.contract, self.learning_rate, self.port.nominal_step, self.cap],
                "first": self.first,
                "second": self.second,
                "count": self.count,
                "step_index": self.step_index,
                "probability_clipped": self.probability_clipped,
                "rng": self.generator.get_state(),
            }
        )


class IdealizedWriteAdam:
    """Digital Adam applied as one idealized CMO write per device and step.

    Every device with a nonzero command receives G <- G + dG + sigma(G + dG) xi
    (``training.reram_cmo``). There is no digital master, no verify read and
    no retry, so earlier write errors persist in the held state. A declared
    ``write_noise_scale`` multiplies the write noise only (0 is the exact-write
    oracle); undeclared, the physical law and its checkpoints are unchanged.
    """

    contract = "crossbar_lifecycle.cmo_idealized_write_adam.v1"

    def __init__(self, array, *, learning_rate, write_noise_scale=None):
        self.array, self.learning_rate = array, float(learning_rate)
        self.write_noise_scale = None if write_noise_scale is None else float(write_noise_scale)
        template = array.read()
        self.first = torch.zeros_like(template)
        self.second = torch.zeros_like(template)
        self.step_index = 0

    def _write(self, command):
        if self.write_noise_scale is None:
            self.array.write(command)
        else:
            self.array.write(command, noise_factor=self.write_noise_scale)

    @torch.no_grad()
    def step(self, gradient):
        command = _adam_command(self, gradient)
        # A zero-learning-rate control must not write.
        if self.learning_rate == 0:
            return
        self._write(command)

    def state_dict(self):
        return clone_cpu(
            {
                "contract": [self.contract, self.learning_rate],
                "first": self.first,
                "second": self.second,
                "step_index": self.step_index,
                **({} if self.write_noise_scale is None else {"write_noise_scale": self.write_noise_scale}),
            }
        )


class ThresholdedWriteAdam(IdealizedWriteAdam):
    """Gate physical commands against a declared nominal mid-range noise scale.

    Optional signed accumulation retains unissued Adam commands digitally. An
    issued command is subtracted even when the addressed cell is stuck: the
    controller receives no hidden fault map or realized noise. Residuals are
    never forcibly flushed at the final endpoint. The physical write law is
    unchanged, including its conductance-dependent noise and held errors.
    """

    contract = "crossbar_lifecycle.cmo_thresholded_write_adam.v1"

    def __init__(self, array, *, learning_rate, threshold_sigma, accumulate_updates,
                 acceptance_percent, noise_scale, max_write_attempts_per_device=None, write_noise_scale=None,
                 write_roles=None):
        super().__init__(array, learning_rate=learning_rate, write_noise_scale=write_noise_scale)
        from training.reram_cmo import HALF_RANGE_US, MID_US, programming_sigma

        # The threshold references the device noise scale, never the write-noise factor.
        self.threshold_sigma = float(threshold_sigma)
        self.noise_std_q = float(noise_scale * programming_sigma(MID_US, acceptance_percent) / HALF_RANGE_US)
        self.threshold_q = self.threshold_sigma * self.noise_std_q
        self.accumulate_updates = bool(accumulate_updates)
        self.pending = torch.zeros_like(self.first) if self.accumulate_updates else None
        self.max_write_attempts_per_device = max_write_attempts_per_device
        self.write_roles = None if write_roles is None else write_roles.detach().clone()
        if self.write_roles is not None and (
                self.write_roles.shape != self.first.shape or self.write_roles.dtype != torch.int8
                or bool(((self.write_roles < -1) | (self.write_roles > 1)).any())):
            raise ValueError("Expected one int8 active/reference/blocked role per logical weight.")
        # Controller-owned issued-command history, never the plant's successful
        # write counters: failed/stuck attempts consume the same allowance.
        self.attempt_counts = (None if max_write_attempts_per_device is None
                               else torch.zeros_like(self.first, dtype=torch.int32))

    @torch.no_grad()
    def step(self, gradient):
        command = _adam_command(self, gradient)
        if self.learning_rate == 0:
            return
        eligible = None if self.write_roles is None else self.write_roles >= 0
        if self.attempt_counts is not None:
            within_cap = self.attempt_counts < self.max_write_attempts_per_device
            eligible = within_cap if eligible is None else eligible & within_cap
        if eligible is not None:
            command = torch.where(eligible, command, 0.0)
        if self.pending is not None:
            self.pending.add_(command)
            command = self.pending
        if not bool(torch.isfinite(command).all()):
            raise FloatingPointError("Nonfinite thresholded write command.")
        requested = command.abs() > self.threshold_q
        if eligible is not None:
            requested &= eligible
        issued = torch.where(requested, command, 0.0)
        self._write(issued)
        if self.attempt_counts is not None:
            self.attempt_counts.add_(requested)
        if self.pending is not None:
            self.pending.sub_(issued)

    def _write(self, command):
        if self.write_roles is None:
            super()._write(command)
        else:
            self.array.write(command, roles=self.write_roles,
                             noise_factor=1.0 if self.write_noise_scale is None else self.write_noise_scale)

    def diagnostics(self):
        values = {
            "threshold_reference": "nominal_midrange",
            "threshold_sigma": self.threshold_sigma,
            "noise_std_q": self.noise_std_q,
            "threshold_q": self.threshold_q,
            "accumulate_updates": self.accumulate_updates,
            "pending_nonzero": 0 if self.pending is None else int((self.pending != 0).sum()),
            "pending_rms_q": 0.0 if self.pending is None else float(self.pending.square().mean().sqrt()),
        }
        if self.write_noise_scale is not None:
            values["write_noise_scale"] = self.write_noise_scale
        if self.write_roles is not None:
            values.update(write_roles="verified_partner", write_role_map_sha256=tensor_hash(self.write_roles),
                          active_training_devices=int((self.write_roles == 0).sum()),
                          reference_training_devices=int((self.write_roles == 1).sum()),
                          blocked_training_pairs=int((self.write_roles < 0).sum()),
                          write_role_map_bytes=self.write_roles.numel() * self.write_roles.element_size())
        if self.attempt_counts is not None:
            values.update(
                max_write_attempts_per_device=self.max_write_attempts_per_device,
                issued_write_attempts=int(self.attempt_counts.sum()),
                max_onchip_device_write_attempts=int(self.attempt_counts.max()),
                capped_devices=int((self.attempt_counts >= self.max_write_attempts_per_device).sum()),
            )
        return values

    def state_dict(self):
        return {**super().state_dict(), **self.diagnostics(),
                "pending": None if self.pending is None else self.pending.detach().cpu().clone(),
                **({"attempt_counts": self.attempt_counts.detach().cpu().clone()}
                   if self.attempt_counts is not None else {}),
                **({"write_role_map": self.write_roles.detach().cpu().clone()}
                   if self.write_roles is not None else {})}


def make_writer(array, lifecycle, *, write_roles=None):
    """The declared OM or CMO update law for a weight-learning arm."""

    onchip = lifecycle.onchip
    if onchip.update_law == "om_closed_loop_pulse_adam":
        return ClosedLoopPulseAdam(
            array.port(),
            learning_rate=onchip.learning_rate,
            tolerance=array.nominal_step * onchip.tolerance_steps,
            pulses_per_update=onchip.pulses_per_update,
            cap=onchip.max_pulses_per_cell,
        )
    if onchip.update_law == "om_open_loop_pulse_adam":
        return OpenLoopPulseAdam(
            array.port(), learning_rate=onchip.learning_rate, cap=onchip.max_pulses_per_cell, seed=onchip.seed
        )
    if onchip.update_law == "cmo_idealized_write_adam":
        return IdealizedWriteAdam(array, learning_rate=onchip.learning_rate,
                                  write_noise_scale=onchip.write_noise_scale)
    if onchip.update_law == "cmo_thresholded_write_adam":
        return ThresholdedWriteAdam(
            array, learning_rate=onchip.learning_rate, threshold_sigma=onchip.threshold_sigma,
            accumulate_updates=onchip.accumulate_updates,
            acceptance_percent=lifecycle.program_verify.acceptance_percent,
            noise_scale=lifecycle.devices.noise_scale,
            max_write_attempts_per_device=onchip.max_write_attempts_per_device,
            write_noise_scale=onchip.write_noise_scale,
            write_roles=write_roles,
        )
    raise ValueError(f"Expected an OM or CMO writer update law. Provided value: {onchip.update_law!r}.")


def evaluations(network, array, cache, lifecycle, device) -> dict:
    """Held apparent-state metrics, plus the named OM persistent diagnostic."""

    evaluate = family(lifecycle.network).evaluate
    q = array.read()
    split = lifecycle.data.evaluation
    values = {"development": evaluate(network, cache["development"], device, q=q)}
    if split == "test":
        values["test"] = evaluate(network, cache["test"], device, q=q)
    for name in data_stage.transfer_splits(lifecycle):
        values[name] = evaluate(network, cache[name], device, q=q)
    persistent = device_stage.persistent_state(array, lifecycle.devices.technology)
    if persistent is not None:
        values[f"persistent_{split}"] = evaluate(network, cache[split], device, q=persistent)
    return values


def train_arm(*, lifecycle, arm, network, array, target, cache, store, label, device, array_seed=None) -> dict:
    """Run one on-chip arm from the array's current (P0) state.

    An arm with a ``fault_rewrite`` first applies that one-shot CMO rewrite
    (``program_verify.fault_rewrite``); learning, its costs and any write-attempt
    cap then start from the rewritten state.
    """

    onchip, data = lifecycle.onchip, lifecycle.data
    network_family = family(lifecycle.network)
    technology = lifecycle.devices.technology
    om, cmo = technology == "om", technology == "cmo"
    # OM and CMO learn through a writer on the held state; PCM reprograms a master.
    direct = om or cmo
    learn, rewrite = arm.weights == "learn", arm.weights == "rewrite"
    network.enable_calibration(arm.calibration)
    network.q.requires_grad_(False)
    adapter = arm.digital_adapter
    if hasattr(network, "enable_lora"):
        network.enable_lora(adapter, onchip.seed)
    base_hash = network.base_state_hash() if adapter else None
    master = torch.nn.Parameter(target.clone(), requires_grad=learn)
    groups = []
    if learn and not direct:
        groups.append({"params": [master], "lr": onchip.learning_rate})
    if arm.calibration:
        groups.append({"params": network.calibration_parameters(), "lr": onchip.calibration_learning_rate})
    if adapter:
        groups.append({"params": network.lora.parameters(), "lr": adapter.learning_rate})
    optimizer = torch.optim.Adam(groups) if groups else None
    initial_hash = tensor_hash(array.read())
    rewritten = None
    controller = {} if arm.write_roles is not None else None
    if arm.fault_rewrite is not None:
        if not cmo or array_seed is None:
            raise ValueError("Expected a CMO array and its assignment seed for a fault rewrite.")
        rewritten = program_verify.fault_rewrite(array, target, lifecycle, arm.fault_rewrite, array_seed,
                                                controller_state=controller)
        store.append_metric({"stage": "onchip_fault_rewrite", "case": label, "arm": arm.arm_id, **rewritten})
    roles = None if controller is None else controller["write_roles"]
    writer = make_writer(array, lifecycle, write_roles=roles) if learn and direct else None
    tolerance = (
        array.nominal_step * onchip.tolerance_steps
        if om and onchip.update_law == "om_closed_loop_pulse_adam"
        else None
    )
    cap = onchip.max_pulses_per_cell if om else None
    baseline = array.cost()
    baseline_writes = array.write_counts(roles).clone() if cmo else None
    fixed = device_stage.fixed_state(array, technology)
    prefix = network.prefix_hash()
    rewrite_count = torch.zeros_like(target, dtype=torch.int64)
    rewrite_reads = 0
    training = cache["recovery_context" if lifecycle.recovery_views else onchip.cohort_key]
    store.append_metric({"stage": "onchip_cohort", "case": label, "arm": arm.arm_id,
                         "cohort": onchip.cohort_key, "examples": network_family.examples(training),
                         "epochs": onchip.epochs if arm.trains else 0, "batch_size": onchip.batch_size,
                         **({"recovery_view": training["cohort"]} if lifecycle.recovery_views else {})})
    curve = []
    global_step = 0
    for epoch in range(1, (onchip.epochs if arm.trains else 0) + 1):
        started = time.monotonic()
        order = torch.randperm(
            network_family.examples(training), generator=torch.Generator().manual_seed(data.data_seed + epoch)
        ) if lifecycle.recovery_views is None else context_views.order(lifecycle, epoch)
        network.train()
        loss_sum = 0.0
        for batch, begin in enumerate(range(0, len(order), onchip.batch_size)):
            index = order[begin : begin + onchip.batch_size]
            network.zero_grad(set_to_none=True)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            observed = array.read().detach().clone()
            if direct:
                q = observed.requires_grad_(learn)
            else:
                q = master + (observed - master).detach() if learn else observed
            microbatch = onchip.microbatch_size or len(index)
            loss_value = 0.0
            for offset in range(0, len(index), microbatch):
                micro_index = index[offset:offset + microbatch]
                x, t, y = network_family.batch(network, training, micro_index, device)
                logits = network.forward_features(x, q)
                if lifecycle.recovery_views:
                    mask = context_views.loss_mask(lifecycle, device)
                    logits, t, y = logits[:, mask], t[:, mask], y[:, mask]
                loss = network_family.objective(logits, t, y, onchip.objective)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite on-chip training objective.")
                fraction = len(micro_index) / len(index)
                # Sum normalized gradients; Adam and the physical writer run once.
                (loss * fraction).backward()
                loss_value += float(loss.detach()) * fraction
                del x, t, y, logits, loss
            if optimizer:
                optimizer.step()
            if writer:
                if onchip.lr_schedule is not None:
                    writer.lr_factor = onchip.lr_schedule.factor(global_step + 1)
                writer.step(q.grad)
            elif learn:
                with torch.no_grad():
                    master.clamp_(-1, 1)
                array.program(master.detach())
            elif rewrite and cmo:
                # Matched to learning: only active devices are rewritten.
                array.program(target, reference=False)
            elif rewrite and not om:
                array.program(target)
            elif rewrite:
                # The same per-update verify opportunity and cap as learning.
                with torch.no_grad():
                    port = array.port()
                    for _ in range(onchip.pulses_per_update):
                        error = target - port.read()
                        rewrite_reads += array.size
                        active = error.abs() > tolerance
                        if cap is not None:
                            active &= rewrite_count < cap
                        if not bool(active.any()):
                            break
                        port.pulse(error.sign().to(torch.int8) * active)
                        rewrite_count += active
            loss_sum += loss_value * len(index)
            global_step += 1
            if batch % 100 == 0:
                store.append_metric(
                    {"stage": "onchip", "case": label, "arm": arm.arm_id, "epoch": epoch, "batch": batch,
                     "train_loss": loss_value}
                )
            if onchip.development_every_steps and global_step % onchip.development_every_steps == 0:
                was_training = network.training
                development = network_family.evaluate(network, cache["development"], device, q=array.read())
                cost_now = {key: value - baseline[key] for key, value in array.cost().items()}
                store.append_metric({"stage": "onchip_development", "case": label, "arm": arm.arm_id,
                                     "epoch": epoch, "global_step": global_step,
                                     "development": development, "cost": cost_now,
                                     **({"lr_factor": writer.lr_factor}
                                        if writer is not None and onchip.lr_schedule is not None else {}),
                                     **({name: network_family.evaluate(network, cache[name], device, q=array.read())
                                         for name in context_views.extra_evaluation_names(lifecycle)
                                         if name.endswith("_development")}),
                                     **({"writer": writer.diagnostics()}
                                        if isinstance(writer, ThresholdedWriteAdam) else {})})
                network.train(was_training)
        device_stage.verify_fixed(array, technology, fixed)
        if network.prefix_hash() != prefix:
            raise RuntimeError("On-chip training changed the frozen digital prefix.")
        if adapter and network.base_state_hash() != base_hash:
            raise RuntimeError("Digital LoRA changed the frozen base model.")
        if adapter and (tensor_hash(array.read()) != initial_hash or array.cost() != baseline):
            raise RuntimeError("Digital LoRA changed the held physical deployment.")
        if rewritten and not learn and tensor_hash(array.read()) != rewritten["rewritten_apparent_sha256"]:
            raise RuntimeError("A held arm changed the array after its fault rewrite.")
        cost = {key: value - baseline[key] for key, value in array.cost().items()}
        if om:
            count = writer.count if writer else rewrite_count
            cost.update(
                verify_reads=writer.verify_reads if writer else rewrite_reads,
                capped_cells=0 if cap is None else int((count >= cap).sum()),
                max_onchip_cell_pulses=int(count.max()),
                probability_clipped=writer.probability_clipped if writer else 0,
            )
        if cmo:
            writes = array.write_counts(roles) - baseline_writes
            cost.update(
                touched_devices=int((writes > 0).sum()),
                max_onchip_device_writes=int(writes.max()),
                out_of_range_devices=array.out_of_range(),
            )
            if roles is not None:
                cost.update(active_device_writes=int(writes[roles == 0].sum()),
                            reference_device_writes=int(writes[roles == 1].sum()),
                            active_touched_devices=int(((writes > 0) & (roles == 0)).sum()),
                            reference_touched_devices=int(((writes > 0) & (roles == 1)).sum()))
            if isinstance(writer, ThresholdedWriteAdam) and writer.attempt_counts is not None:
                attempts = writer.attempt_counts
                if (bool((attempts > writer.max_write_attempts_per_device).any())
                        or bool((writes > attempts).any())
                        or int(attempts.sum()) != cost["device_writes"] + cost["stuck_write_attempts"]):
                    raise RuntimeError("CMO write-attempt cap or physical accounting violated.")
                cost.update(
                    issued_write_attempts=int(attempts.sum()),
                    max_onchip_device_write_attempts=int(attempts.max()),
                    capped_devices=int((attempts >= writer.max_write_attempts_per_device).sum()),
                )
                if roles is not None:
                    cost.update(active_write_attempts=int(attempts[roles == 0].sum()),
                                reference_write_attempts=int(attempts[roles == 1].sum()))
        row = {
            "epoch": epoch,
            "train_loss_mean": loss_sum / len(order),
            network_family.PRESENTATIONS: epoch * len(order),
            **(context_views.accounting(lifecycle, epoch) if lifecycle.recovery_views else {}),
            "cost": cost,
            **({"training_seconds": time.monotonic() - started} if lifecycle.recovery_views else {}),
            **evaluations(network, array, cache, lifecycle, device),
            "epoch_seconds": time.monotonic() - started,
        }
        curve.append(row)
        store.append_metric({"stage": "onchip_epoch", "case": label, "arm": arm.arm_id, **row})
        if torch.device(device).type == "cuda":
            store.append_metric({"stage": "onchip_memory", "case": label, "arm": arm.arm_id,
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        print(json.dumps({"case": label, "arm": arm.arm_id, "epoch": epoch,
                          "teacher_kl": row[lifecycle.data.evaluation]["teacher_kl"]}), flush=True)
    return {
        "arm": arm.arm_id,
        "weights": arm.weights,
        "calibration": arm.calibration,
        **({"write_roles": arm.write_roles, "write_routing": writer.diagnostics()}
           if arm.write_roles is not None else {}),
        **({"digital_adapter": {**adapter.to_dict(),
             "parameters": sum(p.numel() for p in network.lora.parameters()),
             "extra_macs_per_token": sum(p.numel() for p in network.lora.parameters()),
             "base_unchanged": True, "array_unchanged": True}} if adapter else {}),
        "curve": curve,
        **({"fault_rewrite": rewritten} if rewritten else {}),
        "initial_apparent_sha256": initial_hash,
        "final_apparent_sha256": tensor_hash(array.read()),
        "state": {
            "calibration": {
                name: p.detach().cpu().clone()
                for name, p in network.named_parameters()
                if name != "q" and not name.startswith("lora.") and p.requires_grad
            },
            **({"digital_adapter": {"config": adapter.to_dict(), "seed": onchip.seed,
                  "state": clone_cpu(network.lora.state_dict())}} if adapter else {}),
            "master_target": master.detach().cpu().clone(),
            "array": device_stage.array_state(array, technology, portable=bool(adapter)),
            "writer": None if writer is None else writer.state_dict(),
            "rewrite_count": rewrite_count.cpu(),
        },
    }


def restore(network, array, source_model, state, technology, target=None) -> None:
    """Rebuild a final on-chip state from its source model and saved tensors.

    ``target`` rebuilds a compact CMO deployment state (an arm that never wrote).
    """

    if hasattr(network, "enable_lora"):
        network.enable_lora(None)
    network.load_state_dict(source_model)
    if "digital_adapter" in state:
        from workflow.lifecycle import DigitalLora
        saved = state["digital_adapter"]
        config = {k: v for k, v in saved["config"].items() if k != "kind"}
        network.enable_lora(DigitalLora(**config), saved["seed"])
        network.lora.load_state_dict(saved["state"])
    with torch.no_grad():
        parameters = dict(network.named_parameters())
        for name, value in state["calibration"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    device_stage.load_array_state(array, state["array"], technology, target)
