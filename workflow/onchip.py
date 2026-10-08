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
OM open-loop pulse Adam (stochastic one-pulse rounding, no verify reads).
``max_pulses_per_cell`` caps on-chip writes per cell; ``null`` is uncapped.
"""

from __future__ import annotations

import json
import time

import torch

from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.model import tensor_hash
from workflow import devices as device_stage
from workflow.networks import family


def _adam_command(writer, gradient):
    if gradient.shape != writer.first.shape or not bool(torch.isfinite(gradient).all()):
        raise ValueError("Expected a finite normalized gradient for every crosspoint.")
    writer.step_index += 1
    writer.first.mul_(0.9).add_(gradient, alpha=0.1)
    writer.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
    return (
        -writer.learning_rate
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


def make_writer(array, lifecycle):
    """The declared OM update law for a weight-learning arm."""

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
    raise ValueError(f"Expected an OM pulse update law. Provided value: {onchip.update_law!r}.")


def evaluations(network, array, cache, lifecycle, device) -> dict:
    """Held apparent-state metrics, plus the named OM persistent diagnostic."""

    evaluate = family(lifecycle.network).evaluate
    q = array.read()
    split = lifecycle.data.evaluation
    values = {"development": evaluate(network, cache["development"], device, q=q)}
    if split == "test":
        values["test"] = evaluate(network, cache["test"], device, q=q)
    persistent = device_stage.persistent_state(array, lifecycle.devices.technology)
    if persistent is not None:
        values[f"persistent_{split}"] = evaluate(network, cache[split], device, q=persistent)
    return values


def train_arm(*, lifecycle, arm, network, array, target, cache, store, label, device) -> dict:
    """Run one on-chip arm from the array's current (P0) state."""

    onchip, data = lifecycle.onchip, lifecycle.data
    network_family = family(lifecycle.network)
    technology = lifecycle.devices.technology
    om = technology == "om"
    learn, rewrite = arm.weights == "learn", arm.weights == "rewrite"
    network.enable_calibration(arm.calibration)
    network.q.requires_grad_(False)
    master = torch.nn.Parameter(target.clone(), requires_grad=learn)
    groups = []
    if learn and not om:
        groups.append({"params": [master], "lr": onchip.learning_rate})
    if arm.calibration:
        groups.append({"params": network.calibration_parameters(), "lr": onchip.calibration_learning_rate})
    optimizer = torch.optim.Adam(groups) if groups else None
    writer = make_writer(array, lifecycle) if learn and om else None
    tolerance = (
        array.nominal_step * onchip.tolerance_steps
        if om and onchip.update_law == "om_closed_loop_pulse_adam"
        else None
    )
    cap = onchip.max_pulses_per_cell if om else None
    baseline = array.cost()
    fixed = device_stage.fixed_state(array, technology)
    prefix = network.prefix_hash()
    initial_hash = tensor_hash(array.read())
    rewrite_count = torch.zeros_like(target, dtype=torch.int64)
    rewrite_reads = 0
    training = cache["training"]
    curve = []
    for epoch in range(1, (onchip.epochs if arm.trains else 0) + 1):
        started = time.monotonic()
        order = torch.randperm(
            network_family.examples(training), generator=torch.Generator().manual_seed(data.data_seed + epoch)
        )
        network.train()
        loss_sum = 0.0
        for batch, begin in enumerate(range(0, len(order), onchip.batch_size)):
            index = order[begin : begin + onchip.batch_size]
            x, t, y = network_family.batch(network, training, index, device)
            network.zero_grad(set_to_none=True)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            observed = array.read().detach().clone()
            if om:
                q = observed.requires_grad_(learn)
            else:
                q = master + (observed - master).detach() if learn else observed
            loss = network_family.objective(network.forward_features(x, q), t, y, onchip.objective)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite on-chip training objective.")
            loss.backward()
            if optimizer:
                optimizer.step()
            if writer:
                writer.step(q.grad)
            elif learn:
                with torch.no_grad():
                    master.clamp_(-1, 1)
                array.program(master.detach())
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
            loss_sum += float(loss.detach()) * len(index)
            if batch % 100 == 0:
                store.append_metric(
                    {"stage": "onchip", "case": label, "arm": arm.arm_id, "epoch": epoch, "batch": batch,
                     "train_loss": float(loss.detach())}
                )
        device_stage.verify_fixed(array, technology, fixed)
        if network.prefix_hash() != prefix:
            raise RuntimeError("On-chip training changed the frozen digital prefix.")
        cost = {key: value - baseline[key] for key, value in array.cost().items()}
        if om:
            count = writer.count if writer else rewrite_count
            cost.update(
                verify_reads=writer.verify_reads if writer else rewrite_reads,
                capped_cells=0 if cap is None else int((count >= cap).sum()),
                max_onchip_cell_pulses=int(count.max()),
                probability_clipped=writer.probability_clipped if writer else 0,
            )
        row = {
            "epoch": epoch,
            "train_loss_mean": loss_sum / len(order),
            network_family.PRESENTATIONS: epoch * len(order),
            "cost": cost,
            **evaluations(network, array, cache, lifecycle, device),
            "epoch_seconds": time.monotonic() - started,
        }
        curve.append(row)
        store.append_metric({"stage": "onchip_epoch", "case": label, "arm": arm.arm_id, **row})
        print(json.dumps({"case": label, "arm": arm.arm_id, "epoch": epoch,
                          "teacher_kl": row[lifecycle.data.evaluation]["teacher_kl"]}), flush=True)
    return {
        "arm": arm.arm_id,
        "weights": arm.weights,
        "calibration": arm.calibration,
        "curve": curve,
        "initial_apparent_sha256": initial_hash,
        "final_apparent_sha256": tensor_hash(array.read()),
        "state": {
            "calibration": {
                name: p.detach().cpu().clone()
                for name, p in network.named_parameters()
                if name != "q" and p.requires_grad
            },
            "master_target": master.detach().cpu().clone(),
            "array": device_stage.array_state(array, technology),
            "writer": None if writer is None else writer.state_dict(),
            "rewrite_count": rewrite_count.cpu(),
        },
    }


def restore(network, array, source_model, state, technology) -> None:
    """Rebuild a final on-chip state from its source model and saved tensors."""

    network.load_state_dict(source_model)
    with torch.no_grad():
        parameters = dict(network.named_parameters())
        for name, value in state["calibration"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    device_stage.load_array_state(array, state["array"], technology)
