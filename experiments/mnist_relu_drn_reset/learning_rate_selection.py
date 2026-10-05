"""RESET-based bounded learning-rate selection for the MNIST DRN."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import math
from typing import Any

import numpy as np
import torch

from experiments.mnist_shared import teacher_batches
from model.variable.parameter import Bias, DenseWeight
from training.core import engine
from training.core.engine import ExperimentComponents, GradientsReadyEvent
from training.core.guards import FiniteGradientGuard
from training.measured_trace import MeasuredTraceOptimizer


class LearningRateSafetyError(FloatingPointError):
    """A restarted canary or candidate crossed a terminal safety gate."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = dict(failure)
        super().__init__(f"Learning-rate safety rejection: {self.failure!r}")


def _rms(value: torch.Tensor) -> float:
    return float(
        torch.sqrt(value.detach().to(torch.float64).square().mean()).item()
    )


def _combined_rms(values: tuple[torch.Tensor, ...]) -> float:
    """Return an element-count-weighted RMS over one optimizer group."""

    if not values:
        raise ValueError(
            "Expected at least one tensor when computing a grouped RMS. "
            "Provided value: empty."
        )
    squared_sum = 0.0
    element_count = 0
    for value in values:
        resolved = value.detach().to(torch.float64)
        squared_sum += float(resolved.square().sum().item())
        element_count += resolved.numel()
    if element_count <= 0:
        raise ValueError(
            "Expected grouped RMS tensors to contain elements. "
            f"Provided value: element_count={element_count}."
        )
    return math.sqrt(squared_sum / element_count)


def _quantile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-30)


def _state_fingerprint(stack: Any) -> str:
    digest = sha256()
    for binding in stack.bundle.catalog.trainable:
        digest.update(binding.key.encode("utf-8"))
        value = binding.state.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    digest.update(repr(stack.optimizer.learning_rates()).encode("ascii"))
    return digest.hexdigest()


@dataclass
class RuntimeSnapshot:
    parameter_states: tuple[torch.Tensor, ...]
    layer_states: tuple[torch.Tensor, ...]
    optimizer_state: Mapping[str, Any]
    train_generator_state: torch.Tensor
    torch_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]
    fingerprint: str

    @classmethod
    def capture(cls, stack: Any, train_generator: torch.Generator) -> "RuntimeSnapshot":
        return cls(
            parameter_states=tuple(
                binding.state.detach().clone()
                for binding in stack.bundle.catalog.trainable
            ),
            layer_states=tuple(
                layer.state.detach().clone()
                for layer in stack.bundle.energy.layers()
            ),
            optimizer_state=deepcopy(stack.optimizer.state_dict()),
            train_generator_state=train_generator.get_state().detach().cpu().clone(),
            torch_rng_state=torch.get_rng_state().detach().cpu().clone(),
            cuda_rng_states=(
                tuple(state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else ()
            ),
            fingerprint=_state_fingerprint(stack),
        )

    def restore(self, stack: Any, train_generator: torch.Generator) -> None:
        bindings = stack.bundle.catalog.trainable
        layers = stack.bundle.energy.layers()
        if len(bindings) != len(self.parameter_states) or len(layers) != len(
            self.layer_states
        ):
            raise RuntimeError(
                "Expected stable parameter and layer counts during LR-selection "
                "restart. Provided value: "
                f"parameters={len(bindings)}/{len(self.parameter_states)}, "
                f"layers={len(layers)}/{len(self.layer_states)}."
            )
        with torch.no_grad():
            for binding, state in zip(bindings, self.parameter_states):
                binding.state.copy_(state)
        for layer, state in zip(layers, self.layer_states):
            layer.state = state.detach().clone()
        stack.optimizer.load_state_dict(deepcopy(self.optimizer_state))
        train_generator.set_state(self.train_generator_state)
        torch.set_rng_state(self.torch_rng_state)
        if self.cuda_rng_states:
            torch.cuda.set_rng_state_all(list(self.cuda_rng_states))
        restored = _state_fingerprint(stack)
        if restored != self.fingerprint:
            raise RuntimeError(
                "Expected exact RESET-state restoration during LR selection. "
                f"Provided value: expected={self.fingerprint!r}, restored={restored!r}."
            )


class SafetyMonitor:
    def __init__(self, names: tuple[str, ...], settings: Mapping[str, Any]) -> None:
        self.names = names
        self.warmup = int(settings["warmup_batches"])
        self.ema_decay = float(settings["loss_ema_decay"])
        self.loss_factor = float(settings["loss_growth_factor"])
        self.gradient_factor = float(settings["gradient_growth_factor"])
        self.persistence = int(settings["persistence_batches"])
        self.step = 0
        self.loss_ema: float | None = None
        self.loss_minimum = math.inf
        self.loss_streak = 0
        self.maximum_loss_streak = 0
        self.gradient_samples: dict[str, list[float]] = {
            name: [] for name in names
        }
        self.gradient_reference: dict[str, float] = {}
        self.gradient_streak: dict[str, int] = {name: 0 for name in names}
        self.maximum_gradient_streak: dict[str, int] = {
            name: 0 for name in names
        }
        self.maximum_gradient_rms: dict[str, float] = {
            name: 0.0 for name in names
        }
        self.failure: dict[str, Any] | None = None

    def observe(
        self,
        loss: float,
        gradients: tuple[torch.Tensor, ...],
        parameters: tuple[Any, ...],
    ) -> None:
        self.step += 1
        if not math.isfinite(loss):
            self._reject("nonfinite_loss", None)
        self.loss_ema = (
            loss
            if self.loss_ema is None
            else self.ema_decay * self.loss_ema + (1.0 - self.ema_decay) * loss
        )
        failures: list[dict[str, Any]] = []
        for name, parameter, gradient in zip(self.names, parameters, gradients):
            if not torch.isfinite(parameter.state).all():
                self._reject("nonfinite_parameter", name)
            if not torch.isfinite(gradient).all():
                self._reject("nonfinite_gradient", name)
            value = _rms(gradient)
            self.maximum_gradient_rms[name] = max(
                self.maximum_gradient_rms[name], value
            )
            if self.step <= self.warmup:
                self.gradient_samples[name].append(value)
                if self.step == self.warmup:
                    self.gradient_reference[name] = _quantile(
                        self.gradient_samples[name], 0.5
                    )
            else:
                reference = self.gradient_reference[name]
                if value > self.gradient_factor * max(reference, 1e-30):
                    self.gradient_streak[name] += 1
                    self.maximum_gradient_streak[name] = max(
                        self.maximum_gradient_streak[name],
                        self.gradient_streak[name],
                    )
                    if self.gradient_streak[name] == self.persistence:
                        failures.append(
                            self._failure("gradient_rms_explosion", name)
                        )
                else:
                    self.gradient_streak[name] = 0
        if self.step <= self.warmup:
            self.loss_minimum = min(self.loss_minimum, self.loss_ema)
        else:
            if self.loss_ema > self.loss_factor * self.loss_minimum:
                self.loss_streak += 1
                self.maximum_loss_streak = max(
                    self.maximum_loss_streak, self.loss_streak
                )
                if self.loss_streak == self.persistence:
                    failures.append(self._failure("loss_ema_explosion", None))
            else:
                self.loss_streak = 0
            self.loss_minimum = min(self.loss_minimum, self.loss_ema)
        if failures:
            failure = min(
                failures,
                key=lambda item: (
                    item["onset_step"],
                    item["kind"],
                    item["parameter"] or "",
                ),
            )
            self.failure = failure
            raise LearningRateSafetyError(failure)

    def finish(self, parameters: tuple[Any, ...]) -> None:
        if self.step < self.warmup:
            raise RuntimeError(
                "Expected LR safety run to reach its warmup budget. "
                f"Provided value: processed={self.step}, warmup={self.warmup}."
            )
        for name, parameter in zip(self.names, parameters):
            if not torch.isfinite(parameter.state).all():
                self._reject("nonfinite_parameter", name)

    def summary(self) -> dict[str, Any]:
        return {
            "processed_batches": self.step,
            "failure": self.failure,
            "loss_ema": self.loss_ema,
            "loss_ema_minimum": (
                None if not math.isfinite(self.loss_minimum) else self.loss_minimum
            ),
            "maximum_loss_streak": self.maximum_loss_streak,
            "gradient_reference_rms": dict(self.gradient_reference),
            "maximum_gradient_rms": dict(self.maximum_gradient_rms),
            "maximum_gradient_streak": dict(self.maximum_gradient_streak),
        }

    def _failure(self, kind: str, parameter: str | None) -> dict[str, Any]:
        return {
            "kind": kind,
            "parameter": parameter,
            "onset_step": max(1, self.step - self.persistence + 1),
            "confirmed_step": self.step,
        }

    def _reject(self, kind: str, parameter: str | None) -> None:
        self.failure = {
            "kind": kind,
            "parameter": parameter,
            "onset_step": max(1, self.step),
            "confirmed_step": max(1, self.step),
        }
        raise LearningRateSafetyError(self.failure)


def _parameter_topology(
    stack: Any,
    *,
    training_reset_input: bool | None = None,
) -> dict[str, Any]:
    bindings = tuple(stack.bundle.catalog.trainable)
    parameters = tuple(stack.bundle.energy.params())
    names = tuple(binding.key for binding in bindings)
    if len(bindings) != len(parameters) or any(
        binding.parameter is not parameter
        for binding, parameter in zip(bindings, parameters)
    ):
        raise RuntimeError(
            "Expected stable catalog and energy-parameter alignment during "
            "RESET LR selection. Provided value: "
            f"bindings={len(bindings)}, parameters={len(parameters)}."
        )
    binding_by_state_id = {id(binding.state): binding for binding in bindings}
    group_members: list[tuple[str, ...]] = []
    covered: list[str] = []
    for group_index, group in enumerate(stack.optimizer.param_groups):
        states = tuple(group.get("params", ()))
        members: list[str] = []
        for state in states:
            binding = binding_by_state_id.get(id(state))
            if binding is None:
                raise RuntimeError(
                    "Expected every RESET optimizer state to have a stable "
                    "catalog binding. Provided value: "
                    f"group={group_index}, state_id={id(state)}."
                )
            members.append(binding.key)
        if not members:
            raise RuntimeError(
                "Expected every RESET optimizer group to contain parameters. "
                f"Provided value: group={group_index}."
            )
        group_members.append(tuple(members))
        covered.extend(members)
    if len(covered) != len(set(covered)) or set(covered) != set(names):
        raise RuntimeError(
            "Expected RESET optimizer groups to cover every catalog binding "
            "exactly once. Provided value: "
            f"groups={group_members!r}, bindings={names!r}."
        )
    weight_names = tuple(
        name
        for name, parameter in zip(names, parameters)
        if isinstance(parameter, DenseWeight)
    )
    bias_names = tuple(
        name
        for name, parameter in zip(names, parameters)
        if isinstance(parameter, Bias)
    )
    by_name = dict(zip(names, parameters))
    binding_by_name = {binding.key: binding for binding in bindings}
    if (
        len(weight_names) == 2
        and len(bias_names) in {0, 1}
        and len(names) == len(weight_names) + len(bias_names)
        and all(len(members) == 1 for members in group_members)
    ):
        encoding = "single"
        rate_group_names = tuple(members[0] for members in group_members)
        weight_group_names = weight_names
        bias_group_names = bias_names
        bias_to_weight = (
            {} if not bias_names else {bias_names[0]: weight_names[0]}
        )
    elif (
        len(weight_names) == 4
        and not bias_names
        and len(names) == len(weight_names)
        and len(group_members) == 2
        and all(len(members) == 2 for members in group_members)
        and all(
            tuple(binding_by_name[name].role for name in members)
            == ("conductance_plus", "conductance_minus")
            for members in group_members
        )
    ):
        encoding = "differential"
        rate_group_names = tuple(
            f"base.differential_pair.{index}"
            for index in range(len(group_members))
        )
        weight_group_names = rate_group_names
        bias_group_names = ()
        bias_to_weight = {}
    else:
        raise ValueError(
            "Expected the RESET topology to expose either two independently "
            "optimized dense weights with zero or one hidden bias, or two "
            "logical differential groups containing ordered G+/G- branches. "
            "Provided value: "
            f"weights={weight_names!r}, biases={bias_names!r}, "
            f"groups={group_members!r}."
        )
    group_members_by_name = dict(zip(rate_group_names, group_members))
    return {
        "encoding": encoding,
        "names": names,
        "parameters": parameters,
        "by_name": by_name,
        "rate_group_names": rate_group_names,
        "group_members": group_members_by_name,
        "weight_names": weight_group_names,
        "bias_names": bias_group_names,
        "bias_to_weight": bias_to_weight,
        # Existing experiment versions infer the historical behavior from
        # topology. Controlled experiments pass this protocol choice
        # explicitly so it cannot be coupled to the bias factor.
        "training_reset_input": (
            not bool(bias_names)
            if training_reset_input is None
            else training_reset_input
        ),
    }


def _run_probe(
    stack: Any,
    teacher: Any,
    train_loader: Any,
    settings: Mapping[str, Any],
    *,
    training_reset_input: bool | None = None,
) -> dict[str, Any]:
    topology = _parameter_topology(
        stack, training_reset_input=training_reset_input
    )
    names = topology["names"]
    initial_weight_rms = {
        name: _combined_rms(
            tuple(
                topology["by_name"][member].state
                for member in topology["group_members"][name]
            )
        )
        for name in topology["weight_names"]
    }
    invalid = {
        name: value
        for name, value in initial_weight_rms.items()
        if not math.isfinite(value) or value <= 0.0
    }
    if invalid:
        raise RuntimeError(
            "Expected every RESET conductance RMS to be positive and finite. "
            f"Provided value: {invalid!r}."
        )
    samples = {name: [] for name in topology["rate_group_names"]}
    checkpoints = tuple(int(value) for value in settings["probe_batches"])
    stable = False
    used = 0
    split_statistics: dict[str, Any] = {}
    tolerance = float(settings["stability_tolerance"])
    bias_quantile = float(settings["bias_quantile"])

    def measure(event: Any) -> None:
        nonlocal used, stable, split_statistics
        if not isinstance(event, GradientsReadyEvent):
            return
        gradient_by_name = dict(zip(names, event.gradients))
        for name in topology["rate_group_names"]:
            attached = topology["bias_to_weight"].get(name, name)
            grouped_gradient = tuple(
                gradient_by_name[member]
                for member in topology["group_members"][name]
            )
            samples[name].append(
                _combined_rms(grouped_gradient) / initial_weight_rms[attached]
            )
        used += 1
        if used not in checkpoints:
            return
        half = used // 2
        split_statistics = {}
        unstable = []
        for name, values in samples.items():
            quantile = (
                bias_quantile
                if name in topology["bias_names"]
                else 0.5
            )
            first = _quantile(values[:half], quantile)
            second = _quantile(values[half:used], quantile)
            difference = _relative_difference(first, second)
            split_statistics[name] = {
                "first_half": first,
                "second_half": second,
                "relative_difference": difference,
            }
            if difference > tolerance:
                unstable.append(name)
        stable = not unstable

    # A gradient pass with no update. The loader stops before pulling the
    # next batch once the estimate is stable at a checkpoint, and the
    # parameters' gradients are left as they were before the probe.
    parameters = topology["parameters"]
    previous_gradients = [parameter.state.grad for parameter in parameters]
    try:
        engine.train_epoch(
            ExperimentComponents(
                stack.network,
                stack.cost,
                stack.minimizer,
                parameters,
                stack.differentiator,
                None,
            ),
            teacher_batches(
                train_loader,
                teacher,
                stack.device,
                maximum_batches=checkpoints[-1],
                stop=lambda: stable,
            ),
            event_handlers=(
                FiniteGradientGuard(names, label="LR-selection"),
                measure,
            ),
            reset_input=topology["training_reset_input"],
        )
    finally:
        for parameter, gradient in zip(parameters, previous_gradients):
            parameter.state.grad = gradient
    weight_units = {
        name: _quantile(samples[name][:used], 0.5)
        for name in topology["weight_names"]
    }
    bias_units = {
        name: _quantile(samples[name][:used], bias_quantile)
        for name in topology["bias_names"]
    }
    invalid_units = {
        name: value
        for name, value in weight_units.items()
        if not math.isfinite(value) or value <= 0.0
    }
    if used not in checkpoints or not stable or invalid_units:
        raise RuntimeError(
            "Expected a stable positive bounded relative-update probe by the "
            "last configured checkpoint. Provided value: "
            f"used_batches={used}, stable={stable}, "
            f"invalid_units={invalid_units!r}, split_statistics={split_statistics!r}."
        )
    return {
        "schema": "mnist_relu_drn_reset.bounded-relative-update-probe",
        "schema_version": 1,
        "status": "complete",
        "nominal_optimizer": "SGD",
        "nominal_learning_rate": 1.0,
        "optimizer_steps_applied": False,
        "parameter_names": list(names),
        "rate_group_names": list(topology["rate_group_names"]),
        "rate_group_members": {
            name: list(members)
            for name, members in topology["group_members"].items()
        },
        "weight_names": list(topology["weight_names"]),
        "bias_to_weight": dict(topology["bias_to_weight"]),
        "used_batches": used,
        "configured_batch_checkpoints": list(checkpoints),
        "stability_tolerance": tolerance,
        "split_half_statistics": split_statistics,
        "initial_weight_rms_s": initial_weight_rms,
        "normalization_unit_by_weight": weight_units,
        "bias_q90_unit_by_parameter": bias_units,
        "proposal_unit_samples_by_parameter": {
            name: values[:used] for name, values in samples.items()
        },
        "proposal_unit_samples_by_rate_group": {
            name: values[:used] for name, values in samples.items()
        },
        "official_test_read": False,
    }


def _derive_rates(
    probe: Mapping[str, Any], relative_targets: tuple[float, float]
) -> tuple[dict[str, float], tuple[float, ...]]:
    names = tuple(probe.get("rate_group_names", probe["parameter_names"]))
    weight_names = tuple(probe["weight_names"])
    if len(weight_names) != 2 or len(relative_targets) != 2:
        raise ValueError(
            "Expected two weight names and two relative-update targets. "
            f"Provided value: weights={weight_names!r}, "
            f"targets={relative_targets!r}."
        )
    units = probe["normalization_unit_by_weight"]
    target_by_weight = dict(zip(weight_names, relative_targets))
    by_name = {
        name: float(target_by_weight[name]) / float(units[name])
        for name in weight_names
    }
    for bias, attached in probe["bias_to_weight"].items():
        bias_unit = float(probe["bias_q90_unit_by_parameter"][bias])
        attached_rate = by_name[attached]
        target = float(target_by_weight[attached])
        by_name[bias] = (
            attached_rate
            if bias_unit == 0.0
            else min(attached_rate, target / bias_unit)
        )
    rates = tuple(float(by_name[name]) for name in names)
    if any(not math.isfinite(value) or value < 0.0 for value in rates):
        raise RuntimeError(
            "Expected derived learning rates to be finite and non-negative. "
            f"Provided value: {rates!r}."
        )
    return by_name, rates


def _changed_write_count(report: Mapping[str, Any]) -> float:
    steps = int(report["step_count"])
    total = 0.0
    for parameter in report["parameters"].values():
        fraction = parameter.get("pulse_changed_fraction")
        if fraction is None:
            continue
        elements = int(np.prod(parameter["shape"], dtype=np.int64))
        total += float(fraction) * elements * steps
    return total


def select_reset_learning_rates(
    spec: Any,
    stack: Any,
    teacher: Any,
    data: Any,
    *,
    store: Any,
    train_epoch: Callable[..., tuple[dict[str, Any], int]],
    evaluate: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Probe, safety-screen a 3x3 grid, and restore the selected RESET state."""

    if not isinstance(stack.optimizer, MeasuredTraceOptimizer):
        raise TypeError(
            "Expected RESET LR selection to receive a MeasuredTraceOptimizer. "
            f"Provided value: {type(stack.optimizer).__name__}."
        )
    if stack.optimizer.initialized:
        raise RuntimeError(
            "Expected LR selection to own pulse-zero initialization. "
            "Provided value: optimizer is already initialized."
        )
    settings = dict(spec.settings.learning_rate_selection.parameters)
    settings["safety"] = dict(settings["safety"])
    initialization = stack.optimizer.initialize_at_pulse_zero()
    snapshot = RuntimeSnapshot.capture(stack, data.train_generator)
    training_reset_input = spec.settings.reset_input_between_batches
    topology = _parameter_topology(
        stack, training_reset_input=training_reset_input
    )
    names = topology["names"]
    parameters = topology["parameters"]
    training_reset_input = topology["training_reset_input"]

    probe = _run_probe(
        stack,
        teacher,
        data.train,
        settings,
        training_reset_input=training_reset_input,
    )
    store.append_metric(
        {
            "mode": "learning_rate_selection_probe",
            "used_batches": probe["used_batches"],
            "normalization_unit_by_weight": probe[
                "normalization_unit_by_weight"
            ],
            "bias_q90_unit_by_parameter": probe[
                "bias_q90_unit_by_parameter"
            ],
        }
    )
    snapshot.restore(stack, data.train_generator)

    def run_canary(rates: tuple[float, ...]) -> dict[str, Any]:
        stack.optimizer.set_learning_rates(rates)
        monitor = SafetyMonitor(names, settings["safety"])
        try:
            _metrics, batches = train_epoch(
                stack,
                teacher,
                data.train,
                maximum_batches=int(settings["canary_batches"]),
                observer=monitor.observe,
                reset_input=training_reset_input,
            )
            monitor.finish(parameters)
            if batches != int(settings["canary_batches"]):
                raise RuntimeError(
                    "Expected the safety canary to execute exactly "
                    f"{settings['canary_batches']} batches. Provided value: {batches}."
                )
        except LearningRateSafetyError as error:
            return {
                "status": "rejected_safety",
                "failure": dict(error.failure),
                "safety": monitor.summary(),
            }
        return {"status": "clean", "failure": None, "safety": monitor.summary()}

    center = tuple(
        float(value) for value in settings["weight_relative_update_targets"]
    )
    center_attempts: list[dict[str, Any]] = []
    canary_cache: dict[tuple[float, float], dict[str, Any]] = {}
    for reduction in range(int(settings["max_center_reductions"]) + 1):
        by_name, rates = _derive_rates(probe, center)
        snapshot.restore(stack, data.train_generator)
        canary = run_canary(rates)
        canary_cache[center] = canary
        attempt = {
            "reduction": reduction,
            "relative_update_targets": list(center),
            "learning_rates_by_parameter": by_name,
            "learning_rate_vector": list(rates),
            "canary": canary,
        }
        center_attempts.append(attempt)
        store.append_metric(
            {"mode": "learning_rate_selection_center_canary", **attempt}
        )
        if canary["status"] == "clean":
            break
        center = (center[0] / 3.0, center[1] / 3.0)
    if center_attempts[-1]["canary"]["status"] != "clean":
        raise RuntimeError(
            "Expected a safety-clean LR-grid center within the configured "
            f"reduction budget. Provided value: {center_attempts!r}."
        )

    multipliers = tuple(float(value) for value in settings["grid_multipliers"])
    cells: list[dict[str, Any]] = []
    for index, (first, second) in enumerate(product(multipliers, repeat=2)):
        targets = (center[0] * first, center[1] * second)
        by_name, rates = _derive_rates(probe, targets)
        canary = canary_cache.get(targets)
        if canary is None:
            snapshot.restore(stack, data.train_generator)
            canary = run_canary(rates)
            canary_cache[targets] = canary
        cell: dict[str, Any] = {
            "id": f"cell_{index:02d}",
            "grid_indices": [multipliers.index(first), multipliers.index(second)],
            "multipliers": [first, second],
            "relative_update_targets": list(targets),
            "learning_rates_by_parameter": by_name,
            "learning_rate_vector": list(rates),
            "canary": canary,
        }
        store.append_metric(
            {"mode": "learning_rate_selection_canary", **cell}
        )
        if canary["status"] != "clean":
            cell["candidate"] = {
                "status": "skipped_unsafe_canary",
                "failure": canary["failure"],
            }
            cells.append(cell)
            continue
        snapshot.restore(stack, data.train_generator)
        stack.optimizer.set_learning_rates(rates)
        monitor = SafetyMonitor(names, settings["safety"])
        epochs: list[dict[str, Any]] = []
        failure = None
        try:
            for epoch in range(int(settings["candidate_epochs"])):
                trained, batches = train_epoch(
                    stack,
                    teacher,
                    data.train,
                    maximum_batches=spec.settings.max_batches,
                    observer=monitor.observe,
                    reset_input=training_reset_input,
                )
                validation = evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                epochs.append(
                    {
                        "epoch": epoch,
                        "batches": batches,
                        "train": trained,
                        "validation_examples": validation["examples"],
                        "validation_objective_loss": validation["objective_loss"],
                        "validation_accuracy": validation["student_accuracy"],
                        "validation_kl_teacher_student": validation[
                            "kl_teacher_student"
                        ],
                    }
                )
            monitor.finish(parameters)
        except LearningRateSafetyError as error:
            failure = dict(error.failure)
        if failure is not None:
            candidate = {
                "status": "rejected_safety",
                "failure": failure,
                "epochs": epochs,
                "safety": monitor.summary(),
            }
        else:
            programming = stack.optimizer.programming_report
            candidate = {
                "status": "complete",
                "failure": None,
                "epochs": epochs,
                "final_validation_objective_loss": epochs[-1][
                    "validation_objective_loss"
                ],
                "final_validation_accuracy": epochs[-1]["validation_accuracy"],
                "changed_write_count": _changed_write_count(programming),
                "safety": monitor.summary(),
                "projection": programming,
            }
        cell["candidate"] = candidate
        cells.append(cell)
        store.append_metric(
            {
                "mode": "learning_rate_selection_candidate",
                "id": cell["id"],
                "relative_update_targets": cell["relative_update_targets"],
                "learning_rate_vector": cell["learning_rate_vector"],
                "status": candidate["status"],
                "failure": candidate.get("failure"),
                "epochs": candidate.get("epochs", []),
            }
        )

    complete = [
        cell for cell in cells if cell["candidate"]["status"] == "complete"
    ]
    if not complete:
        raise RuntimeError(
            "Expected at least one safety-clean completed LR-grid candidate. "
            f"Provided value: {cells!r}."
        )
    minimum_loss = min(
        cell["candidate"]["final_validation_objective_loss"]
        for cell in complete
    )
    plateau_limit = minimum_loss * (
        1.0 + float(settings["plateau_relative_tolerance"])
    )
    plateau = [
        cell
        for cell in complete
        if cell["candidate"]["final_validation_objective_loss"] <= plateau_limit
    ]
    selected = min(
        plateau,
        key=lambda cell: (
            -cell["candidate"]["final_validation_accuracy"],
            cell["candidate"]["changed_write_count"],
            max(cell["relative_update_targets"]),
            sum(cell["relative_update_targets"]),
            cell["id"],
        ),
    )
    selected_rates = tuple(
        float(value) for value in selected["learning_rate_vector"]
    )
    snapshot.restore(stack, data.train_generator)
    stack.optimizer.set_learning_rates(selected_rates)
    if _state_fingerprint(stack) == snapshot.fingerprint:
        raise RuntimeError(
            "Expected the selected learning rates to change the RESET snapshot "
            "fingerprint. Provided value: unchanged."
        )
    edge_selected = any(
        item in {0, len(multipliers) - 1}
        for item in selected["grid_indices"]
    )
    return {
        "schema": "mnist_relu_drn_reset.measured-cohort-a-lr-selection",
        "schema_version": 1,
        "status": "complete",
        "evidence_class": "exploratory",
        "objective": spec.settings.objective,
        "active_cohort": "A",
        "initialization": initialization,
        "reset_state_fingerprint": snapshot.fingerprint,
        "official_test_read": False,
        "probe": probe,
        "initial_relative_update_targets": list(
            settings["weight_relative_update_targets"]
        ),
        "resolved_center_relative_update_targets": list(center),
        "center_reductions": len(center_attempts) - 1,
        "center_attempts": center_attempts,
        "grid_multipliers": list(multipliers),
        "cells": cells,
        "selection": {
            "metric": "final_validation_objective_loss",
            "minimum_loss": minimum_loss,
            "inclusive_plateau_relative_tolerance": float(
                settings["plateau_relative_tolerance"]
            ),
            "plateau_limit": plateau_limit,
            "plateau_cell_ids": [cell["id"] for cell in plateau],
            "tie_break": [
                "higher_final_validation_accuracy",
                "fewer_changed_writes",
                "lower_max_relative_update_target",
                "lower_sum_relative_update_targets",
                "stable_cell_id",
            ],
            "selected_cell_id": selected["id"],
            "selected_relative_update_targets": selected[
                "relative_update_targets"
            ],
            "selected_learning_rates_by_parameter": selected[
                "learning_rates_by_parameter"
            ],
            "selected_learning_rate_vector": list(selected_rates),
            "selected_final_validation_objective_loss": selected[
                "candidate"
            ]["final_validation_objective_loss"],
            "selected_final_validation_accuracy": selected["candidate"][
                "final_validation_accuracy"
            ],
            "selected_changed_write_count": selected["candidate"][
                "changed_write_count"
            ],
            "edge_selected": edge_selected,
            "grid_expanded": False,
        },
    }


__all__ = [
    "LearningRateSafetyError",
    "RuntimeSnapshot",
    "SafetyMonitor",
    "select_reset_learning_rates",
]
