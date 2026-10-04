"""Bounded relative-update learning-rate selection for measured ReRAM runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import islice, product
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.small_network.config import TrainSpec
from model.variable.parameter import Bias, DenseWeight, PoolWeight
from training.core.batch import as_batch
from training.core.engine import (
    FreePhaseEvent,
    GradientsReadyEvent,
    evaluate,
    train_epoch,
)
from training.checkpoint import load_named_weights
from training.measured_trace import MeasuredTraceOptimizer
from training.core.probes import MeanCostProbe, MeanErrorProbe


class LearningRateSafetyError(FloatingPointError):
    """A restarted canary or candidate crossed a terminal safety gate."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = dict(failure)
        super().__init__(f"Learning-rate safety rejection: {self.failure!r}")


def _rms(value: torch.Tensor) -> float:
    return float(
        torch.sqrt(value.detach().to(torch.float64).square().mean()).item()
    )


def _quantile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-30)


@dataclass
class _RuntimeSnapshot:
    parameter_states: tuple[torch.Tensor, ...]
    optimizer_state: Mapping[str, Any]
    layer_state: Mapping[str, Any]
    generator_states: Mapping[str, torch.Tensor]

    @classmethod
    def capture(cls, runtime: Any) -> "_RuntimeSnapshot":
        parameters = runtime.training_components.parameters
        return cls(
            parameter_states=tuple(
                parameter.state.detach().clone() for parameter in parameters
            ),
            optimizer_state=deepcopy(runtime.optimizer.state_dict()),
            layer_state=deepcopy(runtime.runtime_state.state_dict()),
            generator_states={
                name: generator.get_state().detach().cpu().clone()
                for name, generator in runtime.data.dataloader_generators.items()
            },
        )

    def restore(self, runtime: Any) -> None:
        parameters = runtime.training_components.parameters
        if len(parameters) != len(self.parameter_states):
            raise RuntimeError(
                "Expected stable parameter count during LR-selection restart. "
                f"Provided value: initial={len(self.parameter_states)}, "
                f"current={len(parameters)}."
            )
        with torch.no_grad():
            for parameter, state in zip(parameters, self.parameter_states):
                parameter.state.copy_(state)
        runtime.optimizer.load_state_dict(deepcopy(self.optimizer_state))
        runtime.runtime_state.load_state_dict(deepcopy(self.layer_state))
        for name, generator in runtime.data.dataloader_generators.items():
            generator.set_state(self.generator_states[name])


class _SafetyMonitor:
    def __init__(self, parameters: Mapping[str, Any], settings: Mapping[str, Any]):
        self.parameters = dict(parameters)
        self.warmup = int(settings["warmup_batches"])
        self.ema_decay = float(settings["loss_ema_decay"])
        self.loss_factor = float(settings["loss_growth_factor"])
        self.gradient_factor = float(settings["gradient_growth_factor"])
        self.persistence = int(settings["persistence_batches"])
        self.step = 0
        self.pending_loss: float | None = None
        self.loss_ema: float | None = None
        self.loss_minimum = math.inf
        self.loss_streak = 0
        self.maximum_loss_streak = 0
        self.gradient_samples: dict[str, list[float]] = {}
        self.gradient_reference: dict[str, float] = {}
        self.gradient_streak: dict[str, int] = {}
        self.maximum_gradient_streak: dict[str, int] = {}
        self.maximum_gradient_rms: dict[str, float] = {}
        self.failure: dict[str, Any] | None = None

    def __call__(self, event: Any) -> None:
        if isinstance(event, FreePhaseEvent):
            loss = float(event.components.cost_fn.eval().mean().item())
            if not math.isfinite(loss):
                self._reject("nonfinite_loss", None)
            self.pending_loss = loss
            return
        if not isinstance(event, GradientsReadyEvent):
            return
        if self.pending_loss is None:
            raise RuntimeError(
                "Expected a free-phase loss before LR safety gradient checks. "
                "Provided value: None."
            )
        self.step += 1
        loss = self.pending_loss
        self.pending_loss = None
        self.loss_ema = (
            loss
            if self.loss_ema is None
            else self.ema_decay * self.loss_ema
            + (1.0 - self.ema_decay) * loss
        )
        failures: list[dict[str, Any]] = []
        for name, parameter in self.parameters.items():
            if not torch.isfinite(parameter.state).all():
                self._reject("nonfinite_parameter", name)
        for parameter, gradient in zip(
            event.components.parameters,
            event.gradients,
        ):
            if not isinstance(gradient, torch.Tensor) or not torch.isfinite(
                gradient
            ).all():
                self._reject("nonfinite_gradient", self._name(parameter))
            if not isinstance(parameter, DenseWeight):
                continue
            name = self._name(parameter)
            gradient_rms = _rms(gradient)
            self.maximum_gradient_rms[name] = max(
                self.maximum_gradient_rms.get(name, 0.0),
                gradient_rms,
            )
            if self.step <= self.warmup:
                self.gradient_samples.setdefault(name, []).append(gradient_rms)
                if self.step == self.warmup:
                    self.gradient_reference[name] = _quantile(
                        self.gradient_samples[name], 0.5
                    )
                    self.gradient_streak[name] = 0
                    self.maximum_gradient_streak[name] = 0
            else:
                reference = self.gradient_reference[name]
                if gradient_rms > self.gradient_factor * reference:
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
                    self.maximum_loss_streak,
                    self.loss_streak,
                )
                if self.loss_streak == self.persistence:
                    failures.append(self._failure("loss_ema_explosion", None))
            else:
                self.loss_streak = 0
            self.loss_minimum = min(self.loss_minimum, self.loss_ema)
        if failures:
            failure = min(
                failures,
                key=lambda value: (
                    value["onset_step"],
                    value["kind"],
                    value["parameter"] or "",
                ),
            )
            self.failure = failure
            raise LearningRateSafetyError(failure)

    def finish(self) -> None:
        for name, parameter in self.parameters.items():
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

    def _name(self, parameter: Any) -> str:
        for name, candidate in self.parameters.items():
            if candidate is parameter:
                return name
        return type(parameter).__name__

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


def _limited(loader: Iterable[Any], maximum: int | None) -> Iterable[Any]:
    return loader if maximum is None else islice(loader, maximum)


def _parameter_topology(runtime: Any) -> dict[str, Any]:
    bindings = {
        id(binding.parameter): binding.key
        for binding in runtime.stack.bundle.catalog.trainable
    }
    parameters = tuple(
        parameter
        for parameter in runtime.training_components.parameters
        if not isinstance(parameter, PoolWeight)
    )
    if len(parameters) != len(runtime.optimizer.param_groups):
        raise RuntimeError(
            "Expected one optimizer group per LR-selection parameter. "
            f"Provided value: parameters={len(parameters)}, "
            f"groups={len(runtime.optimizer.param_groups)}."
        )
    names = tuple(
        bindings.get(id(parameter), f"cost.{type(parameter).__name__.lower()}.{index}")
        for index, parameter in enumerate(parameters)
    )
    if len(set(names)) != len(names):
        raise RuntimeError(
            "Expected stable unique parameter names for LR selection. "
            f"Provided value: {names!r}."
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
    if len(weight_names) != 2 or len(bias_names) != 1:
        raise ValueError(
            "Expected the measured-cohort MNIST topology to contain exactly "
            "two dense weights and one hidden bias. Provided value: "
            f"weights={weight_names!r}, biases={bias_names!r}."
        )
    return {
        "parameters": parameters,
        "names": names,
        "by_name": dict(zip(names, parameters)),
        "weight_names": weight_names,
        "bias_names": bias_names,
        "bias_to_weight": {bias_names[0]: weight_names[0]},
    }


def _run_probe(runtime: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    topology = _parameter_topology(runtime)
    initial_weight_rms = {
        name: _rms(topology["by_name"][name].state)
        for name in topology["weight_names"]
    }
    invalid_rms = {
        name: value
        for name, value in initial_weight_rms.items()
        if not math.isfinite(value) or value <= 0.0
    }
    if invalid_rms:
        raise RuntimeError(
            "Expected every measured initial weight RMS to be positive and "
            f"finite. Provided value: {invalid_rms!r}."
        )
    samples = {name: [] for name in topology["names"]}
    checkpoints = tuple(int(value) for value in settings["probe_batches"])
    maximum = checkpoints[-1]
    stability_tolerance = float(settings["stability_tolerance"])
    bias_quantile = float(settings["bias_quantile"])
    split_statistics: dict[str, Any] = {}
    stable = False
    used = 0
    components = runtime.training_components
    for raw_batch in islice(runtime.data.train_loader, maximum):
        batch = as_batch(raw_batch)
        components.network.set_input(batch.inputs, reset=False)
        components.energy_minimizer.compute_equilibrium()
        components.cost_fn.set_target(batch.targets)
        gradients = tuple(components.differentiator.compute_gradient())
        if len(gradients) != len(components.parameters):
            raise RuntimeError(
                "Expected one gradient per parameter during the LR probe. "
                f"Provided value: {len(gradients)}."
            )
        gradient_by_identity = {
            id(parameter): gradient
            for parameter, gradient in zip(components.parameters, gradients)
        }
        for name, parameter in zip(topology["names"], topology["parameters"]):
            gradient = gradient_by_identity[id(parameter)]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    "Expected finite gradients during the LR probe. "
                    f"Provided value: parameter={name!r}."
                )
            attached = topology["bias_to_weight"].get(name, name)
            samples[name].append(_rms(gradient) / initial_weight_rms[attached])
        used += 1
        if used not in checkpoints:
            continue
        half = used // 2
        split_statistics = {}
        unstable = []
        for name, values in samples.items():
            quantile = bias_quantile if name in topology["bias_names"] else 0.5
            first = _quantile(values[:half], quantile)
            second = _quantile(values[half:used], quantile)
            difference = _relative_difference(first, second)
            split_statistics[name] = {
                "first_half": first,
                "second_half": second,
                "relative_difference": difference,
            }
            if difference > stability_tolerance:
                unstable.append(name)
        stable = not unstable
        if stable:
            break
    if used not in checkpoints:
        raise RuntimeError(
            "Expected the training loader to reach an LR-probe checkpoint. "
            f"Provided value: used_batches={used}, checkpoints={checkpoints!r}."
        )
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
    if not stable or invalid_units:
        raise RuntimeError(
            "Expected a stable positive bounded relative-update probe by the "
            f"last configured checkpoint. Provided value: stable={stable}, "
            f"invalid_weight_units={invalid_units!r}, "
            f"split_statistics={split_statistics!r}."
        )
    return {
        "schema": "small_drn.bounded-relative-update-probe",
        "schema_version": 1,
        "status": "complete",
        "nominal_optimizer": "SGD",
        "nominal_learning_rate": 1.0,
        "optimizer_steps_applied": False,
        "parameter_names": list(topology["names"]),
        "weight_names": list(topology["weight_names"]),
        "bias_to_weight": dict(topology["bias_to_weight"]),
        "used_batches": used,
        "configured_batch_checkpoints": list(checkpoints),
        "stability_tolerance": stability_tolerance,
        "split_half_statistics": split_statistics,
        "initial_weight_rms_s": initial_weight_rms,
        "normalization_unit_by_weight": weight_units,
        "bias_q90_unit_by_parameter": bias_units,
        "proposal_unit_samples_by_parameter": {
            name: values[:used] for name, values in samples.items()
        },
        "official_test_read": False,
    }


def _derive_rates(
    probe: Mapping[str, Any],
    relative_targets: tuple[float, float],
) -> tuple[dict[str, float], tuple[float, ...]]:
    names = tuple(probe["parameter_names"])
    weight_names = tuple(probe["weight_names"])
    units = probe["normalization_unit_by_weight"]
    by_name: dict[str, float] = {}
    target_by_weight = dict(zip(weight_names, relative_targets))
    for name in weight_names:
        by_name[name] = float(target_by_weight[name]) / float(units[name])
    for bias, attached in probe["bias_to_weight"].items():
        bias_unit = float(probe["bias_q90_unit_by_parameter"][bias])
        attached_rate = by_name[attached]
        target = target_by_weight[attached]
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


def _run_canary(
    spec: TrainSpec,
    runtime: Any,
    *,
    rates: tuple[float, ...],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    runtime.optimizer.set_learning_rates(rates)
    topology = _parameter_topology(runtime)
    monitor = _SafetyMonitor(topology["by_name"], settings["safety"])
    status = "clean"
    failure = None
    try:
        trained = train_epoch(
            runtime.training_components,
            islice(runtime.data.train_loader, int(settings["canary_batches"])),
            event_handlers=(monitor,),
            epoch=0,
            start_global_step=0,
            reset_input=False,
        )
        monitor.finish()
        if trained.batch_count != int(settings["canary_batches"]):
            raise RuntimeError(
                "Expected the safety canary to execute exactly "
                f"{settings['canary_batches']} batches. Provided value: "
                f"{trained.batch_count}."
            )
    except LearningRateSafetyError as error:
        status = "rejected_safety"
        failure = dict(error.failure)
    return {
        "status": status,
        "failure": failure,
        "safety": monitor.summary(),
    }


def _run_candidate(
    spec: TrainSpec,
    runtime: Any,
    *,
    rates: tuple[float, ...],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    runtime.optimizer.set_learning_rates(rates)
    topology = _parameter_topology(runtime)
    monitor = _SafetyMonitor(topology["by_name"], settings["safety"])
    epochs: list[dict[str, Any]] = []
    global_step = 0
    try:
        for epoch in range(int(settings["candidate_epochs"])):
            trained = train_epoch(
                runtime.training_components,
                _limited(runtime.data.train_loader, spec.settings.max_batches),
                event_handlers=(monitor,),
                epoch=epoch,
                start_global_step=global_step,
                reset_input=False,
            )
            global_step = trained.next_global_step
            validation = evaluate(
                runtime.evaluation_components,
                _limited(
                    runtime.data.held_out_loader,
                    spec.settings.max_validation_batches,
                ),
                probes=(MeanCostProbe(), MeanErrorProbe()),
                epoch=epoch,
                split="lr_selection_validation",
                reset_input=True,
            )
            cost = validation.probe_value("mean_cost")["mean"]
            error = validation.probe_value("mean_error")["mean"]
            if (
                cost is None
                or error is None
                or not math.isfinite(float(cost))
                or not math.isfinite(float(error))
            ):
                raise RuntimeError(
                    "Expected LR candidate validation to produce finite cost "
                    "and error measurements. Provided value: "
                    f"examples={validation.example_count}, cost={cost!r}, "
                    f"error={error!r}."
                )
            epochs.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "validation_examples": validation.example_count,
                    "validation_mean_cost": float(cost),
                    "validation_accuracy": 1.0 - float(error),
                }
            )
        monitor.finish()
    except LearningRateSafetyError as error:
        return {
            "status": "rejected_safety",
            "failure": dict(error.failure),
            "epochs": epochs,
            "safety": monitor.summary(),
        }
    return {
        "status": "complete",
        "failure": None,
        "epochs": epochs,
        "final_validation_mean_cost": epochs[-1]["validation_mean_cost"],
        "final_validation_accuracy": epochs[-1]["validation_accuracy"],
        "safety": monitor.summary(),
        "projection": runtime.optimizer.programming_report,
    }


def select_measured_learning_rates(
    spec: TrainSpec,
    runtime: Any,
    *,
    store: Any,
    initial_weights_path: Path | None = None,
) -> dict[str, Any]:
    """Probe, safety-screen a 3x3 grid, and choose a production LR vector."""

    if not isinstance(runtime.optimizer, MeasuredTraceOptimizer):
        raise TypeError(
            "Expected measured LR selection to receive a "
            "MeasuredTraceOptimizer. Provided value: "
            f"{type(runtime.optimizer).__name__}."
        )
    selection = spec.settings.learning_rate_selection
    if selection.type != "bounded_relative_update_grid":
        raise ValueError(
            "Expected bounded_relative_update_grid LR selection. Provided "
            f"value: {selection.type!r}."
        )
    settings = dict(selection.parameters)
    settings["safety"] = dict(settings["safety"])
    if runtime.optimizer.cohort == "A":
        if initial_weights_path is not None:
            raise ValueError(
                "Expected cohort-A LR selection without initial weights. "
                f"Provided value: {str(initial_weights_path)!r}."
            )
        runtime.optimizer.initialize_at_pulse_zero()
    else:
        if initial_weights_path is None:
            raise ValueError(
                "Expected cohort-B LR selection to receive a named cohort-A "
                "weights path. Provided value: None."
            )
        load_named_weights(
            initial_weights_path,
            runtime.stack.bundle.catalog,
        )
        runtime.optimizer.initialize_from_loaded_targets()
    snapshot = _RuntimeSnapshot.capture(runtime)

    probe = _run_probe(runtime, settings)
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
    snapshot.restore(runtime)

    center = tuple(
        float(value) for value in settings["weight_relative_update_targets"]
    )
    if len(center) != 2:
        raise ValueError(
            "Expected exactly two relative-update targets. Provided value: "
            f"{center!r}."
        )
    center_attempts: list[dict[str, Any]] = []
    canary_cache: dict[tuple[float, float], dict[str, Any]] = {}
    for reduction in range(int(settings["max_center_reductions"]) + 1):
        rates_by_name, rates = _derive_rates(probe, center)
        snapshot.restore(runtime)
        canary = _run_canary(
            spec,
            runtime,
            rates=rates,
            settings=settings,
        )
        canary_cache[center] = canary
        attempt = {
            "reduction": reduction,
            "relative_update_targets": list(center),
            "learning_rates_by_parameter": rates_by_name,
            "learning_rate_vector": list(rates),
            "canary": canary,
        }
        center_attempts.append(attempt)
        store.append_metric(
            {
                "mode": "learning_rate_selection_center_canary",
                **attempt,
            }
        )
        if canary["status"] == "clean":
            break
        center = tuple(value / 3.0 for value in center)
    if center_attempts[-1]["canary"]["status"] != "clean":
        raise RuntimeError(
            "Expected a safety-clean LR-grid center within the configured "
            f"reduction budget. Provided value: {center_attempts!r}."
        )

    multipliers = tuple(float(value) for value in settings["grid_multipliers"])
    cells: list[dict[str, Any]] = []
    for index, (first, second) in enumerate(product(multipliers, repeat=2)):
        targets = (center[0] * first, center[1] * second)
        rates_by_name, rates = _derive_rates(probe, targets)
        canary = canary_cache.get(targets)
        if canary is None:
            snapshot.restore(runtime)
            canary = _run_canary(
                spec,
                runtime,
                rates=rates,
                settings=settings,
            )
            canary_cache[targets] = canary
        cell = {
            "id": f"cell_{index:02d}",
            "grid_indices": [
                multipliers.index(first),
                multipliers.index(second),
            ],
            "multipliers": [first, second],
            "relative_update_targets": list(targets),
            "learning_rates_by_parameter": rates_by_name,
            "learning_rate_vector": list(rates),
            "canary": canary,
        }
        store.append_metric(
            {
                "mode": "learning_rate_selection_canary",
                **cell,
            }
        )
        if canary["status"] != "clean":
            cell["candidate"] = {
                "status": "skipped_unsafe_canary",
                "failure": canary["failure"],
            }
            cells.append(cell)
            continue
        snapshot.restore(runtime)
        candidate = _run_candidate(
            spec,
            runtime,
            rates=rates,
            settings=settings,
        )
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
        cell["candidate"]["final_validation_mean_cost"] for cell in complete
    )
    plateau_limit = minimum_loss * (
        1.0 + float(settings["plateau_relative_tolerance"])
    )
    plateau = [
        cell
        for cell in complete
        if cell["candidate"]["final_validation_mean_cost"] <= plateau_limit
    ]
    selected = min(
        plateau,
        key=lambda cell: (
            -cell["candidate"]["final_validation_accuracy"],
            max(cell["relative_update_targets"]),
            sum(cell["relative_update_targets"]),
            cell["relative_update_targets"][0],
            cell["relative_update_targets"][1],
            cell["id"],
        ),
    )
    edge_selected = any(
        index in {0, len(multipliers) - 1}
        for index in selected["grid_indices"]
    )
    return {
        "schema": (
            "small_drn.measured-cohort-"
            f"{runtime.optimizer.cohort.lower()}-lr-selection"
        ),
        "schema_version": 1,
        "status": "complete",
        "evidence_class": "exploratory",
        "active_cohort": runtime.optimizer.cohort,
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
            "metric": "final_validation_mean_cost",
            "minimum_loss": minimum_loss,
            "inclusive_plateau_relative_tolerance": float(
                settings["plateau_relative_tolerance"]
            ),
            "plateau_limit": plateau_limit,
            "plateau_cell_ids": [cell["id"] for cell in plateau],
            "tie_break": [
                "higher_final_validation_accuracy",
                "lower_max_relative_update_target",
                "lower_sum_relative_update_targets",
                "lower_first_weight_target",
                "lower_second_weight_target",
                "stable_cell_id",
            ],
            "selected_cell_id": selected["id"],
            "selected_relative_update_targets": selected[
                "relative_update_targets"
            ],
            "selected_learning_rates_by_parameter": selected[
                "learning_rates_by_parameter"
            ],
            "selected_learning_rate_vector": selected["learning_rate_vector"],
            "selected_final_validation_mean_cost": selected["candidate"][
                "final_validation_mean_cost"
            ],
            "selected_final_validation_accuracy": selected["candidate"][
                "final_validation_accuracy"
            ],
            "edge_selected": edge_selected,
            "grid_expanded": False,
        },
    }


__all__ = [
    "LearningRateSafetyError",
    "select_measured_learning_rates",
]
