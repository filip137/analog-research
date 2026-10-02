"""Pulse-compatible recovery from one exact two-state IBM OM deployment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Iterator

import torch

from model.resistive.builders import ParameterBinding
from training.ibm_om.plants import IbmReramPulsePlant
from training.ibm_om.population import (
    IbmReramArrayPopulation,
    _selected_array_population,
)
from training.ibm_om.programming.controller import (
    adaptive_controller_settings,
    step_estimator,
    step_ratio_tolerance,
)
from training.ibm_om.sampling import (
    _artifact as load_ibm_reram_hwa_artifact,
    sample_om_array_population_layout_external,
)
from training.ibm_reram_program_verify import (
    ControllerSettings,
    PopulationStepEstimator,
    ProgramVerifyResult,
    derive_seed,
    run_program_verify,
)


RECOVERY_SCHEMA = "ebl.ibm_reram.om_deployed_recovery"
RECOVERY_SCHEMA_VERSION = 1


def _plain_data(value: Any) -> Any:
    """Detach strict config data from nested read-only mapping proxies."""

    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "Expected deployed-recovery parameters to contain strict JSON data. "
        f"Provided value: {type(value).__name__}."
    )


def _tensor_summary(value: torch.Tensor) -> dict[str, float | int | None]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "count": int(flat.numel()),
        "minimum": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "maximum": float(flat.max().item()),
    }


def _result_cost(result: ProgramVerifyResult) -> dict[str, int]:
    return {
        "devices": int(result.accepted.numel()),
        "accepted": int(result.accepted.sum().item()),
        "failed": int((~result.accepted).sum().item()),
        "set_pulses": int(result.set_count.sum().item()),
        "reset_pulses": int(result.reset_count.sum().item()),
        "total_pulses": int(result.total_pulses.sum().item()),
        "verify_reads": int(result.verify_count.sum().item()),
        "reversals": int(result.reversals.sum().item()),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
    }


def sample_physical_fast_population(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
    aihwkit_python: Path,
    population_path: Path,
    receipt_path: Path,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    """Sample two repaired OM fast rails for every slow conductance."""

    selected = tuple(bindings)
    keys = tuple(f"fast_pair.{binding.key}" for binding in selected)
    shapes = tuple(
        (2 * int(binding.state.shape[0]), int(binding.state.shape[1]))
        for binding in selected
    )
    return sample_om_array_population_layout_external(
        keys,
        shapes,
        assignment_seed=assignment_seed,
        corruption_policy="counterfactual_repaired",
        aihwkit_python=aihwkit_python,
        population_path=population_path,
        receipt_path=receipt_path,
    )


class _RecoveryLedger:
    def __init__(self, size: int, *, device: torch.device) -> None:
        self.set_count = torch.zeros(size, dtype=torch.int64, device=device)
        self.reset_count = torch.zeros(size, dtype=torch.int64, device=device)
        self.state_change_count = torch.zeros(size, dtype=torch.int64, device=device)
        self.saturated_count = torch.zeros(size, dtype=torch.int64, device=device)
        self.clipped_count = torch.zeros(size, dtype=torch.int64, device=device)

    def record(
        self,
        indices: torch.Tensor,
        directions: torch.Tensor,
        *,
        changed: torch.Tensor,
        saturated: torch.Tensor,
        clipped: torch.Tensor,
    ) -> None:
        if indices.numel() == 0:
            return
        ones = torch.ones(indices.numel(), dtype=torch.int64, device=indices.device)
        self.set_count.index_add_(0, indices, ones * (directions > 0).to(torch.int64))
        self.reset_count.index_add_(0, indices, ones * (directions < 0).to(torch.int64))
        self.state_change_count.index_add_(0, indices, changed.to(torch.int64))
        self.saturated_count.index_add_(0, indices, saturated.to(torch.int64))
        self.clipped_count.index_add_(0, indices, clipped.to(torch.int64))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).detach().cpu().clone()
            for name in (
                "set_count",
                "reset_count",
                "state_change_count",
                "saturated_count",
                "clipped_count",
            )
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "set_count",
            "reset_count",
            "state_change_count",
            "saturated_count",
            "clipped_count",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Expected an exact deployed-recovery pulse ledger.")
        for name in expected:
            value = state[name]
            current = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != current.shape
                or value.dtype != torch.int64
                or bool(torch.any(value < 0))
            ):
                raise ValueError(f"Expected a valid recovery ledger tensor {name!r}.")
            current.copy_(value.to(current.device))

    def summary(self) -> dict[str, Any]:
        total = self.set_count + self.reset_count
        return {
            "set_requested": int(self.set_count.sum().item()),
            "reset_requested": int(self.reset_count.sum().item()),
            "total_requested": int(total.sum().item()),
            "state_changing": int(self.state_change_count.sum().item()),
            "saturated": int(self.saturated_count.sum().item()),
            "clipped_apparent": int(self.clipped_count.sum().item()),
            "unique_cells_touched": int((total > 0).sum().item()),
            "per_cell_requested": _tensor_summary(total),
        }


class IbmOmDeployedRecovery:
    """Optimizer-like state machine for pulse-compatible deployed recovery.

    ``tiki_taka`` is retained as the historical direct-transfer pilot and
    ``ttv2`` as the completed lossless-buffer diagnostic.  The separately
    named ``ttv2_aihwkit_1p1_minibatch_equation`` control pins the native
    ChoppedTransfer buffer/LR equation while declaring its DRN adaptations:
    one averaged gradient per minibatch, an ideal continuous fast array, and
    grouped dual-rail input scans.
    """

    def __init__(
        self,
        bindings: Sequence[ParameterBinding],
        *,
        slow_population: IbmReramArrayPopulation,
        source_raw_apparent: torch.Tensor,
        source_apparent: torch.Tensor,
        source_persistent: torch.Tensor,
        requested_target: torch.Tensor,
        reset_baseline: torch.Tensor,
        generator_state_after_programming: torch.Tensor,
        parameters: Mapping[str, Any],
        learning_rates: Sequence[float],
        conductance_min: float,
        conductance_max: float,
        total_steps: int,
        source_deployment_sha256: str,
        device_model_path: Path,
        fast_population: IbmReramArrayPopulation | None = None,
        fast_population_receipt: Mapping[str, Any] | None = None,
    ) -> None:
        self.bindings = tuple(bindings)
        self.parameters = _plain_data(parameters)
        self.learning_rates = tuple(float(value) for value in learning_rates)
        self.conductance_min = float(conductance_min)
        self.conductance_max = float(conductance_max)
        self.span = self.conductance_max - self.conductance_min
        self.total_steps = int(total_steps)
        self.source_deployment_sha256 = str(source_deployment_sha256)
        self.device_model_path = device_model_path.expanduser().resolve()
        self.device_model, self.device_model_sha256 = load_ibm_reram_hwa_artifact(
            self.device_model_path
        )
        if self.device_model_sha256 != self.parameters["expected_device_model_sha256"]:
            raise ValueError("Expected the recovery device-model digest to match config.")
        if self.total_steps < 1 or self.span <= 0.0:
            raise ValueError("Expected positive recovery steps and conductance span.")
        if len(self.bindings) != len(self.learning_rates):
            raise ValueError("Expected one recovery learning rate per binding.")
        self.keys = tuple(binding.key for binding in self.bindings)
        self.shapes = tuple(tuple(binding.state.shape) for binding in self.bindings)
        self.sizes = tuple(int(binding.state.numel()) for binding in self.bindings)
        self.sections: tuple[slice, ...] = tuple()
        sections = []
        offset = 0
        for size in self.sizes:
            sections.append(slice(offset, offset + size))
            offset += size
        self.sections = tuple(sections)
        self.size = offset
        if (
            self.keys != slow_population.binding_keys
            or self.shapes != slow_population.binding_shapes
            or self.size != slow_population.size
        ):
            raise ValueError("Expected recovery bindings to match the deployment population.")
        self.device = self.bindings[0].state.device
        self.slow_population = slow_population.to(self.device)
        vectors = {
            "source_raw_apparent": source_raw_apparent,
            "source_apparent": source_apparent,
            "source_persistent": source_persistent,
            "requested_target": requested_target,
            "reset_baseline": reset_baseline,
        }
        normalized = {}
        for name, value in vectors.items():
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            if tensor.shape != (self.size,) or not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError(f"Expected finite recovery vector {name!r}.")
            normalized[name] = tensor.clone()
        self.source_raw_apparent = normalized["source_raw_apparent"]
        self.source_apparent = normalized["source_apparent"]
        self.source_persistent = normalized["source_persistent"]
        self.requested_target = normalized["requested_target"]
        self.reset_baseline = normalized["reset_baseline"]
        if not torch.equal(self.source_apparent, self.source_raw_apparent.clamp(0.0, 1.0)):
            raise ValueError("Expected source apparent clipping parity.")

        self.slow_generator = torch.Generator(device=self.device)
        self.slow_generator.manual_seed(1)
        self.slow_plant = IbmReramPulsePlant(
            self.slow_population,
            generator=self.slow_generator,
            device=self.device,
        )
        self.slow_generator.set_state(generator_state_after_programming.detach().cpu())
        self.slow_plant.load_state(
            persistent=2.0 * self.source_persistent - 1.0,
            apparent=2.0 * self.source_raw_apparent - 1.0,
        )
        self.selection_generator = torch.Generator(device=self.device)
        self.selection_generator.manual_seed(int(self.parameters["update_seed"]))
        self.slow_ledger = _RecoveryLedger(self.size, device=self.device)
        self.step_index = 0
        budget = float(self.parameters["slow_pulse_budget_per_cell"])
        raw_cap = budget * self.size
        if not math.isclose(raw_cap, round(raw_cap), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Expected an integral full-run slow-pulse cap.")
        self.slow_pulse_cap = int(round(raw_cap))
        self.method = str(self.parameters["method"])
        self.pulse_step_ratio = float(self.parameters["pulse_step_ratio"])
        self.direct_probability_scale = self.parameters.get(
            "direct_probability_scale"
        )
        self.direct_probability_report: dict[str, Any] | None = None
        self.expected_direct_pulses = 0.0
        self.direct_gradient_thresholds: dict[str, float] | None = None
        self.direct_gradient_threshold_report: dict[str, Any] | None = None
        self.direct_gate_observed = torch.zeros(
            len(self.bindings), dtype=torch.int64, device=self.device
        )
        self.direct_gate_nonzero = torch.zeros_like(self.direct_gate_observed)
        self.direct_gate_eligible = torch.zeros_like(self.direct_gate_observed)
        self.direct_gate_suppressed = torch.zeros_like(self.direct_gate_observed)
        self.refresh_order: torch.Tensor | None = None
        self.ideal_fast: list[torch.Tensor] | None = None
        self.ttv2_hidden: list[torch.Tensor] | None = None
        self.ttv2_transfer_scale: list[float] = []
        self.ttv2_buffer_scale_uncorrected: list[float] = []
        self.ttv2_buffer_scale_corrected: list[float] = []
        self.ttv2_effective_transfer_lr: list[float] = []
        self.ttv2_threshold_crossings = 0
        self.ttv2_dispatched_pulses = 0
        self.ttv2_cap_deferred_crossings = 0
        self.ttv2_fast_saturation_observations = 0
        self.ttv2_buffer_input_abs = 0.0
        self.ttv2_parameter_pulse_caps: list[int] = []
        self.ttv2_parameter_deferred_crossings: list[int] = [
            0 for _ in self.bindings
        ]
        self.ttv2_parameter_freeze_step: list[int | None] = [
            None for _ in self.bindings
        ]
        self.ttv2_parameter_skipped_steps: list[int] = [
            0 for _ in self.bindings
        ]
        self.transfer_start: list[int] = []
        self.transfer_count: list[int] = [0 for _ in self.bindings]
        self.transfer_opportunities = 0
        self.fast_population = fast_population
        self.fast_population_receipt = (
            deepcopy(dict(fast_population_receipt))
            if fast_population_receipt is not None
            else None
        )
        self.fast_plant: IbmReramPulsePlant | None = None
        self.fast_generator: torch.Generator | None = None
        self.fast_baseline: torch.Tensor | None = None
        self.fast_plus_indices: list[torch.Tensor] = []
        self.fast_minus_indices: list[torch.Tensor] = []
        self.fast_ledger: _RecoveryLedger | None = None
        self.fast_reset_verify_reads = 0
        self.fast_reset_reversals = 0
        self.fast_reset_failures = 0
        self.fast_commissioning: dict[str, Any] | None = None

        if self.method == "rail_refresh":
            self.refresh_order = self._build_refresh_order()
        elif self.method == "tiki_taka":
            self.transfer_start = [
                derive_seed(int(self.parameters["update_seed"]), key, "column_start")
                % shape[1]
                for key, shape in zip(self.keys, self.shapes)
            ]
            if self.parameters["fast_backend"] == "ideal":
                self.ideal_fast = [torch.zeros_like(binding.state) for binding in self.bindings]
            else:
                if self.fast_population is None:
                    raise ValueError("Expected a physical-fast OM population.")
                self._initialize_physical_fast()
        elif self.method in {"ttv2", "ttv2_aihwkit_1p1_minibatch_equation"}:
            if self.parameters.get("fast_backend") != "ideal":
                raise ValueError("Expected the first TTv2 control to use an ideal fast array.")
            layouts = dict(
                self.parameters.get("source_dual_rail_layout_by_parameter", {})
            )
            if set(layouts) != set(self.keys):
                raise ValueError("Expected one declared dual-rail layout per TTv2 binding.")
            for key, (rows, columns) in zip(self.keys, self.shapes):
                if rows % 2 or columns % 2 or layouts[key] not in {"halves", "paired"}:
                    raise ValueError(
                        "Expected even four-cell dual-rail tensors for TTv2."
                    )
            if self.method == "ttv2":
                self.transfer_start = [
                    derive_seed(
                        int(self.parameters["update_seed"]),
                        key,
                        "ttv2_input_start",
                    )
                    % (shape[0] // 2)
                    for key, shape in zip(self.keys, self.shapes)
                ]
            else:
                self.transfer_start = [0 for _ in self.bindings]
            self.ideal_fast = [
                torch.zeros_like(binding.state) for binding in self.bindings
            ]
            self.ttv2_hidden = [
                torch.zeros_like(binding.state) for binding in self.bindings
            ]
            transfer_every = int(self.parameters["ttv2_transfer_every"])
            if self.method == "ttv2":
                gamma0 = float(self.parameters["ttv2_gamma0"])
                self.ttv2_transfer_scale = [
                    transfer_every * (shape[0] // 2)
                    / (gamma0 * self.pulse_step_ratio)
                    for shape in self.shapes
                ]
            else:
                raw_transfer_every = self.parameters["ttv2_transfer_every"]
                if (
                    isinstance(raw_transfer_every, bool)
                    or not isinstance(raw_transfer_every, int)
                    or raw_transfer_every < 1
                ):
                    raise ValueError(
                        "Expected a positive integer TTv2 minibatch transfer period."
                    )
                if self.parameters["ttv2_units_in_mbatch"] is not True:
                    raise ValueError("Expected TTv2 transfer units to be minibatches.")
                if self.parameters["ttv2_auto_scale"] is not False:
                    raise ValueError("Expected TTv2 automatic scaling to be disabled.")
                if self.parameters["ttv2_correct_gradient_magnitudes"] is not True:
                    raise ValueError(
                        "Expected TTv2 corrected gradient magnitudes to be enabled."
                    )
                if self.parameters["ttv2_desired_bl"] != 1 or isinstance(
                    self.parameters["ttv2_desired_bl"], bool
                ):
                    raise ValueError("Expected the one-pulse TTv2 desired BL.")
                if float(self.parameters["ttv2_momentum"]) != 0.0:
                    raise ValueError("Expected zero TTv2 buffer momentum.")
                if self.parameters["ttv2_forget_buffer"] is not True:
                    raise ValueError("Expected native TTv2 buffer forgetting.")
                if self.parameters["ttv2_cap_scope"] != "per_parameter_proportional":
                    raise ValueError("Expected proportional per-parameter TTv2 caps.")
                if self.parameters["ttv2_cursor_policy"] != "zero":
                    raise ValueError("Expected every TTv2 scan cursor to begin at zero.")
                if float(self.parameters["ttv2_in_chop_probability"]) != 0.0:
                    raise ValueError("Expected input chopping to be disabled.")
                if not isinstance(self.parameters["ttv2_scale_transfer_lr"], bool):
                    raise ValueError("Expected a Boolean TTv2 transfer-LR scaling flag.")
                fast_lrs = dict(
                    self.parameters["ttv2_fast_lr_by_parameter"]
                )
                fast_granularities = dict(
                    self.parameters["ttv2_fast_granularity_by_parameter"]
                )
                if set(fast_lrs) != set(self.keys) or set(
                    fast_granularities
                ) != set(self.keys):
                    raise ValueError(
                        "Expected exact per-parameter TTv2 fast LR and granularity mappings."
                    )
                transfer_lr = float(self.parameters["ttv2_transfer_lr"])
                scale_transfer_lr = bool(
                    self.parameters["ttv2_scale_transfer_lr"]
                )
                buffer_granularity = float(
                    self.parameters["ttv2_buffer_granularity"]
                )
                auto_granularity = float(
                    self.parameters["ttv2_auto_granularity"]
                )
                correct_magnitudes = bool(
                    self.parameters["ttv2_correct_gradient_magnitudes"]
                )
                positive_controls = {
                    "ttv2_transfer_lr": transfer_lr,
                    "ttv2_buffer_granularity": buffer_granularity,
                    "ttv2_auto_granularity": auto_granularity,
                    "ttv2_fast_weight_limit": float(
                        self.parameters["ttv2_fast_weight_limit"]
                    ),
                    "pulse_step_ratio": self.pulse_step_ratio,
                }
                if any(
                    not math.isfinite(value) or value <= 0.0
                    for value in positive_controls.values()
                ):
                    raise ValueError("Expected positive finite TTv2 equation controls.")
                if any(
                    not math.isfinite(float(value)) or float(value) <= 0.0
                    for value in (*fast_lrs.values(), *fast_granularities.values())
                ):
                    raise ValueError(
                        "Expected positive finite per-parameter TTv2 fast controls."
                    )
                for key, shape, slow_lr in zip(
                    self.keys,
                    self.shapes,
                    self.learning_rates,
                ):
                    if not math.isfinite(slow_lr) or slow_lr <= 0.0:
                        raise ValueError(
                            "Expected positive finite slow optimizer learning rates."
                        )
                    fast_lr = float(fast_lrs[key])
                    fast_granularity = float(fast_granularities[key])
                    input_count = shape[0] // 2
                    buffer_scale = (
                        buffer_granularity
                        * fast_granularity
                        * auto_granularity
                        / (input_count * transfer_every)
                    )
                    self.ttv2_buffer_scale_uncorrected.append(buffer_scale)
                    if correct_magnitudes:
                        buffer_scale *= self.pulse_step_ratio / fast_granularity
                    self.ttv2_buffer_scale_corrected.append(buffer_scale)
                    effective_transfer_lr = transfer_lr * (
                        slow_lr if scale_transfer_lr else 1.0
                    )
                    self.ttv2_effective_transfer_lr.append(
                        effective_transfer_lr
                    )
                    self.ttv2_transfer_scale.append(
                        effective_transfer_lr / (fast_lr * buffer_scale)
                    )
                for size in self.sizes:
                    raw_parameter_cap = (
                        float(self.parameters["slow_pulse_budget_per_cell"])
                        * size
                    )
                    if not math.isclose(
                        raw_parameter_cap,
                        round(raw_parameter_cap),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ):
                        raise ValueError(
                            "Expected an integral per-parameter TTv2 slow-pulse cap."
                        )
                    self.ttv2_parameter_pulse_caps.append(
                        int(round(raw_parameter_cap))
                    )
        elif self.method != "direct_pulse":
            raise ValueError(f"Unsupported deployed recovery method: {self.method!r}.")
        self._apply_slow_apparent()

    @property
    def requires_gradients(self) -> bool:
        return self.method != "rail_refresh"

    def zero_grad(self, set_to_none: bool = True) -> None:
        for binding in self.bindings:
            if set_to_none:
                binding.state.grad = None
            elif binding.state.grad is not None:
                binding.state.grad.zero_()

    def _build_refresh_order(self) -> torch.Tensor:
        pieces = []
        remaining = self.slow_pulse_cap
        while remaining:
            permutation = torch.randperm(
                self.size,
                generator=self.selection_generator,
                device=self.device,
            )
            take = min(remaining, self.size)
            pieces.append(permutation[:take])
            remaining -= take
        return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64, device=self.device)

    def set_direct_probability_scale(
        self,
        scale: float,
        *,
        report: Mapping[str, Any],
    ) -> None:
        numeric = float(scale)
        if self.method != "direct_pulse" or not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("Expected a positive direct-pulse probability scale.")
        self.direct_probability_scale = numeric
        self.direct_probability_report = deepcopy(dict(report))

    def set_direct_gradient_thresholds(
        self,
        thresholds: Mapping[str, float],
        *,
        report: Mapping[str, Any],
    ) -> None:
        """Freeze one strict absolute raw-gradient gate per parameter."""

        percentile = self.parameters.get("direct_gradient_magnitude_percentile")
        if self.method != "direct_pulse" or percentile is None:
            raise ValueError(
                "Expected gradient thresholds only for percentile-gated direct recovery."
            )
        if set(thresholds) != set(self.keys):
            raise ValueError("Expected one direct gradient threshold per parameter.")
        normalized = {}
        for key in self.keys:
            value = float(thresholds[key])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("Expected finite non-negative gradient thresholds.")
            normalized[key] = value
        if float(report.get("percentile", -1.0)) != float(percentile):
            raise ValueError("Expected the threshold report percentile to match config.")
        self.direct_gradient_thresholds = normalized
        self.direct_gradient_threshold_report = deepcopy(dict(report))

    def _flat_gradients(self) -> tuple[torch.Tensor, torch.Tensor]:
        desired = []
        ratios = []
        for binding, rate in zip(self.bindings, self.learning_rates):
            gradient = binding.state.grad
            if gradient is None or not bool(torch.all(torch.isfinite(gradient))):
                raise FloatingPointError(
                    f"Expected a finite recovery gradient for {binding.key!r}."
                )
            update = -rate * gradient.detach()
            desired.append(update.reshape(-1))
            ratios.append(
                update.abs().reshape(-1) / (self.span * self.pulse_step_ratio)
            )
        return torch.cat(desired), torch.cat(ratios)

    def direct_calibration_snapshot(
        self,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Return read-only per-parameter gradient magnitudes and pulse ratios."""

        if self.method != "direct_pulse":
            raise RuntimeError(
                "Expected direct calibration data only for direct-pulse recovery."
            )
        result = {}
        for binding, rate in zip(self.bindings, self.learning_rates):
            gradient = binding.state.grad
            if gradient is None or not bool(torch.all(torch.isfinite(gradient))):
                raise FloatingPointError(
                    f"Expected a finite recovery gradient for {binding.key!r}."
                )
            magnitude = gradient.detach().abs().reshape(-1)
            result[binding.key] = {
                "raw_gradient_magnitude": magnitude.clone(),
                "pulse_probability_ratio": (
                    magnitude * abs(rate) / (self.span * self.pulse_step_ratio)
                ).clone(),
            }
        return result

    def _direct_components(
        self,
        *,
        record_gate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        desired = []
        ratios = []
        percentile = self.parameters.get("direct_gradient_magnitude_percentile")
        if percentile is not None and self.direct_gradient_thresholds is None:
            raise RuntimeError(
                "Expected direct gradient thresholds to be calibrated before training."
            )
        for index, (binding, rate) in enumerate(
            zip(self.bindings, self.learning_rates)
        ):
            gradient = binding.state.grad
            if gradient is None or not bool(torch.all(torch.isfinite(gradient))):
                raise FloatingPointError(
                    f"Expected a finite recovery gradient for {binding.key!r}."
                )
            update = -rate * gradient.detach()
            magnitude = gradient.detach().abs()
            nonzero = magnitude > 0.0
            if self.direct_gradient_thresholds is None:
                eligible = nonzero
            else:
                eligible = magnitude > self.direct_gradient_thresholds[binding.key]
            desired.append(update.reshape(-1))
            ratios.append(
                (
                    update.abs() / (self.span * self.pulse_step_ratio)
                    * eligible
                ).reshape(-1)
            )
            if record_gate:
                self.direct_gate_observed[index].add_(magnitude.numel())
                self.direct_gate_nonzero[index].add_(nonzero.sum())
                self.direct_gate_eligible[index].add_(eligible.sum())
                self.direct_gate_suppressed[index].add_(
                    (nonzero & ~eligible).sum()
                )
        return torch.cat(desired), torch.cat(ratios)

    def direct_probability_ratios(self) -> torch.Tensor:
        """Return the unscaled one-pulse probabilities for calibration."""

        if self.method != "direct_pulse":
            raise RuntimeError(
                "Expected direct-pulse ratios only for direct-pulse recovery."
            )
        _desired, ratios = self._direct_components(record_gate=False)
        return ratios.detach().clone()

    def _apply_slow_apparent(self) -> None:
        endpoint = ((self.slow_plant.apparent + 1.0) / 2.0).clamp(0.0, 1.0)
        with torch.no_grad():
            for binding, section in zip(self.bindings, self.sections):
                value = self.conductance_min + self.span * endpoint[section]
                binding.state.copy_(value.reshape(binding.state.shape).to(binding.state))

    def _trim_to_remaining_cap(self, indices: torch.Tensor) -> torch.Tensor:
        remaining = self.slow_pulse_cap - int(
            (self.slow_ledger.set_count + self.slow_ledger.reset_count).sum().item()
        )
        if remaining <= 0:
            return indices[:0]
        if indices.numel() <= remaining:
            return indices
        order = torch.randperm(
            indices.numel(),
            generator=self.selection_generator,
            device=self.device,
        )
        return indices[order[:remaining]]

    def _pulse_slow(self, flat_directions: torch.Tensor) -> dict[str, int]:
        directions = torch.as_tensor(flat_directions, device=self.device, dtype=torch.int8)
        if directions.shape != (self.size,):
            raise ValueError("Expected one slow pulse direction per conductance.")
        proposed = torch.where(directions != 0)[0]
        indices = self._trim_to_remaining_cap(proposed)
        if indices.numel() != proposed.numel():
            retained = torch.zeros_like(directions)
            retained[indices] = directions[indices]
            directions = retained
        if indices.numel() == 0:
            return {"requested": 0, "changed": 0}
        selected_directions = directions[indices]
        before = self.slow_plant.persistent[indices].clone()
        self.slow_plant.pulse(directions)
        after = self.slow_plant.persistent[indices]
        changed = after != before
        physical = after + self.slow_plant.population.reference[indices]
        saturated = (
            (torch.abs(physical - self.slow_plant.population.min_bound[indices]) <= 2e-6)
            | (torch.abs(physical - self.slow_plant.population.max_bound[indices]) <= 2e-6)
        )
        raw = (self.slow_plant.apparent[indices] + 1.0) / 2.0
        clipped = (raw < 0.0) | (raw > 1.0)
        self.slow_ledger.record(
            indices,
            selected_directions,
            changed=changed,
            saturated=saturated,
            clipped=clipped,
        )
        self._apply_slow_apparent()
        return {
            "requested": int(indices.numel()),
            "changed": int(changed.sum().item()),
        }

    def _step_refresh(self) -> None:
        assert self.refresh_order is not None
        start = (self.step_index * self.slow_pulse_cap) // self.total_steps
        end = ((self.step_index + 1) * self.slow_pulse_cap) // self.total_steps
        indices = self.refresh_order[start:end]
        directions = torch.zeros(self.size, dtype=torch.int8, device=self.device)
        active = self.requested_target[indices] > self.reset_baseline[indices] + 1e-7
        directions[indices] = torch.where(
            active,
            torch.ones_like(indices, dtype=torch.int8),
            -torch.ones_like(indices, dtype=torch.int8),
        )
        self._pulse_slow(directions)

    def _step_direct(self) -> None:
        if self.direct_probability_scale is None:
            raise RuntimeError("Expected direct-pulse probability calibration before training.")
        desired, ratios = self._direct_components(record_gate=True)
        probabilities = (ratios * float(self.direct_probability_scale)).clamp(max=1.0)
        self.expected_direct_pulses += float(probabilities.sum().item())
        sampled = torch.rand(
            self.size,
            generator=self.selection_generator,
            device=self.device,
        ) < probabilities
        directions = torch.zeros(self.size, dtype=torch.int8, device=self.device)
        directions[sampled & (desired > 0.0)] = 1
        directions[sampled & (desired < 0.0)] = -1
        self._pulse_slow(directions)

    def _fast_pair_layout(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        plus = []
        minus = []
        offset = 0
        for size in self.sizes:
            plus.append(torch.arange(offset, offset + size, device=self.device))
            minus.append(torch.arange(offset + size, offset + 2 * size, device=self.device))
            offset += 2 * size
        return plus, minus

    def _controller_settings(self) -> tuple[ControllerSettings, PopulationStepEstimator]:
        # Physical fast rails are always sampled without published
        # corruption, so they use the continuous calibration branch.
        settings = adaptive_controller_settings(self.device_model)
        estimator = step_estimator(self.device_model, "continuous")
        return settings, estimator

    def _initialize_physical_fast(self) -> None:
        assert self.fast_population is not None
        expected_keys = tuple(f"fast_pair.{key}" for key in self.keys)
        expected_shapes = tuple((2 * shape[0], shape[1]) for shape in self.shapes)
        if (
            self.fast_population.binding_keys != expected_keys
            or self.fast_population.binding_shapes != expected_shapes
            or self.fast_population.size != 2 * self.size
        ):
            raise ValueError("Expected the physical fast population to match two rails per slow cell.")
        population = self.fast_population.to(self.device)
        self.fast_population = population
        self.fast_generator = torch.Generator(device=self.device)
        self.fast_generator.manual_seed(int(self.parameters["fast_endpoint_seed"]))
        self.fast_plant = IbmReramPulsePlant(
            population,
            generator=self.fast_generator,
            device=self.device,
        )
        self.fast_plus_indices, self.fast_minus_indices = self._fast_pair_layout()
        self.fast_ledger = _RecoveryLedger(population.size, device=self.device)
        read_samples = int(self.parameters["fast_reset_read_samples"])
        guard = float(self.parameters["fast_reset_guard_standard_errors"])
        reset_directions = -torch.ones(population.size, dtype=torch.int8, device=self.device)
        reads = []
        for _ in range(read_samples):
            self.fast_plant.pulse(reset_directions)
            reads.append((self.fast_plant.apparent + 1.0) / 2.0)
        observed = torch.stack(reads)
        mean = observed.mean(dim=0)
        standard_error = observed.std(dim=0, unbiased=True) / math.sqrt(read_samples)
        guarded = mean + guard * standard_error
        baseline = torch.empty_like(mean)
        for plus, minus in zip(self.fast_plus_indices, self.fast_minus_indices):
            pair = torch.maximum(guarded[plus], guarded[minus]).clamp_min(0.0)
            baseline[plus] = pair
            baseline[minus] = pair
        settings, estimator = self._controller_settings()
        result = run_program_verify(
            self.fast_plant.controller_port(),
            targets=baseline,
            tolerance=step_ratio_tolerance(0.5, population.nominal_dw_min),
            maximum_pulses=128,
            settings=settings,
            estimator=estimator,
        )
        self.fast_baseline = baseline
        self.fast_commissioning = {
            "algorithm": "reset_read_guarded_pair_max_then_equalize",
            "controller_observations_only": True,
            "hidden_device_bounds_consumed": False,
            "assignment_seed": population.assignment_seed,
            "endpoint_seed": int(self.parameters["fast_endpoint_seed"]),
            "read_samples": read_samples,
            "guard_standard_errors": guard,
            "reset_conditioning_pulses": read_samples * population.size,
            "reset_mean": _tensor_summary(mean),
            "reset_standard_error": _tensor_summary(standard_error),
            "pair_baseline": _tensor_summary(baseline[torch.cat(self.fast_plus_indices)]),
            "equalization": _result_cost(result),
            "observed_reads": observed.detach().cpu().clone(),
            "baseline": baseline.detach().cpu().clone(),
        }

    def _record_fast_write(self, directions: torch.Tensor) -> None:
        assert self.fast_plant is not None and self.fast_ledger is not None
        indices = torch.where(directions != 0)[0]
        if indices.numel() == 0:
            return
        selected_directions = directions[indices]
        before = self.fast_plant.persistent[indices].clone()
        self.fast_plant.pulse(directions)
        after = self.fast_plant.persistent[indices]
        physical = after + self.fast_plant.population.reference[indices]
        raw = (self.fast_plant.apparent[indices] + 1.0) / 2.0
        self.fast_ledger.record(
            indices,
            selected_directions,
            changed=after != before,
            saturated=(
                (torch.abs(physical - self.fast_plant.population.min_bound[indices]) <= 2e-6)
                | (torch.abs(physical - self.fast_plant.population.max_bound[indices]) <= 2e-6)
            ),
            clipped=(raw < 0.0) | (raw > 1.0),
        )

    def _physical_fast_accumulate(self, normalized_desired: torch.Tensor) -> None:
        assert self.fast_plant is not None
        probabilities = (normalized_desired.abs() / self.pulse_step_ratio).clamp(max=1.0)
        sampled = torch.rand(
            self.size,
            generator=self.selection_generator,
            device=self.device,
        ) < probabilities
        slow_indices = torch.where(sampled & (normalized_desired != 0.0))[0]
        directions = torch.zeros(
            self.fast_plant.size,
            dtype=torch.int8,
            device=self.device,
        )
        if slow_indices.numel():
            plus_lookup = torch.cat(self.fast_plus_indices)
            minus_lookup = torch.cat(self.fast_minus_indices)
            positive = normalized_desired[slow_indices] > 0.0
            directions[plus_lookup[slow_indices[positive]]] = 1
            directions[minus_lookup[slow_indices[~positive]]] = 1
        self._record_fast_write(directions)

    def _reset_fast_selection(
        self,
        indices: torch.Tensor,
        *,
        event_key: str,
    ) -> None:
        assert self.fast_plant is not None
        assert self.fast_population is not None
        assert self.fast_baseline is not None
        assert self.fast_ledger is not None
        if indices.numel() > 1 and not bool(torch.all(indices[1:] > indices[:-1])):
            raise RuntimeError(
                "Expected strictly ascending fast-rail reset indices; the "
                "selected subpopulation is ordered by cell index."
            )
        selection = torch.zeros(self.fast_plant.size, dtype=torch.bool, device=self.device)
        selection[indices] = True
        population = _selected_array_population(self.fast_population, selection)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(
            derive_seed(
                int(self.parameters["fast_endpoint_seed"]),
                "transfer_reset",
                event_key,
            )
        )
        subplant = IbmReramPulsePlant(
            population,
            generator=generator,
            device=self.device,
        )
        persistent, apparent = self.fast_plant.read_cells(indices)
        subplant.load_state(persistent=persistent, apparent=apparent)
        settings, estimator = self._controller_settings()
        result = run_program_verify(
            subplant.controller_port(),
            targets=self.fast_baseline[indices],
            tolerance=step_ratio_tolerance(0.5, population.nominal_dw_min),
            maximum_pulses=128,
            settings=settings,
            estimator=estimator,
        )
        before = self.fast_plant.persistent[indices].clone()
        self.fast_plant.write_cells(
            indices,
            persistent=subplant.persistent,
            apparent=subplant.apparent,
        )
        physical = subplant.persistent + population.reference
        raw = (subplant.apparent + 1.0) / 2.0
        # Record every individual controller pulse, including reversals.
        self.fast_ledger.set_count.index_add_(0, indices, result.set_count.to(self.device))
        self.fast_ledger.reset_count.index_add_(0, indices, result.reset_count.to(self.device))
        self.fast_ledger.state_change_count.index_add_(
            0,
            indices,
            (subplant.persistent != before).to(torch.int64),
        )
        self.fast_ledger.saturated_count.index_add_(
            0,
            indices,
            (
                (torch.abs(physical - population.min_bound) <= 2e-6)
                | (torch.abs(physical - population.max_bound) <= 2e-6)
            ).to(torch.int64),
        )
        self.fast_ledger.clipped_count.index_add_(
            0,
            indices,
            ((raw < 0.0) | (raw > 1.0)).to(torch.int64),
        )
        self.fast_reset_verify_reads += int(result.verify_count.sum().item())
        self.fast_reset_reversals += int(result.reversals.sum().item())
        self.fast_reset_failures += int((~result.accepted).sum().item())

    def _transfer_due(self, binding_index: int) -> bool:
        columns = self.shapes[binding_index][1]
        visits = float(self.parameters["slow_pulse_budget_per_cell"]) * columns
        if not math.isclose(visits, round(visits), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Expected an integral number of cyclic column visits.")
        count = int(round(visits))
        before = (self.step_index * count) // self.total_steps
        after = ((self.step_index + 1) * count) // self.total_steps
        return after > before

    def _transfer_binding(self, binding_index: int) -> None:
        rows, columns = self.shapes[binding_index]
        column = (
            self.transfer_start[binding_index] + self.transfer_count[binding_index]
        ) % columns
        section = self.sections[binding_index]
        local_global = torch.arange(
            section.start,
            section.stop,
            device=self.device,
        ).reshape(rows, columns)
        slow_indices = local_global[:, column]
        if self.ideal_fast is not None:
            signal = self.ideal_fast[binding_index][:, column].clone()
        else:
            assert self.fast_plant is not None and self.fast_baseline is not None
            plus = self.fast_plus_indices[binding_index].reshape(rows, columns)[:, column]
            minus = self.fast_minus_indices[binding_index].reshape(rows, columns)[:, column]
            apparent = (self.fast_plant.apparent + 1.0) / 2.0
            signal = (
                apparent[plus]
                - self.fast_baseline[plus]
                - apparent[minus]
                + self.fast_baseline[minus]
            )
        probability = (signal.abs() / self.pulse_step_ratio).clamp(max=1.0)
        sampled = torch.rand(
            rows,
            generator=self.selection_generator,
            device=self.device,
        ) < probability
        directions = torch.zeros(self.size, dtype=torch.int8, device=self.device)
        selected = slow_indices[sampled & (signal != 0.0)]
        selected_signal = signal[sampled & (signal != 0.0)]
        directions[selected] = torch.where(
            selected_signal > 0.0,
            torch.ones_like(selected, dtype=torch.int8),
            -torch.ones_like(selected, dtype=torch.int8),
        )
        self.transfer_opportunities += rows
        self._pulse_slow(directions)
        if self.ideal_fast is not None:
            self.ideal_fast[binding_index][:, column].zero_()
        else:
            plus = self.fast_plus_indices[binding_index].reshape(rows, columns)[:, column]
            minus = self.fast_minus_indices[binding_index].reshape(rows, columns)[:, column]
            self._reset_fast_selection(
                torch.cat((plus, minus)),
                event_key=(
                    f"{self.keys[binding_index]}:{self.transfer_count[binding_index]}"
                ),
            )
        self.transfer_count[binding_index] += 1

    def _step_tiki_taka(self) -> None:
        desired, _ratios = self._flat_gradients()
        normalized = desired / self.span
        if self.ideal_fast is not None:
            for accumulator, section, shape in zip(
                self.ideal_fast,
                self.sections,
                self.shapes,
            ):
                accumulator.add_(normalized[section].reshape(shape))
        else:
            self._physical_fast_accumulate(normalized)
        for binding_index in range(len(self.bindings)):
            if self._transfer_due(binding_index):
                self._transfer_binding(binding_index)

    def _ttv2_fast_accumulate(self, normalized_desired: torch.Tensor) -> None:
        """Apply a symmetric soft-bounds write to the persistent ideal A array."""

        assert self.ideal_fast is not None
        limit = float(self.parameters["ttv2_fast_weight_limit"])
        for accumulator, section, shape in zip(
            self.ideal_fast,
            self.sections,
            self.shapes,
        ):
            update = normalized_desired[section].reshape(shape)
            distance = torch.where(
                update >= 0.0,
                limit - accumulator,
                limit + accumulator,
            ) / limit
            accumulator.add_(update * distance).clamp_(-limit, limit)
            self.ttv2_fast_saturation_observations += int(
                (accumulator.abs() >= limit).sum().item()
            )

    def _transfer_ttv2_binding(self, binding_index: int) -> None:
        """Read one complete dual-rail logical input group into digital H."""

        assert self.ideal_fast is not None and self.ttv2_hidden is not None
        rows, columns = self.shapes[binding_index]
        input_count = rows // 2
        logical_input = (
            self.transfer_start[binding_index] + self.transfer_count[binding_index]
        ) % input_count
        row_indices = torch.tensor(
            [logical_input, logical_input + input_count],
            dtype=torch.int64,
            device=self.device,
        )
        section = self.sections[binding_index]
        local_global = torch.arange(
            section.start,
            section.stop,
            device=self.device,
        ).reshape(rows, columns)
        slow_indices = local_global[row_indices, :].reshape(-1)
        signal = self.ideal_fast[binding_index][row_indices, :]
        hidden_slice = self.ttv2_hidden[binding_index][row_indices, :].clone()
        buffer_input = signal * self.ttv2_transfer_scale[binding_index]
        hidden_slice.add_(buffer_input)
        self.ttv2_buffer_input_abs += float(buffer_input.abs().sum().item())

        threshold = float(self.parameters["ttv2_buffer_threshold"])
        crossing = hidden_slice.abs() >= threshold
        proposed = slow_indices[crossing.reshape(-1)]
        retained = self._trim_to_remaining_cap(proposed)
        # ``_trim_to_remaining_cap`` is allowed to return the retained global
        # indices in a random order.  Preserve each global cell's own local H
        # position so that a cap-truncated mixed-sign transfer cannot exchange
        # pulse directions between cells. ``slow_indices`` is strictly sorted
        # because a scan concatenates two complete stored rows in row order.
        if retained.numel():
            retained_local = torch.searchsorted(slow_indices, retained)
            if not torch.equal(slow_indices[retained_local], retained):
                raise RuntimeError(
                    "Expected every retained TTv2 cell to belong to the scanned group."
                )
        else:
            retained_local = torch.empty(
                0,
                dtype=torch.int64,
                device=self.device,
            )
        directions = torch.zeros(self.size, dtype=torch.int8, device=self.device)
        selected_hidden = hidden_slice.reshape(-1)[retained_local]
        directions[retained] = torch.where(
            selected_hidden > 0.0,
            torch.ones_like(retained, dtype=torch.int8),
            -torch.ones_like(retained, dtype=torch.int8),
        )
        self._pulse_slow(directions)

        # Lossless TTv2 residual accounting: only an actually dispatched
        # one-pulse update is removed. Crossings deferred by the safety cap
        # remain in H and are visible in the terminal artifact.
        # ``retained`` follows the cap sampler's (possibly randomized) order,
        # whereas boolean indexing would visit ``dispatched`` in row-major
        # order.  Subtract in the same local order used to assign directions
        # so mixed-sign retained cells keep their own residuals.
        hidden_slice.reshape(-1)[retained_local] -= (
            directions[retained].to(hidden_slice.dtype) * threshold
        )
        self.ttv2_hidden[binding_index][row_indices, :] = hidden_slice
        crossings = int(crossing.sum().item())
        dispatched_count = int(retained.numel())
        self.ttv2_threshold_crossings += crossings
        self.ttv2_dispatched_pulses += dispatched_count
        self.ttv2_cap_deferred_crossings += crossings - dispatched_count
        self.transfer_opportunities += int(slow_indices.numel())
        self.transfer_count[binding_index] += 1

    def _step_ttv2(self) -> None:
        desired, _ratios = self._flat_gradients()
        self._ttv2_fast_accumulate(desired / self.span)
        transfer_every = int(self.parameters["ttv2_transfer_every"])
        if (self.step_index + 1) % transfer_every != 0:
            return
        for binding_index in range(len(self.bindings)):
            self._transfer_ttv2_binding(binding_index)

    def _parameter_slow_pulses(self, binding_index: int) -> int:
        section = self.sections[binding_index]
        return int(
            (
                self.slow_ledger.set_count[section]
                + self.slow_ledger.reset_count[section]
            )
            .sum()
            .item()
        )

    def _ttv2_aihwkit_parameter_frozen(self, binding_index: int) -> bool:
        return self._parameter_slow_pulses(binding_index) >= (
            self.ttv2_parameter_pulse_caps[binding_index]
        )

    def _ttv2_aihwkit_fast_accumulate(self) -> list[bool]:
        """Update persistent A with fixed fast LRs, independently of slow LRs."""

        assert self.ideal_fast is not None
        fast_lrs = dict(self.parameters["ttv2_fast_lr_by_parameter"])
        limit = float(self.parameters["ttv2_fast_weight_limit"])
        active = []
        for binding_index, (binding, accumulator) in enumerate(
            zip(self.bindings, self.ideal_fast)
        ):
            if self._ttv2_aihwkit_parameter_frozen(binding_index):
                active.append(False)
                self.ttv2_parameter_skipped_steps[binding_index] += 1
                if self.ttv2_parameter_freeze_step[binding_index] is None:
                    self.ttv2_parameter_freeze_step[binding_index] = self.step_index
                continue
            gradient = binding.state.grad
            if gradient is None or not bool(torch.all(torch.isfinite(gradient))):
                raise FloatingPointError(
                    f"Expected a finite recovery gradient for {binding.key!r}."
                )
            update = -float(fast_lrs[binding.key]) * gradient.detach() / self.span
            distance = torch.where(
                update >= 0.0,
                limit - accumulator,
                limit + accumulator,
            ) / limit
            accumulator.add_(update * distance).clamp_(-limit, limit)
            self.ttv2_fast_saturation_observations += int(
                (accumulator.abs() >= limit).sum().item()
            )
            active.append(True)
        return active

    def _transfer_ttv2_aihwkit_binding(self, binding_index: int) -> None:
        """Apply the AIHWKit 1.1 Chopped buffer equation to one rail pair."""

        assert self.ideal_fast is not None and self.ttv2_hidden is not None
        rows, columns = self.shapes[binding_index]
        input_count = rows // 2
        logical_input = (
            self.transfer_start[binding_index]
            + self.transfer_count[binding_index]
        ) % input_count
        row_indices = torch.tensor(
            [logical_input, logical_input + input_count],
            dtype=torch.int64,
            device=self.device,
        )
        section = self.sections[binding_index]
        local_global = torch.arange(
            section.start,
            section.stop,
            device=self.device,
        ).reshape(rows, columns)
        slow_indices = local_global[row_indices, :].reshape(-1)
        signal = self.ideal_fast[binding_index][row_indices, :]
        hidden_slice = self.ttv2_hidden[binding_index][row_indices, :].clone()
        buffer_input = signal * self.ttv2_transfer_scale[binding_index]
        omega = hidden_slice + buffer_input
        self.ttv2_buffer_input_abs += float(buffer_input.abs().sum().item())

        desired_bl = int(self.parameters["ttv2_desired_bl"])
        quantized = torch.trunc(omega).clamp(-desired_bl, desired_bl)
        crossing_flat = quantized.reshape(-1) != 0.0
        proposed_local = torch.where(crossing_flat)[0]
        remaining = (
            self.ttv2_parameter_pulse_caps[binding_index]
            - self._parameter_slow_pulses(binding_index)
        )
        dispatch_flat = torch.zeros(
            slow_indices.numel(),
            dtype=torch.bool,
            device=self.device,
        )
        if remaining > 0 and proposed_local.numel() > 0:
            if proposed_local.numel() > remaining:
                order = torch.randperm(
                    proposed_local.numel(),
                    generator=self.selection_generator,
                    device=self.device,
                )
                proposed_local = proposed_local[order[:remaining]]
            dispatch_flat[proposed_local] = True

        selected_global = slow_indices[dispatch_flat]
        selected_quantized = quantized.reshape(-1)[dispatch_flat]
        directions = torch.zeros(self.size, dtype=torch.int8, device=self.device)
        directions[selected_global] = selected_quantized.to(torch.int8)
        pulse_result = self._pulse_slow(directions)
        if pulse_result["requested"] != int(selected_global.numel()):
            raise RuntimeError(
                "Expected the global safety cap to preserve a proportional TTv2 dispatch."
            )

        momentum = float(self.parameters["ttv2_momentum"])
        if bool(self.parameters["ttv2_forget_buffer"]):
            omega.reshape(-1)[dispatch_flat] *= momentum
        else:
            omega.reshape(-1)[dispatch_flat] -= (
                (1.0 - momentum) * selected_quantized
            )
        self.ttv2_hidden[binding_index][row_indices, :] = omega

        crossings = int(crossing_flat.sum().item())
        dispatched_count = int(selected_global.numel())
        deferred = crossings - dispatched_count
        self.ttv2_threshold_crossings += crossings
        self.ttv2_dispatched_pulses += dispatched_count
        self.ttv2_cap_deferred_crossings += deferred
        self.ttv2_parameter_deferred_crossings[binding_index] += deferred
        self.transfer_opportunities += int(slow_indices.numel())
        self.transfer_count[binding_index] += 1
        if (
            self._ttv2_aihwkit_parameter_frozen(binding_index)
            and self.ttv2_parameter_freeze_step[binding_index] is None
        ):
            self.ttv2_parameter_freeze_step[binding_index] = self.step_index + 1

    def _step_ttv2_aihwkit(self) -> None:
        active = self._ttv2_aihwkit_fast_accumulate()
        transfer_every = int(self.parameters["ttv2_transfer_every"])
        if (self.step_index + 1) % transfer_every != 0:
            return
        for binding_index, is_active in enumerate(active):
            if is_active:
                self._transfer_ttv2_aihwkit_binding(binding_index)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Expected deployed recovery not to use an optimizer closure.")
        if self.step_index >= self.total_steps:
            raise RuntimeError("Expected no recovery step beyond the declared terminal epoch.")
        if self.method == "rail_refresh":
            self._step_refresh()
        elif self.method == "direct_pulse":
            self._step_direct()
        elif self.method == "tiki_taka":
            self._step_tiki_taka()
        elif self.method == "ttv2":
            self._step_ttv2()
        else:
            self._step_ttv2_aihwkit()
        self.step_index += 1
        return None

    def current_apparent_endpoint(self, *, clipped: bool = True) -> torch.Tensor:
        value = (self.slow_plant.apparent + 1.0) / 2.0
        return value.clamp(0.0, 1.0) if clipped else value

    def current_persistent_endpoint(self) -> torch.Tensor:
        return (self.slow_plant.persistent + 1.0) / 2.0

    @contextmanager
    def temporary_forward_state(self, state: str) -> Iterator[None]:
        snapshots = tuple(binding.state.detach().clone() for binding in self.bindings)
        if state == "apparent":
            endpoint = self.current_apparent_endpoint(clipped=True)
        elif state == "persistent":
            endpoint = self.current_persistent_endpoint()
        else:
            raise ValueError("Expected recovery state 'apparent' or 'persistent'.")
        try:
            with torch.no_grad():
                for binding, section in zip(self.bindings, self.sections):
                    value = self.conductance_min + self.span * endpoint[section]
                    binding.state.copy_(value.reshape(binding.state.shape).to(binding.state))
            yield
        finally:
            with torch.no_grad():
                for binding, snapshot in zip(self.bindings, snapshots):
                    binding.state.copy_(snapshot)

    def endpoint_diagnostics(self, endpoint: torch.Tensor) -> dict[str, Any]:
        value = torch.as_tensor(endpoint, device=self.device, dtype=torch.float32)
        offsets = torch.tensor(
            [0.0, 0.0479245, 0.095849, 0.1437735, 0.191698],
            device=self.device,
        )
        distance = (value[:, None] - self.reset_baseline[:, None] - offsets[None, :]).abs()
        nearest_distance, nearest_offset_index = distance.min(dim=1)
        sign_flips = 0
        nonzero = 0
        code_crossings = 0
        layout_by_key = dict(self.parameters.get("source_dual_rail_layout_by_parameter", {}))
        for key, shape, section in zip(self.keys, self.shapes, self.sections):
            layout = layout_by_key.get(key)
            if layout not in {"halves", "paired"}:
                continue
            rows, columns = shape
            half_rows = rows // 2
            half_columns = columns // 2
            plus_rows = torch.arange(half_rows, device=self.device)
            minus_rows = plus_rows + half_rows
            if layout == "halves":
                plus_columns = torch.arange(half_columns, device=self.device)
                minus_columns = plus_columns + half_columns
            else:
                plus_columns = 2 * torch.arange(half_columns, device=self.device)
                minus_columns = plus_columns + 1
            observed = value[section].reshape(shape)
            target = self.requested_target[section].reshape(shape)
            observed_contrast = (
                observed[plus_rows[:, None], plus_columns]
                - observed[plus_rows[:, None], minus_columns]
                - observed[minus_rows[:, None], plus_columns]
                + observed[minus_rows[:, None], minus_columns]
            )
            target_contrast = (
                target[plus_rows[:, None], plus_columns]
                - target[plus_rows[:, None], minus_columns]
                - target[minus_rows[:, None], plus_columns]
                + target[minus_rows[:, None], minus_columns]
            )
            active = target_contrast != 0.0
            nonzero += int(active.sum().item())
            sign_flips += int((active & (target_contrast * observed_contrast < 0.0)).sum().item())
            target_code = torch.round(target_contrast / 0.095849).clamp(-4, 4)
            observed_code = torch.round(observed_contrast / 0.095849).clamp(-4, 4)
            code_crossings += int((target_code != observed_code).sum().item())
        histogram = {
            str(index): int((nearest_offset_index == index).sum().item())
            for index in range(5)
        }
        return {
            "nearest_cell_offset_index_histogram": histogram,
            "off_grid_distance": _tensor_summary(nearest_distance),
            "code_crossings": code_crossings,
            "logical_nonzero_targets": nonzero,
            "logical_sign_flips": sign_flips,
            "logical_sign_flip_fraction": sign_flips / nonzero if nonzero else None,
        }

    def report(self) -> dict[str, Any]:
        slow = self.slow_ledger.summary()
        layer_costs = {}
        for key, section in zip(self.keys, self.sections):
            set_count = self.slow_ledger.set_count[section]
            reset_count = self.slow_ledger.reset_count[section]
            layer_costs[key] = {
                "set_requested": int(set_count.sum().item()),
                "reset_requested": int(reset_count.sum().item()),
                "total_requested": int((set_count + reset_count).sum().item()),
                "unique_cells_touched": int(((set_count + reset_count) > 0).sum().item()),
            }
        direct_gate = None
        if self.method == "direct_pulse":
            gate_by_parameter = {}
            for index, key in enumerate(self.keys):
                observed = int(self.direct_gate_observed[index].item())
                nonzero = int(self.direct_gate_nonzero[index].item())
                eligible = int(self.direct_gate_eligible[index].item())
                suppressed = int(self.direct_gate_suppressed[index].item())
                gate_by_parameter[key] = {
                    "observed": observed,
                    "nonzero": nonzero,
                    "eligible": eligible,
                    "suppressed_nonzero": suppressed,
                    "eligible_fraction_of_observed": (
                        eligible / observed if observed else None
                    ),
                    "eligible_fraction_of_nonzero": (
                        eligible / nonzero if nonzero else None
                    ),
                    "suppressed_fraction_of_nonzero": (
                        suppressed / nonzero if nonzero else None
                    ),
                }
            direct_gate = {
                "mode": (
                    "none"
                    if self.direct_gradient_thresholds is None
                    else "fixed_parameter_absolute_raw_gradient"
                ),
                "comparison": "strictly_greater_than",
                "percentile": self.parameters.get(
                    "direct_gradient_magnitude_percentile"
                ),
                "threshold_by_parameter": deepcopy(
                    self.direct_gradient_thresholds
                ),
                "calibration": deepcopy(
                    self.direct_gradient_threshold_report
                ),
                "cumulative_by_parameter": gate_by_parameter,
            }
        ttv2_report = None
        if self.method == "ttv2":
            ttv2_report = {
                "scan_axis": "canonical_logical_input_columns",
                "scan_unit": "complete_dual_rail_input_pair",
                "fast_array_persistent_after_read": True,
                "buffer_residual_mode": self.parameters[
                    "ttv2_buffer_residual_mode"
                ],
                "transfer_every_minibatches": self.parameters[
                    "ttv2_transfer_every"
                ],
                "gamma0": self.parameters["ttv2_gamma0"],
                "fast_update_model": self.parameters["ttv2_fast_update_model"],
                "fast_weight_limit": self.parameters["ttv2_fast_weight_limit"],
                "buffer_threshold_pulse_units": self.parameters[
                    "ttv2_buffer_threshold"
                ],
                "buffer_scale_by_parameter": {
                    key: value
                    for key, value in zip(self.keys, self.ttv2_transfer_scale)
                },
                "threshold_crossing_observations": self.ttv2_threshold_crossings,
                "dispatched_pulses": self.ttv2_dispatched_pulses,
                "cap_deferred_crossing_observations": self.ttv2_cap_deferred_crossings,
                "buffer_input_absolute_sum": self.ttv2_buffer_input_abs,
                "fast_saturation_observations": self.ttv2_fast_saturation_observations,
                "fast_state_by_parameter": {
                    key: _tensor_summary(value)
                    for key, value in zip(self.keys, self.ideal_fast or ())
                },
                "buffer_state_by_parameter": {
                    key: _tensor_summary(value)
                    for key, value in zip(self.keys, self.ttv2_hidden or ())
                },
            }
        elif self.method == "ttv2_aihwkit_1p1_minibatch_equation":
            cap_by_parameter = {}
            cursor_by_parameter = {}
            for index, key in enumerate(self.keys):
                requested = self._parameter_slow_pulses(index)
                cap = self.ttv2_parameter_pulse_caps[index]
                cap_by_parameter[key] = {
                    "pulse_cap": cap,
                    "requested": requested,
                    "remaining": cap - requested,
                    "exhausted": requested >= cap,
                    "cap_reached_after_step": self.ttv2_parameter_freeze_step[
                        index
                    ],
                    "subsequent_steps_skipped": self.ttv2_parameter_skipped_steps[
                        index
                    ],
                    "deferred_crossing_observations": (
                        self.ttv2_parameter_deferred_crossings[index]
                    ),
                }
                input_count = self.shapes[index][0] // 2
                next_cursor = (
                    self.transfer_start[index] + self.transfer_count[index]
                ) % input_count
                cursor_by_parameter[key] = {
                    "executed_scans": self.transfer_count[index],
                    "logical_input_count": input_count,
                    "next_logical_input": next_cursor,
                    "matches_zero_origin_executed_modulo": (
                        next_cursor == self.transfer_count[index] % input_count
                    ),
                }
            ledger_dispatches = int(
                sum(value["total_requested"] for value in layer_costs.values())
            )
            parameter_deferred = int(
                sum(self.ttv2_parameter_deferred_crossings)
            )
            expected_caps = [
                int(
                    round(
                        float(self.parameters["slow_pulse_budget_per_cell"])
                        * size
                    )
                )
                for size in self.sizes
            ]
            ttv2_report = {
                "equation_reference": "aihwkit_1.1.0_chopped_transfer_cpp",
                "scan_axis": "canonical_logical_input_columns",
                "scan_unit": "complete_dual_rail_input_pair",
                "atomicity": {
                    "scan": "complete_dual_rail_input_pair",
                    "pulse": "individual_cell",
                    "four_cell_quad_pulse_atomic": False,
                },
                "fast_array_persistent_after_read": True,
                "fast_update_model": "continuous_symmetric_soft_bounds",
                "fast_weight_limit": self.parameters["ttv2_fast_weight_limit"],
                "fast_lr_by_parameter": deepcopy(
                    self.parameters["ttv2_fast_lr_by_parameter"]
                ),
                "fast_granularity_by_parameter": deepcopy(
                    self.parameters["ttv2_fast_granularity_by_parameter"]
                ),
                "transfer_every_minibatches": self.parameters[
                    "ttv2_transfer_every"
                ],
                "units_in_mbatch": self.parameters["ttv2_units_in_mbatch"],
                "transfer_lr": self.parameters["ttv2_transfer_lr"],
                "scale_transfer_lr": self.parameters[
                    "ttv2_scale_transfer_lr"
                ],
                "buffer_granularity": self.parameters[
                    "ttv2_buffer_granularity"
                ],
                "auto_granularity": self.parameters["ttv2_auto_granularity"],
                "correct_gradient_magnitudes": self.parameters[
                    "ttv2_correct_gradient_magnitudes"
                ],
                "buffer_scale_uncorrected_by_parameter": {
                    key: value
                    for key, value in zip(
                        self.keys,
                        self.ttv2_buffer_scale_uncorrected,
                    )
                },
                "buffer_scale_corrected_by_parameter": {
                    key: value
                    for key, value in zip(
                        self.keys,
                        self.ttv2_buffer_scale_corrected,
                    )
                },
                "effective_transfer_lr_by_parameter": {
                    key: value
                    for key, value in zip(
                        self.keys,
                        self.ttv2_effective_transfer_lr,
                    )
                },
                "lambda_h_by_parameter": {
                    key: value
                    for key, value in zip(self.keys, self.ttv2_transfer_scale)
                },
                "desired_bl": self.parameters["ttv2_desired_bl"],
                "momentum": self.parameters["ttv2_momentum"],
                "forget_buffer": self.parameters["ttv2_forget_buffer"],
                "cap_scope": self.parameters["ttv2_cap_scope"],
                "cursor_policy": self.parameters["ttv2_cursor_policy"],
                "in_chop_probability": self.parameters[
                    "ttv2_in_chop_probability"
                ],
                "cap_by_parameter": cap_by_parameter,
                "cursor_by_parameter": cursor_by_parameter,
                "threshold_crossing_observations": self.ttv2_threshold_crossings,
                "dispatched_pulses": self.ttv2_dispatched_pulses,
                "cap_deferred_crossing_observations": self.ttv2_cap_deferred_crossings,
                "invariants": {
                    "dispatched_equals_parameter_ledger_sum": (
                        self.ttv2_dispatched_pulses == ledger_dispatches
                    ),
                    "dispatched_equals_global_ledger": (
                        self.ttv2_dispatched_pulses
                        == slow["total_requested"]
                    ),
                    "candidates_equal_dispatched_plus_deferred": (
                        self.ttv2_threshold_crossings
                        == self.ttv2_dispatched_pulses
                        + self.ttv2_cap_deferred_crossings
                    ),
                    "parameter_deferred_equals_global_deferred": (
                        parameter_deferred
                        == self.ttv2_cap_deferred_crossings
                    ),
                    "parameter_caps_match_declared_budget": (
                        self.ttv2_parameter_pulse_caps == expected_caps
                    ),
                    "parameter_caps_sum_to_global_cap": (
                        sum(self.ttv2_parameter_pulse_caps)
                        == self.slow_pulse_cap
                    ),
                    "every_parameter_request_within_cap": all(
                        self._parameter_slow_pulses(index)
                        <= self.ttv2_parameter_pulse_caps[index]
                        for index in range(len(self.bindings))
                    ),
                    "cursor_matches_executed_scan_count": all(
                        value["matches_zero_origin_executed_modulo"]
                        for value in cursor_by_parameter.values()
                    ),
                },
                "buffer_input_absolute_sum": self.ttv2_buffer_input_abs,
                "fast_saturation_observations": self.ttv2_fast_saturation_observations,
                "fast_state_by_parameter": {
                    key: _tensor_summary(value)
                    for key, value in zip(self.keys, self.ideal_fast or ())
                },
                "buffer_state_by_parameter": {
                    key: _tensor_summary(value)
                    for key, value in zip(self.keys, self.ttv2_hidden or ())
                },
            }
        result = {
            "method": self.method,
            "algorithm_class": {
                "tiki_taka": "tt_inspired_direct_fast_to_slow_transfer",
                "ttv2": "ttv2_persistent_fast_digital_residual",
                "ttv2_aihwkit_1p1_minibatch_equation": (
                    "ttv2_aihwkit_1p1_minibatch_equation"
                ),
            }.get(self.method, self.method),
            "completed_steps": self.step_index,
            "total_steps": self.total_steps,
            "slow_pulse_budget_per_cell": self.parameters["slow_pulse_budget_per_cell"],
            "slow_pulse_cap": self.slow_pulse_cap,
            "slow": slow,
            "slow_by_parameter": layer_costs,
            "transfer_opportunities": self.transfer_opportunities,
            "transfer_count_by_parameter": {
                key: count for key, count in zip(self.keys, self.transfer_count)
            },
            "direct_probability_scale": self.direct_probability_scale,
            "direct_probability_calibration": self.direct_probability_report,
            "direct_expected_pulses": self.expected_direct_pulses,
            "direct_gradient_gate": direct_gate,
            "ttv2": ttv2_report,
            "apparent_endpoint": self.endpoint_diagnostics(
                self.current_apparent_endpoint(clipped=True)
            ),
            "persistent_endpoint": self.endpoint_diagnostics(
                self.current_persistent_endpoint()
            ),
        }
        if self.fast_ledger is not None:
            result["fast"] = {
                **self.fast_ledger.summary(),
                "reset_verify_reads": self.fast_reset_verify_reads,
                "reset_reversals": self.fast_reset_reversals,
                "reset_failures": self.fast_reset_failures,
                "commissioning": {
                    key: value
                    for key, value in (self.fast_commissioning or {}).items()
                    if not isinstance(value, torch.Tensor)
                },
            }
        else:
            result["fast"] = None
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": RECOVERY_SCHEMA,
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "parameters": deepcopy(self.parameters),
            "source_deployment_sha256": self.source_deployment_sha256,
            "slow_population_fingerprint": self.slow_population.fingerprint,
            "step_index": self.step_index,
            "slow_persistent": self.slow_plant.persistent.detach().cpu().clone(),
            "slow_apparent": self.slow_plant.apparent.detach().cpu().clone(),
            "slow_generator_state": self.slow_generator.get_state().detach().cpu().clone(),
            "selection_generator_state": self.selection_generator.get_state().detach().cpu().clone(),
            "slow_ledger": self.slow_ledger.state_dict(),
            "direct_probability_scale": self.direct_probability_scale,
            "direct_probability_report": deepcopy(self.direct_probability_report),
            "expected_direct_pulses": self.expected_direct_pulses,
            "direct_gradient_thresholds": deepcopy(
                self.direct_gradient_thresholds
            ),
            "direct_gradient_threshold_report": deepcopy(
                self.direct_gradient_threshold_report
            ),
            "direct_gate_observed": self.direct_gate_observed.detach().cpu().clone(),
            "direct_gate_nonzero": self.direct_gate_nonzero.detach().cpu().clone(),
            "direct_gate_eligible": self.direct_gate_eligible.detach().cpu().clone(),
            "direct_gate_suppressed": self.direct_gate_suppressed.detach().cpu().clone(),
            "refresh_order": (
                None if self.refresh_order is None else self.refresh_order.detach().cpu().clone()
            ),
            "ideal_fast": (
                None
                if self.ideal_fast is None
                else [value.detach().cpu().clone() for value in self.ideal_fast]
            ),
            "ttv2_hidden": (
                None
                if self.ttv2_hidden is None
                else [value.detach().cpu().clone() for value in self.ttv2_hidden]
            ),
            "ttv2_threshold_crossings": self.ttv2_threshold_crossings,
            "ttv2_dispatched_pulses": self.ttv2_dispatched_pulses,
            "ttv2_cap_deferred_crossings": self.ttv2_cap_deferred_crossings,
            "ttv2_fast_saturation_observations": (
                self.ttv2_fast_saturation_observations
            ),
            "ttv2_buffer_input_abs": self.ttv2_buffer_input_abs,
            "ttv2_parameter_deferred_crossings": list(
                self.ttv2_parameter_deferred_crossings
            ),
            "ttv2_parameter_freeze_step": list(
                self.ttv2_parameter_freeze_step
            ),
            "ttv2_parameter_skipped_steps": list(
                self.ttv2_parameter_skipped_steps
            ),
            "transfer_start": list(self.transfer_start),
            "transfer_count": list(self.transfer_count),
            "transfer_opportunities": self.transfer_opportunities,
            "fast_population_fingerprint": (
                None if self.fast_population is None else self.fast_population.fingerprint
            ),
            "fast_persistent": (
                None if self.fast_plant is None else self.fast_plant.persistent.detach().cpu().clone()
            ),
            "fast_apparent": (
                None if self.fast_plant is None else self.fast_plant.apparent.detach().cpu().clone()
            ),
            "fast_generator_state": (
                None
                if self.fast_generator is None
                else self.fast_generator.get_state().detach().cpu().clone()
            ),
            "fast_baseline": (
                None if self.fast_baseline is None else self.fast_baseline.detach().cpu().clone()
            ),
            "fast_ledger": None if self.fast_ledger is None else self.fast_ledger.state_dict(),
            "fast_reset_verify_reads": self.fast_reset_verify_reads,
            "fast_reset_reversals": self.fast_reset_reversals,
            "fast_reset_failures": self.fast_reset_failures,
            "fast_commissioning": deepcopy(self.fast_commissioning),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != RECOVERY_SCHEMA
            or state.get("schema_version") != RECOVERY_SCHEMA_VERSION
            or dict(state.get("parameters", {})) != self.parameters
            or state.get("source_deployment_sha256") != self.source_deployment_sha256
            or state.get("slow_population_fingerprint") != self.slow_population.fingerprint
        ):
            raise ValueError("Expected a recovery checkpoint for this exact protocol and deployment.")
        step_index = state.get("step_index")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or not 0 <= step_index <= self.total_steps:
            raise ValueError("Expected a valid saved recovery step index.")
        saved_slow = {}
        for name, current in (
            ("slow_persistent", self.slow_plant.persistent),
            ("slow_apparent", self.slow_plant.apparent),
        ):
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != current.shape:
                raise ValueError(f"Expected saved recovery tensor {name!r}.")
            saved_slow[name] = value.to(current.device)
        self.slow_plant.load_state(
            persistent=saved_slow["slow_persistent"],
            apparent=saved_slow["slow_apparent"],
        )
        self.slow_generator.set_state(state["slow_generator_state"].detach().cpu())
        self.selection_generator.set_state(state["selection_generator_state"].detach().cpu())
        self.slow_ledger.load_state_dict(state["slow_ledger"])
        self.step_index = step_index
        self.direct_probability_scale = state.get("direct_probability_scale")
        self.direct_probability_report = deepcopy(state.get("direct_probability_report"))
        self.expected_direct_pulses = float(state.get("expected_direct_pulses", 0.0))
        saved_thresholds = state.get("direct_gradient_thresholds")
        saved_threshold_report = state.get("direct_gradient_threshold_report")
        percentile = self.parameters.get("direct_gradient_magnitude_percentile")
        if percentile is None:
            if saved_thresholds is not None or saved_threshold_report is not None:
                raise ValueError(
                    "Expected an ungated direct checkpoint not to contain thresholds."
                )
        else:
            if not isinstance(saved_thresholds, Mapping) or not isinstance(
                saved_threshold_report, Mapping
            ):
                raise ValueError(
                    "Expected a gated direct checkpoint to contain its calibration."
                )
            self.set_direct_gradient_thresholds(
                saved_thresholds,
                report=saved_threshold_report,
            )
        for name, current in (
            ("direct_gate_observed", self.direct_gate_observed),
            ("direct_gate_nonzero", self.direct_gate_nonzero),
            ("direct_gate_eligible", self.direct_gate_eligible),
            ("direct_gate_suppressed", self.direct_gate_suppressed),
        ):
            value = state.get(name)
            if value is None and percentile is None:
                current.zero_()
                continue
            if not isinstance(value, torch.Tensor) or value.shape != current.shape:
                raise ValueError(f"Expected valid saved direct-gate tensor {name!r}.")
            current.copy_(value.to(current.device))
        if self.refresh_order is not None:
            value = state.get("refresh_order")
            if not isinstance(value, torch.Tensor) or value.shape != self.refresh_order.shape:
                raise ValueError("Expected the exact saved refresh order.")
            self.refresh_order.copy_(value.to(self.device))
        if self.ideal_fast is not None:
            values = state.get("ideal_fast")
            if not isinstance(values, list) or len(values) != len(self.ideal_fast):
                raise ValueError("Expected every ideal fast accumulator in recovery state.")
            for current, value in zip(self.ideal_fast, values):
                if not isinstance(value, torch.Tensor) or value.shape != current.shape:
                    raise ValueError("Expected matching ideal fast accumulator shapes.")
                current.copy_(value.to(current.device))
        if self.ttv2_hidden is not None:
            values = state.get("ttv2_hidden")
            if not isinstance(values, list) or len(values) != len(self.ttv2_hidden):
                raise ValueError("Expected every TTv2 digital residual buffer.")
            for current, value in zip(self.ttv2_hidden, values):
                if not isinstance(value, torch.Tensor) or value.shape != current.shape:
                    raise ValueError("Expected matching TTv2 buffer shapes.")
                current.copy_(value.to(current.device))
            self.ttv2_threshold_crossings = int(
                state.get("ttv2_threshold_crossings", 0)
            )
            self.ttv2_dispatched_pulses = int(
                state.get("ttv2_dispatched_pulses", 0)
            )
            self.ttv2_cap_deferred_crossings = int(
                state.get("ttv2_cap_deferred_crossings", 0)
            )
            self.ttv2_fast_saturation_observations = int(
                state.get("ttv2_fast_saturation_observations", 0)
            )
            self.ttv2_buffer_input_abs = float(
                state.get("ttv2_buffer_input_abs", 0.0)
            )
            if self.method == "ttv2_aihwkit_1p1_minibatch_equation":
                deferred = state.get("ttv2_parameter_deferred_crossings")
                freeze_step = state.get("ttv2_parameter_freeze_step")
                skipped = state.get("ttv2_parameter_skipped_steps")
                if (
                    not isinstance(deferred, list)
                    or not isinstance(freeze_step, list)
                    or not isinstance(skipped, list)
                    or len(deferred) != len(self.bindings)
                    or len(freeze_step) != len(self.bindings)
                    or len(skipped) != len(self.bindings)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in deferred + skipped
                    )
                    or any(
                        value is not None
                        and (
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 0
                        )
                        for value in freeze_step
                    )
                ):
                    raise ValueError(
                        "Expected exact saved per-parameter TTv2 cap state."
                    )
                self.ttv2_parameter_deferred_crossings = list(deferred)
                self.ttv2_parameter_freeze_step = list(freeze_step)
                self.ttv2_parameter_skipped_steps = list(skipped)
        self.transfer_start = [int(value) for value in state.get("transfer_start", [])]
        self.transfer_count = [int(value) for value in state.get("transfer_count", [])]
        self.transfer_opportunities = int(state.get("transfer_opportunities", 0))
        if self.fast_plant is not None:
            if state.get("fast_population_fingerprint") != self.fast_population.fingerprint:
                raise ValueError("Expected the saved physical fast population fingerprint.")
            self.fast_plant.load_state(
                persistent=state["fast_persistent"].to(self.device),
                apparent=state["fast_apparent"].to(self.device),
            )
            assert self.fast_generator is not None and self.fast_baseline is not None
            self.fast_generator.set_state(state["fast_generator_state"].detach().cpu())
            self.fast_baseline.copy_(state["fast_baseline"].to(self.device))
            assert self.fast_ledger is not None
            self.fast_ledger.load_state_dict(state["fast_ledger"])
            self.fast_reset_verify_reads = int(state["fast_reset_verify_reads"])
            self.fast_reset_reversals = int(state["fast_reset_reversals"])
            self.fast_reset_failures = int(state["fast_reset_failures"])
            self.fast_commissioning = deepcopy(state.get("fast_commissioning"))
        self._apply_slow_apparent()

    def final_bundle(self) -> dict[str, Any]:
        fast_population = None
        if self.fast_population is not None:
            fast_population = {
                "fingerprint": self.fast_population.fingerprint,
                "assignment_seed": self.fast_population.assignment_seed,
                "binding_keys": self.fast_population.binding_keys,
                "binding_shapes": self.fast_population.binding_shapes,
                "scalars": {
                    "nominal_dw_min": self.fast_population.nominal_dw_min,
                    "dw_min_std": self.fast_population.dw_min_std,
                    "write_noise_std": self.fast_population.write_noise_std,
                    "aihwkit_version": self.fast_population.aihwkit_version,
                },
                "tensors": self.fast_population.tensor_state(),
                "receipt": deepcopy(self.fast_population_receipt),
            }
        return {
            "schema": RECOVERY_SCHEMA,
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "source_deployment_sha256": self.source_deployment_sha256,
            "source_population_fingerprint": self.slow_population.fingerprint,
            "device_model_sha256": self.device_model_sha256,
            "parameters": deepcopy(self.parameters),
            "binding_keys": self.keys,
            "binding_shapes": self.shapes,
            "source_raw_apparent_endpoint": self.source_raw_apparent.detach().cpu().clone(),
            "source_apparent_endpoint": self.source_apparent.detach().cpu().clone(),
            "source_persistent_endpoint": self.source_persistent.detach().cpu().clone(),
            "requested_target": self.requested_target.detach().cpu().clone(),
            "reset_baseline": self.reset_baseline.detach().cpu().clone(),
            "apparent_endpoint": self.current_apparent_endpoint(clipped=True).detach().cpu().clone(),
            "raw_apparent_endpoint": self.current_apparent_endpoint(clipped=False).detach().cpu().clone(),
            "persistent_endpoint": self.current_persistent_endpoint().detach().cpu().clone(),
            "generator_state_after_recovery": self.slow_generator.get_state().detach().cpu().clone(),
            "slow_ledger": self.slow_ledger.state_dict(),
            "fast_population": fast_population,
            "fast_persistent_endpoint": (
                None
                if self.fast_plant is None
                else ((self.fast_plant.persistent + 1.0) / 2.0).detach().cpu().clone()
            ),
            "fast_raw_apparent_endpoint": (
                None
                if self.fast_plant is None
                else ((self.fast_plant.apparent + 1.0) / 2.0).detach().cpu().clone()
            ),
            "fast_baseline": (
                None if self.fast_baseline is None else self.fast_baseline.detach().cpu().clone()
            ),
            "ideal_fast_state": (
                None
                if self.ideal_fast is None
                else [value.detach().cpu().clone() for value in self.ideal_fast]
            ),
            "ttv2_hidden_state": (
                None
                if self.ttv2_hidden is None
                else [value.detach().cpu().clone() for value in self.ttv2_hidden]
            ),
            "fast_commissioning": deepcopy(self.fast_commissioning),
            "report": self.report(),
        }


__all__ = [
    "IbmOmDeployedRecovery",
    "RECOVERY_SCHEMA",
    "RECOVERY_SCHEMA_VERSION",
    "sample_physical_fast_population",
]
