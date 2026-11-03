"""methods.optimizers.om — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
import torch
from hardware.om_crossbar import CrossbarTileSpec, IbmOmEffectiveCrossbarPlant, _LocalStarUpdatePort, _layout_receipt, _normalize_crossbar_layout, layer_cell_slices


class PulseAdam:
    """Digital Adam moments followed by one open-loop OM pulse opportunity."""

    STATE_SCHEMA = "ebl.ibm_om_crossbar_pulse_adam"

    def __init__(
        self,
        *,
        size: int,
        layout: Sequence[CrossbarTileSpec],
        learning_rates: Sequence[float],
        betas: Sequence[float],
        epsilon: float,
        layer_scope: str,
        nominal_dw_min: float,
        pulse_cap_per_cell: int | None,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> None:
        tiles = _normalize_crossbar_layout(layout)
        rates = tuple(float(value) for value in learning_rates)
        beta = tuple(float(value) for value in betas)
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or size != sum(tile.cells for tile in tiles)
            or len(rates) != 2
            or any(not math.isfinite(value) or value < 0.0 for value in rates)
            or len(beta) != 2
            or not 0.0 <= beta[0] < 1.0
            or not 0.0 <= beta[1] < 1.0
            or not math.isfinite(epsilon)
            or epsilon <= 0.0
            or not math.isfinite(nominal_dw_min)
            or nominal_dw_min <= 0.0
            or (
                pulse_cap_per_cell is not None
                and (
                    isinstance(pulse_cap_per_cell, bool)
                    or pulse_cap_per_cell < 1
                )
            )
            or layer_scope not in {"all", "input_only", "output_only"}
        ):
            raise ValueError("Expected a valid two-layer pulse-Adam configuration.")
        self.device = torch.device(device)
        self.generator = generator
        self.beta_1, self.beta_2 = beta
        self._learning_rates = rates
        self._layer_scope = layer_scope
        self._layout_receipt_value = _layout_receipt(tiles)
        self.epsilon = float(epsilon)
        self.nominal_dw_min = float(nominal_dw_min)
        self.pulse_cap_per_cell = (
            None if pulse_cap_per_cell is None else int(pulse_cap_per_cell)
        )
        self.step_index = 0
        self.learning_rate_scale = 1.0
        self.first_moment = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.device = self.first_moment.device
        self.second_moment = torch.zeros_like(self.first_moment)
        slices = layer_cell_slices(tiles)
        self.learning_rate = torch.empty_like(self.first_moment)
        self.learning_rate[slices[0]] = rates[0]
        self.learning_rate[slices[1]] = rates[1]
        self.enabled = torch.zeros(size, dtype=torch.bool, device=self.device)
        if layer_scope in {"all", "input_only"}:
            self.enabled[slices[0]] = True
        if layer_scope in {"all", "output_only"}:
            self.enabled[slices[1]] = True
        self.requested = 0
        self.applied = 0
        self.probability_clipped = 0
        self.blocked_at_cap = 0
        self.pulse_count = torch.zeros(size, dtype=torch.int64, device=self.device)
        self._layer_slices = slices

    def step(
        self,
        gradient: torch.Tensor,
        plant: IbmOmEffectiveCrossbarPlant | _LocalStarUpdatePort,
    ) -> dict[str, int | float]:
        value = gradient.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if (
            plant.size != self.first_moment.numel()
            or value.shape != self.first_moment.shape
            or not bool(torch.all(torch.isfinite(value)))
        ):
            raise ValueError("Expected one finite gradient per crosspoint.")
        self.step_index += 1
        self.first_moment.mul_(self.beta_1).add_(value, alpha=1.0 - self.beta_1)
        self.second_moment.mul_(self.beta_2).addcmul_(
            value, value, value=1.0 - self.beta_2
        )
        first_hat = self.first_moment / (1.0 - self.beta_1**self.step_index)
        second_hat = self.second_moment / (1.0 - self.beta_2**self.step_index)
        command = -self.learning_rate * first_hat / (second_hat.sqrt() + self.epsilon)
        if not math.isfinite(self.learning_rate_scale) or not 0 < self.learning_rate_scale <= 1:
            raise ValueError('Expected a finite learning-rate scale in (0,1].')
        command = command * self.learning_rate_scale
        command = torch.where(self.enabled, command, torch.zeros_like(command))
        raw_probability = command.abs() / self.nominal_dw_min
        clipped = raw_probability > 1.0
        probability = raw_probability.clamp(max=1.0)
        candidate = torch.rand(
            probability.shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        ) < probability
        at_cap = (
            torch.zeros_like(self.enabled)
            if self.pulse_cap_per_cell is None
            else self.pulse_count >= self.pulse_cap_per_cell
        )
        blocked = candidate & self.enabled & at_cap
        selected = candidate & self.enabled & ~at_cap
        direction = torch.sign(command).to(torch.int8) * selected.to(torch.int8)
        requested = int((raw_probability > 0.0).sum().item())
        applied = int(selected.sum().item())
        clipped_count = int(clipped.sum().item())
        blocked_count = int(blocked.sum().item())
        self.requested += requested
        self.applied += applied
        self.probability_clipped += clipped_count
        self.blocked_at_cap += blocked_count
        self.pulse_count += selected.to(torch.int64)
        plant.pulse(direction)
        return {
            "requested_nonzero_commands": requested,
            "commanded_pulses": applied,
            "applied_pulses": applied,
            "probability_clipped": clipped_count,
            "blocked_at_cap": blocked_count,
            "maximum_probability_before_clip": float(raw_probability.max().item()),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "size": self.first_moment.numel(),
            "layout": self._layout_receipt_value,
            "learning_rates": self._learning_rates,
            "betas": (self.beta_1, self.beta_2),
            "epsilon": self.epsilon,
            "layer_scope": self._layer_scope,
            "nominal_dw_min": self.nominal_dw_min,
            "pulse_cap_per_cell": self.pulse_cap_per_cell,
            "device": str(self.device),
            "forward_state": "held_apparent_q",
            "write_state": "persistent_q",
            "gradient_handoff": "identity_ste_apparent_q_to_persistent_pulse_update",
        }

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint Adam moments, endurance counts, counters, and RNG exactly."""

        return {
            "schema": self.STATE_SCHEMA,
            "schema_version": 1,
            "contract": self.contract(),
            "step_index": self.step_index,
            "learning_rate_scale": self.learning_rate_scale,
            "first_moment": self.first_moment.detach().cpu().clone(),
            "second_moment": self.second_moment.detach().cpu().clone(),
            "requested": self.requested,
            "applied": self.applied,
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "pulse_count": self.pulse_count.detach().cpu().clone(),
            "generator_state": self.generator.get_state().detach().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore only an exactly matching pulse-Adam configuration."""

        expected = {
            "schema",
            "schema_version",
            "contract",
            "step_index",
            "first_moment",
            "second_moment",
            "requested",
            "applied",
            "probability_clipped",
            "blocked_at_cap",
            "pulse_count",
            "generator_state",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) not in (expected, expected | {'learning_rate_scale'})
            or state.get("schema") != self.STATE_SCHEMA
            or isinstance(state.get("schema_version"), bool)
            or state.get("schema_version") != 1
            or state.get("contract") != self.contract()
        ):
            raise ValueError("Expected a matching IBM OM pulse-Adam state.")

        def nonnegative_integer(name: str) -> int:
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Expected non-negative pulse-Adam counters.")
            return value

        scale = float(state.get('learning_rate_scale', 1.0))
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError('Expected a saved learning-rate scale in (0,1].')
        self.learning_rate_scale = scale
        step_index = nonnegative_integer("step_index")
        requested = nonnegative_integer("requested")
        applied = nonnegative_integer("applied")
        probability_clipped = nonnegative_integer("probability_clipped")
        blocked_at_cap = nonnegative_integer("blocked_at_cap")
        pending: dict[str, torch.Tensor] = {}
        for name, dtype in (
            ("first_moment", torch.float32),
            ("second_moment", torch.float32),
            ("pulse_count", torch.int64),
        ):
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != self.first_moment.shape
                or value.dtype != dtype
            ):
                raise ValueError(f"Expected a matching pulse-Adam tensor {name!r}.")
            pending[name] = value.detach().to(self.device).clone()
        if (
            not bool(torch.all(torch.isfinite(pending["first_moment"])))
            or not bool(torch.all(torch.isfinite(pending["second_moment"])))
            or bool(torch.any(pending["second_moment"] < 0.0))
            or bool(torch.any(pending["pulse_count"] < 0))
            or applied != int(pending["pulse_count"].sum().item())
            or applied + blocked_at_cap > requested
            or probability_clipped > requested
            or (
                self.pulse_cap_per_cell is not None
                and bool(torch.any(pending["pulse_count"] > self.pulse_cap_per_cell))
            )
        ):
            raise ValueError("Expected a self-consistent pulse-Adam state.")
        generator_state = state["generator_state"]
        if (
            not isinstance(generator_state, torch.Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.device.type != "cpu"
            or generator_state.ndim != 1
        ):
            raise ValueError("Expected a saved pulse-Adam generator state tensor.")
        probe = torch.Generator(device=self.device.type)
        try:
            probe.set_state(generator_state.detach().cpu())
        except RuntimeError as error:
            raise ValueError("Expected a valid pulse-Adam generator state.") from error

        self.step_index = step_index
        self.first_moment.copy_(pending["first_moment"])
        self.second_moment.copy_(pending["second_moment"])
        self.requested = requested
        self.applied = applied
        self.probability_clipped = probability_clipped
        self.blocked_at_cap = blocked_at_cap
        self.pulse_count.copy_(pending["pulse_count"])
        self.generator.set_state(generator_state.detach().cpu())

    def report(self) -> dict[str, Any]:
        layer_counts = [
            int(self.pulse_count[layer_slice].sum().item())
            for layer_slice in self._layer_slices
        ]
        return {
            "optimizer_steps": self.step_index,
            "requested_nonzero_commands": self.requested,
            "commanded_pulses": self.applied,
            "applied_pulses": self.applied,
            "applied_pulses_by_layer": layer_counts,
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "pulse_cap_per_cell": self.pulse_cap_per_cell,
            "cells_at_cap": (
                0
                if self.pulse_cap_per_cell is None
                else int(
                    (self.pulse_count >= self.pulse_cap_per_cell).sum().item()
                )
            ),
            "commanded_cells": int((self.pulse_count > 0).sum().item()),
            "changed_cells": int((self.pulse_count > 0).sum().item()),
            "maximum_pulses_per_cell": int(self.pulse_count.max().item()),
            "enabled_cells": int(self.enabled.sum().item()),
        }


class PulseSGD:
    """Moment-free local gradient to stochastic one-pulse conversion."""

    def __init__(
        self,
        *,
        size: int,
        layout: Sequence[CrossbarTileSpec],
        learning_rates: Sequence[float],
        layer_scope: str,
        nominal_dw_min: float,
        pulse_cap_per_cell: int,
        pulse_rule: str,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> None:
        rates = tuple(float(value) for value in learning_rates)
        if (
            size < 1
            or len(rates) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in rates)
            or not math.isfinite(nominal_dw_min)
            or nominal_dw_min <= 0.0
            or isinstance(pulse_cap_per_cell, bool)
            or pulse_cap_per_cell < 1
            or layer_scope not in {"all", "input_only", "output_only"}
            or pulse_rule not in {"stochastic_pulse_sgd", "stochastic_pulse_sign_sgd"}
        ):
            raise ValueError("Expected a valid two-layer moment-free pulse-SGD configuration.")
        self.device = torch.device(device)
        self.generator = generator
        self.nominal_dw_min = float(nominal_dw_min)
        self.pulse_cap_per_cell = int(pulse_cap_per_cell)
        self.pulse_rule = pulse_rule
        self.step_index = 0
        slices = layer_cell_slices(layout)
        self._learning_rates = rates
        self.enabled = torch.zeros(size, dtype=torch.bool, device=self.device)
        if layer_scope in {"all", "input_only"}:
            self.enabled[slices[0]] = True
        if layer_scope in {"all", "output_only"}:
            self.enabled[slices[1]] = True
        self.requested = 0
        self.commanded = 0
        self.probability_clipped = 0
        self.blocked_at_cap = 0
        self.pulse_count = torch.zeros(size, dtype=torch.int64, device=self.device)
        self._layer_slices = slices

    def step(
        self,
        gradient: torch.Tensor,
        plant: IbmOmEffectiveCrossbarPlant | _LocalStarUpdatePort,
    ) -> dict[str, int | float]:
        value = gradient.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if value.shape != self.pulse_count.shape or not bool(torch.all(torch.isfinite(value))):
            raise ValueError("Expected one finite local gradient per crosspoint.")
        self.step_index += 1
        magnitude = (
            value.abs()
            if self.pulse_rule == "stochastic_pulse_sgd"
            else (value != 0).to(torch.float32)
        )
        raw_probability = torch.zeros_like(magnitude)
        for layer_index, layer_slice in enumerate(self._layer_slices):
            raw_probability[layer_slice] = (
                self._learning_rates[layer_index]
                * magnitude[layer_slice]
                / self.nominal_dw_min
            )
        raw_probability = torch.where(
            self.enabled, raw_probability, torch.zeros_like(raw_probability)
        )
        clipped = raw_probability > 1.0
        probability = raw_probability.clamp(max=1.0)
        candidate = torch.rand(
            probability.shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        ) < probability
        at_cap = self.pulse_count >= self.pulse_cap_per_cell
        blocked = candidate & self.enabled & at_cap
        selected = candidate & self.enabled & ~at_cap
        direction = -torch.sign(value).to(torch.int8) * selected.to(torch.int8)
        requested = int((raw_probability > 0.0).sum().item())
        commanded = int(selected.sum().item())
        clipped_count = int(clipped.sum().item())
        blocked_count = int(blocked.sum().item())
        self.requested += requested
        self.commanded += commanded
        self.probability_clipped += clipped_count
        self.blocked_at_cap += blocked_count
        self.pulse_count += selected.to(torch.int64)
        plant.pulse(direction)
        return {
            "requested_nonzero_commands": requested,
            "commanded_pulses": commanded,
            "probability_clipped": clipped_count,
            "blocked_at_cap": blocked_count,
            "maximum_probability_before_clip": float(raw_probability.max().item()),
        }

    def report(self) -> dict[str, Any]:
        layer_counts = [
            int(self.pulse_count[layer_slice].sum().item())
            for layer_slice in self._layer_slices
        ]
        return {
            "optimizer": "moment_free_pulse_sgd",
            "pulse_rule": self.pulse_rule,
            "learning_rates_q": list(self._learning_rates),
            "nominal_dw_min": self.nominal_dw_min,
            "command_direction": "-sign(gradient_q)",
            "selection_probability": (
                "min(1,learning_rate_q*abs(gradient_q)/nominal_dw_min)"
            ),
            "optimizer_steps": self.step_index,
            "requested_nonzero_commands": self.requested,
            "commanded_pulses": self.commanded,
            "commanded_pulses_by_layer": layer_counts,
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "pulse_cap_per_cell": self.pulse_cap_per_cell,
            "cells_at_cap": int(
                (self.pulse_count >= self.pulse_cap_per_cell).sum().item()
            ),
            "commanded_cells": int((self.pulse_count > 0).sum().item()),
            "maximum_pulses_per_cell": int(self.pulse_count.max().item()),
            "enabled_cells": int(self.enabled.sum().item()),
            "digital_first_moment_values": 0,
            "digital_second_moment_values": 0,
            "fp32_shadow_weight_values": 0,
            "persistent_fp32_update_state_values": 0,
            "persistent_per_cell_state": "integer_endurance_counter_only",
        }
