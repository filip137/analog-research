"""Label-local stochastic-pulse BP and Tiki-Taka for IBM-OM crossbars.

This module models the update path of the standard analog-crossbar --
digital-ReLU -- analog-crossbar comparator.  It deliberately contains no
teacher query, autograd handoff, Adam moments, FP32 shadow weights, fault map,
or final program-and-verify operation.

The stochastic-compressed bit-line equations follow the AIHWKit 1.1
``UpdateParameters`` contract: update bit-line management chooses a separate
length (capped at 31) for every example and physical tile, update management
balances activation and error probabilities, and one sampled row/column bit
is shared by all crosspoints on that line.  Exact positive and negative
coincidence counts are retained.  The explicit IBM-OM plant applies those
counts in grouped sign order, so this is distribution-level parity rather
than a claim of native C++ bit-order replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from hardware.om_crossbar import (
    CrossbarTileSpec,
    StandardCrossbarForwardStates,
    tensor_sha256,
)
from hardware.populations import IbmReramArrayPopulation


_PULSE_TYPE = "stochastic_compressed"
_GROUPED_ORDER = "all_positive_rounds_then_all_negative_rounds"


def _layout_receipt(
    layout: Sequence[CrossbarTileSpec],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    tiles = tuple(layout)
    if not tiles or {tile.layer_index for tile in tiles} != {0, 1}:
        raise ValueError("Expected a non-empty two-layer physical crossbar layout.")
    receipt = tuple(
        (
            tile.key,
            tile.layer_index,
            tile.tile_index,
            tile.input_start,
            tile.input_stop,
            tile.out_features,
        )
        for tile in tiles
    )
    if len({tile.key for tile in tiles}) != len(tiles):
        raise ValueError("Expected unique physical crossbar tile keys.")
    for layer_index in (0, 1):
        selected = [tile for tile in tiles if tile.layer_index == layer_index]
        expected_start = 0
        output_width = selected[0].out_features
        for tile in selected:
            if (
                tile.input_start != expected_start
                or tile.input_stop <= tile.input_start
                or tile.out_features != output_width
            ):
                raise ValueError(
                    "Expected contiguous balanced tiles with one output width per layer."
                )
            expected_start = tile.input_stop
    return receipt


def _layout_size(layout: Sequence[CrossbarTileSpec]) -> int:
    return sum(tile.cells for tile in layout)


def _layer_shapes(
    layout: Sequence[CrossbarTileSpec],
) -> tuple[tuple[int, int], tuple[int, int]]:
    result = []
    for layer_index in (0, 1):
        selected = [tile for tile in layout if tile.layer_index == layer_index]
        result.append(
            (sum(tile.shape[0] for tile in selected), selected[0].out_features)
        )
    return result[0], result[1]


def _strict_positive_pair(
    name: str,
    values: Sequence[float],
) -> tuple[float, float]:
    converted = tuple(float(value) for value in values)
    if len(converted) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in converted
    ):
        raise ValueError(f"Expected {name} to contain two finite positive values.")
    return converted[0], converted[1]


def _strict_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Expected {name} to be a non-negative integer.")
    return value


def _validate_stochastic_contract(
    *,
    desired_bl: int,
    fixed_bl: bool,
    update_bl_management: bool,
    update_management: bool,
    um_grad_scale: float,
) -> None:
    if (
        isinstance(desired_bl, bool)
        or desired_bl != 31
        or fixed_bl is not True
        or update_bl_management is not True
        or update_management is not True
        or isinstance(um_grad_scale, bool)
        or not isinstance(um_grad_scale, (int, float))
        or not math.isfinite(float(um_grad_scale))
        or float(um_grad_scale) != 1.0
    ):
        raise ValueError(
            "Expected the AIHWKit-1.1 stochastic-compressed contract: "
            "desired_bl=31, fixed_bl=true, update_bl_management=true, "
            "update_management=true, and um_grad_scale=1.0."
        )


@dataclass(frozen=True)
class FastArrayZeroTarget:
    """Commissioning target for a physical Tiki-Taka accumulation array."""

    target_q: torch.Tensor
    report: dict[str, Any]


def build_tiki_taka_fast_zero_target(
    population: IbmReramArrayPopulation,
) -> FastArrayZeroTarget:
    """Build one strict, defect-blind ``q=0`` target for every fast cell.

    Runtime code must program-and-verify this target before recovery without
    supplying an eligibility or fault mask.  Unreachable cells therefore
    exhaust or stop according to the ordinary P&V controller; their target is
    never rewritten to a hidden per-cell bound or sampled stuck value.

    Sampled support and corruption metadata are inspected only after the
    literal all-zero target exists, to produce an explicitly post-hoc
    feasibility diagnostic.  They do not influence target construction.
    """

    if not isinstance(population, IbmReramArrayPopulation) or population.size < 1:
        raise ValueError("Expected a non-empty IBM-OM fast-array population.")
    target = torch.zeros(
        population.size,
        dtype=torch.float32,
        device=population.logical_min.device,
    )

    # Everything below this line is diagnostic only.  In particular, neither
    # sampled support nor the corruption mask can change ``target``.
    minimum = population.logical_min.detach().to(torch.float32)
    maximum = population.logical_max.detach().to(torch.float32)
    if not bool(torch.all(torch.isfinite(minimum))) or not bool(
        torch.all(torch.isfinite(maximum))
    ) or bool(torch.any(maximum < minimum)):
        raise ValueError("Expected finite ordered fast-array logical support.")
    target_in_support = (minimum <= 0.0) & (maximum >= 0.0)
    corrupt = population.corrupt.to(device=target.device, dtype=torch.bool)
    report = {
        "schema": "ebl.ibm_om_tiki_taka_fast_zero_target",
        "schema_version": 2,
        "policy": "strict_q_zero_target_mask_blind",
        "population_fingerprint": population.fingerprint,
        "aihwkit_version": population.aihwkit_version,
        "cells": population.size,
        "target_q_sha256": tensor_sha256(target),
        "target_all_zero": bool(torch.count_nonzero(target).item() == 0),
        "target_construction": {
            "requested_q": "strict_zero_for_every_physical_cell",
            "uses_per_cell_support": False,
            "uses_corrupt_mask": False,
            "uses_sampled_stuck_values": False,
            "mutates_array": False,
        },
        "required_controller_contract": (
            "ordinary_apparent_read_program_verify_without_fault_mask_or_eligibility"
        ),
        "posthoc_support_diagnostic": {
            "analysis_only_not_used_by_target_or_controller": True,
            "target_in_sampled_support_cells": int(target_in_support.sum().item()),
            "target_outside_sampled_support_cells": int(
                (~target_in_support).sum().item()
            ),
            "published_or_preexisting_corrupt_cells": int(corrupt.sum().item()),
            "corrupt_cells_with_target_in_support": int(
                (corrupt & target_in_support).sum().item()
            ),
            "corrupt_cells_with_target_outside_support": int(
                (corrupt & ~target_in_support).sum().item()
            ),
        },
        "direct_state_assignment_is_physical_parity": False,
    }
    return FastArrayZeroTarget(target_q=target, report=report)


@dataclass(frozen=True)
class StochasticCompressedPulseCounts:
    """Exact sign-separated coincidences from one outer-product event."""

    positive: torch.Tensor
    negative: torch.Tensor
    bit_line_length: int
    uncapped_bit_line_length: int
    input_probability_clipped: int
    error_probability_clipped: int
    input_probability_maximum_before_clip: float
    error_probability_maximum_before_clip: float
    asserted_input_bits: int
    asserted_error_bits: int

    @property
    def coincidences(self) -> int:
        return int((self.positive + self.negative).sum().item())


@torch.no_grad()
def stochastic_compressed_pulse_counts(
    inputs: torch.Tensor,
    errors: torch.Tensor,
    *,
    learning_rate: float,
    weight_granularity: float,
    generator: torch.Generator,
    desired_bl: int = 31,
    fixed_bl: bool = True,
    update_bl_management: bool = True,
    update_management: bool = True,
    um_grad_scale: float = 1.0,
) -> StochasticCompressedPulseCounts:
    """Sample AIHWKit-1.1-style shared row/column stochastic bit lines."""

    _validate_stochastic_contract(
        desired_bl=desired_bl,
        fixed_bl=fixed_bl,
        update_bl_management=update_bl_management,
        update_management=update_management,
        um_grad_scale=um_grad_scale,
    )
    x = inputs.detach()
    d = errors.detach()
    rate = float(learning_rate)
    granularity = float(weight_granularity)
    if (
        x.ndim != 1
        or d.ndim != 1
        or x.numel() < 1
        or d.numel() < 1
        or x.device != d.device
        or x.dtype != d.dtype
        or not x.dtype.is_floating_point
        or not bool(torch.all(torch.isfinite(x)))
        or not bool(torch.all(torch.isfinite(d)))
        or not math.isfinite(rate)
        or rate <= 0.0
        or not math.isfinite(granularity)
        or granularity <= 0.0
        or not isinstance(generator, torch.Generator)
    ):
        raise ValueError(
            "Expected finite floating outer-product vectors, a positive learning "
            "rate/granularity, and an explicit Torch generator."
        )

    positive = torch.zeros((x.numel(), d.numel()), dtype=torch.int64, device=x.device)
    negative = torch.zeros_like(positive)
    x_maximum = float(x.abs().max().item())
    d_maximum = float(d.abs().max().item())
    if x_maximum == 0.0 or d_maximum == 0.0:
        return StochasticCompressedPulseCounts(
            positive=positive,
            negative=negative,
            bit_line_length=0,
            uncapped_bit_line_length=0,
            input_probability_clipped=0,
            error_probability_clipped=0,
            input_probability_maximum_before_clip=0.0,
            error_probability_maximum_before_clip=0.0,
            asserted_input_bits=0,
            asserted_error_bits=0,
        )

    x_value = x_maximum
    d_value = float(um_grad_scale) * d_maximum
    requested = rate * x_value * d_value / granularity
    if not math.isfinite(requested):
        raise ValueError("Expected a finite managed bit-line length.")
    uncapped = max(1, math.ceil(requested))
    bit_line_length = min(desired_bl, uncapped)

    # AIHWKit's update-BL management clips the managed error scale when the
    # requested pulse train exceeds the maximum length.  Update management
    # then uses the square root of the x/d maximum ratio, not that ratio
    # itself, to balance the two Bernoulli streams.
    if requested > desired_bl:
        d_value *= desired_bl / requested
    base = math.sqrt(rate / (granularity * bit_line_length))
    management_root = math.sqrt(x_value / d_value)
    # AIHWKit calls the error-line scale A and the input-line scale B.
    error_scale = base * management_root
    input_scale = base / management_root
    input_probability = x.abs() * input_scale
    error_probability = d.abs() * error_scale
    input_maximum_before_clip = float(input_probability.max().item())
    error_maximum_before_clip = float(error_probability.max().item())
    input_clipped = int((input_probability > 1.0).sum().item())
    error_clipped = int((error_probability > 1.0).sum().item())
    input_probability = input_probability.clamp(max=1.0)
    error_probability = error_probability.clamp(max=1.0)

    try:
        input_bits = torch.rand(
            (bit_line_length, x.numel()),
            dtype=torch.float32,
            device=x.device,
            generator=generator,
        ) < input_probability.unsqueeze(0)
        error_bits = torch.rand(
            (bit_line_length, d.numel()),
            dtype=torch.float32,
            device=d.device,
            generator=generator,
        ) < error_probability.unsqueeze(0)
    except RuntimeError as error:
        raise ValueError(
            "Expected the bit-line generator to match the update device."
        ) from error

    coincidences = torch.matmul(
        input_bits.transpose(0, 1).to(torch.float32),
        error_bits.to(torch.float32),
    ).to(torch.int64)
    gradient_sign = torch.sign(x).unsqueeze(1) * torch.sign(d).unsqueeze(0)
    update_sign = -gradient_sign
    positive.copy_(torch.where(update_sign > 0.0, coincidences, 0))
    negative.copy_(torch.where(update_sign < 0.0, coincidences, 0))
    return StochasticCompressedPulseCounts(
        positive=positive,
        negative=negative,
        bit_line_length=bit_line_length,
        uncapped_bit_line_length=uncapped,
        input_probability_clipped=input_clipped,
        error_probability_clipped=error_clipped,
        input_probability_maximum_before_clip=input_maximum_before_clip,
        error_probability_maximum_before_clip=error_maximum_before_clip,
        asserted_input_bits=int(input_bits.sum().item()),
        asserted_error_bits=int(error_bits.sum().item()),
    )


@dataclass(frozen=True)
class ManualCeBackpropFactors:
    """Manual label-CE BP factors in the physical effective-``q`` coordinate."""

    layer_inputs: tuple[torch.Tensor, torch.Tensor]
    layer_errors_q: tuple[torch.Tensor, torch.Tensor]
    hidden_error: torch.Tensor
    output_error: torch.Tensor
    mean_cross_entropy: torch.Tensor
    correct: int
    states: StandardCrossbarForwardStates


@torch.no_grad()
def manual_label_ce_backprop(
    port: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> ManualCeBackpropFactors:
    """Compute two-layer CE BP using only forward and transpose-MVM ports."""

    contract = port.hardware_contract()
    scales = tuple(float(value) for value in contract.get("q_scales", ()))
    if len(scales) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in scales
    ):
        raise ValueError("Expected two finite positive physical-q scales from the port.")
    if (
        inputs.ndim != 2
        or labels.ndim != 1
        or labels.shape[0] != inputs.shape[0]
        or inputs.shape[0] < 1
        or not bool(torch.all(torch.isfinite(inputs)))
    ):
        raise ValueError("Expected finite rank-2 inputs and one label per example.")
    labels_device = labels.to(device=inputs.device, dtype=torch.int64)
    states = port.forward(inputs)
    logits = states.output_logits
    if (
        bool(torch.any(labels_device < 0))
        or bool(torch.any(labels_device >= logits.shape[1]))
        or not bool(torch.all(torch.isfinite(logits)))
    ):
        raise ValueError("Expected labels to address finite output logits.")
    log_probabilities = torch.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    output_error = probabilities.clone()
    rows = torch.arange(inputs.shape[0], device=inputs.device)
    output_error[rows, labels_device] -= 1.0
    output_error.div_(inputs.shape[0])
    hidden_error = port.output_transpose_mvm(output_error)
    hidden_error.mul_((states.hidden_preactivation > 0.0).to(hidden_error.dtype))
    mean_cross_entropy = -log_probabilities[rows, labels_device].mean()
    return ManualCeBackpropFactors(
        layer_inputs=(inputs.detach(), states.hidden_post_relu.detach()),
        layer_errors_q=(
            hidden_error * scales[0],
            output_error * scales[1],
        ),
        hidden_error=hidden_error,
        output_error=output_error,
        mean_cross_entropy=mean_cross_entropy,
        correct=int((logits.argmax(dim=1) == labels_device).sum().item()),
        states=states,
    )


_ENGINE_COUNTERS = (
    "events",
    "zero_events",
    "bit_line_length_sum",
    "maximum_bit_line_length",
    "uncapped_bit_line_length_sum",
    "maximum_uncapped_bit_line_length",
    "bit_line_cap_events",
    "input_probability_clipped_values",
    "error_probability_clipped_values",
    "row_bernoulli_draws",
    "column_bernoulli_draws",
    "asserted_input_bits",
    "asserted_error_bits",
    "positive_coincidences",
    "negative_coincidences",
)


def _empty_engine_counters() -> dict[str, int]:
    return {name: 0 for name in _ENGINE_COUNTERS}


class _StochasticBitLineEngine:
    """Serializable owner of one exact bit-line RNG stream."""

    STATE_SCHEMA = "ebl.ibm_om_stochastic_compressed_bit_lines"

    def __init__(
        self,
        *,
        generator: torch.Generator,
        device: torch.device | str,
        roles: Sequence[str],
        desired_bl: int,
        fixed_bl: bool,
        update_bl_management: bool,
        update_management: bool,
        um_grad_scale: float,
    ) -> None:
        _validate_stochastic_contract(
            desired_bl=desired_bl,
            fixed_bl=fixed_bl,
            update_bl_management=update_bl_management,
            update_management=update_management,
            um_grad_scale=um_grad_scale,
        )
        role_names = tuple(str(role) for role in roles)
        if (
            not isinstance(generator, torch.Generator)
            or not role_names
            or any(not role.strip() for role in role_names)
            or len(set(role_names)) != len(role_names)
        ):
            raise ValueError("Expected an explicit generator and unique bit-line roles.")
        self.generator = generator
        self.device = torch.device(device)
        if torch.device(generator.device).type != self.device.type:
            raise ValueError("Expected the bit-line generator to match the update device.")
        self.desired_bl = desired_bl
        self.fixed_bl = fixed_bl
        self.update_bl_management = update_bl_management
        self.update_management = update_management
        self.um_grad_scale = float(um_grad_scale)
        self.roles = role_names
        self.counters = {role: _empty_engine_counters() for role in role_names}

    def generate(
        self,
        inputs: torch.Tensor,
        errors: torch.Tensor,
        *,
        learning_rate: float,
        weight_granularity: float,
        role: str,
    ) -> StochasticCompressedPulseCounts:
        if role not in self.counters:
            raise ValueError("Expected a declared stochastic bit-line role.")
        result = stochastic_compressed_pulse_counts(
            inputs,
            errors,
            learning_rate=learning_rate,
            weight_granularity=weight_granularity,
            generator=self.generator,
            desired_bl=self.desired_bl,
            fixed_bl=self.fixed_bl,
            update_bl_management=self.update_bl_management,
            update_management=self.update_management,
            um_grad_scale=self.um_grad_scale,
        )
        counter = self.counters[role]
        counter["events"] += 1
        counter["zero_events"] += int(result.bit_line_length == 0)
        counter["bit_line_length_sum"] += result.bit_line_length
        counter["maximum_bit_line_length"] = max(
            counter["maximum_bit_line_length"], result.bit_line_length
        )
        counter["uncapped_bit_line_length_sum"] += result.uncapped_bit_line_length
        counter["maximum_uncapped_bit_line_length"] = max(
            counter["maximum_uncapped_bit_line_length"],
            result.uncapped_bit_line_length,
        )
        counter["bit_line_cap_events"] += int(
            result.uncapped_bit_line_length > self.desired_bl
        )
        counter["input_probability_clipped_values"] += (
            result.input_probability_clipped
        )
        counter["error_probability_clipped_values"] += (
            result.error_probability_clipped
        )
        counter["row_bernoulli_draws"] += result.bit_line_length * inputs.numel()
        counter["column_bernoulli_draws"] += result.bit_line_length * errors.numel()
        counter["asserted_input_bits"] += result.asserted_input_bits
        counter["asserted_error_bits"] += result.asserted_error_bits
        counter["positive_coincidences"] += int(result.positive.sum().item())
        counter["negative_coincidences"] += int(result.negative.sum().item())
        return result

    def contract(self) -> dict[str, Any]:
        return {
            "pulse_type": _PULSE_TYPE,
            "desired_bl": self.desired_bl,
            "fixed_bl": self.fixed_bl,
            "update_bl_management": self.update_bl_management,
            "update_management": self.update_management,
            "um_grad_scale": self.um_grad_scale,
            "per_example_per_physical_tile_management": True,
            "shared_row_column_bernoulli_streams": True,
            "coincidence_application_order": _GROUPED_ORDER,
            "parity_level": "aihwkit_1_1_distribution_level_not_bitwise_order",
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "schema_version": 1,
            "contract": self.contract(),
            "roles": self.roles,
            "counters": {
                role: dict(self.counters[role]) for role in self.roles
            },
            "generator_state": self.generator.get_state().detach().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema",
            "schema_version",
            "contract",
            "roles",
            "counters",
            "generator_state",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state["schema"] != self.STATE_SCHEMA
            or state["schema_version"] != 1
            or isinstance(state["schema_version"], bool)
            or state["contract"] != self.contract()
            or not isinstance(state["roles"], (tuple, list))
            or tuple(state["roles"]) != self.roles
            or not isinstance(state["counters"], Mapping)
            or set(state["counters"]) != set(self.roles)
        ):
            raise ValueError("Expected a matching stochastic bit-line engine state.")
        pending: dict[str, dict[str, int]] = {}
        for role in self.roles:
            value = state["counters"][role]
            if not isinstance(value, Mapping) or set(value) != set(_ENGINE_COUNTERS):
                raise ValueError("Expected complete stochastic bit-line counters.")
            converted = {}
            for name in _ENGINE_COUNTERS:
                converted[name] = _strict_nonnegative_int(name, value[name])
            if (
                converted["zero_events"] > converted["events"]
                or converted["bit_line_cap_events"] > converted["events"]
                or converted["maximum_bit_line_length"] > self.desired_bl
            ):
                raise ValueError("Expected self-consistent stochastic bit-line counters.")
            pending[role] = converted
        generator_state = state["generator_state"]
        if (
            not isinstance(generator_state, torch.Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.device.type != "cpu"
            or generator_state.ndim != 1
        ):
            raise ValueError("Expected a valid saved bit-line RNG state tensor.")
        probe = torch.Generator(device=self.device.type)
        try:
            probe.set_state(generator_state)
        except RuntimeError as error:
            raise ValueError("Expected a valid saved bit-line RNG state tensor.") from error
        self.counters = pending
        self.generator.set_state(generator_state)

    def report(self) -> dict[str, Any]:
        return {
            "contract": self.contract(),
            "roles": {role: dict(self.counters[role]) for role in self.roles},
            "generator_state_sha256": tensor_sha256(self.generator.get_state()),
        }


def _tile_offsets(
    layout: Sequence[CrossbarTileSpec],
) -> tuple[int, ...]:
    offsets = []
    offset = 0
    for tile in layout:
        offsets.append(offset)
        offset += tile.cells
    return tuple(offsets)


def _batch_tile_outer_product_counts(
    factors: ManualCeBackpropFactors,
    *,
    layout: Sequence[CrossbarTileSpec],
    learning_rates: Sequence[float],
    weight_granularity: float,
    engine: _StochasticBitLineEngine,
    role: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate one independently managed event per example and tile."""

    rates = _strict_positive_pair("learning_rates", learning_rates)
    tiles = tuple(layout)
    offsets = _tile_offsets(tiles)
    device = factors.layer_inputs[0].device
    positive = torch.zeros(_layout_size(tiles), dtype=torch.int64, device=device)
    negative = torch.zeros_like(positive)
    batch_size = factors.layer_inputs[0].shape[0]
    for example_index in range(batch_size):
        for tile_index, tile in enumerate(tiles):
            row = factors.layer_inputs[tile.layer_index][
                example_index, tile.input_start : tile.input_stop
            ]
            error = factors.layer_errors_q[tile.layer_index][example_index]
            counts = engine.generate(
                row,
                error,
                learning_rate=rates[tile.layer_index],
                weight_granularity=weight_granularity,
                role=role,
            )
            start = offsets[tile_index]
            stop = start + tile.cells
            positive[start:stop] += counts.positive.reshape(-1)
            negative[start:stop] += counts.negative.reshape(-1)
    return positive, negative


def _pulse_counts_by_layer(
    counts: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
) -> list[int]:
    result = [0, 0]
    offset = 0
    for tile in layout:
        stop = offset + tile.cells
        result[tile.layer_index] += int(counts[offset:stop].sum().item())
        offset = stop
    return result


def _validate_count_tensor(
    value: Any,
    *,
    size: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (size,)
        or value.dtype != torch.int64
        or bool(torch.any(value < 0))
    ):
        raise ValueError(f"Expected saved {name} to be non-negative int64 counts.")
    return value.detach().to(device).clone()


def _validate_progress(
    state: Mapping[str, Any],
) -> tuple[int, int, float, int]:
    step_index = _strict_nonnegative_int("step_index", state["step_index"])
    examples = _strict_nonnegative_int("examples", state["examples"])
    correct = _strict_nonnegative_int("correct", state["correct"])
    loss_sum = state["cross_entropy_sum"]
    if (
        isinstance(loss_sum, bool)
        or not isinstance(loss_sum, (int, float))
        or not math.isfinite(float(loss_sum))
        or float(loss_sum) < 0.0
        or correct > examples
        or (step_index == 0) != (examples == 0)
    ):
        raise ValueError("Expected self-consistent saved recovery progress.")
    return step_index, examples, float(loss_sum), correct


def _engine_role_counters(
    state: Any,
    role: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(state, Mapping)
        or not isinstance(state.get("counters"), Mapping)
        or not isinstance(state["counters"].get(role), Mapping)
    ):
        raise ValueError("Expected complete updater bit-line role counters.")
    return state["counters"][role]


class IbmOmDirectPulseSgd:
    """Manual label-CE BP with stochastic pulses written to the slow array."""

    STATE_SCHEMA = "ebl.ibm_om_direct_stochastic_pulse_sgd"

    def __init__(
        self,
        *,
        port: Any,
        layout: Sequence[CrossbarTileSpec],
        learning_rates_q: Sequence[float],
        generator: torch.Generator,
        device: torch.device | str,
        desired_bl: int = 31,
        fixed_bl: bool = True,
        update_bl_management: bool = True,
        update_management: bool = True,
        um_grad_scale: float = 1.0,
    ) -> None:
        self.__port = port
        self.layout = tuple(layout)
        self.layout_receipt = _layout_receipt(self.layout)
        self.size = _layout_size(self.layout)
        self.learning_rates_q = _strict_positive_pair(
            "learning_rates_q", learning_rates_q
        )
        self.device = torch.device(device)
        port_contract = port.hardware_contract()
        if (
            port_contract.get("layout") != self.layout_receipt
            or port_contract.get("size") != self.size
            or port_contract.get("device") != str(self.device)
            or port_contract.get("full_weight_tensor_exposed") is not False
            or port_contract.get("fault_mask_exposed") is not False
        ):
            raise ValueError(
                "Expected a matching capability-limited slow recovery port."
            )
        self.nominal_dw_min = float(port_contract.get("nominal_dw_min", math.nan))
        q_scales = tuple(float(value) for value in port_contract.get("q_scales", ()))
        if (
            not math.isfinite(self.nominal_dw_min)
            or self.nominal_dw_min <= 0.0
            or len(q_scales) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in q_scales)
        ):
            raise ValueError("Expected valid OM granularity and physical-q scales.")
        self.q_scales = q_scales
        self.engine = _StochasticBitLineEngine(
            generator=generator,
            device=self.device,
            roles=("slow_gradient",),
            desired_bl=desired_bl,
            fixed_bl=fixed_bl,
            update_bl_management=update_bl_management,
            update_management=update_management,
            um_grad_scale=um_grad_scale,
        )
        self.step_index = 0
        self.examples = 0
        self.cross_entropy_sum = 0.0
        self.correct = 0
        self.positive_counts = torch.zeros(
            self.size, dtype=torch.int64, device=self.device
        )
        self.negative_counts = torch.zeros_like(self.positive_counts)

    def contract(self) -> dict[str, Any]:
        return {
            "algorithm": "direct_analog_sgd_stochastic_compressed",
            "layout": self.layout_receipt,
            "size": self.size,
            "device": str(self.device),
            "learning_rates_q": self.learning_rates_q,
            "q_scales": self.q_scales,
            "nominal_dw_min": self.nominal_dw_min,
            "gradient_engine": "manual_label_cross_entropy_backpropagation",
            "cross_entropy_reduction": "mean",
            "optimizer_state": "none",
            "weight_state": "physical_persistent_and_apparent_device_state_no_shadow",
            "cumulative_pulse_cap": None,
            "teacher_queries": 0,
            "autograd_calls": 0,
            "adam_moment_values": 0,
            "fp32_shadow_weight_values": 0,
            "fault_mask_exposed": False,
            "bit_lines": self.engine.contract(),
        }

    @torch.no_grad()
    def step(self, inputs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        factors = manual_label_ce_backprop(self.__port, inputs, labels)
        positive, negative = _batch_tile_outer_product_counts(
            factors,
            layout=self.layout,
            learning_rates=self.learning_rates_q,
            weight_granularity=self.nominal_dw_min,
            engine=self.engine,
            role="slow_gradient",
        )
        self.__port.apply_grouped_pulse_counts(positive, negative)
        self.positive_counts += positive
        self.negative_counts += negative
        batch_size = inputs.shape[0]
        self.step_index += 1
        self.examples += batch_size
        self.cross_entropy_sum += float(factors.mean_cross_entropy.item()) * batch_size
        self.correct += factors.correct
        total = positive + negative
        return {
            "optimizer_step": self.step_index,
            "examples": batch_size,
            "cross_entropy_sum": (
                float(factors.mean_cross_entropy.item()) * batch_size
            ),
            "mean_label_cross_entropy": float(factors.mean_cross_entropy.item()),
            "pre_update_correct": factors.correct,
            "slow_commanded_positive_pulses": int(positive.sum().item()),
            "slow_commanded_negative_pulses": int(negative.sum().item()),
            "slow_commanded_pulses": int(total.sum().item()),
            "slow_commanded_cells": int((total > 0).sum().item()),
            "slow_commanded_pulses_by_layer": _pulse_counts_by_layer(
                total, self.layout
            ),
            "fast_commanded_pulses": 0,
            "fast_commanded_cells": 0,
            "slow_state_hash_receipt": self.__port.state_hash_receipt(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "schema_version": 1,
            "contract": self.contract(),
            "step_index": self.step_index,
            "examples": self.examples,
            "cross_entropy_sum": self.cross_entropy_sum,
            "correct": self.correct,
            "positive_counts": self.positive_counts.detach().cpu().clone(),
            "negative_counts": self.negative_counts.detach().cpu().clone(),
            "bit_line_engine": self.engine.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema",
            "schema_version",
            "contract",
            "step_index",
            "examples",
            "cross_entropy_sum",
            "correct",
            "positive_counts",
            "negative_counts",
            "bit_line_engine",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state["schema"] != self.STATE_SCHEMA
            or state["schema_version"] != 1
            or isinstance(state["schema_version"], bool)
            or state["contract"] != self.contract()
        ):
            raise ValueError("Expected a matching direct pulse-SGD state.")
        step_index, examples, loss_sum, correct = _validate_progress(state)
        positive = _validate_count_tensor(
            state["positive_counts"],
            size=self.size,
            name="positive_counts",
            device=self.device,
        )
        negative = _validate_count_tensor(
            state["negative_counts"],
            size=self.size,
            name="negative_counts",
            device=self.device,
        )
        role = _engine_role_counters(state["bit_line_engine"], "slow_gradient")
        if (
            role.get("events") != examples * len(self.layout)
            or role.get("positive_coincidences")
            != int(positive.sum().item())
            or role.get("negative_coincidences")
            != int(negative.sum().item())
        ):
            raise ValueError(
                "Expected direct pulse counts to match the exact per-example "
                "physical-tile bit-line events."
            )
        self.engine.load_state_dict(state["bit_line_engine"])
        self.step_index = step_index
        self.examples = examples
        self.cross_entropy_sum = loss_sum
        self.correct = correct
        self.positive_counts.copy_(positive)
        self.negative_counts.copy_(negative)

    def report(self) -> dict[str, Any]:
        total = self.positive_counts + self.negative_counts
        return {
            "schema": "ebl.ibm_om_direct_stochastic_pulse_sgd_report",
            "schema_version": 1,
            "contract": self.contract(),
            "optimizer_steps": self.step_index,
            "examples": self.examples,
            "mean_label_cross_entropy": (
                self.cross_entropy_sum / self.examples if self.examples else None
            ),
            "pre_update_accuracy": (
                self.correct / self.examples if self.examples else None
            ),
            "slow_commanded_positive_pulses": int(self.positive_counts.sum().item()),
            "slow_commanded_negative_pulses": int(self.negative_counts.sum().item()),
            "slow_commanded_pulses": int(total.sum().item()),
            "slow_commanded_cells": int((total > 0).sum().item()),
            "slow_commanded_pulses_by_layer": _pulse_counts_by_layer(
                total, self.layout
            ),
            "maximum_slow_commanded_pulses_per_cell": int(total.max().item()),
            "fast_commanded_pulses": 0,
            "fast_commanded_cells": 0,
            "positive_count_sha256": tensor_sha256(self.positive_counts),
            "negative_count_sha256": tensor_sha256(self.negative_counts),
            "bit_line_engine": self.engine.report(),
            "slow_state_hash_receipt": self.__port.state_hash_receipt(),
            "checkpoint_requires": ["slow_plant_state", "updater_state"],
        }


def _canonical_fast_initialization_receipt(
    value: Mapping[str, Any],
    *,
    fast_state_hash_receipt: Mapping[str, str],
    strict_zero_target_sha256: str,
) -> dict[str, Any]:
    """Validate a physical P&V receipt or an explicitly ideal smoke control."""

    if not isinstance(value, Mapping):
        raise ValueError("Expected a declared fast-array q-zero initialization receipt.")
    policy = value.get("policy")
    physical_program_verify = value.get("physical_program_verify")
    direct_state_assignment = value.get("direct_state_assignment")
    if policy == "program_verify_strict_q_zero_mask_blind":
        canonical_policy = "program_verify_strict_q_zero_mask_blind"
        expected_physical = True
        expected_direct = False
    elif policy == "ideal_direct_strict_q_zero":
        canonical_policy = "ideal_direct_strict_q_zero"
        expected_physical = False
        expected_direct = True
    else:
        raise ValueError(
            "Expected a declared strict-q-zero physical or ideal fast initialization."
        )
    target_hash = value.get("target_q_sha256")
    persistent_hash = value.get("post_program_persistent_sha256")
    apparent_hash = value.get("post_program_apparent_sha256")
    if (
        physical_program_verify is not expected_physical
        or direct_state_assignment is not expected_direct
        or target_hash != strict_zero_target_sha256
        or persistent_hash != fast_state_hash_receipt.get("persistent_sha256")
        or apparent_hash != fast_state_hash_receipt.get("apparent_sha256")
    ):
        raise ValueError(
            "Expected the strict-q-zero initialization receipt to match its "
            "literal target and the exact commissioned fast-array state."
        )
    return {
        "policy": canonical_policy,
        "target_q_sha256": target_hash,
        "post_program_persistent_sha256": persistent_hash,
        "post_program_apparent_sha256": apparent_hash,
        "physical_program_verify": expected_physical,
        "direct_state_assignment": expected_direct,
    }


class IbmOmTikiTakaV1:
    """TT-v1 ``gamma=0`` stochastic fast accumulation and C transfer."""

    STATE_SCHEMA = "ebl.ibm_om_tiki_taka_v1"

    def __init__(
        self,
        *,
        slow_port: Any,
        fast_port: Any,
        layout: Sequence[CrossbarTileSpec],
        learning_rates_q: Sequence[float],
        fast_initialization_receipt: Mapping[str, Any],
        generator: torch.Generator,
        device: torch.device | str,
        fast_lr: float = 1.0,
        gamma: float = 0.0,
        transfer_every: int = 1,
        units_in_mbatch: bool = True,
        n_reads_per_transfer: int = 1,
        transfer_lr: float = 1.0,
        scale_transfer_lr: bool = True,
        transfer_columns: bool = True,
        transfer_selection: str = "sequential_physical_tile_columns",
        with_reset_prob: float = 0.0,
        random_selection: bool = False,
        desired_bl: int = 31,
        fixed_bl: bool = True,
        update_bl_management: bool = True,
        update_management: bool = True,
        um_grad_scale: float = 1.0,
    ) -> None:
        self.__slow_port = slow_port
        self.__fast_port = fast_port
        self.layout = tuple(layout)
        self.layout_receipt = _layout_receipt(self.layout)
        self.size = _layout_size(self.layout)
        self.learning_rates_q = _strict_positive_pair(
            "learning_rates_q", learning_rates_q
        )
        self.device = torch.device(device)
        slow_contract = slow_port.hardware_contract()
        fast_contract = fast_port.hardware_contract()
        if (
            slow_contract.get("layout") != self.layout_receipt
            or fast_contract.get("layout") != self.layout_receipt
            or slow_contract.get("size") != self.size
            or fast_contract.get("size") != self.size
            or slow_contract.get("device") != str(self.device)
            or fast_contract.get("device") != str(self.device)
            or slow_contract.get("full_weight_tensor_exposed") is not False
            or fast_contract.get("full_weight_tensor_exposed") is not False
            or slow_contract.get("fault_mask_exposed") is not False
            or fast_contract.get("fault_mask_exposed") is not False
            or fast_contract.get("selected_slice_only") is not True
        ):
            raise ValueError("Expected matching capability-limited slow/fast ports.")
        self.slow_nominal_dw_min = float(
            slow_contract.get("nominal_dw_min", math.nan)
        )
        self.fast_nominal_dw_min = float(
            fast_contract.get("nominal_dw_min", math.nan)
        )
        q_scales = tuple(float(value) for value in slow_contract.get("q_scales", ()))
        if (
            any(
                not math.isfinite(value) or value <= 0.0
                for value in (self.slow_nominal_dw_min, self.fast_nominal_dw_min)
            )
            or len(q_scales) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in q_scales)
        ):
            raise ValueError("Expected valid slow/fast OM granularity and q scales.")
        self.q_scales = q_scales
        if (
            isinstance(fast_lr, bool)
            or float(fast_lr) != 1.0
            or isinstance(gamma, bool)
            or float(gamma) != 0.0
            or isinstance(transfer_every, bool)
            or not isinstance(transfer_every, int)
            or transfer_every < 1
            or units_in_mbatch is not True
            or isinstance(n_reads_per_transfer, bool)
            or not isinstance(n_reads_per_transfer, int)
            or n_reads_per_transfer < 1
            or isinstance(transfer_lr, bool)
            or not isinstance(transfer_lr, (int, float))
            or not math.isfinite(float(transfer_lr))
            or float(transfer_lr) <= 0.0
            or not isinstance(scale_transfer_lr, bool)
            or transfer_columns is not True
            or transfer_selection != "sequential_physical_tile_columns"
            or isinstance(with_reset_prob, bool)
            or float(with_reset_prob) != 0.0
            or random_selection is not False
        ):
            raise ValueError(
                "Expected TT-v1 gamma=0, fast_lr=1, minibatch units, cyclic "
                "physical-tile column transfer, and no reset/random selection."
            )
        self.fast_lr = 1.0
        self.gamma = 0.0
        self.transfer_every = transfer_every
        self.units_in_mbatch = True
        self.n_reads_per_transfer = n_reads_per_transfer
        self.transfer_lr = float(transfer_lr)
        self.scale_transfer_lr = scale_transfer_lr
        self.transfer_columns = True
        self.transfer_selection = transfer_selection
        self.with_reset_prob = 0.0
        self.random_selection = False
        self.fast_initialization_receipt = _canonical_fast_initialization_receipt(
            fast_initialization_receipt,
            fast_state_hash_receipt=fast_port.state_hash_receipt(),
            strict_zero_target_sha256=tensor_sha256(
                torch.zeros(self.size, dtype=torch.float32, device=self.device)
            ),
        )
        self.engine = _StochasticBitLineEngine(
            generator=generator,
            device=self.device,
            roles=("fast_gradient", "slow_transfer"),
            desired_bl=desired_bl,
            fixed_bl=fixed_bl,
            update_bl_management=update_bl_management,
            update_management=update_management,
            um_grad_scale=um_grad_scale,
        )
        self.tile_offsets = _tile_offsets(self.layout)
        self.tile_cursors = torch.zeros(
            len(self.layout), dtype=torch.int64, device=self.device
        )
        self.tile_read_events = torch.zeros_like(self.tile_cursors)
        self.step_index = 0
        self.examples = 0
        self.cross_entropy_sum = 0.0
        self.correct = 0
        self.transfer_events = 0
        self.fast_positive_counts = torch.zeros(
            self.size, dtype=torch.int64, device=self.device
        )
        self.fast_negative_counts = torch.zeros_like(self.fast_positive_counts)
        self.slow_positive_counts = torch.zeros_like(self.fast_positive_counts)
        self.slow_negative_counts = torch.zeros_like(self.fast_positive_counts)

    def contract(self) -> dict[str, Any]:
        return {
            "algorithm": "tiki_taka_v1",
            "gamma": self.gamma,
            "fast_lr": self.fast_lr,
            "layout": self.layout_receipt,
            "size": self.size,
            "device": str(self.device),
            "learning_rates_q": self.learning_rates_q,
            "q_scales": self.q_scales,
            "slow_nominal_dw_min": self.slow_nominal_dw_min,
            "fast_nominal_dw_min": self.fast_nominal_dw_min,
            "transfer_every": self.transfer_every,
            "units_in_mbatch": self.units_in_mbatch,
            "n_reads_per_transfer": self.n_reads_per_transfer,
            "transfer_lr": self.transfer_lr,
            "scale_transfer_lr": self.scale_transfer_lr,
            "transfer_columns": self.transfer_columns,
            "transfer_selection": self.transfer_selection,
            "with_reset_prob": self.with_reset_prob,
            "random_selection": self.random_selection,
            "fast_initialization": dict(self.fast_initialization_receipt),
            "fast_and_slow_arrays_independent": True,
            "fast_array_visible_to_forward": False,
            "gradient_engine": "manual_label_cross_entropy_backpropagation",
            "cross_entropy_reduction": "mean",
            "optimizer_state": "physical_fast_array_plus_integer_counters_and_rng",
            "weight_state": "physical_persistent_and_apparent_device_state_no_shadow",
            "cumulative_pulse_cap": None,
            "teacher_queries": 0,
            "autograd_calls": 0,
            "adam_moment_values": 0,
            "fp32_shadow_weight_values": 0,
            "fault_mask_exposed": False,
            "fast_reset": False,
            "bit_lines": self.engine.contract(),
        }

    def _transfer_learning_rate(self, layer_index: int) -> float:
        value = self.transfer_lr
        if self.scale_transfer_lr:
            value *= self.learning_rates_q[layer_index]
        return value

    @torch.no_grad()
    def _transfer_due_columns(self) -> tuple[torch.Tensor, torch.Tensor]:
        positive = torch.zeros(self.size, dtype=torch.int64, device=self.device)
        negative = torch.zeros_like(positive)
        for tile_index, tile in enumerate(self.layout):
            tile_positive = torch.zeros(
                tile.shape, dtype=torch.int64, device=self.device
            )
            tile_negative = torch.zeros_like(tile_positive)
            for _ in range(self.n_reads_per_transfer):
                column_index = int(self.tile_cursors[tile_index].item())
                readout = self.__fast_port.read_apparent_column(
                    tile_index=tile_index,
                    column_index=column_index,
                )
                one_hot = torch.zeros(
                    tile.shape[0], dtype=readout.dtype, device=self.device
                )
                one_hot[column_index] = 1.0
                # A stores the negative q-gradient.  The generic pulse engine
                # performs -x*d, so d=-A makes the slow update add A.
                counts = self.engine.generate(
                    one_hot,
                    -readout,
                    learning_rate=self._transfer_learning_rate(tile.layer_index),
                    weight_granularity=self.slow_nominal_dw_min,
                    role="slow_transfer",
                )
                tile_positive += counts.positive
                tile_negative += counts.negative
                self.tile_cursors[tile_index] = (
                    column_index + 1
                ) % tile.shape[0]
                self.tile_read_events[tile_index] += 1
            start = self.tile_offsets[tile_index]
            stop = start + tile.cells
            positive[start:stop] += tile_positive.reshape(-1)
            negative[start:stop] += tile_negative.reshape(-1)
        self.transfer_events += 1
        return positive, negative

    @torch.no_grad()
    def step(self, inputs: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
        factors = manual_label_ce_backprop(self.__slow_port, inputs, labels)
        fast_positive, fast_negative = _batch_tile_outer_product_counts(
            factors,
            layout=self.layout,
            learning_rates=(self.fast_lr, self.fast_lr),
            weight_granularity=self.fast_nominal_dw_min,
            engine=self.engine,
            role="fast_gradient",
        )
        self.__fast_port.apply_grouped_pulse_counts(fast_positive, fast_negative)
        self.fast_positive_counts += fast_positive
        self.fast_negative_counts += fast_negative

        batch_size = inputs.shape[0]
        self.step_index += 1
        self.examples += batch_size
        cross_entropy_sum = float(factors.mean_cross_entropy.item()) * batch_size
        self.cross_entropy_sum += cross_entropy_sum
        self.correct += factors.correct

        transfer_due = self.step_index % self.transfer_every == 0
        slow_positive = torch.zeros_like(fast_positive)
        slow_negative = torch.zeros_like(fast_negative)
        if transfer_due:
            slow_positive, slow_negative = self._transfer_due_columns()
            self.__slow_port.apply_grouped_pulse_counts(
                slow_positive, slow_negative
            )
            self.slow_positive_counts += slow_positive
            self.slow_negative_counts += slow_negative

        fast_total = fast_positive + fast_negative
        slow_total = slow_positive + slow_negative
        return {
            "optimizer_step": self.step_index,
            "examples": batch_size,
            "cross_entropy_sum": cross_entropy_sum,
            "mean_label_cross_entropy": float(factors.mean_cross_entropy.item()),
            "pre_update_correct": factors.correct,
            "fast_commanded_positive_pulses": int(fast_positive.sum().item()),
            "fast_commanded_negative_pulses": int(fast_negative.sum().item()),
            "fast_commanded_pulses": int(fast_total.sum().item()),
            "fast_commanded_cells": int((fast_total > 0).sum().item()),
            "slow_commanded_positive_pulses": int(slow_positive.sum().item()),
            "slow_commanded_negative_pulses": int(slow_negative.sum().item()),
            "slow_commanded_pulses": int(slow_total.sum().item()),
            "slow_commanded_cells": int((slow_total > 0).sum().item()),
            "transfer_due": transfer_due,
            "transfer_event": self.transfer_events if transfer_due else None,
            "tile_cursors": self.tile_cursors.detach().cpu().tolist(),
            "fast_state_hash_receipt": self.__fast_port.state_hash_receipt(),
            "slow_state_hash_receipt": self.__slow_port.state_hash_receipt(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "schema_version": 1,
            "contract": self.contract(),
            "step_index": self.step_index,
            "examples": self.examples,
            "cross_entropy_sum": self.cross_entropy_sum,
            "correct": self.correct,
            "transfer_events": self.transfer_events,
            "tile_cursors": self.tile_cursors.detach().cpu().clone(),
            "tile_read_events": self.tile_read_events.detach().cpu().clone(),
            "fast_positive_counts": self.fast_positive_counts.detach().cpu().clone(),
            "fast_negative_counts": self.fast_negative_counts.detach().cpu().clone(),
            "slow_positive_counts": self.slow_positive_counts.detach().cpu().clone(),
            "slow_negative_counts": self.slow_negative_counts.detach().cpu().clone(),
            "bit_line_engine": self.engine.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema",
            "schema_version",
            "contract",
            "step_index",
            "examples",
            "cross_entropy_sum",
            "correct",
            "transfer_events",
            "tile_cursors",
            "tile_read_events",
            "fast_positive_counts",
            "fast_negative_counts",
            "slow_positive_counts",
            "slow_negative_counts",
            "bit_line_engine",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state["schema"] != self.STATE_SCHEMA
            or state["schema_version"] != 1
            or isinstance(state["schema_version"], bool)
            or state["contract"] != self.contract()
        ):
            raise ValueError("Expected a matching Tiki-Taka-v1 updater state.")
        step_index, examples, loss_sum, correct = _validate_progress(state)
        transfer_events = _strict_nonnegative_int(
            "transfer_events", state["transfer_events"]
        )
        expected_transfer_events = step_index // self.transfer_every
        cursors = state["tile_cursors"]
        reads = state["tile_read_events"]
        if (
            transfer_events != expected_transfer_events
            or not isinstance(cursors, torch.Tensor)
            or cursors.shape != self.tile_cursors.shape
            or cursors.dtype != torch.int64
            or not isinstance(reads, torch.Tensor)
            or reads.shape != self.tile_read_events.shape
            or reads.dtype != torch.int64
            or bool(torch.any(reads < 0))
        ):
            raise ValueError("Expected matching Tiki-Taka transfer progress.")
        expected_reads = transfer_events * self.n_reads_per_transfer
        for tile_index, tile in enumerate(self.layout):
            if (
                int(reads[tile_index].item()) != expected_reads
                or int(cursors[tile_index].item())
                != expected_reads % tile.shape[0]
            ):
                raise ValueError("Expected exact per-physical-tile cyclic cursors.")
        pending_counts = {}
        for name in (
            "fast_positive_counts",
            "fast_negative_counts",
            "slow_positive_counts",
            "slow_negative_counts",
        ):
            pending_counts[name] = _validate_count_tensor(
                state[name], size=self.size, name=name, device=self.device
            )
        fast_role = _engine_role_counters(
            state["bit_line_engine"], "fast_gradient"
        )
        transfer_role = _engine_role_counters(
            state["bit_line_engine"], "slow_transfer"
        )
        if (
            fast_role.get("events") != examples * len(self.layout)
            or transfer_role.get("events")
            != transfer_events * self.n_reads_per_transfer * len(self.layout)
            or fast_role.get("positive_coincidences")
            != int(pending_counts["fast_positive_counts"].sum().item())
            or fast_role.get("negative_coincidences")
            != int(pending_counts["fast_negative_counts"].sum().item())
            or transfer_role.get("positive_coincidences")
            != int(pending_counts["slow_positive_counts"].sum().item())
            or transfer_role.get("negative_coincidences")
            != int(pending_counts["slow_negative_counts"].sum().item())
        ):
            raise ValueError(
                "Expected Tiki-Taka pulse counts to match exact fast-update and "
                "per-tile transfer bit-line events."
            )
        self.engine.load_state_dict(state["bit_line_engine"])
        self.step_index = step_index
        self.examples = examples
        self.cross_entropy_sum = loss_sum
        self.correct = correct
        self.transfer_events = transfer_events
        self.tile_cursors.copy_(cursors.to(self.device))
        self.tile_read_events.copy_(reads.to(self.device))
        for name, value in pending_counts.items():
            getattr(self, name).copy_(value)

    def report(self) -> dict[str, Any]:
        fast_total = self.fast_positive_counts + self.fast_negative_counts
        slow_total = self.slow_positive_counts + self.slow_negative_counts
        tile_progress = []
        for tile_index, tile in enumerate(self.layout):
            reads = int(self.tile_read_events[tile_index].item())
            tile_progress.append(
                {
                    "tile_key": tile.key,
                    "layer_index": tile.layer_index,
                    "input_columns": tile.shape[0],
                    "cursor": int(self.tile_cursors[tile_index].item()),
                    "column_reads": reads,
                    "completed_cursor_passes": reads // tile.shape[0],
                    "cursor_passes": reads / tile.shape[0],
                }
            )
        return {
            "schema": "ebl.ibm_om_tiki_taka_v1_report",
            "schema_version": 1,
            "contract": self.contract(),
            "optimizer_steps": self.step_index,
            "examples": self.examples,
            "mean_label_cross_entropy": (
                self.cross_entropy_sum / self.examples if self.examples else None
            ),
            "pre_update_accuracy": (
                self.correct / self.examples if self.examples else None
            ),
            "transfer_events": self.transfer_events,
            "physical_tile_progress": tile_progress,
            "fast_commanded_positive_pulses": int(
                self.fast_positive_counts.sum().item()
            ),
            "fast_commanded_negative_pulses": int(
                self.fast_negative_counts.sum().item()
            ),
            "fast_commanded_pulses": int(fast_total.sum().item()),
            "fast_commanded_cells": int((fast_total > 0).sum().item()),
            "fast_commanded_pulses_by_layer": _pulse_counts_by_layer(
                fast_total, self.layout
            ),
            "maximum_fast_commanded_pulses_per_cell": int(fast_total.max().item()),
            "slow_commanded_positive_pulses": int(
                self.slow_positive_counts.sum().item()
            ),
            "slow_commanded_negative_pulses": int(
                self.slow_negative_counts.sum().item()
            ),
            "slow_commanded_pulses": int(slow_total.sum().item()),
            "slow_commanded_cells": int((slow_total > 0).sum().item()),
            "slow_commanded_pulses_by_layer": _pulse_counts_by_layer(
                slow_total, self.layout
            ),
            "maximum_slow_commanded_pulses_per_cell": int(slow_total.max().item()),
            "fast_positive_count_sha256": tensor_sha256(self.fast_positive_counts),
            "fast_negative_count_sha256": tensor_sha256(self.fast_negative_counts),
            "slow_positive_count_sha256": tensor_sha256(self.slow_positive_counts),
            "slow_negative_count_sha256": tensor_sha256(self.slow_negative_counts),
            "bit_line_engine": self.engine.report(),
            "fast_state_hash_receipt": self.__fast_port.state_hash_receipt(),
            "slow_state_hash_receipt": self.__slow_port.state_hash_receipt(),
            "checkpoint_requires": [
                "slow_plant_state",
                "fast_plant_state",
                "updater_state",
            ],
        }


__all__ = [
    "FastArrayZeroTarget",
    "IbmOmDirectPulseSgd",
    "IbmOmTikiTakaV1",
    "ManualCeBackpropFactors",
    "StochasticCompressedPulseCounts",
    "build_tiki_taka_fast_zero_target",
    "manual_label_ce_backprop",
    "stochastic_compressed_pulse_counts",
]
