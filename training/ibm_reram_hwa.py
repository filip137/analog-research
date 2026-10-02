"""IBM OM ReRAM modifiers for off-chip HWA and pulse-resolved deployment.

The clean DRN tensors remain the optimizer-owned master weights. A compact
modifier samples one independently programmed endpoint for each minibatch;
the pulse-resolved modifier starts the same fixed identities at their sampled
RESET bounds and runs the declared adaptive program-and-verify controller.
Both executions apply the apparent endpoint to the forward pass and retain the
hidden persistent endpoint as the state from which a later device update must
continue, matching the AIHWKit OM device contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import torch

from experiments.artifacts import sha256_file
from experiments.reram_program_verify.hwa_model import CONDITION_KEY
from model.resistive.builders import ParameterBinding
# Moved to training.ibm_om.constants; facade re-export keeps historical import paths.
from training.ibm_om.constants import (
    _ARRAY_POPULATION_SCHEMA,
    _ARRAY_POPULATION_SCHEMA_VERSION,
    _ARRAY_SAMPLING_RECEIPT_SCHEMA,
    _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION,
    _CORRUPTION_POLICIES,
    _EVALUATION_STREAM_XOR,
    _EXECUTIONS,
    IBM_OM_RAW_ACTIVE_A_MAX,
    IBM_OM_RAW_ACTIVE_A_MIN,
    IBM_OM_RAW_ACTIVE_CONDITIONING_CHANGE_THRESHOLD,
    IBM_OM_RAW_ACTIVE_CONDITIONING_MAXIMUM_PULSES,
    IBM_OM_RAW_ACTIVE_CONDITIONING_QUIET_STEPS,
    IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
    IBM_OM_RAW_ACTIVE_D90,
    IBM_OM_RAW_ACTIVE_QUANTIZED_LEVELS,
    IBM_OM_RAW_ACTIVE_SCALE,
    IBM_OM_RAW_ACTIVE_TOLERANCE,
    IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
    _MAX_EMPTY_COMMON_WINDOW_PAIRS,
    _MAX_EMPTY_COMMON_WINDOW_QUADS,
    _RAW_ACTIVE_MODES,
    _REQUIRED_AIHWKIT_VERSION,
    _RESET_RELATIVE_MODES,
    _STATE_VERSION,
    _STREAMS,
    _TARGET_MAPPINGS,
)
from training.ibm_om.coordinates import LOGICAL, RAW_ACTIVE
# Moved to training.ibm_om.population; facade re-export keeps historical import paths.
from training.ibm_om.population import (
    _ARRAY_POPULATION_FIELDS,
    IbmReramArrayPopulation,
    load_om_array_population,
    _native_seed,
    _npz_scalar,
    _population_fingerprint,
    save_om_array_population,
    _selected_array_population,
    _tensor_sha256,
    _tensor_summary,
)
# Moved to training.ibm_om.plants; facade re-export keeps historical import paths.
from training.ibm_om.plants import (
    _ArrayControllerPort,
    _ArrayPlant,
    IbmReramPulsePlant,
    _RawActiveArrayControllerPort,
    _RawActiveArrayPlant,
)
# Moved to training.ibm_om.sampling; facade re-export keeps historical import paths.
from training.ibm_om.sampling import (
    _artifact,
    _binding_layout,
    sample_om_array_population,
    sample_om_array_population_external,
    _sample_om_array_population_layout,
    sample_om_array_population_layout,
    sample_om_array_population_layout_external,
    _sample_tile_hidden,
)
# Moved to training.ibm_om.topology; facade re-export keeps historical import paths.
from training.ibm_om.topology import (
    _canonical_differential_pair_layout,
    _normalize_dual_rail_layouts,
    _quad_axes,
    _validate_differential_pair_bindings,
)
from training.ibm_reram_endpoint_model import sample_ibm_reram_endpoints
from training.ibm_reram_program_verify import (
    ControllerSettings,
    OM_PRESET,
    PopulationStepEstimator,
    ProgramVerifyResult,
    derive_seed,
    run_program_verify,
)


@dataclass(frozen=True)
class IbmReramHwaConfig:
    """Exact OM cap-128 programming intervention for one modifier."""

    execution: str
    assignment_seed: int
    endpoint_seed: int
    corruption_policy: str
    noisy_evaluation: bool
    endpoint_policy: str = "clip_0_1"
    target_out_of_support: str = "error"
    preset: str = OM_PRESET
    controller: str = "adaptive"
    start_protocol: str = "lower_to_target"
    tolerance_step_ratio: float = 0.5
    maximum_program_pulses: int = 128
    target_mapping: str = "literal_global"
    dual_rail_layout_by_parameter: tuple[tuple[str, str], ...] | None = None
    common_window_margin_fraction: float = 0.0
    reset_relative_mode: str | None = None
    reset_relative_contrast_step: float | None = None
    reset_read_samples: int | None = None
    reset_guard_standard_errors: float | None = None
    raw_active_mode: str | None = None
    raw_active_unsupported_quad_policy: str | None = None
    forward_logit_gain: float | None = None

    def __post_init__(self) -> None:
        if self.execution not in _EXECUTIONS:
            raise ValueError(
                f"Expected execution to be one of {_EXECUTIONS!r}. "
                f"Provided value: {self.execution!r}."
            )
        if self.corruption_policy not in _CORRUPTION_POLICIES:
            raise ValueError(
                "Expected corruption_policy to be 'counterfactual_repaired' "
                f"or 'published'. Provided value: {self.corruption_policy!r}."
            )
        for name in ("assignment_seed", "endpoint_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
                raise ValueError(
                    f"Expected {name} to be an integer in [0, 2**63). "
                    f"Provided value: {value!r}."
                )
        raw_active = self.target_mapping == "raw_active_p90_quad"
        exact = {
            "preset": (self.preset, OM_PRESET),
            "controller": (
                self.controller,
                "one_pulse" if raw_active else "adaptive",
            ),
            "start_protocol": (self.start_protocol, "lower_to_target"),
            "tolerance_step_ratio": (self.tolerance_step_ratio, 0.5),
            "maximum_program_pulses": (self.maximum_program_pulses, 128),
            "endpoint_policy": (
                self.endpoint_policy,
                "preserve" if raw_active else "clip_0_1",
            ),
            "target_out_of_support": (self.target_out_of_support, "error"),
        }
        mismatches = {
            name: {"expected": expected, "provided": provided}
            for name, (provided, expected) in exact.items()
            if provided != expected
        }
        if mismatches:
            raise ValueError(
                "Expected the declared IBM OM lower-from-RESET cap-128 "
                f"protocol. Provided value: {mismatches!r}."
            )
        if not isinstance(self.noisy_evaluation, bool):
            raise ValueError("Expected noisy_evaluation to be a boolean.")
        if self.target_mapping not in _TARGET_MAPPINGS:
            raise ValueError(
                f"Expected target_mapping to be one of {_TARGET_MAPPINGS!r}. "
                f"Provided value: {self.target_mapping!r}."
            )
        layouts = _normalize_dual_rail_layouts(
            self.dual_rail_layout_by_parameter
        )
        object.__setattr__(self, "dual_rail_layout_by_parameter", layouts)
        margin = self.common_window_margin_fraction
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
            or not 0.0 <= float(margin) < 0.5
        ):
            raise ValueError(
                "Expected common_window_margin_fraction to be finite in "
                f"[0, 0.5). Provided value: {margin!r}."
            )
        object.__setattr__(self, "common_window_margin_fraction", float(margin))
        forward_gain = self.forward_logit_gain
        if forward_gain is not None and (
            isinstance(forward_gain, bool)
            or not isinstance(forward_gain, (int, float))
            or not math.isfinite(float(forward_gain))
            or float(forward_gain) <= 0.0
        ):
            raise ValueError(
                "Expected forward_logit_gain to be null or a positive finite "
                f"number. Provided value: {forward_gain!r}."
            )
        object.__setattr__(
            self,
            "forward_logit_gain",
            None if forward_gain is None else float(forward_gain),
        )
        reset_fields = {
            "reset_relative_mode": self.reset_relative_mode,
            "reset_relative_contrast_step": self.reset_relative_contrast_step,
            "reset_read_samples": self.reset_read_samples,
            "reset_guard_standard_errors": self.reset_guard_standard_errors,
        }
        raw_active_fields = {
            "raw_active_mode": self.raw_active_mode,
            "raw_active_unsupported_quad_policy": (
                self.raw_active_unsupported_quad_policy
            ),
        }
        if self.target_mapping in {
            "dual_rail_quad_common_window",
            "shared_reset_relative_quad",
        }:
            if layouts is None:
                raise ValueError(
                    "Expected dual_rail_layout_by_parameter for "
                    f"{self.target_mapping}."
                )
        if self.target_mapping == "shared_reset_relative_quad":
            if float(margin) != 0.0:
                raise ValueError(
                    "Expected shared_reset_relative_quad to use zero "
                    "common_window_margin_fraction; it does not consume a "
                    "characterized common window."
                )
            if self.reset_relative_mode not in _RESET_RELATIVE_MODES:
                raise ValueError(
                    "Expected reset_relative_mode to be 'continuous' or "
                    "'quantized_9_level' for shared_reset_relative_quad. "
                    f"Provided value: {self.reset_relative_mode!r}."
                )
            step = self.reset_relative_contrast_step
            if (
                isinstance(step, bool)
                or not isinstance(step, (int, float))
                or not math.isfinite(float(step))
                or not 0.0 < float(step) <= 0.5
            ):
                raise ValueError(
                    "Expected reset_relative_contrast_step to be finite in "
                    f"(0, 0.5]. Provided value: {step!r}."
                )
            reads = self.reset_read_samples
            if isinstance(reads, bool) or not isinstance(reads, int) or reads < 2:
                raise ValueError(
                    "Expected reset_read_samples to be an integer of at "
                    f"least two. Provided value: {reads!r}."
                )
            guard = self.reset_guard_standard_errors
            if (
                isinstance(guard, bool)
                or not isinstance(guard, (int, float))
                or not math.isfinite(float(guard))
                or float(guard) < 0.0
            ):
                raise ValueError(
                    "Expected reset_guard_standard_errors to be a finite "
                    f"non-negative number. Provided value: {guard!r}."
                )
            object.__setattr__(
                self, "reset_relative_contrast_step", float(step)
            )
            object.__setattr__(
                self, "reset_guard_standard_errors", float(guard)
            )
        elif any(value is not None for value in reset_fields.values()):
            raise ValueError(
                "Expected RESET-relative commissioning fields only for "
                "shared_reset_relative_quad. Provided value: "
                f"{reset_fields!r}."
            )
        if self.target_mapping == "raw_active_p90_quad":
            if layouts is None:
                raise ValueError(
                    "Expected dual_rail_layout_by_parameter for "
                    "raw_active_p90_quad."
                )
            if float(margin) != 0.0:
                raise ValueError(
                    "Expected raw_active_p90_quad to use the frozen D90 "
                    "budget without an additional common-window margin."
                )
            if self.raw_active_mode not in _RAW_ACTIVE_MODES:
                raise ValueError(
                    "Expected raw_active_mode to be 'continuous' or "
                    "'quantized_7_level' for raw_active_p90_quad. "
                    f"Provided value: {self.raw_active_mode!r}."
                )
            if self.raw_active_unsupported_quad_policy != "structural_failure":
                raise ValueError(
                    "Expected raw_active_unsupported_quad_policy to equal "
                    "'structural_failure' until a versioned donor stream is "
                    "declared. Provided value: "
                    f"{self.raw_active_unsupported_quad_policy!r}."
                )
            if self.execution == "compact_endpoint":
                raise ValueError(
                    "Expected raw_active_p90_quad not to use the historical "
                    "q-coordinate compact endpoint model. Use mapped_target "
                    "for minibatch QAT or pulse_resolved for deployment."
                )
        elif any(value is not None for value in raw_active_fields.values()):
            raise ValueError(
                "Expected raw-active fields only for raw_active_p90_quad. "
                f"Provided value: {raw_active_fields!r}."
            )
        if self.execution == "mapped_target" and not raw_active:
            raise ValueError(
                "Expected mapped_target execution only for the raw-active "
                "array-aware training mapper."
            )
        if self.target_mapping == "differential_pair_common_window":
            if layouts is not None:
                raise ValueError(
                    "Expected differential_pair_common_window to have null "
                    "dual_rail_layout_by_parameter; pairing is fixed by the "
                    "canonical adjacent plus/minus binding catalog."
                )
        elif self.target_mapping == "literal_global" and (
            layouts is not None or float(margin) != 0.0
        ):
            raise ValueError(
                "Expected literal_global target mapping to have null "
                "dual_rail_layout_by_parameter and zero "
                "common_window_margin_fraction."
            )


@dataclass(frozen=True)
class IbmReramResetCommissioning:
    """Controller-visible RESET/read estimates used by the shared mapper.

    The target mapper consumes only these observed statistics and the expanded
    per-quad baseline.  Hidden sampled device bounds deliberately do not occur
    in this object.
    """

    population_fingerprint: str
    assignment_seed: int
    commissioning_seed: int
    read_samples: int
    guard_standard_errors: float
    binding_keys: tuple[str, ...]
    binding_shapes: tuple[tuple[int, ...], ...]
    dual_rail_layout_by_parameter: tuple[tuple[str, str], ...]
    reset_mean: torch.Tensor
    reset_standard_error: torch.Tensor
    baseline: torch.Tensor
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        size = sum(math.prod(shape) for shape in self.binding_shapes)
        if (
            not self.population_fingerprint
            or len(self.binding_keys) != len(self.binding_shapes)
            or len(set(self.binding_keys)) != len(self.binding_keys)
            or set(dict(self.dual_rail_layout_by_parameter))
            != set(self.binding_keys)
        ):
            raise ValueError(
                "Expected RESET commissioning to identify one exact binding "
                "layout and physical population."
            )
        for name in ("reset_mean", "reset_standard_error", "baseline"):
            value = getattr(self, name)
            if (
                value.shape != (size,)
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or not bool(torch.all(torch.isfinite(value)))
            ):
                raise ValueError(
                    f"Expected commissioning {name} to be a finite CPU "
                    f"float32 vector of length {size}."
                )
        if bool(torch.any(self.reset_standard_error < 0.0)):
            raise ValueError(
                "Expected non-negative RESET commissioning standard errors."
            )

    @property
    def size(self) -> int:
        return int(self.baseline.numel())

    def tensor_state(self) -> dict[str, torch.Tensor]:
        return {
            "reset_mean": self.reset_mean.detach().cpu().clone(),
            "reset_standard_error": (
                self.reset_standard_error.detach().cpu().clone()
            ),
            "baseline": self.baseline.detach().cpu().clone(),
        }

    def bundle(self) -> dict[str, Any]:
        return {
            "schema": "ebl.ibm_reram.reset_relative_commissioning",
            "schema_version": 1,
            "population_fingerprint": self.population_fingerprint,
            "assignment_seed": self.assignment_seed,
            "commissioning_seed": self.commissioning_seed,
            "read_samples": self.read_samples,
            "guard_standard_errors": self.guard_standard_errors,
            "binding_keys": self.binding_keys,
            "binding_shapes": self.binding_shapes,
            "dual_rail_layout_by_parameter": dict(
                self.dual_rail_layout_by_parameter
            ),
            "observed": self.tensor_state(),
            "report": dict(self.report),
        }


def _round_half_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)


def _raw_active_structural_cell_mask(
    population: IbmReramArrayPopulation,
    layouts: Mapping[str, str] | Sequence[Sequence[str]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact raw-coordinate bounds and the frozen-D90 cell mask."""

    normalized = _normalize_dual_rail_layouts(layouts)
    if normalized is None or set(dict(normalized)) != set(
        population.binding_keys
    ):
        raise ValueError(
            "Expected raw-active layouts to match the population bindings."
        )
    lower, upper = RAW_ACTIVE.cell_bounds(population, device=device)
    corrupt = population.corrupt.to(device=device)
    eligible_cells = torch.zeros(
        population.size, dtype=torch.bool, device=device
    )
    layout_by_key = dict(normalized)
    offset = 0
    for key, shape in zip(
        population.binding_keys, population.binding_shapes
    ):
        count = math.prod(shape)
        parameter_lower = lower[offset : offset + count].reshape(shape)
        parameter_upper = upper[offset : offset + count].reshape(shape)
        parameter_corrupt = corrupt[offset : offset + count].reshape(shape)
        parameter_eligible = eligible_cells[
            offset : offset + count
        ].reshape(shape)
        row_groups, column_groups = _quad_axes(
            shape,
            layout_by_key[key],
            device=device,
        )
        indices = tuple(
            (rows[:, None], columns)
            for rows in row_groups
            for columns in column_groups
        )
        group_lower = torch.stack(
            tuple(parameter_lower[index] for index in indices)
        )
        group_upper = torch.stack(
            tuple(parameter_upper[index] for index in indices)
        )
        capacity = group_upper.amin(dim=0) - group_lower.amax(dim=0)
        group_corrupt = torch.stack(
            tuple(parameter_corrupt[index] for index in indices)
        ).any(dim=0)
        eligible = (
            (capacity >= IBM_OM_RAW_ACTIVE_D90)
            & (group_lower >= 0.0).all(dim=0)
            & ~group_corrupt
        )
        for index in indices:
            parameter_eligible[index] = eligible
        offset += count
    if offset != population.size:  # pragma: no cover - population validates
        raise RuntimeError(
            "Expected raw-active structural mask to cover every cell."
        )
    return eligible_cells, lower, upper


def map_ibm_reram_array_targets(
    global_targets: torch.Tensor,
    population: IbmReramArrayPopulation,
    *,
    target_mapping: str,
    dual_rail_layout_by_parameter: (
        Mapping[str, str] | Sequence[Sequence[str]] | None
    ),
    common_window_margin_fraction: float,
    reset_commissioning: IbmReramResetCommissioning | None = None,
    reset_relative_mode: str | None = None,
    reset_relative_contrast_step: float | None = None,
    raw_active_mode: str | None = None,
    raw_active_unsupported_quad_policy: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Map clean global fractions onto one fixed IBM OM array assignment.

    The quad mapping uses the exact four physical identities assigned to the
    Cartesian product of a logical synapse's two source and two destination
    rails.  Corrupt identities are deliberately retained in the intersection;
    the published arm therefore keeps its collapsed/stuck-cell semantics,
    while the repaired arm naturally uses its sampled donor bounds.
    """

    if (
        not isinstance(global_targets, torch.Tensor)
        or not global_targets.is_floating_point()
        or global_targets.ndim != 1
        or global_targets.numel() != population.size
        or not bool(torch.all(torch.isfinite(global_targets)))
    ):
        raise ValueError(
            "Expected finite floating global_targets with shape "
            f"({population.size},). Provided value: "
            f"shape={getattr(global_targets, 'shape', None)!r}."
        )
    if target_mapping not in _TARGET_MAPPINGS:
        raise ValueError(
            f"Expected target_mapping to be one of {_TARGET_MAPPINGS!r}. "
            f"Provided value: {target_mapping!r}."
        )
    layouts = _normalize_dual_rail_layouts(dual_rail_layout_by_parameter)
    margin = common_window_margin_fraction
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or not 0.0 <= float(margin) < 0.5
    ):
        raise ValueError(
            "Expected common_window_margin_fraction to be finite in "
            f"[0, 0.5). Provided value: {margin!r}."
        )
    margin = float(margin)
    differential_pairs: tuple[
        tuple[int, str, str, tuple[int, int]], ...
    ] = ()
    if target_mapping in {
        "dual_rail_quad_common_window",
        "shared_reset_relative_quad",
        "raw_active_p90_quad",
    }:
        expected_keys = set(population.binding_keys)
        provided_keys = set(dict(layouts or ()))
        if provided_keys != expected_keys:
            raise ValueError(
                "Expected dual_rail_layout_by_parameter keys to equal the "
                "fixed IBM OM population binding keys. Provided value: "
                f"expected={sorted(expected_keys)!r}, "
                f"provided={sorted(provided_keys)!r}."
            )
        outside_global = (global_targets < 0.0) | (global_targets > 1.0)
        if bool(torch.any(outside_global)):
            offending = global_targets[outside_global]
            raise ValueError(
                "Expected four-cell-mapped clean conductance fractions inside "
                "[0, 1]. Provided extrema: "
                f"minimum={float(offending.min().item())}, "
                f"maximum={float(offending.max().item())}."
            )
        if target_mapping == "shared_reset_relative_quad":
            if margin != 0.0:
                raise ValueError(
                    "Expected shared_reset_relative_quad to use zero "
                    "common-window margin."
                )
            if (
                not isinstance(reset_commissioning, IbmReramResetCommissioning)
                or reset_commissioning.population_fingerprint
                != population.fingerprint
                or reset_commissioning.binding_keys != population.binding_keys
                or reset_commissioning.binding_shapes
                != population.binding_shapes
                or reset_commissioning.dual_rail_layout_by_parameter
                != layouts
            ):
                raise ValueError(
                    "Expected shared_reset_relative_quad to receive the "
                    "controller-observed commissioning artifact for this "
                    "exact population and layout."
                )
            if reset_relative_mode not in _RESET_RELATIVE_MODES:
                raise ValueError(
                    "Expected reset_relative_mode to be 'continuous' or "
                    "'quantized_9_level'."
                )
            step = reset_relative_contrast_step
            if (
                isinstance(step, bool)
                or not isinstance(step, (int, float))
                or not math.isfinite(float(step))
                or not 0.0 < float(step) <= 0.5
            ):
                raise ValueError(
                    "Expected a finite RESET-relative contrast step in "
                    f"(0, 0.5]. Provided value: {step!r}."
                )
            reset_relative_contrast_step = float(step)
        elif target_mapping == "raw_active_p90_quad":
            if margin != 0.0:
                raise ValueError(
                    "Expected raw_active_p90_quad to use zero common-window "
                    "margin around the frozen D90 budget."
                )
            if raw_active_mode not in _RAW_ACTIVE_MODES:
                raise ValueError(
                    "Expected raw_active_mode to be 'continuous' or "
                    "'quantized_7_level'."
                )
            if raw_active_unsupported_quad_policy != "structural_failure":
                raise ValueError(
                    "Expected the versioned raw-active unsupported-quad "
                    "policy to equal 'structural_failure'."
                )
            if (
                reset_commissioning is not None
                or reset_relative_mode is not None
                or reset_relative_contrast_step is not None
            ):
                raise ValueError(
                    "Expected raw_active_p90_quad not to consume RESET-relative "
                    "commissioning inputs."
                )
        elif (
            reset_commissioning is not None
            or reset_relative_mode is not None
            or reset_relative_contrast_step is not None
        ):
            raise ValueError(
                "Expected RESET-relative mapper inputs only for "
                "shared_reset_relative_quad."
            )
    elif target_mapping == "differential_pair_common_window":
        if layouts is not None:
            raise ValueError(
                "Expected differential_pair_common_window to have null "
                "layouts; pairing is fixed by the canonical adjacent "
                "plus/minus binding catalog."
            )
        differential_pairs = _canonical_differential_pair_layout(
            population.binding_keys,
            population.binding_shapes,
        )
        outside_global = (global_targets < 0.0) | (global_targets > 1.0)
        if bool(torch.any(outside_global)):
            offending = global_targets[outside_global]
            raise ValueError(
                "Expected differential-pair-mapped clean conductance "
                "fractions inside [0, 1]. Provided extrema: "
                f"minimum={float(offending.min().item())}, "
                f"maximum={float(offending.max().item())}."
            )
    elif layouts is not None or margin != 0.0:
        raise ValueError(
            "Expected literal_global target mapping to have null layouts and "
            "zero common-window margin."
        )

    device = global_targets.device
    dtype = global_targets.dtype
    coordinate = (
        RAW_ACTIVE if target_mapping == "raw_active_p90_quad" else LOGICAL
    )
    lower, upper = coordinate.cell_bounds(population, device=device, dtype=dtype)
    corrupt = population.corrupt.to(device=device)
    published_corrupt = population.published_corrupt.to(device=device)
    noncorrupt = ~corrupt

    def support_counts(
        targets: torch.Tensor,
        selection: torch.Tensor | None = None,
    ) -> dict[str, int]:
        selected = noncorrupt if selection is None else noncorrupt & selection
        return {
            "below_lower_bound": int((selected & (targets < lower)).sum().item()),
            "above_upper_bound": int((selected & (targets > upper)).sum().item()),
            "inside_bounds": int(
                (selected & (targets >= lower) & (targets <= upper)).sum().item()
            ),
        }

    global_support = support_counts(global_targets)
    report: dict[str, Any] = {
        "target_mapping": target_mapping,
        "common_window_margin_fraction": margin,
        "population_fingerprint": population.fingerprint,
        "corruption_policy": population.corruption_policy,
        "devices": population.size,
        "global_target_support": global_support,
        "global_target_outside_0_1": int(
            ((global_targets < 0.0) | (global_targets > 1.0)).sum().item()
        ),
        "parameters": {},
    }
    if target_mapping == "raw_active_p90_quad":
        report["coordinate"] = {
            "version": IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
            "a_min": IBM_OM_RAW_ACTIVE_A_MIN,
            "a_max": IBM_OM_RAW_ACTIVE_A_MAX,
            "scale": IBM_OM_RAW_ACTIVE_SCALE,
            "reference_consumed": False,
        }
    if target_mapping == "literal_global":
        mapped = global_targets.detach().clone()
        report.update(
            {
                "common_window_grouping": None,
                "common_window_group_size": None,
                "common_window_group_count": 0,
                "common_window_empty_group_count": 0,
                "corrupt_group_count": 0,
                "published_corrupt_group_count": 0,
                "differential_pair_binding": None,
                "nonempty_inner_common_span": None,
                "quad_count": 0,
                "common_window_empty_quad_count": 0,
                "corrupt_quad_count": 0,
                "published_corrupt_quad_count": 0,
                "raw_common_span": None,
                "inner_common_span": None,
                "mapped_target_below_lower_bound_nonempty_quad": 0,
                "mapped_target_above_upper_bound_nonempty_quad": 0,
                "mapped_target_below_lower_bound_empty_quad": 0,
                "mapped_target_above_upper_bound_empty_quad": 0,
                "mapped_target_below_lower_bound_nonempty_group": 0,
                "mapped_target_above_upper_bound_nonempty_group": 0,
                "mapped_target_below_lower_bound_empty_group": 0,
                "mapped_target_above_upper_bound_empty_group": 0,
                "mapped_target_support": support_counts(mapped),
            }
        )
        return mapped, report

    if target_mapping == "raw_active_p90_quad":
        assert raw_active_mode is not None
        assert raw_active_unsupported_quad_policy is not None
        layout_by_key = dict(layouts or ())
        mapped = torch.empty_like(global_targets)
        expanded_eligible = torch.zeros(
            population.size, dtype=torch.bool, device=device
        )
        expanded_capacity = torch.empty_like(global_targets)
        expanded_baseline = torch.empty_like(global_targets)
        all_capacity: list[torch.Tensor] = []
        all_baseline: list[torch.Tensor] = []
        all_baseline_low: list[torch.Tensor] = []
        all_baseline_high: list[torch.Tensor] = []
        all_eligible: list[torch.Tensor] = []
        all_exact_support: list[torch.Tensor] = []
        all_corrupt_quad: list[torch.Tensor] = []
        all_passive_compatible: list[torch.Tensor] = []
        all_logical_u: list[torch.Tensor] = []
        all_code: list[torch.Tensor] = []
        all_differential: list[torch.Tensor] = []
        offset = 0
        for key, shape in zip(
            population.binding_keys, population.binding_shapes
        ):
            count = math.prod(shape)
            target = global_targets[offset : offset + count].reshape(shape)
            target_mapped = mapped[offset : offset + count].reshape(shape)
            cell_lower = lower[offset : offset + count].reshape(shape)
            cell_upper = upper[offset : offset + count].reshape(shape)
            cell_corrupt = corrupt[offset : offset + count].reshape(shape)
            cell_eligible = expanded_eligible[
                offset : offset + count
            ].reshape(shape)
            cell_capacity = expanded_capacity[
                offset : offset + count
            ].reshape(shape)
            cell_baseline = expanded_baseline[
                offset : offset + count
            ].reshape(shape)
            row_groups, column_groups = _quad_axes(
                shape,
                layout_by_key[key],
                device=device,
            )
            plus_rows, minus_rows = row_groups
            plus_columns, minus_columns = column_groups
            cell_indices = tuple(
                (rows[:, None], columns)
                for rows in row_groups
                for columns in column_groups
            )
            group_lower = torch.stack(
                tuple(cell_lower[index] for index in cell_indices)
            )
            group_upper = torch.stack(
                tuple(cell_upper[index] for index in cell_indices)
            )
            group_corrupt = torch.stack(
                tuple(cell_corrupt[index] for index in cell_indices)
            ).any(dim=0)
            common_lower = group_lower.amax(dim=0)
            common_upper = group_upper.amin(dim=0)
            capacity = common_upper - common_lower
            passive_compatible = (group_lower >= 0.0).all(dim=0)
            eligible = (
                (capacity >= IBM_OM_RAW_ACTIVE_D90)
                & passive_compatible
                & ~group_corrupt
            )
            baseline_low = common_lower + IBM_OM_RAW_ACTIVE_D90 / 2.0
            baseline_high = common_upper - IBM_OM_RAW_ACTIVE_D90 / 2.0
            # The midpoint is also well-defined for an unsupported quad.  It
            # is retained only to make the structural failure auditable; that
            # quad receives no target-programming pulses.
            baseline = 0.5 * (common_lower + common_upper)

            q_pp = target[plus_rows[:, None], plus_columns]
            q_pm = target[plus_rows[:, None], minus_columns]
            q_mp = target[minus_rows[:, None], plus_columns]
            q_mm = target[minus_rows[:, None], minus_columns]
            logical_u = 0.5 * (q_pp - q_pm - q_mp + q_mm)
            bounded_u = logical_u.clamp(-1.0, 1.0)
            if raw_active_mode == "quantized_7_level":
                code = _round_half_away_from_zero(
                    3.0 * bounded_u
                ).clamp(-3.0, 3.0)
                differential = (
                    code * (IBM_OM_RAW_ACTIVE_D90 / 3.0)
                )
            else:
                code = bounded_u
                differential = bounded_u * IBM_OM_RAW_ACTIVE_D90
            positive = baseline + differential / 2.0
            negative = baseline - differential / 2.0
            target_mapped[plus_rows[:, None], plus_columns] = positive
            target_mapped[plus_rows[:, None], minus_columns] = negative
            target_mapped[minus_rows[:, None], plus_columns] = negative
            target_mapped[minus_rows[:, None], minus_columns] = positive

            for index in cell_indices:
                cell_eligible[index] = eligible
                cell_capacity[index] = capacity
                cell_baseline[index] = baseline
            exact_cell_support = torch.stack(
                tuple(
                    (target_mapped[index] >= cell_lower[index])
                    & (target_mapped[index] <= cell_upper[index])
                    for index in cell_indices
                )
            )
            exact_quad_support = exact_cell_support.all(dim=0)
            parameter_mapped = target_mapped.reshape(-1)
            parameter_noncorrupt = ~cell_corrupt.reshape(-1)
            parameter_below = parameter_noncorrupt & (
                parameter_mapped < cell_lower.reshape(-1)
            )
            parameter_above = parameter_noncorrupt & (
                parameter_mapped > cell_upper.reshape(-1)
            )
            report["parameters"][key] = {
                "dual_rail_layout": layout_by_key[key],
                "devices": count,
                "quad_count": int(capacity.numel()),
                "p90_eligible_quad_count": int(eligible.sum().item()),
                "p90_eligible_quad_fraction": float(
                    eligible.to(torch.float64).mean().item()
                ),
                "structural_failure_quad_count": int(
                    (~eligible).sum().item()
                ),
                "capacity": _tensor_summary(capacity),
                "eligible_baseline": (
                    _tensor_summary(baseline[eligible])
                    if bool(torch.any(eligible))
                    else None
                ),
                "requested_logical_u": _tensor_summary(logical_u),
                "requested_differential": _tensor_summary(differential),
                "exact_target_supported_quad_count": int(
                    exact_quad_support.sum().item()
                ),
                "mapped_target_below_lower_bound": int(
                    parameter_below.sum().item()
                ),
                "mapped_target_above_upper_bound": int(
                    parameter_above.sum().item()
                ),
            }
            all_capacity.append(capacity.reshape(-1))
            all_baseline.append(baseline.reshape(-1))
            all_baseline_low.append(baseline_low.reshape(-1))
            all_baseline_high.append(baseline_high.reshape(-1))
            all_eligible.append(eligible.reshape(-1))
            all_exact_support.append(exact_quad_support.reshape(-1))
            all_corrupt_quad.append(group_corrupt.reshape(-1))
            all_passive_compatible.append(
                passive_compatible.reshape(-1)
            )
            all_logical_u.append(logical_u.reshape(-1))
            all_code.append(code.reshape(-1))
            all_differential.append(differential.reshape(-1))
            offset += count
        if offset != population.size:  # pragma: no cover - validated layout
            raise RuntimeError(
                "Expected raw-active target mapping to cover every cell."
            )
        capacity = torch.cat(all_capacity)
        baseline = torch.cat(all_baseline)
        baseline_low = torch.cat(all_baseline_low)
        baseline_high = torch.cat(all_baseline_high)
        eligible = torch.cat(all_eligible)
        exact_support = torch.cat(all_exact_support)
        corrupt_quad = torch.cat(all_corrupt_quad)
        passive_compatible = torch.cat(all_passive_compatible)
        logical_u = torch.cat(all_logical_u)
        code = torch.cat(all_code)
        differential = torch.cat(all_differential)
        mapped_support = support_counts(mapped)
        code_histogram = None
        if raw_active_mode == "quantized_7_level":
            code_histogram = {
                str(index): int((code == float(index)).sum().item())
                for index in range(-3, 4)
            }
        eligible_cells = expanded_eligible
        report.update(
            {
                "common_window_grouping": "raw_active_p90_quad",
                "common_window_group_size": 4,
                "common_window_group_count": int(capacity.numel()),
                "common_window_empty_group_count": int(
                    (~eligible).sum().item()
                ),
                "quad_count": int(capacity.numel()),
                "candidate_quad_count": int(capacity.numel()),
                "required_p90_quad_count": int(
                    math.ceil(0.9 * capacity.numel())
                ),
                "p90_eligible_quad_count": int(eligible.sum().item()),
                "p90_eligible_quad_fraction": float(
                    eligible.to(torch.float64).mean().item()
                ),
                "structural_failure_quad_count": int(
                    (~eligible).sum().item()
                ),
                "structural_failure_cell_count": int(
                    (~eligible_cells).sum().item()
                ),
                "corrupt_quad_count": int(corrupt_quad.sum().item()),
                "passive_compatible_quad_count": int(
                    passive_compatible.sum().item()
                ),
                "exact_target_supported_quad_count": int(
                    exact_support.sum().item()
                ),
                "raw_active_mode": raw_active_mode,
                "raw_active_unsupported_quad_policy": (
                    raw_active_unsupported_quad_policy
                ),
                "frozen_differential_budget": IBM_OM_RAW_ACTIVE_D90,
                "one_cell_full_scale_displacement": (
                    IBM_OM_RAW_ACTIVE_D90 / 2.0
                ),
                "quantized_level_count": (
                    IBM_OM_RAW_ACTIVE_QUANTIZED_LEVELS
                    if raw_active_mode == "quantized_7_level"
                    else None
                ),
                "quantized_differential_spacing": (
                    IBM_OM_RAW_ACTIVE_D90 / 3.0
                    if raw_active_mode == "quantized_7_level"
                    else None
                ),
                "logical_code_histogram": code_histogram,
                "requested_logical_u": _tensor_summary(logical_u),
                "requested_differential": _tensor_summary(differential),
                "quad_capacity": _tensor_summary(capacity),
                "eligible_quad_baseline": (
                    _tensor_summary(baseline[eligible])
                    if bool(torch.any(eligible))
                    else None
                ),
                "eligible_quad_baseline_low": (
                    _tensor_summary(baseline_low[eligible])
                    if bool(torch.any(eligible))
                    else None
                ),
                "eligible_quad_baseline_high": (
                    _tensor_summary(baseline_high[eligible])
                    if bool(torch.any(eligible))
                    else None
                ),
                "baseline_shared_within_every_quad": True,
                "differential_scale_shared_across_eligible_quads": True,
                "reference_consumed_by_target_mapper": False,
                "mapped_target_below_lower_bound_nonempty_quad": 0,
                "mapped_target_above_upper_bound_nonempty_quad": 0,
                "mapped_target_below_lower_bound_empty_quad": (
                    mapped_support["below_lower_bound"]
                ),
                "mapped_target_above_upper_bound_empty_quad": (
                    mapped_support["above_upper_bound"]
                ),
                "mapped_target_below_lower_bound_nonempty_group": 0,
                "mapped_target_above_upper_bound_nonempty_group": 0,
                "mapped_target_below_lower_bound_empty_group": (
                    mapped_support["below_lower_bound"]
                ),
                "mapped_target_above_upper_bound_empty_group": (
                    mapped_support["above_upper_bound"]
                ),
                "mapped_target_support": mapped_support,
                "mapped_target_outside_0_1": int(
                    ((mapped < 0.0) | (mapped > 1.0)).sum().item()
                ),
                "expanded_quad_capacity_sha256": _tensor_sha256(
                    expanded_capacity
                ),
                "expanded_quad_baseline_sha256": _tensor_sha256(
                    expanded_baseline
                ),
                "expanded_quad_eligibility_sha256": _tensor_sha256(
                    expanded_eligible
                ),
            }
        )
        return mapped, report

    if target_mapping == "shared_reset_relative_quad":
        assert reset_commissioning is not None
        assert reset_relative_mode is not None
        assert reset_relative_contrast_step is not None
        baseline_flat = reset_commissioning.baseline.to(
            device=device, dtype=dtype
        )
        layout_by_key = dict(layouts or ())
        mapped = torch.empty_like(global_targets)
        all_quad_supported: list[torch.Tensor] = []
        all_codes: list[torch.Tensor] = []
        all_nonzero_contrast: list[torch.Tensor] = []
        all_corrupt_quad: list[torch.Tensor] = []
        all_published_corrupt_quad: list[torch.Tensor] = []
        offset = 0
        for key, shape in zip(
            population.binding_keys, population.binding_shapes
        ):
            count = math.prod(shape)
            target = global_targets[offset : offset + count].reshape(shape)
            target_mapped = mapped[offset : offset + count].reshape(shape)
            baseline = baseline_flat[offset : offset + count].reshape(shape)
            cell_lower = lower[offset : offset + count].reshape(shape)
            cell_upper = upper[offset : offset + count].reshape(shape)
            cell_corrupt = corrupt[offset : offset + count].reshape(shape)
            cell_published_corrupt = published_corrupt[
                offset : offset + count
            ].reshape(shape)
            row_groups, column_groups = _quad_axes(
                shape,
                layout_by_key[key],
                device=device,
            )
            plus_rows, minus_rows = row_groups
            plus_columns, minus_columns = column_groups
            q_pp = target[plus_rows[:, None], plus_columns]
            q_pm = target[plus_rows[:, None], minus_columns]
            q_mp = target[minus_rows[:, None], plus_columns]
            q_mm = target[minus_rows[:, None], minus_columns]
            logical_u = 0.5 * (q_pp - q_pm - q_mp + q_mm)
            raw_code = (4.0 * logical_u).clamp(-4.0, 4.0)
            code = (
                _round_half_away_from_zero(raw_code).clamp(-4.0, 4.0)
                if reset_relative_mode == "quantized_9_level"
                else raw_code
            )
            contrast = code * reset_relative_contrast_step
            positive_offset = contrast.clamp_min(0.0) / 2.0
            negative_offset = (-contrast).clamp_min(0.0) / 2.0
            # Every cell receives one of five shared RESET-relative offsets.
            # Only the observed per-quad RESET baseline varies by placement.
            target_mapped[plus_rows[:, None], plus_columns] = (
                baseline[plus_rows[:, None], plus_columns] + positive_offset
            )
            target_mapped[plus_rows[:, None], minus_columns] = (
                baseline[plus_rows[:, None], minus_columns] + negative_offset
            )
            target_mapped[minus_rows[:, None], plus_columns] = (
                baseline[minus_rows[:, None], plus_columns] + negative_offset
            )
            target_mapped[minus_rows[:, None], minus_columns] = (
                baseline[minus_rows[:, None], minus_columns] + positive_offset
            )
            group_inside = torch.stack(
                tuple(
                    (~cell_corrupt[rows[:, None], columns])
                    & (
                        target_mapped[rows[:, None], columns]
                        >= cell_lower[rows[:, None], columns]
                    )
                    & (
                        target_mapped[rows[:, None], columns]
                        <= cell_upper[rows[:, None], columns]
                    )
                    for rows in row_groups
                    for columns in column_groups
                )
            ).all(dim=0)
            group_corrupt = torch.stack(
                tuple(
                    cell_corrupt[rows[:, None], columns]
                    for rows in row_groups
                    for columns in column_groups
                )
            ).any(dim=0)
            group_published_corrupt = torch.stack(
                tuple(
                    cell_published_corrupt[rows[:, None], columns]
                    for rows in row_groups
                    for columns in column_groups
                )
            ).any(dim=0)
            parameter_mapped = target_mapped.reshape(-1)
            parameter_noncorrupt = ~cell_corrupt.reshape(-1)
            parameter_below = parameter_noncorrupt & (
                parameter_mapped < cell_lower.reshape(-1)
            )
            parameter_above = parameter_noncorrupt & (
                parameter_mapped > cell_upper.reshape(-1)
            )
            report["parameters"][key] = {
                "dual_rail_layout": layout_by_key[key],
                "devices": count,
                "quad_count": int(code.numel()),
                "reset_relative_mode": reset_relative_mode,
                "reset_relative_contrast_step": (
                    reset_relative_contrast_step
                ),
                "quad_baseline": _tensor_summary(
                    baseline[plus_rows[:, None], plus_columns]
                ),
                "logical_u": _tensor_summary(logical_u),
                "logical_code": _tensor_summary(code),
                "logical_contrast": _tensor_summary(contrast),
                "fully_supported_quad_count": int(group_inside.sum().item()),
                "fully_supported_quad_fraction": float(
                    group_inside.to(torch.float64).mean().item()
                ),
                "mapped_target_below_lower_bound": int(
                    parameter_below.sum().item()
                ),
                "mapped_target_above_upper_bound": int(
                    parameter_above.sum().item()
                ),
            }
            all_quad_supported.append(group_inside.reshape(-1))
            all_codes.append(code.reshape(-1))
            all_nonzero_contrast.append(contrast.reshape(-1))
            all_corrupt_quad.append(group_corrupt.reshape(-1))
            all_published_corrupt_quad.append(
                group_published_corrupt.reshape(-1)
            )
            offset += count
        if offset != population.size:  # pragma: no cover - validated layout
            raise RuntimeError("Expected RESET-relative mapping to cover all cells.")
        quad_supported = torch.cat(all_quad_supported)
        codes = torch.cat(all_codes)
        contrasts = torch.cat(all_nonzero_contrast)
        corrupt_quad = torch.cat(all_corrupt_quad)
        published_corrupt_quad = torch.cat(all_published_corrupt_quad)
        mapped_support = support_counts(mapped)
        code_histogram = None
        if reset_relative_mode == "quantized_9_level":
            code_histogram = {
                str(index): int((codes == float(index)).sum().item())
                for index in range(-4, 5)
            }
        report.update(
            {
                "common_window_grouping": "shared_reset_relative_quad",
                "common_window_group_size": 4,
                "common_window_group_count": int(quad_supported.numel()),
                "common_window_empty_group_count": 0,
                "corrupt_group_count": int(corrupt_quad.sum().item()),
                "published_corrupt_group_count": int(
                    published_corrupt_quad.sum().item()
                ),
                "differential_pair_binding": None,
                "quad_count": int(quad_supported.numel()),
                "common_window_empty_quad_count": 0,
                "common_window_empty_fraction": 0.0,
                "corrupt_quad_count": int(corrupt_quad.sum().item()),
                "published_corrupt_quad_count": int(
                    published_corrupt_quad.sum().item()
                ),
                "raw_common_span": None,
                "inner_common_span": None,
                "nonempty_inner_common_span": None,
                "reset_relative_mode": reset_relative_mode,
                "reset_relative_contrast_step": reset_relative_contrast_step,
                "reset_relative_signed_levels": (
                    9
                    if reset_relative_mode == "quantized_9_level"
                    else None
                ),
                "reset_relative_cell_offsets": (
                    [
                        0.0,
                        reset_relative_contrast_step / 2.0,
                        reset_relative_contrast_step,
                        1.5 * reset_relative_contrast_step,
                        2.0 * reset_relative_contrast_step,
                    ]
                    if reset_relative_mode == "quantized_9_level"
                    else None
                ),
                "reset_relative_maximum_absolute_contrast": (
                    4.0 * reset_relative_contrast_step
                ),
                "reset_relative_maximum_cell_offset": (
                    2.0 * reset_relative_contrast_step
                ),
                "logical_code": _tensor_summary(codes),
                "logical_code_histogram": code_histogram,
                "logical_contrast": _tensor_summary(contrasts),
                "fully_supported_quad_count": int(
                    quad_supported.sum().item()
                ),
                "fully_supported_quad_fraction": float(
                    quad_supported.to(torch.float64).mean().item()
                ),
                "hidden_support_is_audit_only": True,
                "hidden_device_bounds_consumed_by_target_mapper": False,
                "commissioning": dict(reset_commissioning.report),
                "mapped_target_below_lower_bound_nonempty_quad": (
                    mapped_support["below_lower_bound"]
                ),
                "mapped_target_above_upper_bound_nonempty_quad": (
                    mapped_support["above_upper_bound"]
                ),
                "mapped_target_below_lower_bound_empty_quad": 0,
                "mapped_target_above_upper_bound_empty_quad": 0,
                "mapped_target_below_lower_bound_nonempty_group": (
                    mapped_support["below_lower_bound"]
                ),
                "mapped_target_above_upper_bound_nonempty_group": (
                    mapped_support["above_upper_bound"]
                ),
                "mapped_target_below_lower_bound_empty_group": 0,
                "mapped_target_above_upper_bound_empty_group": 0,
                "mapped_target_support": mapped_support,
                "mapped_target_outside_0_1": int(
                    ((mapped < 0.0) | (mapped > 1.0)).sum().item()
                ),
            }
        )
        return mapped, report

    if target_mapping == "differential_pair_common_window":
        mapped = torch.empty_like(global_targets)
        report["parameter_pairs"] = {}
        provenance_pairs = []
        all_overlap: list[torch.Tensor] = []
        all_corrupt_pair: list[torch.Tensor] = []
        all_published_corrupt_pair: list[torch.Tensor] = []
        all_raw_span: list[torch.Tensor] = []
        all_inner_span: list[torch.Tensor] = []
        all_nonempty_cell: list[torch.Tensor] = []
        offset = 0
        for pair_index, plus_key, minus_key, shape in differential_pairs:
            count = math.prod(shape)
            plus_slice = slice(offset, offset + count)
            minus_slice = slice(offset + count, offset + 2 * count)
            plus_target = global_targets[plus_slice].reshape(shape)
            minus_target = global_targets[minus_slice].reshape(shape)
            plus_lower = lower[plus_slice].reshape(shape)
            minus_lower = lower[minus_slice].reshape(shape)
            plus_upper = upper[plus_slice].reshape(shape)
            minus_upper = upper[minus_slice].reshape(shape)
            plus_corrupt = corrupt[plus_slice].reshape(shape)
            minus_corrupt = corrupt[minus_slice].reshape(shape)
            plus_published_corrupt = published_corrupt[plus_slice].reshape(shape)
            minus_published_corrupt = published_corrupt[minus_slice].reshape(shape)

            # Each logical branch coordinate uses exactly the two identities
            # at that coordinate. Physical support is intersected with the
            # characterized global coordinate before applying the margin.
            common_low = torch.maximum(plus_lower, minus_lower).clamp(0.0, 1.0)
            common_high = torch.minimum(plus_upper, minus_upper).clamp(0.0, 1.0)
            # A point intersection cannot carry a logical fraction.
            overlap = common_high > common_low
            raw_span = (common_high - common_low).clamp_min(0.0)
            inner_span = raw_span * (1.0 - 2.0 * margin)
            inner_low = common_low + margin * raw_span
            baseline = torch.where(
                overlap,
                inner_low,
                (0.5 * (common_low + common_high)).clamp(0.0, 1.0),
            )
            plus_mapped = baseline + plus_target * inner_span
            minus_mapped = baseline + minus_target * inner_span
            mapped[plus_slice] = plus_mapped.reshape(-1)
            mapped[minus_slice] = minus_mapped.reshape(-1)

            group_corrupt = plus_corrupt | minus_corrupt
            group_published_corrupt = (
                plus_published_corrupt | minus_published_corrupt
            )
            nonempty_cell = torch.cat(
                (overlap.reshape(-1), overlap.reshape(-1))
            )
            pair_mapped = torch.cat(
                (plus_mapped.reshape(-1), minus_mapped.reshape(-1))
            )
            pair_lower = torch.cat(
                (plus_lower.reshape(-1), minus_lower.reshape(-1))
            )
            pair_upper = torch.cat(
                (plus_upper.reshape(-1), minus_upper.reshape(-1))
            )
            pair_noncorrupt = torch.cat(
                ((~plus_corrupt).reshape(-1), (~minus_corrupt).reshape(-1))
            )
            pair_below = pair_noncorrupt & (pair_mapped < pair_lower)
            pair_above = pair_noncorrupt & (pair_mapped > pair_upper)
            nonempty_below = int((pair_below & nonempty_cell).sum().item())
            nonempty_above = int((pair_above & nonempty_cell).sum().item())
            empty_below = int((pair_below & ~nonempty_cell).sum().item())
            empty_above = int((pair_above & ~nonempty_cell).sum().item())
            pair_key = f"base.differential_pair.{pair_index}"
            pair_report = {
                "pair_index": pair_index,
                "conductance_plus_key": plus_key,
                "conductance_minus_key": minus_key,
                "shape": list(shape),
                "devices": 2 * count,
                "pair_count": count,
                "common_window_empty_pair_count": int((~overlap).sum().item()),
                "common_window_empty_fraction": float(
                    (~overlap).to(torch.float64).mean().item()
                ),
                "corrupt_pair_count": int(group_corrupt.sum().item()),
                "published_corrupt_pair_count": int(
                    group_published_corrupt.sum().item()
                ),
                "raw_common_span": _tensor_summary(raw_span),
                "inner_common_span": _tensor_summary(inner_span),
                "nonempty_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "nonempty_pair_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "mapped_target_below_lower_bound_nonempty_pair": nonempty_below,
                "mapped_target_above_upper_bound_nonempty_pair": nonempty_above,
                "mapped_target_below_lower_bound_empty_pair": empty_below,
                "mapped_target_above_upper_bound_empty_pair": empty_above,
            }
            report["parameter_pairs"][pair_key] = pair_report
            branch_values = (
                (
                    "conductance_plus",
                    plus_key,
                    minus_key,
                    plus_mapped,
                    plus_lower,
                    plus_upper,
                    plus_corrupt,
                ),
                (
                    "conductance_minus",
                    minus_key,
                    plus_key,
                    minus_mapped,
                    minus_lower,
                    minus_upper,
                    minus_corrupt,
                ),
            )
            for (
                role,
                key,
                peer_key,
                branch_mapped,
                branch_lower,
                branch_upper,
                branch_corrupt,
            ) in branch_values:
                branch_noncorrupt = ~branch_corrupt
                branch_below = branch_noncorrupt & (branch_mapped < branch_lower)
                branch_above = branch_noncorrupt & (branch_mapped > branch_upper)
                report["parameters"][key] = {
                    "differential_role": role,
                    "paired_parameter": peer_key,
                    "pair_index": pair_index,
                    "shape": list(shape),
                    "devices": count,
                    "pair_count": count,
                    "common_window_empty_pair_count": int(
                        (~overlap).sum().item()
                    ),
                    "raw_common_span": _tensor_summary(raw_span),
                    "inner_common_span": _tensor_summary(inner_span),
                    "nonempty_inner_common_span": (
                        _tensor_summary(inner_span[overlap])
                        if bool(torch.any(overlap))
                        else None
                    ),
                    "nonempty_pair_inner_common_span": (
                        _tensor_summary(inner_span[overlap])
                        if bool(torch.any(overlap))
                        else None
                    ),
                    "mapped_target_below_lower_bound_nonempty_pair": int(
                        (branch_below & overlap).sum().item()
                    ),
                    "mapped_target_above_upper_bound_nonempty_pair": int(
                        (branch_above & overlap).sum().item()
                    ),
                    "mapped_target_below_lower_bound_empty_pair": int(
                        (branch_below & ~overlap).sum().item()
                    ),
                    "mapped_target_above_upper_bound_empty_pair": int(
                        (branch_above & ~overlap).sum().item()
                    ),
                }
            provenance_pairs.append(
                {
                    "pair_index": pair_index,
                    "conductance_plus_key": plus_key,
                    "conductance_minus_key": minus_key,
                    "shape": list(shape),
                }
            )
            all_overlap.append(overlap.reshape(-1))
            all_corrupt_pair.append(group_corrupt.reshape(-1))
            all_published_corrupt_pair.append(
                group_published_corrupt.reshape(-1)
            )
            all_raw_span.append(raw_span.reshape(-1))
            all_inner_span.append(inner_span.reshape(-1))
            all_nonempty_cell.extend(
                (overlap.reshape(-1), overlap.reshape(-1))
            )
            offset += 2 * count
        assert offset == population.size

        overlap = torch.cat(all_overlap)
        corrupt_pair = torch.cat(all_corrupt_pair)
        published_corrupt_pair = torch.cat(all_published_corrupt_pair)
        raw_span = torch.cat(all_raw_span)
        inner_span = torch.cat(all_inner_span)
        nonempty_cell = torch.cat(all_nonempty_cell)
        nonempty_support = support_counts(mapped, nonempty_cell)
        empty_support = support_counts(mapped, ~nonempty_cell)
        below_nonempty = nonempty_support["below_lower_bound"]
        above_nonempty = nonempty_support["above_upper_bound"]
        below_empty = empty_support["below_lower_bound"]
        above_empty = empty_support["above_upper_bound"]
        pair_count = int(overlap.numel())
        empty_pair_count = int((~overlap).sum().item())
        corrupt_pair_count = int(corrupt_pair.sum().item())
        published_corrupt_pair_count = int(
            published_corrupt_pair.sum().item()
        )
        report.update(
            {
                "common_window_grouping": "differential_pair",
                "common_window_group_size": 2,
                "common_window_group_count": pair_count,
                "common_window_empty_group_count": empty_pair_count,
                "corrupt_group_count": corrupt_pair_count,
                "published_corrupt_group_count": published_corrupt_pair_count,
                "differential_pair_binding": {
                    "policy": "canonical_adjacent_plus_minus_same_coordinate",
                    "catalog_order": list(population.binding_keys),
                    "pairs": provenance_pairs,
                },
                "pair_count": pair_count,
                "common_window_empty_pair_count": empty_pair_count,
                "common_window_empty_fraction": float(
                    (~overlap).to(torch.float64).mean().item()
                ),
                "corrupt_pair_count": corrupt_pair_count,
                "published_corrupt_pair_count": published_corrupt_pair_count,
                "raw_common_span": _tensor_summary(raw_span),
                "inner_common_span": _tensor_summary(inner_span),
                "nonempty_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "nonempty_pair_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "mapped_target_below_lower_bound_nonempty_group": below_nonempty,
                "mapped_target_above_upper_bound_nonempty_group": above_nonempty,
                "mapped_target_below_lower_bound_empty_group": below_empty,
                "mapped_target_above_upper_bound_empty_group": above_empty,
                "mapped_target_below_lower_bound_nonempty_pair": below_nonempty,
                "mapped_target_above_upper_bound_nonempty_pair": above_nonempty,
                "mapped_target_below_lower_bound_empty_pair": below_empty,
                "mapped_target_above_upper_bound_empty_pair": above_empty,
                "mapped_target_support": support_counts(mapped),
            }
        )
        return mapped, report

    layout_by_key = dict(layouts or ())
    mapped = torch.empty_like(global_targets)
    all_overlap: list[torch.Tensor] = []
    all_corrupt_quad: list[torch.Tensor] = []
    all_published_corrupt_quad: list[torch.Tensor] = []
    all_raw_span: list[torch.Tensor] = []
    all_inner_span: list[torch.Tensor] = []
    all_nonempty_cell: list[torch.Tensor] = []
    offset = 0
    for key, shape in zip(population.binding_keys, population.binding_shapes):
        count = math.prod(shape)
        if len(shape) != 2 or shape[0] % 2 or shape[1] % 2:
            raise ValueError(
                "Expected quad common-window population bindings to be "
                "even-by-even rank-2 tensors. Provided value: "
                f"key={key!r}, shape={shape!r}."
            )
        target = global_targets[offset : offset + count].reshape(shape)
        target_mapped = mapped[offset : offset + count].reshape(shape)
        cell_lower = lower[offset : offset + count].reshape(shape)
        cell_upper = upper[offset : offset + count].reshape(shape)
        cell_corrupt = corrupt[offset : offset + count].reshape(shape)
        cell_published_corrupt = published_corrupt[
            offset : offset + count
        ].reshape(shape)
        row_groups, column_groups = _quad_axes(
            shape,
            layout_by_key[key],
            device=device,
        )

        group_lower = torch.stack(
            tuple(
                cell_lower[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        )
        group_upper = torch.stack(
            tuple(
                cell_upper[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        )
        group_corrupt = torch.stack(
            tuple(
                cell_corrupt[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        ).any(dim=0)
        group_published_corrupt = torch.stack(
            tuple(
                cell_published_corrupt[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        ).any(dim=0)

        # Intersect physical support with the characterized global coordinate.
        common_low = group_lower.amax(dim=0).clamp_min(0.0)
        common_high = group_upper.amin(dim=0).clamp_max(1.0)
        # A point intersection cannot carry a logical fraction and is therefore
        # budgeted as an empty IBM quad, even though that one point is reachable.
        overlap = common_high > common_low
        raw_span = (common_high - common_low).clamp_min(0.0)
        inner_span = raw_span * (1.0 - 2.0 * margin)
        inner_low = common_low + margin * raw_span
        baseline = torch.where(
            overlap,
            inner_low,
            (0.5 * (common_low + common_high)).clamp(0.0, 1.0),
        )
        nonempty_cell = torch.empty(shape, dtype=torch.bool, device=device)
        for rows in row_groups:
            for columns in column_groups:
                target_mapped[rows[:, None], columns] = (
                    baseline
                    + target[rows[:, None], columns] * inner_span
                )
                nonempty_cell[rows[:, None], columns] = overlap

        parameter_noncorrupt = ~cell_corrupt.reshape(-1)
        parameter_mapped = target_mapped.reshape(-1)
        parameter_lower = cell_lower.reshape(-1)
        parameter_upper = cell_upper.reshape(-1)
        parameter_nonempty = nonempty_cell.reshape(-1)
        parameter_below = parameter_noncorrupt & (
            parameter_mapped < parameter_lower
        )
        parameter_above = parameter_noncorrupt & (
            parameter_mapped > parameter_upper
        )

        parameter_report = {
            "dual_rail_layout": layout_by_key[key],
            "devices": count,
            "quad_count": int(overlap.numel()),
            "common_window_empty_quad_count": int((~overlap).sum().item()),
            "common_window_empty_fraction": float(
                (~overlap).to(torch.float64).mean().item()
            ),
            "corrupt_quad_count": int(group_corrupt.sum().item()),
            "published_corrupt_quad_count": int(
                group_published_corrupt.sum().item()
            ),
            "raw_common_span": _tensor_summary(raw_span),
            "inner_common_span": _tensor_summary(inner_span),
            "mapped_target_below_lower_bound_nonempty_quad": int(
                (parameter_below & parameter_nonempty).sum().item()
            ),
            "mapped_target_above_upper_bound_nonempty_quad": int(
                (parameter_above & parameter_nonempty).sum().item()
            ),
            "mapped_target_below_lower_bound_empty_quad": int(
                (parameter_below & ~parameter_nonempty).sum().item()
            ),
            "mapped_target_above_upper_bound_empty_quad": int(
                (parameter_above & ~parameter_nonempty).sum().item()
            ),
        }
        report["parameters"][key] = parameter_report
        all_overlap.append(overlap.reshape(-1))
        all_corrupt_quad.append(group_corrupt.reshape(-1))
        all_published_corrupt_quad.append(group_published_corrupt.reshape(-1))
        all_raw_span.append(raw_span.reshape(-1))
        all_inner_span.append(inner_span.reshape(-1))
        all_nonempty_cell.append(nonempty_cell.reshape(-1))
        offset += count
    assert offset == population.size

    overlap = torch.cat(all_overlap)
    corrupt_quad = torch.cat(all_corrupt_quad)
    published_corrupt_quad = torch.cat(all_published_corrupt_quad)
    raw_span = torch.cat(all_raw_span)
    inner_span = torch.cat(all_inner_span)
    nonempty_cell = torch.cat(all_nonempty_cell)
    nonempty_support = support_counts(mapped, nonempty_cell)
    empty_support = support_counts(mapped, ~nonempty_cell)
    report.update(
        {
            "common_window_grouping": "dual_rail_quad",
            "common_window_group_size": 4,
            "common_window_group_count": int(overlap.numel()),
            "common_window_empty_group_count": int((~overlap).sum().item()),
            "corrupt_group_count": int(corrupt_quad.sum().item()),
            "published_corrupt_group_count": int(
                published_corrupt_quad.sum().item()
            ),
            "differential_pair_binding": None,
            "quad_count": int(overlap.numel()),
            "common_window_empty_quad_count": int((~overlap).sum().item()),
            "common_window_empty_fraction": float(
                (~overlap).to(torch.float64).mean().item()
            ),
            "corrupt_quad_count": int(corrupt_quad.sum().item()),
            "published_corrupt_quad_count": int(
                published_corrupt_quad.sum().item()
            ),
            "raw_common_span": _tensor_summary(raw_span),
            "inner_common_span": _tensor_summary(inner_span),
            "nonempty_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "nonempty_quad_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "mapped_target_below_lower_bound_nonempty_quad": nonempty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_nonempty_quad": nonempty_support[
                "above_upper_bound"
            ],
            "mapped_target_below_lower_bound_empty_quad": empty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_empty_quad": empty_support[
                "above_upper_bound"
            ],
            "mapped_target_below_lower_bound_nonempty_group": nonempty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_nonempty_group": nonempty_support[
                "above_upper_bound"
            ],
            "mapped_target_below_lower_bound_empty_group": empty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_empty_group": empty_support[
                "above_upper_bound"
            ],
            "mapped_target_support": support_counts(mapped),
        }
    )
    return mapped, report


def validate_ibm_reram_target_mapping_preflight(
    report: Mapping[str, Any],
    *,
    maximum_empty_quads: int = _MAX_EMPTY_COMMON_WINDOW_QUADS,
    maximum_empty_pairs: int = _MAX_EMPTY_COMMON_WINDOW_PAIRS,
) -> None:
    """Fail closed on unsupported nonempty common-window groups."""

    target_mapping = report.get("target_mapping")
    if target_mapping not in {
        "dual_rail_quad_common_window",
        "differential_pair_common_window",
        "shared_reset_relative_quad",
        "raw_active_p90_quad",
    }:
        return
    if target_mapping == "raw_active_p90_quad":
        devices = report.get("devices")
        quad_count = report.get("quad_count")
        required = report.get("required_p90_quad_count")
        eligible = report.get("p90_eligible_quad_count")
        structural = report.get("structural_failure_quad_count")
        mapped_support = report.get("mapped_target_support")
        coordinate = report.get("coordinate")
        if (
            not isinstance(devices, int)
            or devices < 1
            or not isinstance(quad_count, int)
            or quad_count < 1
            or devices != 4 * quad_count
            or required != math.ceil(0.9 * quad_count)
            or not isinstance(eligible, int)
            or not 0 <= eligible <= quad_count
            or structural != quad_count - eligible
            or report.get("structural_failure_cell_count")
            != 4 * structural
            or report.get("common_window_grouping")
            != "raw_active_p90_quad"
            or report.get("common_window_group_size") != 4
            or report.get("common_window_group_count") != quad_count
            or report.get("frozen_differential_budget")
            != IBM_OM_RAW_ACTIVE_D90
            or report.get("baseline_shared_within_every_quad") is not True
            or report.get(
                "differential_scale_shared_across_eligible_quads"
            )
            is not True
            or report.get("reference_consumed_by_target_mapper") is not False
            or report.get("raw_active_unsupported_quad_policy")
            != "structural_failure"
            or report.get("raw_active_mode") not in _RAW_ACTIVE_MODES
            or not isinstance(mapped_support, Mapping)
            or set(mapped_support)
            != {"below_lower_bound", "above_upper_bound", "inside_bounds"}
            or sum(mapped_support.values()) != devices
            or not isinstance(coordinate, Mapping)
            or coordinate.get("version")
            != IBM_OM_RAW_ACTIVE_COORDINATE_VERSION
            or coordinate.get("reference_consumed") is not False
        ):
            raise ValueError(
                "Expected a strict raw-active p90 quad target-mapping "
                "preflight report."
            )
        if report.get("global_target_outside_0_1") != 0:
            raise ValueError(
                "Expected raw-active source targets inside [0, 1]."
            )
        if (
            report.get("mapped_target_below_lower_bound_nonempty_quad") != 0
            or report.get("mapped_target_above_upper_bound_nonempty_quad")
            != 0
        ):
            raise ValueError(
                "Expected every p90-eligible raw-active target to remain "
                "inside its exact per-cell support."
            )
        return
    if target_mapping == "shared_reset_relative_quad":
        devices = report.get("devices")
        quad_count = report.get("quad_count")
        supported = report.get("fully_supported_quad_count")
        supported_fraction = report.get("fully_supported_quad_fraction")
        mapped_support = report.get("mapped_target_support")
        commissioning = report.get("commissioning")
        if (
            not isinstance(devices, int)
            or devices < 1
            or not isinstance(quad_count, int)
            or quad_count < 1
            or devices != 4 * quad_count
            or not isinstance(supported, int)
            or not 0 <= supported <= quad_count
            or not isinstance(supported_fraction, (int, float))
            or not math.isfinite(float(supported_fraction))
            or not math.isclose(
                float(supported_fraction),
                supported / quad_count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not isinstance(mapped_support, Mapping)
            or set(mapped_support)
            != {"below_lower_bound", "above_upper_bound", "inside_bounds"}
            or sum(mapped_support.values()) != devices
            or not isinstance(commissioning, Mapping)
            or commissioning.get("controller_observations_only") is not True
            or commissioning.get(
                "hidden_device_bounds_consumed_by_target_mapper"
            )
            is not False
            or report.get("hidden_support_is_audit_only") is not True
            or report.get("hidden_device_bounds_consumed_by_target_mapper")
            is not False
            or report.get("common_window_grouping")
            != "shared_reset_relative_quad"
            or report.get("common_window_group_size") != 4
            or report.get("common_window_group_count") != quad_count
            or report.get("common_window_empty_group_count") != 0
            or report.get("common_window_empty_quad_count") != 0
        ):
            raise ValueError(
                "Expected a strict shared RESET-relative target-mapping "
                "preflight report."
            )
        if float(supported_fraction) < 0.95:
            raise ValueError(
                "Expected at least 95 percent of RESET-relative quads to be "
                "fully inside their hidden per-cell support in the audit. "
                f"Provided value: {float(supported_fraction):.9f}."
            )
        if report.get("global_target_outside_0_1") != 0:
            raise ValueError(
                "Expected RESET-relative source targets inside [0, 1]."
            )
        if report.get("mapped_target_outside_0_1") != 0:
            raise ValueError(
                "Expected RESET-relative commissioned targets inside the "
                "shared characterized coordinate [0, 1] without clipping. "
                f"Provided outside count: "
                f"{report.get('mapped_target_outside_0_1')!r}."
            )
        return
    if target_mapping == "dual_rail_quad_common_window":
        if (
            isinstance(maximum_empty_quads, bool)
            or not isinstance(maximum_empty_quads, int)
            or maximum_empty_quads < 0
        ):
            raise ValueError(
                "Expected maximum_empty_quads to be a non-negative integer."
            )
        below = report.get("mapped_target_below_lower_bound_nonempty_quad")
        above = report.get("mapped_target_above_upper_bound_nonempty_quad")
        empty = report.get("common_window_empty_quad_count")
        if not all(
            isinstance(value, int) and value >= 0
            for value in (below, above, empty)
        ):
            raise ValueError(
                "Expected a strict IBM OM quad target-mapping preflight report."
            )
        if below or above:
            raise ValueError(
                "Expected zero mapped targets outside physical support on "
                "nonempty quad windows. Provided value: "
                f"below={below}, above={above}."
            )
        if empty > maximum_empty_quads:
            raise ValueError(
                "Expected no more than "
                f"{maximum_empty_quads} empty quad common windows. Provided "
                f"value: {empty}."
            )
        return

    if (
        isinstance(maximum_empty_pairs, bool)
        or not isinstance(maximum_empty_pairs, int)
        or maximum_empty_pairs < 0
    ):
        raise ValueError(
            "Expected maximum_empty_pairs to be a non-negative integer."
        )
    below = report.get("mapped_target_below_lower_bound_nonempty_pair")
    above = report.get("mapped_target_above_upper_bound_nonempty_pair")
    empty = report.get("common_window_empty_pair_count")
    pair_count = report.get("pair_count")
    if not all(
        isinstance(value, int) and value >= 0
        for value in (below, above, empty, pair_count)
    ):
        raise ValueError(
            "Expected a strict IBM OM differential-pair target-mapping "
            "preflight report."
        )
    expected_generic = {
        "common_window_grouping": "differential_pair",
        "common_window_group_size": 2,
        "common_window_group_count": pair_count,
        "common_window_empty_group_count": empty,
        "mapped_target_below_lower_bound_nonempty_group": below,
        "mapped_target_above_upper_bound_nonempty_group": above,
        "mapped_target_below_lower_bound_empty_group": report.get(
            "mapped_target_below_lower_bound_empty_pair"
        ),
        "mapped_target_above_upper_bound_empty_group": report.get(
            "mapped_target_above_upper_bound_empty_pair"
        ),
    }
    if any(report.get(key) != value for key, value in expected_generic.items()):
        raise ValueError(
            "Expected differential-pair-specific and generic common-window "
            "preflight fields to agree exactly."
        )
    if empty > pair_count:
        raise ValueError(
            "Expected common_window_empty_pair_count not to exceed pair_count."
        )
    if below or above:
        raise ValueError(
            "Expected zero mapped targets outside physical support on "
            "nonempty differential-pair windows. Provided value: "
            f"below={below}, above={above}."
        )
    if empty > maximum_empty_pairs:
        raise ValueError(
            "Expected no more than "
            f"{maximum_empty_pairs} empty differential-pair common windows. "
            f"Provided value: {empty}."
        )


def commission_ibm_reram_reset_relative_baselines(
    population: IbmReramArrayPopulation,
    *,
    dual_rail_layout_by_parameter: (
        Mapping[str, str] | Sequence[Sequence[str]]
    ),
    read_samples: int,
    guard_standard_errors: float,
) -> IbmReramResetCommissioning:
    """Commission shared targets using only repeated RESET/read observations.

    Each sample applies one RESET pulse and then reads the apparent state.  A
    quad baseline is the maximum over its four cell estimates after adding the
    declared standard-error guard.  The returned artifact intentionally has no
    per-cell minimum or maximum conductance field.
    """

    layouts = _normalize_dual_rail_layouts(dual_rail_layout_by_parameter)
    if layouts is None or set(dict(layouts)) != set(population.binding_keys):
        raise ValueError(
            "Expected RESET-relative commissioning layouts to match the "
            "fixed population bindings exactly."
        )
    if isinstance(read_samples, bool) or not isinstance(read_samples, int) or read_samples < 2:
        raise ValueError(
            "Expected RESET-relative commissioning to use at least two "
            f"RESET/read samples. Provided value: {read_samples!r}."
        )
    if (
        isinstance(guard_standard_errors, bool)
        or not isinstance(guard_standard_errors, (int, float))
        or not math.isfinite(float(guard_standard_errors))
        or float(guard_standard_errors) < 0.0
    ):
        raise ValueError(
            "Expected a finite non-negative RESET standard-error guard. "
            f"Provided value: {guard_standard_errors!r}."
        )
    guard = float(guard_standard_errors)
    seed = derive_seed(
        population.assignment_seed,
        "shared_reset_relative_commissioning",
        population.fingerprint,
        read_samples,
        format(guard, ".17g"),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    plant = _ArrayPlant(population, generator=generator, device=torch.device("cpu"))
    port = plant.controller_port()
    reset_direction = -torch.ones(population.size, dtype=torch.int8)
    reset_count = torch.ones(population.size, dtype=torch.int64)
    observed = []
    for _sample_index in range(read_samples):
        port.apply_identical_pulses(reset_direction, reset_count)
        value = port.verify()
        if not bool(torch.all(torch.isfinite(value))):
            raise RuntimeError(
                "Expected finite apparent RESET reads during commissioning."
            )
        observed.append(value)
    reads = torch.stack(observed).to(dtype=torch.float32, device="cpu")
    reset_mean = reads.mean(dim=0)
    reset_standard_error = reads.std(dim=0, unbiased=True) / math.sqrt(
        read_samples
    )
    guarded_reset = reset_mean + guard * reset_standard_error
    baseline = torch.empty_like(reset_mean)
    layout_by_key = dict(layouts)
    parameter_reports: dict[str, Any] = {}
    offset = 0
    quad_count = 0
    baseline_values = []
    for key, shape in zip(population.binding_keys, population.binding_shapes):
        count = math.prod(shape)
        guarded = guarded_reset[offset : offset + count].reshape(shape)
        mapped_baseline = baseline[offset : offset + count].reshape(shape)
        row_groups, column_groups = _quad_axes(
            shape,
            layout_by_key[key],
            device=torch.device("cpu"),
        )
        group_guarded = torch.stack(
            tuple(
                guarded[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        )
        raw_group_baseline = group_guarded.amax(dim=0)
        # The DRN's characterized normalized conductance coordinate is public
        # and shared by every cell.  RESET observations below that global
        # coordinate cannot be represented by the network, so commissioning
        # raises only those baselines to zero.  It never clips against a
        # hidden per-cell lower or upper bound.
        group_baseline = raw_group_baseline.clamp_min(0.0)
        for rows in row_groups:
            for columns in column_groups:
                mapped_baseline[rows[:, None], columns] = group_baseline
        parameter_reports[key] = {
            "dual_rail_layout": layout_by_key[key],
            "devices": count,
            "quad_count": int(group_baseline.numel()),
            "reset_mean": _tensor_summary(
                reset_mean[offset : offset + count]
            ),
            "reset_standard_error": _tensor_summary(
                reset_standard_error[offset : offset + count]
            ),
            "quad_baseline": _tensor_summary(group_baseline),
            "raw_quad_baseline": _tensor_summary(raw_group_baseline),
        }
        quad_count += int(group_baseline.numel())
        baseline_values.append(group_baseline.reshape(-1))
        offset += count
    if offset != population.size:  # pragma: no cover - population validates it
        raise RuntimeError("Expected commissioning to cover every population cell.")
    compact_baselines = torch.cat(baseline_values)
    report = {
        "schema": "ebl.ibm_reram.reset_relative_commissioning_receipt",
        "schema_version": 1,
        "algorithm": "reset_pulse_verify_mean_quad_max_guarded_se",
        "controller_observations_only": True,
        "hidden_device_bounds_consumed_by_target_mapper": False,
        "population_fingerprint": population.fingerprint,
        "assignment_seed": population.assignment_seed,
        "commissioning_seed": seed,
        "read_samples": read_samples,
        "guard_standard_errors": guard,
        "devices": population.size,
        "quad_count": quad_count,
        "reset_pulses_per_device": read_samples,
        "total_reset_pulses": population.size * read_samples,
        "reset_mean": _tensor_summary(reset_mean),
        "reset_standard_error": _tensor_summary(reset_standard_error),
        "quad_baseline": _tensor_summary(compact_baselines),
        "reset_mean_sha256": _tensor_sha256(reset_mean),
        "reset_standard_error_sha256": _tensor_sha256(
            reset_standard_error
        ),
        "expanded_baseline_sha256": _tensor_sha256(baseline),
        "parameters": parameter_reports,
    }
    return IbmReramResetCommissioning(
        population_fingerprint=population.fingerprint,
        assignment_seed=population.assignment_seed,
        commissioning_seed=seed,
        read_samples=read_samples,
        guard_standard_errors=guard,
        binding_keys=population.binding_keys,
        binding_shapes=population.binding_shapes,
        dual_rail_layout_by_parameter=layouts,
        reset_mean=reset_mean,
        reset_standard_error=reset_standard_error,
        baseline=baseline,
        report=report,
    )


def _result_mapping(result: ProgramVerifyResult) -> dict[str, Any]:
    count = int(result.accepted.numel())
    return {
        "devices": count,
        "accepted": int(result.accepted.sum().item()),
        "success_fraction": float(result.accepted.float().mean().item()),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "nonfinite": int(result.nonfinite.sum().item()),
        "pulse_count": {
            "mean": float(result.total_pulses.float().mean().item()),
            "median": float(result.total_pulses.float().median().item()),
            "maximum": int(result.total_pulses.max().item()),
            "set_mean": float(result.set_count.float().mean().item()),
            "reset_mean": float(result.reset_count.float().mean().item()),
        },
        "verify_count_mean": float(result.verify_count.float().mean().item()),
        "reversal_count_mean": float(result.reversals.float().mean().item()),
    }


class IbmReramHwaParameterModifier:
    """Temporary IBM OM endpoints backed by one fixed physical assignment."""

    def __init__(
        self,
        bindings: Sequence[ParameterBinding],
        config: IbmReramHwaConfig,
        *,
        device_model_path: Path,
        conductance_min: float,
        conductance_max: float,
        population_path: Path | None = None,
        population_receipt_path: Path | None = None,
    ) -> None:
        self._bindings = tuple(bindings)
        if not self._bindings:
            raise ValueError("Expected IBM OM HWA to receive named dense bindings.")
        if not math.isfinite(conductance_min) or not math.isfinite(
            conductance_max
        ) or not conductance_min < conductance_max:
            raise ValueError("Expected finite increasing DRN conductance bounds.")
        self._config = config
        self._conductance_min = float(conductance_min)
        self._conductance_max = float(conductance_max)
        if config.target_mapping == "raw_active_p90_quad" and (
            self._conductance_min != 0.0 or self._conductance_max != 1.0
        ):
            raise ValueError(
                "Expected raw_active_p90_quad to use the shared g coordinate "
                "directly as model conductance bounds [0, 1], without a "
                "second affine rescaling."
            )
        differential_binding_pairs = ()
        if config.target_mapping == "differential_pair_common_window":
            differential_binding_pairs = _validate_differential_pair_bindings(
                self._bindings
            )
        self._artifact, self._artifact_sha256 = _artifact(device_model_path)
        if (population_path is None) != (population_receipt_path is None):
            raise ValueError(
                "Expected IBM OM population and receipt paths together or neither."
            )
        self._population_artifact_paths: tuple[Path, Path] | None = None
        self._population_sampling_receipt: dict[str, Any] | None = None
        if population_path is None:
            self._population = sample_om_array_population(
                self._bindings,
                assignment_seed=config.assignment_seed,
                corruption_policy=config.corruption_policy,
            )
        else:
            raw_sampler = os.environ.get("EBL_AIHWKIT_PYTHON")
            if not raw_sampler:
                raise RuntimeError(
                    "Expected IBM OM DRN execution to set EBL_AIHWKIT_PYTHON "
                    "to the pinned AIHWKit 1.1.0 interpreter."
                )
            assert population_receipt_path is not None
            self._population, self._population_sampling_receipt = (
                sample_om_array_population_external(
                    self._bindings,
                    assignment_seed=config.assignment_seed,
                    corruption_policy=config.corruption_policy,
                    aihwkit_python=Path(raw_sampler),
                    population_path=population_path,
                    receipt_path=population_receipt_path,
                )
            )
            self._population_artifact_paths = (
                population_path.expanduser().resolve(),
                population_receipt_path.expanduser().resolve(),
            )
        if config.target_mapping == "differential_pair_common_window":
            population_pairs = _canonical_differential_pair_layout(
                self._population.binding_keys,
                self._population.binding_shapes,
            )
            if population_pairs != differential_binding_pairs:
                raise ValueError(
                    "Expected the fixed IBM OM population to preserve the "
                    "canonical differential-pair binding keys and shapes. "
                    f"Provided population={population_pairs!r}, "
                    f"bindings={differential_binding_pairs!r}."
                )
        self._population_cache: dict[str, IbmReramArrayPopulation] = {
            "cpu": self._population
        }
        model_step = float(self._artifact["programming"]["nominal_dw_min"])
        if not math.isclose(
            model_step,
            self._population.nominal_dw_min,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Expected device-model nominal OM step to match AIHWKit identity "
                f"sampling. Provided value: model={model_step}, "
                f"sampled={self._population.nominal_dw_min}."
            )
        if config.target_mapping in {
            "dual_rail_quad_common_window",
            "shared_reset_relative_quad",
            "raw_active_p90_quad",
        }:
            expected_layout_keys = set(self._population.binding_keys)
            provided_layout_keys = set(
                dict(config.dual_rail_layout_by_parameter or ())
            )
            if provided_layout_keys != expected_layout_keys:
                raise ValueError(
                    "Expected dual_rail_layout_by_parameter keys to equal the "
                    "fixed IBM OM population binding keys. Provided value: "
                    f"expected={sorted(expected_layout_keys)!r}, "
                    f"provided={sorted(provided_layout_keys)!r}."
                )
            invalid_shapes = {
                key: shape
                for key, shape in zip(
                    self._population.binding_keys,
                    self._population.binding_shapes,
                )
                if len(shape) != 2 or shape[0] % 2 or shape[1] % 2
            }
            if invalid_shapes:
                raise ValueError(
                    "Expected four-cell dual-rail bindings to be even-by-even "
                    f"rank-2 tensors. Provided value: {invalid_shapes!r}."
                )
        self._reset_commissioning: IbmReramResetCommissioning | None = None
        self._reset_commissioning_artifact_paths: tuple[Path, Path] | None = None
        if config.target_mapping == "shared_reset_relative_quad":
            assert config.reset_read_samples is not None
            assert config.reset_guard_standard_errors is not None
            self._reset_commissioning = (
                commission_ibm_reram_reset_relative_baselines(
                    self._population,
                    dual_rail_layout_by_parameter=(
                        config.dual_rail_layout_by_parameter or ()
                    ),
                    read_samples=config.reset_read_samples,
                    guard_standard_errors=(
                        config.reset_guard_standard_errors
                    ),
                )
            )
        self._generators: dict[str, dict[str, torch.Generator]] = {
            stream: {} for stream in _STREAMS
        }
        self._pending_generator_states: dict[str, dict[str, torch.Tensor]] = {
            stream: {} for stream in _STREAMS
        }
        self._active_stream: str | None = None
        self._last_report: dict[str, Any] | None = None
        self._last_deployment: dict[str, Any] | None = None

    def _population_on(self, device: torch.device) -> IbmReramArrayPopulation:
        key = str(device)
        population = self._population_cache.get(key)
        if population is None:
            population = self._population.to(device)
            self._population_cache[key] = population
        return population

    @property
    def config(self) -> IbmReramHwaConfig:
        return self._config

    @property
    def population_fingerprint(self) -> str:
        return self._population.fingerprint

    @property
    def device_model_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def population_artifact_paths(self) -> tuple[Path, Path] | None:
        return self._population_artifact_paths

    @property
    def population_sampling_receipt(self) -> dict[str, Any] | None:
        if self._population_sampling_receipt is None:
            return None
        return json.loads(json.dumps(self._population_sampling_receipt))

    @property
    def reset_commissioning_bundle(self) -> dict[str, Any] | None:
        if self._reset_commissioning is None:
            return None
        return self._reset_commissioning.bundle()

    @property
    def reset_commissioning_report(self) -> dict[str, Any] | None:
        if self._reset_commissioning is None:
            return None
        return dict(self._reset_commissioning.report)

    @property
    def reset_commissioning_artifact_paths(self) -> tuple[Path, Path] | None:
        return self._reset_commissioning_artifact_paths

    def register_reset_commissioning_artifacts(
        self,
        bundle_path: Path,
        receipt_path: Path,
    ) -> None:
        if self._reset_commissioning is None:
            raise RuntimeError(
                "Expected RESET commissioning before registering its artifacts."
            )
        resolved = (
            bundle_path.expanduser().resolve(),
            receipt_path.expanduser().resolve(),
        )
        if not all(path.is_file() for path in resolved):
            raise RuntimeError(
                "Expected durable RESET commissioning bundle and receipt files."
            )
        self._reset_commissioning_artifact_paths = resolved

    @property
    def programming_report(self) -> dict[str, Any] | None:
        return None if self._last_report is None else dict(self._last_report)

    @property
    def last_deployment_bundle(self) -> dict[str, Any] | None:
        return self._last_deployment

    def _generator(self, stream: str, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._generators[stream].get(key)
        if generator is not None:
            return generator
        generator = torch.Generator(device=device)
        seed = self._config.endpoint_seed
        if stream == "evaluation":
            seed ^= _EVALUATION_STREAM_XOR
        generator.manual_seed(seed)
        pending = self._pending_generator_states[stream].pop(key, None)
        if pending is not None:
            generator.set_state(pending)
        self._generators[stream][key] = generator
        return generator

    def _global_targets(
        self,
    ) -> tuple[torch.Tensor, list[tuple[int, tuple[int, ...]]]]:
        flat = []
        layout = []
        offset = 0
        span = self._conductance_max - self._conductance_min
        for binding in self._bindings:
            state = binding.state
            target = (state.detach() - self._conductance_min) / span
            if not bool(torch.all(torch.isfinite(target))):
                raise ValueError(
                    f"Expected finite clean target for {binding.key!r}."
                )
            flat.append(target.reshape(-1))
            layout.append((offset, tuple(state.shape)))
            offset += target.numel()
        return torch.cat(flat), layout

    def preflight_target_mapping(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return authoritative mapped targets and a read-only mapping report."""

        global_targets, _layout = self._global_targets()
        population = self._population_on(global_targets.device)
        mapped, report = map_ibm_reram_array_targets(
            global_targets,
            population,
            target_mapping=self._config.target_mapping,
            dual_rail_layout_by_parameter=(
                self._config.dual_rail_layout_by_parameter
            ),
            common_window_margin_fraction=(
                self._config.common_window_margin_fraction
            ),
            reset_commissioning=self._reset_commissioning,
            reset_relative_mode=self._config.reset_relative_mode,
            reset_relative_contrast_step=(
                self._config.reset_relative_contrast_step
            ),
            raw_active_mode=self._config.raw_active_mode,
            raw_active_unsupported_quad_policy=(
                self._config.raw_active_unsupported_quad_policy
            ),
        )
        validate_ibm_reram_target_mapping_preflight(report)
        return mapped.detach().clone(), report

    def _targets(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[tuple[int, tuple[int, ...]]],
        dict[str, Any],
    ]:
        global_targets, layout = self._global_targets()
        population = self._population_on(global_targets.device)
        targets, report = map_ibm_reram_array_targets(
            global_targets,
            population,
            target_mapping=self._config.target_mapping,
            dual_rail_layout_by_parameter=(
                self._config.dual_rail_layout_by_parameter
            ),
            common_window_margin_fraction=(
                self._config.common_window_margin_fraction
            ),
            reset_commissioning=self._reset_commissioning,
            reset_relative_mode=self._config.reset_relative_mode,
            reset_relative_contrast_step=(
                self._config.reset_relative_contrast_step
            ),
            raw_active_mode=self._config.raw_active_mode,
            raw_active_unsupported_quad_policy=(
                self._config.raw_active_unsupported_quad_policy
            ),
        )
        validate_ibm_reram_target_mapping_preflight(report)
        return global_targets, targets, layout, report

    def _write_endpoints(
        self,
        endpoint: torch.Tensor,
        layout: Sequence[tuple[int, tuple[int, ...]]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        snapshots = []
        span = self._conductance_max - self._conductance_min
        for binding, (offset, shape) in zip(self._bindings, layout):
            state = binding.state
            count = math.prod(shape)
            clean = state.detach().clone()
            snapshots.append((state, clean))
            programmed = endpoint[offset : offset + count].reshape(shape)
            state.copy_(self._conductance_min + span * programmed.to(state))
            if self._config.target_mapping != "raw_active_p90_quad":
                binding.parameter.clamp_()
        return snapshots

    def _mapped_target(
        self,
        targets: torch.Tensor,
        *,
        mapping_report: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Apply the raw p90 mapper without inventing a compact P&V fit.

        Eligible quads receive their exact continuous or seven-level target.
        A quad outside the frozen D90 support is an explicit structural
        failure and is represented at its transformed RESET bound for this
        deterministic training surrogate.  Exact deployment still performs
        boundary conditioning and one-pulse P&V.
        """

        if self._config.target_mapping != "raw_active_p90_quad":
            raise RuntimeError(
                "Expected mapped_target execution only for raw-active p90."
            )
        population = self._population_on(targets.device)
        structural, lower, upper = _raw_active_structural_cell_mask(
            population,
            self._config.dual_rail_layout_by_parameter or (),
            device=targets.device,
        )
        target_in_support = (
            torch.isfinite(targets)
            & (targets >= 0.0)
            & (targets >= lower)
            & (targets <= upper)
        )
        accepted = structural & target_in_support
        endpoint = torch.where(accepted, targets, lower)
        below = targets < lower
        above = targets > upper
        zero_long = torch.zeros(
            population.size, dtype=torch.int64, device=targets.device
        )
        report = {
            "execution": "mapped_target",
            "execution_detail": (
                "deterministic_raw_target_with_structural_reset_bound_state"
            ),
            "devices": population.size,
            "accepted": int(accepted.sum().item()),
            "success_fraction": float(
                accepted.to(torch.float64).mean().item()
            ),
            "structural_failure": int((~structural).sum().item()),
            "exact_target_in_support": int(
                target_in_support.sum().item()
            ),
            "target_below_lower_bound": int(below.sum().item()),
            "target_above_upper_bound": int(above.sum().item()),
            "endpoint_clipped": 0,
            "pulse_count": {
                "mean": 0.0,
                "median": 0.0,
                "maximum": 0,
                "set_mean": 0.0,
                "reset_mean": 0.0,
            },
            "mapping_only_training_surrogate": True,
            "historical_q_compact_endpoint_model_consumed": False,
        }
        deployment = {
            "schema": "ebl.ibm_reram.om_mapped_target_deployment",
            "schema_version": 1,
            "device_model_sha256": self._artifact_sha256,
            "population_fingerprint": population.fingerprint,
            "config": asdict(self._config),
            "binding_keys": population.binding_keys,
            "binding_shapes": population.binding_shapes,
            "requested_target": targets.detach().cpu().clone(),
            "raw_apparent_endpoint": endpoint.detach().cpu().clone(),
            "apparent_endpoint": endpoint.detach().cpu().clone(),
            "persistent_endpoint": endpoint.detach().cpu().clone(),
            "accepted": accepted.detach().cpu().clone(),
            "structural_quad_eligible": structural.detach().cpu().clone(),
            "exact_target_in_support": (
                target_in_support.detach().cpu().clone()
            ),
            "target_below_lower_bound": below.detach().cpu().clone(),
            "target_above_upper_bound": above.detach().cpu().clone(),
            "set_count": zero_long.detach().cpu().clone(),
            "reset_count": zero_long.detach().cpu().clone(),
            "total_pulses": zero_long.detach().cpu().clone(),
            "verify_count": zero_long.detach().cpu().clone(),
            "reversals": zero_long.detach().cpu().clone(),
            "target_mapping_report": dict(mapping_report),
            "report": report,
        }
        return endpoint.to(dtype=targets.dtype), report, deployment

    def _run_exact_programming(
        self,
        targets: torch.Tensor,
        population: IbmReramArrayPopulation,
        *,
        generator: torch.Generator,
    ) -> tuple[ProgramVerifyResult, _ArrayPlant]:
        """Run the declared controller on exactly the supplied identities."""

        plant = _ArrayPlant(population, generator=generator, device=targets.device)
        branch = (
            "continuous"
            if self._config.corruption_policy == "counterfactual_repaired"
            else "published_corruption"
        )
        estimator = PopulationStepEstimator.from_mapping(
            self._artifact["step_estimators"][branch]
        )
        adaptive = self._artifact["programming"]["adaptive"]
        settings = ControllerSettings(
            kind="adaptive",
            eta=float(adaptive["eta"]),
            maximum_batch=int(adaptive["maximum_batch"]),
            epsilon=float(adaptive["epsilon"]),
            force_one_within_steps=float(adaptive["force_one_within_steps"]),
        )
        tolerance = (
            self._config.tolerance_step_ratio
            * population.nominal_dw_min
            / 2.0
        )
        result = run_program_verify(
            plant.controller_port(),
            targets=targets,
            tolerance=tolerance,
            maximum_pulses=self._config.maximum_program_pulses,
            settings=settings,
            estimator=estimator,
        )
        return result, plant

    def _run_raw_active_programming(
        self,
        targets: torch.Tensor,
        population: IbmReramArrayPopulation,
        *,
        generator: torch.Generator,
    ) -> tuple[
        ProgramVerifyResult,
        _RawActiveArrayPlant,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Condition raw ``a`` from zero, then run independent one-pulse P&V."""

        conditioning_seed = derive_seed(
            int(generator.initial_seed()),
            population.fingerprint,
            "raw_active_boundary_conditioning",
        )
        conditioning_generator = torch.Generator(device=targets.device)
        conditioning_generator.manual_seed(conditioning_seed)
        plant = _RawActiveArrayPlant(
            population,
            generator=conditioning_generator,
            device=targets.device,
        )
        conditioning = plant.condition_lower_boundary()
        conditioning_generator_state = (
            conditioning_generator.get_state().detach().cpu().clone()
        )
        # Target programming owns a separate explicit RNG stream.  The
        # conditioned persistent/apparent states remain unchanged.
        plant.set_generator(generator)
        structural, lower, upper = _raw_active_structural_cell_mask(
            population,
            self._config.dual_rail_layout_by_parameter or (),
            device=targets.device,
        )
        exact_target_in_support = (
            torch.isfinite(targets)
            & (targets >= 0.0)
            & (targets >= lower)
            & (targets <= upper)
        )
        conditioning_success = conditioning["success"]
        programming_eligible = (
            structural & exact_target_in_support & conditioning_success
        )
        result = run_program_verify(
            plant.controller_port(),
            targets=targets,
            tolerance=IBM_OM_RAW_ACTIVE_TOLERANCE,
            maximum_pulses=self._config.maximum_program_pulses,
            settings=ControllerSettings(kind="one_pulse"),
            estimator=None,
            eligible=programming_eligible,
        )
        masks = {
            "structural_quad_eligible": structural,
            "exact_target_in_support": exact_target_in_support,
            "programming_eligible": programming_eligible,
            "lower": lower,
            "upper": upper,
            "conditioning_generator_state_after": (
                conditioning_generator_state
            ),
            "conditioning_seed": torch.tensor(
                conditioning_seed, dtype=torch.int64
            ),
        }
        return result, plant, conditioning, masks

    def _compact(
        self,
        targets: torch.Tensor,
        *,
        generator: torch.Generator,
        mapping_report: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        population = self._population_on(targets.device)
        branch = (
            "continuous"
            if self._config.corruption_policy == "counterfactual_repaired"
            else "published_corruption"
        )
        model = self._artifact["endpoint_models"][branch]
        lower, upper = LOGICAL.cell_bounds(population)
        noncorrupt = ~population.corrupt
        below = noncorrupt & (targets < lower)
        above = noncorrupt & (targets > upper)
        inside = noncorrupt & ~(below | above)
        outside_fixed_support = below | above

        # Empty common-window intersections are an explicitly budgeted mapper
        # failure. They can expose a target/reachability class absent from the
        # compact calibration fit. Preserve that null as unsupported and run
        # the physical controller only on those cells; literal/global mapping
        # remains fail-closed in the compact sampler.
        pulse_fallback_mask = torch.zeros_like(population.corrupt)
        mapping_group_suffix = None
        if self._config.target_mapping == "dual_rail_quad_common_window":
            mapping_group_suffix = "quad"
        elif self._config.target_mapping == "differential_pair_common_window":
            mapping_group_suffix = "pair"
        elif self._config.target_mapping == "shared_reset_relative_quad":
            expected_support = mapping_report.get("mapped_target_support")
            if not isinstance(expected_support, Mapping):
                raise RuntimeError(
                    "Expected RESET-relative compact execution to receive "
                    "its authoritative hidden-support audit."
                )
            expected_below = expected_support.get("below_lower_bound")
            expected_above = expected_support.get("above_upper_bound")
            if (
                expected_below != int(below.sum().item())
                or expected_above != int(above.sum().item())
            ):
                raise RuntimeError(
                    "Expected RESET-relative exact-fallback cells to equal "
                    "the audit-only out-of-bound target set."
                )
            pulse_fallback_mask = outside_fixed_support
        if mapping_group_suffix is not None:
            expected_below = mapping_report.get(
                f"mapped_target_below_lower_bound_empty_{mapping_group_suffix}"
            )
            expected_above = mapping_report.get(
                f"mapped_target_above_upper_bound_empty_{mapping_group_suffix}"
            )
            nonempty_below = mapping_report.get(
                "mapped_target_below_lower_bound_nonempty_"
                f"{mapping_group_suffix}"
            )
            nonempty_above = mapping_report.get(
                "mapped_target_above_upper_bound_nonempty_"
                f"{mapping_group_suffix}"
            )
            if (
                expected_below != int(below.sum().item())
                or expected_above != int(above.sum().item())
                or nonempty_below != 0
                or nonempty_above != 0
            ):
                raise RuntimeError(
                    "Expected compact exact-fallback cells to equal the "
                    "preflighted out-of-bound cells from empty "
                    f"{mapping_group_suffix} windows."
                )
            pulse_fallback_mask = outside_fixed_support
        compact_mask = ~pulse_fallback_mask

        stuck = LOGICAL.from_state(population.logical_min)
        write_sigma = (
            population.write_noise_std * population.nominal_dw_min / 2.0
        )
        raw_endpoint = torch.full_like(targets, float("nan"))
        endpoint = torch.full_like(targets, float("nan"))
        persistent_endpoint = torch.full_like(targets, float("nan"))
        accepted = torch.zeros_like(population.corrupt)
        endpoint_was_clipped = torch.zeros_like(population.corrupt)

        if bool(torch.any(compact_mask)):
            corrupt_apparent = stuck[compact_mask] + write_sigma * torch.randn(
                (int(compact_mask.sum().item()),),
                dtype=stuck.dtype,
                device=stuck.device,
                generator=generator,
            )
            sample = sample_ibm_reram_endpoints(
                targets[compact_mask],
                model,
                condition_key=CONDITION_KEY,
                generator=generator,
                corrupt_mask=population.corrupt[compact_mask],
                lower_bound=lower[compact_mask],
                upper_bound=upper[compact_mask],
                corrupt_apparent_endpoint=corrupt_apparent,
                corrupt_persistent_endpoint=stuck[compact_mask],
                target_out_of_support=self._config.target_out_of_support,
                endpoint_policy=self._config.endpoint_policy,
            )
            raw_endpoint[compact_mask] = sample.raw_endpoint
            endpoint[compact_mask] = sample.endpoint
            persistent_endpoint[compact_mask] = sample.persistent_endpoint
            accepted[compact_mask] = sample.accepted
            endpoint_was_clipped[compact_mask] = sample.endpoint_was_clipped

        fallback_indices = torch.where(pulse_fallback_mask)[0]
        fallback_result: ProgramVerifyResult | None = None
        fallback_plant: _ArrayPlant | None = None
        if fallback_indices.numel():
            fallback_population = _selected_array_population(
                population,
                pulse_fallback_mask,
            )
            fallback_result, fallback_plant = self._run_exact_programming(
                targets[pulse_fallback_mask],
                fallback_population,
                generator=generator,
            )
            fallback_raw = fallback_result.apparent_endpoint
            fallback_endpoint = fallback_raw.clamp(0.0, 1.0)
            fallback_persistent = LOGICAL.from_state(fallback_plant.persistent)
            raw_endpoint[pulse_fallback_mask] = fallback_raw
            endpoint[pulse_fallback_mask] = fallback_endpoint
            persistent_endpoint[pulse_fallback_mask] = fallback_persistent
            accepted[pulse_fallback_mask] = fallback_result.accepted
            endpoint_was_clipped[pulse_fallback_mask] = (
                fallback_endpoint != fallback_raw
            )

        if not bool(
            torch.all(torch.isfinite(raw_endpoint))
            and torch.all(torch.isfinite(endpoint))
            and torch.all(torch.isfinite(persistent_endpoint))
        ):
            raise RuntimeError(
                "Expected compact and pulse-resolved fallback endpoints to "
                "partition every physical identity."
            )

        tolerance = (
            self._config.tolerance_step_ratio
            * population.nominal_dw_min
            / 2.0
        )
        window_reachable = noncorrupt & (
            (targets + tolerance >= lower) & (targets - tolerance <= upper)
        )
        fallback_index_bytes = (
            fallback_indices.detach().cpu().to(dtype=torch.int64).numpy().tobytes()
        )
        fallback_index_sha256 = sha256(fallback_index_bytes).hexdigest()
        fallback_policy = (
            "pulse_resolved_noncorrupt_out_of_bound_empty_pair_only"
            if self._config.target_mapping
            == "differential_pair_common_window"
            else (
                "pulse_resolved_noncorrupt_out_of_bound_shared_reset_relative_only"
                if self._config.target_mapping
                == "shared_reset_relative_quad"
                else "pulse_resolved_noncorrupt_out_of_bound_empty_quad_only"
            )
        )
        fallback_report: dict[str, Any] = {
            "policy": fallback_policy,
            "devices": int(fallback_indices.numel()),
            "selection_indices": fallback_indices.detach().cpu().tolist(),
            "selection_indices_sha256": fallback_index_sha256,
            "target_below_lower_bound": int(
                (below & pulse_fallback_mask).sum().item()
            ),
            "target_above_upper_bound": int(
                (above & pulse_fallback_mask).sum().item()
            ),
            "acceptance_window_unreachable": int(
                (pulse_fallback_mask & ~window_reachable).sum().item()
            ),
            "start_state": "sampled_fully_reset_bound",
            "controller": self._config.controller,
            "maximum_program_pulses": self._config.maximum_program_pulses,
            "random_stream_order": "after_compact_endpoint_sampling",
        }
        if fallback_result is not None:
            fallback_report.update(_result_mapping(fallback_result))
            assert fallback_plant is not None
            fallback_physical = (
                fallback_plant.persistent + fallback_plant.population.reference
            )
            fallback_saturated = (
                torch.abs(
                    fallback_physical - fallback_plant.population.min_bound
                )
                <= 2e-6
            ) | (
                torch.abs(
                    fallback_physical - fallback_plant.population.max_bound
                )
                <= 2e-6
            )
            fallback_report["saturated"] = int(
                fallback_saturated.sum().item()
            )
            fallback_report["endpoint_clipped"] = int(
                endpoint_was_clipped[pulse_fallback_mask].sum().item()
            )
        else:
            fallback_report.update(
                {
                    "accepted": 0,
                    "success_fraction": None,
                    "budget_exhausted": 0,
                    "nonfinite": 0,
                    "pulse_count": {
                        "mean": None,
                        "median": None,
                        "maximum": 0,
                        "set_mean": None,
                        "reset_mean": None,
                    },
                    "verify_count_mean": None,
                    "reversal_count_mean": None,
                    "saturated": 0,
                    "endpoint_clipped": 0,
                }
            )

        failed_noncorrupt = noncorrupt & ~accepted
        if self._config.target_mapping == "differential_pair_common_window":
            fallback_execution_detail = (
                "compact_endpoint_with_exact_empty_pair_fallback"
            )
        elif self._config.target_mapping == "shared_reset_relative_quad":
            fallback_execution_detail = (
                "compact_endpoint_with_exact_reset_relative_support_fallback"
            )
        else:
            fallback_execution_detail = (
                "compact_endpoint_with_exact_empty_quad_fallback"
            )
        report = {
            "execution": "compact_endpoint",
            "execution_detail": (
                fallback_execution_detail
                if fallback_indices.numel()
                else "compact_endpoint_only"
            ),
            "devices": int(targets.numel()),
            "compact_endpoint_devices": int(compact_mask.sum().item()),
            "pulse_resolved_fallback_devices": int(fallback_indices.numel()),
            "accepted": int(accepted.sum().item()),
            "success_fraction": float(accepted.float().mean().item()),
            "failed_noncorrupt": int(failed_noncorrupt.sum().item()),
            "corrupt": int(population.corrupt.sum().item()),
            "published_corrupt": int(
                population.published_corrupt.sum().item()
            ),
            "accepted_corrupt": int(
                (accepted & population.corrupt).sum().item()
            ),
            "target_below_lower_bound": int(below.sum().item()),
            "target_inside_bounds": int(inside.sum().item()),
            "target_above_upper_bound": int(above.sum().item()),
            "acceptance_window_unreachable": int(
                (noncorrupt & ~window_reachable).sum().item()
            ),
            "endpoint_clipped": int(endpoint_was_clipped.sum().item()),
            "pulse_resolved_fallback": fallback_report,
        }
        empty_long = torch.empty(0, dtype=torch.int64)
        empty_bool = torch.empty(0, dtype=torch.bool)
        fallback_deployment = {
            "policy": fallback_report["policy"],
            "selection_indices": fallback_indices.detach().cpu().clone(),
            "selection_indices_sha256": fallback_index_sha256,
            "requested_target": (
                targets[pulse_fallback_mask].detach().cpu().clone()
            ),
            "raw_apparent_endpoint": (
                raw_endpoint[pulse_fallback_mask].detach().cpu().clone()
            ),
            "apparent_endpoint": (
                endpoint[pulse_fallback_mask].detach().cpu().clone()
            ),
            "persistent_endpoint": (
                persistent_endpoint[pulse_fallback_mask].detach().cpu().clone()
            ),
            "accepted": accepted[pulse_fallback_mask].detach().cpu().clone(),
            "budget_exhausted": (
                fallback_result.budget_exhausted.detach().cpu().clone()
                if fallback_result is not None
                else empty_bool
            ),
            "set_count": (
                fallback_result.set_count.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "reset_count": (
                fallback_result.reset_count.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "total_pulses": (
                fallback_result.total_pulses.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "verify_count": (
                fallback_result.verify_count.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "reversals": (
                fallback_result.reversals.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "report": fallback_report,
        }
        deployment = {
            "schema": "ebl.ibm_reram.om_compact_endpoint_deployment",
            "schema_version": 1,
            "device_model_sha256": self._artifact_sha256,
            "population_fingerprint": population.fingerprint,
            "config": asdict(self._config),
            "binding_keys": population.binding_keys,
            "binding_shapes": population.binding_shapes,
            "endpoint_generation_policy": (
                "compact_covered_exact_out_of_bound_empty_pair_fallback"
                if self._config.target_mapping
                == "differential_pair_common_window"
                else (
                    "compact_covered_exact_out_of_bound_reset_relative_fallback"
                    if self._config.target_mapping
                    == "shared_reset_relative_quad"
                    else "compact_covered_exact_out_of_bound_empty_quad_fallback"
                )
            ),
            "requested_target": targets.detach().cpu().clone(),
            "apparent_endpoint": endpoint.detach().cpu().clone(),
            "raw_apparent_endpoint": raw_endpoint.detach().cpu().clone(),
            "persistent_endpoint": persistent_endpoint.detach().cpu().clone(),
            "accepted": accepted.detach().cpu().clone(),
            "compact_endpoint_mask": compact_mask.detach().cpu().clone(),
            "pulse_resolved_fallback_mask": (
                pulse_fallback_mask.detach().cpu().clone()
            ),
            "pulse_resolved_fallback": fallback_deployment,
            "generator_state_after_programming": generator.get_state()
            .detach()
            .cpu()
            .clone(),
        }
        return endpoint.to(dtype=targets.dtype), report, deployment

    def _pulse_resolved_raw_active(
        self,
        targets: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        population = self._population_on(targets.device)
        result, plant, conditioning, masks = (
            self._run_raw_active_programming(
                targets,
                population,
                generator=generator,
            )
        )
        lower = masks["lower"]
        upper = masks["upper"]
        structural = masks["structural_quad_eligible"]
        exact_support = masks["exact_target_in_support"]
        programming_eligible = masks["programming_eligible"]
        conditioning_success = conditioning["success"]
        apparent_endpoint = result.apparent_endpoint
        persistent_endpoint = RAW_ACTIVE.from_state(plant.persistent_a)
        conditioned_persistent = RAW_ACTIVE.from_state(
            conditioning["persistent_a"]
        )
        conditioned_apparent = RAW_ACTIVE.from_state(conditioning["apparent_a"])
        below = targets < lower
        above = targets > upper
        acceptance_window_intersects_exact_support = (
            (targets + IBM_OM_RAW_ACTIVE_TOLERANCE >= lower)
            & (targets - IBM_OM_RAW_ACTIVE_TOLERANCE <= upper)
        )
        persistent_within_tolerance = (
            torch.abs(persistent_endpoint - targets)
            <= IBM_OM_RAW_ACTIVE_TOLERANCE
        )
        accepted_persistent_within_tolerance = (
            result.accepted & persistent_within_tolerance
        )
        saturated = (
            torch.abs(plant.persistent_a - population.min_bound)
            <= IBM_OM_RAW_ACTIVE_CONDITIONING_CHANGE_THRESHOLD
        ) | (
            torch.abs(plant.persistent_a - population.max_bound)
            <= IBM_OM_RAW_ACTIVE_CONDITIONING_CHANGE_THRESHOLD
        )

        def residual_summary(
            residual: torch.Tensor,
            selection: torch.Tensor,
        ) -> dict[str, float | int | None]:
            selected = residual[selection]
            if selected.numel() == 0:
                return {
                    "count": 0,
                    "bias": None,
                    "mae": None,
                    "rmse": None,
                }
            return {
                "count": int(selected.numel()),
                "bias": float(selected.mean().item()),
                "mae": float(selected.abs().mean().item()),
                "rmse": float(selected.square().mean().sqrt().item()),
            }

        base_result = _result_mapping(result)
        eligible_count = int(programming_eligible.sum().item())
        eligible_accepted = int(
            (result.accepted & programming_eligible).sum().item()
        )
        report = {
            "execution": "pulse_resolved",
            "execution_detail": (
                "raw_active_boundary_conditioned_one_pulse_verify"
            ),
            **base_result,
            "controller": "one_pulse",
            "coordinate_version": IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
            "coordinate_a_min": IBM_OM_RAW_ACTIVE_A_MIN,
            "coordinate_a_max": IBM_OM_RAW_ACTIVE_A_MAX,
            "coordinate_scale": IBM_OM_RAW_ACTIVE_SCALE,
            "tolerance": IBM_OM_RAW_ACTIVE_TOLERANCE,
            "corrupt": int(population.corrupt.sum().item()),
            "published_corrupt": int(
                population.published_corrupt.sum().item()
            ),
            "structural_quad_eligible": int(structural.sum().item()),
            "structural_target_assignment_failure": int(
                (~structural).sum().item()
            ),
            "exact_target_in_support": int(exact_support.sum().item()),
            "programming_eligible": eligible_count,
            "programming_eligible_accepted": eligible_accepted,
            "programming_eligible_success_fraction": (
                eligible_accepted / eligible_count if eligible_count else None
            ),
            "conditioning_success": int(
                conditioning_success.sum().item()
            ),
            "conditioning_failure": int(
                (~conditioning_success).sum().item()
            ),
            "conditioning_pulse_count": _tensor_summary(
                conditioning["pulse_count"].to(torch.float32)
            ),
            "conditioning_total_pulses": int(
                conditioning["pulse_count"].sum().item()
            ),
            "conditioning_quiet_steps": (
                IBM_OM_RAW_ACTIVE_CONDITIONING_QUIET_STEPS
            ),
            "conditioning_change_threshold_raw_a": (
                IBM_OM_RAW_ACTIVE_CONDITIONING_CHANGE_THRESHOLD
            ),
            "conditioning_maximum_pulses": (
                IBM_OM_RAW_ACTIVE_CONDITIONING_MAXIMUM_PULSES
            ),
            "conditioning_seed": int(masks["conditioning_seed"].item()),
            "target_below_lower_bound": int(below.sum().item()),
            "target_inside_bounds": int((~below & ~above).sum().item()),
            "target_above_upper_bound": int(above.sum().item()),
            "acceptance_window_intersects_exact_support": int(
                acceptance_window_intersects_exact_support.sum().item()
            ),
            "accepted_persistent_within_tolerance": int(
                accepted_persistent_within_tolerance.sum().item()
            ),
            "accepted_persistent_outside_tolerance": int(
                (
                    result.accepted & ~persistent_within_tolerance
                ).sum().item()
            ),
            "saturated": int(saturated.sum().item()),
            "endpoint_clipped": 0,
            "reference_consumed_by_plant_or_controller": False,
            "accepted_apparent_residual": residual_summary(
                apparent_endpoint - targets,
                result.accepted,
            ),
            "accepted_persistent_residual": residual_summary(
                persistent_endpoint - targets,
                result.accepted,
            ),
        }
        deployment = {
            "schema": "ebl.ibm_reram.om_pulse_resolved_deployment",
            "schema_version": 1,
            "device_model_sha256": self._artifact_sha256,
            "population_fingerprint": population.fingerprint,
            "population_sampling_receipt": self.population_sampling_receipt,
            "config": asdict(self._config),
            "binding_keys": population.binding_keys,
            "binding_shapes": population.binding_shapes,
            "binding_sampling_seeds": population.binding_sampling_seeds,
            "donor_sampling_seeds": population.donor_sampling_seeds,
            "population_scalars": {
                "preset": OM_PRESET,
                "aihwkit_version": population.aihwkit_version,
                "nominal_dw_min": population.nominal_dw_min,
                "dw_min_std": population.dw_min_std,
                "write_noise_std": population.write_noise_std,
            },
            "population": population.tensor_state(),
            "coordinate": {
                "version": IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
                "a_min": IBM_OM_RAW_ACTIVE_A_MIN,
                "a_max": IBM_OM_RAW_ACTIVE_A_MAX,
                "scale": IBM_OM_RAW_ACTIVE_SCALE,
                "reference_excluded_from_numerics": True,
            },
            "requested_target": targets.detach().cpu().clone(),
            "persistent_endpoint": (
                persistent_endpoint.detach().cpu().clone()
            ),
            "raw_apparent_endpoint": (
                apparent_endpoint.detach().cpu().clone()
            ),
            "apparent_endpoint": apparent_endpoint.detach().cpu().clone(),
            "accepted": result.accepted.detach().cpu().clone(),
            "nonfinite": result.nonfinite.detach().cpu().clone(),
            "budget_exhausted": (
                result.budget_exhausted.detach().cpu().clone()
            ),
            "corrupt": population.corrupt.detach().cpu().clone(),
            "structural_quad_eligible": structural.detach().cpu().clone(),
            "structural_target_assignment_failure": (
                (~structural).detach().cpu().clone()
            ),
            "exact_target_in_support": (
                exact_support.detach().cpu().clone()
            ),
            "programming_eligible": (
                programming_eligible.detach().cpu().clone()
            ),
            "target_below_lower_bound": below.detach().cpu().clone(),
            "target_inside_bounds": (
                (~below & ~above).detach().cpu().clone()
            ),
            "target_above_upper_bound": above.detach().cpu().clone(),
            "acceptance_window_intersects_exact_support": (
                acceptance_window_intersects_exact_support
                .detach()
                .cpu()
                .clone()
            ),
            "persistent_within_tolerance": (
                persistent_within_tolerance.detach().cpu().clone()
            ),
            "saturated": saturated.detach().cpu().clone(),
            "set_count": result.set_count.detach().cpu().clone(),
            "reset_count": result.reset_count.detach().cpu().clone(),
            "total_pulses": result.total_pulses.detach().cpu().clone(),
            "verify_count": result.verify_count.detach().cpu().clone(),
            "reversals": result.reversals.detach().cpu().clone(),
            "conditioning": {
                "success": conditioning_success.detach().cpu().clone(),
                "pulse_count": (
                    conditioning["pulse_count"].detach().cpu().clone()
                ),
                "persistent_a": (
                    conditioning["persistent_a"].detach().cpu().clone()
                ),
                "apparent_a": (
                    conditioning["apparent_a"].detach().cpu().clone()
                ),
                "persistent_g": (
                    conditioned_persistent.detach().cpu().clone()
                ),
                "apparent_g": conditioned_apparent.detach().cpu().clone(),
                "seed": int(masks["conditioning_seed"].item()),
                "generator_state_after": masks[
                    "conditioning_generator_state_after"
                ],
            },
            "persistent_a": plant.persistent_a.detach().cpu().clone(),
            "apparent_a": plant.apparent_a.detach().cpu().clone(),
            "generator_state_after_programming": generator.get_state()
            .detach()
            .cpu()
            .clone(),
            "endpoint_application_policy": (
                IBM_RERAM_ENDPOINT_APPLICATION_POLICY
            ),
            "report": report,
        }
        return apparent_endpoint.to(dtype=targets.dtype), report, deployment

    def _pulse_resolved(
        self,
        targets: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        if self._config.target_mapping == "raw_active_p90_quad":
            return self._pulse_resolved_raw_active(
                targets,
                generator=generator,
            )
        population = self._population_on(targets.device)
        result, plant = self._run_exact_programming(
            targets,
            population,
            generator=generator,
        )
        tolerance = (
            self._config.tolerance_step_ratio
            * population.nominal_dw_min
            / 2.0
        )
        raw_endpoint = result.apparent_endpoint
        endpoint = raw_endpoint.clamp(0.0, 1.0)
        lower, upper = LOGICAL.cell_bounds(population)
        noncorrupt = ~population.corrupt
        below = noncorrupt & (targets < lower)
        above = noncorrupt & (targets > upper)
        inside = noncorrupt & ~(below | above)
        window_reachable = noncorrupt & (
            (targets + tolerance >= lower) & (targets - tolerance <= upper)
        )
        persistent_endpoint = LOGICAL.from_state(plant.persistent)
        accepted_noncorrupt = result.accepted & noncorrupt
        apparent_residual = raw_endpoint - targets
        persistent_residual = persistent_endpoint - targets
        physical = plant.persistent + population.reference
        saturated = (
            (torch.abs(physical - population.min_bound) <= 2e-6)
            | (torch.abs(physical - population.max_bound) <= 2e-6)
        )

        def residual_summary(
            residual: torch.Tensor,
            selection: torch.Tensor,
        ) -> dict[str, float | int | None]:
            selected = residual[selection]
            if selected.numel() == 0:
                return {"count": 0, "bias": None, "mae": None, "rmse": None}
            return {
                "count": int(selected.numel()),
                "bias": float(selected.mean().item()),
                "mae": float(selected.abs().mean().item()),
                "rmse": float(selected.square().mean().sqrt().item()),
            }

        report = {
            "execution": "pulse_resolved",
            **_result_mapping(result),
            "corrupt": int(population.corrupt.sum().item()),
            "published_corrupt": int(population.published_corrupt.sum().item()),
            "accepted_corrupt": int(
                (result.accepted & population.corrupt).sum().item()
            ),
            "budget_exhausted_noncorrupt": int(
                (result.budget_exhausted & noncorrupt).sum().item()
            ),
            "budget_exhausted_corrupt": int(
                (result.budget_exhausted & population.corrupt).sum().item()
            ),
            "target_below_lower_bound": int(below.sum().item()),
            "target_inside_bounds": int(inside.sum().item()),
            "target_above_upper_bound": int(above.sum().item()),
            "acceptance_window_unreachable": int(
                (noncorrupt & ~window_reachable).sum().item()
            ),
            "saturated": int(saturated.sum().item()),
            "endpoint_clipped": int((endpoint != raw_endpoint).sum().item()),
            "accepted_noncorrupt_apparent_residual": residual_summary(
                apparent_residual,
                accepted_noncorrupt,
            ),
            "accepted_noncorrupt_persistent_residual": residual_summary(
                persistent_residual,
                accepted_noncorrupt,
            ),
        }
        deployment = {
            "schema": "ebl.ibm_reram.om_pulse_resolved_deployment",
            "schema_version": 1,
            "device_model_sha256": self._artifact_sha256,
            "population_fingerprint": population.fingerprint,
            "population_sampling_receipt": self.population_sampling_receipt,
            "config": asdict(self._config),
            "binding_keys": population.binding_keys,
            "binding_shapes": population.binding_shapes,
            "binding_sampling_seeds": population.binding_sampling_seeds,
            "donor_sampling_seeds": population.donor_sampling_seeds,
            "population_scalars": {
                "preset": OM_PRESET,
                "aihwkit_version": population.aihwkit_version,
                "nominal_dw_min": population.nominal_dw_min,
                "dw_min_std": population.dw_min_std,
                "write_noise_std": population.write_noise_std,
            },
            "population": population.tensor_state(),
            "requested_target": targets.detach().cpu().clone(),
            "persistent_endpoint": persistent_endpoint.detach().cpu().clone(),
            "raw_apparent_endpoint": raw_endpoint.detach().cpu().clone(),
            "apparent_endpoint": endpoint.detach().cpu().clone(),
            "accepted": result.accepted.detach().cpu().clone(),
            "budget_exhausted": result.budget_exhausted.detach().cpu().clone(),
            "corrupt": population.corrupt.detach().cpu().clone(),
            "target_below_lower_bound": below.detach().cpu().clone(),
            "target_inside_bounds": inside.detach().cpu().clone(),
            "target_above_upper_bound": above.detach().cpu().clone(),
            "acceptance_window_reachable": window_reachable.detach().cpu().clone(),
            "saturated": saturated.detach().cpu().clone(),
            "set_count": result.set_count.detach().cpu().clone(),
            "reset_count": result.reset_count.detach().cpu().clone(),
            "total_pulses": result.total_pulses.detach().cpu().clone(),
            "verify_count": result.verify_count.detach().cpu().clone(),
            "reversals": result.reversals.detach().cpu().clone(),
            "generator_state_after_programming": generator.get_state()
            .detach()
            .cpu()
            .clone(),
            "report": report,
        }
        return endpoint.to(dtype=targets.dtype), report, deployment

    @contextmanager
    def _context(self, stream: str, *, enabled: bool) -> Iterator["IbmReramHwaParameterModifier"]:
        if self._active_stream is not None:
            raise RuntimeError(
                "Expected IBM OM modifier contexts not to overlap. "
                f"Provided value: active={self._active_stream!r}, requested={stream!r}."
            )
        self._active_stream = stream
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            if enabled:
                with torch.no_grad():
                    global_targets, targets, layout, mapping_report = self._targets()
                    generator = self._generator(stream, targets.device)
                    if self._config.execution == "mapped_target":
                        endpoint, report, deployment = self._mapped_target(
                            targets,
                            mapping_report=mapping_report,
                        )
                    elif self._config.execution == "compact_endpoint":
                        endpoint, report, deployment = self._compact(
                            targets,
                            generator=generator,
                            mapping_report=mapping_report,
                        )
                    else:
                        endpoint, report, deployment = self._pulse_resolved(
                            targets, generator=generator
                        )
                    snapshots = self._write_endpoints(endpoint, layout)
                    deployment["global_requested_target"] = (
                        global_targets.detach().cpu().clone()
                    )
                    deployment["target_mapping_report"] = mapping_report
                    deployment["reset_commissioning"] = (
                        self.reset_commissioning_bundle
                    )
                    self._last_report = {
                        **report,
                        "endpoint_application_policy": (
                            IBM_RERAM_ENDPOINT_APPLICATION_POLICY
                        ),
                        "target_mapping": self._config.target_mapping,
                        "target_mapping_report": mapping_report,
                        "assignment_seed": self._config.assignment_seed,
                        "endpoint_seed": self._config.endpoint_seed,
                        "corruption_policy": self._config.corruption_policy,
                        "population_fingerprint": self._population.fingerprint,
                        "device_model_sha256": self._artifact_sha256,
                        "population_sampling_backend": (
                            self._population_sampling_receipt.get("backend")
                            if self._population_sampling_receipt is not None
                            else "in_process_aihwkit"
                        ),
                        "population_receipt_sha256": (
                            sha256_file(self._population_artifact_paths[1])
                            if self._population_artifact_paths is not None
                            else None
                        ),
                        "reset_commissioning_receipt_sha256": (
                            sha256_file(
                                self._reset_commissioning_artifact_paths[1]
                            )
                            if self._reset_commissioning_artifact_paths
                            is not None
                            else None
                        ),
                    }
                    deployment["population_sampling_receipt"] = (
                        self.population_sampling_receipt
                    )
                    deployment["endpoint_application_policy"] = (
                        IBM_RERAM_ENDPOINT_APPLICATION_POLICY
                    )
                    deployment["report"] = dict(self._last_report)
                    self._last_deployment = deployment
            yield self
        finally:
            with torch.no_grad():
                for state, clean in snapshots:
                    state.copy_(clean)
            self._active_stream = None

    def training_context(self):
        return self._context("train", enabled=True)

    def evaluation_context(self):
        return self._context(
            "evaluation", enabled=self._config.noisy_evaluation
        )

    def state_dict(self) -> dict[str, Any]:
        if self._active_stream is not None:
            raise RuntimeError("Expected modifier state outside an active context.")
        streams = {}
        for stream in _STREAMS:
            states = {
                key: value.detach().cpu().clone()
                for key, value in self._pending_generator_states[stream].items()
            }
            states.update(
                {
                    key: generator.get_state().detach().cpu().clone()
                    for key, generator in self._generators[stream].items()
                }
            )
            streams[stream] = states
        return {
            "version": _STATE_VERSION,
            "endpoint_application_policy": (
                IBM_RERAM_ENDPOINT_APPLICATION_POLICY
            ),
            "config": asdict(self._config),
            "conductance_bounds": [self._conductance_min, self._conductance_max],
            "device_model_sha256": self._artifact_sha256,
            "population": {
                "fingerprint": self._population.fingerprint,
                "binding_keys": self._population.binding_keys,
                "binding_shapes": self._population.binding_shapes,
                "binding_sampling_seeds": self._population.binding_sampling_seeds,
                "donor_sampling_seeds": self._population.donor_sampling_seeds,
                "tensors": self._population.tensor_state(),
            },
            "generator_states": streams,
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        if self._active_stream is not None:
            raise RuntimeError("Expected modifier restore outside an active context.")
        expected = {
            "version",
            "endpoint_application_policy",
            "config",
            "conductance_bounds",
            "device_model_sha256",
            "population",
            "generator_states",
        }
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("Expected an exact IBM OM HWA modifier state mapping.")
        if (
            state_dict["endpoint_application_policy"]
            != IBM_RERAM_ENDPOINT_APPLICATION_POLICY
        ):
            raise ValueError(
                "Expected the saved IBM OM endpoint application policy to "
                "match the apparent-forward, persistent-update-state contract."
            )
        saved_config = state_dict["config"]
        if isinstance(saved_config, Mapping):
            saved_config = dict(saved_config)
            saved_config.setdefault("target_mapping", "literal_global")
            saved_config.setdefault("dual_rail_layout_by_parameter", None)
            saved_config.setdefault("common_window_margin_fraction", 0.0)
            saved_config.setdefault("reset_relative_mode", None)
            saved_config.setdefault("reset_relative_contrast_step", None)
            saved_config.setdefault("reset_read_samples", None)
            saved_config.setdefault("reset_guard_standard_errors", None)
            saved_config.setdefault("raw_active_mode", None)
            saved_config.setdefault(
                "raw_active_unsupported_quad_policy", None
            )
            saved_config.setdefault("forward_logit_gain", None)
        if state_dict["version"] != _STATE_VERSION or saved_config != asdict(
            self._config
        ):
            raise ValueError("Expected saved IBM OM HWA version and config to match.")
        if list(state_dict["conductance_bounds"]) != [
            self._conductance_min,
            self._conductance_max,
        ] or state_dict["device_model_sha256"] != self._artifact_sha256:
            raise ValueError(
                "Expected saved conductance bounds and device-model digest to match."
            )
        population = state_dict["population"]
        if not isinstance(population, Mapping) or population.get(
            "fingerprint"
        ) != self._population.fingerprint:
            raise ValueError("Expected saved IBM OM physical assignment to match.")
        if tuple(population.get("binding_keys", ())) != self._population.binding_keys or tuple(
            tuple(shape) for shape in population.get("binding_shapes", ())
        ) != self._population.binding_shapes:
            raise ValueError("Expected saved IBM OM binding layout to match.")
        tensors = population.get("tensors")
        if not isinstance(tensors, Mapping) or any(
            name not in tensors
            or not isinstance(tensors[name], torch.Tensor)
            or not torch.equal(tensors[name].cpu(), current)
            for name, current in self._population.tensor_state().items()
        ):
            raise ValueError("Expected saved IBM OM population tensors to match exactly.")
        raw_streams = state_dict["generator_states"]
        if not isinstance(raw_streams, Mapping) or set(raw_streams) != set(_STREAMS):
            raise ValueError("Expected train and evaluation IBM OM RNG streams.")
        pending: dict[str, dict[str, torch.Tensor]] = {stream: {} for stream in _STREAMS}
        for stream in _STREAMS:
            if not isinstance(raw_streams[stream], Mapping):
                raise ValueError("Expected IBM OM RNG states to be keyed by device.")
            for key, value in raw_streams[stream].items():
                if (
                    not isinstance(key, str)
                    or not isinstance(value, torch.Tensor)
                    or value.dtype != torch.uint8
                    or value.device.type != "cpu"
                ):
                    raise ValueError("Expected CPU uint8 IBM OM generator states.")
                probe = torch.Generator(device=torch.device(key))
                probe.set_state(value)
                pending[stream][key] = value.detach().clone()
        self._generators = {stream: {} for stream in _STREAMS}
        self._pending_generator_states = pending


def build_ibm_reram_hwa_modifier(
    bindings: Sequence[ParameterBinding],
    config: IbmReramHwaConfig,
    *,
    device_model_path: Path,
    conductance_min: float,
    conductance_max: float,
    population_path: Path | None = None,
    population_receipt_path: Path | None = None,
) -> IbmReramHwaParameterModifier:
    return IbmReramHwaParameterModifier(
        bindings,
        config,
        device_model_path=device_model_path,
        conductance_min=conductance_min,
        conductance_max=conductance_max,
        population_path=population_path,
        population_receipt_path=population_receipt_path,
    )


__all__ = [
    "IBM_RERAM_ENDPOINT_APPLICATION_POLICY",
    "IBM_OM_RAW_ACTIVE_A_MAX",
    "IBM_OM_RAW_ACTIVE_A_MIN",
    "IBM_OM_RAW_ACTIVE_COORDINATE_VERSION",
    "IBM_OM_RAW_ACTIVE_D90",
    "IBM_OM_RAW_ACTIVE_SCALE",
    "IBM_OM_RAW_ACTIVE_TOLERANCE",
    "IbmReramArrayPopulation",
    "IbmReramHwaConfig",
    "IbmReramHwaParameterModifier",
    "IbmReramResetCommissioning",
    "build_ibm_reram_hwa_modifier",
    "commission_ibm_reram_reset_relative_baselines",
    "load_om_array_population",
    "map_ibm_reram_array_targets",
    "sample_om_array_population",
    "sample_om_array_population_external",
    "sample_om_array_population_layout",
    "sample_om_array_population_layout_external",
    "save_om_array_population",
    "validate_ibm_reram_target_mapping_preflight",
]
