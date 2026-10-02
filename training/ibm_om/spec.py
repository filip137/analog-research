"""IBM OM HWA configuration: the stored flat config and its validation.

The flat ``IbmReramHwaConfig`` is the canonical stored form.  Besides the
historical fields it declares the device coordinate, the programming start
state and verify tolerance, the readback clamp and the characterization
stage explicitly, so each layer is selected from the config rather than
inferred from another layer.  Configs and artifacts written before these
fields existed are completed with ``legacy_explicit_defaults``, the single
place where the old inference from ``target_mapping`` survives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from training.ibm_om.constants import (
    IBM_OM_RAW_ACTIVE_TOLERANCE,
    _CORRUPTION_POLICIES,
    _EXECUTIONS,
    _RAW_ACTIVE_MODES,
    _RESET_RELATIVE_MODES,
    _TARGET_MAPPINGS,
)
from training.ibm_om.coordinates import (
    DEVICE_COORDINATES,
    LOGICAL_REFERENCE_SUBTRACTED,
    RAW_ACTIVE_V1,
)
from training.ibm_om.topology import _normalize_dual_rail_layouts
from training.ibm_reram_program_verify import OM_PRESET


INITIAL_STATES = ("exact_reset_bound", "conditioned_lower_boundary")
CHARACTERIZATIONS = ("none", "reset_reads")
EXPLICIT_LAYER_FIELDS = (
    "device_coordinate",
    "initial_state",
    "verify_tolerance_absolute",
    "clamp_to_parameter_bounds",
    "characterization",
)


def legacy_explicit_defaults(target_mapping: str) -> dict[str, Any]:
    """Return the layer settings that ``target_mapping`` used to imply."""

    raw_active = target_mapping == "raw_active_p90_quad"
    return {
        "device_coordinate": (
            RAW_ACTIVE_V1 if raw_active else LOGICAL_REFERENCE_SUBTRACTED
        ),
        "initial_state": (
            "conditioned_lower_boundary" if raw_active else "exact_reset_bound"
        ),
        "verify_tolerance_absolute": (
            IBM_OM_RAW_ACTIVE_TOLERANCE if raw_active else None
        ),
        "clamp_to_parameter_bounds": not raw_active,
        "characterization": (
            "reset_reads"
            if target_mapping == "shared_reset_relative_quad"
            else "none"
        ),
    }


def backfill_config_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    """Complete a stored flat config dict that predates the explicit fields."""

    completed = dict(value)
    defaults = legacy_explicit_defaults(
        str(completed.get("target_mapping", "literal_global"))
    )
    for name in EXPLICIT_LAYER_FIELDS:
        completed.setdefault(name, defaults[name])
    return completed


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
    device_coordinate: str | None = None
    initial_state: str | None = None
    verify_tolerance_absolute: float | None = None
    clamp_to_parameter_bounds: bool | None = None
    characterization: str | None = None

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
        self._resolve_explicit_layer_fields()

    def _resolve_explicit_layer_fields(self) -> None:
        """Fill absent layer fields and keep the declared combination valid.

        Only the combinations the code supports are accepted: each explicit
        field must equal what ``target_mapping`` implies.
        """

        defaults = legacy_explicit_defaults(self.target_mapping)
        for name in EXPLICIT_LAYER_FIELDS:
            if getattr(self, name) is None and defaults[name] is not None:
                object.__setattr__(self, name, defaults[name])
        if self.device_coordinate not in DEVICE_COORDINATES:
            raise ValueError(
                "Expected device_coordinate to be one of "
                f"{DEVICE_COORDINATES!r}. Provided value: "
                f"{self.device_coordinate!r}."
            )
        if self.initial_state not in INITIAL_STATES:
            raise ValueError(
                f"Expected initial_state to be one of {INITIAL_STATES!r}. "
                f"Provided value: {self.initial_state!r}."
            )
        tolerance = self.verify_tolerance_absolute
        if tolerance is not None:
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(float(tolerance))
                or float(tolerance) <= 0.0
            ):
                raise ValueError(
                    "Expected verify_tolerance_absolute to be null or a "
                    f"positive finite number. Provided value: {tolerance!r}."
                )
            object.__setattr__(self, "verify_tolerance_absolute", float(tolerance))
        if not isinstance(self.clamp_to_parameter_bounds, bool):
            raise ValueError(
                "Expected clamp_to_parameter_bounds to be a boolean. Provided "
                f"value: {self.clamp_to_parameter_bounds!r}."
            )
        if self.characterization not in CHARACTERIZATIONS:
            raise ValueError(
                f"Expected characterization to be one of {CHARACTERIZATIONS!r}. "
                f"Provided value: {self.characterization!r}."
            )
        mismatches = {
            name: {"declared": getattr(self, name), "supported": defaults[name]}
            for name in EXPLICIT_LAYER_FIELDS
            if getattr(self, name) != defaults[name]
        }
        if mismatches:
            raise ValueError(
                "Expected the declared device coordinate, programming start, "
                "verify tolerance, readback clamp and characterization to be "
                f"the supported combination for target_mapping "
                f"{self.target_mapping!r}. Provided value: {mismatches!r}."
            )


def backfill_weight_modifier_metadata(value: Any) -> Any:
    """Back-fill saved ``{"type", "parameters"}`` modifier provenance.

    Checkpoint metadata written before the explicit layer fields existed
    stores IBM OM parameters without them; completing them makes it compare
    equal to the current normalized config.  Other values pass through.
    """

    if (
        isinstance(value, Mapping)
        and value.get("type") == "ibm_reram_om_program_verify"
        and isinstance(value.get("parameters"), Mapping)
    ):
        return {**value, "parameters": backfill_config_dict(value["parameters"])}
    return value
