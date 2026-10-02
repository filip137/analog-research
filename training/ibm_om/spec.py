"""IBM OM HWA configuration: the stored flat config and its validation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from training.ibm_om.constants import (
    _CORRUPTION_POLICIES,
    _EXECUTIONS,
    _RAW_ACTIVE_MODES,
    _RESET_RELATIVE_MODES,
    _TARGET_MAPPINGS,
)
from training.ibm_om.topology import _normalize_dual_rail_layouts
from training.ibm_reram_program_verify import OM_PRESET


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
