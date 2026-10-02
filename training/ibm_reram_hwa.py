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
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import torch

from experiments.artifacts import sha256_file
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
# Moved to training.ibm_om.characterization; facade re-export keeps historical import paths.
from training.ibm_om.characterization import (
    IbmReramResetCommissioning,
    commission_ibm_reram_reset_relative_baselines,
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
# Moved to training.ibm_om.mappings; facade re-export keeps historical import paths.
from training.ibm_om.mappings import (
    map_ibm_reram_array_targets,
    validate_ibm_reram_target_mapping_preflight,
)
# facade re-export (used by raw-active initialization and analysis tools)
from training.ibm_om.mappings.raw_active import _raw_active_structural_cell_mask
from training.ibm_om.mappings import run_mapping
from training.ibm_om.mappings.base import MappingResult
from training.ibm_om.programming import PROGRAMMERS
from training.ibm_om.programming.base import ProgrammingContext, ProgramRequest
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
        # Legacy derivations, replaced by declared config fields later: the
        # raw-active mapping implies the raw-active coordinate, and the
        # corruption policy selects the calibrated programming branch.
        self._coordinate = (
            RAW_ACTIVE
            if config.target_mapping == "raw_active_p90_quad"
            else LOGICAL
        )
        self._calibration_branch = (
            "continuous"
            if config.corruption_policy == "counterfactual_repaired"
            else "published_corruption"
        )
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
        list[tuple[int, tuple[int, ...]]],
        MappingResult,
    ]:
        global_targets, layout = self._global_targets()
        population = self._population_on(global_targets.device)
        mapping = run_mapping(
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
        return global_targets, layout, mapping

    def _programming_context(self, device: torch.device) -> ProgrammingContext:
        return ProgrammingContext(
            population=self._population_on(device),
            coordinate=self._coordinate,
            device_model=self._artifact,
            device_model_sha256=self._artifact_sha256,
            calibration_branch=self._calibration_branch,
            controller=self._config.controller,
            tolerance_step_ratio=self._config.tolerance_step_ratio,
            maximum_program_pulses=self._config.maximum_program_pulses,
            target_out_of_support=self._config.target_out_of_support,
            endpoint_policy=self._config.endpoint_policy,
        )

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
                    global_targets, layout, mapping = self._targets()
                    targets = mapping.targets
                    mapping_report = mapping.report
                    generator = self._generator(stream, targets.device)
                    endpoint, report, deployment = PROGRAMMERS[
                        self._config.execution
                    ](
                        ProgramRequest(
                            targets=targets,
                            program_mask=mapping.program_mask,
                            out_of_support=mapping.out_of_support,
                        ),
                        self._programming_context(targets.device),
                        generator=generator,
                    )
                    deployment["config"] = asdict(self._config)
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
