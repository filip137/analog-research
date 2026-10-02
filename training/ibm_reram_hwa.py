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
from training.ibm_om.mappings.raw_active import _raw_active_structural_cell_mask
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
