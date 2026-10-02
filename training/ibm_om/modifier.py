"""DRN parameter modifier that swaps IBM OM device endpoints into the network.

Per context it maps the clean master conductances, programs the fixed cell
assignment with the configured execution, writes the endpoints into the DRN
for the forward/gradient pass and restores the clean values afterwards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import torch

from experiments.artifacts import sha256_file
from model.resistive.builders import ParameterBinding
from training.ibm_om.characterization import (
    IbmReramResetCommissioning,
    commission_ibm_reram_reset_relative_baselines,
)
from training.ibm_om.constants import (
    _EVALUATION_STREAM_XOR,
    _STATE_VERSION,
    _STREAMS,
    IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
)
from training.ibm_om.coordinates import LOGICAL, RAW_ACTIVE
from training.ibm_om.mappings import (
    map_ibm_reram_array_targets,
    run_mapping,
    validate_ibm_reram_target_mapping_preflight,
)
from training.ibm_om.mappings.base import MappingResult
from training.ibm_om.population import IbmReramArrayPopulation
from training.ibm_om.programming import PROGRAMMERS
from training.ibm_om.programming.base import ProgrammingContext, ProgramRequest
from training.ibm_om.readback import (
    fractions_from_bindings,
    restore,
    write_endpoints,
)
from training.ibm_om.sampling import (
    _artifact,
    sample_om_array_population,
    sample_om_array_population_external,
)
from training.ibm_om.spec import IbmReramHwaConfig
from training.ibm_om.topology import (
    _canonical_differential_pair_layout,
    _validate_differential_pair_bindings,
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
        self._clamp_to_parameter_bounds = (
            config.target_mapping != "raw_active_p90_quad"
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
        return fractions_from_bindings(
            self._bindings,
            conductance_min=self._conductance_min,
            conductance_max=self._conductance_max,
        )

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
        return write_endpoints(
            self._bindings,
            endpoint,
            layout,
            conductance_min=self._conductance_min,
            conductance_max=self._conductance_max,
            clamp_to_parameter_bounds=self._clamp_to_parameter_bounds,
        )

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
            restore(snapshots)
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
