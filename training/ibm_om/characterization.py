"""Controller-observed IBM OM characterization performed before mapping.

RESET-relative commissioning pulses every cell to RESET, reads it several
times and records one guarded baseline per dual-rail quad.  It sees only
controller reads, never the hidden device bounds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch

from training.ibm_om.plants import _ArrayPlant
from training.ibm_om.population import (
    IbmReramArrayPopulation,
    _tensor_sha256,
    _tensor_summary,
)
from training.ibm_om.topology import _normalize_dual_rail_layouts, _quad_axes
from training.ibm_reram_program_verify import derive_seed


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
