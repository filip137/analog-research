"""Raw-active IBM OM P&V for one global nine-level cell codebook."""

from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import sys
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from experiments.reram_program_verify.integrity import (
    validate_trajectory_database,
)
from ebl.schema.raw_active_cell import (
    RAW_ACTIVE_CELL_CODEBOOK_BASELINE,
    RAW_ACTIVE_CELL_CODEBOOK_LEVELS,
    RAW_ACTIVE_CELL_CODEBOOK_MAXIMUM,
    RAW_ACTIVE_CELL_CODEBOOK_MINIMUM,
    RAW_ACTIVE_CELL_CODEBOOK_STEP,
    RawActiveCellProgramVerifySpec,
)
from experiments.reram_program_verify.storage import TrajectoryStore
from ebl.schema.base import to_plain_data
from methods.ibm_om.api import (
    IBM_OM_RAW_ACTIVE_A_MIN,
    IBM_OM_RAW_ACTIVE_SCALE,
    IbmReramArrayPopulation,
    sample_om_array_population_layout,
    sample_om_array_population_layout_external,
    save_om_array_population,
)
from methods.ibm_reram_program_verify import (
    ConditioningResult,
    ControllerSettings,
    IbmReramPopulation,
    IbmReramRawActivePlant,
    VerifyObservation,
    derive_seed,
    partition_device_identities,
    run_program_verify,
)

if TYPE_CHECKING:
    from ebl.cli import CharacterizeRequest


_ROOT = Path(__file__).resolve().parents[2]


def _g_from_a(value: torch.Tensor) -> torch.Tensor:
    return (value - IBM_OM_RAW_ACTIVE_A_MIN) / IBM_OM_RAW_ACTIVE_SCALE


def _codebook() -> np.ndarray:
    return np.linspace(
        RAW_ACTIVE_CELL_CODEBOOK_MINIMUM,
        RAW_ACTIVE_CELL_CODEBOOK_MAXIMUM,
        RAW_ACTIVE_CELL_CODEBOOK_LEVELS,
        dtype=np.float64,
    )


def _population_support_audit(
    population: IbmReramArrayPopulation,
    targets: np.ndarray,
) -> dict[str, Any]:
    lower = (
        population.min_bound.detach().cpu().numpy().astype(np.float64)
        - IBM_OM_RAW_ACTIVE_A_MIN
    ) / IBM_OM_RAW_ACTIVE_SCALE
    upper = (
        population.max_bound.detach().cpu().numpy().astype(np.float64)
        - IBM_OM_RAW_ACTIVE_A_MIN
    ) / IBM_OM_RAW_ACTIVE_SCALE
    all_levels = (lower <= targets[0]) & (upper >= targets[-1])
    per_target = []
    for index, target in enumerate(targets.tolist()):
        supported = (lower <= target) & (upper >= target)
        per_target.append(
            {
                "target_index": index,
                "target_g": target,
                "target_a": (
                    IBM_OM_RAW_ACTIVE_A_MIN
                    + IBM_OM_RAW_ACTIVE_SCALE * target
                ),
                "supported_cells": int(supported.sum()),
                "support_fraction": float(supported.mean()),
            }
        )
    return {
        "cells": population.size,
        "all_nine_supported_cells": int(all_levels.sum()),
        "all_nine_support_fraction": float(all_levels.mean()),
        "per_target": per_target,
        "lower_g": {
            "minimum": float(lower.min()),
            "median": float(np.median(lower)),
            "maximum": float(lower.max()),
        },
        "upper_g": {
            "minimum": float(upper.min()),
            "median": float(np.median(upper)),
            "maximum": float(upper.max()),
        },
    }


def _sample_population_for_runtime(
    spec: RawActiveCellProgramVerifySpec,
    *,
    population_path: Path,
    receipt_path: Path,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    raw_sampler = os.environ.get("EBL_AIHWKIT_PYTHON")
    if raw_sampler:
        return sample_om_array_population_layout_external(
            spec.device.binding_keys,
            spec.device.binding_shapes,
            assignment_seed=spec.runtime.assignment_seed,
            corruption_policy=spec.device.corruption_policy,
            aihwkit_python=Path(raw_sampler),
            population_path=population_path,
            receipt_path=receipt_path,
        )
    population = sample_om_array_population_layout(
        spec.device.binding_keys,
        spec.device.binding_shapes,
        assignment_seed=spec.runtime.assignment_seed,
        corruption_policy=spec.device.corruption_policy,
    )
    save_om_array_population(population_path, population)
    receipt = {
        "schema": "ebl.ibm_om.raw_active_array_sampling_receipt",
        "schema_version": 1,
        "backend": "in_process_pinned_aihwkit",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "aihwkit_version": population.aihwkit_version,
        "assignment_seed": population.assignment_seed,
        "corruption_policy": population.corruption_policy,
        "population_fingerprint": population.fingerprint,
        "population_sha256": sha256_file(population_path),
        "cells": population.size,
        "published_corrupt_cells": int(population.published_corrupt.sum()),
        "final_corrupt_cells": int(population.corrupt.sum()),
    }
    atomic_write_json(receipt_path, receipt)
    return population, receipt


def _cell_tile_seeds(population: IbmReramArrayPopulation) -> torch.Tensor:
    pieces = []
    for shape, seed in zip(
        population.binding_shapes,
        population.binding_sampling_seeds,
    ):
        pieces.append(
            torch.full(
                (math.prod(shape),),
                int(seed),
                dtype=torch.int64,
            )
        )
    return torch.cat(pieces)


def _expanded_programming_population(
    population: IbmReramArrayPopulation,
    global_cell_ids: torch.Tensor,
) -> IbmReramPopulation:
    indices = global_cell_ids.to(dtype=torch.long, device="cpu")
    construction_seeds = _cell_tile_seeds(population)[indices]
    return IbmReramPopulation(
        preset="reram_array_om",
        aihwkit_version=population.aihwkit_version,
        nominal_dw_min=population.nominal_dw_min,
        dw_min_std=population.dw_min_std,
        write_noise_std=population.write_noise_std,
        mult_noise=False,
        construction_seeds=construction_seeds.clone(),
        max_bound=population.max_bound[indices].clone(),
        min_bound=population.min_bound[indices].clone(),
        dwmin_up=population.dwmin_up[indices].clone(),
        dwmin_down=population.dwmin_down[indices].clone(),
        reference=population.reference[indices].clone(),
        corrupt=population.corrupt[indices].clone(),
        preset_parameters={
            "source": "ReRamArrayOMPresetDevice",
            "array_population_fingerprint": population.fingerprint,
            "reference_consumed": False,
        },
    )


def _trajectory_seeds(
    *,
    base_seed: int,
    assignment_seed: int,
    global_cell_ids: torch.Tensor,
    repeat_ids: torch.Tensor,
    role: str,
    target_index: int | None = None,
) -> list[int]:
    values = []
    for cell_id, repeat_id in zip(
        global_cell_ids.tolist(), repeat_ids.tolist()
    ):
        parts: tuple[object, ...] = (
            "reram_array_om",
            assignment_seed,
            int(cell_id),
            int(repeat_id),
            role,
        )
        if target_index is not None:
            parts = (*parts, target_index)
        values.append(derive_seed(base_seed, *parts))
    return values


def _select_identities(
    population: IbmReramArrayPopulation,
    *,
    count: int,
    seed: int,
) -> torch.Tensor:
    if count > population.size:
        raise ValueError(
            "Expected sampled_devices not to exceed the array population. "
            f"Provided value: sampled={count}, population={population.size}."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(population.size, generator=generator)[:count]


def _write_selection(
    path: Path,
    *,
    selected_global_cell_ids: torch.Tensor,
    partitions: tuple[str, ...],
    population_fingerprint: str,
    selection_seed: int,
) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray("ebl.ibm_om.raw_active_cell_selection"),
        schema_version=np.asarray(1, dtype=np.int64),
        population_fingerprint=np.asarray(population_fingerprint),
        selection_seed=np.asarray(selection_seed, dtype=np.int64),
        global_cell_ids=selected_global_cell_ids.numpy().astype(np.int64),
        partitions=np.asarray(partitions),
    )


def _condition_population(
    population: IbmReramPopulation,
    *,
    seeds: list[int],
    spec: RawActiveCellProgramVerifySpec,
) -> tuple[ConditioningResult, IbmReramRawActivePlant]:
    plant = IbmReramRawActivePlant(
        population,
        seeds=seeds,
        coordinate_min=IBM_OM_RAW_ACTIVE_A_MIN,
        coordinate_scale=IBM_OM_RAW_ACTIVE_SCALE,
        device=spec.runtime.device,
    )
    plant.reset_logical_zero()
    settings = spec.settings.conditioning
    result = plant.condition_boundary(
        start_protocol="lower_to_target",
        quiet_steps=settings.quiet_steps,
        change_threshold=settings.persistent_change_threshold_raw_a,
        maximum_pulses=settings.maximum_pulses,
    )
    return result, plant


def _run_target(
    *,
    trajectory_store: TrajectoryStore,
    population: IbmReramPopulation,
    local_device_ids: torch.Tensor,
    global_cell_ids: torch.Tensor,
    repeat_ids: torch.Tensor,
    partition_labels: tuple[str, ...],
    repeat_seeds: list[int],
    conditioning_seeds: list[int],
    conditioning: ConditioningResult,
    target_index: int,
    target: float,
    spec: RawActiveCellProgramVerifySpec,
) -> dict[str, int]:
    pulse_seeds = _trajectory_seeds(
        base_seed=spec.runtime.pulse_seed,
        assignment_seed=spec.runtime.assignment_seed,
        global_cell_ids=global_cell_ids,
        repeat_ids=repeat_ids,
        role="target_programming",
        target_index=target_index,
    )
    plant = IbmReramRawActivePlant(
        population,
        seeds=pulse_seeds,
        coordinate_min=IBM_OM_RAW_ACTIVE_A_MIN,
        coordinate_scale=IBM_OM_RAW_ACTIVE_SCALE,
        device=spec.runtime.device,
    )
    plant.persistent.copy_(conditioning.persistent)
    plant.apparent.copy_(conditioning.apparent)

    tolerance = spec.settings.tolerance
    tolerance_ratio = tolerance / (
        population.nominal_dw_min / IBM_OM_RAW_ACTIVE_SCALE
    )
    conditioned_apparent = _g_from_a(conditioning.apparent).tolist()
    conditioned_persistent = _g_from_a(conditioning.persistent).tolist()
    lower_g = _g_from_a(population.min_bound).tolist()
    upper_g = _g_from_a(population.max_bound).tolist()
    conditioning_success = conditioning.success.tolist()
    conditioning_pulses = conditioning.pulse_count.tolist()
    construction_seeds = population.construction_seeds.tolist()
    base_rows = []
    for index in range(population.size):
        base_rows.append(
            {
                "preset": "reram_array_om",
                "corrupt_population": 0,
                "controller": "one_pulse",
                "start_protocol": "lower_to_target",
                "tolerance_ratio": tolerance_ratio,
                "tolerance": tolerance,
                "target_index": target_index,
                "target": target,
                "device_id": int(local_device_ids[index]),
                "repeat_id": int(repeat_ids[index]),
                "partition_name": partition_labels[index],
                "construction_seed": int(construction_seeds[index]),
                "repeat_seed": int(repeat_seeds[index]),
                "conditioning_seed": int(conditioning_seeds[index]),
                "pulse_seed": int(pulse_seeds[index]),
                "controller_seed": None,
                "corrupt": int(population.corrupt[index]),
                "conditioning_success": int(conditioning_success[index]),
                "conditioning_pulses": int(conditioning_pulses[index]),
                "conditioned_apparent": float(conditioned_apparent[index]),
                "conditioned_persistent": float(conditioned_persistent[index]),
                "sampled_lower_persistent": float(lower_g[index]),
                "sampled_upper_persistent": float(upper_g[index]),
            }
        )
    trajectory_ids = trajectory_store.begin_trajectories(base_rows)
    previous_apparent = _g_from_a(conditioning.apparent)

    def observe(observation: VerifyObservation) -> None:
        nonlocal previous_apparent
        if observation.verify_index == 0:
            record_mask = conditioning.success.clone()
        else:
            record_mask = observation.pulse_count > 0
        indices = torch.nonzero(record_mask, as_tuple=False).reshape(-1)
        persistent = _g_from_a(plant.persistent)
        rows = []
        for index in indices.tolist():
            direction = int(observation.direction[index])
            count = int(observation.pulse_count[index])
            rows.append(
                (
                    trajectory_ids[index],
                    observation.verify_index,
                    float(previous_apparent[index]),
                    float(observation.apparent[index]),
                    float(persistent[index]),
                    direction,
                    count,
                    direction * count,
                    int(observation.total_pulses[index]),
                )
            )
        if rows:
            trajectory_store.insert_events(rows)
        previous_apparent = torch.where(
            record_mask,
            observation.apparent,
            previous_apparent,
        )

    targets = torch.full(
        (population.size,),
        float(target),
        dtype=torch.float32,
        device=plant.device,
    )
    result = run_program_verify(
        plant.controller_port(),
        targets=targets,
        tolerance=tolerance,
        maximum_pulses=spec.settings.maximum_program_pulses,
        settings=ControllerSettings(kind="one_pulse"),
        eligible=conditioning.success,
        observer=observe,
    )
    persistent_endpoint = _g_from_a(plant.persistent)
    threshold = spec.settings.conditioning.persistent_change_threshold_raw_a
    saturated = (
        torch.abs(plant.persistent - population.min_bound) <= threshold
    ) | (
        torch.abs(plant.persistent - population.max_bound) <= threshold
    )
    apparent_values = result.apparent_endpoint.tolist()
    persistent_values = persistent_endpoint.tolist()
    final_rows = []
    for index, trajectory_id in enumerate(trajectory_ids):
        apparent = float(apparent_values[index])
        persistent = float(persistent_values[index])
        final_rows.append(
            {
                "trajectory_id": trajectory_id,
                "accepted": int(result.accepted[index]),
                "initialization_failed": int(not conditioning_success[index]),
                "nonfinite": int(result.nonfinite[index]),
                "budget_exhausted": int(result.budget_exhausted[index]),
                "saturated": int(saturated[index]),
                "endpoint_apparent": apparent,
                "endpoint_persistent": persistent,
                "residual_apparent": apparent - target,
                "residual_persistent": persistent - target,
                "set_count": int(result.set_count[index]),
                "reset_count": int(result.reset_count[index]),
                "total_pulses": int(result.total_pulses[index]),
                "verify_count": int(result.verify_count[index]),
                "reversals": int(result.reversals[index]),
            }
        )
    trajectory_store.finish_trajectories(final_rows)
    return {
        "trajectories": len(final_rows),
        "accepted": int(result.accepted.sum()),
        "conditioning_failed": int((~conditioning.success).sum()),
        "budget_exhausted": int(result.budget_exhausted.sum()),
        "nonfinite": int(result.nonfinite.sum()),
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _analyze_database(
    database_path: Path,
    *,
    targets: np.ndarray,
    tolerance: float,
    support_audit: dict[str, Any],
    assignment_seed: int,
) -> dict[str, Any]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT target_index, target, device_id, repeat_id, partition_name,
                   sampled_lower_persistent, sampled_upper_persistent,
                   accepted, initialization_failed, nonfinite,
                   budget_exhausted, saturated, endpoint_apparent,
                   endpoint_persistent, set_count, reset_count,
                   total_pulses, verify_count, reversals
            FROM trajectories
            ORDER BY target_index, device_id, repeat_id
            """
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise RuntimeError("Expected raw-active P&V trajectories for analysis.")

    target_index = np.asarray([row[0] for row in rows], dtype=np.int64)
    target = np.asarray([row[1] for row in rows], dtype=np.float64)
    device_id = np.asarray([row[2] for row in rows], dtype=np.int64)
    repeat_id = np.asarray([row[3] for row in rows], dtype=np.int64)
    partition = np.asarray([row[4] for row in rows])
    lower = np.asarray([row[5] for row in rows], dtype=np.float64)
    upper = np.asarray([row[6] for row in rows], dtype=np.float64)
    accepted = np.asarray([row[7] for row in rows], dtype=np.bool_)
    initialization_failed = np.asarray([row[8] for row in rows], dtype=np.bool_)
    nonfinite = np.asarray([row[9] for row in rows], dtype=np.bool_)
    budget_exhausted = np.asarray([row[10] for row in rows], dtype=np.bool_)
    saturated = np.asarray([row[11] for row in rows], dtype=np.bool_)
    apparent = np.asarray([row[12] for row in rows], dtype=np.float64)
    persistent = np.asarray([row[13] for row in rows], dtype=np.float64)
    set_count = np.asarray([row[14] for row in rows], dtype=np.int64)
    reset_count = np.asarray([row[15] for row in rows], dtype=np.int64)
    total_pulses = np.asarray([row[16] for row in rows], dtype=np.int64)
    verify_count = np.asarray([row[17] for row in rows], dtype=np.int64)
    reversals = np.asarray([row[18] for row in rows], dtype=np.int64)

    exact_support = (lower <= target) & (upper >= target)
    all_nine_support = (lower <= targets[0]) & (upper >= targets[-1])
    verify_window_intersects_support = (
        (target + tolerance >= lower) & (target - tolerance <= upper)
    )
    persistent_within_tolerance = np.abs(persistent - target) <= tolerance
    distances_apparent = np.abs(apparent[:, None] - targets[None, :])
    distances_persistent = np.abs(persistent[:, None] - targets[None, :])
    finite_apparent = np.isfinite(apparent)
    finite_persistent = np.isfinite(persistent)
    nearest_apparent = np.argmin(
        np.where(np.isfinite(distances_apparent), distances_apparent, np.inf),
        axis=1,
    )
    nearest_persistent = np.argmin(
        np.where(np.isfinite(distances_persistent), distances_persistent, np.inf),
        axis=1,
    )
    nearest_apparent_correct = finite_apparent & (nearest_apparent == target_index)
    nearest_persistent_correct = finite_persistent & (
        nearest_persistent == target_index
    )
    apparent_window_memberships = (
        distances_apparent <= tolerance
    ).sum(axis=1)
    persistent_window_memberships = (
        distances_persistent <= tolerance
    ).sum(axis=1)
    half_spacing = 0.5 * (targets[1] - targets[0])
    persistent_within_half_spacing = (
        np.abs(persistent - target) <= half_spacing
    )

    def summarize(selection: np.ndarray) -> dict[str, Any]:
        count = int(selection.sum())
        exact = selection & exact_support
        accepted_exact = exact & accepted
        return {
            "trajectories": count,
            "exact_target_support": int(exact.sum()),
            "exact_target_support_fraction": _fraction(int(exact.sum()), count),
            "all_nine_support": int((selection & all_nine_support).sum()),
            "verify_window_intersects_support": int(
                (selection & verify_window_intersects_support).sum()
            ),
            "accepted": int((selection & accepted).sum()),
            "accepted_fraction": _fraction(int((selection & accepted).sum()), count),
            "accepted_exact_support": int(accepted_exact.sum()),
            "accepted_exact_support_fraction": _fraction(
                int(accepted_exact.sum()), int(exact.sum())
            ),
            "accepted_outside_exact_support": int(
                (selection & accepted & ~exact_support).sum()
            ),
            "persistent_within_tolerance_after_acceptance": int(
                (accepted_exact & persistent_within_tolerance).sum()
            ),
            "persistent_within_tolerance_fraction_after_acceptance": _fraction(
                int((accepted_exact & persistent_within_tolerance).sum()),
                int(accepted_exact.sum()),
            ),
            "persistent_nearest_code_correct_after_acceptance": int(
                (accepted_exact & nearest_persistent_correct).sum()
            ),
            "persistent_nearest_code_correct_fraction_after_acceptance": _fraction(
                int((accepted_exact & nearest_persistent_correct).sum()),
                int(accepted_exact.sum()),
            ),
            "persistent_within_half_spacing_after_acceptance": int(
                (accepted_exact & persistent_within_half_spacing).sum()
            ),
            "persistent_within_half_spacing_fraction_after_acceptance": _fraction(
                int((accepted_exact & persistent_within_half_spacing).sum()),
                int(accepted_exact.sum()),
            ),
            "apparent_acceptance_ambiguous_between_codes": int(
                (accepted_exact & (apparent_window_memberships > 1)).sum()
            ),
            "apparent_acceptance_ambiguity_fraction": _fraction(
                int((accepted_exact & (apparent_window_memberships > 1)).sum()),
                int(accepted_exact.sum()),
            ),
            "persistent_endpoint_inside_multiple_acceptance_windows": int(
                (accepted_exact & (persistent_window_memberships > 1)).sum()
            ),
            "initialization_failed": int((selection & initialization_failed).sum()),
            "budget_exhausted": int((selection & budget_exhausted).sum()),
            "nonfinite": int((selection & nonfinite).sum()),
            "saturated": int((selection & saturated).sum()),
            "mean_total_pulses": (
                float(total_pulses[selection].mean()) if count else None
            ),
            "mean_set_pulses": (
                float(set_count[selection].mean()) if count else None
            ),
            "mean_reset_pulses": (
                float(reset_count[selection].mean()) if count else None
            ),
            "mean_verifies": (
                float(verify_count[selection].mean()) if count else None
            ),
            "mean_reversals": (
                float(reversals[selection].mean()) if count else None
            ),
        }

    target_records = []
    validation = partition == "validation"
    confusion = np.zeros(
        (len(targets), len(targets)), dtype=np.int64
    )
    for index, requested in enumerate(targets.tolist()):
        selected = validation & (target_index == index)
        accepted_exact = selected & accepted & exact_support
        for predicted in nearest_persistent[accepted_exact].tolist():
            confusion[index, int(predicted)] += 1
        endpoints = persistent[accepted_exact]
        target_records.append(
            {
                "target_index": index,
                "target_g": requested,
                "target_a": (
                    IBM_OM_RAW_ACTIVE_A_MIN
                    + IBM_OM_RAW_ACTIVE_SCALE * requested
                ),
                "validation": summarize(selected),
                "accepted_exact_persistent_endpoint": (
                    {
                        "count": int(endpoints.size),
                        "mean": float(endpoints.mean()),
                        "standard_deviation": float(endpoints.std(ddof=1))
                        if endpoints.size > 1
                        else 0.0,
                        "p10": float(np.quantile(endpoints, 0.1)),
                        "median": float(np.median(endpoints)),
                        "p90": float(np.quantile(endpoints, 0.9)),
                    }
                    if endpoints.size
                    else None
                ),
            }
        )

    trajectories_per_target = len(rows) // len(targets)
    expected_shape = (len(targets), trajectories_per_target)
    if target_index.shape != (len(rows),) or not np.array_equal(
        target_index.reshape(expected_shape)[:, 0],
        np.arange(len(targets)),
    ):
        raise RuntimeError("Expected a rectangular target-by-trajectory matrix.")
    matrix_validation = validation.reshape(expected_shape)[0]
    matrix_accepted = accepted.reshape(expected_shape)
    matrix_exact = exact_support.reshape(expected_shape)
    matrix_nearest = nearest_persistent_correct.reshape(expected_shape)
    matrix_half = persistent_within_half_spacing.reshape(expected_shape)
    identity_repeat_count = int(matrix_validation.sum())
    all_targets_accepted_exact = (
        matrix_accepted & matrix_exact
    ).all(axis=0) & matrix_validation
    all_targets_nearest = (
        matrix_accepted & matrix_exact & matrix_nearest
    ).all(axis=0) & matrix_validation
    all_targets_half = (
        matrix_accepted & matrix_exact & matrix_half
    ).all(axis=0) & matrix_validation

    return {
        "schema": "ebl.ibm_om.raw_active_common_cell_9.analysis",
        "schema_version": 1,
        "assignment_seed": assignment_seed,
        "coordinate": {
            "definition": "g=(a-A_min)/S",
            "a_min": IBM_OM_RAW_ACTIVE_A_MIN,
            "scale": IBM_OM_RAW_ACTIVE_SCALE,
            "reference_consumed": False,
        },
        "codebook": {
            "definition": "g_k=B+(k-4)*Delta_g, k=0..8",
            "baseline_g": RAW_ACTIVE_CELL_CODEBOOK_BASELINE,
            "step_g": RAW_ACTIVE_CELL_CODEBOOK_STEP,
            "targets_g": targets.tolist(),
            "levels": len(targets),
            "tolerance_g": tolerance,
            "half_spacing_g": half_spacing,
            "two_tolerance_g": 2.0 * tolerance,
            "acceptance_windows_overlap": bool(2.0 * tolerance > targets[1] - targets[0]),
        },
        "population_support_audit": support_audit,
        "all_partitions": summarize(np.ones(len(rows), dtype=np.bool_)),
        "validation": summarize(validation),
        "validation_complete_identity_repeats": {
            "count": identity_repeat_count,
            "all_nine_accepted_with_exact_support": int(
                all_targets_accepted_exact.sum()
            ),
            "all_nine_accepted_with_exact_support_fraction": _fraction(
                int(all_targets_accepted_exact.sum()), identity_repeat_count
            ),
            "all_nine_persistent_nearest_code_correct": int(
                all_targets_nearest.sum()
            ),
            "all_nine_persistent_nearest_code_correct_fraction": _fraction(
                int(all_targets_nearest.sum()), identity_repeat_count
            ),
            "all_nine_persistent_within_half_spacing": int(
                all_targets_half.sum()
            ),
            "all_nine_persistent_within_half_spacing_fraction": _fraction(
                int(all_targets_half.sum()), identity_repeat_count
            ),
        },
        "persistent_nearest_code_confusion_validation_accepted_exact_support": (
            confusion.tolist()
        ),
        "targets": target_records,
        "identity_columns": {
            "device_id_count": int(np.unique(device_id).size),
            "repeat_id_count": int(np.unique(repeat_id).size),
        },
        "apparent_nearest_code_correct_validation": int(
            (validation & accepted & exact_support & nearest_apparent_correct).sum()
        ),
    }


def _write_report(path: Path, analysis: dict[str, Any]) -> None:
    validation = analysis["validation"]
    complete = analysis["validation_complete_identity_repeats"]
    codebook = analysis["codebook"]
    support = analysis["population_support_audit"]

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.3f}%"

    lines = [
        "# Raw-active common-cell nine-level program-and-verify",
        "",
        f"- Assignment seed: `{analysis['assignment_seed']}`",
        "- Physical coordinate: one active state `a_i`; AIHWKit reference excluded.",
        f"- Common baseline `B`: `{codebook['baseline_g']:.15f}`",
        f"- Common cell step `Delta_g`: `{codebook['step_g']:.15f}`",
        f"- Controller tolerance: `{codebook['tolerance_g']:.15f}`",
        f"- Full-array all-nine support: {percent(support['all_nine_support_fraction'])}",
        "",
        "## Primary diagnostics",
        "",
        f"- Exact-support apparent acceptance: {percent(validation['accepted_exact_support_fraction'])}",
        "- Persistent nearest-code correctness after acceptance: "
        f"{percent(validation['persistent_nearest_code_correct_fraction_after_acceptance'])}",
        "- Persistent within half a code step after acceptance: "
        f"{percent(validation['persistent_within_half_spacing_fraction_after_acceptance'])}",
        "- Apparent acceptances compatible with multiple codes: "
        f"{percent(validation['apparent_acceptance_ambiguity_fraction'])}",
        "- Complete identity/repeats with all nine nearest-code correct: "
        f"{percent(complete['all_nine_persistent_nearest_code_correct_fraction'])}",
        "",
        "The canonical verify windows overlap because `2*tau > Delta_g`. "
        "Apparent acceptance is therefore not interpreted as nine-state "
        "distinguishability; nearest-code and half-spacing metrics are reported separately.",
        "",
        "## Target-wise validation",
        "",
        "| k | target g | full-array support | accepted / exact support | persistent nearest-code |",
        "|---:|---:|---:|---:|---:|",
    ]
    population_targets = support["per_target"]
    for record in analysis["targets"]:
        index = record["target_index"]
        target_validation = record["validation"]
        lines.append(
            f"| {index} | {record['target_g']:.9f} | "
            f"{percent(population_targets[index]['support_fraction'])} | "
            f"{percent(target_validation['accepted_exact_support_fraction'])} | "
            f"{percent(target_validation['persistent_nearest_code_correct_fraction_after_acceptance'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, analysis: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = np.asarray(analysis["codebook"]["targets_g"], dtype=np.float64)
    medians = []
    p10 = []
    p90 = []
    for record in analysis["targets"]:
        endpoint = record["accepted_exact_persistent_endpoint"]
        medians.append(np.nan if endpoint is None else endpoint["median"])
        p10.append(np.nan if endpoint is None else endpoint["p10"])
        p90.append(np.nan if endpoint is None else endpoint["p90"])
    confusion = np.asarray(
        analysis[
            "persistent_nearest_code_confusion_validation_accepted_exact_support"
        ],
        dtype=np.float64,
    )
    row_sum = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_sum,
        out=np.zeros_like(confusion),
        where=row_sum > 0,
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.8), constrained_layout=True)
    x = np.arange(len(targets))
    axes[0].plot(x, targets, "o--", color="#555555", label="requested")
    axes[0].errorbar(
        x,
        medians,
        yerr=[np.asarray(medians) - np.asarray(p10), np.asarray(p90) - np.asarray(medians)],
        fmt="o",
        color="#1775b8",
        capsize=3,
        label="persistent median (p10-p90)",
    )
    axes[0].set_xlabel("Requested cell code k")
    axes[0].set_ylabel("Array-wide g coordinate")
    axes[0].set_title("Persistent programmed endpoints")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    image = axes[1].imshow(
        normalized,
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        origin="lower",
        aspect="equal",
    )
    axes[1].set_xlabel("Nearest persistent code")
    axes[1].set_ylabel("Requested code")
    axes[1].set_title("Validation confusion")
    axes[1].set_xticks(x)
    axes[1].set_yticks(x)
    figure.colorbar(image, ax=axes[1], label="row fraction", shrink=0.85)
    figure.savefig(path, dpi=240, facecolor="white")
    plt.close(figure)


def run_raw_active_cell_characterize(request: "CharacterizeRequest") -> int:
    spec = request.spec
    if not isinstance(spec, RawActiveCellProgramVerifySpec):
        raise TypeError(
            "Expected raw-active cell characterize to resolve "
            "RawActiveCellProgramVerifySpec. Provided value: "
            f"{type(spec).__name__}."
        )
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        resume_capability="unsupported",
    )
    try:
        settings = spec.settings
        artifacts_dir = store.run_dir / "artifacts"
        population_path = artifacts_dir / "device_population.npz"
        population_receipt_path = artifacts_dir / "population_sampling_receipt.json"
        selection_path = artifacts_dir / "selected_identities.npz"
        codebook_path = artifacts_dir / "codebook_and_support.json"
        database_path = artifacts_dir / "trajectories.sqlite3"
        integrity_path = artifacts_dir / "integrity_report.json"
        analysis_path = artifacts_dir / "analysis.json"
        report_path = artifacts_dir / "report.md"
        plot_path = artifacts_dir / "programming.png"

        population, sampling_receipt = _sample_population_for_runtime(
            spec,
            population_path=population_path,
            receipt_path=population_receipt_path,
        )
        if population.aihwkit_version != spec.runtime.required_aihwkit_version:
            raise RuntimeError("Expected the sampled AIHWKit version to match config.")
        targets = _codebook()
        support_audit = _population_support_audit(population, targets)
        codebook_bundle = {
            "schema": "ebl.ibm_om.raw_active_common_cell_9.codebook",
            "schema_version": 1,
            "population_fingerprint": population.fingerprint,
            "coordinate": {
                "definition": "g=(a-A_min)/S",
                "a_min": IBM_OM_RAW_ACTIVE_A_MIN,
                "scale": IBM_OM_RAW_ACTIVE_SCALE,
                "reference_consumed": False,
            },
            "definition": "g_k=B+(k-4)*Delta_g, k=0..8",
            "baseline_g": RAW_ACTIVE_CELL_CODEBOOK_BASELINE,
            "step_g": RAW_ACTIVE_CELL_CODEBOOK_STEP,
            "targets_g": targets.tolist(),
            "tolerance_g": settings.tolerance,
            "maximum_program_pulses": settings.maximum_program_pulses,
            "support_audit": support_audit,
        }
        atomic_write_json(codebook_path, codebook_bundle)
        store.append_metric(
            {
                "mode": "characterize",
                "phase": "population_audited",
                "assignment_seed": spec.runtime.assignment_seed,
                "population_fingerprint": population.fingerprint,
                "all_nine_support_fraction": support_audit[
                    "all_nine_support_fraction"
                ],
            }
        )

        selected = _select_identities(
            population,
            count=settings.sampled_devices,
            seed=spec.runtime.selection_seed,
        )
        partitions = partition_device_identities(
            settings.sampled_devices,
            seed=spec.runtime.partition_seed,
            fractions=(
                settings.partitions.calibration,
                settings.partitions.fit,
                settings.partitions.validation,
            ),
        )
        _write_selection(
            selection_path,
            selected_global_cell_ids=selected,
            partitions=partitions,
            population_fingerprint=population.fingerprint,
            selection_seed=spec.runtime.selection_seed,
        )
        local_device_ids = torch.arange(settings.sampled_devices).repeat_interleave(
            settings.repeats_per_device
        )
        repeat_ids = torch.arange(settings.repeats_per_device).repeat(
            settings.sampled_devices
        )
        global_cell_ids = selected[local_device_ids]
        expanded = _expanded_programming_population(population, global_cell_ids)
        expanded_partition_labels = tuple(
            partitions[int(index)] for index in local_device_ids.tolist()
        )
        repeat_seeds = _trajectory_seeds(
            base_seed=spec.runtime.repeat_seed,
            assignment_seed=spec.runtime.assignment_seed,
            global_cell_ids=global_cell_ids,
            repeat_ids=repeat_ids,
            role="repeat",
        )
        conditioning_seeds = _trajectory_seeds(
            base_seed=spec.runtime.conditioning_seed,
            assignment_seed=spec.runtime.assignment_seed,
            global_cell_ids=global_cell_ids,
            repeat_ids=repeat_ids,
            role="lower_boundary_conditioning",
        )
        conditioning, _conditioning_plant = _condition_population(
            expanded,
            seeds=conditioning_seeds,
            spec=spec,
        )
        totals = Counter()
        metadata = {
            "execution_profile": settings.profile,
            "execution_device": spec.runtime.device,
            "preset": "reram_array_om",
            "coordinate": "g=(a-A_min)/S",
            "reference_consumed": False,
            "controller": "one_pulse",
            "start_protocol": "lower_to_target",
            "conditioning_scope": "per_start",
            "conditioning_state_reuse": (
                "one immutable independently seeded conditioned state per "
                "cell identity/repeat, cloned across all nine targets"
            ),
            "population_sampling": sampling_receipt,
            "seeds": to_plain_data(spec.runtime),
            "selected_identity_artifact": selection_path.name,
            "codebook": codebook_bundle,
        }
        with TrajectoryStore(database_path) as trajectory_store:
            trajectory_store.insert_metadata(
                {
                    "characterization": json.dumps(
                        metadata, allow_nan=False, sort_keys=True
                    ),
                    "resolved_spec": json.dumps(
                        to_plain_data(spec), allow_nan=False, sort_keys=True
                    ),
                }
            )
            for target_index, target in enumerate(targets.tolist()):
                summary = _run_target(
                    trajectory_store=trajectory_store,
                    population=expanded,
                    local_device_ids=local_device_ids,
                    global_cell_ids=global_cell_ids,
                    repeat_ids=repeat_ids,
                    partition_labels=expanded_partition_labels,
                    repeat_seeds=repeat_seeds,
                    conditioning_seeds=conditioning_seeds,
                    conditioning=conditioning,
                    target_index=target_index,
                    target=target,
                    spec=spec,
                )
                totals.update(summary)
                store.append_metric(
                    {
                        "mode": "characterize",
                        "phase": "programming",
                        "target_index": target_index,
                        "target_g": target,
                        "completed_targets": target_index + 1,
                        "total_targets": len(targets),
                    }
                )

        tolerance_ratio = settings.tolerance / (
            population.nominal_dw_min / IBM_OM_RAW_ACTIVE_SCALE
        )
        expected_per_condition = (
            settings.sampled_devices * settings.repeats_per_device
        )
        validate_trajectory_database(
            database_path,
            output_path=integrity_path,
            expected_trajectory_count=expected_per_condition * len(targets),
            expected_condition_count=len(targets),
            trajectories_per_condition=expected_per_condition,
            maximum_program_pulses=settings.maximum_program_pulses,
            controllers=("one_pulse",),
            start_protocols=("lower_to_target",),
            tolerance_ratios=(tolerance_ratio,),
            target_points=len(targets),
            repeats_per_device=settings.repeats_per_device,
            expected_partition_device_counts=dict(Counter(partitions)),
            conditioning_scope="per_start",
        )
        analysis = _analyze_database(
            database_path,
            targets=targets,
            tolerance=settings.tolerance,
            support_audit=support_audit,
            assignment_seed=spec.runtime.assignment_seed,
        )
        atomic_write_json(analysis_path, analysis)
        _write_report(report_path, analysis)
        _write_plot(plot_path, analysis)

        artifacts = [
            store.artifact_record(population_path, kind="sampled_device_population"),
            store.artifact_record(
                population_receipt_path, kind="population_sampling_receipt"
            ),
            store.artifact_record(selection_path, kind="selected_device_identities"),
            store.artifact_record(codebook_path, kind="physical_cell_codebook"),
            store.artifact_record(database_path, kind="pulse_trajectory_database"),
            store.artifact_record(integrity_path, kind="trajectory_integrity_report"),
            store.artifact_record(analysis_path, kind="program_verify_analysis"),
            store.artifact_record(report_path, kind="characterization_report"),
            store.artifact_record(plot_path, kind="characterization_plot"),
        ]
        validation = analysis["validation"]
        store.complete(
            metrics={
                "evidence_class": (
                    "operational_smoke"
                    if settings.profile == "smoke"
                    else "model_based_aihwkit_preset"
                ),
                "execution_profile": settings.profile,
                "assignment_seed": spec.runtime.assignment_seed,
                "population_fingerprint": population.fingerprint,
                "population_cells": population.size,
                "sampled_devices": settings.sampled_devices,
                "repeats_per_device": settings.repeats_per_device,
                "trajectories": totals["trajectories"],
                "all_nine_population_support_fraction": support_audit[
                    "all_nine_support_fraction"
                ],
                "validation_exact_support_acceptance_fraction": validation[
                    "accepted_exact_support_fraction"
                ],
                "validation_persistent_nearest_code_correct_fraction": validation[
                    "persistent_nearest_code_correct_fraction_after_acceptance"
                ],
                "validation_apparent_acceptance_ambiguity_fraction": validation[
                    "apparent_acceptance_ambiguity_fraction"
                ],
                "conditioning_failed": totals["conditioning_failed"],
                "budget_exhausted": totals["budget_exhausted"],
                "nonfinite": totals["nonfinite"],
                "reference_consumed": False,
                "hwa_or_on_chip_training_performed": False,
            },
            artifacts=artifacts,
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_raw_active_cell_characterize"]
