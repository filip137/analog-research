"""Shared crossbar sampling, programming, evaluation and off-chip adaptation.

Kept independent of experiment entry points and equilibrium solvers.
"""

from __future__ import annotations
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Iterable, Mapping
import torch
import torch.nn.functional as F
from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_shared import limited
from experiments.schema import to_plain_data
from training.parameters import ParameterBinding
from training.parameters import DenseWeight
from training.checkpoint import atomic_torch_save
from training.ibm_om_standard_crossbar import (
    CROSSBAR_TRAJECTORY_SEED_DERIVATION,
    CrossbarTileSpec,
    IbmOmEffectiveCrossbarPlant,
    apply_population_bound_policy,
    crossbar_trajectory_seeds,
    layer_cell_slices,
    project_to_nearest_effective_code_device,
    standard_crossbar_logits,
    tensor_sha256,
    validate_population_layout,
)
from training.ibm_reram_endpoint_model import IbmReramAcceptedEndpointModel
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    sample_om_array_population_external,
    save_om_array_population,
)
from training.ibm_reram_program_verify import (
    ControllerSettings,
    OM_PRESET,
    PUBLISHED_CORRUPT_PROBABILITY,
    PUBLISHED_CORRUPT_RANGE,
    derive_seed,
    run_program_verify,
)
from training.star_targets import build_calibration_binding


def _sampling_bindings(layout: Iterable[CrossbarTileSpec]) -> tuple[ParameterBinding, ...]:
    bindings = []
    for tile in layout:
        parameter = DenseWeight(
            (tile.shape[0],),
            (tile.shape[1],),
            gain=1.0,
            device="cpu",
            clamp=False,
        )
        bindings.append(
            ParameterBinding(
                key=tile.key,
                parameter=parameter,
                group="crossbar",
                role="dense_weight",
                trainable=False,
                checkpointed=False,
            )
        )
    return tuple(bindings)


def _sample_population(
    *,
    layout: tuple[CrossbarTileSpec, ...],
    assignment_seed: int,
    corruption_policy: str,
    sampler: Path,
    artifact_root: Path,
    role: str,
    bound_policy: str,
    required_aihwkit_version: str,
) -> tuple[
    IbmReramArrayPopulation,
    dict[str, Any],
    tuple[tuple[Path, str], ...],
]:
    sampled_path = artifact_root / f"{role}_sampled_population.npz"
    sampled_receipt_path = artifact_root / f"{role}_sampled_population.receipt.json"
    sampled, sampling_receipt = sample_om_array_population_external(
        _sampling_bindings(layout),
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        aihwkit_python=sampler,
        population_path=sampled_path,
        receipt_path=sampled_receipt_path,
    )
    population, bound_report = apply_population_bound_policy(
        sampled,
        policy=bound_policy,
    )
    if population.aihwkit_version != required_aihwkit_version:
        raise RuntimeError(
            "Expected the sampled IBM OM population to match the pinned "
            "AIHWKit version."
        )
    validate_population_layout(population, layout)
    artifacts: tuple[tuple[Path, str], ...] = (
        (sampled_path, "ibm_om_sampled_population"),
        (sampled_receipt_path, "population_sampling_receipt"),
    )
    effective_artifact = None
    if population.fingerprint != sampled.fingerprint:
        effective_path = artifact_root / f"{role}_effective_population.npz"
        effective_receipt_path = artifact_root / f"{role}_effective_population.receipt.json"
        save_om_array_population(effective_path, population)
        effective_receipt = {
            "schema": "ebl.ibm_om_crossbar_effective_population_receipt",
            "schema_version": 1,
            "artifact": effective_path.name,
            "artifact_sha256": sha256_file(effective_path),
            "population_fingerprint": population.fingerprint,
            "source_population_fingerprint": sampled.fingerprint,
            "source_population_sha256": sha256_file(sampled_path),
            "sampling_receipt_sha256": sha256_file(sampled_receipt_path),
            "bound_treatment": bound_report,
        }
        atomic_write_json(effective_receipt_path, effective_receipt)
        artifacts += (
            (effective_path, "ibm_om_effective_population"),
            (effective_receipt_path, "effective_population_receipt"),
        )
        effective_artifact = effective_receipt
    provenance = {
        "sampling": sampling_receipt,
        "bound_treatment": bound_report,
        "effective_artifact": effective_artifact,
    }
    return population, provenance, artifacts


def _prediction_digest(predictions: list[torch.Tensor]) -> str:
    joined = torch.cat(predictions).to(dtype=torch.int64, device="cpu").contiguous()
    return sha256(joined.numpy().tobytes()).hexdigest()


def _evaluate(
    *,
    effective_state: torch.Tensor,
    digital_scales: tuple[float, float],
    layout: tuple[CrossbarTileSpec, ...],
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    device: torch.device,
    maximum_batches: int | None,
    sample_limit: int | None,
) -> dict[str, Any]:
    totals = {
        "student_correct": 0,
        "teacher_correct": 0,
        "agreement": 0,
        "kl": 0.0,
        "cross_entropy": 0.0,
        "student_score_squared": 0.0,
        "teacher_score_squared": 0.0,
    }
    examples = 0
    student_predictions: list[torch.Tensor] = []
    teacher_predictions: list[torch.Tensor] = []
    state = effective_state.detach().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for inputs, labels in limited(loader, maximum_batches):
            if sample_limit is not None:
                remaining = sample_limit - examples
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                labels = labels[:remaining]
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            teacher_logits = teacher.logits(inputs)
            student_logits = standard_crossbar_logits(
                inputs,
                state,
                layout,
                digital_scales=digital_scales,
            )
            teacher_log_probability = F.log_softmax(teacher_logits, dim=1)
            teacher_probability = teacher_log_probability.exp()
            student_log_probability = F.log_softmax(student_logits, dim=1)
            student_prediction = student_logits.argmax(dim=1)
            teacher_prediction = teacher_logits.argmax(dim=1)
            totals["student_correct"] += int(student_prediction.eq(labels).sum().item())
            totals["teacher_correct"] += int(teacher_prediction.eq(labels).sum().item())
            totals["agreement"] += int(student_prediction.eq(teacher_prediction).sum().item())
            totals["kl"] += float(
                (teacher_probability * (teacher_log_probability - student_log_probability)).sum().item()
            )
            totals["cross_entropy"] += float(
                F.cross_entropy(student_logits, labels, reduction="sum").item()
            )
            totals["student_score_squared"] += float(student_logits.square().sum().item())
            totals["teacher_score_squared"] += float(teacher_logits.square().sum().item())
            examples += int(labels.shape[0])
            student_predictions.append(student_prediction.detach().cpu())
            teacher_predictions.append(teacher_prediction.detach().cpu())
    if examples == 0:
        raise ValueError("Expected evaluation to process at least one example.")
    score_count = examples * 10
    return {
        "examples": examples,
        "student_correct": totals["student_correct"],
        "student_accuracy": totals["student_correct"] / examples,
        "teacher_correct": totals["teacher_correct"],
        "teacher_accuracy": totals["teacher_correct"] / examples,
        "teacher_agreement": totals["agreement"] / examples,
        "prediction_flips_from_teacher": examples - totals["agreement"],
        "kl_teacher_student": totals["kl"] / examples,
        "cross_entropy": totals["cross_entropy"] / examples,
        "student_score_rms": math.sqrt(totals["student_score_squared"] / score_count),
        "teacher_score_rms": math.sqrt(totals["teacher_score_squared"] / score_count),
        "student_prediction_sha256": _prediction_digest(student_predictions),
        "teacher_prediction_sha256": _prediction_digest(teacher_predictions),
    }


def _evaluate_plant_states(
    *,
    plant: IbmOmEffectiveCrossbarPlant,
    digital_scales: tuple[float, float],
    layout: tuple[CrossbarTileSpec, ...],
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    device: torch.device,
    maximum_batches: int | None,
    sample_limit: int | None,
) -> dict[str, Any]:
    """Evaluate the AIHWKit-visible state and hidden update state separately."""

    common = {
        "digital_scales": digital_scales,
        "layout": layout,
        "teacher": teacher,
        "loader": loader,
        "device": device,
        "maximum_batches": maximum_batches,
        "sample_limit": sample_limit,
    }
    return {
        "network_forward_state": "apparent_q",
        "hidden_update_state": "persistent_q",
        "apparent_forward": _evaluate(
            effective_state=plant.apparent,
            **common,
        ),
        "persistent_diagnostic": _evaluate(
            effective_state=plant.persistent,
            **common,
        ),
    }


def _float_summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().cpu().to(torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.all(torch.isfinite(flat))):
        raise ValueError("Expected a non-empty finite tensor summary input.")
    quantiles = torch.quantile(flat, torch.tensor((0.1, 0.5, 0.9, 0.99), dtype=torch.float64))
    return {
        "minimum": float(flat.min().item()),
        "p10": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "mean": float(flat.mean().item()),
        "p90": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
        "maximum": float(flat.max().item()),
        "rms": float(flat.square().mean().sqrt().item()),
    }


def _ordered_labeled_cohort_report(
    *,
    inputs: list[torch.Tensor],
    labels: list[torch.Tensor],
    split: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Hash an exact ordered update stream and its per-example identities."""

    if not inputs or len(inputs) != len(labels):
        raise ValueError("Expected non-empty matched cohort input and label batches.")
    joined_inputs = torch.cat(inputs).detach().cpu().to(torch.float32).contiguous()
    joined_labels = torch.cat(labels).detach().cpu().to(torch.int64).contiguous()
    if joined_inputs.shape[0] != joined_labels.numel():
        raise ValueError("Expected one cohort label per model input.")
    sample_ids = torch.arange(joined_labels.numel(), dtype=torch.int64)
    binding = build_calibration_binding(
        dataset_id="mnist",
        split=split,
        ordered_sample_ids=sample_ids,
        model_inputs=joined_inputs,
        labels=joined_labels,
    )
    identities = tuple(
        content_hash(
            {
                "model_input_sha256": tensor_sha256(joined_inputs[index]),
                "label": int(joined_labels[index].item()),
            }
        )
        for index in range(joined_labels.numel())
    )
    report = to_plain_data(binding)
    report["ordered_example_identity_sequence_sha256"] = content_hash(
        list(identities)
    )
    report["unique_example_identities"] = len(set(identities))
    return report, identities


def _population_defect_report(
    population: IbmReramArrayPopulation,
) -> dict[str, Any]:
    """Report the sampled defect intervention without inferring it from levels."""

    final = population.corrupt.detach().cpu()
    published = population.published_corrupt.detach().cpu()
    collapsed_zero_step = (
        (torch.abs(population.max_bound.detach().cpu() - population.min_bound.detach().cpu()) <= 1e-12)
        & (population.dwmin_up.detach().cpu() == 0.0)
        & (population.dwmin_down.detach().cpu() == 0.0)
    )
    if not torch.equal(final, collapsed_zero_step):
        raise RuntimeError(
            "Expected final corrupt cells to remain exactly the collapsed "
            "zero-step IBM OM identities."
        )
    if population.corruption_policy == "published" and not torch.equal(
        final, published
    ):
        raise RuntimeError(
            "Expected the published corruption policy to retain every sampled defect."
        )
    cells = population.size
    final_count = int(final.sum().item())
    published_count = int(published.sum().item())
    per_binding = []
    offset = 0
    for key, shape in zip(
        population.binding_keys,
        population.binding_shapes,
        strict=True,
    ):
        binding_cells = math.prod(shape)
        stop = offset + binding_cells
        binding_final = final[offset:stop]
        binding_published = published[offset:stop]
        binding_collapsed = collapsed_zero_step[offset:stop]
        per_binding.append(
            {
                "binding_key": key,
                "binding_shape": list(shape),
                "cell_offset_start": offset,
                "cell_offset_stop": stop,
                "cells": binding_cells,
                "final_corrupt_cells": int(binding_final.sum().item()),
                "published_corrupt_cells": int(binding_published.sum().item()),
                "repaired_published_corrupt_cells": int(
                    (binding_published & ~binding_final).sum().item()
                ),
                "unattributed_final_corrupt_cells": int(
                    (binding_final & ~binding_published).sum().item()
                ),
                "collapsed_zero_step_cells": int(binding_collapsed.sum().item()),
            }
        )
        offset = stop
    if offset != cells:  # pragma: no cover - population validation is upstream
        raise RuntimeError("Expected IBM OM binding offsets to cover every cell.")
    return {
        "corruption_policy": population.corruption_policy,
        "cells": cells,
        "final_corrupt_cells": final_count,
        "published_corrupt_cells": published_count,
        "repaired_published_corrupt_cells": int((published & ~final).sum().item()),
        "unattributed_final_corrupt_cells": int((final & ~published).sum().item()),
        "collapsed_zero_step_cells": int(collapsed_zero_step.sum().item()),
        "final_corrupt_fraction": final_count / cells,
        "published_corrupt_fraction": published_count / cells,
        "per_binding": per_binding,
    }


def _mapping_report(
    *,
    population: IbmReramArrayPopulation,
    requested: torch.Tensor,
    continuous: torch.Tensor,
    codebook: torch.Tensor,
    pulse_indices: torch.Tensor,
    level_counts: torch.Tensor,
) -> dict[str, Any]:
    logical_minimum = population.logical_min.detach().cpu()
    logical_maximum = population.logical_max.detach().cpu()
    below = requested < logical_minimum
    above = requested > logical_maximum
    requested_nonzero = requested != 0.0
    return {
        "requested_sha256": tensor_sha256(requested),
        "continuous_sha256": tensor_sha256(continuous),
        "codebook_sha256": tensor_sha256(codebook),
        "deterministic_projected_target_sha256": tensor_sha256(codebook),
        "requested": _float_summary(requested),
        "continuous": _float_summary(continuous),
        "codebook": _float_summary(codebook),
        "continuous_error": _float_summary(continuous - requested),
        "codebook_error": _float_summary(codebook - requested),
        "requested_below_support": int(below.sum().item()),
        "requested_above_support": int(above.sum().item()),
        "requested_outside_support": int((below | above).sum().item()),
        "requested_sign_counts": {
            "negative": int((requested < 0.0).sum().item()),
            "zero": int((requested == 0.0).sum().item()),
            "positive": int((requested > 0.0).sum().item()),
        },
        "continuous_sign_flips": int(((requested * continuous) < 0.0).sum().item()),
        "continuous_nonzero_to_zero": int(
            (requested_nonzero & (continuous == 0.0)).sum().item()
        ),
        "codebook_sign_flips": int(((requested * codebook) < 0.0).sum().item()),
        "codebook_nonzero_to_zero": int(
            (requested_nonzero & (codebook == 0.0)).sum().item()
        ),
        "pulse_indices": _float_summary(pulse_indices.to(torch.float32)),
        "effective_level_counts": _float_summary(level_counts.to(torch.float32)),
        "one_level_cells": int((level_counts == 1).sum().item()),
        "final_corrupt_one_level_cells": int(
            ((level_counts == 1) & population.corrupt.detach().cpu()).sum().item()
        ),
        "defects": _population_defect_report(population),
        "tie_rule": "lowest_pulse_index",
        "reference_projection_count": 0,
        "reference_reassignment_count": 0,
        "reference_failure_count": 0,
    }


def _program_endpoint(
    *,
    population: IbmReramArrayPopulation,
    requested: torch.Tensor,
    assignment_seed: int,
    endpoint_seed: int,
    maximum_pulses: int,
    tolerance_x: float,
    device: torch.device,
    stream_role: str,
    random_stream_fingerprint: str,
) -> tuple[IbmOmEffectiveCrossbarPlant, dict[str, Any]]:
    if assignment_seed != population.assignment_seed:
        raise ValueError(
            "Expected the programming assignment seed to match the sampled population."
        )
    trajectory_seeds = crossbar_trajectory_seeds(
        population,
        endpoint_seed=endpoint_seed,
        stream_role=stream_role,
        random_stream_population_fingerprint=random_stream_fingerprint,
    )
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        trajectory_seeds=trajectory_seeds,
        trajectory_seed_derivation=CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        device=device,
    )
    result = run_program_verify(
        plant.controller_port(),
        targets=(requested.to(device=device, dtype=torch.float32) + 1.0) / 2.0,
        tolerance=tolerance_x,
        maximum_pulses=maximum_pulses,
        settings=ControllerSettings(kind="one_pulse"),
    )
    persistent = plant.persistent.detach().cpu()
    apparent = plant.apparent.detach().cpu()
    target = requested.detach().cpu().to(torch.float32)
    logical_minimum = population.logical_min.detach().cpu()
    logical_maximum = population.logical_max.detach().cpu()
    tolerance_q = 2.0 * tolerance_x
    exact_in_support = (target >= logical_minimum) & (target <= logical_maximum)
    verify_window_intersects_support = (
        (target + tolerance_q >= logical_minimum)
        & (target - tolerance_q <= logical_maximum)
    )
    persistent_within_tolerance = torch.abs(persistent - target) <= tolerance_q
    apparent_q = apparent
    final_corrupt = population.corrupt.detach().cpu()
    result_accepted = result.accepted.detach().cpu()
    result_budget_exhausted = result.budget_exhausted.detach().cpu()
    result_nonfinite = result.nonfinite.detach().cpu()
    result_verify_count = result.verify_count.detach().cpu()
    result_total_pulses = result.total_pulses.detach().cpu()
    final_active = persistent + population.reference.detach().cpu()
    final_saturated_lower = torch.isclose(
        final_active,
        population.min_bound.detach().cpu(),
        atol=1e-7,
        rtol=0.0,
    )
    final_saturated_upper = torch.isclose(
        final_active,
        population.max_bound.detach().cpu(),
        atol=1e-7,
        rtol=0.0,
    )
    rng_state = plant.state_dict()
    trajectory_seed_origin = {
        "schema": "ebl.ibm_om_crossbar_trajectory_seed_origin",
        "schema_version": 1,
        "derivation": CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        "stream_role": stream_role.strip(),
        "assignment_seed": assignment_seed,
        "endpoint_seed": endpoint_seed,
        "random_stream_population_fingerprint": random_stream_fingerprint,
        "trajectory_seeds_sha256": tensor_sha256(
            rng_state["trajectory_seeds"]
        ),
    }
    report = {
        "endpoint_seed": endpoint_seed,
        "trajectory_seed_origin": trajectory_seed_origin,
        "trajectory_rng_backend": rng_state["rng_backend"],
        "trajectory_rng_reproducibility_scope": rng_state[
            "rng_reproducibility_scope"
        ],
        "trajectory_rng_statistical_contract": rng_state[
            "rng_statistical_contract"
        ],
        "trajectory_seed_derivation": rng_state["trajectory_seed_derivation"],
        "trajectory_seeds_sha256": tensor_sha256(
            rng_state["trajectory_seeds"]
        ),
        "trajectory_draw_indices_sha256": tensor_sha256(
            rng_state["trajectory_draw_indices"]
        ),
        "random_stream_population_fingerprint": random_stream_fingerprint,
        "persistent_sha256": tensor_sha256(persistent),
        "apparent_sha256": tensor_sha256(apparent),
        "cells": population.size,
        "defects": _population_defect_report(population),
        "exact_target_in_support": int(exact_in_support.sum().item()),
        "exact_target_outside_support": int((~exact_in_support).sum().item()),
        "verify_window_intersects_support": int(
            verify_window_intersects_support.sum().item()
        ),
        "apparent_accepted": int(result.accepted.sum().item()),
        "persistent_within_tolerance": int(persistent_within_tolerance.sum().item()),
        "apparent_accepted_persistent_outside_tolerance": int(
            (result_accepted & ~persistent_within_tolerance).sum().item()
        ),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "nonfinite": int(result.nonfinite.sum().item()),
        "verify_reads": int(result.verify_count.sum().item()),
        "total_programming_pulses": int(result.total_pulses.sum().item()),
        "mean_programming_pulses_per_cell": float(result.total_pulses.float().mean().item()),
        "maximum_programming_pulses_per_cell": int(result.total_pulses.max().item()),
        "direction_reversals": int(result.reversals.sum().item()),
        "target_q": _float_summary(target),
        "apparent_error_q": _float_summary(apparent_q - target),
        "persistent_error_q": _float_summary(persistent - target),
        "verify_tolerance_x": tolerance_x,
        "verify_tolerance_q": tolerance_q,
        "target_clipping": False,
        "network_forward_state": "apparent_q",
        "hidden_update_state": "persistent_q",
        "apparent_endpoint_applied_to_network": True,
        "plant_pulses": plant.pulse_statistics(),
        "final_corrupt_endpoint": {
            "cells": int(final_corrupt.sum().item()),
            "exact_target_in_support": int(
                (final_corrupt & exact_in_support).sum().item()
            ),
            "exact_target_outside_support": int(
                (final_corrupt & ~exact_in_support).sum().item()
            ),
            "verify_window_intersects_support": int(
                (final_corrupt & verify_window_intersects_support).sum().item()
            ),
            "verify_window_disjoint_support": int(
                (final_corrupt & ~verify_window_intersects_support).sum().item()
            ),
            "apparent_accepted": int(
                (final_corrupt & result_accepted).sum().item()
            ),
            "persistent_within_tolerance": int(
                (final_corrupt & persistent_within_tolerance).sum().item()
            ),
            "apparent_accepted_persistent_outside_tolerance": int(
                (
                    final_corrupt
                    & result_accepted
                    & ~persistent_within_tolerance
                ).sum().item()
            ),
            "budget_exhausted": int(
                (final_corrupt & result_budget_exhausted).sum().item()
            ),
            "nonfinite": int((final_corrupt & result_nonfinite).sum().item()),
            "verify_reads": int(result_verify_count[final_corrupt].sum().item()),
            "programming_pulses": int(
                result_total_pulses[final_corrupt].sum().item()
            ),
            "final_saturated_lower": int(
                (final_corrupt & final_saturated_lower).sum().item()
            ),
            "final_saturated_upper": int(
                (final_corrupt & final_saturated_upper).sum().item()
            ),
        },
    }
    return plant, report


def _matched_published_fault_overlay(
    *,
    healthy: IbmReramArrayPopulation,
    published: IbmReramArrayPopulation,
    preset_default_corrupt_devices_prob: float,
    enabled_corrupt_devices_prob: float,
    corrupt_devices_range: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Bind a repaired healthy identity to its published stuck-cell companion."""

    if (
        not math.isclose(
            preset_default_corrupt_devices_prob,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            enabled_corrupt_devices_prob,
            PUBLISHED_CORRUPT_PROBABILITY[OM_PRESET],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            corrupt_devices_range,
            PUBLISHED_CORRUPT_RANGE[OM_PRESET],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Expected the pinned AIHWKit OM corrupt-device settings.")
    if (
        healthy.assignment_seed != published.assignment_seed
        or healthy.binding_keys != published.binding_keys
        or healthy.binding_shapes != published.binding_shapes
        or healthy.binding_sampling_seeds != published.binding_sampling_seeds
        or healthy.size != published.size
        or healthy.nominal_dw_min != published.nominal_dw_min
        or healthy.dw_min_std != published.dw_min_std
        or healthy.write_noise_std != published.write_noise_std
        or healthy.aihwkit_version != published.aihwkit_version
        or healthy.corruption_policy != "counterfactual_repaired"
        or published.corruption_policy != "published"
    ):
        raise ValueError("Expected matched repaired and published IBM OM populations.")
    mask = published.corrupt.detach().cpu().to(torch.bool)
    if (
        not bool(torch.any(mask))
        or bool(torch.any(healthy.corrupt.detach().cpu()))
        or not torch.equal(healthy.published_corrupt.detach().cpu(), mask)
        or not torch.equal(published.published_corrupt.detach().cpu(), mask)
    ):
        raise ValueError("Expected the matched published defect mask to survive only in its companion.")
    healthy_mask = ~mask
    for name in (
        "min_bound",
        "max_bound",
        "dwmin_up",
        "dwmin_down",
        "reference",
    ):
        left = getattr(healthy, name).detach().cpu()[healthy_mask]
        right = getattr(published, name).detach().cpu()[healthy_mask]
        if not torch.equal(left, right):
            raise ValueError(
                f"Expected non-fault IBM OM field {name!r} to match its published companion."
            )
    stuck_persistent_q = published.logical_min.detach().cpu().to(torch.float32)
    stuck_active = published.min_bound.detach().cpu().to(torch.float32)
    corrupt_range = corrupt_devices_range
    if not torch.equal(
        stuck_persistent_q[mask],
        published.logical_max.detach().cpu()[mask],
    ) or bool(torch.any(stuck_active[mask].abs() > corrupt_range + 1e-7)):
        raise ValueError("Expected every published fault source cell to have collapsed support.")
    report = {
        "policy": "post_deployment_published_companion_replay",
        "healthy_population_fingerprint": healthy.fingerprint,
        "published_population_fingerprint": published.fingerprint,
        "assignment_seed": healthy.assignment_seed,
        "fault_mask_sha256": tensor_sha256(mask),
        "stuck_persistent_q_sha256": tensor_sha256(stuck_persistent_q[mask]),
        "stuck_state_source": (
            "AIHWKit_1.1.0_sampled_corrupt_active_bound_minus_intrinsic_reference"
        ),
        "published_corrupt_devices_probability": enabled_corrupt_devices_prob,
        "preset_default_corrupt_devices_probability": (
            preset_default_corrupt_devices_prob
        ),
        "corrupt_devices_range": corrupt_range,
        "persistent_pulse_increments": "zero_up_and_down",
        "apparent_write_noise_retained": True,
        "faulted_cells": int(mask.sum().item()),
        "cells": healthy.size,
        "fault_fraction": float(mask.float().mean().item()),
        "non_fault_identity_equal": True,
        "shared_nominal_device_parameters_equal": True,
    }
    return mask, stuck_persistent_q, report


def _recovery_seed(
    *,
    runtime_seed: int,
    assignment_seed: int,
    endpoint_seed: int,
) -> int:
    # Deliberately excludes recovery policy and layer scope so matched arms see
    # the same Bernoulli stream.
    return derive_seed(runtime_seed, assignment_seed, endpoint_seed, "crossbar_pulse_selection")


def _offchip_realized_state(
    *,
    master: torch.Tensor,
    policy: str,
    logical_minimum: torch.Tensor,
    logical_maximum: torch.Tensor,
    codebook_rows: torch.Tensor,
    validate_codebook: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if policy in {"none", "population_programming_error_hwa"}:
        return master, None
    if policy in {
        "support_clamped_no_update",
        "continuous_hwa",
        "stochastic_apparent_hwa",
    }:
        return torch.maximum(torch.minimum(master, logical_maximum), logical_minimum), None
    if policy == "deterministic_qat":
        return project_to_nearest_effective_code_device(
            codebook_rows,
            master,
            rowwise=True,
            validate=validate_codebook,
        )
    raise ValueError(f"Unsupported off-chip policy: {policy!r}.")


def _offchip_deployment_state(
    *,
    master: torch.Tensor,
    realized: torch.Tensor,
    deployment_target: str,
) -> torch.Tensor:
    """Select the state requested from P&V without changing training forwards."""

    if deployment_target == "policy_realized_state":
        return realized
    if deployment_target == "fixed_final_master_fault_blind_pv":
        return master
    raise ValueError(f"Unsupported off-chip deployment target: {deployment_target!r}.")


def _sample_aihwkit_om_apparent_write_noise(
    *,
    persistent_q: torch.Tensor,
    nominal_dw_min: float,
    write_noise_std: float,
    relative_scale: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample AIHWKit 1.1 OM apparent ``q`` from persistent ``q``.

    ``SoftBoundsReferenceDevice`` uses
    ``q_app = q_persistent + write_noise_std * dw_min * N(0, 1)``.  The
    intrinsic reference is already included in the effective ``q=a-r`` state,
    so it is not sampled again here.  This helper intentionally uses an
    explicit PyTorch generator for replayability; it claims equation and
    distribution parity, not native C++ RNG-stream parity.
    """

    if persistent_q.dtype != torch.float32 or not bool(torch.all(torch.isfinite(persistent_q))):
        raise ValueError("Expected finite float32 persistent q for apparent-noise sampling.")
    values = (float(nominal_dw_min), float(write_noise_std), float(relative_scale))
    if (
        not all(math.isfinite(value) for value in values)
        or nominal_dw_min <= 0.0
        or write_noise_std < 0.0
        or relative_scale <= 0.0
    ):
        raise ValueError("Expected finite positive OM write-noise parameters.")
    standard_normal = torch.randn(
        persistent_q.shape,
        dtype=torch.float32,
        device=persistent_q.device,
        generator=generator,
    )
    noise = standard_normal * (
        float(nominal_dw_min) * float(write_noise_std) * float(relative_scale)
    )
    return persistent_q + noise, noise


def _evaluate_held_apparent_hwa_state(
    *,
    support_clamped_digital_master_q: torch.Tensor,
    digital_scales: tuple[float, float],
    layout: tuple[CrossbarTileSpec, ...],
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    device: torch.device,
    maximum_batches: int | None,
    sample_limit: int | None,
    nominal_dw_min: float,
    write_noise_std: float,
    relative_scale: float,
    generator: torch.Generator,
    evaluation_id: str,
    resolved_seed: int,
) -> dict[str, Any]:
    """Evaluate one held apparent HWA draw and its digital-master diagnostic.

    The sampled apparent ``q`` is drawn exactly once and then held for the
    complete evaluation cohort.  The support-clamped digital master is useful
    as a deterministic diagnostic, but it is not a persistent physical device
    state and must never be reported under the persistent-state label.
    """

    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        raise ValueError("Expected a non-empty held-apparent evaluation identity.")
    if isinstance(resolved_seed, bool) or not isinstance(resolved_seed, int):
        raise ValueError("Expected an integer held-apparent evaluation seed.")
    base = support_clamped_digital_master_q.detach().to(
        device=device,
        dtype=torch.float32,
    )
    generator_before = generator.get_state().detach().cpu().clone()
    held_apparent, noise = _sample_aihwkit_om_apparent_write_noise(
        persistent_q=base,
        nominal_dw_min=nominal_dw_min,
        write_noise_std=write_noise_std,
        relative_scale=relative_scale,
        generator=generator,
    )
    generator_after = generator.get_state().detach().cpu().clone()
    apparent_metrics = _evaluate(
        effective_state=held_apparent,
        digital_scales=digital_scales,
        layout=layout,
        teacher=teacher,
        loader=loader,
        device=device,
        maximum_batches=maximum_batches,
        sample_limit=sample_limit,
    )
    diagnostic_metrics = _evaluate(
        effective_state=base,
        digital_scales=digital_scales,
        layout=layout,
        teacher=teacher,
        loader=loader,
        device=device,
        maximum_batches=maximum_batches,
        sample_limit=sample_limit,
    )
    noise64 = noise.detach().to(torch.float64)
    observed_mean = float(noise64.mean().item())
    observed_variance = max(
        0.0,
        float(torch.mean(torch.square(noise64)).item()) - observed_mean**2,
    )
    return {
        "network_forward_state": "held_apparent_q",
        "primary_state": "held_apparent_q",
        "diagnostic_state_role": (
            "nonpersistent_support_clamped_digital_master_q"
        ),
        "persistent_device_state_present": False,
        "apparent_forward": apparent_metrics,
        "nonpersistent_support_clamped_digital_master_diagnostic": (
            diagnostic_metrics
        ),
        "held_apparent_state_receipt": {
            "schema": "ebl.ibm_om_crossbar_held_apparent_hwa_evaluation",
            "schema_version": 1,
            "evaluation_id": evaluation_id.strip(),
            "resolved_seed": resolved_seed,
            "resampling": (
                "one_full_array_draw_held_across_complete_evaluation_cohort"
            ),
            "support_clamped_digital_master_q_sha256": tensor_sha256(base),
            "held_apparent_q_sha256": tensor_sha256(held_apparent),
            "write_noise_q_sha256": tensor_sha256(noise),
            "generator_state_before_sha256": tensor_sha256(generator_before),
            "generator_state_after_sha256": tensor_sha256(generator_after),
            "configured_sigma_q": float(
                nominal_dw_min * write_noise_std * relative_scale
            ),
            "observed_mean_q": observed_mean,
            "observed_std_q": math.sqrt(observed_variance),
            "observed_minimum_q": float(noise.min().item()),
            "observed_maximum_q": float(noise.max().item()),
        },
    }


def _offchip_adapt(
    *,
    source_requested: torch.Tensor,
    population: IbmReramArrayPopulation,
    codebook_values: torch.Tensor,
    spec: Any,
    layout: tuple[CrossbarTileSpec, ...],
    digital_scales: tuple[float, float],
    teacher: BiasFreeReluTeacher,
    validation_loader: Iterable,
    train_loader: Iterable,
    device: torch.device,
    test_loader: Iterable | None = None,
    programming_error_model: IbmReramAcceptedEndpointModel | None = None,
) -> tuple[torch.Tensor, dict[str, Any], Mapping[str, Any]]:
    """Run fixed-final deterministic/stochastic HWA or codebook QAT.

    Adam acts on two effective-``q`` shadow tensors.  Dividing each parameter
    group's rate by its fixed digital layer scale makes the Adam displacement
    correspond to the declared logical-weight displacement (up to Adam's
    epsilon term).  The forward pass uses a straight-through estimator; only
    the continuous support clamp or deterministic codebook state reaches the
    network.  ``stochastic_apparent_hwa`` additionally samples the AIHWKit
    1.1.0 ``SoftBoundsReferenceDevice`` additive apparent-write-noise equation
    once per minibatch.  This is equation/distribution parity with an explicit
    PyTorch RNG, not native AIHWKit RNG-stream parity and not a P&V-conditioned
    endpoint sampler.  ``population_programming_error_hwa`` instead keeps one
    global ``q`` range and draws accepted, non-corrupt, target-conditioned P&V
    residuals from a characterized population on every minibatch.  It never
    exposes a fixed array identity, bound tuple, or defect mask to training.
    """

    slices = layer_cell_slices(layout)
    source = source_requested.detach().to(device=device, dtype=torch.float32)
    parameters = [
        torch.nn.Parameter(source[layer_slice].clone()) for layer_slice in slices
    ]
    master = torch.cat(parameters)
    minimum = population.logical_min.detach().to(device=device, dtype=torch.float32)
    maximum = population.logical_max.detach().to(device=device, dtype=torch.float32)
    codebook_rows = (
        codebook_values.detach()
        .to(device=device, dtype=torch.float32)
        .transpose(0, 1)
        .contiguous()
    )
    initial_realized, initial_indices = _offchip_realized_state(
        master=master,
        policy=spec.offchip.policy,
        logical_minimum=minimum,
        logical_maximum=maximum,
        codebook_rows=codebook_rows,
        validate_codebook=True,
    )
    noise_settings = spec.offchip.forward_noise
    evaluation_forward_policy = getattr(
        spec.offchip,
        "evaluation_forward_policy",
        "legacy_policy_realized_state",
    )
    if evaluation_forward_policy not in {
        "legacy_policy_realized_state",
        "one_sampled_held_apparent_q_per_evaluation",
    }:
        raise ValueError("Expected a supported off-chip HWA evaluation-forward policy.")
    held_apparent_evaluation = (
        evaluation_forward_policy
        == "one_sampled_held_apparent_q_per_evaluation"
    )
    if held_apparent_evaluation and (
        spec.offchip.policy != "stochastic_apparent_hwa"
        or noise_settings is None
    ):
        raise ValueError(
            "Expected held apparent HWA evaluation only with stochastic apparent HWA."
        )

    def evaluate_offchip_state(
        realized: torch.Tensor,
        loader: Iterable,
        *,
        evaluation_id: str,
        maximum_batches: int | None,
    ) -> dict[str, Any]:
        if not held_apparent_evaluation:
            return _evaluate(
                effective_state=realized,
                digital_scales=digital_scales,
                layout=layout,
                teacher=teacher,
                loader=loader,
                device=device,
                maximum_batches=maximum_batches,
                sample_limit=spec.evaluation.sample_limit,
            )
        if noise_settings is None:  # pragma: no cover - guarded above
            raise RuntimeError("Expected stochastic HWA evaluation-noise settings.")
        resolved_seed = derive_seed(
            noise_settings.seed,
            population.assignment_seed,
            "offchip_stochastic_apparent_hwa_held_evaluation",
            evaluation_id,
        )
        evaluation_generator = torch.Generator(device=device)
        evaluation_generator.manual_seed(resolved_seed)
        return _evaluate_held_apparent_hwa_state(
            support_clamped_digital_master_q=realized,
            digital_scales=digital_scales,
            layout=layout,
            teacher=teacher,
            loader=loader,
            device=device,
            maximum_batches=maximum_batches,
            sample_limit=spec.evaluation.sample_limit,
            nominal_dw_min=population.nominal_dw_min,
            write_noise_std=population.write_noise_std,
            relative_scale=noise_settings.relative_scale,
            generator=evaluation_generator,
            evaluation_id=evaluation_id,
            resolved_seed=resolved_seed,
        )

    initial_validation = evaluate_offchip_state(
        initial_realized,
        validation_loader,
        evaluation_id="initial.validation",
        maximum_batches=spec.evaluation.maximum_validation_batches,
    )
    evaluate_epoch_test = spec.offchip.epoch_evaluation == "validation_and_test"
    if evaluate_epoch_test and test_loader is None:
        raise ValueError(
            "Expected a test loader for off-chip validation-and-test epoch diagnostics."
        )
    initial_test = (
        None
        if not evaluate_epoch_test
        else evaluate_offchip_state(
            initial_realized,
            test_loader,
            evaluation_id="initial.test",
            maximum_batches=None,
        )
    )
    initial_master = master.detach().cpu().clone()
    initial_realized_cpu = initial_realized.detach().cpu().clone()
    if spec.offchip.policy in {"none", "support_clamped_no_update"}:
        policy = spec.offchip.policy
        deployment = _offchip_deployment_state(
            master=initial_master,
            realized=initial_realized_cpu,
            deployment_target=spec.offchip.deployment_target,
        ).detach().cpu().clone()
        state = {
            "schema": "ebl.ibm_om_crossbar_offchip_state",
            "schema_version": 2,
            "policy": policy,
            "initial_master_q": initial_master,
            "fixed_final_master_q": initial_master.clone(),
            "fixed_final_realized_q": initial_realized_cpu,
            "fixed_final_deployment_q": deployment,
            "optimizer_state_dict": None,
            "forward_noise_generator_initial_state": None,
            "forward_noise_generator_final_state": None,
            "programming_error_model_fingerprint": None,
            "programming_error_generator_initial_state": None,
            "programming_error_generator_final_state": None,
        }
        return deployment, {
            "policy": policy,
            "training_protocol": spec.offchip.training_protocol,
            "objective": spec.offchip.objective,
            "initial_master_sha256": tensor_sha256(initial_master),
            "initial_realized_sha256": tensor_sha256(initial_realized_cpu),
            "initial_codebook_index_sha256": None,
            "initial_validation": initial_validation,
            "initial_test": initial_test,
            "epochs": [],
            "fixed_final_epoch": 0,
            "fixed_final_master_sha256": tensor_sha256(initial_master),
            "fixed_final_realized_sha256": tensor_sha256(initial_realized_cpu),
            "fixed_final_deployment_sha256": tensor_sha256(deployment),
            "fixed_final_codebook_index_sha256": None,
            "fixed_final_validation": initial_validation,
            "fixed_final_test": initial_test,
            "optimizer_steps": 0,
            "logical_learning_rates": [0.0, 0.0],
            "effective_q_learning_rates": [0.0, 0.0],
            "checkpoint_policy": (
                "fixed_source_no_selection"
                if policy == "none"
                else "fixed_source_support_clamp_no_selection"
            ),
            "deployment_target": spec.offchip.deployment_target,
            "stochastic_programming_during_training": False,
            "stochastic_apparent_forward_noise_during_training": False,
            "stochastic_programming_error_model_during_training": False,
            "persistent_state_updates_during_training": False,
            "forward_noise": None,
            "programming_error": None,
        }, state

    effective_rates = tuple(
        logical_rate / digital_scale
        for logical_rate, digital_scale in zip(
            spec.offchip.logical_learning_rates,
            digital_scales,
            strict=True,
        )
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter], "lr": rate}
            for parameter, rate in zip(parameters, effective_rates, strict=True)
        ],
        betas=(spec.offchip.beta_1, spec.offchip.beta_2),
        eps=spec.offchip.epsilon,
    )
    noise_generator: torch.Generator | None = None
    noise_resolved_seed: int | None = None
    noise_initial_state: torch.Tensor | None = None
    noise_scale_q: float | None = None
    if spec.offchip.policy == "stochastic_apparent_hwa":
        if noise_settings is None:
            raise RuntimeError("Expected stochastic HWA forward-noise settings.")
        noise_resolved_seed = derive_seed(
            noise_settings.seed,
            population.assignment_seed,
            "offchip_stochastic_apparent_hwa",
        )
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(noise_resolved_seed)
        noise_initial_state = noise_generator.get_state().detach().cpu().clone()
        noise_scale_q = float(
            population.write_noise_std
            * population.nominal_dw_min
            * noise_settings.relative_scale
        )
    programming_settings = spec.offchip.programming_error
    programming_generator: torch.Generator | None = None
    programming_resolved_seed: int | None = None
    programming_initial_state: torch.Tensor | None = None
    compiled_programming_model: IbmReramAcceptedEndpointModel | None = None
    if spec.offchip.policy == "population_programming_error_hwa":
        if programming_settings is None or programming_error_model is None:
            raise RuntimeError(
                "Expected population HWA settings and a compiled endpoint model."
            )
        compiled_programming_model = programming_error_model.to(
            device,
            dtype=torch.float32,
        )
        programming_resolved_seed = derive_seed(
            programming_settings.seed,
            compiled_programming_model.fingerprint,
            "offchip_population_programming_error_hwa",
        )
        programming_generator = torch.Generator(device=device)
        programming_generator.manual_seed(programming_resolved_seed)
        programming_initial_state = (
            programming_generator.get_state().detach().cpu().clone()
        )
    elif programming_error_model is not None:
        raise RuntimeError(
            "Expected no compiled programming-error model outside population HWA."
        )
    epoch_reports: list[dict[str, Any]] = []
    optimizer_steps = 0
    final_validation = initial_validation
    final_test = initial_test
    final_realized = initial_realized
    final_indices = initial_indices
    first_epoch_inputs: list[torch.Tensor] = []
    first_epoch_labels: list[torch.Tensor] = []
    for epoch in range(1, spec.offchip.epochs + 1):
        loss_sum = 0.0
        examples = 0
        batches = 0
        noise_value_count = 0
        noise_sum = 0.0
        noise_square_sum = 0.0
        noise_minimum = math.inf
        noise_maximum = -math.inf
        noise_sequence_digest = sha256()
        noise_generator_before = (
            None
            if noise_generator is None
            else noise_generator.get_state().detach().cpu().clone()
        )
        programming_value_count = 0
        programming_residual_sum = 0.0
        programming_residual_square_sum = 0.0
        programming_residual_minimum = math.inf
        programming_residual_maximum = -math.inf
        programming_apparent_outside_bounds = 0
        programming_sequence_digest = sha256()
        programming_generator_before = (
            None
            if programming_generator is None
            else programming_generator.get_state().detach().cpu().clone()
        )
        programming_strength: float | None = None
        if programming_generator is not None:
            if programming_settings is None:  # pragma: no cover - invariant
                raise RuntimeError("Expected population HWA settings.")
            ramp_fraction = min(
                float(epoch) / float(programming_settings.ramp_epochs),
                1.0,
            )
            programming_strength = (
                programming_settings.initial_strength
                + ramp_fraction
                * (
                    programming_settings.final_strength
                    - programming_settings.initial_strength
                )
            )
        for batch_index, (inputs, labels) in enumerate(
            limited(train_loader, spec.offchip.maximum_batches)
        ):
            if epoch == 1:
                first_epoch_inputs.append(
                    inputs.detach().cpu().to(torch.float32).contiguous()
                )
                first_epoch_labels.append(
                    labels.detach().cpu().to(torch.int64).contiguous()
                )
            inputs = inputs.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                teacher_logits = teacher.logits(inputs)
                teacher_log_probability = F.log_softmax(teacher_logits, dim=1)
                teacher_probability = teacher_log_probability.exp()
            optimizer.zero_grad(set_to_none=True)
            master = torch.cat(parameters)
            realized, _indices = _offchip_realized_state(
                master=master,
                policy=spec.offchip.policy,
                logical_minimum=minimum,
                logical_maximum=maximum,
                codebook_rows=codebook_rows,
                validate_codebook=False,
            )
            forward_state = realized
            if noise_generator is not None:
                if noise_settings is None:
                    raise RuntimeError("Expected stochastic HWA noise settings.")
                forward_state, noise = _sample_aihwkit_om_apparent_write_noise(
                    persistent_q=realized,
                    nominal_dw_min=population.nominal_dw_min,
                    write_noise_std=population.write_noise_std,
                    relative_scale=noise_settings.relative_scale,
                    generator=noise_generator,
                )
                noise64 = noise.detach().to(torch.float64)
                noise_value_count += int(noise.numel())
                noise_sum += float(noise64.sum().item())
                noise_square_sum += float(torch.square(noise64).sum().item())
                noise_minimum = min(noise_minimum, float(noise.min().item()))
                noise_maximum = max(noise_maximum, float(noise.max().item()))
                noise_sequence_digest.update(
                    bytes.fromhex(tensor_sha256(noise.detach()))
                )
            if programming_generator is not None:
                if (
                    compiled_programming_model is None
                    or programming_strength is None
                ):  # pragma: no cover - invariant
                    raise RuntimeError("Expected a compiled population HWA model.")
                target_x = (realized.detach() + 1.0) * 0.5
                endpoint_sample = compiled_programming_model.sample(
                    target_x,
                    generator=programming_generator,
                )
                residual_q = endpoint_sample.residual_x * 2.0
                applied_error_q = residual_q * programming_strength
                forward_state = realized + applied_error_q
                residual64 = residual_q.detach().to(torch.float64)
                programming_value_count += int(residual_q.numel())
                programming_residual_sum += float(residual64.sum().item())
                programming_residual_square_sum += float(
                    torch.square(residual64).sum().item()
                )
                programming_residual_minimum = min(
                    programming_residual_minimum,
                    float(residual_q.min().item()),
                )
                programming_residual_maximum = max(
                    programming_residual_maximum,
                    float(residual_q.max().item()),
                )
                programming_apparent_outside_bounds += int(
                    ((forward_state < -1.0) | (forward_state > 1.0))
                    .sum()
                    .item()
                )
                programming_sequence_digest.update(
                    bytes.fromhex(tensor_sha256(residual_q.detach()))
                )
            straight_through = master + (forward_state - master).detach()
            student_logits = standard_crossbar_logits(
                inputs,
                straight_through,
                layout,
                digital_scales=digital_scales,
            )
            student_log_probability = F.log_softmax(student_logits, dim=1)
            loss = (
                teacher_probability
                * (teacher_log_probability - student_log_probability)
            ).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for parameter in parameters:
                    parameter.clamp_(
                        min=spec.offchip.master_q_bounds[0],
                        max=spec.offchip.master_q_bounds[1],
                    )
            examples += int(inputs.shape[0])
            loss_sum += float(loss.item()) * int(inputs.shape[0])
            batches = batch_index + 1
            optimizer_steps += 1
        if examples == 0:
            raise ValueError("Expected off-chip HWA/QAT to process training examples.")
        if any(not bool(torch.all(torch.isfinite(parameter))) for parameter in parameters):
            raise RuntimeError("Expected finite off-chip HWA/QAT shadow weights.")
        master = torch.cat(parameters)
        final_realized, final_indices = _offchip_realized_state(
            master=master,
            policy=spec.offchip.policy,
            logical_minimum=minimum,
            logical_maximum=maximum,
            codebook_rows=codebook_rows,
            validate_codebook=False,
        )
        final_validation = evaluate_offchip_state(
            final_realized,
            validation_loader,
            evaluation_id=f"epoch_{epoch:03d}.validation",
            maximum_batches=spec.evaluation.maximum_validation_batches,
        )
        final_test = (
            None
            if not evaluate_epoch_test
            else evaluate_offchip_state(
                final_realized,
                test_loader,
                evaluation_id=f"epoch_{epoch:03d}.test",
                maximum_batches=None,
            )
        )
        epoch_report: dict[str, Any] = {
            "epoch": epoch,
            "batches": batches,
            "examples": examples,
            "train_kl_teacher_student": loss_sum / examples,
            "master_sha256": tensor_sha256(master),
            "realized_sha256": tensor_sha256(final_realized),
            "codebook_index_sha256": (
                None if final_indices is None else tensor_sha256(final_indices)
            ),
            "validation": final_validation,
            "test": final_test,
            "forward_noise": None,
            "programming_error": None,
        }
        if noise_generator is not None:
            if noise_value_count <= 0 or noise_scale_q is None:
                raise RuntimeError("Expected stochastic HWA to draw forward noise.")
            observed_mean = noise_sum / noise_value_count
            observed_variance = max(
                0.0,
                noise_square_sum / noise_value_count - observed_mean**2,
            )
            generator_after = noise_generator.get_state().detach().cpu().clone()
            epoch_report["forward_noise"] = {
                "full_array_draws": batches,
                "scalar_values": noise_value_count,
                "configured_sigma_q": noise_scale_q,
                "observed_mean_q": observed_mean,
                "observed_std_q": math.sqrt(observed_variance),
                "observed_minimum_q": noise_minimum,
                "observed_maximum_q": noise_maximum,
                "noise_tensor_sequence_sha256": noise_sequence_digest.hexdigest(),
                "generator_state_before_sha256": tensor_sha256(
                    noise_generator_before
                ),
                "generator_state_after_sha256": tensor_sha256(generator_after),
            }
        if programming_generator is not None:
            if programming_value_count <= 0 or programming_strength is None:
                raise RuntimeError(
                    "Expected population HWA to draw programming errors."
                )
            observed_mean = (
                programming_residual_sum / programming_value_count
            )
            observed_variance = max(
                0.0,
                programming_residual_square_sum / programming_value_count
                - observed_mean**2,
            )
            programming_generator_after = (
                programming_generator.get_state().detach().cpu().clone()
            )
            epoch_report["programming_error"] = {
                "full_array_draws": batches,
                "scalar_values": programming_value_count,
                "strength": programming_strength,
                "unscaled_observed_mean_q": observed_mean,
                "unscaled_observed_std_q": math.sqrt(observed_variance),
                "unscaled_observed_minimum_q": (
                    programming_residual_minimum
                ),
                "unscaled_observed_maximum_q": (
                    programming_residual_maximum
                ),
                "applied_observed_mean_q": (
                    programming_strength * observed_mean
                ),
                "applied_observed_std_q": (
                    programming_strength * math.sqrt(observed_variance)
                ),
                "applied_apparent_values_outside_global_q_bounds": (
                    programming_apparent_outside_bounds
                ),
                "unscaled_residual_tensor_sequence_sha256": (
                    programming_sequence_digest.hexdigest()
                ),
                "generator_state_before_sha256": tensor_sha256(
                    programming_generator_before
                ),
                "generator_state_after_sha256": tensor_sha256(
                    programming_generator_after
                ),
            }
        epoch_reports.append(epoch_report)

    final_master_cpu = torch.cat(parameters).detach().cpu().clone()
    final_realized_cpu = final_realized.detach().cpu().clone()
    final_deployment_cpu = _offchip_deployment_state(
        master=final_master_cpu,
        realized=final_realized_cpu,
        deployment_target=spec.offchip.deployment_target,
    ).detach().cpu().clone()
    first_epoch_training_cohort, _first_epoch_identities = (
        _ordered_labeled_cohort_report(
            inputs=first_epoch_inputs,
            labels=first_epoch_labels,
            split="predeployment_offchip_first_epoch_update_stream",
        )
    )
    state = {
        "schema": "ebl.ibm_om_crossbar_offchip_state",
        "schema_version": (
            3 if compiled_programming_model is not None else 2
        ),
        "policy": spec.offchip.policy,
        "initial_master_q": initial_master,
        "fixed_final_master_q": final_master_cpu,
        "fixed_final_realized_q": final_realized_cpu,
        "fixed_final_deployment_q": final_deployment_cpu,
        "optimizer_state_dict": optimizer.state_dict(),
        "forward_noise_generator_initial_state": noise_initial_state,
        "forward_noise_generator_final_state": (
            None
            if noise_generator is None
            else noise_generator.get_state().detach().cpu().clone()
        ),
        "programming_error_model_fingerprint": (
            None
            if compiled_programming_model is None
            else compiled_programming_model.fingerprint
        ),
        "programming_error_generator_initial_state": programming_initial_state,
        "programming_error_generator_final_state": (
            None
            if programming_generator is None
            else programming_generator.get_state().detach().cpu().clone()
        ),
    }
    return final_deployment_cpu, {
        "policy": spec.offchip.policy,
        "training_protocol": spec.offchip.training_protocol,
        "objective": spec.offchip.objective,
        "initial_master_sha256": tensor_sha256(initial_master),
        "initial_realized_sha256": tensor_sha256(initial_realized_cpu),
        "initial_codebook_index_sha256": (
            None if initial_indices is None else tensor_sha256(initial_indices)
        ),
        "initial_validation": initial_validation,
        "initial_test": initial_test,
        "first_epoch_training_cohort": first_epoch_training_cohort,
        "epochs": epoch_reports,
        "fixed_final_epoch": spec.offchip.epochs,
        "fixed_final_master_sha256": tensor_sha256(final_master_cpu),
        "fixed_final_realized_sha256": tensor_sha256(final_realized_cpu),
        "fixed_final_deployment_sha256": tensor_sha256(final_deployment_cpu),
        "fixed_final_codebook_index_sha256": (
            None if final_indices is None else tensor_sha256(final_indices)
        ),
        "fixed_final_validation": final_validation,
        "fixed_final_test": final_test,
        "optimizer_steps": optimizer_steps,
        "logical_learning_rates": list(spec.offchip.logical_learning_rates),
        "effective_q_learning_rates": list(effective_rates),
        "checkpoint_policy": spec.offchip.checkpoint_policy,
        "deployment_target": spec.offchip.deployment_target,
        "stochastic_programming_during_training": False,
        "stochastic_apparent_forward_noise_during_training": (
            noise_generator is not None
        ),
        "stochastic_programming_error_model_during_training": (
            programming_generator is not None
        ),
        "persistent_state_updates_during_training": False,
        **(
            {
                "evaluation_forward_policy": evaluation_forward_policy,
                "primary_evaluation_state": "held_apparent_q",
                "diagnostic_evaluation_state": (
                    "nonpersistent_support_clamped_digital_master_q"
                ),
                "persistent_device_state_present_during_offchip_hwa": False,
                "evaluation_noise_stream": {
                    "derivation": (
                        "derive_seed(configured_forward_noise_seed,assignment_seed,"
                        "offchip_stochastic_apparent_hwa_held_evaluation,"
                        "evaluation_id)"
                    ),
                    "one_independent_generator_per_evaluation": True,
                    "test_evaluation_changes_training_noise_stream": False,
                    "test_evaluation_changes_validation_noise_stream": False,
                },
            }
            if held_apparent_evaluation
            else {}
        ),
        "forward_noise": (
            None
            if noise_settings is None
            else {
                **to_plain_data(noise_settings),
                "configured_seed": noise_settings.seed,
                "resolved_seed": noise_resolved_seed,
                "nominal_dw_min": float(population.nominal_dw_min),
                "write_noise_std": float(population.write_noise_std),
                "sigma_q": noise_scale_q,
                "intrinsic_reference_semantics": "effective_q_equals_active_a_minus_fixed_r",
                "native_rng_stream_parity": False,
                "program_verify_conditioned_endpoint_sampling": False,
                "generator_initial_state_sha256": tensor_sha256(
                    noise_initial_state
                ),
                "generator_final_state_sha256": tensor_sha256(
                    noise_generator.get_state().detach().cpu()
                ),
            }
        ),
        "programming_error": (
            None
            if programming_settings is None
            else {
                **to_plain_data(programming_settings),
                "configured_seed": programming_settings.seed,
                "resolved_seed": programming_resolved_seed,
                "compiled_model_fingerprint": (
                    compiled_programming_model.fingerprint
                ),
                "global_master_q_bounds": [-1.0, 1.0],
                "target_conversion": "x_equals_q_plus_one_over_two",
                "apparent_conversion": (
                    "q_apparent_equals_q_master_plus_strength_times_"
                    "two_times_sampled_residual_x"
                ),
                "accepted_noncorrupt_residuals_only": True,
                "fixed_array_identity_during_training": False,
                "fixed_per_cell_bounds_during_training": False,
                "corrupt_device_mask_during_training": False,
                "persistent_endpoint_state_during_training": False,
                "physical_programming_during_training": False,
                "same_sample_for_forward_and_backward": True,
                "native_rng_stream_parity": False,
                "generator_initial_state_sha256": tensor_sha256(
                    programming_initial_state
                ),
                "generator_final_state_sha256": tensor_sha256(
                    programming_generator.get_state().detach().cpu()
                ),
            }
        ),
    }, state


def _state_tree_equal(left: Any, right: Any) -> bool:
    """Compare a weights-only recovery artifact without coercing tensor values."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_state_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(
                _state_tree_equal(left_value, right_value)
                for left_value, right_value in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def _save_and_verify_on_chip_recovery_state(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist and immediately reload the exact stochastic-recovery cursor."""

    atomic_torch_save(dict(payload), path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    if not _state_tree_equal(payload, reloaded):
        raise RuntimeError(
            "Expected the on-chip recovery artifact to reload bit-exactly."
        )
    return {
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "artifact_reload_bit_exact": True,
    }


def population_nominal_step(population: IbmReramArrayPopulation) -> float:
    value = float(population.nominal_dw_min)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Expected a positive IBM OM nominal pulse scale.")
    return value
