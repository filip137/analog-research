"""Execution ``pulse_resolved``: simulate program-and-verify pulse by pulse."""

from __future__ import annotations

import torch

from training.ibm_om.constants import (
    IBM_OM_RAW_ACTIVE_A_MAX,
    IBM_OM_RAW_ACTIVE_A_MIN,
    IBM_OM_RAW_ACTIVE_CONDITIONING_CHANGE_THRESHOLD,
    IBM_OM_RAW_ACTIVE_CONDITIONING_MAXIMUM_PULSES,
    IBM_OM_RAW_ACTIVE_CONDITIONING_QUIET_STEPS,
    IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
    IBM_OM_RAW_ACTIVE_SCALE,
    IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
)
from training.ibm_om.coordinates import LOGICAL, RAW_ACTIVE
from training.ibm_om.population import _tensor_summary
from training.ibm_om.programming.base import (
    ProgramRequest,
    ProgrammingContext,
    ProgrammingResult,
    _result_mapping,
)
from training.ibm_om.programming.controller import (
    run_exact_programming,
    run_raw_active_programming,
)
from training.ibm_reram_program_verify import OM_PRESET


def program_pulse_resolved(
    request: ProgramRequest,
    context: ProgrammingContext,
    *,
    generator: torch.Generator,
) -> ProgrammingResult:
    """Program every cell from its RESET start with the declared controller."""

    if context.initial_state == "conditioned_lower_boundary":
        return _program_raw_active(request, context, generator=generator)
    return _program_logical(request, context, generator=generator)


def _program_raw_active(
    request: ProgramRequest,
    context: ProgrammingContext,
    *,
    generator: torch.Generator,
) -> ProgrammingResult:
    targets = request.targets
    population = context.population
    result, plant, conditioning, masks = (
        run_raw_active_programming(
            targets,
            population,
            generator=generator,
            program_mask=request.program_mask,
            maximum_program_pulses=context.maximum_program_pulses,
            verify_tolerance=context.verify_tolerance_absolute,
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
        (targets + context.verify_tolerance_absolute >= lower)
        & (targets - context.verify_tolerance_absolute <= upper)
    )
    persistent_within_tolerance = (
        torch.abs(persistent_endpoint - targets)
        <= context.verify_tolerance_absolute
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
        "tolerance": context.verify_tolerance_absolute,
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
        "device_model_sha256": context.device_model_sha256,
        "population_fingerprint": population.fingerprint,
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
    return ProgrammingResult(apparent_endpoint.to(dtype=targets.dtype), report, deployment)


def _program_logical(
    request: ProgramRequest,
    context: ProgrammingContext,
    *,
    generator: torch.Generator,
) -> ProgrammingResult:
    targets = request.targets
    population = context.population
    result, plant = run_exact_programming(
        targets,
        population,
        generator=generator,
        context=context,
    )
    tolerance = (
        context.tolerance_step_ratio
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
        "device_model_sha256": context.device_model_sha256,
        "population_fingerprint": population.fingerprint,
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
    return ProgrammingResult(endpoint.to(dtype=targets.dtype), report, deployment)
