"""Execution ``mapped_target``: the mapped target is the endpoint."""

from __future__ import annotations


import torch

from training.ibm_om.programming.base import (
    ProgrammingContext,
    ProgrammingResult,
    ProgramRequest,
)


def program_mapped_target(
    request: ProgramRequest,
    context: ProgrammingContext,
    *,
    generator: torch.Generator,
) -> ProgrammingResult:
    """Apply the raw p90 mapper without inventing a compact P&V fit.

    Eligible quads receive their exact continuous or seven-level target.
    A quad outside the frozen D90 support is an explicit structural
    failure and is represented at its transformed RESET bound for this
    deterministic training surrogate.  Exact deployment still performs
    boundary conditioning and one-pulse P&V.
    """

    if request.program_mask is None:
        raise RuntimeError(
            "Expected mapped_target execution only for raw-active p90."
        )
    targets = request.targets
    population = context.population
    structural = request.program_mask
    lower, upper = context.coordinate.cell_bounds(population, device=targets.device)
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
        "device_model_sha256": context.device_model_sha256,
        "population_fingerprint": population.fingerprint,
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
        "report": report,
    }
    return ProgrammingResult(endpoint.to(dtype=targets.dtype), report, deployment)
