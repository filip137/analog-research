"""Execution ``compact_endpoint``: sample programmed endpoints statistically.

Cells the mapping declares out of support are programmed exactly on a
pulse-resolved fallback plant after the compact samples.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import torch

from experiments.reram_program_verify.hwa_model import CONDITION_KEY
from training.ibm_om.coordinates import LOGICAL
from training.ibm_om.plants import _ArrayPlant
from training.ibm_om.population import _selected_array_population
from training.ibm_om.programming.base import (
    ProgrammingContext,
    ProgrammingResult,
    ProgramRequest,
    _result_mapping,
)
from training.ibm_om.programming.controller import run_exact_programming
from training.ibm_reram_endpoint_model import sample_ibm_reram_endpoints
from training.ibm_reram_program_verify import ProgramVerifyResult


def program_compact(
    request: ProgramRequest,
    context: ProgrammingContext,
    *,
    generator: torch.Generator,
) -> ProgrammingResult:
    targets = request.targets
    population = context.population
    branch = context.calibration_branch
    model = context.device_model["endpoint_models"][branch]
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
    contract = request.out_of_support
    contract.check(below=int(below.sum().item()), above=int(above.sum().item()))
    if contract.mode == "pulse_fallback":
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
            target_out_of_support=context.target_out_of_support,
            endpoint_policy=context.endpoint_policy,
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
        fallback_result, fallback_plant = run_exact_programming(
            targets[pulse_fallback_mask],
            fallback_population,
            generator=generator,
            context=context,
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
        context.tolerance_step_ratio
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
    fallback_policy = contract.fallback_policy
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
        "controller": context.controller,
        "maximum_program_pulses": context.maximum_program_pulses,
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
    fallback_execution_detail = contract.execution_detail
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
        "device_model_sha256": context.device_model_sha256,
        "population_fingerprint": population.fingerprint,
        "binding_keys": population.binding_keys,
        "binding_shapes": population.binding_shapes,
        "endpoint_generation_policy": contract.endpoint_generation_policy,
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
    return ProgrammingResult(endpoint.to(dtype=targets.dtype), report, deployment)
