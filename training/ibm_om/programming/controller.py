"""Program-and-verify controller settings and the two exact programming runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from training.ibm_om.coordinates import RAW_ACTIVE
from training.ibm_om.plants import _ArrayPlant, _RawActiveArrayPlant
from training.ibm_om.population import IbmReramArrayPopulation
from training.ibm_om.programming.base import ProgrammingContext
from training.ibm_reram_program_verify import (
    ControllerSettings,
    PopulationStepEstimator,
    ProgramVerifyResult,
    derive_seed,
    run_program_verify,
)


def adaptive_controller_settings(
    device_model: Mapping[str, Any],
) -> ControllerSettings:
    """Return the device model's calibrated adaptive controller settings."""

    adaptive = device_model["programming"]["adaptive"]
    return ControllerSettings(
        kind="adaptive",
        eta=float(adaptive["eta"]),
        maximum_batch=int(adaptive["maximum_batch"]),
        epsilon=float(adaptive["epsilon"]),
        force_one_within_steps=float(adaptive["force_one_within_steps"]),
    )


def step_estimator(
    device_model: Mapping[str, Any],
    branch: str,
) -> PopulationStepEstimator:
    """Return the calibrated step estimator for ``continuous`` or
    ``published_corruption`` populations."""

    return PopulationStepEstimator.from_mapping(
        device_model["step_estimators"][branch]
    )


def step_ratio_tolerance(step_ratio: float, nominal_dw_min: float) -> float:
    """Return the verify tolerance as a fraction of one nominal pulse."""

    return step_ratio * nominal_dw_min / 2.0


def run_exact_programming(
    targets: torch.Tensor,
    population: IbmReramArrayPopulation,
    *,
    generator: torch.Generator,
    context: ProgrammingContext,
) -> tuple[ProgramVerifyResult, _ArrayPlant]:
    """Run the declared controller on exactly the supplied identities."""

    plant = _ArrayPlant(population, generator=generator, device=targets.device)
    branch = context.calibration_branch
    estimator = PopulationStepEstimator.from_mapping(
        context.device_model["step_estimators"][branch]
    )
    adaptive = context.device_model["programming"]["adaptive"]
    settings = ControllerSettings(
        kind="adaptive",
        eta=float(adaptive["eta"]),
        maximum_batch=int(adaptive["maximum_batch"]),
        epsilon=float(adaptive["epsilon"]),
        force_one_within_steps=float(adaptive["force_one_within_steps"]),
    )
    tolerance = (
        context.tolerance_step_ratio
        * population.nominal_dw_min
        / 2.0
    )
    result = run_program_verify(
        plant.controller_port(),
        targets=targets,
        tolerance=tolerance,
        maximum_pulses=context.maximum_program_pulses,
        settings=settings,
        estimator=estimator,
    )
    return result, plant


def run_raw_active_programming(
    targets: torch.Tensor,
    population: IbmReramArrayPopulation,
    *,
    generator: torch.Generator,
    program_mask: torch.Tensor,
    maximum_program_pulses: int,
    verify_tolerance: float,
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
    structural = program_mask
    lower, upper = RAW_ACTIVE.cell_bounds(population, device=targets.device)
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
        tolerance=verify_tolerance,
        maximum_pulses=maximum_program_pulses,
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
