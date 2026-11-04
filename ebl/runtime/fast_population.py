"""Execution assembly for physical fast-rail populations."""
from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from networks.resistive.builders import ParameterBinding
from hardware.ibm_om.population import IbmReramArrayPopulation
from ebl.runtime.device_sampling import sample_om_array_population_layout_external


def sample_physical_fast_population(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
    aihwkit_python: Path,
    population_path: Path,
    receipt_path: Path,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    """Sample two repaired OM fast rails for every slow conductance."""

    selected = tuple(bindings)
    keys = tuple(f"fast_pair.{binding.key}" for binding in selected)
    shapes = tuple(
        (2 * int(binding.state.shape[0]), int(binding.state.shape[1]))
        for binding in selected
    )
    return sample_om_array_population_layout_external(
        keys,
        shapes,
        assignment_seed=assignment_seed,
        corruption_policy="counterfactual_repaired",
        aihwkit_python=aihwkit_python,
        population_path=population_path,
        receipt_path=receipt_path,
    )
