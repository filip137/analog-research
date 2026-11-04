"""Translate declared DRN bindings into physical population metadata."""
from __future__ import annotations
from collections.abc import Sequence
from networks.resistive.builders import ParameterBinding
from networks.variable.parameter import DenseWeight
from hardware.ibm_om.population import IbmReramArrayPopulation
from hardware.ibm_om.sampling import _sample_om_array_population_layout


def _binding_layout(
    bindings: Sequence[ParameterBinding],
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    selected = tuple(bindings)
    if not selected or any(
        not isinstance(binding.parameter, DenseWeight) for binding in selected
    ):
        raise ValueError(
            "Expected at least one named DenseWeight binding for IBM OM HWA."
        )
    keys = tuple(binding.key for binding in selected)
    shapes = tuple(tuple(binding.state.shape) for binding in selected)
    if len(set(keys)) != len(keys):
        raise ValueError("Expected unique IBM OM DenseWeight binding keys.")
    return keys, shapes

def sample_om_array_population(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
    corruption_policy: str,
) -> IbmReramArrayPopulation:
    """Sample a literal published assignment, optionally repairing defects only."""

    keys, shapes = _binding_layout(bindings)
    return _sample_om_array_population_layout(
        keys,
        shapes,
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
    )
