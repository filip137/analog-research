"""Program-and-verify controller settings shared by HWA and recovery."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from training.ibm_reram_program_verify import (
    ControllerSettings,
    PopulationStepEstimator,
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
