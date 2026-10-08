"""Reusable crossbar experiment lifecycle.

A campaign owns lifecycle definitions (``campaigns/<campaign>/lifecycles``).
Each lifecycle stage calls one reusable function here:

- ``workflow.devices``: devices and their permanent defects (fresh arrays);
- ``workflow.deployment``: mapping of the digital weights onto those arrays;
- ``workflow.hwa``: optional noise- or corruption-aware off-chip training;
- ``workflow.program_verify``: programming, verify cycles and relaxation;
- ``workflow.onchip``: same-array on-chip training and matched controls.

``workflow.networks`` holds one module per network family (ResNet-32 CIFAR
suffix, OPT with analog decoder MLPs); the stages reach the network only
through it. ``workflow.lifecycle`` is the strict stdlib-only schema, ``workflow.runtime``
the native ``ebl train`` entry point for one stage and ``workflow.plan`` the
generator of stage configs and an ``ebl campaign run`` manifest. This package
keeps its import light: numerical modules load only when a stage runs.
"""

from workflow.lifecycle import (
    EXPERIMENT_ID,
    Lifecycle,
    StageConfig,
    load_lifecycle,
    parse_lifecycle,
    parse_stage_config,
)

__all__ = [
    "EXPERIMENT_ID",
    "Lifecycle",
    "StageConfig",
    "load_lifecycle",
    "parse_lifecycle",
    "parse_stage_config",
]
