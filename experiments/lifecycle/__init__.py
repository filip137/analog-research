"""Run lifecycle: the phases every workflow-managed run executes.

A *phase* is one run bundle (one ``RunStore``).  ``train_phase`` holds the
parts every train phase shares: global-best selection, the epoch loop with
its epoch-boundary checkpoints, and terminal completion or failure.
Experiment families supply only what differs: how to build their stack,
train one epoch, validate, and describe a selection.
"""

from experiments.lifecycle.train_phase import (
    BestCheckpoint,
    EpochResult,
    run_epochs,
    run_phase,
)

__all__ = ["BestCheckpoint", "EpochResult", "run_epochs", "run_phase"]
