"""Teacher targets prepared at the training-engine batch seam."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional
import torch
from methods.core.batch import Batch, prepared_batches

@dataclass(frozen=True)
class TeacherTargets:
    """Teacher logits and labels for one batch; ``len`` is the example count."""

    logits: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])

def teacher_batches(
    loader: Iterable,
    teacher: Any,
    device: Any,
    *,
    maximum_batches: Optional[int] = None,
    sample_limit: Optional[int] = None,
    stop: Optional[Callable[[], bool]] = None,
    before_transfer: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
):
    """Engine batches of ``(inputs, TeacherTargets)`` from an MNIST loader.

    ``before_transfer`` sees the loader-side tensors after truncation, e.g. to
    hash the exact cohort. Teacher logits are computed without gradients on
    the transferred inputs, when the engine pulls the batch.
    """

    def prepare(batch: Batch) -> Batch:
        if before_transfer is not None:
            before_transfer(batch.inputs, batch.targets)
        inputs = batch.inputs.to(device, dtype=torch.float32)
        labels = batch.targets.to(device, dtype=torch.long)
        with torch.no_grad():
            logits = teacher.logits(inputs)
        return Batch(inputs, TeacherTargets(logits, labels))

    return prepared_batches(
        loader,
        maximum_batches=maximum_batches,
        sample_limit=sample_limit,
        prepare=prepare,
        stop=stop,
    )
