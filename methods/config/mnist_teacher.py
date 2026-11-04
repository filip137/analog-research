from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class TeacherTrainSettings:
    num_epochs: int
    learning_rate: float
    weight_decay: float
    log_every: int
    max_batches: int | None
    max_validation_batches: int | None
    minimum_validation_accuracy: float
