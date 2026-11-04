from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from core.components import ComponentSettings


@dataclass(frozen=True)
class TrainSettings:
    num_epochs: int
    algorithm: str
    learning_rates: Tuple[float, ...]
    bias_learning_rates: Tuple[float, ...]
    nudging: float
    log_every: int
    max_batches: Optional[int]
    max_validation_batches: Optional[int]
    weight_modifier: ComponentSettings
    update_backend: ComponentSettings
    learning_rate_selection: ComponentSettings
