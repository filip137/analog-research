from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class DataSettings:
    batch_size: int
    validation_points: int
    num_points: int | None
    shuffle: bool

@dataclass(frozen=True)
class ModelSettings:
    dims: tuple[int, int, int]
    bias: bool
