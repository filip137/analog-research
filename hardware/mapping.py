"""hardware.mapping — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    shape: tuple[int, ...]
    offset: int
    size: int
    scale: float
