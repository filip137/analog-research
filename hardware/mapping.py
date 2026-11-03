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


@dataclass(frozen=True)
class FaultDomain:
    """Physical fault-group identity supplied by the network mapping."""

    group: str
    adapter: bool


DEVICES_PER_WEIGHT = {
    "differential_pair_25us": 2,
    "single_cell_intrinsic_reference": 1,
    # CMO: an active device and a mid-range reference device per weight.
    "active_reference_pair": 2,
}
