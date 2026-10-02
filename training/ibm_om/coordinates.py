"""Device coordinates: how one physical OM cell state maps onto [0, 1].

Two coordinates exist.  ``logical_reference_subtracted`` reads the
AIHWKit logical state ``w = internal - reference`` as ``(w + 1) / 2``.
``raw_active_v1`` reads the raw active state ``a`` through the frozen
array-wide affine map ``(a - A_MIN) / SCALE`` and never consults the
reference.  Mapping uses a coordinate for per-cell bounds, programming for
its plant and endpoint readout, so both see one shared, declared choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from training.ibm_om.constants import (
    IBM_OM_RAW_ACTIVE_A_MIN,
    IBM_OM_RAW_ACTIVE_SCALE,
)
from training.ibm_om.plants import _ArrayPlant, _RawActiveArrayPlant
from training.ibm_om.population import IbmReramArrayPopulation


LOGICAL_REFERENCE_SUBTRACTED = "logical_reference_subtracted"
RAW_ACTIVE_V1 = "raw_active_v1"
DEVICE_COORDINATES = (LOGICAL_REFERENCE_SUBTRACTED, RAW_ACTIVE_V1)


def _prepare(
    value: torch.Tensor,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if device is None and dtype is None:
        return value
    if dtype is None:
        return value.to(device=device)
    return value.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class DeviceCoordinate:
    """One declared cell coordinate and the plant that realizes it."""

    name: str
    _lower_state: Callable[[IbmReramArrayPopulation], torch.Tensor]
    _upper_state: Callable[[IbmReramArrayPopulation], torch.Tensor]
    _from_state: Callable[[torch.Tensor], torch.Tensor]
    _plant_type: type

    def from_state(self, value: torch.Tensor) -> torch.Tensor:
        """Convert a plant state tensor into this coordinate."""

        return self._from_state(value)

    def cell_bounds(
        self,
        population: IbmReramArrayPopulation,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return each cell's reachable [lower, upper] in this coordinate."""

        lower = self._from_state(
            _prepare(self._lower_state(population), device, dtype)
        )
        upper = self._from_state(
            _prepare(self._upper_state(population), device, dtype)
        )
        return lower, upper

    def new_plant(
        self,
        population: IbmReramArrayPopulation,
        *,
        generator: torch.Generator,
        device: torch.device,
    ):
        """Return a fresh plant whose controller port reads this coordinate."""

        return self._plant_type(population, generator=generator, device=device)


def _logical_from_state(value: torch.Tensor) -> torch.Tensor:
    return (value + 1.0) / 2.0


def _raw_active_from_state(value: torch.Tensor) -> torch.Tensor:
    return (value - IBM_OM_RAW_ACTIVE_A_MIN) / IBM_OM_RAW_ACTIVE_SCALE


LOGICAL = DeviceCoordinate(
    name=LOGICAL_REFERENCE_SUBTRACTED,
    _lower_state=lambda population: population.logical_min,
    _upper_state=lambda population: population.logical_max,
    _from_state=_logical_from_state,
    _plant_type=_ArrayPlant,
)

RAW_ACTIVE = DeviceCoordinate(
    name=RAW_ACTIVE_V1,
    _lower_state=lambda population: population.min_bound,
    _upper_state=lambda population: population.max_bound,
    _from_state=_raw_active_from_state,
    _plant_type=_RawActiveArrayPlant,
)

_BY_NAME = {coordinate.name: coordinate for coordinate in (LOGICAL, RAW_ACTIVE)}


def coordinate_for(name: str) -> DeviceCoordinate:
    """Return the declared device coordinate named ``name``."""

    try:
        return _BY_NAME[name]
    except KeyError as error:
        raise ValueError(
            f"Expected device_coordinate to be one of {DEVICE_COORDINATES!r}. "
            f"Provided value: {name!r}."
        ) from error
