"""Torch IBM OM pulse plants driven by AIHWKit-sampled identities.

The plants reproduce AIHWKit's soft-bounds OM update with additive cycle
noise and persistent/apparent write noise.  Their random-draw order is part
of the numerical contract: ``__init__`` draws once for the full array and
every non-empty ``pulse`` draws twice for the full array (cycle, then write),
regardless of how many cells are active.
"""

from __future__ import annotations

import torch

from training.ibm_om.constants import (
    IBM_OM_RAW_ACTIVE_A_MIN,
    IBM_OM_RAW_ACTIVE_CONDITIONING_CHANGE_THRESHOLD,
    IBM_OM_RAW_ACTIVE_CONDITIONING_MAXIMUM_PULSES,
    IBM_OM_RAW_ACTIVE_CONDITIONING_QUIET_STEPS,
    IBM_OM_RAW_ACTIVE_SCALE,
)
from training.ibm_om.population import IbmReramArrayPopulation


class _ArrayPlant:
    """Network-scale pulse plant using one exact checkpointable RNG stream."""

    def __init__(
        self,
        population: IbmReramArrayPopulation,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        self.population = population.to(device)
        self.device = device
        self.generator = generator
        self.persistent = self.population.logical_min.clone()
        write_scale = self.population.write_noise_std * self.population.nominal_dw_min
        self.apparent = self.persistent + write_scale * self._normal_all()

    @property
    def size(self) -> int:
        return self.population.size

    def _normal_all(self) -> torch.Tensor:
        return torch.randn(
            (self.size,),
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        )

    def pulse(self, directions: torch.Tensor) -> None:
        direction = torch.as_tensor(directions, dtype=torch.int8, device=self.device)
        if direction.shape != (self.size,) or bool(
            torch.any((direction < -1) | (direction > 1))
        ):
            raise ValueError("Expected one {-1,0,1} pulse direction per device.")
        active = direction != 0
        if not bool(torch.any(active)):
            return
        population = self.population
        physical = self.persistent + population.reference
        cycle = self._normal_all()
        candidate = physical.clone()
        up = direction > 0
        down = direction < 0
        if bool(torch.any(up)):
            normalized = torch.where(
                population.max_bound > 0.0,
                physical / population.max_bound,
                torch.zeros_like(physical),
            )
            response = population.dwmin_up * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[up] = physical[up] + response[up]
        if bool(torch.any(down)):
            normalized = torch.where(
                population.min_bound < 0.0,
                physical / population.min_bound,
                torch.zeros_like(physical),
            )
            response = population.dwmin_down * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[down] = physical[down] - response[down]
        candidate = torch.maximum(candidate, population.min_bound)
        candidate = torch.minimum(candidate, population.max_bound)
        self.persistent[active] = (candidate - population.reference)[active]
        write_scale = population.write_noise_std * population.nominal_dw_min
        apparent = self.persistent + write_scale * self._normal_all()
        self.apparent[active] = apparent[active]

    def controller_port(self) -> "_ArrayControllerPort":
        return _ArrayControllerPort(self)

    def set_generator(self, generator: torch.Generator) -> None:
        """Continue on another explicit RNG stream; the state is unchanged."""

        self.generator = generator

    def load_state(
        self,
        *,
        persistent: torch.Tensor,
        apparent: torch.Tensor,
    ) -> None:
        """Overwrite the full persistent and apparent states in place."""

        self.persistent.copy_(persistent)
        self.apparent.copy_(apparent)

    def read_cells(
        self,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return copies of the persistent and apparent states of ``indices``."""

        return self.persistent[indices], self.apparent[indices]

    def write_cells(
        self,
        indices: torch.Tensor,
        *,
        persistent: torch.Tensor,
        apparent: torch.Tensor,
    ) -> None:
        """Overwrite the persistent and apparent states of ``indices``."""

        self.persistent[indices] = persistent
        self.apparent[indices] = apparent



IbmReramPulsePlant = _ArrayPlant



class _ArrayControllerPort:
    def __init__(self, plant: _ArrayPlant) -> None:
        self._plant = plant

    @property
    def size(self) -> int:
        return self._plant.size

    def verify(self) -> torch.Tensor:
        return (self._plant.apparent.clone() + 1.0) / 2.0

    def apply_identical_pulses(
        self,
        directions: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        direction = torch.as_tensor(
            directions, dtype=torch.int8, device=self._plant.device
        )
        count = torch.as_tensor(counts, dtype=torch.int64, device=self._plant.device)
        if direction.shape != (self.size,) or count.shape != (self.size,) or bool(
            torch.any(count < 0)
        ):
            raise ValueError("Expected valid directions and pulse counts per device.")
        for pulse_index in range(int(count.max().item()) if count.numel() else 0):
            self._plant.pulse(
                torch.where(
                    count > pulse_index,
                    direction,
                    torch.zeros_like(direction),
                )
            )



class _RawActiveArrayPlant:
    """Explicit IBM OM plant whose persistent state is the active state ``a``.

    Unlike :class:`_ArrayPlant`, this plant never subtracts or consults the
    AIHWKit reference tensor.  It starts at raw ``a=0`` and exposes only the
    frozen array-wide ``g`` coordinate through its controller port.
    """

    def __init__(
        self,
        population: IbmReramArrayPopulation,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        self.population = population.to(device)
        self.device = device
        self.generator = generator
        self.persistent_a = torch.zeros(
            population.size, dtype=torch.float32, device=device
        )
        self.apparent_a = self.persistent_a.clone()

    @property
    def size(self) -> int:
        return self.population.size

    def _normal_all(self) -> torch.Tensor:
        return torch.randn(
            (self.size,),
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        )

    def pulse(self, directions: torch.Tensor) -> None:
        direction = torch.as_tensor(
            directions, dtype=torch.int8, device=self.device
        )
        if direction.shape != (self.size,) or bool(
            torch.any((direction < -1) | (direction > 1))
        ):
            raise ValueError(
                "Expected one raw-active {-1,0,1} pulse direction per cell."
            )
        active = direction != 0
        if not bool(torch.any(active)):
            return
        population = self.population
        current = self.persistent_a
        cycle = self._normal_all()
        candidate = current.clone()
        up = direction > 0
        down = direction < 0
        if bool(torch.any(up)):
            normalized = torch.where(
                population.max_bound > 0.0,
                current / population.max_bound,
                torch.zeros_like(current),
            )
            response = population.dwmin_up * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[up] = current[up] + response[up]
        if bool(torch.any(down)):
            normalized = torch.where(
                population.min_bound < 0.0,
                current / population.min_bound,
                torch.zeros_like(current),
            )
            response = population.dwmin_down * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[down] = current[down] - response[down]
        candidate = torch.maximum(candidate, population.min_bound)
        candidate = torch.minimum(candidate, population.max_bound)
        self.persistent_a[active] = candidate[active]
        write_scale = (
            population.write_noise_std * population.nominal_dw_min
        )
        apparent = self.persistent_a + write_scale * self._normal_all()
        self.apparent_a[active] = apparent[active]

    def condition_lower_boundary(self) -> dict[str, torch.Tensor]:
        active = torch.ones(
            self.size, dtype=torch.bool, device=self.device
        )
        consecutive = torch.zeros(
            self.size, dtype=torch.int64, device=self.device
        )
        pulse_count = torch.zeros_like(consecutive)
        for _pulse_index in range(
            IBM_OM_RAW_ACTIVE_CONDITIONING_MAXIMUM_PULSES
        ):
            before = self.persistent_a.clone()
            self.pulse(-active.to(dtype=torch.int8))
            change = torch.abs(self.persistent_a - before)
            consecutive = torch.where(
                active
                & (
                    change
                    < IBM_OM_RAW_ACTIVE_CONDITIONING_CHANGE_THRESHOLD
                ),
                consecutive + 1,
                torch.where(
                    active,
                    torch.zeros_like(consecutive),
                    consecutive,
                ),
            )
            pulse_count[active] += 1
            active = consecutive < IBM_OM_RAW_ACTIVE_CONDITIONING_QUIET_STEPS
            if not bool(torch.any(active)):
                break
        return {
            "success": ~active,
            "pulse_count": pulse_count,
            "persistent_a": self.persistent_a.clone(),
            "apparent_a": self.apparent_a.clone(),
        }

    def controller_port(self) -> "_RawActiveArrayControllerPort":
        return _RawActiveArrayControllerPort(self)

    def set_generator(self, generator: torch.Generator) -> None:
        """Continue on another explicit RNG stream; the state is unchanged."""

        self.generator = generator



class _RawActiveArrayControllerPort:
    """Capability-limited apparent ``g`` view of a raw-active plant."""

    def __init__(self, plant: _RawActiveArrayPlant) -> None:
        self.__plant = plant

    @property
    def size(self) -> int:
        return self.__plant.size

    def verify(self) -> torch.Tensor:
        return (
            self.__plant.apparent_a.clone() - IBM_OM_RAW_ACTIVE_A_MIN
        ) / IBM_OM_RAW_ACTIVE_SCALE

    def apply_identical_pulses(
        self,
        directions: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        direction = torch.as_tensor(
            directions, dtype=torch.int8, device=self.__plant.device
        )
        count = torch.as_tensor(
            counts, dtype=torch.int64, device=self.__plant.device
        )
        if (
            direction.shape != (self.size,)
            or count.shape != (self.size,)
            or bool(torch.any(count < 0))
        ):
            raise ValueError(
                "Expected valid raw-active pulse directions and counts."
            )
        maximum = int(count.max().item()) if count.numel() else 0
        for pulse_index in range(maximum):
            self.__plant.pulse(
                torch.where(
                    count > pulse_index,
                    direction,
                    torch.zeros_like(direction),
                )
            )
