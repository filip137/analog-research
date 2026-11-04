"""Explicit-RNG ReRAM plants and their restricted observation ports."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Callable, Iterable, Mapping, Protocol, Sequence
import torch
from hardware.ibm_om.constants import OM_PRESET, HFO2_PRESET, SUPPORTED_PRESETS, PUBLISHED_CORRUPT_PROBABILITY

def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable positive 63-bit seed from a recorded base seed."""

    digest = sha256(
        "\x1f".join((str(int(base_seed)), *(str(part) for part in parts))).encode(
            "utf-8"
        )
    ).digest()
    # Torch accepts signed 64-bit seeds.  Avoid zero because AIHWKit treats a
    # zero construction seed specially.
    return int.from_bytes(digest[:8], "little") % (2**63 - 1) + 1

@dataclass(frozen=True)
class IbmReramPopulation:
    """Fixed device-to-device parameters for independent ReRAM identities."""

    preset: str
    aihwkit_version: str
    nominal_dw_min: float
    dw_min_std: float
    write_noise_std: float
    mult_noise: bool
    construction_seeds: torch.Tensor
    max_bound: torch.Tensor
    min_bound: torch.Tensor
    dwmin_up: torch.Tensor
    dwmin_down: torch.Tensor
    reference: torch.Tensor
    corrupt: torch.Tensor
    preset_parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        size = int(self.max_bound.numel())
        if size < 1:
            raise ValueError("Expected at least one IBM ReRAM device identity.")
        names = (
            "construction_seeds",
            "min_bound",
            "dwmin_up",
            "dwmin_down",
            "reference",
            "corrupt",
        )
        for name in names:
            value = getattr(self, name)
            if value.ndim != 1 or int(value.numel()) != size:
                raise ValueError(
                    "Expected every IBM ReRAM population tensor to be a "
                    f"one-dimensional vector of length {size}. Provided "
                    f"value: {name} shape={tuple(value.shape)!r}."
                )
        if self.construction_seeds.dtype != torch.int64:
            raise ValueError(
                "Expected construction_seeds to use torch.int64. "
                f"Provided value: {self.construction_seeds.dtype}."
            )
        if self.corrupt.dtype != torch.bool:
            raise ValueError(
                "Expected corrupt flags to use torch.bool. "
                f"Provided value: {self.corrupt.dtype}."
            )
        for name in ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference"):
            value = getattr(self, name)
            if value.dtype != torch.float32 or not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    "Expected sampled IBM ReRAM device parameters to be finite "
                    f"torch.float32 vectors. Provided value: {name} dtype={value.dtype}."
                )
        scalars = {
            "nominal_dw_min": self.nominal_dw_min,
            "dw_min_std": self.dw_min_std,
            "write_noise_std": self.write_noise_std,
        }
        if (
            not math.isfinite(self.nominal_dw_min)
            or self.nominal_dw_min <= 0.0
            or not math.isfinite(self.dw_min_std)
            or self.dw_min_std < 0.0
            or not math.isfinite(self.write_noise_std)
            or self.write_noise_std < 0.0
        ):
            raise ValueError(
                "Expected a positive nominal step and non-negative finite noise "
                f"scales. Provided value: {scalars!r}."
            )
        if bool(torch.any(self.max_bound < self.min_bound)):
            raise ValueError("Expected every sampled maximum bound to exceed its minimum bound.")
        detected_corrupt = (
            (torch.abs(self.max_bound - self.min_bound) <= 1e-12)
            & (self.dwmin_up == 0.0)
            & (self.dwmin_down == 0.0)
        )
        if not torch.equal(detected_corrupt, self.corrupt):
            raise ValueError(
                "Expected corrupt flags to identify exactly the collapsed, zero-step devices."
            )
        if self.preset not in SUPPORTED_PRESETS:
            raise ValueError(
                f"Expected preset to be one of {SUPPORTED_PRESETS!r}. "
                f"Provided value: {self.preset!r}."
            )
        if self.mult_noise:
            raise ValueError(
                "Expected the IBM array presets to use additive cycle-to-cycle "
                "noise. Provided value: mult_noise=True."
            )

    @property
    def size(self) -> int:
        return int(self.max_bound.numel())

    @property
    def logical_min(self) -> torch.Tensor:
        return self.min_bound - self.reference

    @property
    def logical_max(self) -> torch.Tensor:
        return self.max_bound - self.reference

    def select(self, indices: torch.Tensor | Sequence[int]) -> "IbmReramPopulation":
        index = torch.as_tensor(indices, dtype=torch.long)
        return IbmReramPopulation(
            preset=self.preset,
            aihwkit_version=self.aihwkit_version,
            nominal_dw_min=self.nominal_dw_min,
            dw_min_std=self.dw_min_std,
            write_noise_std=self.write_noise_std,
            mult_noise=self.mult_noise,
            construction_seeds=self.construction_seeds[index].clone(),
            max_bound=self.max_bound[index].clone(),
            min_bound=self.min_bound[index].clone(),
            dwmin_up=self.dwmin_up[index].clone(),
            dwmin_down=self.dwmin_down[index].clone(),
            reference=self.reference[index].clone(),
            corrupt=self.corrupt[index].clone(),
            preset_parameters=self.preset_parameters,
        )

    def to(self, device: torch.device | str) -> "IbmReramPopulation":
        """Move sampled per-identity tensors without changing device parameters."""

        target = torch.device(device)
        return IbmReramPopulation(
            preset=self.preset,
            aihwkit_version=self.aihwkit_version,
            nominal_dw_min=self.nominal_dw_min,
            dw_min_std=self.dw_min_std,
            write_noise_std=self.write_noise_std,
            mult_noise=self.mult_noise,
            construction_seeds=self.construction_seeds.to(target),
            max_bound=self.max_bound.to(target),
            min_bound=self.min_bound.to(target),
            dwmin_up=self.dwmin_up.to(target),
            dwmin_down=self.dwmin_down.to(target),
            reference=self.reference.to(target),
            corrupt=self.corrupt.to(target),
            preset_parameters=self.preset_parameters,
        )

def _plain_parameter(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_parameter(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_parameter(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain_parameter(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)

def _load_aihwkit_preset(preset: str):
    try:
        import aihwkit
        from aihwkit.simulator.presets.devices import (
            ReRamArrayHfO2PresetDevice,
            ReRamArrayOMPresetDevice,
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Expected AIHWKit to sample the published IBM ReRAM preset "
            f"parameters. Provided value: import failed ({error})."
        ) from error

    classes = {
        OM_PRESET: ReRamArrayOMPresetDevice,
        HFO2_PRESET: ReRamArrayHfO2PresetDevice,
    }
    try:
        preset_class = classes[preset]
    except KeyError as error:
        raise ValueError(
            f"Expected preset to be one of {SUPPORTED_PRESETS!r}. "
            f"Provided value: {preset!r}."
        ) from error
    return aihwkit, preset_class

def sample_ibm_reram_population(
    *,
    preset: str,
    num_devices: int,
    construction_seed: int,
    enable_published_corruption: bool,
    required_aihwkit_version: str | None = "1.1.0",
) -> IbmReramPopulation:
    """Sample independent preset identities with order-stable construction seeds.

    One 1x1 tile is constructed per identity.  Consequently device ``i`` is
    fully determined by ``derive_seed(construction_seed, preset, i)`` and does
    not change when the population is chunked or traversed differently.
    """

    if isinstance(num_devices, bool) or not isinstance(num_devices, int) or num_devices < 1:
        raise ValueError(
            "Expected num_devices to be a positive integer. "
            f"Provided value: {num_devices!r}."
        )
    aihwkit, preset_class = _load_aihwkit_preset(preset)
    version = str(getattr(aihwkit, "__version__", "unknown"))
    if required_aihwkit_version is not None and version != required_aihwkit_version:
        raise RuntimeError(
            "Expected the declared AIHWKit reference version to match exactly. "
            f"Provided value: required={required_aihwkit_version!r}, installed={version!r}."
        )

    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.tiles import AnalogTile

    template = preset_class()
    if enable_published_corruption:
        template.corrupt_devices_prob = PUBLISHED_CORRUPT_PROBABILITY[preset]
    parameters = _plain_parameter(vars(template))
    if not isinstance(parameters, Mapping):  # pragma: no cover - defensive
        raise RuntimeError("Expected preset parameters to normalize to a mapping.")

    names = ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference")
    sampled: dict[str, list[float]] = {name: [] for name in names}
    corrupt: list[bool] = []
    seeds: list[int] = []
    for device_id in range(num_devices):
        device = preset_class()
        if enable_published_corruption:
            device.corrupt_devices_prob = PUBLISHED_CORRUPT_PROBABILITY[preset]
        seed = derive_seed(construction_seed, preset, device_id)
        # AIHWKit's C++ construction seed is an int; keep it positive and in
        # the portable signed-31-bit range.
        native_seed = seed % (2**31 - 2) + 1
        device.construction_seed = int(native_seed)
        tile = AnalogTile(1, 1, SingleRPUConfig(device=device), bias=False)
        hidden = tile.get_hidden_parameters()
        missing = tuple(name for name in names if name not in hidden)
        if missing:
            raise RuntimeError(
                "Expected AIHWKit SoftBoundsReference hidden parameters. "
                f"Provided value: missing={missing!r}, available={tuple(hidden)!r}."
            )
        for name in names:
            sampled[name].append(float(hidden[name].reshape(-1)[0].item()))
        is_corrupt = (
            abs(sampled["max_bound"][-1] - sampled["min_bound"][-1]) <= 1e-12
            and sampled["dwmin_up"][-1] == 0.0
            and sampled["dwmin_down"][-1] == 0.0
        )
        corrupt.append(is_corrupt)
        seeds.append(int(native_seed))

    if len(set(seeds)) != len(seeds):
        raise RuntimeError(
            "Expected identity-derived native AIHWKit construction seeds to "
            "be unique within the sampled population. Provided value: a seed "
            "collision; choose a different construction_seed."
        )

    tensor = lambda values: torch.tensor(values, dtype=torch.float32)
    return IbmReramPopulation(
        preset=preset,
        aihwkit_version=version,
        nominal_dw_min=float(template.dw_min),
        dw_min_std=float(template.dw_min_std),
        write_noise_std=float(template.write_noise_std),
        mult_noise=bool(template.mult_noise),
        construction_seeds=torch.tensor(seeds, dtype=torch.int64),
        max_bound=tensor(sampled["max_bound"]),
        min_bound=tensor(sampled["min_bound"]),
        dwmin_up=tensor(sampled["dwmin_up"]),
        dwmin_down=tensor(sampled["dwmin_down"]),
        reference=tensor(sampled["reference"]),
        corrupt=torch.tensor(corrupt, dtype=torch.bool),
        preset_parameters=dict(parameters),
    )

class VerifyPort(Protocol):
    """Information and actions available to a realistic P&V controller."""

    @property
    def size(self) -> int: ...

    def verify(self) -> torch.Tensor: ...

    def apply_identical_pulses(
        self,
        directions: torch.Tensor,
        counts: torch.Tensor,
    ) -> None: ...

def make_buffered_normal_draws(
    seeds: Sequence[int],
    *,
    maximum_random_draws: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Generate independent per-trajectory streams on CPU, then stage them.

    AIHWKit's published device-to-device sampling remains in its pinned CPU
    environment.  CUDA only evaluates the explicit pulse equation.  Drawing
    each trajectory's complete stream with an independently seeded CPU
    generator preserves identity/order independence while avoiding millions
    of scalar RNG dispatches during the GPU pulse loop.
    """

    if (
        isinstance(maximum_random_draws, bool)
        or not isinstance(maximum_random_draws, int)
        or maximum_random_draws < 1
    ):
        raise ValueError(
            "Expected maximum_random_draws to be a positive integer. "
            f"Provided value: {maximum_random_draws!r}."
        )
    staged = torch.empty(
        (len(seeds), maximum_random_draws), dtype=torch.float32, device="cpu"
    )
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        torch.randn(
            (maximum_random_draws,),
            generator=generator,
            out=staged[index],
        )
    return staged.to(torch.device(device))

class IbmReramPlant:
    """Vectorized pulse plant with independent explicit RNG streams."""

    def __init__(
        self,
        population: IbmReramPopulation,
        *,
        seeds: Sequence[int],
        device: torch.device | str = "cpu",
        maximum_random_draws: int | None = None,
        normal_draws: torch.Tensor | None = None,
    ) -> None:
        if len(seeds) != population.size:
            raise ValueError(
                "Expected one pulse seed per device trajectory. "
                f"Provided value: seeds={len(seeds)}, devices={population.size}."
            )
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "Expected the IBM ReRAM pulse plant device to be 'cpu' or "
                f"'cuda'. Provided value: {str(self.device)!r}."
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Expected CUDA to be available for the requested IBM ReRAM "
                "pulse-plant execution device."
            )
        self.population = population.to(self.device)
        self.persistent = torch.zeros(
            population.size, dtype=torch.float32, device=self.device
        )
        self.apparent = self.persistent.clone()
        self._generators: list[torch.Generator] = []
        self._seeds: tuple[int, ...] = ()
        self._maximum_random_draws = maximum_random_draws
        self._normal_draws: torch.Tensor | None = None
        self._draw_indices: torch.Tensor | None = None
        self._configure_rng(seeds, normal_draws=normal_draws)

    @property
    def size(self) -> int:
        return self.population.size

    @property
    def rng_backend(self) -> str:
        return (
            "per_trajectory_torch_cpu"
            if self.device.type == "cpu"
            else "per_trajectory_buffered_torch_cpu"
        )

    def _configure_rng(
        self,
        seeds: Sequence[int],
        *,
        normal_draws: torch.Tensor | None,
    ) -> None:
        if len(seeds) != self.size:
            raise ValueError(
                "Expected one seed for each plant element. "
                f"Provided value: {len(seeds)!r}."
            )
        self._seeds = tuple(int(seed) for seed in seeds)
        if self.device.type == "cpu":
            if normal_draws is not None:
                raise ValueError(
                    "Expected externally buffered normal draws only for CUDA execution."
                )
            generators = []
            for seed in self._seeds:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                generators.append(generator)
            self._generators = generators
            self._normal_draws = None
            self._draw_indices = None
            return

        maximum = self._maximum_random_draws
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError(
                "Expected CUDA pulse plants to declare a positive "
                "maximum_random_draws budget. "
                f"Provided value: {maximum!r}."
            )
        if normal_draws is None:
            draws = make_buffered_normal_draws(
                self._seeds,
                maximum_random_draws=maximum,
                device=self.device,
            )
        else:
            draws = torch.as_tensor(
                normal_draws, dtype=torch.float32, device=self.device
            )
            expected = (self.size, maximum)
            if draws.shape != expected or not bool(torch.all(torch.isfinite(draws))):
                raise ValueError(
                    "Expected buffered CUDA normal draws to be finite float32 "
                    f"with shape {expected!r}. Provided value: "
                    f"shape={tuple(draws.shape)!r}, dtype={draws.dtype}."
                )
        self._generators = []
        self._normal_draws = draws
        self._draw_indices = torch.zeros(
            self.size, dtype=torch.int64, device=self.device
        )

    def set_seeds(self, seeds: Sequence[int]) -> None:
        self._configure_rng(seeds, normal_draws=None)

    def reset_logical_zero(self) -> None:
        """Reset simulator state before declared boundary conditioning."""

        self.persistent.zero_()
        self.apparent.zero_()

    def state_dict(self) -> dict[str, object]:
        """Return exact pulse-continuation state without exposing it to a controller."""

        if self.device.type == "cuda":
            assert self._draw_indices is not None
            return {
                "schema_version": 2,
                "preset": self.population.preset,
                "construction_seeds": self.population.construction_seeds.detach().cpu(),
                "persistent": self.persistent.clone(),
                "apparent": self.apparent.clone(),
                "rng_backend": self.rng_backend,
                "seeds": list(self._seeds),
                "maximum_random_draws": self._maximum_random_draws,
                "draw_indices": self._draw_indices.detach().cpu(),
            }
        return {
            "schema_version": 1,
            "preset": self.population.preset,
            "construction_seeds": self.population.construction_seeds.detach().cpu(),
            "persistent": self.persistent.clone(),
            "apparent": self.apparent.clone(),
            "generator_states": [
                generator.get_state().clone() for generator in self._generators
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore an exact state produced for the same sampled population."""

        schema_version = state.get("schema_version")
        expected_v1 = {
            "schema_version", "preset", "construction_seeds", "persistent",
            "apparent", "generator_states",
        }
        expected_v2 = {
            "schema_version", "preset", "construction_seeds", "persistent",
            "apparent", "rng_backend", "seeds", "maximum_random_draws",
            "draw_indices",
        }
        expected = expected_v1 if schema_version == 1 else expected_v2
        if set(state) != expected or schema_version not in {1, 2}:
            raise ValueError(
                "Expected an IBM ReRAM plant state with schema version 1 or 2 "
                f"and its exact keys. Provided value: schema={schema_version!r}, "
                f"keys={sorted(state)!r}."
            )
        if (schema_version == 1) != (self.device.type == "cpu"):
            raise ValueError(
                "Expected saved IBM ReRAM RNG state schema to match the plant "
                f"execution device. Provided value: schema={schema_version}, "
                f"device={self.device.type!r}."
            )
        if state["preset"] != self.population.preset:
            raise ValueError(
                "Expected plant state preset to match the sampled population. "
                f"Provided value: saved={state['preset']!r}, current={self.population.preset!r}."
            )
        saved_seeds = state["construction_seeds"]
        persistent = state["persistent"]
        apparent = state["apparent"]
        if not isinstance(saved_seeds, torch.Tensor) or not torch.equal(
            saved_seeds.detach().cpu(),
            self.population.construction_seeds.detach().cpu(),
        ):
            raise ValueError(
                "Expected plant state construction seeds to identify the same devices."
            )
        if (
            not isinstance(persistent, torch.Tensor)
            or not isinstance(apparent, torch.Tensor)
            or persistent.shape != self.persistent.shape
            or apparent.shape != self.apparent.shape
            or persistent.dtype != torch.float32
            or apparent.dtype != torch.float32
        ):
            raise ValueError(
                "Expected saved persistent and apparent tensors to match the plant shape and float32 dtype."
            )
        if schema_version == 1:
            generator_states = state["generator_states"]
            if not isinstance(generator_states, (list, tuple)) or len(generator_states) != self.size:
                raise ValueError(
                    "Expected one saved generator state per device trajectory."
                )
            staged_generators = []
            for generator_state in generator_states:
                if not isinstance(generator_state, torch.Tensor):
                    raise ValueError("Expected every saved generator state to be a tensor.")
                generator = torch.Generator(device="cpu")
                generator.set_state(generator_state.detach().cpu())
                staged_generators.append(generator)
            self._generators = staged_generators
        else:
            if state["rng_backend"] != "per_trajectory_buffered_torch_cpu":
                raise ValueError("Expected the saved CUDA RNG backend to match exactly.")
            if state["maximum_random_draws"] != self._maximum_random_draws:
                raise ValueError(
                    "Expected the saved CUDA random-draw budget to match the plant."
                )
            seeds = state["seeds"]
            draw_indices = state["draw_indices"]
            if not isinstance(seeds, (list, tuple)) or not isinstance(
                draw_indices, torch.Tensor
            ):
                raise ValueError("Expected saved CUDA seeds and draw indices.")
            self._configure_rng(seeds, normal_draws=None)
            assert self._draw_indices is not None
            if draw_indices.shape != self._draw_indices.shape:
                raise ValueError("Expected saved CUDA draw indices to match the plant.")
            self._draw_indices.copy_(draw_indices.to(self.device))
        self.persistent.copy_(persistent.to(self.device))
        self.apparent.copy_(apparent.to(self.device))

    def _normal(self, active: torch.Tensor) -> torch.Tensor:
        values = torch.zeros(
            self.size, dtype=torch.float32, device=self.device
        )
        if self.device.type == "cuda":
            assert self._normal_draws is not None
            assert self._draw_indices is not None
            indices = torch.nonzero(active, as_tuple=False).reshape(-1)
            counters = self._draw_indices[indices]
            if bool(torch.any(counters >= self._normal_draws.shape[1])):
                raise RuntimeError(
                    "Expected the declared CUDA random-draw budget to cover "
                    "every requested stochastic pulse."
                )
            values[indices] = self._normal_draws[indices, counters]
            self._draw_indices[indices] += 1
            return values
        for index in torch.nonzero(active, as_tuple=False).reshape(-1).tolist():
            values[index] = torch.randn((), generator=self._generators[index])
        return values

    def pulse(self, directions: torch.Tensor) -> None:
        """Apply one fixed-amplitude pulse to every non-zero direction."""

        direction = torch.as_tensor(
            directions, dtype=torch.int8, device=self.device
        )
        if direction.shape != self.persistent.shape or bool(
            torch.any((direction < -1) | (direction > 1))
        ):
            raise ValueError(
                "Expected directions to be a vector containing only -1, 0, "
                f"or 1. Provided value: shape={tuple(direction.shape)!r}."
            )
        active = direction != 0
        if not bool(torch.any(active)):
            return

        population = self.population
        physical = self.persistent + population.reference
        cycle = self._normal(active)
        up = direction > 0
        down = direction < 0
        candidate = physical.clone()
        if bool(torch.any(up)):
            normalized = torch.where(
                population.max_bound > 0.0,
                physical / population.max_bound,
                torch.zeros_like(physical),
            )
            response = population.dwmin_up * (
                1.0
                - normalized
                + population.dw_min_std * cycle
            )
            candidate[up] = physical[up] + response[up]
        if bool(torch.any(down)):
            normalized = torch.where(
                population.min_bound < 0.0,
                physical / population.min_bound,
                torch.zeros_like(physical),
            )
            response = population.dwmin_down * (
                1.0
                - normalized
                + population.dw_min_std * cycle
            )
            candidate[down] = physical[down] - response[down]
        candidate = torch.maximum(candidate, population.min_bound)
        candidate = torch.minimum(candidate, population.max_bound)
        self.persistent[active] = (
            candidate - population.reference
        )[active]

        write_scale = population.write_noise_std * population.nominal_dw_min
        if write_scale > 0.0:
            write = self._normal(active)
            self.apparent[active] = (
                self.persistent + float(write_scale) * write
            )[active]
        else:
            self.apparent[active] = self.persistent[active]

    def condition_boundary(
        self,
        *,
        start_protocol: str,
        quiet_steps: int,
        change_threshold: float,
        maximum_pulses: int,
    ) -> "ConditioningResult":
        if start_protocol not in {"lower_to_target", "upper_to_target"}:
            raise ValueError(
                "Expected start_protocol to be 'lower_to_target' or "
                f"'upper_to_target'. Provided value: {start_protocol!r}."
            )
        direction = -1 if start_protocol == "lower_to_target" else 1
        active = torch.ones(
            self.size, dtype=torch.bool, device=self.device
        )
        consecutive = torch.zeros(
            self.size, dtype=torch.int64, device=self.device
        )
        counts = torch.zeros_like(consecutive)
        for _ in range(maximum_pulses):
            before = self.persistent.clone()
            self.pulse(active.to(torch.int8) * direction)
            change = torch.abs(self.persistent - before)
            consecutive = torch.where(
                active & (change < change_threshold),
                consecutive + 1,
                torch.where(active, torch.zeros_like(consecutive), consecutive),
            )
            counts[active] += 1
            active = consecutive < quiet_steps
            if not bool(torch.any(active)):
                break
        return ConditioningResult(
            success=~active,
            pulse_count=counts,
            persistent=self.persistent.clone(),
            apparent=self.apparent.clone(),
        )

    def controller_port(self) -> VerifyPort:
        return _IbmReramControllerPort(self)

class _IbmReramControllerPort:
    """Capability-limited view: no hidden state or device parameters."""

    def __init__(self, plant: IbmReramPlant) -> None:
        self.__plant = plant

    @property
    def size(self) -> int:
        return self.__plant.size

    def verify(self) -> torch.Tensor:
        return (self.__plant.apparent.clone() + 1.0) / 2.0

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
        if direction.shape != (self.size,) or count.shape != (self.size,):
            raise ValueError(
                "Expected directions and counts to match the controller port "
                f"size {self.size}. Provided value: direction={tuple(direction.shape)!r}, "
                f"count={tuple(count.shape)!r}."
            )
        if bool(torch.any(count < 0)):
            raise ValueError(
                "Expected pulse counts to be non-negative. "
                f"Provided value: minimum={int(count.min().item())}."
            )
        maximum = int(count.max().item()) if count.numel() else 0
        for pulse_index in range(maximum):
            active_direction = torch.where(
                count > pulse_index,
                direction,
                torch.zeros_like(direction),
            )
            self.__plant.pulse(active_direction)

class IbmReramRawActivePlant(IbmReramPlant):
    """Pulse plant whose persistent state is the single active state ``a``.

    ``IbmReramPlant`` reproduces AIHWKit's tile-facing reference-relative
    coordinate ``w=a-r``.  This variant evaluates the same sampled pulse
    response directly in ``a`` and deliberately never reads ``reference``.
    The controller sees one caller-declared array-wide affine coordinate
    ``g=(a-coordinate_min)/coordinate_scale``.
    """

    def __init__(
        self,
        population: IbmReramPopulation,
        *,
        seeds: Sequence[int],
        coordinate_min: float,
        coordinate_scale: float,
        device: torch.device | str = "cpu",
        maximum_random_draws: int | None = None,
        normal_draws: torch.Tensor | None = None,
    ) -> None:
        if (
            not math.isfinite(coordinate_min)
            or not math.isfinite(coordinate_scale)
            or coordinate_scale <= 0.0
        ):
            raise ValueError(
                "Expected a finite raw-active coordinate minimum and a "
                "positive finite scale. Provided value: "
                f"minimum={coordinate_min!r}, scale={coordinate_scale!r}."
            )
        self.coordinate_min = float(coordinate_min)
        self.coordinate_scale = float(coordinate_scale)
        super().__init__(
            population,
            seeds=seeds,
            device=device,
            maximum_random_draws=maximum_random_draws,
            normal_draws=normal_draws,
        )

    @property
    def persistent_a(self) -> torch.Tensor:
        return self.persistent

    @property
    def apparent_a(self) -> torch.Tensor:
        return self.apparent

    def pulse(self, directions: torch.Tensor) -> None:
        """Apply one raw-active pulse without subtracting a reference state."""

        direction = torch.as_tensor(
            directions, dtype=torch.int8, device=self.device
        )
        if direction.shape != self.persistent.shape or bool(
            torch.any((direction < -1) | (direction > 1))
        ):
            raise ValueError(
                "Expected directions to be a vector containing only -1, 0, "
                f"or 1. Provided value: shape={tuple(direction.shape)!r}."
            )
        active = direction != 0
        if not bool(torch.any(active)):
            return

        population = self.population
        current = self.persistent
        cycle = self._normal(active)
        up = direction > 0
        down = direction < 0
        candidate = current.clone()
        if bool(torch.any(up)):
            normalized = torch.where(
                population.max_bound > 0.0,
                current / population.max_bound,
                torch.zeros_like(current),
            )
            response = population.dwmin_up * (
                1.0
                - normalized
                + population.dw_min_std * cycle
            )
            candidate[up] = current[up] + response[up]
        if bool(torch.any(down)):
            normalized = torch.where(
                population.min_bound < 0.0,
                current / population.min_bound,
                torch.zeros_like(current),
            )
            response = population.dwmin_down * (
                1.0
                - normalized
                + population.dw_min_std * cycle
            )
            candidate[down] = current[down] - response[down]
        candidate = torch.maximum(candidate, population.min_bound)
        candidate = torch.minimum(candidate, population.max_bound)
        self.persistent[active] = candidate[active]

        write_scale = population.write_noise_std * population.nominal_dw_min
        if write_scale > 0.0:
            write = self._normal(active)
            self.apparent[active] = (
                self.persistent + float(write_scale) * write
            )[active]
        else:
            self.apparent[active] = self.persistent[active]

    def controller_port(self) -> VerifyPort:
        return _IbmReramRawActiveControllerPort(self)

class _IbmReramRawActiveControllerPort:
    """Capability-limited apparent-``g`` view of a raw-active plant."""

    def __init__(self, plant: IbmReramRawActivePlant) -> None:
        self.__plant = plant

    @property
    def size(self) -> int:
        return self.__plant.size

    def verify(self) -> torch.Tensor:
        return (
            self.__plant.apparent.clone() - self.__plant.coordinate_min
        ) / self.__plant.coordinate_scale

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
        if direction.shape != (self.size,) or count.shape != (self.size,):
            raise ValueError(
                "Expected directions and counts to match the raw-active port "
                f"size {self.size}. Provided value: direction="
                f"{tuple(direction.shape)!r}, count={tuple(count.shape)!r}."
            )
        if bool(torch.any(count < 0)):
            raise ValueError(
                "Expected pulse counts to be non-negative. Provided value: "
                f"minimum={int(count.min().item())}."
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

@dataclass(frozen=True)
class ConditioningResult:
    success: torch.Tensor
    pulse_count: torch.Tensor
    persistent: torch.Tensor
    apparent: torch.Tensor

def partition_device_identities(
    num_devices: int,
    *,
    seed: int,
    fractions: Sequence[float] = (0.2, 0.6, 0.2),
) -> tuple[str, ...]:
    """Assign whole identities to calibration, fit, and validation splits."""

    if num_devices < 3:
        raise ValueError(
            "Expected at least three device identities for non-empty "
            f"partitions. Provided value: {num_devices!r}."
        )
    if len(fractions) != 3 or any(value <= 0.0 for value in fractions) or not math.isclose(
        sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "Expected three positive partition fractions summing to one. "
            f"Provided value: {tuple(fractions)!r}."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(num_devices, generator=generator).tolist()
    calibration_count = max(1, int(math.floor(num_devices * fractions[0])))
    validation_count = max(1, int(math.floor(num_devices * fractions[2])))
    if calibration_count + validation_count >= num_devices:
        calibration_count = 1
        validation_count = 1
    fit_count = num_devices - calibration_count - validation_count
    labels = ["fit"] * num_devices
    for index in order[:calibration_count]:
        labels[index] = "calibration"
    for index in order[calibration_count + fit_count :]:
        labels[index] = "validation"
    return tuple(labels)
