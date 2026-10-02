"""Fixed IBM OM physical-cell assignments and their pickle-free persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.ibm_om.constants import (
    _ARRAY_POPULATION_SCHEMA,
    _ARRAY_POPULATION_SCHEMA_VERSION,
    _REQUIRED_AIHWKIT_VERSION,
)
from training.ibm_reram_program_verify import OM_PRESET, derive_seed


@dataclass(frozen=True)
class IbmReramArrayPopulation:
    """One fixed, flattened physical-cell assignment across named DRN tensors."""

    assignment_seed: int
    corruption_policy: str
    binding_keys: tuple[str, ...]
    binding_shapes: tuple[tuple[int, ...], ...]
    binding_sampling_seeds: tuple[int, ...]
    donor_sampling_seeds: tuple[int, ...]
    nominal_dw_min: float
    dw_min_std: float
    write_noise_std: float
    max_bound: torch.Tensor
    min_bound: torch.Tensor
    dwmin_up: torch.Tensor
    dwmin_down: torch.Tensor
    reference: torch.Tensor
    corrupt: torch.Tensor
    published_corrupt: torch.Tensor
    fingerprint: str
    aihwkit_version: str

    def __post_init__(self) -> None:
        size = int(self.max_bound.numel())
        if size < 1 or sum(math.prod(shape) for shape in self.binding_shapes) != size:
            raise ValueError("Expected population tensors to match binding shapes.")
        if len(self.binding_keys) != len(self.binding_shapes) or len(set(self.binding_keys)) != len(
            self.binding_keys
        ):
            raise ValueError("Expected unique population binding keys and matching shapes.")
        for name in (
            "max_bound",
            "min_bound",
            "dwmin_up",
            "dwmin_down",
            "reference",
        ):
            value = getattr(self, name)
            if value.shape != (size,) or value.dtype != torch.float32 or not bool(
                torch.all(torch.isfinite(value))
            ):
                raise ValueError(f"Expected {name} to be a finite float32 vector.")
        for name in ("corrupt", "published_corrupt"):
            value = getattr(self, name)
            if value.shape != (size,) or value.dtype != torch.bool:
                raise ValueError(f"Expected {name} to be a boolean vector.")
        if bool(torch.any(self.max_bound < self.min_bound)):
            raise ValueError("Expected sampled maximum bounds not below minimum bounds.")
        detected = (
            (torch.abs(self.max_bound - self.min_bound) <= 1e-12)
            & (self.dwmin_up == 0.0)
            & (self.dwmin_down == 0.0)
        )
        if not torch.equal(detected, self.corrupt):
            raise ValueError("Expected final corrupt flags to match collapsed zero-step cells.")
        if self.corruption_policy == "published" and not torch.equal(
            self.corrupt, self.published_corrupt
        ):
            raise ValueError("Expected published policy to retain the sampled corrupt mask.")
        if self.corruption_policy == "counterfactual_repaired" and bool(
            torch.any(self.corrupt)
        ):
            raise ValueError("Expected repaired policy to replace every corrupt cell.")

    @property
    def size(self) -> int:
        return int(self.max_bound.numel())

    @property
    def logical_min(self) -> torch.Tensor:
        return self.min_bound - self.reference

    @property
    def logical_max(self) -> torch.Tensor:
        return self.max_bound - self.reference

    def to(self, device: torch.device | str) -> "IbmReramArrayPopulation":
        target = torch.device(device)
        return IbmReramArrayPopulation(
            assignment_seed=self.assignment_seed,
            corruption_policy=self.corruption_policy,
            binding_keys=self.binding_keys,
            binding_shapes=self.binding_shapes,
            binding_sampling_seeds=self.binding_sampling_seeds,
            donor_sampling_seeds=self.donor_sampling_seeds,
            nominal_dw_min=self.nominal_dw_min,
            dw_min_std=self.dw_min_std,
            write_noise_std=self.write_noise_std,
            max_bound=self.max_bound.to(target),
            min_bound=self.min_bound.to(target),
            dwmin_up=self.dwmin_up.to(target),
            dwmin_down=self.dwmin_down.to(target),
            reference=self.reference.to(target),
            corrupt=self.corrupt.to(target),
            published_corrupt=self.published_corrupt.to(target),
            fingerprint=self.fingerprint,
            aihwkit_version=self.aihwkit_version,
        )

    def tensor_state(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).detach().cpu().clone()
            for name in (
                "max_bound",
                "min_bound",
                "dwmin_up",
                "dwmin_down",
                "reference",
                "corrupt",
                "published_corrupt",
            )
        }



def _selected_array_population(
    population: IbmReramArrayPopulation,
    selection: torch.Tensor,
) -> IbmReramArrayPopulation:
    """Build an internal physical-identity slice for exact fallback writes."""

    mask = torch.as_tensor(
        selection,
        device=population.max_bound.device,
        dtype=torch.bool,
    )
    if mask.shape != (population.size,) or not bool(torch.any(mask)):
        raise ValueError(
            "Expected a non-empty boolean selection matching the IBM OM "
            "population."
        )
    indices = torch.where(mask)[0].detach().cpu().to(dtype=torch.int64)
    digest = sha256()
    digest.update(population.fingerprint.encode())
    digest.update(indices.numpy().tobytes())
    count = int(indices.numel())
    return IbmReramArrayPopulation(
        assignment_seed=population.assignment_seed,
        corruption_policy=population.corruption_policy,
        binding_keys=("__pulse_resolved_fallback__",),
        binding_shapes=((count, 1),),
        binding_sampling_seeds=(population.assignment_seed,),
        donor_sampling_seeds=(population.assignment_seed,),
        nominal_dw_min=population.nominal_dw_min,
        dw_min_std=population.dw_min_std,
        write_noise_std=population.write_noise_std,
        max_bound=population.max_bound[mask].clone(),
        min_bound=population.min_bound[mask].clone(),
        dwmin_up=population.dwmin_up[mask].clone(),
        dwmin_down=population.dwmin_down[mask].clone(),
        reference=population.reference[mask].clone(),
        corrupt=population.corrupt[mask].clone(),
        published_corrupt=population.published_corrupt[mask].clone(),
        fingerprint=digest.hexdigest(),
        aihwkit_version=population.aihwkit_version,
    )



def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    flattened = value.detach().reshape(-1)
    if flattened.numel() == 0:
        return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0}
    return {
        "minimum": float(flattened.min().item()),
        "mean": float(flattened.mean().item()),
        "maximum": float(flattened.max().item()),
    }



def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()



def _native_seed(seed: int) -> int:
    return int(seed % (2**31 - 2) + 1)



def _population_fingerprint(
    *,
    assignment_seed: int,
    corruption_policy: str,
    keys: Sequence[str],
    shapes: Sequence[tuple[int, ...]],
    binding_sampling_seeds: Sequence[int],
    donor_sampling_seeds: Sequence[int],
    scalar_parameters: Mapping[str, float | str],
    tensors: Mapping[str, torch.Tensor],
) -> str:
    digest = sha256()
    digest.update(str(assignment_seed).encode())
    digest.update(corruption_policy.encode())
    for key, shape, binding_seed, donor_seed in zip(
        keys,
        shapes,
        binding_sampling_seeds,
        donor_sampling_seeds,
    ):
        digest.update(key.encode())
        digest.update(repr(tuple(shape)).encode())
        digest.update(str(int(binding_seed)).encode())
        digest.update(str(int(donor_seed)).encode())
    for name in sorted(scalar_parameters):
        value = scalar_parameters[name]
        digest.update(name.encode())
        digest.update(
            (float(value).hex() if isinstance(value, float) else str(value)).encode()
        )
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()



_ARRAY_POPULATION_FIELDS = {
    "schema",
    "schema_version",
    "assignment_seed",
    "corruption_policy",
    "binding_keys_json",
    "binding_shapes_json",
    "binding_sampling_seeds",
    "donor_sampling_seeds",
    "nominal_dw_min",
    "dw_min_std",
    "write_noise_std",
    "max_bound",
    "min_bound",
    "dwmin_up",
    "dwmin_down",
    "reference",
    "corrupt",
    "published_corrupt",
    "fingerprint",
    "aihwkit_version",
}



def save_om_array_population(
    path: Path,
    population: IbmReramArrayPopulation,
) -> None:
    """Write a cross-PyTorch-version, pickle-free OM array population."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(_ARRAY_POPULATION_SCHEMA),
            schema_version=np.asarray(
                _ARRAY_POPULATION_SCHEMA_VERSION, dtype=np.int64
            ),
            assignment_seed=np.asarray(population.assignment_seed, dtype=np.int64),
            corruption_policy=np.asarray(population.corruption_policy),
            binding_keys_json=np.asarray(
                json.dumps(
                    list(population.binding_keys),
                    allow_nan=False,
                    separators=(",", ":"),
                )
            ),
            binding_shapes_json=np.asarray(
                json.dumps(
                    [list(shape) for shape in population.binding_shapes],
                    allow_nan=False,
                    separators=(",", ":"),
                )
            ),
            binding_sampling_seeds=np.asarray(
                population.binding_sampling_seeds, dtype=np.int64
            ),
            donor_sampling_seeds=np.asarray(
                population.donor_sampling_seeds, dtype=np.int64
            ),
            nominal_dw_min=np.asarray(population.nominal_dw_min, dtype=np.float64),
            dw_min_std=np.asarray(population.dw_min_std, dtype=np.float64),
            write_noise_std=np.asarray(
                population.write_noise_std, dtype=np.float64
            ),
            max_bound=population.max_bound.detach().cpu().numpy(),
            min_bound=population.min_bound.detach().cpu().numpy(),
            dwmin_up=population.dwmin_up.detach().cpu().numpy(),
            dwmin_down=population.dwmin_down.detach().cpu().numpy(),
            reference=population.reference.detach().cpu().numpy(),
            corrupt=population.corrupt.detach().cpu().numpy(),
            published_corrupt=population.published_corrupt.detach().cpu().numpy(),
            fingerprint=np.asarray(population.fingerprint),
            aihwkit_version=np.asarray(population.aihwkit_version),
        )



def _npz_scalar(payload: Mapping[str, np.ndarray], name: str) -> Any:
    value = payload[name]
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"Expected scalar non-object OM population field {name!r}.")
    return value.item()



def load_om_array_population(path: Path) -> IbmReramArrayPopulation:
    """Load and independently revalidate an OM array population artifact."""

    source = path.expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as raw:
            if set(raw.files) != _ARRAY_POPULATION_FIELDS:
                raise ValueError(
                    "Expected exact OM array population fields. "
                    f"Provided value: {sorted(raw.files)!r}."
                )
            payload = {name: raw[name].copy() for name in raw.files}
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Expected a readable pickle-free OM array population at {str(source)!r}."
        ) from error
    if (
        str(_npz_scalar(payload, "schema")) != _ARRAY_POPULATION_SCHEMA
        or int(_npz_scalar(payload, "schema_version"))
        != _ARRAY_POPULATION_SCHEMA_VERSION
    ):
        raise ValueError("Expected OM array population schema version 1.")
    if payload["schema_version"].dtype != np.dtype(np.int64):
        raise ValueError("Expected int64 OM array population schema version.")
    if payload["assignment_seed"].dtype != np.dtype(np.int64):
        raise ValueError("Expected int64 OM array assignment seed.")
    try:
        keys_value = json.loads(str(_npz_scalar(payload, "binding_keys_json")))
        shapes_value = json.loads(str(_npz_scalar(payload, "binding_shapes_json")))
    except json.JSONDecodeError as error:
        raise ValueError("Expected canonical JSON OM array binding layout.") from error
    if (
        not isinstance(keys_value, list)
        or not all(isinstance(key, str) and key for key in keys_value)
        or not isinstance(shapes_value, list)
        or not all(
            isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape)
            for shape in shapes_value
        )
    ):
        raise ValueError("Expected valid named two-dimensional OM binding layout.")
    keys = tuple(keys_value)
    shapes = tuple(tuple(shape) for shape in shapes_value)
    binding_seeds_array = payload["binding_sampling_seeds"]
    donor_seeds_array = payload["donor_sampling_seeds"]
    if (
        binding_seeds_array.dtype != np.dtype(np.int64)
        or donor_seeds_array.dtype != np.dtype(np.int64)
        or binding_seeds_array.shape != (len(keys),)
        or donor_seeds_array.shape != (len(keys),)
    ):
        raise ValueError("Expected one int64 base and donor seed per OM binding.")
    tensor_names = (
        "max_bound",
        "min_bound",
        "dwmin_up",
        "dwmin_down",
        "reference",
    )
    size = sum(math.prod(shape) for shape in shapes)
    tensors: dict[str, torch.Tensor] = {}
    for name in tensor_names:
        value = payload[name]
        if (
            value.dtype != np.dtype(np.float32)
            or value.shape != (size,)
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"Expected finite float32 OM population vector {name!r}.")
        tensors[name] = torch.from_numpy(value.copy())
    for name in ("corrupt", "published_corrupt"):
        value = payload[name]
        if value.dtype != np.dtype(np.bool_) or value.shape != (size,):
            raise ValueError(f"Expected boolean OM population vector {name!r}.")
        tensors[name] = torch.from_numpy(value.copy())
    scalar_names = ("nominal_dw_min", "dw_min_std", "write_noise_std")
    scalars = {}
    for name in scalar_names:
        if payload[name].dtype != np.dtype(np.float64):
            raise ValueError(f"Expected float64 OM population scalar {name!r}.")
        value = float(_npz_scalar(payload, name))
        if not math.isfinite(value):
            raise ValueError(f"Expected finite OM population scalar {name!r}.")
        scalars[name] = value
    aihwkit_version = str(_npz_scalar(payload, "aihwkit_version"))
    if aihwkit_version != _REQUIRED_AIHWKIT_VERSION:
        raise ValueError(
            "Expected AIHWKit version '1.1.0' in the OM population artifact."
        )
    assignment_seed = int(_npz_scalar(payload, "assignment_seed"))
    corruption_policy = str(_npz_scalar(payload, "corruption_policy"))
    base_seeds = tuple(int(value) for value in binding_seeds_array.tolist())
    donor_seeds = tuple(int(value) for value in donor_seeds_array.tolist())
    expected_base_seeds = tuple(
        _native_seed(derive_seed(assignment_seed, OM_PRESET, key, "published"))
        for key in keys
    )
    expected_donor_seeds = tuple(
        _native_seed(derive_seed(assignment_seed, OM_PRESET, key, "repair_donor"))
        for key in keys
    )
    if base_seeds != expected_base_seeds or donor_seeds != expected_donor_seeds:
        raise ValueError("Expected binding seeds derived from the OM assignment seed.")
    fingerprint = _population_fingerprint(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        keys=keys,
        shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        scalar_parameters={"aihwkit_version": aihwkit_version, **scalars},
        tensors=tensors,
    )
    if str(_npz_scalar(payload, "fingerprint")) != fingerprint:
        raise ValueError("Expected recomputed OM population fingerprint to match.")
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        nominal_dw_min=scalars["nominal_dw_min"],
        dw_min_std=scalars["dw_min_std"],
        write_noise_std=scalars["write_noise_std"],
        max_bound=tensors["max_bound"],
        min_bound=tensors["min_bound"],
        dwmin_up=tensors["dwmin_up"],
        dwmin_down=tensors["dwmin_down"],
        reference=tensors["reference"],
        corrupt=tensors["corrupt"],
        published_corrupt=tensors["published_corrupt"],
        fingerprint=fingerprint,
        aihwkit_version=aihwkit_version,
    )
