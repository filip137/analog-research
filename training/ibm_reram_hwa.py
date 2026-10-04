"""IBM OM crossbar population sampling, fingerprints and pickle-free storage.

The module path and artifact schema are retained for sampler receipts and
historical population files.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any
import numpy as np
import torch
from experiments.artifacts import sha256_file
from training.parameters import ParameterBinding
from training.parameters import DenseWeight
from training.ibm_reram_program_verify import (
    OM_PRESET,
    PUBLISHED_CORRUPT_PROBABILITY,
    PUBLISHED_CORRUPT_RANGE,
    derive_seed,
)


_CORRUPTION_POLICIES = ("counterfactual_repaired", "published")


_ROOT = Path(__file__).resolve().parents[1]


_ARRAY_POPULATION_SCHEMA = "ebl.ibm_reram.om_array_population"


_ARRAY_POPULATION_SCHEMA_VERSION = 1


_ARRAY_SAMPLING_RECEIPT_SCHEMA = "ebl.ibm_reram.om_array_population_receipt"


_ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION = 1


_REQUIRED_AIHWKIT_VERSION = "1.1.0"


@dataclass(frozen=True)
class IbmReramArrayPopulation:
    """One fixed, flattened physical-cell assignment across named crossbar tiles."""

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


def _native_seed(seed: int) -> int:
    return int(seed % (2**31 - 2) + 1)


def _sample_tile_hidden(
    shape: tuple[int, ...],
    *,
    construction_seed: int,
    published_corruption: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
    if len(shape) != 2:
        raise ValueError(
            "Expected IBM OM HWA only for two-dimensional DenseWeight tensors. "
            f"Provided value: {shape!r}."
        )
    try:
        import aihwkit
        from aihwkit.simulator.configs import SingleRPUConfig
        from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice
        from aihwkit.simulator.tiles import AnalogTile
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Expected AIHWKit 1.1.0 to sample the published IBM OM cell "
            f"assignment. Provided value: {error}."
        ) from error
    version = str(getattr(aihwkit, "__version__", "unknown"))
    if version != "1.1.0":
        raise RuntimeError(
            "Expected AIHWKit version '1.1.0' for IBM OM identity sampling. "
            f"Provided value: {version!r}."
        )
    device = ReRamArrayOMPresetDevice()
    preset_default_corrupt_probability = float(device.corrupt_devices_prob)
    preset_corrupt_range = float(device.corrupt_devices_range)
    if not math.isclose(
        preset_default_corrupt_probability,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        preset_corrupt_range,
        PUBLISHED_CORRUPT_RANGE[OM_PRESET],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Expected the pinned AIHWKit OM preset default corruption probability "
            "and corrupt-device range."
        )
    device.corrupt_devices_prob = (
        PUBLISHED_CORRUPT_PROBABILITY[OM_PRESET] if published_corruption else 0.0
    )
    device.construction_seed = _native_seed(construction_seed)
    tile = AnalogTile(
        shape[1],
        shape[0],
        SingleRPUConfig(device=device),
        bias=False,
    )
    raw = tile.get_hidden_parameters()
    names = ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference")
    missing = tuple(name for name in names if name not in raw)
    if missing:
        raise RuntimeError(
            "Expected AIHWKit OM hidden device parameters. "
            f"Provided value: missing={missing!r}."
        )
    hidden = {}
    for name in names:
        tensor = raw[name].detach().to(device="cpu", dtype=torch.float32)
        if tensor.shape != (shape[1], shape[0]):
            raise RuntimeError(
                "Expected AIHWKit hidden arrays in (post, pre) layout. "
                f"Provided value: name={name!r}, shape={tuple(tensor.shape)!r}."
            )
        hidden[name] = tensor.transpose(0, 1).contiguous().reshape(-1)
    metadata: dict[str, float | str] = {
        "aihwkit_version": version,
        "nominal_dw_min": float(device.dw_min),
        "dw_min_std": float(device.dw_min_std),
        "write_noise_std": float(device.write_noise_std),
    }
    return hidden, metadata


def om_array_population_fingerprint(
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


def _sample_om_array_population_layout(
    binding_keys: Sequence[str],
    binding_shapes: Sequence[tuple[int, ...]],
    *,
    assignment_seed: int,
    corruption_policy: str,
) -> IbmReramArrayPopulation:
    """Sample a literal published assignment for one canonical binding layout."""

    if corruption_policy not in _CORRUPTION_POLICIES:
        raise ValueError(
            f"Expected corruption_policy in {_CORRUPTION_POLICIES!r}."
        )
    keys = tuple(binding_keys)
    shapes = tuple(tuple(int(value) for value in shape) for shape in binding_shapes)
    if (
        not keys
        or len(keys) != len(shapes)
        or len(set(keys)) != len(keys)
        or any(not key for key in keys)
        or any(len(shape) != 2 or any(value < 1 for value in shape) for shape in shapes)
    ):
        raise ValueError(
            "Expected unique named two-dimensional IBM OM binding keys and shapes."
        )
    base_seeds = tuple(
        _native_seed(derive_seed(assignment_seed, OM_PRESET, key, "published"))
        for key in keys
    )
    donor_seeds = tuple(
        _native_seed(derive_seed(assignment_seed, OM_PRESET, key, "repair_donor"))
        for key in keys
    )
    names = ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference")
    pieces: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    corrupt_pieces = []
    published_pieces = []
    reference_metadata: dict[str, float | str] | None = None
    for key, shape, base_seed, donor_seed in zip(keys, shapes, base_seeds, donor_seeds):
        base, metadata = _sample_tile_hidden(
            shape,
            construction_seed=base_seed,
            published_corruption=True,
        )
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise RuntimeError("Expected identical OM preset scalar parameters per binding.")
        published_corrupt = (
            (torch.abs(base["max_bound"] - base["min_bound"]) <= 1e-12)
            & (base["dwmin_up"] == 0.0)
            & (base["dwmin_down"] == 0.0)
        )
        final = {name: value.clone() for name, value in base.items()}
        if corruption_policy == "counterfactual_repaired" and bool(
            torch.any(published_corrupt)
        ):
            donor, donor_metadata = _sample_tile_hidden(
                shape,
                construction_seed=donor_seed,
                published_corruption=False,
            )
            if donor_metadata != metadata:
                raise RuntimeError("Expected repair donors from the same OM preset.")
            for name in names:
                final[name][published_corrupt] = donor[name][published_corrupt]
        final_corrupt = (
            (torch.abs(final["max_bound"] - final["min_bound"]) <= 1e-12)
            & (final["dwmin_up"] == 0.0)
            & (final["dwmin_down"] == 0.0)
        )
        for name in names:
            pieces[name].append(final[name])
        corrupt_pieces.append(final_corrupt)
        published_pieces.append(published_corrupt)
    assert reference_metadata is not None
    joined = {name: torch.cat(values) for name, values in pieces.items()}
    joined["corrupt"] = torch.cat(corrupt_pieces)
    joined["published_corrupt"] = torch.cat(published_pieces)
    fingerprint = om_array_population_fingerprint(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        keys=keys,
        shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        scalar_parameters=reference_metadata,
        tensors=joined,
    )
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        nominal_dw_min=float(reference_metadata["nominal_dw_min"]),
        dw_min_std=float(reference_metadata["dw_min_std"]),
        write_noise_std=float(reference_metadata["write_noise_std"]),
        max_bound=joined["max_bound"],
        min_bound=joined["min_bound"],
        dwmin_up=joined["dwmin_up"],
        dwmin_down=joined["dwmin_down"],
        reference=joined["reference"],
        corrupt=joined["corrupt"],
        published_corrupt=joined["published_corrupt"],
        fingerprint=fingerprint,
        aihwkit_version=str(reference_metadata["aihwkit_version"]),
    )


def _binding_layout(
    bindings: Sequence[ParameterBinding],
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    selected = tuple(bindings)
    if not selected or any(
        not isinstance(binding.parameter, DenseWeight) for binding in selected
    ):
        raise ValueError(
            "Expected at least one named DenseWeight binding for IBM OM HWA."
        )
    keys = tuple(binding.key for binding in selected)
    shapes = tuple(tuple(binding.state.shape) for binding in selected)
    if len(set(keys)) != len(keys):
        raise ValueError("Expected unique IBM OM DenseWeight binding keys.")
    return keys, shapes


def sample_om_array_population(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
    corruption_policy: str,
) -> IbmReramArrayPopulation:
    """Sample a literal published assignment, optionally repairing defects only."""

    keys, shapes = _binding_layout(bindings)
    return _sample_om_array_population_layout(
        keys,
        shapes,
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
    )


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
    fingerprint = om_array_population_fingerprint(
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


def sample_om_array_population_external(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
    corruption_policy: str,
    aihwkit_python: Path,
    population_path: Path,
    receipt_path: Path,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    """Sample OM cells in pinned AIHWKit while training stays in CUDA PyTorch."""

    keys, shapes = _binding_layout(bindings)
    sampler = aihwkit_python.expanduser().resolve()
    if not sampler.is_file() or not os.access(sampler, os.X_OK):
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to identify an executable pinned "
            f"AIHWKit interpreter. Provided value: {str(sampler)!r}."
        )
    destination = population_path.expanduser().resolve()
    receipt_destination = receipt_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_destination.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "preset": OM_PRESET,
        "assignment_seed": assignment_seed,
        "corruption_policy": corruption_policy,
        "preset_default_corrupt_devices_prob": 0.0,
        "published_corrupt_devices_prob": PUBLISHED_CORRUPT_PROBABILITY[
            OM_PRESET
        ],
        "corrupt_devices_range": PUBLISHED_CORRUPT_RANGE[OM_PRESET],
        "binding_keys": list(keys),
        "binding_shapes": [list(shape) for shape in shapes],
        "required_aihwkit_version": _REQUIRED_AIHWKIT_VERSION,
    }
    command = [
        str(sampler),
        "-m",
        "experiments.reram_program_verify.hwa_population_sampler",
        "--request-json",
        json.dumps(request, allow_nan=False, sort_keys=True, separators=(",", ":")),
        "--output",
        str(destination),
        "--receipt",
        str(receipt_destination),
    ]
    environment = os.environ.copy()
    environment.pop("EBL_AIHWKIT_PYTHON", None)
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            "Expected the pinned external OM HWA population sampler to "
            f"complete successfully. Exit={completed.returncode}; tail={detail!r}."
        )
    try:
        receipt = json.loads(receipt_destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Expected a readable OM array sampling receipt.") from error
    expected_receipt_keys = {
        "schema",
        "schema_version",
        "backend",
        "python_executable",
        "python_version",
        "torch_version",
        "aihwkit_version",
        "request",
        "num_cells",
        "population_fingerprint",
        "population_sha256",
        "sampler_source_sha256",
        "population_implementation_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_receipt_keys
        or receipt.get("schema") != _ARRAY_SAMPLING_RECEIPT_SCHEMA
        or receipt.get("schema_version") != _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION
        or receipt.get("backend") != "external_pinned_aihwkit_python"
        or receipt.get("aihwkit_version") != _REQUIRED_AIHWKIT_VERSION
        or receipt.get("request") != request
        or receipt.get("population_sha256") != sha256_file(destination)
        or receipt.get("sampler_source_sha256")
        != sha256_file(_ROOT / "experiments/reram_program_verify/hwa_population_sampler.py")
        or receipt.get("population_implementation_sha256")
        != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("Expected the OM array sampling receipt to match its request and sources.")
    population = load_om_array_population(destination)
    if (
        population.assignment_seed != assignment_seed
        or population.corruption_policy != corruption_policy
        or population.binding_keys != keys
        or population.binding_shapes != shapes
        or population.fingerprint != receipt.get("population_fingerprint")
        or population.size != receipt.get("num_cells")
    ):
        raise RuntimeError("Expected the sampled OM population to match its declared layout.")
    return population, receipt
