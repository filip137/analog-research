"""IBM OM device-model loading and AIHWKit identity sampling.

This is the only IBM OM module that imports AIHWKit.  Sampling runs either
in-process or in the pinned external AIHWKit interpreter, whose receipt
binds the population artifact to this module's source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import torch

from experiments.artifacts import sha256_file
from experiments.reram_program_verify.hwa_model import (
    CONDITION_KEY,
    SCHEMA as DEVICE_MODEL_SCHEMA,
    SCHEMA_VERSION as DEVICE_MODEL_SCHEMA_VERSION,
)
from model.resistive.builders import ParameterBinding
from model.variable.parameter import DenseWeight
from training.ibm_om.constants import (
    _ARRAY_SAMPLING_RECEIPT_SCHEMA,
    _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION,
    _CORRUPTION_POLICIES,
    _REQUIRED_AIHWKIT_VERSION,
)
from training.ibm_om.population import (
    IbmReramArrayPopulation,
    _native_seed,
    _population_fingerprint,
    load_om_array_population,
)
from training.ibm_reram_program_verify import (
    OM_PRESET,
    PUBLISHED_CORRUPT_PROBABILITY,
    PopulationStepEstimator,
    derive_seed,
)


# Repository root (this module lives in training/ibm_om/).
_ROOT = Path(__file__).resolve().parents[2]



def _artifact(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Expected --device-model to name a readable IBM OM HWA JSON "
            f"artifact. Provided value: {str(resolved)!r}."
        ) from error
    if not isinstance(value, dict) or (
        value.get("schema") != DEVICE_MODEL_SCHEMA
        or value.get("schema_version") != DEVICE_MODEL_SCHEMA_VERSION
    ):
        raise ValueError(
            "Expected --device-model to use the IBM OM HWA schema version 1."
        )
    programming = value.get("programming")
    expected = {
        "controller": "adaptive",
        "start_protocol": "lower_to_target",
        "condition_key": CONDITION_KEY,
        "tolerance_step_ratio": 0.5,
        "maximum_program_pulses": 128,
    }
    if not isinstance(programming, Mapping) or any(
        programming.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError(
            "Expected --device-model to contain the declared OM cap-128 "
            "adaptive lower-from-RESET protocol."
        )
    endpoint_models = value.get("endpoint_models")
    estimators = value.get("step_estimators")
    if (
        not isinstance(endpoint_models, Mapping)
        or set(endpoint_models) != {"continuous", "published_corruption"}
        or not isinstance(estimators, Mapping)
        or set(estimators) != {"continuous", "published_corruption"}
    ):
        raise ValueError(
            "Expected --device-model to contain matched continuous and "
            "published-corruption endpoint models and step estimators."
        )
    for branch, published_corruption in (
        ("continuous", False),
        ("published_corruption", True),
    ):
        model = endpoint_models[branch]
        if (
            not isinstance(model, Mapping)
            or model.get("schema")
            != "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model"
            or model.get("schema_version") != 2
        ):
            raise ValueError(
                f"Expected {branch} to contain endpoint-model schema version 2."
            )
        condition = model.get("conditions", {}).get(CONDITION_KEY)
        if not isinstance(condition, Mapping) or condition.get("adequate") is not True:
            raise ValueError(
                f"Expected embedded {branch} endpoint condition to pass its "
                "held-out adequacy gate."
            )
        metadata = model.get("metadata")
        expected_metadata = {
            "preset": OM_PRESET,
            "execution_profile": "hwa_production_cap128",
            "enable_published_corruption": published_corruption,
        }
        if not isinstance(metadata, Mapping) or any(
            metadata.get(key) != expected_value
            for key, expected_value in expected_metadata.items()
        ):
            raise ValueError(
                f"Expected embedded {branch} endpoint metadata to match the "
                "declared OM cap-128 corruption arm."
            )
        PopulationStepEstimator.from_mapping(estimators[branch])
    return value, sha256_file(resolved)



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
    fingerprint = _population_fingerprint(
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



def sample_om_array_population_layout(
    binding_keys: Sequence[str],
    binding_shapes: Sequence[tuple[int, ...]],
    *,
    assignment_seed: int,
    corruption_policy: str,
) -> IbmReramArrayPopulation:
    """Sample a fixed OM population for an explicit auxiliary-array layout."""

    return _sample_om_array_population_layout(
        binding_keys,
        binding_shapes,
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
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
    return sample_om_array_population_layout_external(
        keys,
        shapes,
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        aihwkit_python=aihwkit_python,
        population_path=population_path,
        receipt_path=receipt_path,
    )



def sample_om_array_population_layout_external(
    binding_keys: Sequence[str],
    binding_shapes: Sequence[tuple[int, ...]],
    *,
    assignment_seed: int,
    corruption_policy: str,
    aihwkit_python: Path,
    population_path: Path,
    receipt_path: Path,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    """Sample pinned OM cells for an explicit auxiliary-array layout."""

    keys = tuple(str(key) for key in binding_keys)
    shapes = tuple(tuple(int(value) for value in shape) for shape in binding_shapes)
    if (
        not keys
        or len(keys) != len(shapes)
        or len(set(keys)) != len(keys)
        or any(not key for key in keys)
        or any(
            len(shape) != 2 or any(value < 1 for value in shape)
            for shape in shapes
        )
    ):
        raise ValueError(
            "Expected unique named two-dimensional auxiliary OM bindings."
        )
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
