"""Versioned, atomic checkpoint codecs for experiment runtimes.

The named-weight format is the canonical model contract.  Positional tensor
lists remain available only through explicit legacy import profiles.  Resume
checkpoints represent an epoch boundary and restore model tensors in place so
optimizers keep valid tensor identities.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from model.resistive.builders import ParameterBinding, ParameterCatalog
from training.core.modifier import ParameterModifier


NAMED_WEIGHTS_SCHEMA = "drn.named-weights"
NAMED_WEIGHTS_SCHEMA_VERSION = 1
EPOCH_BOUNDARY_SCHEMA = "drn.epoch-boundary-resume"
EPOCH_BOUNDARY_SCHEMA_VERSION = 3
RESUME_CAPABILITIES = frozenset({"exact", "stateful_nondeterministic"})


class CheckpointError(ValueError):
    """Raised when checkpoint bytes do not satisfy the declared contract."""


@dataclass(frozen=True)
class LegacyImportProfile:
    """Explicit mapping from a positional legacy list to catalog bindings."""

    name: str
    groups: Optional[tuple[str, ...]]

    def select(self, catalog: ParameterCatalog) -> tuple[ParameterBinding, ...]:
        if self.groups is None:
            return catalog.checkpointed
        allowed = set(self.groups)
        return tuple(
            binding
            for binding in catalog.checkpointed
            if binding.group in allowed
        )


LEGACY_FULL = LegacyImportProfile(name="full", groups=None)
LEGACY_BASE_ONLY = LegacyImportProfile(name="base-only", groups=("base",))


@dataclass(frozen=True)
class CheckpointLoadResult:
    schema: str
    schema_version: int
    restored_keys: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class EpochBoundaryResume:
    epoch: int
    global_step: int
    restored_keys: tuple[str, ...]
    metadata: Mapping[str, Any]
    progress_state: Mapping[str, Any]
    selected_weights: Optional[Mapping[str, Any]]
    resume_capability: str
    optimizer_restored: bool
    modifier_restored: bool
    scheduler_restored: bool
    runtime_state_restored: bool
    progress_restored: bool
    rng_restored: bool
    dataloader_generators_restored: tuple[str, ...]

    @property
    def generators_restored(self) -> bool:
        """Whether at least one named dataloader generator was restored."""

        return bool(self.dataloader_generators_restored)


def _normalize_metadata(
    metadata: Optional[Mapping[str, Any]],
    *,
    name: str = "metadata",
) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise CheckpointError(
            f"Expected {name} to be a JSON-compatible object. "
            f"Provided value: {metadata!r}."
        )
    try:
        encoded = json.dumps(
            dict(metadata),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            f"Expected {name} to be a JSON-compatible object without NaN or "
            f"infinity. Provided value: {metadata!r}."
        ) from exc
    return normalized


def _validate_non_negative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError(
            f"Expected {name} to be a non-negative integer. "
            f"Provided value: {value!r}."
        )
    return value


def _validate_resume_capability(value: Any) -> str:
    if value == "unsupported":
        raise CheckpointError(
            "Expected runs with resume_capability='unsupported' not to "
            "produce an epoch-boundary resume checkpoint. "
            f"Provided value: {value!r}."
        )
    if not isinstance(value, str) or value not in RESUME_CAPABILITIES:
        raise CheckpointError(
            "Expected resume_capability to be 'exact' or "
            "'stateful_nondeterministic'. "
            f"Provided value: {value!r}."
        )
    return value


def _component_state_dict(component: Any, *, name: str) -> dict[Any, Any]:
    try:
        state = component.state_dict()
    except Exception as exc:
        raise CheckpointError(
            f"Expected {name}.state_dict() to return checkpoint state. "
            f"Provided value raised: {str(exc)!r}."
        ) from exc
    if not isinstance(state, Mapping):
        raise CheckpointError(
            f"Expected {name}.state_dict() to return an object. "
            f"Provided value: {state!r}."
        )
    return copy.deepcopy(dict(state))


def _validated_runtime_generators(
    value: Optional[Mapping[str, torch.Generator]],
) -> dict[str, torch.Generator]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CheckpointError(
            "Expected dataloader_generators to be an object mapping names to "
            f"torch.Generator instances. Provided value: {value!r}."
        )
    generators: dict[str, torch.Generator] = {}
    for name, generator in value.items():
        if not isinstance(name, str) or not name:
            raise CheckpointError(
                "Expected every dataloader generator name to be a non-empty "
                f"string. Provided value: {name!r}."
            )
        if not isinstance(generator, torch.Generator):
            raise CheckpointError(
                f"Expected dataloader generator {name!r} to be a "
                "torch.Generator. "
                f"Provided value: {type(generator).__name__}."
            )
        generators[name] = generator
    return dict(sorted(generators.items()))


def _capture_generator_states(
    generators: Mapping[str, torch.Generator],
) -> dict[str, torch.Tensor]:
    return {
        name: generator.get_state().detach().cpu().clone()
        for name, generator in generators.items()
    }


def _validated_generator_states(
    value: Any,
    *,
    generators: Mapping[str, torch.Generator],
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise CheckpointError(
            "Expected dataloader generator states to be an object keyed by "
            f"generator name. Provided value: {value!r}."
        )
    expected_names = set(generators)
    provided_names = set(value)
    if provided_names != expected_names:
        missing = sorted(expected_names - provided_names)
        extra = sorted(provided_names - expected_names, key=repr)
        raise CheckpointError(
            "Expected runtime dataloader generator names to match the resume "
            "checkpoint exactly. "
            f"Provided value: missing={missing!r}, extra={extra!r}."
        )

    normalized: dict[str, torch.Tensor] = {}
    for name, generator in generators.items():
        state = value[name]
        if (
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
        ):
            raise CheckpointError(
                f"Expected dataloader generator state {name!r} to be a "
                "one-dimensional uint8 tensor. "
                f"Provided value: {state!r}."
            )
        staged = state.detach().cpu().clone()
        try:
            probe = torch.Generator(device=generator.device)
            probe.set_state(staged)
        except Exception as exc:
            raise CheckpointError(
                f"Expected dataloader generator state {name!r} to be "
                f"compatible with device {generator.device}. "
                f"Provided value raised: {str(exc)!r}."
            ) from exc
        normalized[name] = staged
    return normalized


def _apply_generator_states(
    generators: Mapping[str, torch.Generator],
    states: Mapping[str, torch.Tensor],
) -> None:
    for name, generator in generators.items():
        generator.set_state(states[name])


def _atomic_torch_save(payload: Any, path: Path | str) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_descriptor = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_torch_save(payload: Any, path: Path | str) -> Path:
    """Persist an arbitrary torch artifact with the checkpoint durability path."""

    return _atomic_torch_save(payload, path)


def _torch_load(path: Path | str) -> Any:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            "Expected checkpoint path to name an existing file. "
            f"Provided value: {str(source)!r}."
        )
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(source, map_location="cpu")


def _explicit_bound(parameter: Any, name: str) -> Optional[float]:
    value = getattr(parameter, name, None)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointError(
            f"Expected parameter {name} to be numeric or null. "
            f"Provided value: {value!r}."
        ) from exc
    if math.isnan(numeric):
        raise CheckpointError(
            f"Expected parameter {name} not to be NaN. "
            f"Provided value: {value!r}."
        )
    return numeric


def _stage_tensor(
    binding: ParameterBinding,
    value: Any,
    *,
    source: str,
) -> torch.Tensor:
    target = binding.state
    if (
        not isinstance(value, torch.Tensor)
        or not torch.is_floating_point(value)
        or value.layout != torch.strided
    ):
        raise CheckpointError(
            f"Expected {source} for {binding.key!r} to be a dense "
            f"floating-point tensor. Provided value: {type(value).__name__}."
        )
    if tuple(value.shape) != tuple(target.shape):
        raise CheckpointError(
            f"Expected {source} for {binding.key!r} to have shape "
            f"{tuple(target.shape)!r}. Provided value: {tuple(value.shape)!r}."
        )
    if value.dtype != target.dtype:
        raise CheckpointError(
            f"Expected {source} for {binding.key!r} to have dtype "
            f"{target.dtype}. Provided value: {value.dtype}."
        )
    if not bool(torch.isfinite(value).all()):
        raise CheckpointError(
            f"Expected {source} for {binding.key!r} to contain only finite "
            "values. Provided value: non-finite tensor."
        )

    staged = value.detach().to(device=target.device, dtype=target.dtype).clone()
    if not bool(torch.isfinite(staged).all()):
        raise CheckpointError(
            f"Expected {source} for {binding.key!r} to remain finite on "
            f"{target.device}. Provided value: non-finite converted tensor."
        )
    lower = _explicit_bound(binding.parameter, "min_cond")
    upper = _explicit_bound(binding.parameter, "max_cond")
    # Parameter clamps operate in the tensor dtype.  Compare against the same
    # representable endpoints so a value produced by ``clamp_(max=0.00011)``
    # is not rejected merely because float32 stores that decimal slightly
    # above the Python float.
    represented_lower = (
        None
        if lower is None
        else float(torch.as_tensor(lower, dtype=target.dtype).item())
    )
    represented_upper = (
        None
        if upper is None
        else float(torch.as_tensor(upper, dtype=target.dtype).item())
    )
    if staged.numel():
        minimum = float(staged.min().item())
        maximum = float(staged.max().item())
        if represented_lower is not None and minimum < represented_lower:
            raise CheckpointError(
                f"Expected {source} for {binding.key!r} to be >= {lower}. "
                f"Provided value: minimum={minimum!r}."
            )
        if represented_upper is not None and maximum > represented_upper:
            raise CheckpointError(
                f"Expected {source} for {binding.key!r} to be <= {upper}. "
                f"Provided value: maximum={maximum!r}."
            )
    return staged


def _apply_staged(
    staged: tuple[tuple[ParameterBinding, torch.Tensor], ...],
) -> tuple[str, ...]:
    snapshots = tuple(
        (binding, binding.state.detach().clone()) for binding, _ in staged
    )
    try:
        with torch.no_grad():
            for binding, value in staged:
                binding.state.copy_(value)
    except Exception as exc:
        with torch.no_grad():
            for binding, snapshot in snapshots:
                binding.state.copy_(snapshot)
        raise CheckpointError(
            "Expected validated checkpoint tensors to copy into the model "
            f"without mutation failure. Provided value: {str(exc)!r}."
        ) from exc
    return tuple(binding.key for binding, _ in staged)


def _structural_descriptor(binding: ParameterBinding) -> dict[str, Any]:
    descriptor = binding.descriptor()
    return {
        "key": descriptor["key"],
        "group": descriptor["group"],
        "role": descriptor["role"],
        "shape": descriptor["shape"],
        "dtype": descriptor["dtype"],
    }


def encode_named_weights(
    catalog: ParameterCatalog,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Encode checkpointed parameters by stable key."""

    bindings = catalog.checkpointed
    return {
        "schema": NAMED_WEIGHTS_SCHEMA,
        "schema_version": NAMED_WEIGHTS_SCHEMA_VERSION,
        "catalog": [_structural_descriptor(binding) for binding in bindings],
        "weights": {
            binding.key: binding.state.detach().cpu().clone()
            for binding in bindings
        },
        "metadata": _normalize_metadata(metadata),
    }


def _stage_named_weights(
    payload: Any,
    catalog: ParameterCatalog,
) -> tuple[
    tuple[tuple[ParameterBinding, torch.Tensor], ...],
    dict[str, Any],
]:
    if not isinstance(payload, Mapping):
        raise CheckpointError(
            "Expected named checkpoint to contain an object. "
            f"Provided value: {type(payload).__name__}."
        )
    schema = payload.get("schema")
    version = payload.get("schema_version")
    if schema != NAMED_WEIGHTS_SCHEMA:
        raise CheckpointError(
            f"Expected named checkpoint schema {NAMED_WEIGHTS_SCHEMA!r}. "
            f"Provided value: {schema!r}."
        )
    if version != NAMED_WEIGHTS_SCHEMA_VERSION:
        raise CheckpointError(
            "Expected named checkpoint schema_version to equal "
            f"{NAMED_WEIGHTS_SCHEMA_VERSION}. Provided value: {version!r}."
        )

    bindings = catalog.checkpointed
    expected_catalog = [
        _structural_descriptor(binding) for binding in bindings
    ]
    provided_catalog = payload.get("catalog")
    if provided_catalog != expected_catalog:
        raise CheckpointError(
            "Expected named checkpoint catalog to match the target parameter "
            "keys, groups, roles, shapes, and dtypes. "
            f"Provided value: {provided_catalog!r}."
        )

    weights = payload.get("weights")
    if not isinstance(weights, Mapping):
        raise CheckpointError(
            "Expected named checkpoint weights to be an object keyed by "
            f"parameter name. Provided value: {weights!r}."
        )
    expected_keys = tuple(binding.key for binding in bindings)
    provided_keys = tuple(weights.keys())
    if set(provided_keys) != set(expected_keys) or len(provided_keys) != len(
        expected_keys
    ):
        missing = sorted(set(expected_keys) - set(provided_keys))
        extra = sorted(set(provided_keys) - set(expected_keys), key=repr)
        raise CheckpointError(
            "Expected named checkpoint weights to contain every checkpointed "
            "parameter key exactly once. "
            f"Provided value: missing={missing!r}, extra={extra!r}."
        )

    staged = tuple(
        (
            binding,
            _stage_tensor(
                binding,
                weights[binding.key],
                source="named checkpoint tensor",
            ),
        )
        for binding in bindings
    )
    metadata = _normalize_metadata(
        payload.get("metadata"),
        name="named checkpoint metadata",
    )
    return staged, metadata


def _clone_validated_named_weights(
    payload: Any,
    catalog: ParameterCatalog,
) -> dict[str, Any]:
    staged, metadata = _stage_named_weights(payload, catalog)
    return {
        "schema": NAMED_WEIGHTS_SCHEMA,
        "schema_version": NAMED_WEIGHTS_SCHEMA_VERSION,
        "catalog": [
            _structural_descriptor(binding) for binding, _value in staged
        ],
        "weights": {
            binding.key: value.detach().cpu().clone()
            for binding, value in staged
        },
        "metadata": metadata,
    }


def save_named_weights(
    path: Path | str,
    catalog: ParameterCatalog,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    return _atomic_torch_save(
        encode_named_weights(catalog, metadata=metadata),
        path,
    )


def save_encoded_named_weights(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    catalog: ParameterCatalog,
) -> Path:
    """Atomically persist validated named weights without applying them."""

    validated = _clone_validated_named_weights(payload, catalog)
    return _atomic_torch_save(validated, path)


def load_named_weights(
    path: Path | str,
    catalog: ParameterCatalog,
) -> CheckpointLoadResult:
    payload = _torch_load(path)
    staged, metadata = _stage_named_weights(payload, catalog)
    restored = _apply_staged(staged)
    return CheckpointLoadResult(
        schema=NAMED_WEIGHTS_SCHEMA,
        schema_version=NAMED_WEIGHTS_SCHEMA_VERSION,
        restored_keys=restored,
        metadata=metadata,
    )


def load_legacy_positional_weights(
    path: Path | str,
    catalog: ParameterCatalog,
    *,
    profile: LegacyImportProfile,
) -> CheckpointLoadResult:
    """Import a legacy tensor list through an explicit positional profile."""

    if not isinstance(profile, LegacyImportProfile):
        raise TypeError(
            "Expected profile to be LEGACY_FULL, LEGACY_BASE_ONLY, or an "
            f"explicit LegacyImportProfile. Provided value: {profile!r}."
        )
    payload = _torch_load(path)
    bindings = profile.select(catalog)
    if not isinstance(payload, (list, tuple)):
        raise CheckpointError(
            "Expected positional legacy checkpoint to contain a list or tuple "
            f"of {len(bindings)} tensors. Provided value: "
            f"{type(payload).__name__}."
        )
    if len(payload) != len(bindings):
        raise CheckpointError(
            f"Expected legacy profile {profile.name!r} to contain exactly "
            f"{len(bindings)} tensors. Provided value: {len(payload)} tensors."
        )
    staged = tuple(
        (
            binding,
            _stage_tensor(
                binding,
                value,
                source=(
                    f"legacy profile {profile.name!r} tensor at index {index}"
                ),
            ),
        )
        for index, (binding, value) in enumerate(zip(bindings, payload))
    )
    restored = _apply_staged(staged)
    return CheckpointLoadResult(
        schema=f"legacy.positional.{profile.name}",
        schema_version=0,
        restored_keys=restored,
        metadata={},
    )


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].astype(np.uint32, copy=False).tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state().cpu().clone(),
        "torch_cuda": (
            [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _validated_rng_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(
            "Expected RNG state to be an object. "
            f"Provided value: {value!r}."
        )
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(value) != required:
        raise CheckpointError(
            "Expected RNG state keys to be python, numpy, torch_cpu, and "
            f"torch_cuda. Provided value: {sorted(value, key=repr)!r}."
        )

    python_state = value["python"]
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            "Expected python RNG state to be accepted by random.setstate(). "
            f"Provided value: {python_state!r}."
        ) from exc

    numpy_value = value["numpy"]
    if not isinstance(numpy_value, Mapping):
        raise CheckpointError(
            "Expected numpy RNG state to be an object. "
            f"Provided value: {numpy_value!r}."
        )
    try:
        numpy_state = (
            str(numpy_value["bit_generator"]),
            np.asarray(numpy_value["state"], dtype=np.uint32),
            int(numpy_value["position"]),
            int(numpy_value["has_gauss"]),
            float(numpy_value["cached_gaussian"]),
        )
        probe = np.random.RandomState()
        probe.set_state(numpy_state)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise CheckpointError(
            "Expected numpy RNG state to be accepted by RandomState.set_state(). "
            f"Provided value: {numpy_value!r}."
        ) from exc

    torch_cpu = value["torch_cpu"]
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
    ):
        raise CheckpointError(
            "Expected torch_cpu RNG state to be a one-dimensional uint8 "
            f"tensor. Provided value: {torch_cpu!r}."
        )
    cuda_value = value["torch_cuda"]
    if not isinstance(cuda_value, (list, tuple)) or not all(
        isinstance(state, torch.Tensor)
        and state.dtype == torch.uint8
        and state.ndim == 1
        for state in cuda_value
    ):
        raise CheckpointError(
            "Expected torch_cuda RNG state to be a list of one-dimensional "
            f"uint8 tensors. Provided value: {cuda_value!r}."
        )
    if cuda_value and not torch.cuda.is_available():
        raise CheckpointError(
            "Expected CUDA to be available when restoring non-empty CUDA RNG "
            f"state. Provided value: {len(cuda_value)} CUDA states."
        )
    if torch.cuda.is_available() and len(cuda_value) not in (
        0,
        torch.cuda.device_count(),
    ):
        raise CheckpointError(
            "Expected CUDA RNG state count to match the visible CUDA device "
            f"count. Provided value: states={len(cuda_value)}, "
            f"devices={torch.cuda.device_count()}."
        )
    return {
        "python": python_state,
        "numpy": numpy_state,
        "torch_cpu": torch_cpu.detach().cpu().clone(),
        "torch_cuda": [
            state.detach().cpu().clone() for state in cuda_value
        ],
    }


def restore_rng_state(value: Mapping[str, Any]) -> None:
    normalized = _validated_rng_state(value)
    previous = capture_rng_state()
    try:
        _apply_validated_rng_state(normalized)
    except Exception:
        rollback = _validated_rng_state(previous)
        _apply_validated_rng_state(rollback)
        raise


def _apply_validated_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.random.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"]:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def save_epoch_boundary_checkpoint(
    path: Path | str,
    *,
    catalog: ParameterCatalog,
    epoch: int,
    global_step: int = 0,
    optimizer: Optional[Any] = None,
    modifier: Optional[ParameterModifier] = None,
    scheduler: Optional[Any] = None,
    runtime_state: Optional[Any] = None,
    progress: Optional[Any] = None,
    progress_state: Optional[Mapping[str, Any]] = None,
    selected_weights: Optional[Mapping[str, Any]] = None,
    dataloader_generators: Optional[
        Mapping[str, torch.Generator]
    ] = None,
    resume_capability: str = "exact",
    metadata: Optional[Mapping[str, Any]] = None,
    include_rng: bool = True,
) -> Path:
    epoch = _validate_non_negative_integer(
        epoch,
        name="epoch at an epoch boundary",
    )
    global_step = _validate_non_negative_integer(
        global_step,
        name="global_step at an epoch boundary",
    )
    resume_capability = _validate_resume_capability(resume_capability)
    if not isinstance(include_rng, bool):
        raise CheckpointError(
            "Expected include_rng to be a bool. "
            f"Provided value: {include_rng!r}."
        )
    normalized_selected_weights = (
        _clone_validated_named_weights(selected_weights, catalog)
        if selected_weights is not None
        else None
    )
    if progress is not None and progress_state is not None:
        raise CheckpointError(
            "Expected at most one of progress or progress_state when saving "
            "an epoch-boundary checkpoint. "
            "Provided value: both were present."
        )
    optimizer_state = (
        _component_state_dict(optimizer, name="optimizer")
        if optimizer is not None
        else None
    )
    modifier_state = (
        _component_state_dict(modifier, name="modifier")
        if modifier is not None
        else None
    )
    scheduler_state = (
        _component_state_dict(scheduler, name="scheduler")
        if scheduler is not None
        else None
    )
    runtime_state_state = (
        _component_state_dict(runtime_state, name="runtime_state")
        if runtime_state is not None
        else None
    )
    if progress is not None:
        normalized_progress = _normalize_metadata(
            _component_state_dict(progress, name="progress"),
            name="progress state",
        )
    else:
        normalized_progress = _normalize_metadata(
            progress_state,
            name="progress state",
        )
    generators = _validated_runtime_generators(dataloader_generators)
    generator_states = _capture_generator_states(generators)
    normalized_metadata = _normalize_metadata(metadata)
    payload = {
        "schema": EPOCH_BOUNDARY_SCHEMA,
        "schema_version": EPOCH_BOUNDARY_SCHEMA_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "weights": encode_named_weights(catalog),
        "optimizer_state": optimizer_state,
        "modifier_state": modifier_state,
        "scheduler_state": scheduler_state,
        "runtime_state_state": runtime_state_state,
        "progress_state": normalized_progress,
        "selected_weights": normalized_selected_weights,
        "rng_state": capture_rng_state() if include_rng else None,
        "dataloader_generator_states": generator_states,
        "resume_capability": resume_capability,
        "metadata": normalized_metadata,
    }
    return _atomic_torch_save(payload, path)


def _resume_version(payload: Mapping[str, Any]) -> int:
    version = payload.get("schema_version")
    if version not in (1, 2, EPOCH_BOUNDARY_SCHEMA_VERSION):
        raise CheckpointError(
            "Expected resume schema_version to equal 1, 2, or "
            f"{EPOCH_BOUNDARY_SCHEMA_VERSION}. Provided value: {version!r}."
        )
    return version


def _validate_saved_component_state(
    saved: Any,
    *,
    runtime: Any,
    name: str,
) -> None:
    if (runtime is None) != (saved is None):
        raise CheckpointError(
            f"Expected runtime {name} presence to match the resume checkpoint. "
            f"Provided value: runtime_present={runtime is not None}, "
            f"checkpoint_present={saved is not None}."
        )
    if saved is not None and not isinstance(saved, Mapping):
        raise CheckpointError(
            f"Expected saved {name} state to be an object. "
            f"Provided value: {saved!r}."
        )


def _rollback_component(
    component: Any,
    state: Optional[Mapping[Any, Any]],
    *,
    name: str,
    errors: list[str],
) -> None:
    if component is None or state is None:
        return
    try:
        component.load_state_dict(copy.deepcopy(state))
    except Exception as exc:  # pragma: no cover - catastrophic
        errors.append(f"{name}: {exc}")


def _rollback_parameters(
    snapshots: tuple[tuple[ParameterBinding, torch.Tensor], ...],
    *,
    errors: list[str],
) -> None:
    try:
        with torch.no_grad():
            for binding, snapshot in snapshots:
                binding.state.copy_(snapshot)
    except Exception as exc:  # pragma: no cover - catastrophic
        errors.append(f"parameters: {exc}")


def _legacy_resume_fields(
    payload: Mapping[str, Any],
    *,
    version: int,
) -> tuple[int, Any, Any, Any, Any, str]:
    if version == 1:
        return 0, None, None, {}, {}, "stateful_nondeterministic"
    return (
        payload.get("global_step"),
        payload.get("modifier_state"),
        (
            payload.get("runtime_state_state")
            if version >= 3
            else None
        ),
        payload.get("progress_state"),
        payload.get("dataloader_generator_states"),
        payload.get("resume_capability"),
    )


def load_epoch_boundary_checkpoint(
    path: Path | str,
    *,
    catalog: ParameterCatalog,
    optimizer: Optional[Any] = None,
    modifier: Optional[ParameterModifier] = None,
    scheduler: Optional[Any] = None,
    runtime_state: Optional[Any] = None,
    progress: Optional[Any] = None,
    dataloader_generators: Optional[
        Mapping[str, torch.Generator]
    ] = None,
    restore_rng: bool = True,
    restore_generators: bool = True,
) -> EpochBoundaryResume:
    """Restore a complete epoch boundary, rolling back on application failure."""

    if not isinstance(restore_rng, bool):
        raise CheckpointError(
            "Expected restore_rng to be a bool. "
            f"Provided value: {restore_rng!r}."
        )
    if not isinstance(restore_generators, bool):
        raise CheckpointError(
            "Expected restore_generators to be a bool. "
            f"Provided value: {restore_generators!r}."
        )
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise CheckpointError(
            "Expected epoch-boundary checkpoint to contain an object. "
            f"Provided value: {type(payload).__name__}."
        )
    if payload.get("schema") != EPOCH_BOUNDARY_SCHEMA:
        raise CheckpointError(
            f"Expected resume schema {EPOCH_BOUNDARY_SCHEMA!r}. "
            f"Provided value: {payload.get('schema')!r}."
        )
    version = _resume_version(payload)
    epoch = _validate_non_negative_integer(
        payload.get("epoch"),
        name="resume epoch",
    )
    (
        global_step,
        modifier_state,
        runtime_state_state,
        progress_state,
        generator_states,
        resume_capability,
    ) = _legacy_resume_fields(payload, version=version)
    global_step = _validate_non_negative_integer(
        global_step,
        name="resume global_step",
    )
    resume_capability = _validate_resume_capability(resume_capability)

    staged, _weight_metadata = _stage_named_weights(
        payload.get("weights"),
        catalog,
    )
    selected_weights = payload.get("selected_weights")
    normalized_selected_weights = (
        _clone_validated_named_weights(selected_weights, catalog)
        if selected_weights is not None
        else None
    )
    metadata = _normalize_metadata(
        payload.get("metadata"),
        name="resume metadata",
    )
    normalized_progress = _normalize_metadata(
        progress_state,
        name="resume progress state",
    )
    optimizer_state = payload.get("optimizer_state")
    scheduler_state = payload.get("scheduler_state")
    rng_state = payload.get("rng_state")
    for name, runtime, saved in (
        ("optimizer", optimizer, optimizer_state),
        ("modifier", modifier, modifier_state),
        ("scheduler", scheduler, scheduler_state),
        ("runtime_state", runtime_state, runtime_state_state),
    ):
        _validate_saved_component_state(
            saved,
            runtime=runtime,
            name=name,
        )
    if progress is not None:
        current_progress = _component_state_dict(progress, name="progress")
        _normalize_metadata(
            current_progress,
            name="runtime progress state",
        )
    else:
        current_progress = None

    normalized_rng = None
    if restore_rng:
        if rng_state is None:
            raise CheckpointError(
                "Expected resume checkpoint to include RNG state when "
                "restore_rng=True. Provided value: null."
            )
        normalized_rng = _validated_rng_state(rng_state)

    generators = _validated_runtime_generators(dataloader_generators)
    normalized_generator_states: dict[str, torch.Tensor] = {}
    if restore_generators:
        normalized_generator_states = _validated_generator_states(
            generator_states,
            generators=generators,
        )

    parameter_snapshots = tuple(
        (binding, binding.state.detach().clone()) for binding, _ in staged
    )
    optimizer_snapshot = (
        _component_state_dict(optimizer, name="optimizer")
        if optimizer is not None
        else None
    )
    modifier_snapshot = (
        _component_state_dict(modifier, name="modifier")
        if modifier is not None
        else None
    )
    scheduler_snapshot = (
        _component_state_dict(scheduler, name="scheduler")
        if scheduler is not None
        else None
    )
    runtime_state_snapshot = (
        _component_state_dict(runtime_state, name="runtime_state")
        if runtime_state is not None
        else None
    )
    rng_snapshot = capture_rng_state() if restore_rng else None
    generator_snapshots = (
        _capture_generator_states(generators) if restore_generators else {}
    )

    try:
        restored_keys = _apply_staged(staged)
        if optimizer is not None:
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        if modifier is not None:
            modifier.load_state_dict(copy.deepcopy(modifier_state))
        if scheduler is not None:
            scheduler.load_state_dict(copy.deepcopy(scheduler_state))
        if runtime_state is not None:
            runtime_state.load_state_dict(
                copy.deepcopy(runtime_state_state)
            )
        if progress is not None:
            progress.load_state_dict(copy.deepcopy(normalized_progress))
        if normalized_rng is not None:
            _apply_validated_rng_state(normalized_rng)
        if restore_generators:
            _apply_generator_states(
                generators,
                normalized_generator_states,
            )
    except Exception as exc:
        rollback_errors: list[str] = []
        if restore_generators:
            try:
                _apply_generator_states(generators, generator_snapshots)
            except Exception as rollback_exc:  # pragma: no cover - catastrophic
                rollback_errors.append(
                    f"dataloader generators: {rollback_exc}"
                )
        if rng_snapshot is not None:
            try:
                _apply_validated_rng_state(
                    _validated_rng_state(rng_snapshot)
                )
            except Exception as rollback_exc:  # pragma: no cover - catastrophic
                rollback_errors.append(f"rng: {rollback_exc}")
        _rollback_component(
            progress,
            current_progress,
            name="progress",
            errors=rollback_errors,
        )
        _rollback_component(
            runtime_state,
            runtime_state_snapshot,
            name="runtime_state",
            errors=rollback_errors,
        )
        _rollback_component(
            scheduler,
            scheduler_snapshot,
            name="scheduler",
            errors=rollback_errors,
        )
        _rollback_component(
            modifier,
            modifier_snapshot,
            name="modifier",
            errors=rollback_errors,
        )
        _rollback_component(
            optimizer,
            optimizer_snapshot,
            name="optimizer",
            errors=rollback_errors,
        )
        _rollback_parameters(parameter_snapshots, errors=rollback_errors)
        rollback_suffix = (
            f" Rollback errors: {rollback_errors!r}." if rollback_errors else ""
        )
        raise CheckpointError(
            "Expected epoch-boundary checkpoint to restore atomically. "
            f"Provided value: {str(exc)!r}.{rollback_suffix}"
        ) from exc

    return EpochBoundaryResume(
        epoch=epoch,
        global_step=global_step,
        restored_keys=restored_keys,
        metadata=metadata,
        progress_state=normalized_progress,
        selected_weights=normalized_selected_weights,
        resume_capability=resume_capability,
        optimizer_restored=optimizer is not None,
        modifier_restored=modifier is not None,
        scheduler_restored=scheduler is not None,
        runtime_state_restored=runtime_state is not None,
        progress_restored=progress is not None,
        rng_restored=normalized_rng is not None,
        dataloader_generators_restored=tuple(
            normalized_generator_states
        ),
    )


__all__ = [
    "CheckpointError",
    "CheckpointLoadResult",
    "EPOCH_BOUNDARY_SCHEMA",
    "EPOCH_BOUNDARY_SCHEMA_VERSION",
    "EpochBoundaryResume",
    "LEGACY_BASE_ONLY",
    "LEGACY_FULL",
    "LegacyImportProfile",
    "NAMED_WEIGHTS_SCHEMA",
    "NAMED_WEIGHTS_SCHEMA_VERSION",
    "RESUME_CAPABILITIES",
    "atomic_torch_save",
    "capture_rng_state",
    "encode_named_weights",
    "load_epoch_boundary_checkpoint",
    "load_legacy_positional_weights",
    "load_named_weights",
    "restore_rng_state",
    "save_encoded_named_weights",
    "save_epoch_boundary_checkpoint",
    "save_named_weights",
]
