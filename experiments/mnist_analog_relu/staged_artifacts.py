"""Relocatable artifacts for the staged IBM-OM crossbar experiment.

The public run manifest authenticates the bytes of every artifact.  These
codecs additionally validate their internal physical identities so a copied
artifact can be restored without consulting an untracked sidecar file.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.artifacts import content_hash, sha256_file
from training.checkpoint import atomic_torch_save
from training.ibm_om_standard_crossbar import (
    CROSSBAR_TRAJECTORY_SEED_DERIVATION,
    CrossbarTileSpec,
    IbmOmCrossbarStateBundle,
    build_crossbar_layout,
    crossbar_trajectory_seeds,
    tensor_sha256,
)
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    om_array_population_fingerprint,
)


HWA_SCHEMA = "ebl.ibm_om_crossbar_hwa_master"
DEVICE_STATE_SCHEMA = "ebl.ibm_om_crossbar_staged_device_state"
SCHEMA_VERSION = 1
STAGED_PROGRAM_STREAM_ROLE = "staged_healthy_program_verify"
_POPULATION_FIELDS = {
    "assignment_seed",
    "corruption_policy",
    "binding_keys",
    "binding_shapes",
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _plain_layout(layout: Sequence[CrossbarTileSpec]) -> list[dict[str, Any]]:
    return [
        {
            "key": tile.key,
            "layer_index": tile.layer_index,
            "tile_index": tile.tile_index,
            "input_start": tile.input_start,
            "input_stop": tile.input_stop,
            "out_features": tile.out_features,
        }
        for tile in layout
    ]


def _layout(value: Any) -> tuple[CrossbarTileSpec, ...]:
    fields = {
        "key",
        "layer_index",
        "tile_index",
        "input_start",
        "input_stop",
        "out_features",
    }
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a non-empty serialized crossbar layout.")
    result: list[CrossbarTileSpec] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError("Expected exact serialized crossbar layout fields.")
        result.append(CrossbarTileSpec(**dict(item)))
    return tuple(result)


def population_state(population: IbmReramArrayPopulation) -> dict[str, Any]:
    """Encode one complete OM assignment using tensors and plain containers."""

    return {
        "assignment_seed": population.assignment_seed,
        "corruption_policy": population.corruption_policy,
        "binding_keys": list(population.binding_keys),
        "binding_shapes": [list(shape) for shape in population.binding_shapes],
        "binding_sampling_seeds": list(population.binding_sampling_seeds),
        "donor_sampling_seeds": list(population.donor_sampling_seeds),
        "nominal_dw_min": population.nominal_dw_min,
        "dw_min_std": population.dw_min_std,
        "write_noise_std": population.write_noise_std,
        **population.tensor_state(),
        "fingerprint": population.fingerprint,
        "aihwkit_version": population.aihwkit_version,
    }


def population_from_state(value: Any) -> IbmReramArrayPopulation:
    """Decode and recompute the identity fingerprint of an embedded array."""

    if not isinstance(value, Mapping) or set(value) != _POPULATION_FIELDS:
        raise ValueError("Expected an exact embedded IBM OM population.")
    keys = tuple(value["binding_keys"])
    shapes = tuple(tuple(item) for item in value["binding_shapes"])
    base_seeds = tuple(value["binding_sampling_seeds"])
    donor_seeds = tuple(value["donor_sampling_seeds"])
    if (
        not keys
        or any(not isinstance(key, str) or not key for key in keys)
        or len(keys) != len(shapes)
        or len(base_seeds) != len(keys)
        or len(donor_seeds) != len(keys)
        or any(
            len(shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape)
            for shape in shapes
        )
        or any(isinstance(item, bool) or not isinstance(item, int) for item in (*base_seeds, *donor_seeds))
    ):
        raise ValueError("Expected a valid embedded OM binding layout and seeds.")
    tensors: dict[str, torch.Tensor] = {}
    size = sum(math.prod(shape) for shape in shapes)
    for name, dtype in (
        ("max_bound", torch.float32),
        ("min_bound", torch.float32),
        ("dwmin_up", torch.float32),
        ("dwmin_down", torch.float32),
        ("reference", torch.float32),
        ("corrupt", torch.bool),
        ("published_corrupt", torch.bool),
    ):
        item = value[name]
        if (
            not isinstance(item, torch.Tensor)
            or item.dtype != dtype
            or item.shape != (size,)
        ):
            raise ValueError(f"Expected embedded population tensor {name!r} to match.")
        tensors[name] = item.detach().cpu().clone()
    scalars: dict[str, float] = {}
    for name in ("nominal_dw_min", "dw_min_std", "write_noise_std"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"Expected finite embedded population scalar {name!r}.")
        scalars[name] = float(item)
    assignment_seed = value["assignment_seed"]
    if isinstance(assignment_seed, bool) or not isinstance(assignment_seed, int) or assignment_seed < 1:
        raise ValueError("Expected a positive embedded assignment seed.")
    if value["aihwkit_version"] != "1.1.0":
        raise ValueError("Expected an AIHWKit 1.1.0 embedded population.")
    fingerprint = om_array_population_fingerprint(
        assignment_seed=assignment_seed,
        corruption_policy=value["corruption_policy"],
        keys=keys,
        shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        scalar_parameters={"aihwkit_version": value["aihwkit_version"], **scalars},
        tensors=tensors,
    )
    if value["fingerprint"] != fingerprint:
        raise ValueError("Expected the embedded OM population fingerprint to match.")
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy=value["corruption_policy"],
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
        aihwkit_version=value["aihwkit_version"],
    )


@dataclass(frozen=True)
class HwaMaster:
    dims: tuple[int, int, int]
    layout: tuple[CrossbarTileSpec, ...]
    digital_scales: tuple[float, float]
    fixed_final_master_q: torch.Tensor
    teacher_sha256: str
    hwa_population_fingerprint: str
    metadata: Mapping[str, Any]
    report: Mapping[str, Any]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": HWA_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "dims": list(self.dims),
            "layout": _plain_layout(self.layout),
            "digital_scales": list(self.digital_scales),
            "fixed_final_master_q": self.fixed_final_master_q.detach().cpu().clone(),
            "fixed_final_master_q_sha256": tensor_sha256(self.fixed_final_master_q),
            "teacher_sha256": self.teacher_sha256,
            "hwa_population_fingerprint": self.hwa_population_fingerprint,
            "metadata": dict(self.metadata),
            "metadata_sha256": content_hash(dict(self.metadata)),
            "report": dict(self.report),
        }


def save_hwa_master(path: Path, master: HwaMaster) -> Path:
    return atomic_torch_save(master.state_dict(), path)


def load_hwa_master(path: Path) -> HwaMaster:
    raw = _load_torch_mapping(path)
    expected = {
        "schema",
        "schema_version",
        "dims",
        "layout",
        "digital_scales",
        "fixed_final_master_q",
        "fixed_final_master_q_sha256",
        "teacher_sha256",
        "hwa_population_fingerprint",
        "metadata",
        "metadata_sha256",
        "report",
    }
    if (
        set(raw) != expected
        or raw.get("schema") != HWA_SCHEMA
        or raw.get("schema_version") != SCHEMA_VERSION
        or not _is_sha256(raw.get("teacher_sha256"))
        or not isinstance(raw.get("metadata"), Mapping)
        or raw.get("metadata_sha256") != content_hash(dict(raw["metadata"]))
    ):
        raise ValueError("Expected a valid staged IBM OM HWA master.")
    dims = tuple(raw["dims"])
    scales = tuple(float(item) for item in raw["digital_scales"])
    state = raw["fixed_final_master_q"]
    if (
        len(dims) != 3
        or len(scales) != 2
        or any(not math.isfinite(item) or item <= 0.0 for item in scales)
        or not isinstance(state, torch.Tensor)
        or state.dtype != torch.float32
        or state.ndim != 1
        or raw["fixed_final_master_q_sha256"] != tensor_sha256(state)
    ):
        raise ValueError("Expected matching HWA dimensions, scales, and master q.")
    return HwaMaster(
        dims=dims,
        layout=_layout(raw["layout"]),
        digital_scales=(scales[0], scales[1]),
        fixed_final_master_q=state.detach().cpu().clone(),
        teacher_sha256=raw["teacher_sha256"],
        hwa_population_fingerprint=raw["hwa_population_fingerprint"],
        metadata=dict(raw["metadata"]),
        report=dict(raw["report"]),
    )


@dataclass(frozen=True)
class StagedDeviceState:
    role: str
    source_kind: str
    dims: tuple[int, int, int]
    assignment_seed: int
    endpoint_seed: int
    teacher_sha256: str
    source_artifact_sha256: str | None
    parent_device_state_sha256: str | None
    healthy_population: IbmReramArrayPopulation
    published_population: IbmReramArrayPopulation
    healthy_p0: IbmOmCrossbarStateBundle
    current: IbmOmCrossbarStateBundle
    recovery: Mapping[str, Any] | None

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": DEVICE_STATE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "role": self.role,
            "source_kind": self.source_kind,
            "dims": list(self.dims),
            "assignment_seed": self.assignment_seed,
            "endpoint_seed": self.endpoint_seed,
            "teacher_sha256": self.teacher_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "parent_device_state_sha256": self.parent_device_state_sha256,
            "healthy_population": population_state(self.healthy_population),
            "published_population": population_state(self.published_population),
            "healthy_p0": self.healthy_p0.state_dict(),
            "current": self.current.state_dict(),
            "recovery": None if self.recovery is None else dict(self.recovery),
        }


def save_device_state(path: Path, state: StagedDeviceState) -> Path:
    return atomic_torch_save(state.state_dict(), path)


def load_device_state(path: Path) -> StagedDeviceState:
    raw = _load_torch_mapping(path)
    expected = {
        "schema",
        "schema_version",
        "role",
        "source_kind",
        "dims",
        "assignment_seed",
        "endpoint_seed",
        "teacher_sha256",
        "source_artifact_sha256",
        "parent_device_state_sha256",
        "healthy_population",
        "published_population",
        "healthy_p0",
        "current",
        "recovery",
    }
    if (
        set(raw) != expected
        or raw.get("schema") != DEVICE_STATE_SCHEMA
        or raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("role") not in {"healthy_p0", "faulted_p0", "adam_final"}
        or raw.get("source_kind") not in {"teacher", "hwa_master", "scratch"}
        or not _is_sha256(raw.get("teacher_sha256"))
        or any(
            item is not None and not _is_sha256(item)
            for item in (
                raw.get("source_artifact_sha256"),
                raw.get("parent_device_state_sha256"),
            )
        )
    ):
        raise ValueError("Expected a valid staged IBM OM device-state artifact.")
    dims = tuple(raw["dims"])
    assignment_seed = raw["assignment_seed"]
    endpoint_seed = raw["endpoint_seed"]
    if (
        dims != (784, 256, 10)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in (assignment_seed, endpoint_seed)
        )
        or (raw["recovery"] is not None and not isinstance(raw["recovery"], Mapping))
    ):
        raise ValueError("Expected fixed dimensions, positive seeds, and recovery metadata.")
    healthy = population_from_state(raw["healthy_population"])
    published = population_from_state(raw["published_population"])
    healthy_p0 = IbmOmCrossbarStateBundle.from_state_dict(raw["healthy_p0"])
    current = IbmOmCrossbarStateBundle.from_state_dict(raw["current"])
    expected_layout = build_crossbar_layout(dims, maximum_input_size=512)
    _validate_population_companions(healthy, published)
    expected_trajectory_seeds = crossbar_trajectory_seeds(
        healthy,
        endpoint_seed=endpoint_seed,
        stream_role=STAGED_PROGRAM_STREAM_ROLE,
        random_stream_population_fingerprint=healthy.fingerprint,
    )
    expected_seed_origin = {
        "schema": "ebl.ibm_om_crossbar_trajectory_seed_origin",
        "schema_version": 1,
        "derivation": CROSSBAR_TRAJECTORY_SEED_DERIVATION,
        "stream_role": STAGED_PROGRAM_STREAM_ROLE,
        "assignment_seed": assignment_seed,
        "endpoint_seed": endpoint_seed,
        "random_stream_population_fingerprint": healthy.fingerprint,
        "trajectory_seeds_sha256": tensor_sha256(expected_trajectory_seeds),
    }
    for name, bundle in (("healthy_p0", healthy_p0), ("current", current)):
        plant_state = bundle.plant_state
        saved_seeds = plant_state.get("trajectory_seeds")
        if (
            plant_state.get("trajectory_seed_derivation")
            != CROSSBAR_TRAJECTORY_SEED_DERIVATION
            or not isinstance(saved_seeds, torch.Tensor)
            or saved_seeds.device.type != "cpu"
            or saved_seeds.dtype != torch.int64
            or saved_seeds.shape != expected_trajectory_seeds.shape
            or not torch.equal(saved_seeds, expected_trajectory_seeds)
        ):
            raise ValueError(
                f"Expected {name} to retain the exact declared staged trajectory seeds."
            )
    if healthy_p0.metadata.get("trajectory_seed_origin") != expected_seed_origin:
        raise ValueError(
            "Expected healthy P0 metadata to authenticate the staged trajectory-seed origin."
        )
    transition = current.plant_state.get("fault_transition")
    healthy_plant = healthy_p0.plant_state
    transition_matches_p0 = (
        isinstance(transition, Mapping)
        and transition.get("pre_fault_persistent_sha256")
        == tensor_sha256(healthy_plant["persistent"])
        and transition.get("pre_fault_apparent_sha256")
        == tensor_sha256(healthy_plant["apparent"])
    )
    if (
        healthy.assignment_seed != assignment_seed
        or published.assignment_seed != assignment_seed
        or healthy.corruption_policy != "counterfactual_repaired"
        or published.corruption_policy != "published"
        or healthy_p0.population_fingerprint != healthy.fingerprint
        or current.population_fingerprint != healthy.fingerprint
        or healthy_p0.layout != expected_layout
        or current.layout != expected_layout
        or healthy_p0.digital_scales != current.digital_scales
        or healthy_p0.state_kind != "healthy"
        or (raw["role"] == "healthy_p0" and current.state_kind != "healthy")
        or (raw["role"] == "faulted_p0" and current.state_kind != "faulted")
        or (raw["role"] == "adam_final" and raw["recovery"] is None)
        or (raw["role"] != "adam_final" and raw["recovery"] is not None)
        or (raw["role"] == "healthy_p0" and raw["parent_device_state_sha256"] is not None)
        or (raw["role"] != "healthy_p0" and raw["parent_device_state_sha256"] is None)
        or (
            raw["role"] == "healthy_p0"
            and current.plant_state_sha256 != healthy_p0.plant_state_sha256
        )
        or (current.state_kind == "faulted" and not transition_matches_p0)
    ):
        raise ValueError("Expected staged populations, seeds, and state roles to agree.")
    return StagedDeviceState(
        role=raw["role"],
        source_kind=raw["source_kind"],
        dims=dims,
        assignment_seed=assignment_seed,
        endpoint_seed=endpoint_seed,
        teacher_sha256=raw["teacher_sha256"],
        source_artifact_sha256=raw["source_artifact_sha256"],
        parent_device_state_sha256=raw["parent_device_state_sha256"],
        healthy_population=healthy,
        published_population=published,
        healthy_p0=healthy_p0,
        current=current,
        recovery=None if raw["recovery"] is None else dict(raw["recovery"]),
    )


def _validate_population_companions(
    healthy: IbmReramArrayPopulation,
    published: IbmReramArrayPopulation,
) -> None:
    """Authenticate repaired/published views of one sampled assignment."""

    if (
        healthy.assignment_seed != published.assignment_seed
        or healthy.binding_keys != published.binding_keys
        or healthy.binding_shapes != published.binding_shapes
        or healthy.binding_sampling_seeds != published.binding_sampling_seeds
        or healthy.donor_sampling_seeds != published.donor_sampling_seeds
        or healthy.nominal_dw_min != published.nominal_dw_min
        or healthy.dw_min_std != published.dw_min_std
        or healthy.write_noise_std != published.write_noise_std
        or healthy.aihwkit_version != published.aihwkit_version
        or bool(torch.any(healthy.corrupt))
        or not bool(torch.any(published.corrupt))
        or not torch.equal(healthy.published_corrupt, published.corrupt)
        or not torch.equal(published.published_corrupt, published.corrupt)
    ):
        raise ValueError("Expected matched repaired and published OM companions.")
    unaffected = ~published.corrupt
    for name in ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference"):
        if not torch.equal(getattr(healthy, name)[unaffected], getattr(published, name)[unaffected]):
            raise ValueError(
                f"Expected non-fault companion field {name!r} to match bit-exactly."
            )


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Expected an existing staged artifact: {source}.")
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Expected a readable weights-only artifact: {source}.") from error
    if not isinstance(value, Mapping):
        raise ValueError("Expected a mapping in the staged artifact.")
    return dict(value)


def artifact_identity(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Expected an existing input artifact: {source}.")
    return {"path": str(source), "sha256": sha256_file(source)}


__all__ = [
    "DEVICE_STATE_SCHEMA",
    "HWA_SCHEMA",
    "HwaMaster",
    "STAGED_PROGRAM_STREAM_ROLE",
    "StagedDeviceState",
    "artifact_identity",
    "load_device_state",
    "load_hwa_master",
    "population_from_state",
    "population_state",
    "save_device_state",
    "save_hwa_master",
]
