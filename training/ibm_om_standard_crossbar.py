"""Checkpointable IBM-OM primitives for a standard tiled crossbar MLP.

The DRN uses full non-negative conductances in a passive equilibrium solve.
This module implements the deliberately different AIHWKit-style comparator:
each programmable crosspoint exposes the effective signed state ``q=a-r`` to
an ordinary matrix-vector multiply, the hidden non-linearity is digital ReLU,
and a second ordinary MVM produces the logits.

AIHWKit 1.1.0 is still authoritative for sampled OM identity parameters.  The
pulse equation is evaluated here with explicit counter-keyed per-trajectory
random streams because the native CPU tile does not expose a serializable
cycle-to-cycle RNG state.  This keeps persistent deployments and same-backend
recovery forks exactly replayable without a dense future-draw buffer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import torch

from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    om_array_population_fingerprint,
)
from training.ibm_reram_program_verify import (
    OM_PRESET,
    PUBLISHED_CORRUPT_RANGE,
    derive_seed,
)


CROSSBAR_TRAJECTORY_RNG_BACKEND = (
    "per_trajectory_stateless_counter_box_muller_v1"
)
CROSSBAR_TRAJECTORY_REPRODUCIBILITY_SCOPE = (
    "bit_exact_for_identical_saved_state_command_sequence_torch_version_and_"
    "execution_backend;cpu_cuda_cross_backend_values_are_not_claimed_bit_exact"
)
CROSSBAR_TRAJECTORY_STATISTICAL_CONTRACT = (
    "two_independently_projected_31_bit_murmur_finalizer_lanes_form_each_box_muller_"
    "normal;62_bit_lane_pair_key_collision_space_and_float32_output;simulator_"
    "pseudorandom_not_cryptographic"
)
CROSSBAR_TRAJECTORY_SEED_DERIVATION = (
    "derive_seed(endpoint_seed,assignment_seed,random_stream_population_"
    "fingerprint,stream_role,ibm_om_crossbar_per_cell_v1), reduced to a "
    "nonwrapping positive 63-bit base followed by unique contiguous "
    "per-cell trajectory seeds in canonical flattened tile order"
)
_LEGACY_GENERATOR_SEED_DERIVATION = (
    "derive_seed(generator.initial_seed,assignment_seed,population_fingerprint,"
    "ibm_om_crossbar_legacy_generator_adapter_v1), reduced to a nonwrapping "
    "positive 63-bit base followed by unique contiguous per-cell trajectory "
    "seeds in canonical flattened tile order"
)
_MAX_TRAJECTORY_SEED = 2**63 - 1


@dataclass(frozen=True)
class CrossbarTileSpec:
    """One physical input-sliced crossbar tile in canonical order."""

    key: str
    layer_index: int
    tile_index: int
    input_start: int
    input_stop: int
    out_features: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.input_stop - self.input_start, self.out_features)

    @property
    def cells(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class IbmOmCrossbarStateBundle:
    """Validated, reloadable state boundary for one standard crossbar.

    The physical population remains a separately authenticated artifact.  A
    bundle binds its fingerprint to the exact tiled layout, digital scales,
    persistent/apparent plant state, RNG streams, and caller-owned provenance.
    ``state_dict()`` emits only tensors and plain Python containers so the
    experiment runtime can place it inside its normal hashed checkpoint.
    """

    STATE_SCHEMA = "ebl.ibm_om_crossbar_state_bundle"
    STATE_SCHEMA_VERSION = 1

    layout: tuple[CrossbarTileSpec, ...]
    digital_scales: tuple[float, float]
    state_kind: str
    population_fingerprint: str
    plant_state: Mapping[str, Any]
    plant_state_sha256: str
    metadata: Mapping[str, Any]
    metadata_sha256: str

    def state_dict(self) -> dict[str, Any]:
        """Return a detached plain-container representation of the bundle."""

        state = {
            "schema": self.STATE_SCHEMA,
            "schema_version": self.STATE_SCHEMA_VERSION,
            "layout": tuple(
                {
                    "key": tile.key,
                    "layer_index": tile.layer_index,
                    "tile_index": tile.tile_index,
                    "input_start": tile.input_start,
                    "input_stop": tile.input_stop,
                    "out_features": tile.out_features,
                }
                for tile in self.layout
            ),
            "digital_scales": self.digital_scales,
            "state_kind": self.state_kind,
            "population_fingerprint": self.population_fingerprint,
            "plant_state": _clone_checkpoint_value(self.plant_state),
            "plant_state_sha256": self.plant_state_sha256,
            "metadata": _clone_checkpoint_value(self.metadata),
            "metadata_sha256": self.metadata_sha256,
        }
        # Frozen dataclasses do not recursively freeze a caller's nested
        # mappings.  Fail closed if one was mutated after construction.
        if _structured_sha256(state["plant_state"]) != self.plant_state_sha256:
            raise RuntimeError("Expected the bundled plant state to remain immutable.")
        _, metadata_sha256 = _canonical_metadata(state["metadata"])
        if metadata_sha256 != self.metadata_sha256:
            raise RuntimeError("Expected bundled provenance metadata to remain immutable.")
        return state

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "IbmOmCrossbarStateBundle":
        """Parse and independently validate a serialized bundle."""

        expected = {
            "schema",
            "schema_version",
            "layout",
            "digital_scales",
            "state_kind",
            "population_fingerprint",
            "plant_state",
            "plant_state_sha256",
            "metadata",
            "metadata_sha256",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state.get("schema") != cls.STATE_SCHEMA
            or isinstance(state.get("schema_version"), bool)
            or state.get("schema_version") != cls.STATE_SCHEMA_VERSION
        ):
            raise ValueError("Expected an IBM OM crossbar state bundle version 1.")
        layout = _layout_from_bundle_state(state["layout"])
        scales = _normalize_digital_scales(state["digital_scales"])
        state_kind = state["state_kind"]
        population_fingerprint = state["population_fingerprint"]
        plant_state = state["plant_state"]
        plant_state_sha256 = state["plant_state_sha256"]
        metadata_sha256 = state["metadata_sha256"]
        if (
            state_kind not in {"healthy", "faulted"}
            or not isinstance(population_fingerprint, str)
            or not population_fingerprint
            or not isinstance(plant_state, Mapping)
            or not _is_sha256(plant_state_sha256)
            or not _is_sha256(metadata_sha256)
        ):
            raise ValueError("Expected valid crossbar bundle identity fields.")
        if _structured_sha256(plant_state) != plant_state_sha256:
            raise ValueError("Expected the bundled plant-state digest to match.")
        if plant_state.get("population_fingerprint") != population_fingerprint:
            raise ValueError("Expected the bundle and plant population fingerprints to match.")
        plant_schema_version = plant_state.get("schema_version")
        transition = plant_state.get("fault_transition")
        mask = plant_state.get("post_deployment_fault_mask")
        inferred_kind = "healthy"
        if plant_schema_version in {2, 3} and transition is not None:
            inferred_kind = "faulted"
        if inferred_kind != state_kind or (
            plant_schema_version in {2, 3}
            and isinstance(mask, torch.Tensor)
            and bool(torch.any(mask)) != (state_kind == "faulted")
        ):
            raise ValueError("Expected the declared bundle state kind to match the plant.")
        canonical_metadata, actual_metadata_sha256 = _canonical_metadata(
            state["metadata"]
        )
        if actual_metadata_sha256 != metadata_sha256:
            raise ValueError("Expected the bundled provenance digest to match.")
        return cls(
            layout=layout,
            digital_scales=scales,
            state_kind=state_kind,
            population_fingerprint=population_fingerprint,
            plant_state=_clone_checkpoint_value(plant_state),
            plant_state_sha256=plant_state_sha256,
            metadata=canonical_metadata,
            metadata_sha256=metadata_sha256,
        )


def build_crossbar_layout(
    dims: Sequence[int],
    *,
    maximum_input_size: int,
) -> tuple[CrossbarTileSpec, ...]:
    """Split each dense input dimension as AIHWKit's balanced tile array does."""

    dimensions = tuple(int(value) for value in dims)
    if (
        len(dimensions) != 3
        or any(value < 1 for value in dimensions)
        or isinstance(maximum_input_size, bool)
        or maximum_input_size < 1
    ):
        raise ValueError(
            "Expected three positive dimensions and a positive maximum input size."
        )
    result: list[CrossbarTileSpec] = []
    for layer_index, (inputs, outputs) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        tile_count = math.ceil(inputs / maximum_input_size)
        base, remainder = divmod(inputs, tile_count)
        start = 0
        for tile_index in range(tile_count):
            width = base + (1 if tile_index < remainder else 0)
            stop = start + width
            result.append(
                CrossbarTileSpec(
                    key=f"crossbar.layer{layer_index}.tile{tile_index}",
                    layer_index=layer_index,
                    tile_index=tile_index,
                    input_start=start,
                    input_stop=stop,
                    out_features=outputs,
                )
            )
            start = stop
        if start != inputs:  # pragma: no cover - arithmetic invariant
            raise RuntimeError("Expected the balanced tile split to cover its input.")
    return tuple(result)


def _normalize_crossbar_layout(
    layout: Sequence[CrossbarTileSpec],
) -> tuple[CrossbarTileSpec, ...]:
    tiles = tuple(layout)
    if not tiles or any(not isinstance(tile, CrossbarTileSpec) for tile in tiles):
        raise ValueError("Expected a non-empty standard-crossbar tile layout.")
    expected_order: list[CrossbarTileSpec] = []
    layer_outputs: list[int] = []
    layer_inputs: list[int] = []
    for layer_index in (0, 1):
        layer = tuple(tile for tile in tiles if tile.layer_index == layer_index)
        if not layer:
            raise ValueError("Expected both logical layers in the crossbar layout.")
        start = 0
        output_features = layer[0].out_features
        for tile_index, tile in enumerate(layer):
            if (
                tile.key != f"crossbar.layer{layer_index}.tile{tile_index}"
                or tile.tile_index != tile_index
                or tile.input_start != start
                or tile.input_stop <= tile.input_start
                or tile.out_features != output_features
                or output_features < 1
            ):
                raise ValueError("Expected a canonical contiguous crossbar tile layout.")
            start = tile.input_stop
            expected_order.append(tile)
        layer_inputs.append(start)
        layer_outputs.append(output_features)
    if (
        tiles != tuple(expected_order)
        or layer_inputs[1] != layer_outputs[0]
        or len({tile.key for tile in tiles}) != len(tiles)
    ):
        raise ValueError("Expected an ordered compatible two-layer crossbar layout.")
    return tiles


def _layout_from_bundle_state(value: Any) -> tuple[CrossbarTileSpec, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("Expected a serialized crossbar tile layout.")
    fields = {
        "key",
        "layer_index",
        "tile_index",
        "input_start",
        "input_stop",
        "out_features",
    }
    tiles = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError("Expected exact serialized crossbar tile fields.")
        integer_names = fields - {"key"}
        if (
            not isinstance(item["key"], str)
            or not item["key"]
            or any(
                isinstance(item[name], bool) or not isinstance(item[name], int)
                for name in integer_names
            )
        ):
            raise ValueError("Expected typed serialized crossbar tile fields.")
        tiles.append(
            CrossbarTileSpec(
                key=item["key"],
                layer_index=item["layer_index"],
                tile_index=item["tile_index"],
                input_start=item["input_start"],
                input_stop=item["input_stop"],
                out_features=item["out_features"],
            )
        )
    return _normalize_crossbar_layout(tiles)


def _normalize_digital_scales(value: Sequence[float]) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("Expected two serialized digital scales.")
    if len(value) != 2 or any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise ValueError("Expected two finite positive digital scales.")
    scales = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item <= 0.0 for item in scales):
        raise ValueError("Expected two finite positive digital scales.")
    return scales[0], scales[1]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _clone_checkpoint_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_checkpoint_value(item) for item in value]
    return deepcopy(value)


def _update_structured_digest(digest: Any, value: Any) -> None:
    """Hash the tensor/plain-container subset used by crossbar checkpoints."""

    if value is None:
        digest.update(b"none;")
    elif isinstance(value, bool):
        digest.update(b"bool:1;" if value else b"bool:0;")
    elif isinstance(value, int):
        digest.update(f"int:{value};".encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Expected finite values in a crossbar checkpoint.")
        digest.update(f"float:{value.hex()};".encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        digest.update(b";")
    elif isinstance(value, torch.Tensor):
        digest.update(b"tensor:")
        digest.update(tensor_sha256(value).encode("ascii"))
        digest.update(b";")
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Expected string keys in a crossbar checkpoint mapping.")
        digest.update(f"mapping:{len(value)}:".encode("ascii"))
        for key in sorted(value):
            _update_structured_digest(digest, key)
            _update_structured_digest(digest, value[key])
        digest.update(b";")
    elif isinstance(value, (tuple, list)):
        kind = "tuple" if isinstance(value, tuple) else "list"
        digest.update(f"{kind}:{len(value)}:".encode("ascii"))
        for item in value:
            _update_structured_digest(digest, item)
        digest.update(b";")
    else:
        raise ValueError(
            "Expected only tensors and plain containers in a crossbar checkpoint."
        )


def _structured_sha256(value: Any) -> str:
    digest = sha256()
    _update_structured_digest(digest, value)
    return digest.hexdigest()


def _canonical_metadata(value: Any) -> tuple[dict[str, Any], str]:
    """Return strict JSON provenance and its canonical content digest."""

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool)):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Expected finite crossbar provenance numbers.")
            return item
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Expected string keys in crossbar provenance metadata.")
            return {key: normalize(item[key]) for key in sorted(item)}
        raise ValueError("Expected JSON-compatible crossbar provenance metadata.")

    if not isinstance(value, Mapping):
        raise ValueError("Expected crossbar provenance metadata to be a mapping.")
    canonical = normalize(value)
    encoded = json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return canonical, sha256(encoded).hexdigest()


def aihwkit_analog_linear_default_logical_weights(
    dims: Sequence[int],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the default bias-free ``AnalogLinear`` initialization.

    AIHWKit 1.1.0 delegates ``AnalogLinear.reset_parameters`` to PyTorch's
    ``Linear.reset_parameters``.  The weight draw is therefore Kaiming-uniform
    with ``a=sqrt(5)`` in the native ``[out_features, in_features]`` layout.
    This comparator stores logical matrices transposed as ``[in, out]``.

    A dedicated CPU generator makes the starting logical weights independent
    of device-population sampling and CUDA RNG state.  This copies the
    initialization equation and draw order; device programming remains the
    explicit IBM-OM program-and-verify step performed by the experiment.
    """

    dimensions = tuple(int(value) for value in dims)
    if len(dimensions) != 3 or any(value < 1 for value in dimensions):
        raise ValueError("Expected three positive logical dimensions.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Expected a non-negative integer initialization seed.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    logical = []
    for in_features, out_features in zip(
        dimensions[:-1], dimensions[1:], strict=True
    ):
        native_weight = torch.empty(
            (out_features, in_features),
            dtype=torch.float32,
            device="cpu",
        )
        torch.nn.init.kaiming_uniform_(
            native_weight,
            a=math.sqrt(5.0),
            generator=generator,
        )
        logical.append(native_weight.transpose(0, 1).contiguous())
    return logical[0], logical[1]


def validate_population_layout(
    population: IbmReramArrayPopulation,
    layout: Sequence[CrossbarTileSpec],
) -> None:
    expected_keys = tuple(tile.key for tile in layout)
    expected_shapes = tuple(tile.shape for tile in layout)
    if (
        population.binding_keys != expected_keys
        or population.binding_shapes != expected_shapes
        or population.size != sum(tile.cells for tile in layout)
    ):
        raise ValueError(
            "Expected the IBM OM population to match the standard-crossbar "
            f"layout exactly. Provided value: keys={population.binding_keys!r}, "
            f"shapes={population.binding_shapes!r}."
        )


def apply_population_bound_policy(
    population: IbmReramArrayPopulation,
    *,
    policy: str,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    """Apply the declared native or DRN-matched bound treatment.

    The matched policy reproduces the recent four-device DRN control's only
    bound intervention: sample and repair the complete identity first, then
    intersect each raw active-state support with ``a in [-1, 1]``.  It never
    resamples a cell and it leaves pulse sizes and intrinsic references fixed.
    """

    if policy not in {
        "native_sampled_bounds",
        "winsorize_raw_active_a_to_unit_interval",
    }:
        raise ValueError("Expected a supported IBM OM population-bound policy.")
    original_minimum = population.min_bound.detach().cpu().to(torch.float32)
    original_maximum = population.max_bound.detach().cpu().to(torch.float32)
    if policy == "native_sampled_bounds":
        minimum = original_minimum.clone()
        maximum = original_maximum.clone()
    else:
        minimum = torch.maximum(original_minimum, torch.full_like(original_minimum, -1.0))
        maximum = torch.minimum(original_maximum, torch.full_like(original_maximum, 1.0))
    empty = maximum < minimum
    if bool(torch.any(empty)):
        raise RuntimeError(
            "An IBM OM identity has no support after applying the declared "
            "bound policy; refusing to resample it."
        )
    reference = population.reference.detach().cpu().to(torch.float32)
    reference_outside = (reference < minimum) | (reference > maximum)

    lower_changed = minimum != original_minimum
    upper_changed = maximum != original_maximum
    fingerprint = population.fingerprint
    if policy != "native_sampled_bounds":
        fingerprint = om_array_population_fingerprint(
            assignment_seed=population.assignment_seed,
            corruption_policy=population.corruption_policy,
            keys=population.binding_keys,
            shapes=population.binding_shapes,
            binding_sampling_seeds=population.binding_sampling_seeds,
            donor_sampling_seeds=population.donor_sampling_seeds,
            scalar_parameters={
                "aihwkit_version": population.aihwkit_version,
                "nominal_dw_min": population.nominal_dw_min,
                "dw_min_std": population.dw_min_std,
                "write_noise_std": population.write_noise_std,
            },
            tensors={
                "max_bound": maximum,
                "min_bound": minimum,
                "dwmin_up": population.dwmin_up,
                "dwmin_down": population.dwmin_down,
                "reference": reference,
                "corrupt": population.corrupt,
                "published_corrupt": population.published_corrupt,
            },
        )
    result = IbmReramArrayPopulation(
        assignment_seed=population.assignment_seed,
        corruption_policy=population.corruption_policy,
        binding_keys=population.binding_keys,
        binding_shapes=population.binding_shapes,
        binding_sampling_seeds=population.binding_sampling_seeds,
        donor_sampling_seeds=population.donor_sampling_seeds,
        nominal_dw_min=population.nominal_dw_min,
        dw_min_std=population.dw_min_std,
        write_noise_std=population.write_noise_std,
        max_bound=maximum.clone(),
        min_bound=minimum.clone(),
        dwmin_up=population.dwmin_up.detach().cpu().clone(),
        dwmin_down=population.dwmin_down.detach().cpu().clone(),
        reference=reference.clone(),
        corrupt=population.corrupt.detach().cpu().clone(),
        published_corrupt=population.published_corrupt.detach().cpu().clone(),
        fingerprint=fingerprint,
        aihwkit_version=population.aihwkit_version,
    )
    return result, {
        "policy": policy,
        "source_population_fingerprint": population.fingerprint,
        "effective_population_fingerprint": result.fingerprint,
        "identity_resampled": False,
        "raw_active_interval": None if policy == "native_sampled_bounds" else [-1.0, 1.0],
        "lower_bound_changed": int(lower_changed.sum().item()),
        "upper_bound_changed": int(upper_changed.sum().item()),
        "any_bound_changed": int((lower_changed | upper_changed).sum().item()),
        "reference_changed": False,
        "reference_outside_effective_support": int(reference_outside.sum().item()),
    }


def layer_cell_slices(
    layout: Sequence[CrossbarTileSpec],
) -> tuple[slice, slice]:
    """Return the two contiguous flattened cell slices."""

    counts = [0, 0]
    seen = []
    for tile in layout:
        if tile.layer_index not in {0, 1}:
            raise ValueError("Expected exactly two crossbar layers.")
        counts[tile.layer_index] += tile.cells
        seen.append(tile.layer_index)
    if set(seen) != {0, 1}:
        raise ValueError("Expected the layout to contain both crossbar layers.")
    return (slice(0, counts[0]), slice(counts[0], counts[0] + counts[1]))


def map_logical_weights(
    logical_weights: Sequence[torch.Tensor],
    layout: Sequence[CrossbarTileSpec],
    *,
    weight_scaling_omega: Sequence[float],
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Map logical weights to AIHWKit-style effective device targets ``q``.

    One positive digital output scale is shared by all tiles in a logical
    layer.  This avoids the tile-wise calibration confound that would arise if
    each input slice independently used its own maximum.  An omega of zero
    copies AIHWKit's default ``MappingParameter`` behavior: no weight remapping
    and a unit digital scale, so the logical weights are sent directly to the
    device as signed ``q`` targets.
    """

    weights = tuple(
        value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for value in logical_weights
    )
    if len(weights) != 2:
        raise ValueError("Expected exactly two logical weight tensors.")
    omega = tuple(float(value) for value in weight_scaling_omega)
    if len(omega) != 2 or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in omega
    ):
        raise ValueError("Expected two finite weight-scaling omega values in [0, 1].")

    expected_shapes: list[tuple[int, int]] = []
    for layer_index in (0, 1):
        selected = [tile for tile in layout if tile.layer_index == layer_index]
        expected_shapes.append(
            (sum(tile.shape[0] for tile in selected), selected[0].out_features)
        )
    if tuple(value.shape for value in weights) != tuple(expected_shapes):
        raise ValueError(
            "Expected logical weight shapes to match the tiled layout. "
            f"Provided value: {tuple(tuple(value.shape) for value in weights)!r}."
        )

    scales = []
    for layer_index, value in enumerate(weights):
        absolute_maximum = float(value.abs().max().item())
        if not math.isfinite(absolute_maximum) or absolute_maximum <= 0.0:
            raise ValueError("Expected every logical layer to have non-zero finite weights.")
        scales.append(
            1.0
            if omega[layer_index] == 0.0
            else absolute_maximum / omega[layer_index]
        )

    pieces = []
    for tile in layout:
        logical = weights[tile.layer_index]
        target = (
            logical[tile.input_start : tile.input_stop]
            / scales[tile.layer_index]
        )
        pieces.append(target.reshape(-1))
    flattened = torch.cat(pieces)
    if not bool(torch.all(torch.isfinite(flattened))):
        raise RuntimeError("Expected finite effective crossbar targets.")
    return flattened, (float(scales[0]), float(scales[1]))


def effective_state_to_logical_weights(
    effective_state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    digital_scales: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reassemble flattened device states into two logical dense matrices."""

    flat = effective_state.reshape(-1)
    expected = sum(tile.cells for tile in layout)
    scales = tuple(float(value) for value in digital_scales)
    if flat.numel() != expected or len(scales) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in scales
    ):
        raise ValueError("Expected a complete effective state and two positive scales.")
    offsets = 0
    layers: list[list[torch.Tensor]] = [[], []]
    for tile in layout:
        stop = offsets + tile.cells
        value = flat[offsets:stop].reshape(tile.shape)
        layers[tile.layer_index].append(value)
        offsets = stop
    logical = []
    for layer_index in (0, 1):
        logical.append(torch.cat(layers[layer_index], dim=0) * scales[layer_index])
    return logical[0], logical[1]


def standard_crossbar_logits(
    inputs: torch.Tensor,
    effective_state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    digital_scales: Sequence[float],
) -> torch.Tensor:
    """Analog MVM -- digital ReLU -- analog MVM forward equation."""

    return standard_crossbar_forward_states(
        inputs,
        effective_state,
        layout,
        digital_scales=digital_scales,
    ).output_logits


@dataclass(frozen=True)
class StandardCrossbarForwardStates:
    """Locally observable states of the two-crossbar digital-ReLU path."""

    hidden_preactivation: torch.Tensor
    hidden_post_relu: torch.Tensor
    output_logits: torch.Tensor


def standard_crossbar_forward_states(
    inputs: torch.Tensor,
    effective_state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    digital_scales: Sequence[float],
) -> StandardCrossbarForwardStates:
    """Return ``u``, ``h=ReLU(u)``, and ``z`` without hiding local states."""

    weight_0, weight_1 = effective_state_to_logical_weights(
        effective_state,
        layout,
        digital_scales=digital_scales,
    )
    weight_0 = weight_0.to(device=inputs.device, dtype=inputs.dtype)
    weight_1 = weight_1.to(device=inputs.device, dtype=inputs.dtype)
    hidden_preactivation = inputs @ weight_0
    hidden_post_relu = torch.relu(hidden_preactivation)
    return StandardCrossbarForwardStates(
        hidden_preactivation=hidden_preactivation,
        hidden_post_relu=hidden_post_relu,
        output_logits=hidden_post_relu @ weight_1,
    )


def flatten_logical_crossbar_matrices(
    matrices: Sequence[torch.Tensor],
    layout: Sequence[CrossbarTileSpec],
) -> torch.Tensor:
    """Flatten two logical matrices in the physical tile order."""

    values = tuple(matrices)
    if len(values) != 2:
        raise ValueError("Expected exactly two logical crossbar matrices.")
    pieces: list[torch.Tensor] = []
    for tile in layout:
        value = values[tile.layer_index]
        expected_rows = sum(
            item.shape[0]
            for item in layout
            if item.layer_index == tile.layer_index
        )
        if value.ndim != 2 or value.shape != (expected_rows, tile.out_features):
            raise ValueError("Expected logical matrices to match the tiled layout.")
        pieces.append(value[tile.input_start : tile.input_stop].reshape(-1))
    if not pieces:
        raise ValueError("Expected a non-empty two-layer crossbar layout.")
    return torch.cat(pieces)


@dataclass(frozen=True)
class LocalStarCrossbarStep:
    """One label-addressed STAR step made only from layer-local errors."""

    gradient_q: torch.Tensor
    hidden_error: torch.Tensor
    output_error: torch.Tensor
    hidden_loss: torch.Tensor
    output_loss: torch.Tensor
    states: StandardCrossbarForwardStates


@torch.no_grad()
def local_star_crossbar_step(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    effective_state: torch.Tensor,
    layout: Sequence[CrossbarTileSpec],
    *,
    digital_scales: Sequence[float],
    hidden_targets: torch.Tensor,
    output_targets: torch.Tensor,
    hidden_gain: float,
    output_gain: float,
) -> LocalStarCrossbarStep:
    """Compute stopped, layer-local STAR errors and physical-``q`` gradients.

    The class label only addresses the stored target row.  There is no
    softmax/cross-entropy term, teacher query, transpose MVM, or cross-layer
    error propagation.  The hidden and output outer products are gradients of
    two separate local state-matching objectives.
    """

    if inputs.ndim != 2 or labels.ndim != 1 or labels.shape[0] != inputs.shape[0]:
        raise ValueError("Expected rank-2 inputs and one label per example.")
    gains = (float(hidden_gain), float(output_gain))
    if any(not math.isfinite(value) or value < 0.0 for value in gains):
        raise ValueError("Expected finite non-negative STAR gains.")
    scales = tuple(float(value) for value in digital_scales)
    if len(scales) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in scales
    ):
        raise ValueError("Expected two finite positive digital scales.")
    labels_device = labels.to(device=inputs.device, dtype=torch.int64)
    hidden_table = hidden_targets.to(device=inputs.device, dtype=inputs.dtype)
    output_table = output_targets.to(device=inputs.device, dtype=inputs.dtype)
    if (
        hidden_table.ndim != 2
        or output_table.ndim != 2
        or hidden_table.shape[0] != output_table.shape[0]
        or hidden_table.shape[0] < 1
        or bool(torch.any(labels_device < 0))
        or bool(torch.any(labels_device >= hidden_table.shape[0]))
        or not bool(torch.all(torch.isfinite(hidden_table)))
        or not bool(torch.all(torch.isfinite(output_table)))
    ):
        raise ValueError("Expected finite class-indexed STAR target tables.")

    states = standard_crossbar_forward_states(
        inputs,
        effective_state,
        layout,
        digital_scales=scales,
    )
    if (
        hidden_table.shape[1] != states.hidden_post_relu.shape[1]
        or output_table.shape[1] != states.output_logits.shape[1]
    ):
        raise ValueError("Expected STAR target widths to match local network states.")
    batch_size = inputs.shape[0]
    hidden_delta = states.hidden_post_relu - hidden_table[labels_device]
    output_delta = states.output_logits - output_table[labels_device]
    hidden_error = (
        gains[0]
        * hidden_delta
        * (states.hidden_preactivation > 0).to(inputs.dtype)
        / batch_size
    )
    output_error = gains[1] * output_delta / batch_size

    # Stop the inter-layer path: W1 sees only its hidden target error, while W2
    # sees the current hidden state solely as its locally available row signal.
    gradient_w0 = inputs.transpose(0, 1) @ hidden_error
    gradient_w1 = states.hidden_post_relu.detach().transpose(0, 1) @ output_error
    gradient_q = flatten_logical_crossbar_matrices(
        (scales[0] * gradient_w0, scales[1] * gradient_w1),
        layout,
    )
    return LocalStarCrossbarStep(
        gradient_q=gradient_q,
        hidden_error=hidden_error,
        output_error=output_error,
        hidden_loss=0.5 * gains[0] * hidden_delta.square().sum(dim=1).mean(),
        output_loss=0.5 * gains[1] * output_delta.square().sum(dim=1).mean(),
        states=states,
    )


@dataclass(frozen=True)
class DeterministicEffectiveCodebook:
    """Per-cell lower-to-SET no-noise effective-state codebook."""

    values: torch.Tensor
    effective_level_counts: torch.Tensor

    @property
    def maximum_pulses(self) -> int:
        return int(self.values.shape[0]) - 1


def build_deterministic_effective_codebook(
    population: IbmReramArrayPopulation,
    *,
    maximum_pulses: int,
) -> DeterministicEffectiveCodebook:
    """Enumerate exact ``xi=0`` OM states and retain the sampled reference."""

    if isinstance(maximum_pulses, bool) or maximum_pulses < 1:
        raise ValueError("Expected a positive deterministic pulse cap.")
    minimum = population.min_bound.detach().cpu().to(torch.float32)
    active = minimum.clone()
    maximum = population.max_bound.detach().cpu().to(torch.float32)
    step = population.dwmin_up.detach().cpu().to(torch.float32)
    reference = population.reference.detach().cpu().to(torch.float32)
    levels = [active - reference]
    for _ in range(maximum_pulses):
        normalized = torch.where(maximum > 0.0, active / maximum, torch.zeros_like(active))
        active = torch.minimum(
            torch.maximum(active + step * (1.0 - normalized), minimum),
            maximum,
        )
        levels.append(active - reference)
    values = torch.stack(levels)
    if not bool(torch.all(torch.isfinite(values))) or bool(
        torch.any(values[1:] < values[:-1])
    ):
        raise RuntimeError("Expected finite monotone deterministic OM codebooks.")
    counts = 1 + (values[1:] != values[:-1]).sum(dim=0).to(torch.int64)
    corrupt = population.corrupt.detach().cpu()
    if bool(torch.any(counts[corrupt] != 1)) or bool(
        torch.any(values[:, corrupt] != values[0, corrupt].unsqueeze(0))
    ):
        raise RuntimeError(
            "Expected every collapsed zero-step IBM OM cell to retain one "
            "immutable deterministic code."
        )
    return DeterministicEffectiveCodebook(values=values, effective_level_counts=counts)


def project_to_nearest_effective_code(
    codebook: DeterministicEffectiveCodebook,
    targets: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project targets with the lowest-pulse index winning exact ties."""

    target = targets.detach().cpu().to(torch.float32).reshape(-1)
    values = codebook.values
    if (
        values.ndim != 2
        or values.shape[1] != target.numel()
        or chunk_size < 1
        or not bool(torch.all(torch.isfinite(target)))
    ):
        raise ValueError("Expected one finite target for every deterministic codebook.")
    realized = torch.empty_like(target)
    indices = torch.empty(target.shape, dtype=torch.int64)
    for start in range(0, target.numel(), chunk_size):
        stop = min(start + chunk_size, target.numel())
        local = values[:, start:stop]
        selected = torch.abs(local - target[start:stop].unsqueeze(0)).argmin(dim=0)
        realized[start:stop] = local.gather(0, selected.unsqueeze(0)).squeeze(0)
        indices[start:stop] = selected
    return realized, indices


def project_to_nearest_effective_code_device(
    codebook_values: torch.Tensor,
    targets: torch.Tensor,
    *,
    rowwise: bool = False,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project row-wise monotone codebooks without leaving the target device.

    This is the training-time counterpart of
    :func:`project_to_nearest_effective_code`. ``torch.searchsorted`` avoids a
    dense levels-by-cells distance tensor on every QAT minibatch. Exact ties
    select the lower pulse index. Callers may cache a contiguous
    cells-by-levels tensor with ``rowwise=True`` and, after one validated call,
    use ``validate=False`` inside a fixed-codebook training loop.
    """

    if not isinstance(rowwise, bool) or not isinstance(validate, bool):
        raise TypeError("Expected boolean rowwise and validate flags.")
    values = codebook_values
    target = targets.reshape(-1)
    level_count = values.shape[1] if rowwise and values.ndim == 2 else (
        values.shape[0] if values.ndim == 2 else 0
    )
    expected_shape = (
        (target.numel(), values.shape[1])
        if rowwise and values.ndim == 2
        else (values.shape[0], target.numel())
        if values.ndim == 2
        else ()
    )
    if (
        values.ndim != 2
        or values.shape != expected_shape
        or level_count < 2
        or target.numel() < 1
        or values.device != target.device
        or values.dtype != target.dtype
        or (validate and not bool(torch.all(torch.isfinite(target))))
    ):
        raise ValueError("Expected finite monotone device-local codebooks and targets.")
    table = values if rowwise else values.transpose(0, 1).contiguous()
    if validate and (
        not bool(torch.all(torch.isfinite(table)))
        or bool(torch.any(table[:, 1:] < table[:, :-1]))
    ):
        raise ValueError("Expected finite monotone device-local codebooks and targets.")
    insertion = torch.searchsorted(
        table,
        target.unsqueeze(1),
        right=False,
    ).squeeze(1)
    high_index = insertion.clamp(min=1, max=table.shape[1] - 1)
    low_index = high_index - 1
    rows = torch.arange(target.numel(), device=target.device)
    low = table[rows, low_index]
    high = table[rows, high_index]
    choose_high = torch.abs(high - target) < torch.abs(target - low)
    indices = torch.where(choose_high, high_index, low_index)
    realized = table[rows, indices]
    return realized, indices


def _contiguous_trajectory_seeds(*, base_seed: int, size: int) -> torch.Tensor:
    """Expand one authenticated seed into unique nonwrapping cell streams."""

    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or size >= _MAX_TRAJECTORY_SEED
    ):
        raise ValueError("Expected a valid base seed and positive trajectory count.")
    # ``derive_seed`` spans the complete positive signed-63-bit range.  Reduce
    # it before adding the cell coordinate so construction can never wrap or
    # collide within one population, even when the derived value is near the
    # signed-int64 ceiling.
    first = int(base_seed) % (_MAX_TRAJECTORY_SEED - size) + 1
    return torch.arange(first, first + size, dtype=torch.int64, device="cpu")


def crossbar_trajectory_seeds(
    population: IbmReramArrayPopulation,
    *,
    endpoint_seed: int,
    stream_role: str,
    random_stream_population_fingerprint: str | None = None,
) -> torch.Tensor:
    """Derive one explicit independent trajectory seed per physical cell.

    The population fingerprint and stream role are deliberately part of the
    derivation.  Replaying one endpoint on the same frozen array therefore
    reproduces every cell stream, while a different assignment or declared
    physical intervention receives a disjoint keyed stream.  Canonical cell
    order is the population's flattened binding/tile order.
    """

    if not isinstance(population, IbmReramArrayPopulation):
        raise TypeError("Expected one frozen IBM OM array population.")
    if isinstance(endpoint_seed, bool) or not isinstance(endpoint_seed, int):
        raise TypeError("Expected endpoint_seed to be an integer.")
    if not isinstance(stream_role, str) or not stream_role.strip():
        raise ValueError("Expected a non-empty physical random-stream role.")
    fingerprint = (
        population.fingerprint
        if random_stream_population_fingerprint is None
        else random_stream_population_fingerprint
    )
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError("Expected a non-empty random-stream population fingerprint.")
    base = derive_seed(
        endpoint_seed,
        population.assignment_seed,
        fingerprint,
        stream_role.strip(),
        "ibm_om_crossbar_per_cell_v1",
    )
    return _contiguous_trajectory_seeds(base_seed=base, size=population.size)


def _normalize_trajectory_seeds(
    value: Sequence[int] | torch.Tensor,
    *,
    size: int,
) -> torch.Tensor:
    """Validate an auditable seed vector without silently coercing floats."""

    if isinstance(value, torch.Tensor):
        if value.dtype != torch.int64 or value.ndim != 1:
            raise ValueError("Expected trajectory seeds to be a rank-1 int64 tensor.")
        seeds = value.detach().to(device="cpu").clone()
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("Expected one integer trajectory seed per crosspoint.")
        items = tuple(value)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in items):
            raise ValueError("Expected every trajectory seed to be an integer.")
        seeds = torch.tensor(items, dtype=torch.int64, device="cpu")
    if seeds.shape != (size,) or bool(torch.any(seeds < 1)):
        raise ValueError(
            "Expected one positive signed-63-bit trajectory seed per crosspoint."
        )
    if int(torch.unique(seeds).numel()) != size:
        raise ValueError("Expected independent unique per-crosspoint trajectory seeds.")
    return seeds


def _counter_keyed_hash_pair(
    seeds: torch.Tensor,
    counters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two independently projected 31-bit words for every RNG key.

    A 31-bit adaptation of the MurmurHash3 integer finalizer turns the full
    signed-63-bit ``(seed, counter)`` key into two words.  Each lane uses a
    distinct coefficient-and-salt schedule across the seed and counter limbs
    before finalization; it is not a differently salted view of one collapsed
    31-bit preimage.  The operands are masked before each multiply, so
    every intermediate is below signed-int64 overflow on both CPU and CUDA.
    The resulting 31+31-bit output has a 62-bit pair space.  Even a
    deliberately loose upper bound of seven
    billion campaign draws has only about five expected pair collisions under
    the birthday model (below one part per billion of draws), while float32
    normal quantization itself already admits repeated output values.
    """

    if (
        seeds.dtype != torch.int64
        or counters.dtype != torch.int64
        or seeds.shape != counters.shape
        or seeds.ndim != 1
    ):
        raise ValueError("Expected matching rank-1 int64 RNG keys and counters.")
    mask = 0x7FFFFFFF

    def hashed(lane: int) -> torch.Tensor:
        if lane == 0:
            coefficients = (
                0x9E3779B1,
                0x85EBCA6B,
                0xC2B2AE35,
                0x27D4EB2F,
                0x165667B1,
                0x243F6A88,
            )
        elif lane == 1:
            # A second projection of every seed/counter limb is required.
            # Merely changing a final salt would leave both outputs as a
            # function of one 31-bit intermediate and would not provide a
            # 62-bit Box--Muller pair space.
            coefficients = (
                0x27D4EB2F,
                0x165667B1,
                0x9E3779B1,
                0x85EBCA6B,
                0xC2B2AE35,
                0x13198A2E,
            )
        else:  # pragma: no cover - private fixed-lane invariant
            raise ValueError("Expected one of two counter-RNG lanes.")
        seed_low = seeds & mask
        seed_mid = (seeds >> 31) & mask
        seed_high = (seeds >> 62) & mask
        counter_low = counters & mask
        counter_mid = (counters >> 31) & mask
        counter_high = (counters >> 62) & mask
        value = (
            seed_low
            ^ ((seed_mid * coefficients[0]) & mask)
            ^ ((seed_high * coefficients[1]) & mask)
            ^ ((counter_low * coefficients[2]) & mask)
            ^ ((counter_mid * coefficients[3]) & mask)
            ^ ((counter_high * coefficients[4]) & mask)
            ^ coefficients[5]
        ) & mask
        value = value ^ (value >> 16)
        value = (value * 0x85EBCA6B) & mask
        value = value ^ (value >> 13)
        value = (value * 0xC2B2AE35) & mask
        return (value ^ (value >> 16)) & mask

    return hashed(0), hashed(1)


def _counter_keyed_standard_normal(
    seeds: torch.Tensor,
    counters: torch.Tensor,
) -> torch.Tensor:
    """Return counter-keyed Box--Muller normals without future-draw buffers.

    This is a deterministic stochastic-simulator RNG, not an attempted replay
    of AIHWKit's hidden native tile RNG and not a cryptographic primitive.
    """

    radius_word, angle_word = _counter_keyed_hash_pair(seeds, counters)
    # Keep transcendental work in float32: recovery can request thousands of
    # sparse physical updates on consumer CUDA devices with very weak float64
    # throughput.  Half-bin centering and a one-ULP upper clamp retain valid
    # Box--Muller inputs after the int-to-float32 conversion rounds its edge.
    reciprocal = 1.0 / float(2**31)
    upper = 1.0 - torch.finfo(torch.float32).eps
    uniform_radius = (
        (radius_word.to(torch.float32) + 0.5) * reciprocal
    ).clamp(max=upper)
    uniform_angle = (
        (angle_word.to(torch.float32) + 0.5) * reciprocal
    ).clamp(max=upper)
    normal = torch.sqrt(-2.0 * torch.log(uniform_radius)) * torch.cos(
        (2.0 * math.pi) * uniform_angle
    )
    return normal


class IbmOmEffectiveCrossbarPlant:
    """Persistent ``q=a-r`` OM plant with independent cell RNG streams."""

    STATE_SCHEMA = "ebl.ibm_om_effective_crossbar_plant"

    def __init__(
        self,
        population: IbmReramArrayPopulation,
        *,
        trajectory_seeds: Sequence[int] | torch.Tensor | None = None,
        trajectory_seed_derivation: str | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.population = population.to(self.device)
        # ``torch.device("cuda")`` is not equal to the concrete
        # ``torch.device("cuda:0")`` reported by tensors allocated through it.
        # Keep the plant's device contract canonical so capability ports accept
        # CUDA-resident minibatches produced with an unindexed CUDA config.
        self.device = self.population.reference.device
        if trajectory_seeds is not None and generator is not None:
            raise ValueError(
                "Expected either explicit trajectory seeds or the legacy "
                "generator adapter, not both."
            )
        if trajectory_seeds is None:
            if not isinstance(generator, torch.Generator):
                raise ValueError("Expected explicit per-cell trajectory seeds.")
            legacy_base = derive_seed(
                int(generator.initial_seed()),
                population.assignment_seed,
                population.fingerprint,
                "ibm_om_crossbar_legacy_generator_adapter_v1",
            )
            seeds_cpu = _contiguous_trajectory_seeds(
                base_seed=legacy_base,
                size=population.size,
            )
            derivation = _LEGACY_GENERATOR_SEED_DERIVATION
        else:
            seeds_cpu = _normalize_trajectory_seeds(
                trajectory_seeds,
                size=population.size,
            )
            derivation = (
                "caller_supplied_explicit_per_cell_v1"
                if trajectory_seed_derivation is None
                else trajectory_seed_derivation
            )
        if not isinstance(derivation, str) or not derivation.strip():
            raise ValueError("Expected a non-empty trajectory-seed derivation receipt.")
        self.trajectory_seed_derivation = derivation.strip()
        self.trajectory_seeds = seeds_cpu.to(self.device)
        self.trajectory_draw_indices = torch.zeros(
            population.size,
            dtype=torch.int64,
            device=self.device,
        )
        self.persistent = self.population.logical_min.clone()
        self.apparent = self.persistent.clone()
        if self._write_scale > 0.0:
            initial_mask = torch.ones(
                population.size,
                dtype=torch.bool,
                device=self.device,
            )
            indices, write_noise = self._normal(initial_mask)
            self.apparent[indices] = (
                self.persistent[indices] + self._write_scale * write_noise
            )
        self.upward_pulses = torch.zeros(
            self.population.size, dtype=torch.int64, device=self.device
        )
        self.downward_pulses = torch.zeros_like(self.upward_pulses)
        self.post_deployment_fault_mask = torch.zeros(
            self.population.size, dtype=torch.bool, device=self.device
        )
        self.post_deployment_stuck_persistent_q = torch.zeros(
            self.population.size, dtype=torch.float32, device=self.device
        )
        self.post_deployment_fault_pulse_baseline = torch.zeros(
            self.population.size, dtype=torch.int64, device=self.device
        )
        self.fault_transition: dict[str, Any] | None = None

    @property
    def size(self) -> int:
        return self.population.size

    @property
    def state_kind(self) -> str:
        """Return the public recovery-state class without exposing a fault map."""

        return "faulted" if self.fault_transition is not None else "healthy"

    @property
    def _write_scale(self) -> float:
        return float(self.population.write_noise_std * self.population.nominal_dw_min)

    def _normal(self, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw once for selected trajectories and advance no other stream."""

        mask = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        if mask.shape != (self.size,):
            raise ValueError("Expected one stochastic-selection flag per crosspoint.")
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            return indices, torch.empty(
                (0,), dtype=torch.float32, device=self.device
            )
        counters = self.trajectory_draw_indices[indices]
        if bool(torch.any(counters == torch.iinfo(torch.int64).max)):
            raise RuntimeError("A per-cell random-draw counter is exhausted.")
        values = _counter_keyed_standard_normal(
            self.trajectory_seeds[indices],
            counters,
        )
        self.trajectory_draw_indices[indices] += 1
        return indices, values

    def pulse(self, directions: torch.Tensor) -> None:
        direction = torch.as_tensor(directions, dtype=torch.int8, device=self.device)
        if direction.shape != (self.size,) or bool(
            torch.any((direction < -1) | (direction > 1))
        ):
            raise ValueError("Expected one {-1,0,1} pulse direction per crosspoint.")
        active = direction != 0
        if not bool(torch.any(active)):
            return
        movable = active & ~self.post_deployment_fault_mask
        population = self.population
        active_state = self.persistent + population.reference
        cycle_indices, cycle_values = self._normal(active)
        cycle = torch.zeros_like(self.persistent)
        cycle[cycle_indices] = cycle_values
        candidate = active_state.clone()
        upward = direction > 0
        downward = direction < 0
        if bool(torch.any(upward)):
            normalized = torch.where(
                population.max_bound > 0.0,
                active_state / population.max_bound,
                torch.zeros_like(active_state),
            )
            response = population.dwmin_up * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[upward] = active_state[upward] + response[upward]
        if bool(torch.any(downward)):
            normalized = torch.where(
                population.min_bound < 0.0,
                active_state / population.min_bound,
                torch.zeros_like(active_state),
            )
            response = population.dwmin_down * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[downward] = active_state[downward] - response[downward]
        candidate = torch.maximum(candidate, population.min_bound)
        candidate = torch.minimum(candidate, population.max_bound)
        self.persistent[movable] = (candidate - population.reference)[movable]
        # AIHWKit corrupt cells have zero persistent increments, but write
        # noise still changes the apparent state whenever a write is attempted.
        if self._write_scale > 0.0:
            write_indices, write_values = self._normal(active)
            self.apparent[write_indices] = (
                self.persistent[write_indices]
                + self._write_scale * write_values
            )
        else:
            self.apparent[active] = self.persistent[active]
        self.upward_pulses += upward.to(torch.int64)
        self.downward_pulses += downward.to(torch.int64)

    def apply_grouped_pulse_counts(
        self,
        positive_counts: torch.Tensor,
        negative_counts: torch.Tensor,
    ) -> None:
        """Apply counted coincidence pulses with an explicit grouped ordering.

        Stochastic-compressed bit lines can command both update directions for
        one cell over a minibatch.  The caller retains the exact positive and
        negative coincidence counts; this plant applies all positive pulse
        rounds first and all negative pulse rounds second.  The grouping is a
        declared distribution-level approximation to a native tile's hidden
        bit-line ordering, not a claim of bitwise AIHWKit replay.
        """

        positive = torch.as_tensor(
            positive_counts, dtype=torch.int64, device=self.device
        )
        negative = torch.as_tensor(
            negative_counts, dtype=torch.int64, device=self.device
        )
        if (
            positive.shape != (self.size,)
            or negative.shape != (self.size,)
            or bool(torch.any(positive < 0))
            or bool(torch.any(negative < 0))
        ):
            raise ValueError(
                "Expected non-negative positive/negative pulse counts per crosspoint."
            )
        positive_maximum = int(positive.max().item()) if positive.numel() else 0
        negative_maximum = int(negative.max().item()) if negative.numel() else 0
        for pulse_index in range(positive_maximum):
            self.pulse((positive > pulse_index).to(torch.int8))
        for pulse_index in range(negative_maximum):
            self.pulse(-(negative > pulse_index).to(torch.int8))

    def apply_stuck_at_fault_transition(
        self,
        *,
        mask: torch.Tensor,
        stuck_persistent_q: torch.Tensor,
        transition_id: str,
        source_population_fingerprint: str,
    ) -> dict[str, Any]:
        """Apply one immutable post-deployment stuck-state transition.

        The mask is transition-authority data: it is stored inside the plant
        for pulse blocking and reporting, but is never required by the STAR
        gradient or pulse-writer interfaces.
        """

        fault_mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        values = torch.as_tensor(
            stuck_persistent_q,
            dtype=torch.float32,
            device=self.device,
        )
        if self.fault_transition is not None:
            raise RuntimeError("Expected exactly one post-deployment fault transition.")
        if (
            not isinstance(transition_id, str)
            or not transition_id.strip()
            or not isinstance(source_population_fingerprint, str)
            or not source_population_fingerprint.strip()
            or fault_mask.shape != (self.size,)
            or values.shape != (self.size,)
            or not bool(torch.any(fault_mask))
            or not bool(torch.all(torch.isfinite(values[fault_mask])))
        ):
            raise ValueError("Expected a non-empty finite stuck-at transition.")
        pre_persistent_state = self.persistent.detach().clone()
        pre_apparent_state = self.apparent.detach().clone()
        pre_persistent = tensor_sha256(pre_persistent_state)
        pre_apparent = tensor_sha256(pre_apparent_state)
        self.post_deployment_fault_mask.copy_(fault_mask)
        self.post_deployment_stuck_persistent_q[fault_mask] = values[fault_mask]
        self.post_deployment_fault_pulse_baseline.copy_(
            self.upward_pulses + self.downward_pulses
        )
        self.persistent[fault_mask] = values[fault_mask]
        if self._write_scale > 0.0:
            fault_indices, transition_noise = self._normal(fault_mask)
            self.apparent[fault_indices] = (
                self.persistent[fault_indices]
                + self._write_scale * transition_noise
            )
        else:
            self.apparent[fault_mask] = self.persistent[fault_mask]
        non_fault_persistent_unchanged = bool(
            torch.equal(
                self.persistent[~fault_mask],
                pre_persistent_state[~fault_mask],
            )
        )
        non_fault_apparent_unchanged = bool(
            torch.equal(
                self.apparent[~fault_mask],
                pre_apparent_state[~fault_mask],
            )
        )
        if not non_fault_persistent_unchanged or not non_fault_apparent_unchanged:
            raise RuntimeError(
                "Expected the post-deployment fault transition to leave every "
                "non-fault cell bit-exact."
            )
        transition = {
            "schema": "ebl.ibm_om_crossbar_post_deployment_fault_transition",
            "schema_version": 1,
            "transition_id": transition_id.strip(),
            "policy": "aihwkit_sampled_corrupt_device_overlay",
            "source_population_fingerprint": source_population_fingerprint,
            "healthy_population_fingerprint": self.population.fingerprint,
            "pre_fault_persistent_sha256": pre_persistent,
            "pre_fault_apparent_sha256": pre_apparent,
            "fault_mask_sha256": tensor_sha256(fault_mask),
            "stuck_persistent_q_sha256": tensor_sha256(values[fault_mask]),
            "persistent_fault_state": "sampled_collapsed_bound_with_zero_pulse_increments",
            "apparent_fault_state": "write_noise_resampled_at_transition_and_each_write_attempt",
            "faulted_cells": int(fault_mask.sum().item()),
            "cells": self.size,
            "fault_fraction": float(fault_mask.float().mean().item()),
            "post_fault_persistent_sha256": tensor_sha256(self.persistent),
            "post_fault_apparent_sha256": tensor_sha256(self.apparent),
            "non_fault_persistent_unchanged": non_fault_persistent_unchanged,
            "non_fault_apparent_unchanged": non_fault_apparent_unchanged,
            "fault_mask_exposed_to_local_update_kernel": False,
        }
        self.fault_transition = transition
        return dict(transition)

    def controller_port(self) -> "_EffectiveControllerPort":
        return _EffectiveControllerPort(self)

    def restricted_recovery_update_port(self) -> "_LocalStarUpdatePort":
        """Expose apparent state and pulse writes for healthy or faulted recovery.

        The public state-kind label is retained for provenance, while the
        fault mask, persistent tensor, population, and RNG remain hidden.
        """

        return _LocalStarUpdatePort(self)

    def local_star_update_port(self) -> "_LocalStarUpdatePort":
        """Backward-compatible name for the restricted recovery update port."""

        return self.restricted_recovery_update_port()

    def on_chip_recovery_port(
        self,
        *,
        layout: Sequence[CrossbarTileSpec],
        digital_scales: Sequence[float],
    ) -> "_OnChipCrossbarRecoveryPort":
        """Open a forward/BP/pulse capability on a healthy or faulted plant.

        The returned learner view never exposes the full apparent or
        persistent tensors, the population, RNG, or fault metadata.  Forward
        MVMs and the one required transpose MVM remain capabilities of the
        physical plant rather than weight-tensor handoffs.
        """

        validate_population_layout(self.population, layout)
        return _OnChipCrossbarRecoveryPort(
            self,
            layout=layout,
            digital_scales=digital_scales,
        )

    def tiki_taka_fast_port(
        self,
        *,
        layout: Sequence[CrossbarTileSpec],
    ) -> "_TikiTakaFastArrayPort":
        """Open a restricted auxiliary-array pulse/read capability.

        Tiki-Taka transfer may read only the selected physical input column.
        It cannot obtain a full apparent fast matrix or any hidden plant
        authority.
        """

        validate_population_layout(self.population, layout)
        return _TikiTakaFastArrayPort(self, layout=layout)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.STATE_SCHEMA,
            "schema_version": 3,
            "population_fingerprint": self.population.fingerprint,
            "persistent": self.persistent.detach().cpu().clone(),
            "apparent": self.apparent.detach().cpu().clone(),
            "upward_pulses": self.upward_pulses.detach().cpu().clone(),
            "downward_pulses": self.downward_pulses.detach().cpu().clone(),
            "post_deployment_fault_mask": self.post_deployment_fault_mask.detach().cpu().clone(),
            "post_deployment_stuck_persistent_q": (
                self.post_deployment_stuck_persistent_q.detach().cpu().clone()
            ),
            "post_deployment_fault_pulse_baseline": (
                self.post_deployment_fault_pulse_baseline.detach().cpu().clone()
            ),
            "fault_transition": (
                None if self.fault_transition is None else dict(self.fault_transition)
            ),
            "rng_backend": CROSSBAR_TRAJECTORY_RNG_BACKEND,
            "rng_reproducibility_scope": (
                CROSSBAR_TRAJECTORY_REPRODUCIBILITY_SCOPE
            ),
            "rng_statistical_contract": CROSSBAR_TRAJECTORY_STATISTICAL_CONTRACT,
            "trajectory_seed_derivation": self.trajectory_seed_derivation,
            "trajectory_seeds": self.trajectory_seeds.detach().cpu().clone(),
            "trajectory_draw_indices": (
                self.trajectory_draw_indices.detach().cpu().clone()
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        expected_fault_source_population: IbmReramArrayPopulation | None = None,
        expected_pre_fault_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Restore one plant state, requiring authority for faulted replays.

        Healthy checkpoints remain self-contained.  A checkpoint carrying a
        post-deployment fault overlay is different: its stuck values and
        chronology are meaningful only relative to the published AIHWKit
        companion population and the exact healthy P0 state that preceded the
        transition.  Requiring both objects here prevents a coherently edited
        checkpoint from authenticating its own fault values or historical
        hashes.
        """
        expected_legacy = {
            "schema",
            "schema_version",
            "population_fingerprint",
            "persistent",
            "apparent",
            "upward_pulses",
            "downward_pulses",
            "generator_state",
        }
        expected_legacy_fault = expected_legacy | {
            "post_deployment_fault_mask",
            "post_deployment_stuck_persistent_q",
            "post_deployment_fault_pulse_baseline",
            "fault_transition",
        }
        expected_v3 = (expected_legacy_fault - {"generator_state"}) | {
            "rng_backend",
            "rng_reproducibility_scope",
            "rng_statistical_contract",
            "trajectory_seed_derivation",
            "trajectory_seeds",
            "trajectory_draw_indices",
        }
        if (
            not isinstance(state, Mapping)
            or "schema_version" not in state
        ):
            raise ValueError("Expected a matching effective-crossbar plant state.")
        schema_version = state.get("schema_version")
        if (
            set(state) != expected_v3
            or state["schema"] != self.STATE_SCHEMA
            or isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 3
            or state["population_fingerprint"] != self.population.fingerprint
        ):
            if schema_version in {1, 2} and frozenset(state) in {
                frozenset(expected_legacy),
                frozenset(expected_legacy_fault),
            }:
                raise ValueError(
                    "Legacy single-generator crossbar states cannot be exactly "
                    "continued with the required per-trajectory RNG backend."
                )
            raise ValueError("Expected a matching effective-crossbar plant state version 3.")
        pending: dict[str, torch.Tensor] = {}
        for name, target, dtype in (
            ("persistent", self.persistent, torch.float32),
            ("apparent", self.apparent, torch.float32),
            ("upward_pulses", self.upward_pulses, torch.int64),
            ("downward_pulses", self.downward_pulses, torch.int64),
        ):
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != dtype:
                raise ValueError(f"Expected saved crossbar field {name!r} to match.")
            pending[name] = value.detach().to(self.device).clone()
        if (
            not bool(torch.all(torch.isfinite(pending["persistent"])))
            or not bool(torch.all(torch.isfinite(pending["apparent"])))
            or bool(torch.any(pending["upward_pulses"] < 0))
            or bool(torch.any(pending["downward_pulses"] < 0))
        ):
            raise ValueError("Expected finite states and non-negative pulse counters.")

        pending_fault_mask = torch.zeros_like(self.post_deployment_fault_mask)
        pending_fault_q = torch.zeros_like(
            self.post_deployment_stuck_persistent_q
        )
        pending_fault_baseline = torch.zeros_like(
            self.post_deployment_fault_pulse_baseline
        )
        pending_transition: dict[str, Any] | None = None
        if schema_version == 3:
            fault_mask = state["post_deployment_fault_mask"]
            fault_q = state["post_deployment_stuck_persistent_q"]
            fault_pulse_baseline = state["post_deployment_fault_pulse_baseline"]
            transition = state["fault_transition"]
            if (
                not isinstance(fault_mask, torch.Tensor)
                or fault_mask.shape != self.post_deployment_fault_mask.shape
                or fault_mask.dtype != torch.bool
                or not isinstance(fault_q, torch.Tensor)
                or fault_q.shape != self.post_deployment_stuck_persistent_q.shape
                or fault_q.dtype != torch.float32
                or not isinstance(fault_pulse_baseline, torch.Tensor)
                or fault_pulse_baseline.shape
                != self.post_deployment_fault_pulse_baseline.shape
                or fault_pulse_baseline.dtype != torch.int64
                or (transition is not None and not isinstance(transition, Mapping))
                or (transition is None and bool(torch.any(fault_mask)))
                or (transition is not None and not bool(torch.any(fault_mask)))
            ):
                raise ValueError("Expected a valid saved stuck-at fault overlay.")
            pending_fault_mask = fault_mask.detach().to(self.device).clone()
            pending_fault_q = fault_q.detach().to(self.device).clone()
            pending_fault_baseline = (
                fault_pulse_baseline.detach().to(self.device).clone()
            )
            total_pulses = pending["upward_pulses"] + pending["downward_pulses"]
            if (
                not bool(torch.all(torch.isfinite(pending_fault_q)))
                or bool(torch.any(pending_fault_baseline < 0))
                or bool(torch.any(pending_fault_baseline > total_pulses))
            ):
                raise ValueError(
                    "Expected finite fault states and an ordered non-negative pulse baseline."
                )
            if transition is None:
                if (
                    expected_fault_source_population is not None
                    or expected_pre_fault_state is not None
                ):
                    raise ValueError(
                        "Expected fault replay bindings only for a saved fault transition."
                    )
                if (
                    bool(torch.any(pending_fault_q != 0.0))
                    or bool(torch.any(pending_fault_baseline != 0))
                ):
                    raise ValueError("Expected an empty saved fault overlay without a transition.")
            else:
                if not isinstance(
                    expected_fault_source_population,
                    IbmReramArrayPopulation,
                ) or not isinstance(expected_pre_fault_state, Mapping):
                    raise ValueError(
                        "Expected the published companion population and exact "
                        "healthy P0 state for faulted checkpoint replay."
                    )
                pending_transition = dict(transition)
                transition_fields = {
                    "schema",
                    "schema_version",
                    "transition_id",
                    "policy",
                    "source_population_fingerprint",
                    "healthy_population_fingerprint",
                    "pre_fault_persistent_sha256",
                    "pre_fault_apparent_sha256",
                    "fault_mask_sha256",
                    "stuck_persistent_q_sha256",
                    "persistent_fault_state",
                    "apparent_fault_state",
                    "faulted_cells",
                    "cells",
                    "fault_fraction",
                    "post_fault_persistent_sha256",
                    "post_fault_apparent_sha256",
                    "non_fault_persistent_unchanged",
                    "non_fault_apparent_unchanged",
                    "fault_mask_exposed_to_local_update_kernel",
                }

                def valid_sha256(value: Any) -> bool:
                    return (
                        isinstance(value, str)
                        and len(value) == 64
                        and all(character in "0123456789abcdef" for character in value)
                    )

                if set(pending_transition) != transition_fields:
                    raise ValueError(
                        "Expected exact saved fault-transition receipt fields."
                    )
                expected_fraction = float(
                    pending_fault_mask.to(torch.float32).mean().item()
                )
                reported_fraction = pending_transition["fault_fraction"]
                transition_schema_version = pending_transition["schema_version"]
                if (
                    pending_transition["schema"]
                    != "ebl.ibm_om_crossbar_post_deployment_fault_transition"
                    or isinstance(transition_schema_version, bool)
                    or not isinstance(transition_schema_version, int)
                    or transition_schema_version != 1
                    or pending_transition["policy"]
                    != "aihwkit_sampled_corrupt_device_overlay"
                    or pending_transition["persistent_fault_state"]
                    != "sampled_collapsed_bound_with_zero_pulse_increments"
                    or pending_transition["apparent_fault_state"]
                    != "write_noise_resampled_at_transition_and_each_write_attempt"
                    or not isinstance(pending_transition["transition_id"], str)
                    or not pending_transition["transition_id"].strip()
                    or not isinstance(
                        pending_transition["source_population_fingerprint"], str
                    )
                    or not pending_transition["source_population_fingerprint"].strip()
                    or pending_transition["healthy_population_fingerprint"]
                    != self.population.fingerprint
                    or any(
                        not valid_sha256(pending_transition[name])
                        for name in (
                            "pre_fault_persistent_sha256",
                            "pre_fault_apparent_sha256",
                            "fault_mask_sha256",
                            "stuck_persistent_q_sha256",
                            "post_fault_persistent_sha256",
                            "post_fault_apparent_sha256",
                        )
                    )
                    or pending_transition["fault_mask_sha256"]
                    != tensor_sha256(pending_fault_mask)
                    or pending_transition["stuck_persistent_q_sha256"]
                    != tensor_sha256(pending_fault_q[pending_fault_mask])
                    or pending_transition["faulted_cells"]
                    != int(pending_fault_mask.sum().item())
                    or pending_transition["cells"] != self.size
                    or isinstance(reported_fraction, bool)
                    or not isinstance(reported_fraction, (int, float))
                    or not math.isfinite(float(reported_fraction))
                    or not math.isclose(
                        float(reported_fraction),
                        expected_fraction,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or pending_transition["non_fault_persistent_unchanged"] is not True
                    or pending_transition["non_fault_apparent_unchanged"] is not True
                    or pending_transition[
                        "fault_mask_exposed_to_local_update_kernel"
                    ] is not False
                    or bool(torch.any(pending_fault_q[~pending_fault_mask] != 0.0))
                ):
                    raise ValueError("Expected a self-consistent saved fault-transition receipt.")
                if not torch.equal(
                    pending["persistent"][pending_fault_mask],
                    pending_fault_q[pending_fault_mask],
                ):
                    raise ValueError("Expected saved persistent stuck cells to be immutable.")

                fault_source = expected_fault_source_population.to(self.device)
                source_mask = fault_source.corrupt
                if (
                    fault_source.corruption_policy != "published"
                    or fault_source.assignment_seed
                    != self.population.assignment_seed
                    or fault_source.binding_keys != self.population.binding_keys
                    or fault_source.binding_shapes != self.population.binding_shapes
                    or fault_source.binding_sampling_seeds
                    != self.population.binding_sampling_seeds
                    or fault_source.donor_sampling_seeds
                    != self.population.donor_sampling_seeds
                    or fault_source.nominal_dw_min
                    != self.population.nominal_dw_min
                    or fault_source.dw_min_std != self.population.dw_min_std
                    or fault_source.write_noise_std
                    != self.population.write_noise_std
                    or fault_source.aihwkit_version
                    != self.population.aihwkit_version
                    or pending_transition["source_population_fingerprint"]
                    != fault_source.fingerprint
                    or not torch.equal(source_mask, fault_source.published_corrupt)
                    or not torch.equal(source_mask, pending_fault_mask)
                    or not torch.equal(
                        self.population.published_corrupt,
                        pending_fault_mask,
                    )
                ):
                    raise ValueError(
                        "Expected the saved fault overlay to match its published "
                        "AIHWKit companion population exactly."
                    )
                healthy_mask = ~pending_fault_mask
                for name in (
                    "min_bound",
                    "max_bound",
                    "dwmin_up",
                    "dwmin_down",
                    "reference",
                ):
                    if not torch.equal(
                        getattr(fault_source, name)[healthy_mask],
                        getattr(self.population, name)[healthy_mask],
                    ):
                        raise ValueError(
                            "Expected every non-fault companion identity field "
                            "to match the healthy population."
                        )
                corrupt_range = float(PUBLISHED_CORRUPT_RANGE[OM_PRESET])
                collapsed_q = fault_source.logical_min
                if (
                    not torch.equal(
                        fault_source.min_bound[pending_fault_mask],
                        fault_source.max_bound[pending_fault_mask],
                    )
                    or not torch.equal(
                        collapsed_q[pending_fault_mask],
                        fault_source.logical_max[pending_fault_mask],
                    )
                    or not torch.equal(
                        fault_source.dwmin_up[pending_fault_mask],
                        torch.zeros_like(
                            fault_source.dwmin_up[pending_fault_mask]
                        ),
                    )
                    or not torch.equal(
                        fault_source.dwmin_down[pending_fault_mask],
                        torch.zeros_like(
                            fault_source.dwmin_down[pending_fault_mask]
                        ),
                    )
                    or bool(
                        torch.any(
                            torch.abs(
                                fault_source.min_bound[pending_fault_mask]
                            )
                            > corrupt_range + 1e-7
                        )
                    )
                    or not torch.equal(
                        pending_fault_q[pending_fault_mask],
                        collapsed_q[pending_fault_mask],
                    )
                ):
                    raise ValueError(
                        "Expected exact AIHWKit sampled collapsed stuck values, "
                        "zero pulse increments, and configured corrupt range."
                    )

                pre_fault = expected_pre_fault_state
                pre_fault_legacy = {
                    "schema",
                    "schema_version",
                    "population_fingerprint",
                    "persistent",
                    "apparent",
                    "upward_pulses",
                    "downward_pulses",
                    "generator_state",
                }
                pre_fault_legacy_fault = pre_fault_legacy | {
                    "post_deployment_fault_mask",
                    "post_deployment_stuck_persistent_q",
                    "post_deployment_fault_pulse_baseline",
                    "fault_transition",
                }
                pre_fault_v3 = (pre_fault_legacy_fault - {"generator_state"}) | {
                    "rng_backend",
                    "rng_reproducibility_scope",
                    "rng_statistical_contract",
                    "trajectory_seed_derivation",
                    "trajectory_seeds",
                    "trajectory_draw_indices",
                }
                pre_schema_version = pre_fault.get("schema_version")
                pre_persistent = pre_fault.get("persistent")
                pre_apparent = pre_fault.get("apparent")
                pre_upward = pre_fault.get("upward_pulses")
                pre_downward = pre_fault.get("downward_pulses")
                pre_rng_backend = pre_fault.get("rng_backend")
                pre_reproducibility = pre_fault.get("rng_reproducibility_scope")
                pre_statistical_contract = pre_fault.get("rng_statistical_contract")
                pre_seed_derivation = pre_fault.get("trajectory_seed_derivation")
                pre_trajectory_seeds = pre_fault.get("trajectory_seeds")
                pre_draw_indices = pre_fault.get("trajectory_draw_indices")
                if (
                    set(pre_fault) != pre_fault_v3
                    or pre_fault.get("schema") != self.STATE_SCHEMA
                    or isinstance(pre_schema_version, bool)
                    or not isinstance(pre_schema_version, int)
                    or pre_schema_version != 3
                    or pre_fault.get("population_fingerprint")
                    != self.population.fingerprint
                    or not isinstance(pre_persistent, torch.Tensor)
                    or pre_persistent.shape != self.persistent.shape
                    or pre_persistent.dtype != torch.float32
                    or not bool(torch.all(torch.isfinite(pre_persistent)))
                    or not isinstance(pre_apparent, torch.Tensor)
                    or pre_apparent.shape != self.apparent.shape
                    or pre_apparent.dtype != torch.float32
                    or not bool(torch.all(torch.isfinite(pre_apparent)))
                    or not isinstance(pre_upward, torch.Tensor)
                    or pre_upward.shape != self.upward_pulses.shape
                    or pre_upward.dtype != torch.int64
                    or bool(torch.any(pre_upward < 0))
                    or not isinstance(pre_downward, torch.Tensor)
                    or pre_downward.shape != self.downward_pulses.shape
                    or pre_downward.dtype != torch.int64
                    or bool(torch.any(pre_downward < 0))
                    or pre_rng_backend != CROSSBAR_TRAJECTORY_RNG_BACKEND
                    or pre_reproducibility
                    != CROSSBAR_TRAJECTORY_REPRODUCIBILITY_SCOPE
                    or pre_statistical_contract
                    != CROSSBAR_TRAJECTORY_STATISTICAL_CONTRACT
                    or not isinstance(pre_seed_derivation, str)
                    or not pre_seed_derivation.strip()
                    or not isinstance(pre_trajectory_seeds, torch.Tensor)
                    or pre_trajectory_seeds.device.type != "cpu"
                    or pre_trajectory_seeds.dtype != torch.int64
                    or pre_trajectory_seeds.shape != (self.size,)
                    or not isinstance(pre_draw_indices, torch.Tensor)
                    or pre_draw_indices.device.type != "cpu"
                    or pre_draw_indices.dtype != torch.int64
                    or pre_draw_indices.shape != (self.size,)
                    or bool(torch.any(pre_draw_indices < 0))
                    or not isinstance(state.get("trajectory_seeds"), torch.Tensor)
                    or not torch.equal(
                        pre_trajectory_seeds,
                        state["trajectory_seeds"],
                    )
                    or pre_seed_derivation
                    != state.get("trajectory_seed_derivation")
                ):
                    raise ValueError(
                        "Expected a complete healthy P0 state for fault chronology."
                    )
                if pre_schema_version == 3:
                    pre_mask = pre_fault["post_deployment_fault_mask"]
                    pre_stuck = pre_fault[
                        "post_deployment_stuck_persistent_q"
                    ]
                    pre_baseline = pre_fault[
                        "post_deployment_fault_pulse_baseline"
                    ]
                    if (
                        not isinstance(pre_mask, torch.Tensor)
                        or pre_mask.shape != self.post_deployment_fault_mask.shape
                        or pre_mask.dtype != torch.bool
                        or not isinstance(pre_stuck, torch.Tensor)
                        or pre_stuck.shape
                        != self.post_deployment_stuck_persistent_q.shape
                        or pre_stuck.dtype != torch.float32
                        or not bool(torch.all(torch.isfinite(pre_stuck)))
                        or not isinstance(pre_baseline, torch.Tensor)
                        or pre_baseline.shape
                        != self.post_deployment_fault_pulse_baseline.shape
                        or pre_baseline.dtype != torch.int64
                        or pre_fault["fault_transition"] is not None
                        or bool(torch.any(pre_mask))
                        or bool(torch.any(pre_stuck != 0.0))
                        or bool(torch.any(pre_baseline != 0))
                    ):
                        raise ValueError(
                            "Expected the replay P0 state to precede every "
                            "fault transition."
                        )
                pre_upward_device = pre_upward.detach().to(self.device)
                pre_downward_device = pre_downward.detach().to(self.device)
                if (
                    pending_transition["pre_fault_persistent_sha256"]
                    != tensor_sha256(pre_persistent)
                    or pending_transition["pre_fault_apparent_sha256"]
                    != tensor_sha256(pre_apparent)
                    or not torch.equal(
                        pending_fault_baseline,
                        pre_upward_device + pre_downward_device,
                    )
                    or bool(
                        torch.any(
                            pending["upward_pulses"] < pre_upward_device
                        )
                    )
                    or bool(
                        torch.any(
                            pending["downward_pulses"] < pre_downward_device
                        )
                    )
                ):
                    raise ValueError(
                        "Expected saved fault chronology to match the exact healthy P0 hashes."
                    )
                if torch.equal(total_pulses, pending_fault_baseline) and (
                    pending_transition["post_fault_persistent_sha256"]
                    != tensor_sha256(pending["persistent"])
                    or pending_transition["post_fault_apparent_sha256"]
                    != tensor_sha256(pending["apparent"])
                ):
                    raise ValueError(
                        "Expected the immediate post-fault state hashes to match."
                    )
                if torch.equal(total_pulses, pending_fault_baseline) and (
                    not torch.equal(
                        pending["persistent"][healthy_mask],
                        pre_persistent.detach().to(self.device)[healthy_mask],
                    )
                    or not torch.equal(
                        pending["apparent"][healthy_mask],
                        pre_apparent.detach().to(self.device)[healthy_mask],
                    )
                ):
                    raise ValueError(
                        "Expected the immediate fault transition to leave P0 "
                        "non-fault states bit-exact."
                    )
        healthy = ~pending_fault_mask
        if bool(
            torch.any(
                pending["persistent"][healthy]
                < self.population.logical_min[healthy]
            )
        ) or bool(
            torch.any(
                pending["persistent"][healthy]
                > self.population.logical_max[healthy]
            )
        ):
            raise ValueError("Expected saved programmable states to remain in support.")
        rng_backend = state["rng_backend"]
        reproducibility_scope = state["rng_reproducibility_scope"]
        statistical_contract = state["rng_statistical_contract"]
        seed_derivation = state["trajectory_seed_derivation"]
        trajectory_seeds = state["trajectory_seeds"]
        draw_indices = state["trajectory_draw_indices"]
        if (
            rng_backend != CROSSBAR_TRAJECTORY_RNG_BACKEND
            or reproducibility_scope
            != CROSSBAR_TRAJECTORY_REPRODUCIBILITY_SCOPE
            or statistical_contract != CROSSBAR_TRAJECTORY_STATISTICAL_CONTRACT
            or not isinstance(seed_derivation, str)
            or not seed_derivation.strip()
            or not isinstance(trajectory_seeds, torch.Tensor)
            or trajectory_seeds.device.type != "cpu"
            or trajectory_seeds.dtype != torch.int64
            or trajectory_seeds.shape != (self.size,)
            or not isinstance(draw_indices, torch.Tensor)
            or draw_indices.device.type != "cpu"
            or draw_indices.dtype != torch.int64
            or draw_indices.shape != (self.size,)
            or bool(torch.any(draw_indices < 0))
        ):
            raise ValueError(
                "Expected exact saved per-trajectory seeds and draw counters."
            )
        normalized_seeds = _normalize_trajectory_seeds(
            trajectory_seeds,
            size=self.size,
        )
        total_pulses = pending["upward_pulses"] + pending["downward_pulses"]
        noisy_apparent = int(self._write_scale > 0.0)
        expected_draw_indices = (
            torch.full_like(total_pulses, noisy_apparent)
            + total_pulses * (1 + noisy_apparent)
            + pending_fault_mask.to(torch.int64) * noisy_apparent
        )
        if not torch.equal(
            draw_indices,
            expected_draw_indices.detach().cpu(),
        ):
            raise ValueError(
                "Expected per-cell draw counters to match conditioning, pulse, "
                "write-noise, and fault-transition chronology exactly."
            )

        # The held apparent state is not an independently mutable checkpoint
        # field.  Every cell receives one apparent write-noise draw at
        # conditioning, and thereafter its final draw is the write-noise draw
        # associated with its most recent pulse (or fault transition).  The
        # stateless per-trajectory RNG therefore lets us authenticate that
        # relation exactly when restoring on the same execution backend.  A
        # cross-backend restore is intentionally outside the bit-exact scope
        # recorded by ``rng_reproducibility_scope``.
        if self._write_scale > 0.0:
            saved_counters = draw_indices.to(self.device)
            if bool(torch.any(saved_counters < 1)):
                raise ValueError(
                    "Expected every noisy apparent state to have a conditioning draw."
                )
            last_write_noise = _counter_keyed_standard_normal(
                normalized_seeds.to(self.device),
                saved_counters - 1,
            )
            expected_apparent = (
                pending["persistent"] + self._write_scale * last_write_noise
            )
        else:
            expected_apparent = pending["persistent"]
        if not torch.equal(pending["apparent"], expected_apparent):
            raise ValueError(
                "Expected held apparent state to match the deterministic last "
                "write-noise draw on this execution backend."
            )

        self.persistent.copy_(pending["persistent"])
        self.apparent.copy_(pending["apparent"])
        self.upward_pulses.copy_(pending["upward_pulses"])
        self.downward_pulses.copy_(pending["downward_pulses"])
        self.post_deployment_fault_mask.copy_(pending_fault_mask)
        self.post_deployment_stuck_persistent_q.copy_(pending_fault_q)
        self.post_deployment_fault_pulse_baseline.copy_(pending_fault_baseline)
        self.fault_transition = pending_transition
        self.trajectory_seed_derivation = seed_derivation.strip()
        self.trajectory_seeds.copy_(normalized_seeds.to(self.device))
        self.trajectory_draw_indices.copy_(draw_indices.to(self.device))

    def pulse_statistics(self) -> dict[str, Any]:
        total = self.upward_pulses + self.downward_pulses
        active = self.persistent + self.population.reference
        commanded = total > 0
        corrupt = self.population.corrupt
        post_fault = self.post_deployment_fault_mask
        post_transition_pulses = total - self.post_deployment_fault_pulse_baseline
        diagnostic_minimum = self.population.min_bound.clone()
        diagnostic_maximum = self.population.max_bound.clone()
        if bool(torch.any(post_fault)):
            collapsed_active = (
                self.post_deployment_stuck_persistent_q
                + self.population.reference
            )
            diagnostic_minimum[post_fault] = collapsed_active[post_fault]
            diagnostic_maximum[post_fault] = collapsed_active[post_fault]
        return {
            "total": int(total.sum().item()),
            "upward": int(self.upward_pulses.sum().item()),
            "downward": int(self.downward_pulses.sum().item()),
            # Retain the historical key, but make its command semantics explicit.
            "changed_cells": int(commanded.sum().item()),
            "commanded_cells": int(commanded.sum().item()),
            "persistent_moved_from_reset_cells": int(
                (~torch.isclose(
                    active,
                    self.population.min_bound,
                    atol=1e-7,
                    rtol=0.0,
                )).sum().item()
            ),
            "final_corrupt_cells": int(corrupt.sum().item()),
            "published_corrupt_cells": int(
                self.population.published_corrupt.sum().item()
            ),
            "commanded_final_corrupt_cells": int((commanded & corrupt).sum().item()),
            "pulses_to_final_corrupt_cells": int(total[corrupt].sum().item()),
            "post_deployment_fault_cells": int(post_fault.sum().item()),
            "commanded_post_deployment_fault_cells": int(
                ((post_transition_pulses > 0) & post_fault).sum().item()
            ),
            "pulses_to_post_deployment_fault_cells": int(
                post_transition_pulses[post_fault].sum().item()
            ),
            "maximum_per_cell": int(total.max().item()) if total.numel() else 0,
            "saturated_lower": int(
                torch.isclose(active, diagnostic_minimum, atol=1e-7, rtol=0.0).sum().item()
            ),
            "saturated_upper": int(
                torch.isclose(active, diagnostic_maximum, atol=1e-7, rtol=0.0).sum().item()
            ),
            "post_deployment_fault_bound_diagnostics": (
                "sampled_collapsed_active_bound_overlay"
            ),
        }


def crossbar_state_bundle(
    plant: IbmOmEffectiveCrossbarPlant,
    *,
    layout: Sequence[CrossbarTileSpec],
    digital_scales: Sequence[float],
    metadata: Mapping[str, Any] | None = None,
) -> IbmOmCrossbarStateBundle:
    """Freeze an exact healthy or faulted deployment/recovery boundary."""

    if not isinstance(plant, IbmOmEffectiveCrossbarPlant):
        raise TypeError("Expected an IBM OM effective-crossbar plant.")
    tiles = _normalize_crossbar_layout(layout)
    validate_population_layout(plant.population, tiles)
    scales = _normalize_digital_scales(digital_scales)
    canonical_metadata, metadata_sha256 = _canonical_metadata(
        {} if metadata is None else metadata
    )
    plant_state = plant.state_dict()
    return IbmOmCrossbarStateBundle(
        layout=tiles,
        digital_scales=scales,
        state_kind=plant.state_kind,
        population_fingerprint=plant.population.fingerprint,
        plant_state=plant_state,
        plant_state_sha256=_structured_sha256(plant_state),
        metadata=canonical_metadata,
        metadata_sha256=metadata_sha256,
    )


def restore_crossbar_state_bundle(
    bundle: IbmOmCrossbarStateBundle | Mapping[str, Any],
    *,
    population: IbmReramArrayPopulation,
    trajectory_seeds: Sequence[int] | torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str,
    expected_layout: Sequence[CrossbarTileSpec],
    expected_digital_scales: Sequence[float],
    expected_metadata_sha256: str | None = None,
    expected_fault_source_population: IbmReramArrayPopulation | None = None,
    expected_pre_fault_state: Mapping[str, Any] | None = None,
) -> tuple[IbmOmEffectiveCrossbarPlant, IbmOmCrossbarStateBundle]:
    """Restore a bundle against external population and configuration authority.

    A faulted bundle additionally requires the published companion population
    and exact healthy P0 plant state accepted by ``load_state_dict``.  The
    returned bundle is a newly validated copy; callers can use its canonical
    metadata, layout, scales, and digests in stage receipts.
    """

    parsed = IbmOmCrossbarStateBundle.from_state_dict(
        bundle.state_dict()
        if isinstance(bundle, IbmOmCrossbarStateBundle)
        else bundle
    )
    tiles = _normalize_crossbar_layout(expected_layout)
    scales = _normalize_digital_scales(expected_digital_scales)
    validate_population_layout(population, tiles)
    if (
        parsed.layout != tiles
        or parsed.digital_scales != scales
        or parsed.population_fingerprint != population.fingerprint
    ):
        raise ValueError(
            "Expected bundle layout, scales, and physical population to match "
            "the external recovery configuration."
        )
    if expected_metadata_sha256 is not None and (
        not _is_sha256(expected_metadata_sha256)
        or parsed.metadata_sha256 != expected_metadata_sha256
    ):
        raise ValueError("Expected the crossbar bundle provenance digest to match.")
    if parsed.state_kind == "healthy" and (
        expected_fault_source_population is not None
        or expected_pre_fault_state is not None
    ):
        raise ValueError("Expected fault replay authority only for a faulted bundle.")
    if parsed.state_kind == "faulted" and (
        not isinstance(expected_fault_source_population, IbmReramArrayPopulation)
        or not isinstance(expected_pre_fault_state, Mapping)
    ):
        raise ValueError(
            "Expected a published companion population and exact healthy P0 "
            "state for a faulted bundle."
        )
    saved_seeds = parsed.plant_state.get("trajectory_seeds")
    saved_derivation = parsed.plant_state.get("trajectory_seed_derivation")
    if not isinstance(saved_seeds, torch.Tensor):
        raise ValueError(
            "Expected a per-trajectory RNG state; legacy generator checkpoints "
            "cannot be exactly restored."
        )
    normalized_saved_seeds = _normalize_trajectory_seeds(
        saved_seeds,
        size=population.size,
    )
    if trajectory_seeds is not None and not torch.equal(
        _normalize_trajectory_seeds(trajectory_seeds, size=population.size),
        normalized_saved_seeds,
    ):
        raise ValueError(
            "Expected caller-supplied trajectory seeds to match the saved "
            "physical continuation state exactly."
        )
    # ``generator`` remains an accepted keyword for v1 call-site source
    # compatibility only.  A v3 bundle is its own RNG authority; silently
    # deriving replacement streams from the placeholder would break replay.
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("Expected generator to be a Torch generator when provided.")
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        trajectory_seeds=normalized_saved_seeds,
        trajectory_seed_derivation=saved_derivation,
        device=device,
    )
    plant.load_state_dict(
        parsed.plant_state,
        expected_fault_source_population=expected_fault_source_population,
        expected_pre_fault_state=expected_pre_fault_state,
    )
    if plant.state_kind != parsed.state_kind:
        raise RuntimeError("Expected the restored plant state kind to match its bundle.")
    return plant, parsed


class _EffectiveControllerPort:
    """P&V port exposing only apparent normalized effective state."""

    def __init__(self, plant: IbmOmEffectiveCrossbarPlant) -> None:
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
        direction = torch.as_tensor(directions, dtype=torch.int8, device=self._plant.device)
        count = torch.as_tensor(counts, dtype=torch.int64, device=self._plant.device)
        if direction.shape != (self.size,) or count.shape != (self.size,) or bool(
            torch.any(count < 0)
        ):
            raise ValueError("Expected valid P&V directions and counts per crosspoint.")
        maximum = int(count.max().item()) if count.numel() else 0
        for index in range(maximum):
            self._plant.pulse(
                torch.where(count > index, direction, torch.zeros_like(direction))
            )


class _LocalStarUpdatePort:
    """Restricted local-recovery view with no fault map or persistent state."""

    __slots__ = (
        "__size",
        "__nominal_dw_min",
        "__state_kind",
        "__apparent_state",
        "__persistent_state",
        "__pulse_writer",
    )

    def __init__(self, plant: IbmOmEffectiveCrossbarPlant) -> None:
        self.__size = plant.size
        self.__nominal_dw_min = float(plant.population.nominal_dw_min)
        self.__state_kind = plant.state_kind
        self.__apparent_state = plant.apparent
        self.__persistent_state = plant.persistent
        self.__pulse_writer = plant.pulse

    @property
    def size(self) -> int:
        return self.__size

    @property
    def nominal_dw_min(self) -> float:
        return self.__nominal_dw_min

    @property
    def apparent(self) -> torch.Tensor:
        return self.__apparent_state.detach().clone()

    def pulse(self, directions: torch.Tensor) -> None:
        self.__pulse_writer(directions)

    def state_hash_receipt(self) -> dict[str, str]:
        return {
            "plant_state_kind": self.__state_kind,
            "apparent_sha256": tensor_sha256(self.__apparent_state),
            "persistent_sha256": tensor_sha256(self.__persistent_state),
        }


def _layout_receipt(
    layout: Sequence[CrossbarTileSpec],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    """Return the immutable scalar layout identity used by restricted ports."""

    return tuple(
        (
            tile.key,
            tile.layer_index,
            tile.tile_index,
            tile.input_start,
            tile.input_stop,
            tile.out_features,
        )
        for tile in layout
    )


class _OnChipCrossbarRecoveryPort:
    """Capability-limited slow-array view for label-supervised recovery."""

    __slots__ = (
        "__size",
        "__nominal_dw_min",
        "__layout",
        "__layout_receipt_value",
        "__digital_scales",
        "__state_kind",
        "__device",
        "__apparent_state",
        "__persistent_state",
        "__grouped_pulse_writer",
    )

    def __init__(
        self,
        plant: IbmOmEffectiveCrossbarPlant,
        *,
        layout: Sequence[CrossbarTileSpec],
        digital_scales: Sequence[float],
    ) -> None:
        scales = tuple(float(value) for value in digital_scales)
        if len(scales) != 2 or any(
            not math.isfinite(value) or value <= 0.0 for value in scales
        ):
            raise ValueError("Expected two finite positive physical-q scales.")
        self.__size = plant.size
        self.__nominal_dw_min = float(plant.population.nominal_dw_min)
        self.__layout = tuple(layout)
        self.__layout_receipt_value = _layout_receipt(self.__layout)
        self.__digital_scales = scales
        self.__state_kind = plant.state_kind
        self.__device = plant.device
        self.__apparent_state = plant.apparent
        self.__persistent_state = plant.persistent
        self.__grouped_pulse_writer = plant.apply_grouped_pulse_counts

    @property
    def size(self) -> int:
        return self.__size

    @property
    def nominal_dw_min(self) -> float:
        return self.__nominal_dw_min

    @property
    def q_scales(self) -> tuple[float, float]:
        return self.__digital_scales

    def hardware_contract(self) -> dict[str, Any]:
        return {
            "size": self.__size,
            "nominal_dw_min": self.__nominal_dw_min,
            "layout": self.__layout_receipt_value,
            "q_scales": self.__digital_scales,
            "device": str(self.__device),
            "plant_state_kind": self.__state_kind,
            "forward_state": "apparent_q",
            "full_weight_tensor_exposed": False,
            "fault_mask_exposed": False,
        }

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> StandardCrossbarForwardStates:
        if inputs.ndim != 2 or inputs.device != self.__device or not bool(
            torch.all(torch.isfinite(inputs))
        ):
            raise ValueError(
                "Expected finite rank-2 inputs on the recovery-port device."
            )
        return standard_crossbar_forward_states(
            inputs,
            self.__apparent_state,
            self.__layout,
            digital_scales=self.__digital_scales,
        )

    @torch.no_grad()
    def output_transpose_mvm(self, output_errors: torch.Tensor) -> torch.Tensor:
        """Apply the apparent second-layer transpose without exporting W2."""

        if output_errors.ndim != 2 or output_errors.device != self.__device or not bool(
            torch.all(torch.isfinite(output_errors))
        ):
            raise ValueError(
                "Expected finite rank-2 output errors on the recovery-port device."
            )
        _, output_weight = effective_state_to_logical_weights(
            self.__apparent_state,
            self.__layout,
            digital_scales=self.__digital_scales,
        )
        if output_errors.shape[1] != output_weight.shape[1]:
            raise ValueError("Expected output errors to match the output crossbar.")
        return output_errors @ output_weight.transpose(0, 1)

    def apply_grouped_pulse_counts(
        self,
        positive_counts: torch.Tensor,
        negative_counts: torch.Tensor,
    ) -> None:
        self.__grouped_pulse_writer(positive_counts, negative_counts)

    def state_hash_receipt(self) -> dict[str, str]:
        return {
            "plant_state_kind": self.__state_kind,
            "apparent_sha256": tensor_sha256(self.__apparent_state),
            "persistent_sha256": tensor_sha256(self.__persistent_state),
        }


class _TikiTakaFastArrayPort:
    """Restricted fast-array port exposing only selected transfer columns."""

    __slots__ = (
        "__size",
        "__nominal_dw_min",
        "__layout_receipt_value",
        "__tile_offsets",
        "__tile_shapes",
        "__device",
        "__apparent_state",
        "__persistent_state",
        "__grouped_pulse_writer",
    )

    def __init__(
        self,
        plant: IbmOmEffectiveCrossbarPlant,
        *,
        layout: Sequence[CrossbarTileSpec],
    ) -> None:
        tiles = tuple(layout)
        offsets = []
        offset = 0
        for tile in tiles:
            offsets.append(offset)
            offset += tile.cells
        self.__size = plant.size
        self.__nominal_dw_min = float(plant.population.nominal_dw_min)
        self.__layout_receipt_value = _layout_receipt(tiles)
        self.__tile_offsets = tuple(offsets)
        self.__tile_shapes = tuple(tile.shape for tile in tiles)
        self.__device = plant.device
        self.__apparent_state = plant.apparent
        self.__persistent_state = plant.persistent
        self.__grouped_pulse_writer = plant.apply_grouped_pulse_counts

    @property
    def size(self) -> int:
        return self.__size

    @property
    def nominal_dw_min(self) -> float:
        return self.__nominal_dw_min

    def hardware_contract(self) -> dict[str, Any]:
        return {
            "size": self.__size,
            "nominal_dw_min": self.__nominal_dw_min,
            "layout": self.__layout_receipt_value,
            "device": str(self.__device),
            "selected_slice_only": True,
            "full_weight_tensor_exposed": False,
            "fault_mask_exposed": False,
        }

    @torch.no_grad()
    def read_apparent_column(
        self,
        *,
        tile_index: int,
        column_index: int,
    ) -> torch.Tensor:
        """Read one physical input column in AIHWKit's ``[out,in]`` sense."""

        if (
            isinstance(tile_index, bool)
            or not isinstance(tile_index, int)
            or tile_index < 0
            or tile_index >= len(self.__tile_shapes)
        ):
            raise ValueError("Expected a valid physical Tiki-Taka tile index.")
        inputs, outputs = self.__tile_shapes[tile_index]
        if (
            isinstance(column_index, bool)
            or not isinstance(column_index, int)
            or column_index < 0
            or column_index >= inputs
        ):
            raise ValueError("Expected a valid physical input-column index.")
        start = self.__tile_offsets[tile_index] + column_index * outputs
        return self.__apparent_state[start : start + outputs].detach().clone()

    def apply_grouped_pulse_counts(
        self,
        positive_counts: torch.Tensor,
        negative_counts: torch.Tensor,
    ) -> None:
        self.__grouped_pulse_writer(positive_counts, negative_counts)

    def state_hash_receipt(self) -> dict[str, str]:
        return {
            "apparent_sha256": tensor_sha256(self.__apparent_state),
            "persistent_sha256": tensor_sha256(self.__persistent_state),
        }


class PulseAdam:
    """Digital Adam moments followed by one open-loop OM pulse opportunity."""

    STATE_SCHEMA = "ebl.ibm_om_crossbar_pulse_adam"

    def __init__(
        self,
        *,
        size: int,
        layout: Sequence[CrossbarTileSpec],
        learning_rates: Sequence[float],
        betas: Sequence[float],
        epsilon: float,
        layer_scope: str,
        nominal_dw_min: float,
        pulse_cap_per_cell: int | None,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> None:
        tiles = _normalize_crossbar_layout(layout)
        rates = tuple(float(value) for value in learning_rates)
        beta = tuple(float(value) for value in betas)
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or size != sum(tile.cells for tile in tiles)
            or len(rates) != 2
            or any(not math.isfinite(value) or value < 0.0 for value in rates)
            or len(beta) != 2
            or not 0.0 <= beta[0] < 1.0
            or not 0.0 <= beta[1] < 1.0
            or not math.isfinite(epsilon)
            or epsilon <= 0.0
            or not math.isfinite(nominal_dw_min)
            or nominal_dw_min <= 0.0
            or (
                pulse_cap_per_cell is not None
                and (
                    isinstance(pulse_cap_per_cell, bool)
                    or pulse_cap_per_cell < 1
                )
            )
            or layer_scope not in {"all", "input_only", "output_only"}
        ):
            raise ValueError("Expected a valid two-layer pulse-Adam configuration.")
        self.device = torch.device(device)
        self.generator = generator
        self.beta_1, self.beta_2 = beta
        self._learning_rates = rates
        self._layer_scope = layer_scope
        self._layout_receipt_value = _layout_receipt(tiles)
        self.epsilon = float(epsilon)
        self.nominal_dw_min = float(nominal_dw_min)
        self.pulse_cap_per_cell = (
            None if pulse_cap_per_cell is None else int(pulse_cap_per_cell)
        )
        self.step_index = 0
        self.learning_rate_scale = 1.0
        self.first_moment = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.device = self.first_moment.device
        self.second_moment = torch.zeros_like(self.first_moment)
        slices = layer_cell_slices(tiles)
        self.learning_rate = torch.empty_like(self.first_moment)
        self.learning_rate[slices[0]] = rates[0]
        self.learning_rate[slices[1]] = rates[1]
        self.enabled = torch.zeros(size, dtype=torch.bool, device=self.device)
        if layer_scope in {"all", "input_only"}:
            self.enabled[slices[0]] = True
        if layer_scope in {"all", "output_only"}:
            self.enabled[slices[1]] = True
        self.requested = 0
        self.applied = 0
        self.probability_clipped = 0
        self.blocked_at_cap = 0
        self.pulse_count = torch.zeros(size, dtype=torch.int64, device=self.device)
        self._layer_slices = slices

    def step(
        self,
        gradient: torch.Tensor,
        plant: IbmOmEffectiveCrossbarPlant | _LocalStarUpdatePort,
    ) -> dict[str, int | float]:
        value = gradient.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if (
            plant.size != self.first_moment.numel()
            or value.shape != self.first_moment.shape
            or not bool(torch.all(torch.isfinite(value)))
        ):
            raise ValueError("Expected one finite gradient per crosspoint.")
        self.step_index += 1
        self.first_moment.mul_(self.beta_1).add_(value, alpha=1.0 - self.beta_1)
        self.second_moment.mul_(self.beta_2).addcmul_(
            value, value, value=1.0 - self.beta_2
        )
        first_hat = self.first_moment / (1.0 - self.beta_1**self.step_index)
        second_hat = self.second_moment / (1.0 - self.beta_2**self.step_index)
        command = -self.learning_rate * first_hat / (second_hat.sqrt() + self.epsilon)
        if not math.isfinite(self.learning_rate_scale) or not 0 < self.learning_rate_scale <= 1:
            raise ValueError('Expected a finite learning-rate scale in (0,1].')
        command = command * self.learning_rate_scale
        command = torch.where(self.enabled, command, torch.zeros_like(command))
        raw_probability = command.abs() / self.nominal_dw_min
        clipped = raw_probability > 1.0
        probability = raw_probability.clamp(max=1.0)
        candidate = torch.rand(
            probability.shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        ) < probability
        at_cap = (
            torch.zeros_like(self.enabled)
            if self.pulse_cap_per_cell is None
            else self.pulse_count >= self.pulse_cap_per_cell
        )
        blocked = candidate & self.enabled & at_cap
        selected = candidate & self.enabled & ~at_cap
        direction = torch.sign(command).to(torch.int8) * selected.to(torch.int8)
        requested = int((raw_probability > 0.0).sum().item())
        applied = int(selected.sum().item())
        clipped_count = int(clipped.sum().item())
        blocked_count = int(blocked.sum().item())
        self.requested += requested
        self.applied += applied
        self.probability_clipped += clipped_count
        self.blocked_at_cap += blocked_count
        self.pulse_count += selected.to(torch.int64)
        plant.pulse(direction)
        return {
            "requested_nonzero_commands": requested,
            "commanded_pulses": applied,
            "applied_pulses": applied,
            "probability_clipped": clipped_count,
            "blocked_at_cap": blocked_count,
            "maximum_probability_before_clip": float(raw_probability.max().item()),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "size": self.first_moment.numel(),
            "layout": self._layout_receipt_value,
            "learning_rates": self._learning_rates,
            "betas": (self.beta_1, self.beta_2),
            "epsilon": self.epsilon,
            "layer_scope": self._layer_scope,
            "nominal_dw_min": self.nominal_dw_min,
            "pulse_cap_per_cell": self.pulse_cap_per_cell,
            "device": str(self.device),
            "forward_state": "held_apparent_q",
            "write_state": "persistent_q",
            "gradient_handoff": "identity_ste_apparent_q_to_persistent_pulse_update",
        }

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint Adam moments, endurance counts, counters, and RNG exactly."""

        return {
            "schema": self.STATE_SCHEMA,
            "schema_version": 1,
            "contract": self.contract(),
            "step_index": self.step_index,
            "learning_rate_scale": self.learning_rate_scale,
            "first_moment": self.first_moment.detach().cpu().clone(),
            "second_moment": self.second_moment.detach().cpu().clone(),
            "requested": self.requested,
            "applied": self.applied,
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "pulse_count": self.pulse_count.detach().cpu().clone(),
            "generator_state": self.generator.get_state().detach().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore only an exactly matching pulse-Adam configuration."""

        expected = {
            "schema",
            "schema_version",
            "contract",
            "step_index",
            "first_moment",
            "second_moment",
            "requested",
            "applied",
            "probability_clipped",
            "blocked_at_cap",
            "pulse_count",
            "generator_state",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) not in (expected, expected | {'learning_rate_scale'})
            or state.get("schema") != self.STATE_SCHEMA
            or isinstance(state.get("schema_version"), bool)
            or state.get("schema_version") != 1
            or state.get("contract") != self.contract()
        ):
            raise ValueError("Expected a matching IBM OM pulse-Adam state.")

        def nonnegative_integer(name: str) -> int:
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Expected non-negative pulse-Adam counters.")
            return value

        scale = float(state.get('learning_rate_scale', 1.0))
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError('Expected a saved learning-rate scale in (0,1].')
        self.learning_rate_scale = scale
        step_index = nonnegative_integer("step_index")
        requested = nonnegative_integer("requested")
        applied = nonnegative_integer("applied")
        probability_clipped = nonnegative_integer("probability_clipped")
        blocked_at_cap = nonnegative_integer("blocked_at_cap")
        pending: dict[str, torch.Tensor] = {}
        for name, dtype in (
            ("first_moment", torch.float32),
            ("second_moment", torch.float32),
            ("pulse_count", torch.int64),
        ):
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != self.first_moment.shape
                or value.dtype != dtype
            ):
                raise ValueError(f"Expected a matching pulse-Adam tensor {name!r}.")
            pending[name] = value.detach().to(self.device).clone()
        if (
            not bool(torch.all(torch.isfinite(pending["first_moment"])))
            or not bool(torch.all(torch.isfinite(pending["second_moment"])))
            or bool(torch.any(pending["second_moment"] < 0.0))
            or bool(torch.any(pending["pulse_count"] < 0))
            or applied != int(pending["pulse_count"].sum().item())
            or applied + blocked_at_cap > requested
            or probability_clipped > requested
            or (
                self.pulse_cap_per_cell is not None
                and bool(torch.any(pending["pulse_count"] > self.pulse_cap_per_cell))
            )
        ):
            raise ValueError("Expected a self-consistent pulse-Adam state.")
        generator_state = state["generator_state"]
        if (
            not isinstance(generator_state, torch.Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.device.type != "cpu"
            or generator_state.ndim != 1
        ):
            raise ValueError("Expected a saved pulse-Adam generator state tensor.")
        probe = torch.Generator(device=self.device.type)
        try:
            probe.set_state(generator_state.detach().cpu())
        except RuntimeError as error:
            raise ValueError("Expected a valid pulse-Adam generator state.") from error

        self.step_index = step_index
        self.first_moment.copy_(pending["first_moment"])
        self.second_moment.copy_(pending["second_moment"])
        self.requested = requested
        self.applied = applied
        self.probability_clipped = probability_clipped
        self.blocked_at_cap = blocked_at_cap
        self.pulse_count.copy_(pending["pulse_count"])
        self.generator.set_state(generator_state.detach().cpu())

    def report(self) -> dict[str, Any]:
        layer_counts = [
            int(self.pulse_count[layer_slice].sum().item())
            for layer_slice in self._layer_slices
        ]
        return {
            "optimizer_steps": self.step_index,
            "requested_nonzero_commands": self.requested,
            "commanded_pulses": self.applied,
            "applied_pulses": self.applied,
            "applied_pulses_by_layer": layer_counts,
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "pulse_cap_per_cell": self.pulse_cap_per_cell,
            "cells_at_cap": (
                0
                if self.pulse_cap_per_cell is None
                else int(
                    (self.pulse_count >= self.pulse_cap_per_cell).sum().item()
                )
            ),
            "commanded_cells": int((self.pulse_count > 0).sum().item()),
            "changed_cells": int((self.pulse_count > 0).sum().item()),
            "maximum_pulses_per_cell": int(self.pulse_count.max().item()),
            "enabled_cells": int(self.enabled.sum().item()),
        }


class PulseSGD:
    """Moment-free local gradient to stochastic one-pulse conversion."""

    def __init__(
        self,
        *,
        size: int,
        layout: Sequence[CrossbarTileSpec],
        learning_rates: Sequence[float],
        layer_scope: str,
        nominal_dw_min: float,
        pulse_cap_per_cell: int,
        pulse_rule: str,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> None:
        rates = tuple(float(value) for value in learning_rates)
        if (
            size < 1
            or len(rates) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in rates)
            or not math.isfinite(nominal_dw_min)
            or nominal_dw_min <= 0.0
            or isinstance(pulse_cap_per_cell, bool)
            or pulse_cap_per_cell < 1
            or layer_scope not in {"all", "input_only", "output_only"}
            or pulse_rule not in {"stochastic_pulse_sgd", "stochastic_pulse_sign_sgd"}
        ):
            raise ValueError("Expected a valid two-layer moment-free pulse-SGD configuration.")
        self.device = torch.device(device)
        self.generator = generator
        self.nominal_dw_min = float(nominal_dw_min)
        self.pulse_cap_per_cell = int(pulse_cap_per_cell)
        self.pulse_rule = pulse_rule
        self.step_index = 0
        slices = layer_cell_slices(layout)
        self._learning_rates = rates
        self.enabled = torch.zeros(size, dtype=torch.bool, device=self.device)
        if layer_scope in {"all", "input_only"}:
            self.enabled[slices[0]] = True
        if layer_scope in {"all", "output_only"}:
            self.enabled[slices[1]] = True
        self.requested = 0
        self.commanded = 0
        self.probability_clipped = 0
        self.blocked_at_cap = 0
        self.pulse_count = torch.zeros(size, dtype=torch.int64, device=self.device)
        self._layer_slices = slices

    def step(
        self,
        gradient: torch.Tensor,
        plant: IbmOmEffectiveCrossbarPlant | _LocalStarUpdatePort,
    ) -> dict[str, int | float]:
        value = gradient.detach().to(device=self.device, dtype=torch.float32).reshape(-1)
        if value.shape != self.pulse_count.shape or not bool(torch.all(torch.isfinite(value))):
            raise ValueError("Expected one finite local gradient per crosspoint.")
        self.step_index += 1
        magnitude = (
            value.abs()
            if self.pulse_rule == "stochastic_pulse_sgd"
            else (value != 0).to(torch.float32)
        )
        raw_probability = torch.zeros_like(magnitude)
        for layer_index, layer_slice in enumerate(self._layer_slices):
            raw_probability[layer_slice] = (
                self._learning_rates[layer_index]
                * magnitude[layer_slice]
                / self.nominal_dw_min
            )
        raw_probability = torch.where(
            self.enabled, raw_probability, torch.zeros_like(raw_probability)
        )
        clipped = raw_probability > 1.0
        probability = raw_probability.clamp(max=1.0)
        candidate = torch.rand(
            probability.shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        ) < probability
        at_cap = self.pulse_count >= self.pulse_cap_per_cell
        blocked = candidate & self.enabled & at_cap
        selected = candidate & self.enabled & ~at_cap
        direction = -torch.sign(value).to(torch.int8) * selected.to(torch.int8)
        requested = int((raw_probability > 0.0).sum().item())
        commanded = int(selected.sum().item())
        clipped_count = int(clipped.sum().item())
        blocked_count = int(blocked.sum().item())
        self.requested += requested
        self.commanded += commanded
        self.probability_clipped += clipped_count
        self.blocked_at_cap += blocked_count
        self.pulse_count += selected.to(torch.int64)
        plant.pulse(direction)
        return {
            "requested_nonzero_commands": requested,
            "commanded_pulses": commanded,
            "probability_clipped": clipped_count,
            "blocked_at_cap": blocked_count,
            "maximum_probability_before_clip": float(raw_probability.max().item()),
        }

    def report(self) -> dict[str, Any]:
        layer_counts = [
            int(self.pulse_count[layer_slice].sum().item())
            for layer_slice in self._layer_slices
        ]
        return {
            "optimizer": "moment_free_pulse_sgd",
            "pulse_rule": self.pulse_rule,
            "learning_rates_q": list(self._learning_rates),
            "nominal_dw_min": self.nominal_dw_min,
            "command_direction": "-sign(gradient_q)",
            "selection_probability": (
                "min(1,learning_rate_q*abs(gradient_q)/nominal_dw_min)"
            ),
            "optimizer_steps": self.step_index,
            "requested_nonzero_commands": self.requested,
            "commanded_pulses": self.commanded,
            "commanded_pulses_by_layer": layer_counts,
            "probability_clipped": self.probability_clipped,
            "blocked_at_cap": self.blocked_at_cap,
            "pulse_cap_per_cell": self.pulse_cap_per_cell,
            "cells_at_cap": int(
                (self.pulse_count >= self.pulse_cap_per_cell).sum().item()
            ),
            "commanded_cells": int((self.pulse_count > 0).sum().item()),
            "maximum_pulses_per_cell": int(self.pulse_count.max().item()),
            "enabled_cells": int(self.enabled.sum().item()),
            "digital_first_moment_values": 0,
            "digital_second_moment_values": 0,
            "fp32_shadow_weight_values": 0,
            "persistent_fp32_update_state_values": 0,
            "persistent_per_cell_state": "integer_endurance_counter_only",
        }


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "CROSSBAR_TRAJECTORY_REPRODUCIBILITY_SCOPE",
    "CROSSBAR_TRAJECTORY_RNG_BACKEND",
    "CROSSBAR_TRAJECTORY_SEED_DERIVATION",
    "CROSSBAR_TRAJECTORY_STATISTICAL_CONTRACT",
    "CrossbarTileSpec",
    "DeterministicEffectiveCodebook",
    "IbmOmCrossbarStateBundle",
    "IbmOmEffectiveCrossbarPlant",
    "LocalStarCrossbarStep",
    "PulseAdam",
    "PulseSGD",
    "StandardCrossbarForwardStates",
    "aihwkit_analog_linear_default_logical_weights",
    "apply_population_bound_policy",
    "build_crossbar_layout",
    "build_deterministic_effective_codebook",
    "crossbar_state_bundle",
    "crossbar_trajectory_seeds",
    "effective_state_to_logical_weights",
    "flatten_logical_crossbar_matrices",
    "layer_cell_slices",
    "local_star_crossbar_step",
    "map_logical_weights",
    "project_to_nearest_effective_code",
    "project_to_nearest_effective_code_device",
    "restore_crossbar_state_bundle",
    "standard_crossbar_logits",
    "standard_crossbar_forward_states",
    "tensor_sha256",
    "validate_population_layout",
]
