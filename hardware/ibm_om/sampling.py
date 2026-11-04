"""Physical population sampling from explicit identity and shape metadata."""
from __future__ import annotations
from collections.abc import Sequence
import torch
from hardware.ibm_om.constants import _CORRUPTION_POLICIES
from hardware.ibm_om.population import IbmReramArrayPopulation, _native_seed, _population_fingerprint
from hardware.devices.reram import OM_PRESET, PUBLISHED_CORRUPT_PROBABILITY, derive_seed


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
