"""AIHWKit 1.1.0 CPU population sampler, isolated from GPU training Python."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from experiments.cifar_crossbar.model import state_hash


def sample(kind, size, seed, policy, variation):
    import aihwkit
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.presets.devices import (
        PCMPresetDevice,
        ReRamArrayOMPresetDevice,
    )
    from aihwkit.simulator.tiles import AnalogTile
    from training.ibm_reram_program_verify import (
        OM_PRESET,
        PUBLISHED_CORRUPT_PROBABILITY,
    )

    if aihwkit.__version__ != "1.1.0":
        raise RuntimeError("Expected AIHWKit 1.1.0 for population sampling.")
    cls = ReRamArrayOMPresetDevice if kind == "om" else PCMPresetDevice
    preset = cls()
    for name in ("dw_min_dtod", "w_min_dtod", "w_max_dtod", "up_down_dtod"):
        setattr(preset, name, getattr(preset, name) * variation)
    if kind == "om":
        preset.corrupt_devices_prob = PUBLISHED_CORRUPT_PROBABILITY[OM_PRESET]
    preset.construction_seed = seed % (2**31 - 2) + 1
    tile = AnalogTile(
        1 if kind == "om" else 2, size, SingleRPUConfig(device=preset), bias=False
    )
    hidden = {
        k: v.detach().clone().float()
        for k, v in tile.get_hidden_parameters().items()
        if k != "persistent_weights"
    }
    corrupt = (
        ((hidden["max_bound"] - hidden["min_bound"]).abs() <= 1e-12)
        & (hidden["dwmin_up"] == 0)
        & (hidden["dwmin_down"] == 0)
    )
    published = corrupt.clone()
    if policy == "repaired" and corrupt.any():
        for attempt in range(10):
            preset.corrupt_devices_prob = 0.0
            preset.construction_seed = (seed + 7919 * (attempt + 1)) % (2**31 - 2) + 1
            donor = AnalogTile(
                1 if kind == "om" else 2,
                size,
                SingleRPUConfig(device=preset),
                bias=False,
            ).get_hidden_parameters()
            for key in hidden:
                hidden[key][corrupt] = donor[key][corrupt]
            corrupt = (
                ((hidden["max_bound"] - hidden["min_bound"]).abs() <= 1e-12)
                & (hidden["dwmin_up"] == 0)
                & (hidden["dwmin_down"] == 0)
            )
            if not corrupt.any():
                break
        if corrupt.any():
            raise RuntimeError("Expected successful independent donor repair.")
    hidden["corrupt"] = corrupt
    hidden["published_corrupt"] = published
    constants = {
        name: getattr(preset, name)
        for name in (
            "dw_min",
            "dw_min_std",
            "write_noise_std",
            "reset",
            "reset_std",
            "reset_dtod",
            "w_max",
            "w_min",
        )
    }
    if kind == "pcm":
        constants.update(
            {
                name: getattr(preset, name)
                for name in (
                    "A_up",
                    "gamma_up",
                    "a",
                    "b",
                    "dw_min_std_add",
                    "dw_min_std_slope",
                )
            }
        )
    return {
        "schema": "cifar_crossbar.population.v1",
        "kind": kind,
        "size": size,
        "seed": seed,
        "policy": policy,
        "variation": variation,
        "aihwkit_version": "1.1.0",
        "constants": constants,
        "hidden": hidden,
        "hidden_sha256": state_hash(hidden),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("om", "pcm"), required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--policy", choices=("native", "repaired"), required=True)
    p.add_argument("--variation", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    result = sample(a.kind, a.size, a.seed, a.policy, a.variation)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, a.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("hidden",)}))


if __name__ == "__main__":
    main()
