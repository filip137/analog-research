"""Declared retention encodings, native endpoints and held observations."""
from __future__ import annotations
import torch
import hardware.devices.cmo as cmo
from hardware.drift.inference import wan_snapshot

MODELS = {
    "pcm": {"class": "PCMLikeNoiseModel", "g_min": 0.0, "g_max": 25.0,
            "encoding": "differential_pair", "normalizer": 25.0},
    "reram_cmo": {"class": "ReRamCMONoiseModel", "g_min": cmo.G_MIN_US, "g_max": cmo.G_MAX_US,
                  "encoding": "active_reference_pair", "normalizer": cmo.HALF_RANGE_US},
    "reram_wan2022": {"class": "ReRamWan2022NoiseModel", "g_min": 0.0, "g_max": 40.0,
                      "encoding": "differential_pair", "normalizer": 40.0},
}


def targets(q, model):
    if not bool(torch.isfinite(q).all()) or bool((q.abs() > 1).any()):
        raise ValueError("Expected finite normalized frozen weights in [-1, 1].")
    if model == "reram_cmo":
        return torch.stack((cmo.to_conductance(q), torch.full_like(q, cmo.MID_US)))
    gmax = MODELS[model]["g_max"]
    return torch.stack((q.clamp_min(0), (-q).clamp_min(0))) * gmax


def mapped_faults(mask, stuck, case, model, device):
    """Keep device locations and uniform quantiles matched, not raw CMO uS."""
    mask, stuck = mask.to(device), stuck.to(device)
    spec = MODELS[model]
    if model == "reram_cmo":
        return mask, stuck
    if case.kind == "random":
        quantile = (stuck - cmo.G_MIN_US) / (cmo.G_MAX_US - cmo.G_MIN_US)
        mapped = spec["g_min"] + quantile * (spec["g_max"] - spec["g_min"])
    else:
        value = spec["g_max"] if case.kind == "gmax" else spec["g_min"]
        mapped = torch.full_like(stuck, value)
    return mask, torch.where(mask, mapped, 0.0)


@torch.no_grad()
def program(target, model, generator, *, acceptance, noise_scale):
    """One native programming draw; Wan's programming fit is its 1 s fit."""
    if model == "reram_wan2022":
        return wan_snapshot(target, generator, 1, g_max=40, noise_scale=noise_scale)
    noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
    if model == "reram_cmo":
        return target + noise_scale * cmo.programming_sigma(target, acceptance) * noise
    relative = target / 25.0
    sigma = 0.26348 + relative * 1.9650 + relative.square() * -1.1731
    return (target + noise_scale * sigma * noise).clamp_min(0)


def apparent(series, model, seconds, scale=1.0):
    pieces = []
    for trace in series:
        # Reuse the actual 1 s programming realization as Wan's first snapshot.
        g = (trace.baseline() if seconds is None or (model == "reram_wan2022" and seconds == 1)
             else trace.snapshot(seconds, scale))
        if not torch.equal(g[trace.mask], trace.stuck[trace.mask]):
            raise RuntimeError("Retention changed a permanent defect.")
        pieces.append((g[0] - g[1]) / MODELS[model]["normalizer"])
    return torch.cat(pieces)
