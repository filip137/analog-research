"""hardware.faults — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from hardware.plants import population_fingerprint, prepare_population


def om_population(raw, kind, rate, seed):
    """Repair baseline is sampled before this independent Bernoulli fault mask.

    Low/high/random refer to each active cell's own native bound interval;
    the sampled intrinsic reference is retained. Not physical PCM G=0/25 uS.
    """
    if raw["kind"] != "om" or raw["policy"] != "repaired" or raw["aihwkit_version"] != "1.1.0":
        raise ValueError("Expected literal repaired AIHWKit 1.1.0 OM identities.")
    p = prepare_population(raw)
    h = p["hidden"]
    if h["corrupt"].any():
        raise ValueError("Expected a counterfactual repaired OM baseline.")
    rng = torch.Generator().manual_seed(seed)
    mask = torch.rand(h["max_bound"].shape, generator=rng) < rate
    fraction = torch.rand(mask.shape, generator=rng)
    if kind == "open":
        fraction.zero_()
    elif kind == "gmax":
        fraction.fill_(1)
    elif kind != "random":
        raise ValueError("Unknown failure kind.")
    stuck = h["min_bound"] + fraction * (h["max_bound"] - h["min_bound"])
    for key in ("min_bound", "max_bound"):
        h[key][mask] = stuck[mask]
    for key in ("dwmin_up", "dwmin_down"):
        h[key][mask] = 0
    h["corrupt"] |= mask
    h["injected_faults"] = mask
    p.update(fault_kind=kind, fault_rate=rate, fault_seed=seed)
    p["fingerprint"] = population_fingerprint(p)
    return p
