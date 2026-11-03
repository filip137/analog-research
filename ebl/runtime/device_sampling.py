"""ebl.runtime.device_sampling — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
from core.tensor_identity import state_hash
from core.serialization import load
from core.seeds import binding_seed
from hardware.arrays import ROOT


def native_sample(*, kind, size, seed, policy, variation, output: Path) -> dict:
    """One literal AIHWKit 1.1.0 population drawn by the isolated sampler."""

    interpreter = os.environ.get("EBL_AIHWKIT_PYTHON")
    if not interpreter:
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to name a pinned AIHWKit 1.1.0 CPU interpreter."
        )
    command = [
        interpreter,
        "-m",
        "experiments.cifar_crossbar.native",
        "--kind",
        kind,
        "--size",
        str(size),
        "--seed",
        str(seed),
        "--policy",
        policy,
        "--variation",
        str(variation),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run(command, cwd=ROOT, env=environment, check=True, stdout=subprocess.DEVNULL)
    raw = load(output)
    output.unlink()
    return raw


def sample_populations(lifecycle, layout, scratch: Path, sampler=None) -> dict:
    """Layer-bound OM populations for every array plus characterization cells."""

    devices, dataset = lifecycle.devices, lifecycle.network.binding_namespace
    sampler = native_sample if sampler is None else sampler
    scratch.mkdir(parents=True, exist_ok=True)

    def draw(size, seed):
        return sampler(
            kind="om",
            size=size,
            seed=seed,
            policy=devices.policy,
            variation=devices.variation_scale,
            output=scratch / f"population_{seed}.pt",
        )

    layers = {}
    for seed in devices.assignment_seeds + devices.selection_seeds:
        layers[str(seed)] = {
            m.name: draw(m.size, binding_seed(seed, dataset, m.name)) for m in layout
        }
    c = devices.characterization
    return {
        "schema": "crossbar_lifecycle.om_populations.v1",
        "dataset": dataset,
        "layers": layers,
        "kernel_fit": draw(c.bins * c.samples, c.fit_seed),
        "kernel_heldout": draw(c.bins * c.samples, c.heldout_seed),
    }


def population_receipt(populations, technology: str) -> dict:
    """Small identity summary of a population bundle for run metrics."""

    if populations is None:
        return {"kind": f"analytic_{technology}_endpoint", "sampled_cells": 0}
    return {
        "kind": "aihwkit_1.1.0_om",
        "arrays": sorted(populations["layers"]),
        "layer_hidden_sha256": {
            seed: {name: state_hash(raw["hidden"]) for name, raw in item.items()}
            for seed, item in populations["layers"].items()
        },
        "kernel_hidden_sha256": {
            role: state_hash(populations["kernel_" + role]["hidden"]) for role in ("fit", "heldout")
        },
    }
