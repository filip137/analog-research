"""(a) Devices and permanent defects: fresh physical arrays before programming.

One assignment seed names one array identity. Every layer binds its own
population, fault map and programming stream through ``binding_seed``, so
identities do not depend on layer order or on other arrays. Defect cases are
applied to the same assignment, giving matched fault conditions rather than
independent arrays. Controllers later see only reads and writes; fault masks
and hidden cell parameters stay on the analysis side.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import torch

from experiments.cifar_crossbar.devices import (
    clone_cpu,
    make_plant,
    population_fingerprint,
)
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults
from experiments.cifar_crossbar.runtime import load
from experiments.cifar_crossbar.sweep_devices import binding_seed, combined_om

ROOT = Path(__file__).resolve().parents[1]


class PcmEndpointArray:
    """Layer-bound Gaussian PCM endpoint array with permanent per-bank faults.

    The fault map exists from construction; conductances exist after the
    first ``program`` call. With nominal noise and no relaxation this is the
    historical ``LayeredPcmArray``. A relaxation law, when attached by the P&V
    stage, produces one held apparent state per programming event.
    """

    kind = "layered_pcm"

    def __init__(self, layout, dataset, case, assignment_seed, endpoint_seed, noise_scale, device):
        self.layout = tuple(layout)
        self.dataset = dataset
        self.endpoint_seed = endpoint_seed
        self.noise_scale = noise_scale
        self.device = torch.device(device)
        self.layer_faults = [
            PermanentFaults.sample(
                m.size,
                case.sampler_kind,
                case.rate,
                binding_seed(assignment_seed, dataset, m.name),
                self.device,
            )
            for m in self.layout
        ]
        self.faults = PermanentFaults(
            torch.cat([f.mask for f in self.layer_faults], 1),
            torch.cat([f.conductance for f in self.layer_faults], 1),
            case.sampler_kind,
            case.rate,
            assignment_seed,
        )
        self.size = sum(m.size for m in self.layout)
        self.children = None
        self.relaxation = None
        self.held = None

    @property
    def programmed(self) -> bool:
        return self.children is not None

    def _slices(self, target):
        if target.shape != (self.size,):
            raise ValueError("Expected one normalized target per logical weight.")
        return [target[m.offset : m.offset + m.size] for m in self.layout]

    def program(self, target):
        pieces = self._slices(target)
        if self.children is None:
            self.children = [
                GaussianEndpointArray(
                    piece,
                    faults,
                    binding_seed(self.endpoint_seed, self.dataset, m.name),
                    self.noise_scale,
                )
                for piece, faults, m in zip(pieces, self.layer_faults, self.layout)
            ]
        else:
            for child, piece in zip(self.children, pieces):
                child.program(piece)
        self.held = None if self.relaxation is None else self.relaxation.apply(self)

    def programmed_read(self):
        """Immediate post-programming state, before any declared relaxation."""

        if self.children is None:
            raise RuntimeError("Expected a programmed PCM array before reading it.")
        return torch.cat([child.read() for child in self.children])

    def read(self):
        if self.held is not None:
            return self.held.detach().clone()
        return self.programmed_read()

    def verify_permanence(self):
        for child in self.children or ():
            child.verify_permanence()

    def cost(self):
        if self.children is None:
            return {
                "array_reprogram_calls": 0,
                "logical_target_requests": 0,
                "healthy_device_programs": 0,
                "failed_device_write_attempts": 0,
            }
        rows = [child.cost() for child in self.children]
        if len({row["array_reprogram_calls"] for row in rows}) != 1:
            raise RuntimeError("Expected every layer of the array to be reprogrammed together.")
        return {
            key: rows[0][key] if key == "array_reprogram_calls" else sum(row[key] for row in rows)
            for key in rows[0]
        }

    def state_dict(self):
        if self.children is None:
            raise RuntimeError("Expected a programmed PCM array before saving its state.")
        return {
            "kind": self.kind,
            "children": [child.state_dict() for child in self.children],
            "held": None if self.held is None else self.held.detach().cpu().clone(),
            "relaxation": None if self.relaxation is None else self.relaxation.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("kind") != self.kind or len(state["children"]) != len(self.layout):
            raise ValueError("Expected the same layer-bound PCM array layout.")
        if (state["relaxation"] is None) != (self.relaxation is None):
            raise ValueError("Expected the same declared PCM relaxation law.")
        if self.children is None:
            # Construction draws a placeholder endpoint; the saved RNG and
            # conductances then replace it exactly.
            self.program(torch.zeros(self.size, device=self.device))
        for child, value in zip(self.children, state["children"]):
            child.load_state_dict(value)
        if self.relaxation is not None:
            self.relaxation.load_state_dict(state["relaxation"])
        self.held = None if state["held"] is None else state["held"].to(self.device).clone()


def fresh_array(lifecycle, layout, assignment_seed, case, populations, device):
    """Construct one assignment under one defect case, before programming."""

    devices, dataset = lifecycle.devices, lifecycle.network.dataset
    endpoint = devices.endpoint_seed(assignment_seed)
    if devices.technology == "pcm":
        return PcmEndpointArray(
            layout, dataset, case, assignment_seed, endpoint, devices.noise_scale, device
        )
    if populations is None:
        raise ValueError("Expected sampled OM populations for an OM lifecycle.")
    population = combined_om(
        populations, layout, dataset, assignment_seed, endpoint, case.sampler_kind, case.rate
    )
    if devices.noise_scale != 1.0:
        population = dict(population, noise_scale=devices.noise_scale)
        population["fingerprint"] = population_fingerprint(population)
    return make_plant(population, endpoint, device)


def array_identity(array, technology: str) -> dict:
    """Analysis-side receipt of the physical identity and its defects."""

    if technology == "pcm":
        return array.faults.receipt()
    hidden = array.source["hidden"]
    return {
        "identity_sha256": array.source["fingerprint"],
        "failed_devices": int(hidden["injected_faults"].sum()),
        "physical_devices": array.size,
        "published_corrupt_repaired": int(hidden["published_corrupt"].sum()),
        "fault_kind": array.source["fault_kind"],
        "requested_rate": array.source["fault_rate"],
        "placement": "independent Bernoulli per active cell; fixed throughout recovery",
        "semantics": "one active OM cell per logical weight; exact intrinsic reference; "
        "stuck persistent state with apparent write noise",
    }


def array_state(array, technology: str) -> dict:
    """Compact state: OM populations are restored from the device bundle."""

    state = array.state_dict()
    if technology == "om":
        return {
            "kind": "om",
            "population_fingerprint": array.source["fingerprint"],
            "engine": state["engine"],
        }
    return state


def load_array_state(array, state: dict, technology: str) -> None:
    if technology == "om":
        if state.get("kind") != "om" or state["population_fingerprint"] != array.source["fingerprint"]:
            raise ValueError("Expected an OM state for the same physical population.")
        array.engine.load_state_dict(state["engine"])
    else:
        array.load_state_dict(state)


def fixed_state(array, technology: str):
    """The physical state that permanent defects must preserve."""

    if technology == "pcm":
        return None
    mask = array.source["hidden"]["injected_faults"].flatten().to(array.engine.device)
    return mask, array.engine.persistent[mask].detach().clone()


def verify_fixed(array, technology: str, fixed) -> None:
    if technology == "pcm":
        array.verify_permanence()
        return
    mask, values = fixed
    if not torch.equal(array.engine.persistent[mask], values):
        raise RuntimeError("A permanent OM defect changed state.")


def persistent_state(array, technology: str):
    """Named diagnostic only; it never drives gradients or selection."""

    return array.engine.persistent if technology == "om" else None


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

    devices, dataset = lifecycle.devices, lifecycle.network.dataset
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


def population_receipt(populations) -> dict:
    """Small identity summary of a population bundle for run metrics."""

    if populations is None:
        return {"kind": "analytic_pcm_endpoint", "sampled_cells": 0}
    from experiments.cifar_crossbar.model import state_hash

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


__all__ = [
    "PcmEndpointArray",
    "array_identity",
    "array_state",
    "clone_cpu",
    "fixed_state",
    "fresh_array",
    "load_array_state",
    "native_sample",
    "persistent_state",
    "population_receipt",
    "sample_populations",
    "verify_fixed",
]
