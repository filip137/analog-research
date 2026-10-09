"""(a) Devices and permanent defects: fresh physical arrays before programming.

One assignment seed names one array identity. Every layer binds its own
population, fault map and programming stream through ``binding_seed``, so
identities do not depend on layer order or on other arrays. Defect cases are
applied to the same assignment, giving matched fault conditions rather than
independent arrays. Controllers later see only reads and writes; fault masks
and hidden cell parameters stay on the analysis side.

Technologies: ``om`` (literal AIHWKit OM cell populations), ``pcm`` (Gaussian
endpoints on a differential pair with permanent faults) and ``cmo`` (Gaussian
CMO ReRAM endpoints, an active and a reference device per weight, hierarchical
stuck devices).

CMO defects are placed hierarchically over physical devices, active and
reference alike. A weight's two devices sit on adjacent rows of one physical
array, so a tile_size x tile_size array holds tile_size/2 input rows of one
deployment tile, and each tile spans two arrays. Consecutive arrays of a group
(an OPT decoder layer's MLP) fill dies of ``arrays_per_die``. Die rates
p_d ~ Beta(mu k_d, (1-mu) k_d) and array rates
p_da | p_d ~ Beta(p_d k_a, (1-p_d) k_a), with an infinite concentration meaning
no variation at that level. Exactly round(mu N) of the group's N devices fail:
counts are allocated to arrays multinomially in proportion to p_da times the
array size, then devices are chosen uniformly within each array. Fault maps
are drawn on the CPU with NumPy, so identities do not depend on the execution
device, and each group has its own stream.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments.cifar_crossbar.devices import (
    clone_cpu,
    make_plant,
    population_fingerprint,
)
from experiments.cifar_crossbar.model import state_hash, tensor_hash
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults
from experiments.cifar_crossbar.runtime import load
from experiments.cifar_crossbar.sweep_devices import binding_seed, combined_om
from training.reram_cmo import G_MAX_US, G_MIN_US, STUCK_US as CMO_STUCK_US, CmoMatrix

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


class CmoArray:
    """Layer-bound CMO ReRAM array: an active and a reference device per weight, held conductances.

    Each analog matrix owns a ``CmoMatrix`` and its noise stream, bound from
    the endpoint seed. Conductances exist after the first ``program`` call;
    every read returns the held state until a program or write changes it.
    """

    kind = "layered_cmo"

    def __init__(self, layout, dataset, endpoint_seed, acceptance, noise_scale, device, faults=None):
        self.layout = tuple(layout)
        self.device = torch.device(device)
        self.size = sum(m.size for m in self.layout)
        self.endpoint_seeds = {m.name: binding_seed(endpoint_seed, dataset, m.name) for m in self.layout}
        stuck = {} if faults is None else faults["matrices"]
        self.fault_receipt = None if faults is None else faults["receipt"]
        self.children = [
            CmoMatrix(m.size, self.endpoint_seeds[m.name], acceptance, noise_scale, self.device, stuck.get(m.name))
            for m in self.layout
        ]

    def _slices(self, value):
        if value.shape != (self.size,):
            raise ValueError("Expected one normalized value per logical weight.")
        return [value[m.offset : m.offset + m.size] for m in self.layout]

    def program(self, target, reference=True):
        for child, piece in zip(self.children, self._slices(target)):
            child.program(piece, reference)

    def write(self, delta):
        for child, piece in zip(self.children, self._slices(delta)):
            child.write(piece)

    def read(self):
        return torch.cat([child.read() for child in self.children])

    def write_counts(self):
        """Programs plus writes per device since construction."""

        return torch.cat([child.writes for child in self.children])

    def out_of_range(self) -> int:
        return sum(child.out_of_range() for child in self.children)

    def freshly_programmed(self) -> bool:
        """Programmed exactly once (both devices) and never written since."""

        return all(child.program_calls == 1 and child.device_writes == 0 for child in self.children)

    def verify_permanence(self) -> None:
        for child in self.children:
            child.verify_permanence()

    def cost(self) -> dict:
        rows = [child.cost() for child in self.children]
        if len({row["array_program_calls"] for row in rows}) != 1:
            raise RuntimeError("Expected every layer of the array to be programmed together.")
        return {
            key: rows[0][key] if key == "array_program_calls" else sum(row[key] for row in rows) for key in rows[0]
        }

    def state_dict(self):
        return {"kind": self.kind, "children": [child.state_dict() for child in self.children]}

    def load_state_dict(self, state):
        if state.get("kind") != self.kind or len(state["children"]) != len(self.layout):
            raise ValueError("Expected the same layer-bound CMO array layout.")
        for child, value in zip(self.children, state["children"]):
            child.load_state_dict(value)


def fault_group(name: str) -> str:
    """The module owning an analog matrix, e.g. ``decoder.layers.11`` for its fc1 and fc2."""

    return name.rsplit(".", 1)[0]


def physical_arrays(layout, tile_size, devices_per_weight=2):
    """Physical arrays in mapping order, each with its devices' flat indices.

    Each deployment tile is split along its inputs into ``devices_per_weight``
    arrays, so that every weight's devices (role-major indices ``role * size +
    weight``, i.e. a (2, size) layout flattened) share one array.
    """

    arrays = []
    for spec in layout:
        outputs, inputs = spec.shape[0], spec.size // spec.shape[0]
        for output in range(0, outputs, tile_size):
            rows = np.arange(output, min(output + tile_size, outputs))
            for row in range(0, inputs, tile_size):
                tile_inputs = np.arange(row, min(row + tile_size, inputs))
                for part, columns in enumerate(np.array_split(tile_inputs, devices_per_weight)):
                    weights = (rows[:, None] * inputs + columns[None, :]).reshape(-1)
                    indices = np.concatenate([role * spec.size + weights for role in range(devices_per_weight)])
                    arrays.append(
                        {"matrix": spec.name, "tile": [output // tile_size, row // tile_size, part], "indices": indices}
                    )
    return arrays


def _beta(rng, mean, concentration):
    if concentration is None or mean <= 0 or mean >= 1:
        return float(mean)
    return float(rng.beta(mean * concentration, (1 - mean) * concentration))


def _allocate(rng, total, sizes, weights):
    """Multinomial counts proportional to ``weights`` summing to ``total``, capped at each size."""

    counts = np.zeros(len(sizes), dtype=np.int64)
    while total:
        room = sizes - counts
        w = np.where(room > 0, weights, 0.0)
        w = w if w.sum() > 0 else room.astype(float)
        counts += rng.multinomial(total, w / w.sum())
        total = int(np.clip(counts - sizes, 0, None).sum())
        counts = np.minimum(counts, sizes)
    return counts


def hierarchical_faults(layout, tile_size, arrays_per_die, case, assignment_seed, namespace):
    """Permanent stuck devices of one assignment and defect case (see the module docstring)."""

    mu = case.rate
    by_matrix = {m.name: (np.zeros(2 * m.size, dtype=bool), np.zeros(2 * m.size, dtype=np.float32)) for m in layout}
    arrays = physical_arrays(layout, tile_size)
    groups, receipt_groups = {}, []
    for array in arrays:
        groups.setdefault(fault_group(array["matrix"]), []).append(array)
    for group, members in groups.items():
        seed = binding_seed(assignment_seed, namespace, f"{group}/faults")
        die_rng, array_rng, count_rng, device_rng, value_rng = (
            np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(5)
        )
        sizes = np.array([len(a["indices"]) for a in members], dtype=np.int64)
        dies = [
            range(start, min(start + arrays_per_die, len(members))) for start in range(0, len(members), arrays_per_die)
        ]
        die_rates = [_beta(die_rng, mu, case.die_concentration) for _ in dies]
        rates = np.zeros(len(members))
        for die, rate in zip(dies, die_rates):
            for index in die:
                rates[index] = _beta(array_rng, rate, case.array_concentration)
        total = int(round(mu * sizes.sum()))
        weights = rates * sizes
        counts = _allocate(count_rng, total, sizes, weights if weights.sum() > 0 else sizes.astype(float))
        values = (
            value_rng.uniform(G_MIN_US, G_MAX_US, size=total)
            if case.kind == "random"
            else np.full(total, CMO_STUCK_US[case.kind])
        )
        used = 0
        for array, count in zip(members, counts):
            chosen = array["indices"][device_rng.choice(len(array["indices"]), size=int(count), replace=False)]
            mask, stuck = by_matrix[array["matrix"]]
            mask[chosen] = True
            stuck[chosen] = values[used : used + count]
            used += int(count)
        receipt_groups.append(
            {
                "group": group,
                "devices": int(sizes.sum()),
                "failed_devices": total,
                "dies": [
                    {"die": d, "arrays": len(die), "rate": rate, "failed_devices": int(counts[list(die)].sum())}
                    for d, (die, rate) in enumerate(zip(dies, die_rates))
                ],
                "arrays": [
                    {
                        "matrix": a["matrix"],
                        "tile": a["tile"],
                        "die": d,
                        "devices": int(size),
                        "rate": float(rate),
                        "failed_devices": int(count),
                    }
                    for d, die in enumerate(dies)
                    for a, size, rate, count in zip(
                        [members[i] for i in die], sizes[list(die)], rates[list(die)], counts[list(die)]
                    )
                ],
            }
        )
    matrices = {
        name: (torch.from_numpy(mask).reshape(2, -1), torch.from_numpy(stuck).reshape(2, -1))
        for name, (mask, stuck) in by_matrix.items()
    }
    fractions = [a["failed_devices"] / a["devices"] for g in receipt_groups for a in g["arrays"]]
    receipt = {
        "kind": case.kind,
        "requested_rate": mu,
        "die_concentration": case.die_concentration,
        "array_concentration": case.array_concentration,
        "arrays_per_die": arrays_per_die,
        "array_shape": [tile_size, tile_size],
        "array_layout": "active and reference rows of tile_size/2 inputs x tile_size outputs",
        "physical_devices": 2 * sum(m.size for m in layout),
        "failed_active": sum(int(mask[0].sum()) for mask, _ in matrices.values()),
        "failed_reference": sum(int(mask[1].sum()) for mask, _ in matrices.values()),
        "failed_devices": sum(g["failed_devices"] for g in receipt_groups),
        "array_fault_fraction": {
            "mean": float(np.mean(fractions)),
            "std": float(np.std(fractions)),
            "min": float(np.min(fractions)),
            "max": float(np.max(fractions)),
        },
        "identity_sha256": state_hash(
            {f"{name}.{part}": value for name, pair in matrices.items() for part, value in zip(("mask", "stuck"), pair)}
        ),
        "placement": "hierarchical die/array rates; exact fault total per group; uniform devices (active or "
        "reference) within each array",
        "groups": receipt_groups,
    }
    return {"matrices": matrices, "receipt": receipt}


def fresh_array(lifecycle, layout, assignment_seed, case, populations, device):
    """Construct one assignment under one defect case, before programming."""

    devices, dataset = lifecycle.devices, lifecycle.network.binding_namespace
    endpoint = devices.endpoint_seed(assignment_seed)
    if devices.technology == "cmo":
        faults = None
        if case.kind != "none":
            faults = hierarchical_faults(
                layout, lifecycle.deployment.tile_size, lifecycle.defects.arrays_per_die, case, assignment_seed, dataset
            )
        return CmoArray(
            layout, dataset, endpoint, lifecycle.program_verify.acceptance_percent, devices.noise_scale, device, faults
        )
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
    if technology == "cmo":
        semantics = (
            "an active and a mid-range reference CMO device per logical weight; held Gaussian P&V endpoints; "
            "stuck devices keep their conductance through every program and write"
        )
        if array.fault_receipt is None:
            return {
                "kind": "none",
                "requested_rate": 0.0,
                "physical_devices": 2 * array.size,
                "failed_devices": 0,
                "endpoint_seeds": array.endpoint_seeds,
                "placement": "no defects",
                "semantics": semantics,
            }
        return {**array.fault_receipt, "endpoint_seeds": array.endpoint_seeds, "semantics": semantics}
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
    """Compact state: OM populations are restored from the device bundle.

    A CMO array programmed once and never written is fully determined by its
    target, seeds and fault map; it is stored as identities and rebuilt by
    reprogramming on the same execution-device type, then verified.
    """

    if technology == "cmo" and array.freshly_programmed():
        return {
            "kind": "layered_cmo_deployment",
            "faults_sha256": [child.fault_digest() for child in array.children],
            "apparent_sha256": tensor_hash(array.read()),
            "device_type": array.device.type,
        }
    state = array.state_dict()
    if technology == "om":
        return {
            "kind": "om",
            "population_fingerprint": array.source["fingerprint"],
            "engine": state["engine"],
        }
    return state


def load_array_state(array, state: dict, technology: str, target=None) -> None:
    if technology == "om":
        if state.get("kind") != "om" or state["population_fingerprint"] != array.source["fingerprint"]:
            raise ValueError("Expected an OM state for the same physical population.")
        array.engine.load_state_dict(state["engine"])
    elif state.get("kind") == "layered_cmo_deployment":
        if target is None or state["device_type"] != array.device.type:
            raise ValueError("Expected the deployment target on the same device type to rebuild a CMO deployment.")
        if [child.fault_digest() for child in array.children] != state["faults_sha256"]:
            raise ValueError("Expected the same permanent CMO defects.")
        array.program(target)
        if tensor_hash(array.read()) != state["apparent_sha256"]:
            raise RuntimeError("The rebuilt CMO deployment differs from the saved one.")
    else:
        array.load_state_dict(state)


def fixed_state(array, technology: str):
    """The physical state that permanent defects must preserve."""

    if technology != "om":
        return None
    mask = array.source["hidden"]["injected_faults"].flatten().to(array.engine.device)
    return mask, array.engine.persistent[mask].detach().clone()


def verify_fixed(array, technology: str, fixed) -> None:
    if technology in ("pcm", "cmo"):
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


__all__ = [
    "CMO_STUCK_US",
    "CmoArray",
    "fault_group",
    "hierarchical_faults",
    "physical_arrays",
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
