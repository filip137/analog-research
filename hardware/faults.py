"""Pinned HWA b0c1f06 hierarchical fault placement, generalized to native DRN cells.

The default two-role path retains the upstream arithmetic and CPU RNG order.
DRNs use one physical device per native matrix entry (four per logical weight).
"""
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
import torch
from hardware.devices.cmo import G_MIN_US, G_MAX_US, STUCK_US as CMO_STUCK_US


def binding_seed(seed, dataset, layer):
    return int.from_bytes(sha256(f'{seed}:{dataset}:{layer}'.encode()).digest()[:4], 'little') % 2000000000 + 1


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = sha256(str((tuple(value.shape), value.dtype)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_hash(state):
    h = sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    shape: tuple
    offset: int
    size: int
    scale: float = 1.0


@dataclass(frozen=True)
class FaultCase:
    kind: str
    rate: float
    die_concentration: float | None = None
    array_concentration: float | None = None


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

def hierarchical_faults(layout, tile_size, arrays_per_die, case, assignment_seed, namespace, devices_per_weight=2, *, domains):
    """Permanent stuck devices of one assignment and defect case (see the module docstring)."""

    if set(domains) != {m.name for m in layout} or any(not isinstance(v, str) or not v for v in domains.values()):
        raise ValueError("Expected one explicit fault domain per physical matrix.")
    mu = case.rate
    by_matrix = {m.name: (np.zeros(devices_per_weight * m.size, dtype=bool), np.zeros(devices_per_weight * m.size, dtype=np.float32)) for m in layout}
    arrays = physical_arrays(layout, tile_size, devices_per_weight)
    groups, receipt_groups = {}, []
    for array in arrays:
        groups.setdefault(domains[array["matrix"]], []).append(array)
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
        name: (torch.from_numpy(mask).reshape(devices_per_weight, -1), torch.from_numpy(stuck).reshape(devices_per_weight, -1))
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
        "array_layout": ("active and reference rows of tile_size/2 inputs x tile_size outputs" if devices_per_weight == 2 else "native DRN conductance matrix tiles"),
        "physical_devices": devices_per_weight * sum(m.size for m in layout),
        "failed_active": sum(int(mask[0].sum()) for mask, _ in matrices.values()),
        "failed_reference": sum(int(mask[1:].sum()) for mask, _ in matrices.values()),
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
