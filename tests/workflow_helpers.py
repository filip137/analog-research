"""Shared synthetic fixtures for the lifecycle workflow tests."""

import json
from pathlib import Path

import torch

from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
from experiments.cifar_crossbar.sweep_devices import binding_seed
from workflow import program_verify
from workflow.lifecycle import DefectCase, parse_lifecycle

LIFECYCLES = Path(__file__).resolve().parents[1] / "campaigns/cifar-crossbar-hwa-recovery/lifecycles"
GMAX = DefectCase("gmax", 50000)
NOMINAL = DefectCase("none", 0)


def lifecycle_document(name):
    return json.loads((LIFECYCLES / f"{name}.json").read_text())


class Store:
    def __init__(self, root):
        self.run_dir = Path(root)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records = []

    def append_metric(self, record):
        self.records.append(record)


def lifecycle(name="cifar10-pcm-conv4-smoke", **changes):
    doc = lifecycle_document(name)
    for dotted, value in changes.items():
        *parents, key = dotted.split("__")
        target = doc
        for part in parents:
            target = target[part]
        target[key] = value
    return parse_lifecycle(doc)


def network_for(convolutions=0, seed=31):
    torch.manual_seed(seed)
    teacher = CifarResNet32(10).eval()
    suffix = {0: "head", 2: "last_block", 4: "last_two_blocks"}[convolutions]
    return teacher, CrossbarSuffix(teacher, suffix)


def synthetic_cache(teacher, network, n=24, seed=5):
    generator = torch.Generator().manual_seed(seed)
    cache = {
        "hwa_raw": torch.randint(0, 256, (n, 3, 32, 32), generator=generator, dtype=torch.uint8),
        "hwa_labels": torch.randint(0, 10, (n,), generator=generator),
        "hwa_cohort": {"count": n, "indices_sha256": "synthetic"},
    }
    for split in ("training", "development", "test"):
        x = torch.randn(n, 3, 32, 32, generator=generator)
        with torch.no_grad():
            cache[split] = {
                "features": network.features(x),
                "teacher_logits": teacher(x),
                "labels": torch.randint(0, 10, (n,), generator=generator),
                "cohort": {"count": n, "indices_sha256": split},
            }
    return cache


def om_raw(size, seed, step=0.05):
    generator = torch.Generator().manual_seed(seed)

    def uniform():
        return torch.rand((1, size), generator=generator)

    hidden = {
        "min_bound": -1 + 0.1 * uniform(),
        "max_bound": 1 - 0.1 * uniform(),
        "dwmin_up": step * (0.8 + 0.4 * uniform()),
        "dwmin_down": step * (0.8 + 0.4 * uniform()),
        "reference": 0.05 * (2 * uniform() - 1),
        "corrupt": torch.zeros((1, size), dtype=torch.bool),
        "published_corrupt": torch.zeros((1, size), dtype=torch.bool),
    }
    constants = {
        "dw_min": step, "dw_min_std": 0.2, "write_noise_std": 0.3, "reset": 0.01,
        "reset_std": 0.01, "reset_dtod": 0.02, "w_max": 1.0, "w_min": -1.0,
    }
    return {
        "schema": "cifar_crossbar.population.v1", "kind": "om", "size": size, "seed": seed,
        "policy": "repaired", "variation": 1.0, "aihwkit_version": "1.1.0",
        "constants": constants, "hidden": hidden,
    }


def om_populations(layout, seeds, dataset="cifar10", kernel_size=None):
    populations = {
        "dataset": dataset,
        "layers": {
            str(seed): {m.name: om_raw(m.size, binding_seed(seed, dataset, m.name)) for m in layout}
            for seed in seeds
        },
    }
    if kernel_size:
        populations["kernel_fit"] = om_raw(kernel_size, 231001)
        populations["kernel_heldout"] = om_raw(kernel_size, 231002)
    return populations


def deployed(lc, network, target, case, populations, seed=271001):
    array = program_verify.fresh(lc, network.layout, seed, case, populations, "cpu")
    report = program_verify.program(array, target, lc)
    return array, report
