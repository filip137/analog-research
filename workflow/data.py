"""Fixed CIFAR cohorts and cached frozen-prefix features for every stage.

The class-balanced 45000/5000 split and its nested prefixes come from the
existing CIFAR helpers. HWA consumes raw training images (augmented every
epoch); calibration and on-chip training consume cached prefix features and
teacher logits. Test images are cached only for test-evaluated lifecycles.
"""

from __future__ import annotations

import torch

from experiments.cifar_crossbar.data import (
    NORMALIZATION,
    cache_features,
    cohort_receipt,
    get_dataset,
    split_indices,
)
from experiments.cifar_crossbar.hwa_fault_runtime import augment_images
from workflow.lifecycle import DEVELOPMENT_SPLIT, TRAINING_SPLIT

FEATURE_SPLITS = ("training", "development", "test")


def evaluation_split(lifecycle) -> str:
    return lifecycle.data.evaluation


def build_cache(teacher, network, lifecycle, device) -> dict:
    """Materialize every cohort a lifecycle reads, truncated for smoke runs."""

    data, dataset = lifecycle.data, lifecycle.network.dataset
    training_set = get_dataset(dataset, True)
    training, development = split_indices(training_set.targets, seed=data.data_seed)
    if (
        len(training) != TRAINING_SPLIT
        or len(development) != DEVELOPMENT_SPLIT
        or set(training) & set(development)
    ):
        raise ValueError("Expected a disjoint 45000/5000 CIFAR training/development split.")
    hwa = training[: data.cohort_size("hwa")]
    recovery = training[: data.cohort_size("training")]
    selection = development[: data.cohort_size("development")]
    cache = {
        "hwa_raw": torch.from_numpy(training_set.data[hwa].copy()).permute(0, 3, 1, 2).contiguous(),
        "hwa_labels": torch.tensor(training_set.targets)[torch.from_numpy(hwa)],
        "hwa_cohort": cohort_receipt(hwa),
        "training": cache_features(teacher, network, training_set, recovery, device=device),
        "development": cache_features(teacher, network, training_set, selection, device=device),
    }
    if data.evaluation == "test":
        test_set = get_dataset(dataset, False)
        cache["test"] = cache_features(
            teacher, network, test_set, range(data.cohort_size("test")), device=device
        )
    return cache


def check_cache(cache: dict, lifecycle) -> None:
    """Reject caches whose cohorts differ from the lifecycle's data contract."""

    data = lifecycle.data
    expected = {
        "training": data.cohort_size("training"),
        "development": data.cohort_size("development"),
    }
    if data.evaluation == "test":
        expected["test"] = data.cohort_size("test")
    for split, size in expected.items():
        if split not in cache or len(cache[split]["labels"]) != size:
            raise ValueError(f"Expected the {split} cohort to contain {size} examples.")
    if len(cache["hwa_labels"]) != data.cohort_size("hwa") or len(cache["hwa_raw"]) != data.cohort_size("hwa"):
        raise ValueError("Expected the HWA image cohort declared by the lifecycle.")


def to_device(cache: dict, device) -> dict:
    """Move cached tensors once; batch slicing then stays on the device."""

    moved = dict(cache)
    for split in FEATURE_SPLITS:
        if split in cache:
            moved[split] = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in cache[split].items()
            }
    return moved


def normalize_images(raw: torch.Tensor, dataset: str) -> torch.Tensor:
    x = raw.float() / 255
    mean, sd = NORMALIZATION[dataset]
    return (x - x.new_tensor(mean)[None, :, None, None]) / x.new_tensor(sd)[None, :, None, None]


def training_images(raw: torch.Tensor, dataset: str, generator, augment: bool) -> torch.Tensor:
    """Random crop/flip before normalization when the lifecycle augments."""

    return augment_images(raw, dataset, generator) if augment else normalize_images(raw, dataset)
