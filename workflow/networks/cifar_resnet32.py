"""``cifar_resnet32_suffix``: ResNet-32 on CIFAR-10/100 with analog suffix convolutions.

The pretrained ResNet-32 stays complete. Its digital prefix is frozen; the
last ``analog_convolutions`` layer3 convolutions and the classifier become
crossbar matrices (``CrossbarSuffix``). The fixed class-balanced 45000/5000
split and its nested prefixes come from the existing CIFAR helpers. HWA
consumes raw training images (augmented every epoch); calibration and on-chip
training consume cached prefix features and teacher logits. Test images are
cached only for test-evaluated lifecycles.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.artifacts import sha256_file
from experiments.cifar_crossbar.data import (
    NORMALIZATION,
    cache_features,
    cohort_receipt,
    get_dataset,
    split_indices,
)
from experiments.cifar_crossbar.hwa_fault_runtime import augment_images
from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
from experiments.cifar_crossbar.prepare import SOURCES
from experiments.cifar_crossbar.runtime import evaluate as _evaluate
from experiments.cifar_crossbar.runtime import load
from workflow.lifecycle import DEVELOPMENT_SPLIT, TRAINING_SPLIT

FEATURE_SPLITS = ("training", "development", "test")
SUMMARY_METRICS = ("teacher_kl", "accuracy_percent")
PRESENTATIONS = "image_presentations"


def pinned_digest(network) -> str:
    return SOURCES[network.dataset][1]


def load_teacher(path, network, device):
    """Load the pinned public ResNet-32 checkpoint for the lifecycle dataset."""

    path = Path(path)
    digest = sha256_file(path)
    if not digest.startswith(SOURCES[network.dataset][1]):
        raise ValueError(
            "Expected the declared public pretrained ResNet-32 checkpoint digest "
            f"for {network.dataset}. Provided value: {digest}."
        )
    teacher = CifarResNet32(network.classes).to(device)
    teacher.load_state_dict(load(path), strict=True)
    teacher.eval().requires_grad_(False)
    return teacher, digest


def build_network(teacher, lifecycle) -> CrossbarSuffix:
    return CrossbarSuffix(teacher, lifecycle.network.suffix, lifecycle.deployment.tile_size)


# --- cohorts ------------------------------------------------------------------


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


def cohorts(cache: dict) -> dict:
    value = {name: cache[name]["cohort"] for name in FEATURE_SPLITS if name in cache}
    value["hwa"] = cache["hwa_cohort"]
    return value


def normalize_images(raw: torch.Tensor, dataset: str) -> torch.Tensor:
    x = raw.float() / 255
    mean, sd = NORMALIZATION[dataset]
    return (x - x.new_tensor(mean)[None, :, None, None]) / x.new_tensor(sd)[None, :, None, None]


def training_images(raw: torch.Tensor, dataset: str, generator, augment: bool) -> torch.Tensor:
    """Random crop/flip before normalization when the lifecycle augments."""

    return augment_images(raw, dataset, generator) if augment else normalize_images(raw, dataset)


# --- batches, objective and metrics --------------------------------------------


def evaluate(network, split: dict, device, *, q=None) -> dict:
    return _evaluate(network, split, device, q=q)


def examples(split: dict) -> int:
    return len(split["labels"])


def batch(network, split: dict, index, device):
    """Cached prefix features, teacher logits and labels of one minibatch."""

    return (
        split["features"][index].to(device),
        split["teacher_logits"][index].to(device),
        split["labels"][index].to(device),
    )


def objective(logits, teacher_logits, labels, name):
    if name == "teacher_kl":
        return F.kl_div(logits.log_softmax(1), teacher_logits.softmax(1), reduction="batchmean")
    return F.cross_entropy(logits, labels)


def hwa_inputs(cache: dict, device) -> dict:
    return {"raw": cache["hwa_raw"].to(device), "labels": cache["hwa_labels"].to(device)}


def hwa_examples(inputs: dict) -> int:
    return len(inputs["raw"])


def hwa_batch(inputs: dict, index, generator, network, teacher, lifecycle):
    """Raw images, augmented with the epoch generator, through the frozen prefix."""

    x = training_images(inputs["raw"][index], lifecycle.network.dataset, generator, lifecycle.hwa.augment)
    with torch.no_grad():
        teacher_logits = teacher(x) if lifecycle.hwa.objective == "teacher_kl" else None
        features = network.features(x)
    return features, teacher_logits, inputs["labels"][index]


def prepare_report(teacher, network, cache: dict, lifecycle, device) -> dict:
    """Digital suffix metrics; the full test split must reproduce the published model."""

    split = lifecycle.data.evaluation
    digital = evaluate(network, cache[split], device)
    teacher_accuracy = float(
        (cache[split]["teacher_logits"].argmax(1) == cache[split]["labels"]).float().mean() * 100
    )
    published = SOURCES[lifecycle.network.dataset][2]
    if split == "test" and not lifecycle.data.smoke and abs(teacher_accuracy - published) > 0.25:
        raise RuntimeError("Digital checkpoint accuracy does not reproduce the published source within 0.25 pp.")
    return {
        "digital": digital,
        "teacher_accuracy_percent": teacher_accuracy,
        "published_accuracy_percent": published,
    }


def deployment_summary(clean: dict, initial: dict) -> dict:
    return {"deployment_drop_pp": clean["accuracy_percent"] - initial["accuracy_percent"]}
