"""Fixed CIFAR splits and nested calibration cohorts; no test-set selection."""

from __future__ import annotations
from hashlib import sha256
import os
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms

NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
}


def get_dataset(name, train, *, augment=False):
    root = Path(os.environ.get("EBL_CIFAR_ROOT", "data/cifar"))
    operations = []
    if augment:
        operations += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    operations += [transforms.ToTensor(), transforms.Normalize(*NORMALIZATION[name])]
    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    return cls(
        str(root), train=train, transform=transforms.Compose(operations), download=False
    )


def split_indices(labels, *, seed=20260917, validation_size=5000):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if validation_size % len(classes) or validation_size >= len(labels):
        raise ValueError(
            "Expected a class-balanced development split smaller than training data."
        )
    rng = np.random.default_rng(seed)
    validation, per_class = [], []
    for label in classes:
        ordered = rng.permutation(np.flatnonzero(labels == label))
        n = validation_size // len(classes)
        validation.append(ordered[:n])
        per_class.append(ordered[n:])
    # Interleaving gives class-balanced nested prefixes for every budget.
    training = np.stack(per_class, axis=1).reshape(-1).astype(np.int64)
    return training, np.stack(validation, axis=1).reshape(-1).astype(np.int64)


def cohort_receipt(indices):
    value = np.asarray(indices, dtype="<i8")
    return {"count": len(value), "indices_sha256": sha256(value.tobytes()).hexdigest()}


@torch.no_grad()
def cache_features(teacher, student, dataset, indices, *, batch_size=256, device="cpu"):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, list(map(int, indices))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    features, logits, labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        features.append(student.features(x).cpu())
        logits.append(teacher(x).cpu())
        labels.append(y)
    return {
        "features": torch.cat(features),
        "teacher_logits": torch.cat(logits),
        "labels": torch.cat(labels),
        "cohort": cohort_receipt(indices),
    }
