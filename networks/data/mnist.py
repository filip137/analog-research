"""Shared MNIST loading and teacher-supervised batches for the DRN experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms



class FlattenedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image.reshape(-1), label


@dataclass(frozen=True)
class MnistLoaders:
    train: Iterable
    validation: Iterable
    test: Iterable
    calibration: Iterable
    train_generator: torch.Generator


def _stratified_indices(
    targets: torch.Tensor,
    *,
    validation_points: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    labels = targets.detach().cpu().numpy().astype(np.int64, copy=False)
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    classes = sorted(int(item) for item in np.unique(labels))
    base, remainder = divmod(validation_points, len(classes))
    for class_index, label in enumerate(classes):
        indices = np.flatnonzero(labels == label)
        take = base + (1 if class_index < remainder else 0)
        if take >= len(indices):
            raise ValueError(
                "Expected every MNIST class to retain training examples after "
                f"the validation split. Provided value: class={label}, "
                f"available={len(indices)}, holdout={take}."
            )
        shuffled = generator.permutation(indices)
        validation.extend(int(item) for item in shuffled[:take])
        train.extend(int(item) for item in shuffled[take:])
    return sorted(train), sorted(validation)


def build_mnist_loaders(
    data,
    *,
    data_seed: int,
    calibration_examples: int = 1024,
    calibration_batch_size: int | None = None,
) -> MnistLoaders:
    root = Path(
        os.environ.get(
            "EBL_MNIST_ROOT",
            str(Path.home() / "datasets" / "mnist"),
        )
    ).expanduser()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3,)),
        ]
    )
    try:
        raw_train = datasets.MNIST(
            root=str(root),
            train=True,
            download=False,
            transform=transform,
        )
        raw_test = datasets.MNIST(
            root=str(root),
            train=False,
            download=False,
            transform=transform,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Expected an existing torchvision MNIST dataset root containing "
            f"MNIST/raw or MNIST/processed. Provided root: {root}"
        ) from error

    train_indices, validation_indices = _stratified_indices(
        raw_train.targets,
        validation_points=int(data.validation_points),
        seed=int(data_seed),
    )
    flattened_train = FlattenedDataset(raw_train)
    train_dataset: Dataset = Subset(flattened_train, train_indices)
    if data.num_points is not None:
        if data.num_points > len(train_dataset):
            raise ValueError(
                "Expected config.data.num_points not to exceed the MNIST "
                f"training split. Provided value: {data.num_points}."
            )
        subset_generator = torch.Generator().manual_seed(data_seed)
        selected = torch.randperm(
            len(train_dataset),
            generator=subset_generator,
        )[: data.num_points].tolist()
        train_dataset = Subset(train_dataset, selected)

    train_generator = torch.Generator().manual_seed(data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=data.batch_size,
        shuffle=data.shuffle,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        Subset(flattened_train, validation_indices),
        batch_size=data.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        FlattenedDataset(raw_test),
        batch_size=data.batch_size,
        shuffle=False,
    )

    available_calibration = min(calibration_examples, len(train_indices))
    if available_calibration < calibration_examples:
        raise ValueError(
            "Expected the training split to contain at least the configured "
            f"calibration examples. Provided value: available={len(train_indices)}, "
            f"configured={calibration_examples}."
        )
    calibration_loader = DataLoader(
        Subset(flattened_train, train_indices[:available_calibration]),
        batch_size=calibration_batch_size or data.batch_size,
        shuffle=False,
    )
    return MnistLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        calibration=calibration_loader,
        train_generator=train_generator,
    )


def limited(loader: Iterable, maximum_batches: int | None):
    if maximum_batches is None:
        return loader
    from itertools import islice

    return islice(loader, maximum_batches)






__all__ = [
    "MnistLoaders",
    "build_mnist_loaders",
    "limited",
]
