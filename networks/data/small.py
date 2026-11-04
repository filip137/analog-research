from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import os
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from networks.data.reference import DigitsDataset, MnistDataset, MoonsDataset, YinYangDataset
from networks.resistive.small import _resolve_device


class _FlattenedImageDataset(Dataset):
    """Flatten torchvision images while preserving labels and optional indices."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        return (sample[0].reshape(-1), *sample[1:])

@dataclass(frozen=True)
class DataBundle:
    train_loader: Iterable[Any]
    held_out_loader: Iterable[Any]
    test_loader: Iterable[Any]
    dataloader_generators: Mapping[str, torch.Generator]

def build_data(common: Any) -> DataBundle:
    """Build legacy datasets and an explicitly seeded training sampler."""

    device = _resolve_device(common.runtime.device)
    logical_input_dim = common.model.dims[0] // 2
    data_seed = (
        common.runtime.data_seed
        if common.runtime.data_seed is not None
        else common.runtime.seed
    )
    if common.data.dataset in {"moons", "yinyang"}:
        if common.data.num_points is None:
            raise ValueError(
                "Expected config.data.num_points to be an integer for "
                f"dataset {common.data.dataset!r}. Provided value: None."
            )

    if common.data.dataset == "moons":
        dataset = MoonsDataset(
            name="moons",
            batch_size=common.data.batch_size,
            device=device,
            num_samples=common.data.num_points,
            input_dim=logical_input_dim,
        )
    elif common.data.dataset == "yinyang":
        dataset = YinYangDataset(
            name="yinyang",
            batch_size=common.data.batch_size,
            device=device,
            num_samples=common.data.num_points,
            seed=42 if data_seed is None else data_seed,
        )
    elif common.data.dataset == "digits":
        dataset = DigitsDataset(
            name="digits",
            batch_size=common.data.batch_size,
            device=device,
            num_samples=common.data.num_points,
            seed=0 if data_seed is None else data_seed,
        )
    elif common.data.dataset == "mnist":
        mnist_root = Path(
            os.environ.get(
                "EBL_MNIST_ROOT",
                str(Path.home() / "datasets" / "mnist"),
            )
        ).expanduser()
        dataset = MnistDataset(
            name="mnist",
            batch_size=common.data.batch_size,
            device=torch.device("cpu"),
            root=str(mnist_root),
            train=True,
            download=False,
            normalize=True,
            normalize_mean=0.1307,
            normalize_std=0.3,
        )
    else:  # pragma: no cover - rejected by the pure schema
        raise ValueError(
            "Expected config.data.dataset to be 'moons', 'yinyang', "
            f"'digits', or 'mnist'. Provided value: {common.data.dataset!r}."
        )

    try:
        legacy_train, held_out = dataset.build()
    except RuntimeError as exc:
        if common.data.dataset != "mnist":
            raise
        raise RuntimeError(
            "Expected an existing torchvision MNIST dataset root containing "
            f"MNIST/raw or MNIST/processed. Provided root: {mnist_root}"
        ) from exc

    train_dataset = legacy_train.dataset
    test_loader = held_out
    if common.data.dataset == "mnist":
        raw_train_dataset = train_dataset
        flattened_train = _FlattenedImageDataset(raw_train_dataset)
        test_loader = DataLoader(
            _FlattenedImageDataset(held_out.dataset),
            batch_size=common.data.batch_size,
            shuffle=False,
        )
        if common.data.validation_points is None:
            train_dataset = flattened_train
            held_out = test_loader
        else:
            train_indices, validation_indices = _stratified_split_indices(
                raw_train_dataset,
                validation_points=common.data.validation_points,
                seed=0 if data_seed is None else data_seed,
            )
            train_dataset = Subset(flattened_train, train_indices)
            held_out = DataLoader(
                Subset(flattened_train, validation_indices),
                batch_size=common.data.batch_size,
                shuffle=False,
            )
        if common.data.num_points is not None:
            if common.data.num_points > len(train_dataset):
                raise ValueError(
                    "Expected config.data.num_points to be no larger than the "
                    "available MNIST training split after validation holdout. "
                    f"Provided value: {common.data.num_points}."
                )
            subset_generator = torch.Generator()
            subset_generator.manual_seed(0 if data_seed is None else data_seed)
            indices = torch.randperm(
                len(train_dataset),
                generator=subset_generator,
            )[: common.data.num_points].tolist()
            train_dataset = Subset(train_dataset, indices)

    generator = torch.Generator()
    if data_seed is None:
        generator.seed()
    else:
        generator.manual_seed(data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=common.data.batch_size,
        shuffle=common.data.shuffle,
        generator=generator,
    )
    return DataBundle(
        train_loader=train_loader,
        held_out_loader=held_out,
        test_loader=test_loader,
        dataloader_generators={"train": generator},
    )

def _stratified_split_indices(
    dataset: Dataset,
    *,
    validation_points: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic class-stratified train/validation indices."""

    raw_targets = getattr(dataset, "targets", None)
    if raw_targets is None and hasattr(dataset, "tensors"):
        tensors = getattr(dataset, "tensors")
        if isinstance(tensors, (tuple, list)) and len(tensors) >= 2:
            raw_targets = tensors[1]
    if raw_targets is None:
        raw_targets = [dataset[index][1] for index in range(len(dataset))]
    targets = np.asarray(
        raw_targets.detach().cpu().numpy()
        if isinstance(raw_targets, torch.Tensor)
        else raw_targets,
        dtype=np.int64,
    ).reshape(-1)
    if targets.size != len(dataset):
        raise ValueError(
            "Expected MNIST target count to match the training dataset. "
            f"Provided value: targets={targets.size}, dataset={len(dataset)}."
        )
    if validation_points >= targets.size:
        raise ValueError(
            "Expected config.data.validation_points to be smaller than the "
            f"MNIST training set ({targets.size}). Provided value: "
            f"{validation_points}."
        )
    labels = sorted(int(value) for value in np.unique(targets))
    if not labels:
        raise ValueError(
            "Expected MNIST training targets to contain at least one class. "
            "Provided value: empty."
        )
    base, remainder = divmod(validation_points, len(labels))
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for class_index, label in enumerate(labels):
        indices = np.flatnonzero(targets == label)
        take = base + (1 if class_index < remainder else 0)
        if take >= indices.size:
            raise ValueError(
                "Expected every MNIST class to retain at least one training "
                "example after the stratified validation split. Provided "
                f"value: class={label}, available={indices.size}, holdout={take}."
            )
        shuffled = generator.permutation(indices)
        validation.extend(int(value) for value in shuffled[:take])
        train.extend(int(value) for value in shuffled[take:])
    # Sort the fixed subsets; only the explicit training DataLoader controls
    # epoch order, so split membership and shuffle order stay independent.
    return sorted(train), sorted(validation)
