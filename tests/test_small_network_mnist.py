from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ebl.registry import RunMode, get_definition
from ebl.schema.base import ConfigError
import networks.data.small as components
from networks.data.small import build_data
from ebl.schema.small_network import parse_small_drn_config


EXAMPLE_PATH = Path("examples/small_drn/mnist_perfect_diode.json")


def _example_payload() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def test_mnist_example_resolves_full_differential_input_shape() -> None:
    document = parse_small_drn_config(_example_payload())
    spec = get_definition("small_drn.v1").resolve(document, RunMode.TRAIN)

    assert document.common.data.dataset == "mnist"
    assert document.common.model.dims == (1568, 100, 20)
    assert document.linspace is None
    assert spec.settings.algorithm == "backprop"
    assert spec.settings.nudging == 0.0


def test_mnist_rejects_non_784_logical_input_width() -> None:
    payload = _example_payload()
    payload["model"]["dims"][0] = 1536

    with pytest.raises(
        ConfigError,
        match=r"config.model.dims\[0\].*equal 1568",
    ):
        parse_small_drn_config(payload)


def test_mnist_rejects_linspace_mode() -> None:
    payload = _example_payload()
    payload["modes"]["linspace"] = {
        "minimum": -1.0,
        "maximum": 1.0,
        "samples": 3,
        "record_states": False,
    }

    with pytest.raises(
        ConfigError,
        match="only 'train' and/or 'validate' for dataset 'mnist'",
    ):
        parse_small_drn_config(payload)


def test_build_data_flattens_mnist_and_honors_num_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _example_payload()
    payload["runtime"]["device"] = "cpu"
    payload["data"]["num_points"] = 5
    document = parse_small_drn_config(payload)

    train_images = torch.arange(
        8 * 28 * 28,
        dtype=torch.float32,
    ).reshape(8, 1, 28, 28)
    train_targets = torch.arange(8) % 10
    test_images = torch.zeros(3, 1, 28, 28)
    test_targets = torch.arange(3)

    class FakeMnistDataset:
        def __init__(self, **_: object) -> None:
            pass

        def build(self):
            return (
                DataLoader(
                    TensorDataset(train_images, train_targets),
                    batch_size=2,
                ),
                DataLoader(
                    TensorDataset(test_images, test_targets),
                    batch_size=2,
                ),
            )

    monkeypatch.setattr(components, "MnistDataset", FakeMnistDataset)

    bundle = build_data(document.common)
    train_inputs, _ = next(iter(bundle.train_loader))
    test_inputs, _ = next(iter(bundle.held_out_loader))

    assert len(bundle.train_loader.dataset) == 5
    assert train_inputs.shape[1:] == (784,)
    assert test_inputs.shape[1:] == (784,)


def test_build_data_makes_seeded_stratified_validation_and_keeps_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _example_payload()
    payload["runtime"]["device"] = "cpu"
    payload["data"]["validation_points"] = 10
    document = parse_small_drn_config(payload)

    train_images = torch.zeros(30, 1, 28, 28)
    train_targets = torch.arange(30) % 10
    test_images = torch.ones(7, 1, 28, 28)
    test_targets = torch.arange(7) % 10

    class FakeMnistDataset:
        def __init__(self, **_: object) -> None:
            pass

        def build(self):
            return (
                DataLoader(
                    TensorDataset(train_images, train_targets),
                    batch_size=2,
                ),
                DataLoader(
                    TensorDataset(test_images, test_targets),
                    batch_size=2,
                ),
            )

    monkeypatch.setattr(components, "MnistDataset", FakeMnistDataset)
    first = build_data(document.common)
    second = build_data(document.common)

    assert len(first.train_loader.dataset) == 20
    assert len(first.held_out_loader.dataset) == 10
    assert len(first.test_loader.dataset) == 7
    assert first.train_loader.dataset.indices == second.train_loader.dataset.indices
    assert (
        first.held_out_loader.dataset.indices
        == second.held_out_loader.dataset.indices
    )
    held_out_labels = torch.tensor(
        [target for _, target in first.held_out_loader.dataset]
    )
    assert torch.bincount(held_out_labels, minlength=10).tolist() == [1] * 10
