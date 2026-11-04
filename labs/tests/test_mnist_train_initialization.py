import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from networks.data.reference import _build_torchvision_image_loaders
from labs.mnist_train import _load_initial_parameters


class _Energy:
    def __init__(self, *parameters, all_parameters=None):
        self._parameters = list(parameters)
        if all_parameters is not None:
            self._params = list(all_parameters)

    def params(self):
        return self._parameters


def _parameter(state, *, lower=0.0, upper=1.0):
    return SimpleNamespace(
        state=state,
        min_cond=lower,
        max_cond=upper,
        _non_negative=True,
    )


def test_initial_parameter_loader_copies_exact_tensors_without_replacing_state(
    tmp_path,
):
    parameters = [
        _parameter(torch.zeros(2, 3)),
        _parameter(torch.zeros(3)),
    ]
    state_objects = [parameter.state for parameter in parameters]
    expected = [
        torch.linspace(0.1, 0.6, 6).reshape(2, 3),
        torch.tensor([0.2, 0.4, 0.8]),
    ]
    checkpoint = tmp_path / "model.pt"
    torch.save(expected, checkpoint)

    metadata = _load_initial_parameters(
        _Energy(*parameters),
        checkpoint,
        torch.device("cpu"),
    )

    assert all(
        parameter.state is state
        for parameter, state in zip(parameters, state_objects)
    )
    for parameter, target in zip(parameters, expected):
        torch.testing.assert_close(parameter.state, target, rtol=0.0, atol=0.0)
    assert metadata["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_initial_parameter_loader_validates_all_tensors_before_mutating(tmp_path):
    parameters = [
        _parameter(torch.full((2,), 0.25)),
        _parameter(torch.full((2,), 0.75)),
    ]
    before = [parameter.state.clone() for parameter in parameters]
    checkpoint = tmp_path / "bad_model.pt"
    torch.save([torch.full((2,), 0.5), torch.zeros(3)], checkpoint)

    with pytest.raises(ValueError, match="shape to match the DRN"):
        _load_initial_parameters(
            _Energy(*parameters),
            checkpoint,
            torch.device("cpu"),
        )

    for parameter, target in zip(parameters, before):
        torch.testing.assert_close(parameter.state, target, rtol=0.0, atol=0.0)


def test_initial_parameter_loader_includes_fixed_native_saved_parameters(tmp_path):
    trainable = _parameter(torch.zeros(2))
    fixed = _parameter(torch.zeros(3))
    expected_trainable = torch.tensor([0.2, 0.4])
    expected_fixed = torch.tensor([0.1, 0.3, 0.5])
    checkpoint = tmp_path / "model_with_fixed_parameter.pt"
    torch.save([expected_trainable, expected_fixed], checkpoint)

    metadata = _load_initial_parameters(
        _Energy(trainable, all_parameters=[trainable, fixed]),
        checkpoint,
        torch.device("cpu"),
    )

    torch.testing.assert_close(
        trainable.state,
        expected_trainable,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fixed.state,
        expected_fixed,
        rtol=0.0,
        atol=0.0,
    )
    assert [item["shape"] for item in metadata["parameters"]] == [[2], [3]]


class _SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, *, transform, **_kwargs):
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        image = np.full((2, 2), 255, dtype=np.uint8)
        return self.transform(image), 0


def test_mnist_normalization_scale_is_applied_after_standardization(tmp_path):
    train_loader, _ = _build_torchvision_image_loaders(
        dataset_cls=_SingleImageDataset,
        batch_size=1,
        root=tmp_path,
        download=False,
        normalize=True,
        normalize_mean=0.1307,
        normalize_std=0.3081,
        normalize_scale=0.3,
    )

    images, _ = next(iter(train_loader))
    expected = ((1.0 - 0.1307) / 0.3081) * 0.3
    torch.testing.assert_close(
        images,
        torch.full_like(images, expected),
        rtol=0.0,
        atol=1e-6,
    )
