from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.definitions import parse_experiment_config
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_relu.runtime import (
    _device,
    _runtime_device_receipt,
    _teacher_acceptance_gate,
    _validate_request,
)
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "examples" / "mnist_relu" / "teacher.json"
V2_CONFIG = (
    ROOT / "examples" / "mnist_relu" / "teacher_784_256_10_cuda.json"
)


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_teacher_config_freezes_campaign_architecture_and_training() -> None:
    definition, document = parse_experiment_config(_payload(V2_CONFIG))
    train = definition.resolve(document, RunMode.TRAIN)
    validate = definition.resolve(document, RunMode.VALIDATE)

    assert definition.experiment_id == "mnist_relu.v2"
    assert train.experiment_id == "mnist_relu.v2"
    assert train.runtime.device == "cuda"
    assert train.runtime.seed == 42
    assert train.runtime.data_seed == 42
    assert train.data.batch_size == 16
    assert train.data.validation_points == 5000
    assert train.model.dims == (784, 256, 10)
    assert train.model.bias is False
    assert train.settings.num_epochs == 30
    assert train.settings.learning_rate == 1e-3
    assert train.settings.weight_decay == 0.0
    assert train.settings.minimum_validation_accuracy == 0.97
    assert validate.settings.split == "test"
    assert validate.settings.sample_limit is None


def test_v2_teacher_model_uses_configured_hidden_width_and_stable_keys() -> None:
    torch.manual_seed(29)
    teacher = BiasFreeReluTeacher(
        device=torch.device("cpu"),
        dims=(784, 256, 10),
    )

    assert teacher.dims == (784, 256, 10)
    assert tuple(teacher.input_weight.state.shape) == (784, 256)
    assert tuple(teacher.output_weight.state.shape) == (256, 10)
    assert tuple(binding.key for binding in teacher.catalog.checkpointed) == (
        "teacher.dense_weight.0",
        "teacher.dense_weight.1",
    )
    logits = teacher.logits(torch.zeros((3, 784), dtype=torch.float32))
    assert tuple(logits.shape) == (3, 10)


def test_v2_acceptance_gate_is_strictly_above_threshold() -> None:
    with pytest.raises(RuntimeError, match="strictly above 0.97"):
        _teacher_acceptance_gate(
            experiment_id="mnist_relu.v2",
            accuracy=0.97,
            threshold=0.97,
        )
    with pytest.raises(RuntimeError, match="strictly above 0.97"):
        _teacher_acceptance_gate(
            experiment_id="mnist_relu.v2",
            accuracy=0.969999,
            threshold=0.97,
        )

    assert _teacher_acceptance_gate(
        experiment_id="mnist_relu.v2",
        accuracy=0.970001,
        threshold=0.97,
    ) == {
        "minimum_validation_accuracy": 0.97,
        "comparison_operator": ">",
        "passed": True,
    }


def test_v1_keeps_fixed_width_and_inclusive_gate() -> None:
    v1_payload = _payload(V1_CONFIG)
    definition, document = parse_experiment_config(v1_payload)
    train = definition.resolve(document, RunMode.TRAIN)
    assert train.model.dims == (784, 50, 10)
    assert _teacher_acceptance_gate(
        experiment_id="mnist_relu.v1",
        accuracy=0.97,
        threshold=0.97,
    ) == {
        "minimum_validation_accuracy": 0.97,
        "passed": True,
    }

    v1_payload["model"]["dims"] = [784, 256, 10]
    with pytest.raises(ConfigError, match=r"\[784, 50, 10\]"):
        parse_experiment_config(v1_payload)


@pytest.mark.parametrize(
    "dims",
    (
        [783, 256, 10],
        [784, 256, 11],
        [784, 0, 10],
        [784, True, 10],
        [784, 256],
    ),
)
def test_v2_rejects_non_mnist_or_invalid_dimensions(dims: list[object]) -> None:
    payload = _payload(V2_CONFIG)
    payload["model"]["dims"] = dims
    with pytest.raises(ConfigError, match="dims"):
        parse_experiment_config(payload)


@pytest.mark.parametrize("name", ("device_state", "selection_receipt"))
def test_teacher_runtime_rejects_staged_crossbar_inputs(name: str) -> None:
    request = SimpleNamespace(
        spec=SimpleNamespace(experiment_id="mnist_relu.v2"),
        weights=None,
        base_weights=None,
        resume=None,
        device_data=None,
        teacher_weights=None,
        device_state=None,
        selection_receipt=None,
    )
    setattr(request, name, Path("wrong-artifact"))

    with pytest.raises(ValueError, match=name.replace("_", "-")):
        _validate_request(request, training=True)


def test_v2_cuda_runtime_device_receipt_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda device: f"mock-device-for-{device}",
    )

    configured_device = "cuda"
    resolved_device = _device(configured_device)
    receipt = _runtime_device_receipt(
        configured_device=configured_device,
        resolved_device=resolved_device,
    )

    assert resolved_device.type == "cuda"
    assert receipt == {
        "configured_device": "cuda",
        "resolved_device": "cuda",
        "cuda_available": True,
        "device_name": "mock-device-for-cuda",
    }
