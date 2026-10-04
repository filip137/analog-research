from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.definitions import parse_experiment_config
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.schema import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]


def test_teacher_kaiming_initialization_uses_logical_fan_in() -> None:
    torch.manual_seed(19)
    teacher = BiasFreeReluTeacher(device=torch.device("cpu"))
    torch.manual_seed(19)
    expected_input = torch.empty((784, 50), dtype=torch.float32)
    expected_output = torch.empty((50, 10), dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(
        expected_input.T,
        a=0.0,
        nonlinearity="relu",
    )
    torch.nn.init.kaiming_uniform_(
        expected_output.T,
        a=0.0,
        nonlinearity="linear",
    )

    torch.testing.assert_close(teacher.input_weight.state, expected_input)
    torch.testing.assert_close(teacher.output_weight.state, expected_output)
    assert float(teacher.input_weight.state.abs().max()) <= (6.0 / 784.0) ** 0.5
    assert float(teacher.output_weight.state.abs().max()) <= (3.0 / 50.0) ** 0.5


@pytest.mark.parametrize(
    "relative",
    [
        "examples/mnist_relu/teacher.json",
    ],
)
def test_new_examples_resolve_train_and_validate(relative: str) -> None:
    payload = json.loads((ROOT / relative).read_text())
    definition, document = parse_experiment_config(payload)
    assert definition.resolve(document, RunMode.TRAIN).experiment_id == payload["experiment_id"]
    assert definition.resolve(document, RunMode.VALIDATE).experiment_id == payload["experiment_id"]
