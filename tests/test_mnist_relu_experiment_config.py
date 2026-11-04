from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ebl.registry import parse_experiment_config
from networks.teachers.mnist import BiasFreeReluTeacher
from ebl.schema.base import ConfigError, RunMode


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
        "examples/mnist_relu_drn/ideal_single.json",
        "examples/mnist_relu_drn/ideal_differential.json",
        "examples/mnist_relu_drn/ideal_differential_general_ab.json",
        "examples/mnist_relu_drn/measured_raw_single.json",
        "examples/mnist_relu_drn/measured_raw_differential.json",
        "examples/mnist_relu_drn/measured_raw_single_pairwise_common_window_finetune_10ep.json",
        "examples/mnist_relu_drn/ideal_single_centered_70_90us_finetune_10ep.json",
    ],
)
def test_new_examples_resolve_train_and_validate(relative: str) -> None:
    payload = json.loads((ROOT / relative).read_text())
    definition, document = parse_experiment_config(payload)
    assert definition.resolve(document, RunMode.TRAIN).experiment_id == payload["experiment_id"]
    assert definition.resolve(document, RunMode.VALIDATE).experiment_id == payload["experiment_id"]


def test_student_rejects_unknown_nested_keys() -> None:
    payload = json.loads(
        (ROOT / "examples/mnist_relu_drn/ideal_single.json").read_text()
    )
    payload["mapping"]["surprise"] = 1
    with pytest.raises(ConfigError, match="Expected config.mapping"):
        parse_experiment_config(payload)


def test_student_requires_explicit_empty_diode_parameter_dicts() -> None:
    payload = json.loads(
        (ROOT / "examples/mnist_relu_drn/ideal_single.json").read_text()
    )
    del payload["model"]["non_linearity"]["quadratic_diode_param"]
    with pytest.raises(ConfigError, match="quadratic_diode_param"):
        parse_experiment_config(payload)


def test_student_accepts_arbitrary_positive_finite_amplifier_magnitudes() -> None:
    payload = json.loads(
        (
            ROOT
            / "examples/mnist_relu_drn/ideal_differential_general_ab.json"
        ).read_text()
    )
    definition, document = parse_experiment_config(payload)
    train_spec = definition.resolve(document, RunMode.TRAIN)
    validate_spec = definition.resolve(document, RunMode.VALIDATE)

    assert train_spec.model.voltage_amp == 3.0
    assert train_spec.model.current_amp == 2.0
    assert validate_spec.model.voltage_amp == 3.0
    assert validate_spec.model.current_amp == 2.0


def test_student_keeps_legacy_single_device_amplifier_contract() -> None:
    payload = json.loads(
        (ROOT / "examples/mnist_relu_drn/ideal_single.json").read_text()
    )
    payload["model"]["voltage_amp"] = 3.0
    payload["model"]["current_amp"] = 2.0
    with pytest.raises(ConfigError, match="legacy single-device encoding"):
        parse_experiment_config(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("voltage_amp", 0.0),
        ("voltage_amp", -1.0),
        ("voltage_amp", float("inf")),
        ("voltage_amp", float("nan")),
        ("current_amp", 0.0),
        ("current_amp", -1.0),
        ("current_amp", float("inf")),
        ("current_amp", float("nan")),
    ],
)
def test_student_rejects_nonpositive_or_nonfinite_amplifier_magnitude(
    field: str,
    value: float,
) -> None:
    payload = json.loads(
        (ROOT / "examples/mnist_relu_drn/ideal_differential.json").read_text()
    )
    payload["model"][field] = value
    with pytest.raises(ConfigError, match=field):
        parse_experiment_config(payload)


def test_paired_affine_mapping_requires_differential_encoding() -> None:
    payload = json.loads(
        (ROOT / "examples/mnist_relu_drn/measured_raw_single.json").read_text()
    )
    payload["modes"]["train"]["update_backend"]["parameters"][
        "initial_target_mapping"
    ] = "paired_affine_common_window"

    with pytest.raises(ConfigError, match="differential model encoding"):
        parse_experiment_config(payload)


def test_dual_rail_pairwise_mapping_requires_single_encoding() -> None:
    payload = json.loads(
        (
            ROOT
            / "examples/mnist_relu_drn/"
            "measured_raw_single_pairwise_common_window_finetune_10ep.json"
        ).read_text()
    )
    payload["model"]["encoding"] = "differential"

    with pytest.raises(ConfigError, match="single model encoding"):
        parse_experiment_config(payload)


def test_dual_rail_pairwise_mapping_requires_model_local_layout_keys() -> None:
    payload = json.loads(
        (
            ROOT
            / "examples/mnist_relu_drn/"
            "measured_raw_single_pairwise_common_window_finetune_10ep.json"
        ).read_text()
    )
    layouts = payload["modes"]["train"]["update_backend"]["parameters"][
        "dual_rail_layout_by_parameter"
    ]
    layouts["base.dense_weight.0"] = "paired"

    with pytest.raises(ConfigError, match="model-local dual-rail layouts"):
        parse_experiment_config(payload)
