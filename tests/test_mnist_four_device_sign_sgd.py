from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from ebl.registry import parse_experiment_config
from ebl.runtime.mnist_relu_drn.components import build_student_stack
from ebl.runtime.mnist_relu_drn.runtime import (
    _amplification_index_report,
    _single_topology_signature,
)
from ebl.schema.base import RunMode
from methods.sign_sgd import SignSGD


ROOT = Path(__file__).resolve().parents[1]
STANDARD = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_single_quad_common_window_finetune_10ep.json"
)
CONFIG_DIR = (
    ROOT
    / "examples/mnist_relu_drn/four_device_cohort_a_sign_sgd"
)
MATCHED = CONFIG_DIR / "matched_numeric_lr.json"
BALANCED = CONFIG_DIR / "balanced_0p21ns.json"


def _train(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition, document = parse_experiment_config(payload)
    return definition.resolve(document, RunMode.TRAIN)


def test_sign_sgd_configs_change_only_backend_and_declared_step_scale() -> None:
    standard = _train(STANDARD)
    matched = _train(MATCHED)
    balanced = _train(BALANCED)

    assert matched.runtime == balanced.runtime == standard.runtime
    assert matched.data == balanced.data == standard.data
    assert matched.model == balanced.model == standard.model
    assert matched.solver == balanced.solver == standard.solver
    assert matched.mapping == balanced.mapping == standard.mapping
    assert matched.settings.num_epochs == balanced.settings.num_epochs == 10
    assert matched.settings.update_backend.parameters == (
        balanced.settings.update_backend.parameters
    ) == standard.settings.update_backend.parameters
    assert matched.settings.update_backend.type == (
        balanced.settings.update_backend.type
    ) == "measured_cohort_a_sign_sgd"
    assert matched.settings.learning_rates == standard.settings.learning_rates
    assert balanced.settings.learning_rates == (2.1e-10, 2.1e-10)


def test_sign_sgd_backend_builds_sign_optimizer_and_is_model_local() -> None:
    spec = replace(
        _train(MATCHED),
        runtime=replace(_train(MATCHED).runtime, device="cpu"),
    )
    first = build_student_stack(spec, enable_measured=False)
    second = build_student_stack(spec, enable_measured=False)

    assert isinstance(first.optimizer, SignSGD)
    assert not isinstance(first.optimizer, torch.optim.SGD)
    expected = ((0, 1, 1.0, 1.0), (1, 2, 4.0, 0.0625))
    for stack in (first, second):
        assert _single_topology_signature(
            _amplification_index_report(stack),
            voltage_amp=4.0,
            current_amp=0.25,
        ) == expected


def test_sign_sgd_capability_fails_closed_outside_four_device_protocol() -> None:
    payload = json.loads(MATCHED.read_text(encoding="utf-8"))
    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    parameters["initial_target_mapping"] = "literal"
    parameters.pop("dual_rail_layout_by_parameter")

    with pytest.raises(ValueError, match="single-encoding four-device"):
        parse_experiment_config(payload)
