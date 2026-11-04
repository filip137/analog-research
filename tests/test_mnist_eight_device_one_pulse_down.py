from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from ebl.registry import parse_experiment_config
from ebl.runtime.mnist_relu_drn.components import (
    build_student_stack,
    measured_optimizer_type,
)
from ebl.runtime.mnist_relu_drn.runtime import (
    _amplification_index_report,
    _differential_topology_signature,
)
from ebl.schema.base import RunMode
from methods.measured_trace import MeasuredCohortAOnePulseDownOptimizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/eight_device_cohort_a_one_pulse_down/"
    "isotonic_10ep.json"
)
HOLD_CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/eight_device_cohort_a_one_pulse_down/"
    "initialization_hold_1ep.json"
)
EXPECTED_KEYS = {
    "base.conductance_plus.0",
    "base.conductance_minus.0",
    "base.conductance_plus.1",
    "base.conductance_minus.1",
}


def _resolved(path: Path = CONFIG, payload: dict | None = None):
    raw = (
        json.loads(path.read_text(encoding="utf-8"))
        if payload is None
        else payload
    )
    definition, document = parse_experiment_config(raw)
    return definition.resolve(document, RunMode.TRAIN)


def test_eight_device_one_pulse_down_config_is_explicit() -> None:
    spec = _resolved()

    assert spec.model.encoding == "differential"
    assert spec.settings.learning_rates == (0.0, 0.0)
    assert spec.settings.update_backend.type == (
        "measured_cohort_a_one_pulse_down"
    )
    assert spec.settings.update_backend.parameters[
        "initial_target_mapping"
    ] == "paired_affine_common_window"
    assert spec.settings.update_backend.parameters[
        "curve_preprocessing"
    ] == "isotonic_nonincreasing"
    assert spec.settings.update_backend.parameters[
        "positive_gradient_threshold_by_parameter"
    ] is None
    assert measured_optimizer_type(spec.settings.update_backend.type) is (
        MeasuredCohortAOnePulseDownOptimizer
    )


def test_eight_device_threshold_keys_and_mapping_fail_closed() -> None:
    hold = json.loads(HOLD_CONFIG.read_text(encoding="utf-8"))
    spec = _resolved(HOLD_CONFIG)
    thresholds = dict(
        spec.settings.update_backend.parameters[
            "positive_gradient_threshold_by_parameter"
        ]
    )
    assert set(thresholds) == EXPECTED_KEYS
    assert set(thresholds.values()) == {1e30}

    missing = deepcopy(hold)
    del missing["modes"]["train"]["update_backend"]["parameters"][
        "positive_gradient_threshold_by_parameter"
    ]["base.conductance_minus.1"]
    with pytest.raises(ValueError, match="map exactly the stable parameter"):
        _resolved(payload=missing)

    wrong_mapping = deepcopy(hold)
    parameters = wrong_mapping["modes"]["train"]["update_backend"][
        "parameters"
    ]
    parameters["initial_target_mapping"] = "dual_rail_quad_common_window"
    parameters["dual_rail_layout_by_parameter"] = {
        "base.dense_weight.0": "halves",
        "base.dense_weight.1": "paired",
    }
    parameters["positive_gradient_threshold_by_parameter"] = {
        "base.dense_weight.0": 1.0,
        "base.dense_weight.1": 1.0,
    }
    with pytest.raises(ValueError, match="single model encoding"):
        _resolved(payload=wrong_mapping)


def test_eight_device_topology_is_process_order_independent() -> None:
    spec = _resolved()
    cpu_spec = replace(spec, runtime=replace(spec.runtime, device="cpu"))

    first = build_student_stack(cpu_spec, enable_measured=False)
    second = build_student_stack(cpu_spec, enable_measured=False)
    expected = ((0, 1, 1.0, 1.0), (1, 2, 4.0, 0.0625))
    assert _differential_topology_signature(
        _amplification_index_report(first)
    ) == expected
    assert _differential_topology_signature(
        _amplification_index_report(second)
    ) == expected
