from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from ebl.registry import parse_experiment_config
from ebl.runtime.mnist_relu_drn.components import (
    build_student_stack,
    measured_optimizer_type,
)
from ebl.runtime.mnist_relu_drn.runtime import (
    _amplification_index_report,
    _single_topology_signature,
)
from ebl.schema.base import RunMode
from networks.resistive.builders import ParameterBinding, ParameterCatalog
from networks.variable.parameter import DenseWeight
from methods.measured_trace import MeasuredCohortAOnePulseDownOptimizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/four_device_cohort_a_one_pulse_down/"
    "isotonic_10ep.json"
)


def _write_device_data(path: Path) -> None:
    resistance = {
        "R00_00": [10000, 12000, 11000, 16000, 18000],
        "R00_00_bar": [11000, 13000, 12500, 17000, 19000],
        "R00_01": [10500, 14000, 11500, 18000, 20000],
        "R00_01_bar": [11500, 15000, 13000, 19000, 21000],
        "R01_00": [12000, 16000, 13500, 20000, 22000],
        "R01_00_bar": [12500, 17000, 14000, 21000, 23000],
        "R01_01": [13000, 18000, 14500, 22000, 24000],
        "R01_01_bar": [3_000_000, 3_100_000, 2_900_000, 3_200_000, 3_000_000],
    }
    with h5py.File(path, "w") as handle:
        for name, values in resistance.items():
            dataset = handle.create_dataset(
                name,
                data=np.asarray(values, dtype=np.float32),
            )
            row, col = name.removesuffix("_bar").removeprefix("R").split("_")
            dataset.attrs["row"] = int(row)
            dataset.attrs["col"] = int(col)
            dataset.attrs["bar"] = name.endswith("_bar")


def _parameters(
    thresholds: dict[str, float] | None = None,
) -> dict:
    result = {
        "curve_preprocessing": "isotonic_nonincreasing",
        "split_seed": 42,
        "assignment_seed": 42,
        "formed_resistance_max_ohm": 30000.0,
        "cohort_fraction": 0.5,
        "cohort": "A",
        "source_traces_per_cell": 2,
        "initial_pulse_index": 0,
        "projection": "global_nearest",
        "expected_trace_length": 5,
        "initial_target_mapping": "dual_rail_quad_common_window",
        "dual_rail_layout_by_parameter": {
            "base.dense_weight.0": "halves"
        },
        "programming_deadband_mode": "none",
        "programming_deadband_relative": 0.0,
        "probabilistic_write_mode": "none",
        "probabilistic_write_probability": 1.0,
        "probabilistic_write_scale_relative": 0.0,
        "probabilistic_write_seed": 0,
    }
    if thresholds is not None:
        result["positive_gradient_threshold_by_parameter"] = thresholds
    return result


def _optimizer(
    path: Path,
    thresholds: dict[str, float] | None = None,
):
    weight = DenseWeight(
        (4,),
        (4,),
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=1.1e-4,
    )
    binding = ParameterBinding(
        key="base.dense_weight.0",
        parameter=weight,
        role="dense_weight",
    )
    catalog = ParameterCatalog((binding,))
    logical = torch.optim.SGD(
        [{"params": weight.state, "lr": 123.0}],
        lr=123.0,
    )
    optimizer = MeasuredCohortAOnePulseDownOptimizer(
        logical,
        catalog,
        _parameters(thresholds),
        path,
    )
    with torch.no_grad():
        weight.state.copy_(
            torch.linspace(
                2.0e-5,
                1.0e-4,
                weight.state.numel(),
            ).reshape(weight.state.shape)
        )
    optimizer.initialize_from_reset_targets()
    return weight, optimizer


def _resolved_train(payload: dict | None = None):
    raw = (
        json.loads(CONFIG.read_text(encoding="utf-8"))
        if payload is None
        else payload
    )
    definition, document = parse_experiment_config(raw)
    return definition.resolve(document, RunMode.TRAIN)


def test_one_pulse_down_config_is_explicit_and_fail_closed() -> None:
    spec = _resolved_train()

    assert spec.settings.update_backend.type == (
        "measured_cohort_a_one_pulse_down"
    )
    assert spec.settings.learning_rates == (0.0, 0.0)
    assert spec.settings.update_backend.parameters["curve_preprocessing"] == (
        "isotonic_nonincreasing"
    )
    assert spec.settings.update_backend.parameters[
        "positive_gradient_threshold_by_parameter"
    ] is None
    assert measured_optimizer_type(spec.settings.update_backend.type) is (
        MeasuredCohortAOnePulseDownOptimizer
    )

    nonzero = json.loads(CONFIG.read_text(encoding="utf-8"))
    nonzero["modes"]["train"]["learning_rates"] = [1e-9, 0.0]
    with pytest.raises(ValueError, match="ignores learning-rate magnitude"):
        _resolved_train(nonzero)

    raw_curves = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw_curves["modes"]["train"]["update_backend"]["parameters"][
        "curve_preprocessing"
    ] = "raw"
    with pytest.raises(ValueError, match="isotonic_nonincreasing"):
        _resolved_train(raw_curves)

    wrong_mapping = json.loads(CONFIG.read_text(encoding="utf-8"))
    parameters = wrong_mapping["modes"]["train"]["update_backend"][
        "parameters"
    ]
    parameters["initial_target_mapping"] = "literal"
    parameters.pop("dual_rail_layout_by_parameter")
    with pytest.raises(ValueError, match="dual_rail_quad_common_window"):
        _resolved_train(wrong_mapping)

    thresholded = json.loads(CONFIG.read_text(encoding="utf-8"))
    thresholded["modes"]["train"]["update_backend"]["parameters"][
        "positive_gradient_threshold_by_parameter"
    ] = {
        "base.dense_weight.0": 494.015869140625,
        "base.dense_weight.1": 5849.4638671875,
    }
    thresholded_spec = _resolved_train(thresholded)
    assert dict(
        thresholded_spec.settings.update_backend.parameters[
            "positive_gradient_threshold_by_parameter"
        ]
    ) == {
        "base.dense_weight.0": 494.015869140625,
        "base.dense_weight.1": 5849.4638671875,
    }

    missing_key = deepcopy(thresholded)
    del missing_key["modes"]["train"]["update_backend"]["parameters"][
        "positive_gradient_threshold_by_parameter"
    ]["base.dense_weight.1"]
    with pytest.raises(ValueError, match="map exactly the stable parameter"):
        _resolved_train(missing_key)

    negative = deepcopy(thresholded)
    negative["modes"]["train"]["update_backend"]["parameters"][
        "positive_gradient_threshold_by_parameter"
    ]["base.dense_weight.0"] = -1.0
    with pytest.raises(ValueError, match="finite number >= 0.0"):
        _resolved_train(negative)


def test_one_pulse_down_model_construction_is_process_order_independent() -> None:
    spec = _resolved_train()
    cpu_spec = replace(
        spec,
        runtime=replace(spec.runtime, device="cpu"),
    )

    first = build_student_stack(cpu_spec, enable_measured=False)
    second = build_student_stack(cpu_spec, enable_measured=False)
    expected = ((0, 1, 1.0, 1.0), (1, 2, 4.0, 0.0625))
    for stack in (first, second):
        assert _single_topology_signature(
            _amplification_index_report(stack),
            voltage_amp=4.0,
            current_amp=0.25,
        ) == expected


def test_one_pulse_down_moves_only_requested_cells_by_one_and_resumes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(path)
    key = "base.dense_weight.0"

    initial_pulses = optimizer._plant.pulse_indices[key].to(torch.long).clone()
    initial = weight.state.detach().clone()
    gradient = torch.tensor(
        [1.0, -1.0, 0.0, 2.0] * 4,
        dtype=weight.state.dtype,
    ).reshape(weight.state.shape)
    weight.state.grad = gradient.clone()
    optimizer.step()

    expected_applied = (gradient.reshape(-1) > 0.0) & (initial_pulses < 4)
    assert torch.equal(
        optimizer._plant.pulse_indices[key].to(torch.long),
        initial_pulses + expected_applied.to(torch.long),
    )
    assert bool((weight.state <= initial).all())
    assert torch.equal(optimizer._shadows[key], weight.state)
    assert torch.equal(optimizer._last_programmed_shadows[key], weight.state)
    report = optimizer.programming_report
    parameter_report = report["parameters"][key]
    assert report["pulse_model"] is True
    assert report["global_nearest_fine_tuning"] is False
    assert report["learning_rate_magnitude_used"] is False
    assert parameter_report["max_abs_pulse_jump"] == 1
    assert parameter_report["conductance_increase_fraction"] == 0.0
    assert parameter_report["one_pulse_down_invariant_passed"] is True

    checkpoint = deepcopy(optimizer.state_dict())
    restored_weight, restored = _optimizer(path)
    with torch.no_grad():
        restored_weight.state.copy_(weight.state)
    restored.load_state_dict(checkpoint)

    next_gradient = torch.linspace(
        -2.0,
        2.0,
        weight.state.numel(),
    ).reshape(weight.state.shape)
    weight.state.grad = next_gradient.clone()
    restored_weight.state.grad = next_gradient.clone()
    optimizer.step()
    restored.step()
    assert torch.equal(restored_weight.state, weight.state)
    assert restored.programming_report == optimizer.programming_report


def test_one_pulse_down_saturates_at_last_pulse_without_increase(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(path)
    key = "base.dense_weight.0"
    previous = weight.state.detach().clone()

    for _ in range(6):
        weight.state.grad = torch.ones_like(weight.state)
        optimizer.step()
        assert bool((weight.state <= previous).all())
        previous = weight.state.detach().clone()

    assert bool((optimizer._plant.pulse_indices[key] == 4).all())
    saturated = weight.state.detach().clone()
    weight.state.grad = torch.ones_like(weight.state)
    optimizer.step()
    assert torch.equal(weight.state, saturated)
    report = optimizer.programming_report["parameters"][key]
    assert report["current_last_pulse_fraction"] == 1.0
    assert report["down_request_blocked_at_last_pulse_fraction"] > 0.0
    assert report["max_abs_pulse_jump"] == 1


def test_one_pulse_down_uses_strict_per_parameter_gradient_threshold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    key = "base.dense_weight.0"
    threshold = 0.5
    weight, optimizer = _optimizer(path, {key: threshold})
    initial_pulses = optimizer._plant.pulse_indices[key].to(torch.long).clone()
    gradient = torch.tensor(
        [1.0, 0.5, 0.5001, 0.1, -1.0, 2.0, 0.0, 0.49] * 2,
        dtype=weight.state.dtype,
    ).reshape(weight.state.shape)
    weight.state.grad = gradient.clone()
    optimizer.step()

    eligible = gradient.reshape(-1) > threshold
    expected_applied = eligible & (initial_pulses < 4)
    assert torch.equal(
        optimizer._plant.pulse_indices[key].to(torch.long),
        initial_pulses + expected_applied.to(torch.long),
    )
    report = optimizer.programming_report
    parameter_report = report["parameters"][key]
    assert report["gradient_gate"] == (
        "raw_gradient_strictly_greater_than_parameter_threshold"
    )
    assert report["positive_gradient_threshold_by_parameter"] == {
        key: threshold
    }
    assert parameter_report["positive_gradient_threshold"] == threshold
    assert parameter_report["threshold_eligible_fraction"] == pytest.approx(
        float(eligible.to(torch.float64).mean().item())
    )
    suppressed = (gradient > 0.0) & ~eligible.reshape(gradient.shape)
    assert parameter_report[
        "positive_gradient_suppressed_by_threshold_fraction"
    ] == pytest.approx(
        float(suppressed.to(torch.float64).mean().item())
    )

    checkpoint = deepcopy(optimizer.state_dict())
    restored_weight, restored = _optimizer(path, {key: threshold})
    with torch.no_grad():
        restored_weight.state.copy_(weight.state)
    restored.load_state_dict(checkpoint)
    assert restored.programming_report == optimizer.programming_report

    _changed_weight, changed = _optimizer(path, {key: threshold + 0.1})
    with pytest.raises(ValueError, match="configuration to match"):
        changed.load_state_dict(checkpoint)


def test_one_pulse_down_rejects_non_teacher_mapped_initialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    _weight, optimizer = _optimizer(path)

    with pytest.raises(RuntimeError, match="reset-target initialization"):
        optimizer.initialize_at_pulse_zero()
