from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from networks.resistive.builders import ParameterBinding, ParameterCatalog
from networks.variable.parameter import DenseWeight
from methods.measured_trace import (
    MeasuredCohortAOptimizer,
    MeasuredCohortBOptimizer,
    MeasuredCohortBLoRAOptimizer,
    _pava_nonincreasing,
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


def _config(
    preprocessing: str,
    cohort: str = "A",
    *,
    initial_target_mapping: str = "literal",
    deadband_relative: float = 0.0,
    explicit_deadband: bool = False,
    probabilistic_mode: str = "none",
    write_probability: float = 1.0,
    write_scale_relative: float = 0.0,
    write_seed: int = 0,
    dual_rail_layout_by_parameter: dict[str, str] | None = None,
) -> dict:
    config = {
        "curve_preprocessing": preprocessing,
        "split_seed": 42,
        "assignment_seed": 42,
        "formed_resistance_max_ohm": 30000.0,
        "cohort_fraction": 0.5,
        "cohort": cohort,
        "source_traces_per_cell": 2,
        "initial_pulse_index": 0,
        "projection": "global_nearest",
        "expected_trace_length": 5,
    }
    if initial_target_mapping != "literal":
        config["initial_target_mapping"] = initial_target_mapping
    if dual_rail_layout_by_parameter is not None:
        config["dual_rail_layout_by_parameter"] = (
            dual_rail_layout_by_parameter
        )
    if explicit_deadband or deadband_relative > 0.0:
        config.update(
            {
                "programming_deadband_mode": (
                    "accumulated_shadow_relative_rms"
                    if deadband_relative > 0.0
                    else "none"
                ),
                "programming_deadband_relative": deadband_relative,
            }
        )
    if probabilistic_mode != "none":
        config.update(
            {
                "probabilistic_write_mode": probabilistic_mode,
                "probabilistic_write_probability": write_probability,
                "probabilistic_write_scale_relative": write_scale_relative,
                "probabilistic_write_seed": write_seed,
            }
        )
    return config


def _optimizer(
    path: Path,
    preprocessing: str,
    cohort: str = "A",
    *,
    initial_target_mapping: str = "literal",
    deadband_relative: float = 0.0,
    explicit_deadband: bool = False,
    probabilistic_mode: str = "none",
    write_probability: float = 1.0,
    write_scale_relative: float = 0.0,
    write_seed: int = 0,
):
    weight = DenseWeight(
        (4,),
        (3,),
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
    direct = torch.optim.SGD(
        [{"params": weight.state, "lr": 1.0}],
        lr=1.0,
    )
    optimizer_type = (
        MeasuredCohortAOptimizer if cohort == "A" else MeasuredCohortBOptimizer
    )
    measured = optimizer_type(
        direct,
        catalog,
        _config(
            preprocessing,
            cohort,
            initial_target_mapping=initial_target_mapping,
            deadband_relative=deadband_relative,
            explicit_deadband=explicit_deadband,
            probabilistic_mode=probabilistic_mode,
            write_probability=write_probability,
            write_scale_relative=write_scale_relative,
            write_seed=write_seed,
        ),
        path,
    )
    return weight, measured


def _differential_optimizer(path: Path, cohort: str = "A"):
    bindings = []
    for role in ("conductance_plus", "conductance_minus"):
        weight = DenseWeight(
            (4,),
            (3,),
            gain=1.0,
            device="cpu",
            clamp=True,
            clamp_min=0.0,
            clamp_max=1.1e-4,
        )
        bindings.append(
            ParameterBinding(
                key=f"base.{role}.0",
                parameter=weight,
                role=role,
            )
        )
    catalog = ParameterCatalog(bindings)
    direct = torch.optim.SGD(
        [{"params": [binding.state for binding in bindings], "lr": 1.0}],
        lr=1.0,
    )
    optimizer_type = (
        MeasuredCohortAOptimizer
        if cohort == "A"
        else MeasuredCohortBOptimizer
    )
    measured = optimizer_type(
        direct,
        catalog,
        _config(
            "raw",
            cohort,
            initial_target_mapping="paired_affine_common_window",
        ),
        path,
    )
    return catalog, measured


def _dual_rail_single_optimizer(
    path: Path,
    *,
    layout: str,
    initial_target_mapping: str = "dual_rail_pairwise_common_window",
    cohort: str = "A",
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
    direct = torch.optim.SGD(
        [{"params": weight.state, "lr": 1.0}],
        lr=1.0,
    )
    optimizer_type = (
        MeasuredCohortAOptimizer
        if cohort == "A"
        else MeasuredCohortBOptimizer
    )
    measured = optimizer_type(
        direct,
        catalog,
        _config(
            "raw",
            cohort,
            initial_target_mapping=initial_target_mapping,
            dual_rail_layout_by_parameter={
                "base.dense_weight.0": layout
            },
        ),
        path,
    )
    return weight, measured


def _lora_optimizer(path: Path):
    specs = (
        ("base.dense_weight.0", "base", "dense_weight", False, (4, 3)),
        ("base.dense_weight.1", "base", "dense_weight", False, (3, 2)),
        ("adapter.input_factor.0", "adapter", "input_factor", True, (4, 2)),
        ("adapter.output_factor.0", "adapter", "output_factor", True, (2, 3)),
        ("adapter.input_factor.1", "adapter", "input_factor", True, (3, 2)),
        ("adapter.output_factor.1", "adapter", "output_factor", True, (2, 2)),
    )
    bindings = []
    for key, group, role, trainable, shape in specs:
        weight = DenseWeight(
            (shape[0],),
            (shape[1],),
            gain=1.0,
            device="cpu",
            clamp=True,
            clamp_min=0.0,
            clamp_max=1.1e-4,
        )
        bindings.append(
            ParameterBinding(
                key=key,
                parameter=weight,
                group=group,
                role=role,
                trainable=trainable,
            )
        )
    catalog = ParameterCatalog(bindings)
    direct = torch.optim.SGD(
        [
            {"params": binding.state, "lr": 1e-6}
            for binding in catalog.trainable
        ],
        lr=1e-6,
    )
    parameters = _config("raw", "B")
    parameters["initial_pulse_index"] = 4
    optimizer = MeasuredCohortBLoRAOptimizer(
        direct,
        catalog,
        parameters,
        path,
    )
    return catalog, optimizer


def test_pava_fit_is_nonincreasing_and_preserves_constant_mean() -> None:
    source = np.asarray([4.0, 2.0, 3.0, 1.0], dtype=np.float32)
    fitted = _pava_nonincreasing(source)

    assert np.all(np.diff(fitted) <= 0.0)
    assert fitted.tolist() == pytest.approx([4.0, 2.5, 2.5, 1.0])
    assert float(fitted.mean()) == pytest.approx(float(source.mean()))


@pytest.mark.parametrize("preprocessing", ["raw", "isotonic_nonincreasing"])
def test_cohort_a_reset_then_target_initialization_is_one_exact_write(
    tmp_path: Path,
    preprocessing: str,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(path, preprocessing)
    target = torch.linspace(4.5e-5, 9.5e-5, weight.state.numel()).reshape(
        weight.state.shape
    )
    with torch.no_grad():
        weight.state.copy_(target)

    report = optimizer.initialize_from_reset_targets()

    key = "base.dense_weight.0"
    table = optimizer._tables[key]
    source = optimizer._source_curves[key]
    curves = (
        table.alpha[:, None] * source[table.source_left]
        + (1.0 - table.alpha[:, None]) * source[table.source_right]
    )
    expected_pulses = (
        curves - target.reshape(-1, 1)
    ).abs().argmin(dim=1)
    expected_values = curves.gather(
        1,
        expected_pulses[:, None],
    ).squeeze(1)
    assert torch.equal(
        optimizer._pulse_indices[key].to(torch.long),
        expected_pulses,
    )
    assert torch.equal(weight.state.reshape(-1), expected_values)
    initial = report["parameters"][key]["initial_write"]
    assert initial["kind"] == "reset_then_loaded_target_global_nearest"
    assert initial["reset_pulse_index"] == 0
    assert initial["initial_write_count_per_cell"] == 1
    assert initial["reset_to_target_rms_distance_s"] > 0.0


def test_per_device_affine_initialization_maps_nominal_fraction_to_curve_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(
        path,
        "raw",
        initial_target_mapping="per_device_affine",
    )
    fraction = torch.linspace(0.0, 1.0, weight.state.numel()).reshape(
        weight.state.shape
    )
    with torch.no_grad():
        weight.state.copy_(fraction * 1.1e-4)

    report = optimizer.initialize_from_reset_targets()

    key = "base.dense_weight.0"
    curve_min, curve_max = optimizer._curve_ranges(key)
    expected_target = curve_min + fraction.reshape(-1) * (
        curve_max - curve_min
    )
    torch.testing.assert_close(
        optimizer._shadows[key].reshape(-1),
        expected_target,
    )
    initial = report["parameters"][key]["initial_write"]
    assert initial["initial_target_mapping"] == "per_device_affine"
    assert initial["projection_max_abs_error_s"] <= 1.0e-5


def test_paired_affine_initialization_uses_shared_reachable_baseline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    catalog, optimizer = _differential_optimizer(path)
    plus = catalog.by_key["base.conductance_plus.0"].state
    minus = catalog.by_key["base.conductance_minus.0"].state
    with torch.no_grad():
        plus.zero_()
        minus.zero_()

    report = optimizer.initialize_from_reset_targets()

    plus_shadow = optimizer._shadows["base.conductance_plus.0"]
    minus_shadow = optimizer._shadows["base.conductance_minus.0"]
    torch.testing.assert_close(plus_shadow, minus_shadow)
    for key in (
        "base.conductance_plus.0",
        "base.conductance_minus.0",
    ):
        initial = report["parameters"][key]["initial_write"]
        assert (
            initial["initial_target_mapping"]
            == "paired_affine_common_window"
        )
        assert 0.0 <= initial["common_window_empty_fraction"] <= 1.0


@pytest.mark.parametrize("layout", ["halves", "paired"])
def test_dual_rail_pairwise_mapping_cancels_each_complementary_baseline(
    tmp_path: Path,
    layout: str,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _dual_rail_single_optimizer(path, layout=layout)
    fractions = torch.tensor(
        [
            [0.75, 0.25, 0.25, 0.75],
            [0.25, 0.75, 0.75, 0.25],
            [0.25, 0.75, 0.75, 0.25],
            [0.75, 0.25, 0.25, 0.75],
        ],
        dtype=weight.state.dtype,
    )
    with torch.no_grad():
        weight.state.copy_(fractions * 1.1e-4)
    nominal_state = weight.state.detach().clone()
    preview = optimizer.preview_reset_targets()
    assert optimizer.initialized is False
    assert optimizer._pulse_indices == {}
    torch.testing.assert_close(weight.state, nominal_state)

    _nominal, _clipped, targets, details = (
        optimizer._mapped_initial_targets()
    )
    target = targets["base.dense_weight.0"]
    curve_min, curve_max = optimizer._curve_ranges(
        "base.dense_weight.0"
    )
    curve_min = curve_min.reshape(weight.state.shape)
    curve_max = curve_max.reshape(weight.state.shape)
    if layout == "halves":
        plus_columns = torch.tensor([0, 1])
        minus_columns = torch.tensor([2, 3])
    else:
        plus_columns = torch.tensor([0, 2])
        minus_columns = torch.tensor([1, 3])
    for rows in (torch.tensor([0, 1]), torch.tensor([2, 3])):
        low = torch.maximum(
            curve_min[rows[:, None], plus_columns],
            curve_min[rows[:, None], minus_columns],
        )
        high = torch.minimum(
            curve_max[rows[:, None], plus_columns],
            curve_max[rows[:, None], minus_columns],
        )
        span = (high - low).clamp_min(0.0)
        expected_difference = span * (
            fractions[rows[:, None], plus_columns]
            - fractions[rows[:, None], minus_columns]
        )
        torch.testing.assert_close(
            target[rows[:, None], plus_columns]
            - target[rows[:, None], minus_columns],
            expected_difference,
        )

    report = optimizer.initialize_from_reset_targets()
    torch.testing.assert_close(
        weight.state,
        preview["base.dense_weight.0"],
    )
    initial = report["parameters"]["base.dense_weight.0"][
        "initial_write"
    ]
    assert (
        initial["initial_target_mapping"]
        == "dual_rail_pairwise_common_window"
    )
    assert initial["dual_rail_layout"] == layout
    assert initial["pair_count"] == 8
    assert 0.0 <= initial["common_window_empty_fraction"] <= 1.0


@pytest.mark.parametrize("layout", ["halves", "paired"])
def test_dual_rail_quad_mapping_uses_one_window_for_each_four_cell_block(
    tmp_path: Path,
    layout: str,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _dual_rail_single_optimizer(
        path,
        layout=layout,
        initial_target_mapping="dual_rail_quad_common_window",
    )
    fractions = torch.tensor(
        [
            [0.75, 0.25, 0.25, 0.75],
            [0.25, 0.75, 0.75, 0.25],
            [0.25, 0.75, 0.75, 0.25],
            [0.75, 0.25, 0.25, 0.75],
        ],
        dtype=weight.state.dtype,
    )
    with torch.no_grad():
        weight.state.copy_(fractions * 1.1e-4)

    preview = optimizer.preview_reset_targets()
    _nominal, _clipped, targets, details = (
        optimizer._mapped_initial_targets()
    )
    key = "base.dense_weight.0"
    target = targets[key]
    curve_min, curve_max = optimizer._curve_ranges(key)
    curve_min = curve_min.reshape(weight.state.shape)
    curve_max = curve_max.reshape(weight.state.shape)
    plus_rows = torch.tensor([0, 1])
    minus_rows = torch.tensor([2, 3])
    if layout == "halves":
        plus_columns = torch.tensor([0, 1])
        minus_columns = torch.tensor([2, 3])
    else:
        plus_columns = torch.tensor([0, 2])
        minus_columns = torch.tensor([1, 3])

    group_minimum = torch.stack(
        tuple(
            curve_min[rows[:, None], columns]
            for rows in (plus_rows, minus_rows)
            for columns in (plus_columns, minus_columns)
        )
    )
    group_maximum = torch.stack(
        tuple(
            curve_max[rows[:, None], columns]
            for rows in (plus_rows, minus_rows)
            for columns in (plus_columns, minus_columns)
        )
    )
    common_low = group_minimum.amax(dim=0)
    common_high = group_maximum.amin(dim=0)
    overlap = common_high >= common_low
    baseline = torch.where(
        overlap,
        common_low,
        0.5 * (common_low + common_high),
    )
    span = (common_high - common_low).clamp_min(0.0)
    for rows in (plus_rows, minus_rows):
        for columns in (plus_columns, minus_columns):
            torch.testing.assert_close(
                target[rows[:, None], columns],
                baseline
                + fractions[rows[:, None], columns] * span,
            )
    torch.testing.assert_close(
        target[plus_rows[:, None], plus_columns],
        target[minus_rows[:, None], minus_columns],
    )
    torch.testing.assert_close(
        target[plus_rows[:, None], minus_columns],
        target[minus_rows[:, None], plus_columns],
    )

    assert details[key]["group_size"] == 4
    assert details[key]["quad_count"] == 4
    assert details[key]["logical_synapse_count"] == 4
    report = optimizer.initialize_from_reset_targets()
    torch.testing.assert_close(weight.state, preview[key])
    initial = report["parameters"][key]["initial_write"]
    assert (
        initial["initial_target_mapping"]
        == "dual_rail_quad_common_window"
    )
    assert initial["dual_rail_layout"] == layout
    assert initial["group_size"] == 4
    assert initial["quad_count"] == 4
    assert initial["logical_synapse_count"] == 4
    assert 0.0 <= initial["common_window_empty_fraction"] <= 1.0


@pytest.mark.parametrize("layout", ["halves", "paired"])
def test_cohort_b_quad_deployment_preserves_four_way_nominal_symmetry(
    tmp_path: Path,
    layout: str,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _dual_rail_single_optimizer(
        path,
        layout=layout,
        initial_target_mapping="dual_rail_quad_common_window",
        cohort="B",
    )
    fractions = torch.tensor(
        [
            [0.75, 0.25, 0.25, 0.75],
            [0.25, 0.75, 0.75, 0.25],
            [0.25, 0.75, 0.75, 0.25],
            [0.75, 0.25, 0.25, 0.75],
        ],
        dtype=weight.state.dtype,
    )
    with torch.no_grad():
        weight.state.copy_(fractions * 1.1e-4)

    report = optimizer.initialize_from_loaded_targets()
    target = optimizer._shadows["base.dense_weight.0"]
    plus_rows = torch.tensor([0, 1])
    minus_rows = torch.tensor([2, 3])
    if layout == "halves":
        plus_columns = torch.tensor([0, 1])
        minus_columns = torch.tensor([2, 3])
    else:
        plus_columns = torch.tensor([0, 2])
        minus_columns = torch.tensor([1, 3])
    torch.testing.assert_close(
        target[plus_rows[:, None], plus_columns],
        target[minus_rows[:, None], minus_columns],
    )
    torch.testing.assert_close(
        target[plus_rows[:, None], minus_columns],
        target[minus_rows[:, None], plus_columns],
    )

    initial = report["parameters"]["base.dense_weight.0"][
        "initial_write"
    ]
    assert report["active_cohort"] == "B"
    assert initial["kind"] == "mapped_loaded_target_global_nearest"
    assert (
        initial["initial_target_mapping"]
        == "dual_rail_quad_common_window"
    )
    assert initial["dual_rail_layout"] == layout
    assert initial["group_size"] == 4
    assert initial["quad_count"] == 4


def test_cohort_b_paired_deployment_preserves_a_shared_differential_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    catalog, optimizer = _differential_optimizer(path, cohort="B")
    plus = catalog.by_key["base.conductance_plus.0"].state
    minus = catalog.by_key["base.conductance_minus.0"].state
    fraction = torch.linspace(0.0, 1.0, plus.numel()).reshape(plus.shape)
    with torch.no_grad():
        plus.copy_(fraction * 1.1e-4)
        minus.copy_((1.0 - fraction) * 1.1e-4)

    report = optimizer.initialize_from_loaded_targets()

    plus_shadow = optimizer._shadows["base.conductance_plus.0"]
    minus_shadow = optimizer._shadows["base.conductance_minus.0"]
    plus_fraction = plus_shadow.reshape(-1)
    minus_fraction = minus_shadow.reshape(-1)
    assert torch.equal(
        torch.sign(plus_fraction - minus_fraction),
        torch.sign((2.0 * fraction - 1.0).reshape(-1)),
    )
    for key in (
        "base.conductance_plus.0",
        "base.conductance_minus.0",
    ):
        initial = report["parameters"][key]["initial_write"]
        assert initial["kind"] == "mapped_loaded_target_global_nearest"
        assert (
            initial["initial_target_mapping"]
            == "paired_affine_common_window"
        )
        assert 0.0 <= initial["common_window_empty_fraction"] <= 1.0


def test_measured_lora_freezes_deployed_base_and_resets_every_factor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    catalog, optimizer = _lora_optimizer(path)
    with torch.no_grad():
        catalog.by_key["base.dense_weight.0"].state.fill_(7.0e-5)
        catalog.by_key["base.dense_weight.1"].state.fill_(6.0e-5)

    report = optimizer.initialize_from_base_and_reset_adapters()

    assert tuple(optimizer._tables) == (
        "base.dense_weight.0",
        "base.dense_weight.1",
        "adapter.input_factor.0",
        "adapter.output_factor.0",
        "adapter.input_factor.1",
        "adapter.output_factor.1",
    )
    assert set(report["frozen_base"]) == {
        "base.dense_weight.0",
        "base.dense_weight.1",
    }
    for binding in catalog.trainable:
        key = binding.key
        table = optimizer._tables[key]
        source = optimizer._source_curves[key]
        expected = (
            table.alpha * source[table.source_left, 4]
            + (1.0 - table.alpha) * source[table.source_right, 4]
        ).reshape(binding.state.shape)
        assert torch.equal(binding.state, expected)
        assert torch.all(optimizer._pulse_indices[key] == 4)
        assert (
            optimizer.programming_report["parameters"][key]["initial_write"][
                "kind"
            ]
            == "fully_reset_trace_endpoint"
        )
        assert (
            optimizer.programming_report["parameters"][key]["initial_write"][
                "last_pulse_fraction"
            ]
            == 1.0
        )

    base_before = {
        key: catalog.by_key[key].state.clone()
        for key in MeasuredCohortBLoRAOptimizer._BASE_KEYS
    }
    for binding in catalog.trainable:
        binding.state.grad = torch.ones_like(binding.state)
    optimizer.step()
    for key, expected in base_before.items():
        assert torch.equal(catalog.by_key[key].state, expected)


def test_measured_lora_resume_preserves_the_next_physical_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    catalog, optimizer = _lora_optimizer(path)
    with torch.no_grad():
        catalog.by_key["base.dense_weight.0"].state.fill_(7.0e-5)
        catalog.by_key["base.dense_weight.1"].state.fill_(6.0e-5)
    optimizer.initialize_from_base_and_reset_adapters()

    for binding in catalog.trainable:
        binding.state.grad = torch.ones_like(binding.state)
    optimizer.step()
    checkpoint = optimizer.state_dict()

    restored_catalog, restored = _lora_optimizer(path)
    with torch.no_grad():
        for binding in catalog:
            restored_catalog.by_key[binding.key].state.copy_(binding.state)
    restored.load_state_dict(checkpoint)

    assert restored.programming_report == optimizer.programming_report
    for binding in catalog.trainable:
        key = binding.key
        assert torch.equal(
            restored._pulse_indices[key],
            optimizer._pulse_indices[key],
        )
        assert torch.equal(restored._shadows[key], optimizer._shadows[key])
        gradient = torch.linspace(
            -1.0,
            1.0,
            binding.state.numel(),
            dtype=binding.state.dtype,
        ).reshape(binding.state.shape)
        binding.state.grad = gradient.clone()
        restored_catalog.by_key[key].state.grad = gradient.clone()

    optimizer.step()
    restored.step()
    for binding in catalog:
        assert torch.equal(
            restored_catalog.by_key[binding.key].state,
            binding.state,
        )
    assert restored.programming_report == optimizer.programming_report


@pytest.mark.parametrize("preprocessing", ["raw", "isotonic_nonincreasing"])
def test_measured_optimizer_uses_exact_global_nearest_and_resumes(
    tmp_path: Path,
    preprocessing: str,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(path, preprocessing)

    initial = optimizer.initialize_at_pulse_zero()
    assert initial["formed_trace_count"] == 7
    assert initial["screened_trace_names"] == ["R01_01_bar"]
    assert initial["physical_cell_count"] == 4
    assert initial["cohort_a_cell_count"] == 2
    assert initial["cohort_b_cell_count"] == 2
    assert initial["cohort_b_reserved_not_used"] is True

    target = torch.linspace(4.5e-5, 9.5e-5, weight.state.numel())
    shadow_before = optimizer._shadows["base.dense_weight.0"].reshape(-1).clone()
    weight.state.grad = (shadow_before - target).reshape(weight.state.shape)
    optimizer.step()

    table = optimizer._tables["base.dense_weight.0"]
    source = optimizer._source_curves["base.dense_weight.0"]
    curves = (
        table.alpha[:, None] * source[table.source_left]
        + (1.0 - table.alpha[:, None]) * source[table.source_right]
    )
    expected_pulses = (curves - target[:, None]).abs().argmin(dim=1)
    expected_values = curves.gather(1, expected_pulses[:, None]).squeeze(1)
    assert torch.equal(
        optimizer._pulse_indices["base.dense_weight.0"].to(torch.long),
        expected_pulses,
    )
    assert torch.allclose(weight.state.reshape(-1), expected_values)
    assert torch.allclose(
        optimizer._shadows["base.dense_weight.0"].reshape(-1),
        target,
    )

    checkpoint = optimizer.state_dict()
    restored_weight, restored = _optimizer(path, preprocessing)
    restored_weight.state.copy_(weight.state)
    restored.load_state_dict(checkpoint)
    assert restored.learning_rates() == optimizer.learning_rates()
    assert torch.equal(
        restored._pulse_indices["base.dense_weight.0"],
        optimizer._pulse_indices["base.dense_weight.0"],
    )
    assert torch.equal(
        restored._shadows["base.dense_weight.0"],
        optimizer._shadows["base.dense_weight.0"],
    )

    gradient = torch.linspace(-1e-6, 1e-6, weight.state.numel()).reshape(
        weight.state.shape
    )
    weight.state.grad = gradient.clone()
    restored_weight.state.grad = gradient.clone()
    optimizer.step()
    restored.step()
    assert torch.equal(restored_weight.state, weight.state)
    assert restored.programming_report == optimizer.programming_report


def test_measured_resume_rejects_different_device_bytes(tmp_path: Path) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    _, optimizer = _optimizer(path, "raw")
    optimizer.initialize_at_pulse_zero()
    checkpoint = optimizer.state_dict()

    with h5py.File(path, "r+") as handle:
        handle["R00_00"][0] = 10001.0
    _, changed = _optimizer(path, "raw")
    with pytest.raises(ValueError, match="data and virtual-device assignments"):
        changed.load_state_dict(checkpoint)


@pytest.mark.parametrize("preprocessing", ["raw", "isotonic_nonincreasing"])
def test_cohort_b_deploys_loaded_targets_on_disjoint_curves_and_resumes(
    tmp_path: Path,
    preprocessing: str,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(path, preprocessing, cohort="B")
    target = torch.linspace(4.5e-5, 9.5e-5, weight.state.numel()).reshape(
        weight.state.shape
    )
    weight.state.copy_(target)

    deployed = optimizer.initialize_from_loaded_targets()

    assert deployed["active_cohort"] == "B"
    assert deployed["cohort_a_used"] is False
    assert deployed["cohort_b_used"] is True
    assert deployed["cohort_a_reserved_not_used"] is True
    assert set(optimizer._dataset.active_trace_names) == set(
        optimizer._dataset.cohort_b_trace_names
    )
    assert set(optimizer._dataset.cohort_a_trace_names).isdisjoint(
        optimizer._dataset.active_trace_names
    )

    table = optimizer._tables["base.dense_weight.0"]
    source = optimizer._source_curves["base.dense_weight.0"]
    curves = (
        table.alpha[:, None] * source[table.source_left]
        + (1.0 - table.alpha[:, None]) * source[table.source_right]
    )
    expected_pulses = (curves - target.reshape(-1, 1)).abs().argmin(dim=1)
    expected_values = curves.gather(1, expected_pulses[:, None]).squeeze(1)
    assert torch.equal(
        optimizer._pulse_indices["base.dense_weight.0"].to(torch.long),
        expected_pulses,
    )
    assert torch.allclose(weight.state.reshape(-1), expected_values)
    assert torch.equal(optimizer._shadows["base.dense_weight.0"], target)
    initial_write = deployed["parameters"]["base.dense_weight.0"][
        "initial_write"
    ]
    assert initial_write["kind"] == "loaded_target_global_nearest"
    assert initial_write["projection_mean_abs_error_s"] >= 0.0

    checkpoint = optimizer.state_dict()
    restored_weight, restored = _optimizer(path, preprocessing, cohort="B")
    restored_weight.state.copy_(weight.state)
    restored.load_state_dict(checkpoint)
    assert restored.programming_report == optimizer.programming_report


def test_current_optimizer_loads_legacy_cohort_a_state(tmp_path: Path) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(path, "raw")
    optimizer.initialize_at_pulse_zero()
    legacy = optimizer.state_dict()
    legacy["version"] = 1
    legacy["cohort_a_trace_names"] = legacy.pop(
        "active_cohort_trace_names"
    )
    legacy.pop("initialization_reports")
    legacy.pop("last_programmed_shadows")
    legacy.pop("programming_threshold_s")
    legacy.pop("probabilistic_write_scale_s")
    legacy.pop("probabilistic_write_rng_states")
    legacy["configuration"].pop("programming_deadband_mode")
    legacy["configuration"].pop("programming_deadband_relative")
    legacy["configuration"].pop("probabilistic_write_mode")
    legacy["configuration"].pop("probabilistic_write_probability")
    legacy["configuration"].pop("probabilistic_write_scale_relative")
    legacy["configuration"].pop("probabilistic_write_seed")
    for accumulator in legacy["accumulators"].values():
        accumulator.pop("programming_eligible_count")
        accumulator.pop("programming_suppressed_count")
        accumulator.pop("write_probability_sum")
        accumulator.pop("write_probability_count")

    restored_weight, restored = _optimizer(path, "raw")
    restored_weight.state.copy_(weight.state)
    restored.load_state_dict(legacy)

    assert restored.initialized is True
    assert restored.programming_report["parameters"][
        "base.dense_weight.0"
    ]["initial_write"] == {"kind": "legacy_cohort_a_checkpoint"}


def test_zero_deadband_is_exactly_equivalent_to_legacy_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    legacy_weight, legacy = _optimizer(path, "raw", cohort="B")
    explicit_weight, explicit = _optimizer(
        path,
        "raw",
        cohort="B",
        explicit_deadband=True,
    )
    target = torch.linspace(4.5e-5, 9.5e-5, legacy_weight.state.numel()).reshape(
        legacy_weight.state.shape
    )
    legacy_weight.state.copy_(target)
    explicit_weight.state.copy_(target)
    legacy.initialize_from_loaded_targets()
    explicit.initialize_from_loaded_targets()

    gradient = torch.linspace(-2e-6, 2e-6, target.numel()).reshape(target.shape)
    legacy_weight.state.grad = gradient.clone()
    explicit_weight.state.grad = gradient.clone()
    legacy.step()
    explicit.step()

    assert torch.equal(explicit_weight.state, legacy_weight.state)
    assert explicit.state_dict()["configuration"] == legacy.state_dict()[
        "configuration"
    ]
    assert explicit.programming_report == legacy.programming_report
    report = explicit.programming_report["parameters"][
        "base.dense_weight.0"
    ]
    assert report["programming_event_fraction"] == 1.0
    assert report["programming_suppressed_fraction"] == 0.0


def test_accumulated_deadband_holds_then_programs_and_resumes_exactly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(
        path,
        "raw",
        cohort="B",
        deadband_relative=0.01,
    )
    target = torch.full_like(weight.state, 7.5e-5)
    weight.state.copy_(target)
    optimizer.initialize_from_loaded_targets()
    initial_live = weight.state.clone()
    threshold = optimizer._programming_threshold_s["base.dense_weight.0"]
    delta = 0.4 * threshold

    weight.state.grad = torch.full_like(weight.state, -delta)
    optimizer.step()
    assert torch.equal(weight.state, initial_live)
    checkpoint = optimizer.state_dict()

    restored_weight, restored = _optimizer(
        path,
        "raw",
        cohort="B",
        deadband_relative=0.01,
    )
    restored_weight.state.copy_(weight.state)
    restored.load_state_dict(checkpoint)
    assert torch.equal(
        restored._last_programmed_shadows["base.dense_weight.0"],
        optimizer._last_programmed_shadows["base.dense_weight.0"],
    )

    for _ in range(2):
        gradient = torch.full_like(weight.state, -delta)
        weight.state.grad = gradient.clone()
        restored_weight.state.grad = gradient.clone()
        optimizer.step()
        restored.step()

    key = "base.dense_weight.0"
    expected, _ = optimizer._project(
        key,
        optimizer._shadows[key].reshape(-1),
    )
    assert torch.equal(weight.state.reshape(-1), expected)
    assert torch.equal(restored_weight.state, weight.state)
    assert restored.programming_report == optimizer.programming_report
    report = optimizer.programming_report["parameters"][key]
    assert report["programming_event_fraction"] == pytest.approx(1.0 / 3.0)
    assert report["programming_suppressed_fraction"] == pytest.approx(2.0 / 3.0)
    assert report["pending_programming_max_abs_s"] == pytest.approx(0.0)


def test_current_optimizer_loads_version_two_state(tmp_path: Path) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(path, "raw", cohort="B")
    weight.state.fill_(7.5e-5)
    optimizer.initialize_from_loaded_targets()
    weight.state.grad = torch.full_like(weight.state, 1e-6)
    optimizer.step()
    previous = optimizer.state_dict()
    previous["version"] = 2
    previous.pop("last_programmed_shadows")
    previous.pop("programming_threshold_s")
    previous.pop("probabilistic_write_scale_s")
    previous.pop("probabilistic_write_rng_states")
    previous["configuration"].pop("programming_deadband_mode")
    previous["configuration"].pop("programming_deadband_relative")
    previous["configuration"].pop("probabilistic_write_mode")
    previous["configuration"].pop("probabilistic_write_probability")
    previous["configuration"].pop("probabilistic_write_scale_relative")
    previous["configuration"].pop("probabilistic_write_seed")
    for accumulator in previous["accumulators"].values():
        accumulator.pop("programming_eligible_count")
        accumulator.pop("programming_suppressed_count")
        accumulator.pop("write_probability_sum")
        accumulator.pop("write_probability_count")

    restored_weight, restored = _optimizer(path, "raw", cohort="B")
    restored_weight.state.copy_(weight.state)
    restored.load_state_dict(previous)

    report = restored.programming_report["parameters"][
        "base.dense_weight.0"
    ]
    assert report["programming_event_fraction"] == 1.0
    assert report["programming_suppressed_fraction"] == 0.0
    assert torch.equal(
        restored._last_programmed_shadows["base.dense_weight.0"],
        restored._shadows["base.dense_weight.0"],
    )


def test_current_optimizer_loads_version_three_deadband_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(
        path,
        "raw",
        cohort="B",
        deadband_relative=0.01,
    )
    weight.state.fill_(7.5e-5)
    optimizer.initialize_from_loaded_targets()
    weight.state.grad = torch.full_like(weight.state, -1e-7)
    optimizer.step()
    previous = optimizer.state_dict()
    previous["version"] = 3
    previous.pop("probabilistic_write_scale_s")
    previous.pop("probabilistic_write_rng_states")
    previous["configuration"].pop("probabilistic_write_mode")
    previous["configuration"].pop("probabilistic_write_probability")
    previous["configuration"].pop("probabilistic_write_scale_relative")
    previous["configuration"].pop("probabilistic_write_seed")
    for accumulator in previous["accumulators"].values():
        accumulator.pop("write_probability_sum")
        accumulator.pop("write_probability_count")

    restored_weight, restored = _optimizer(
        path,
        "raw",
        cohort="B",
        deadband_relative=0.01,
    )
    restored_weight.state.copy_(weight.state)
    restored.load_state_dict(previous)

    assert restored.programming_report == optimizer.programming_report


def test_uniform_probabilistic_writes_are_seeded_and_resume_exactly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(
        path,
        "raw",
        cohort="B",
        probabilistic_mode="uniform_bernoulli",
        write_probability=0.4,
        write_seed=123,
    )
    weight.state.fill_(7.5e-5)
    optimizer.initialize_from_loaded_targets()
    key = "base.dense_weight.0"
    for _ in range(3):
        weight.state.grad = torch.full_like(weight.state, -2e-7)
        optimizer.step()
    checkpoint = optimizer.state_dict()

    restored_weight, restored = _optimizer(
        path,
        "raw",
        cohort="B",
        probabilistic_mode="uniform_bernoulli",
        write_probability=0.4,
        write_seed=123,
    )
    restored_weight.state.copy_(weight.state)
    restored.load_state_dict(checkpoint)
    assert torch.equal(
        restored.state_dict()["probabilistic_write_rng_states"][key],
        checkpoint["probabilistic_write_rng_states"][key],
    )

    for _ in range(5):
        gradient = torch.full_like(weight.state, -2e-7)
        weight.state.grad = gradient.clone()
        restored_weight.state.grad = gradient.clone()
        optimizer.step()
        restored.step()

    assert torch.equal(restored_weight.state, weight.state)
    assert restored.programming_report == optimizer.programming_report
    report = optimizer.programming_report["parameters"][key]
    assert report["mean_write_probability"] == pytest.approx(0.4)
    assert report["programming_event_fraction"] == pytest.approx(
        0.4,
        abs=0.15,
    )

    same_weight, same = _optimizer(
        path,
        "raw",
        cohort="B",
        probabilistic_mode="uniform_bernoulli",
        write_probability=0.4,
        write_seed=123,
    )
    same_weight.state.fill_(7.5e-5)
    same.initialize_from_loaded_targets()
    same_weight.state.grad = torch.full_like(same_weight.state, -2e-7)
    same.step()
    different_weight, different = _optimizer(
        path,
        "raw",
        cohort="B",
        probabilistic_mode="uniform_bernoulli",
        write_probability=0.4,
        write_seed=124,
    )
    different_weight.state.fill_(7.5e-5)
    different.initialize_from_loaded_targets()
    different_weight.state.grad = torch.full_like(different_weight.state, -2e-7)
    different.step()
    assert not torch.equal(
        different._last_programmed_shadows[key],
        same._last_programmed_shadows[key],
    )


def test_displacement_proportional_write_probability_tracks_pending_update(
    tmp_path: Path,
) -> None:
    path = tmp_path / "devices.hdf5"
    _write_device_data(path)
    weight, optimizer = _optimizer(
        path,
        "raw",
        cohort="B",
        probabilistic_mode="displacement_proportional",
        write_scale_relative=0.01,
        write_seed=321,
    )
    weight.state.fill_(7.5e-5)
    optimizer.initialize_from_loaded_targets()
    key = "base.dense_weight.0"
    scale = optimizer._probabilistic_write_scale_s[key]
    relative_changes = torch.tensor(
        [0.0, 0.25, 0.5, 1.0, 2.0, 0.75] * 2,
        dtype=weight.state.dtype,
    ).reshape(weight.state.shape)
    weight.state.grad = -relative_changes * scale
    optimizer.step()

    expected_probability = relative_changes.clamp(max=1.0).mean().item()
    report = optimizer.programming_report["parameters"][key]
    assert report["probabilistic_write_scale_s"] == pytest.approx(scale)
    assert report["mean_write_probability"] == pytest.approx(
        expected_probability
    )
    assert 0.0 <= report["programming_event_fraction"] <= 1.0
