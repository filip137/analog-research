from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from methods.ttv2_buffered_reference import (
    buffered_transfer_update,
    effective_fast_learning_rate,
    effective_transfer_learning_rate,
    read_sequential_slice,
    sequential_unit_soft_bounds_pulses,
)


_ROOT = Path(__file__).resolve().parents[1]
_NATIVE_PARITY_TOOL = (
    _ROOT / "labs" / "tools" / "verify_ttv2_aihwkit_native_parity.py"
)


def test_buffered_transfer_matches_native_aihwkit_1p1_oracle() -> None:
    # Native CPU oracle: deterministic 3x4 ConstantStep devices, dw=0.1,
    # perfect transfer IO, cursor 0, desired_BL=1, forget=False, momentum=0.
    result = buffered_transfer_update(
        torch.tensor([0.8, 0.79, -0.8]),
        torch.tensor([0.4, 0.2, -0.3]),
        transfer_learning_rate=1.0,
        threshold=1.0,
        desired_bl=1,
        momentum=0.0,
        forget_buffer=False,
    )

    assert torch.equal(result.pulse_count, torch.tensor([1, 0, -1]))
    assert torch.allclose(result.omega, torch.tensor([1.2, 0.99, -1.1]))
    assert torch.allclose(result.hidden, torch.tensor([0.2, 0.99, -0.1]))


@pytest.mark.parametrize(
    ("forget", "momentum", "expected"),
    [
        (True, 0.0, [0.0, 0.0, -0.0]),
        (True, 0.25, [0.4375, 0.6, -0.3]),
        (False, 0.0, [0.75, 0.4, -0.2]),
        (False, 0.25, [1.0, 0.9, -0.45]),
    ],
)
def test_buffered_transfer_forget_residual_momentum_and_bl(
    forget: bool,
    momentum: float,
    expected: list[float],
) -> None:
    result = buffered_transfer_update(
        torch.tensor([1.75, 2.4, -1.2]),
        torch.zeros(3),
        transfer_learning_rate=1.0,
        threshold=1.0,
        desired_bl=2,
        momentum=momentum,
        forget_buffer=forget,
    )

    assert torch.equal(result.pulse_count, torch.tensor([1, 2, -1]))
    assert torch.allclose(result.hidden, torch.tensor(expected))


def test_native_learning_rate_placement_separates_fast_and_transfer_paths() -> None:
    assert effective_fast_learning_rate(optimizer_lr=0.1, fast_lr=0.5) == 0.5
    assert effective_fast_learning_rate(optimizer_lr=0.01, fast_lr=0.5) == 0.5
    assert effective_fast_learning_rate(optimizer_lr=0.01, fast_lr=0.0) == 0.01
    assert effective_transfer_learning_rate(
        optimizer_lr=0.1,
        transfer_lr=2.0,
        scale_transfer_lr=True,
    ) == pytest.approx(0.2)
    assert effective_transfer_learning_rate(
        optimizer_lr=0.01,
        transfer_lr=2.0,
        scale_transfer_lr=True,
    ) == pytest.approx(0.02)
    assert effective_transfer_learning_rate(
        optimizer_lr=0.01,
        transfer_lr=2.0,
        scale_transfer_lr=False,
    ) == pytest.approx(2.0)


def test_native_scan_uses_logical_input_columns_on_non_square_matrix() -> None:
    weight = torch.arange(12).reshape(3, 4)

    first, cursor = read_sequential_slice(weight, cursor=0)
    second, cursor = read_sequential_slice(weight, cursor=cursor)
    assert torch.equal(first, torch.tensor([0, 4, 8]))
    assert torch.equal(second, torch.tensor([1, 5, 9]))
    assert cursor == 2

    row, row_cursor = read_sequential_slice(
        weight,
        cursor=2,
        transfer_columns=False,
    )
    assert torch.equal(row, torch.tensor([8, 9, 10, 11]))
    assert row_cursor == 0


def test_native_soft_bounds_pulses_are_sequential_not_one_bulk_euler_step() -> None:
    observed = sequential_unit_soft_bounds_pulses(
        torch.tensor([0.5]),
        torch.tensor([2]),
        dw_min=0.1,
    )
    bulk = torch.tensor([0.5]) + 0.2 * (1.0 - torch.tensor([0.5]))

    assert observed.item() == pytest.approx(0.595)
    assert bulk.item() == pytest.approx(0.6)
    assert not torch.equal(observed, bulk)


@pytest.mark.skipif(
    not os.environ.get("EBL_AIHWKIT_PYTHON"),
    reason="Set EBL_AIHWKIT_PYTHON to an AIHWKit 1.1.0 Python executable.",
)
def test_optional_native_aihwkit_1p1_buffer_axis_and_cursor_parity() -> None:
    python_path = Path(os.environ["EBL_AIHWKIT_PYTHON"]).expanduser()
    completed = subprocess.run(
        [str(python_path), str(_NATIVE_PARITY_TOOL)],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    receipt = json.loads(completed.stdout)

    assert receipt["schema"] == "ebl.ttv2_aihwkit_native_parity"
    assert receipt["schema_version"] == 1
    assert receipt["aihwkit_version"] == "1.1.0"
    assert receipt["shape"] == {"out": 3, "in": 4}
    assert all(receipt["checks"].values())

    initial_fast = torch.tensor(receipt["initial"]["fast"])
    initial_buffer = torch.tensor(receipt["initial"]["buffer"])
    lambda_h = receipt["equation"]["lambda_h"]

    first_read, cursor = read_sequential_slice(initial_fast, cursor=0)
    first_reference = buffered_transfer_update(
        initial_buffer[:, 0],
        first_read,
        transfer_learning_rate=lambda_h,
        threshold=1.0,
        desired_bl=1,
        momentum=0.0,
        forget_buffer=True,
    )
    assert cursor == 1
    assert torch.equal(first_reference.pulse_count, torch.zeros(3, dtype=torch.int64))
    assert torch.allclose(
        first_reference.hidden,
        torch.tensor(receipt["after_first_transfer"]["buffer"])[:, 0],
        rtol=0.0,
        atol=2e-6,
    )

    second_read, cursor = read_sequential_slice(initial_fast, cursor=cursor)
    second_reference = buffered_transfer_update(
        initial_buffer[:, 1],
        second_read,
        transfer_learning_rate=lambda_h,
        threshold=1.0,
        desired_bl=1,
        momentum=0.0,
        forget_buffer=True,
    )
    assert cursor == 2
    assert torch.equal(second_reference.pulse_count, torch.tensor([1, -1, 1]))
    assert torch.allclose(
        second_reference.hidden,
        torch.tensor(receipt["after_second_transfer"]["buffer"])[:, 1],
        rtol=0.0,
        atol=2e-6,
    )
    expected_slow = sequential_unit_soft_bounds_pulses(
        torch.zeros(3),
        second_reference.pulse_count,
        dw_min=receipt["controls"]["slow_dw"],
    )
    assert torch.allclose(
        expected_slow,
        torch.tensor(receipt["after_second_transfer"]["slow"])[:, 1],
        rtol=0.0,
        atol=2e-6,
    )
    assert receipt["initial"]["cursor"] == [0, 0]
    assert receipt["after_first_transfer"]["cursor"] == [1, 0]
    assert receipt["after_second_transfer"]["cursor"] == [2, 0]
