from __future__ import annotations

import copy

import pytest

from experiments.mnist_analog_relu.staged_analysis import (
    PRODUCTION_ENDPOINTS,
    START_STATES,
    grouped_assignment_summary,
    select_adam_hyperparameters,
    validate_selection_receipt,
)
from experiments.mnist_analog_relu.staged_config import (
    ADAM_LEARNING_RATE_GRID,
    ADAM_PULSE_CAP_GRID,
)


def _rows(*, tied: bool = False):
    result = []
    for rate_index, rate in enumerate(ADAM_LEARNING_RATE_GRID):
        for cap_index, cap in enumerate(ADAM_PULSE_CAP_GRID):
            base = 0.4 if tied else 0.4 + rate_index * 0.1 + cap_index * 0.01
            for start_index, start in enumerate(START_STATES):
                result.append(
                    {
                        "learning_rate": rate,
                        "pulse_cap_per_cell": cap,
                        "start_state": start,
                        "final_validation_cross_entropy": base + start_index * 1e-4,
                        "result_sha256": f"{len(result) + 1:064x}",
                        "origin_device_state_sha256": f"{100 + start_index:064x}",
                    }
                )
    return result


def test_selection_requires_complete_six_by_four_coverage() -> None:
    with pytest.raises(ValueError, match="six-candidate by four-start"):
        select_adam_hyperparameters(
            _rows()[:-1],
            study_id="study",
            source_plan_sha256="a" * 64,
        )


def test_selection_and_strict_replay_choose_lowest_score() -> None:
    receipt = select_adam_hyperparameters(
        _rows(),
        study_id="study",
        source_plan_sha256="a" * 64,
    )
    selection = validate_selection_receipt(receipt)
    assert selection.learning_rate == 3e-4
    assert selection.pulse_cap_per_cell == 128
    assert len(receipt["candidates"]) == 6
    assert receipt["origin_device_state_sha256_by_start"] == {
        start: f"{100 + index:064x}"
        for index, start in enumerate(START_STATES)
    }


def test_selection_requires_one_exact_origin_for_each_matched_start() -> None:
    rows = _rows()
    rows[-1]["origin_device_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="same exact origin"):
        select_adam_hyperparameters(
            rows,
            study_id="study",
            source_plan_sha256="a" * 64,
        )


def test_receipt_replay_rejects_tampered_common_origin() -> None:
    receipt = select_adam_hyperparameters(
        _rows(),
        study_id="study",
        source_plan_sha256="a" * 64,
    )
    bad = copy.deepcopy(receipt)
    bad["origin_device_state_sha256_by_start"][START_STATES[0]] = "f" * 64
    with pytest.raises(ValueError, match="same exact origin"):
        validate_selection_receipt(bad)


def test_tie_break_prefers_cap_then_lower_learning_rate() -> None:
    receipt = select_adam_hyperparameters(
        _rows(tied=True),
        study_id="study",
        source_plan_sha256="b" * 64,
    )
    assert receipt["winner"]["pulse_cap_per_cell"] == 128
    assert receipt["winner"]["learning_rate"] == 3e-4


def test_receipt_rejects_coherently_wrong_declared_winner() -> None:
    receipt = select_adam_hyperparameters(
        _rows(),
        study_id="study",
        source_plan_sha256="c" * 64,
    )
    bad = copy.deepcopy(receipt)
    bad["winner"]["learning_rate"] = 3e-3
    with pytest.raises(ValueError, match="winner"):
        validate_selection_receipt(bad)


def test_assignment_summary_uses_four_assignment_means_as_primary() -> None:
    rows = [
        {
            "assignment_seed": assignment,
            "endpoint_seed": endpoint,
            "accuracy": assignment / 100_000_000.0 + write_index / 10.0,
        }
        for assignment, endpoints in PRODUCTION_ENDPOINTS.items()
        for write_index, endpoint in enumerate(endpoints)
    ]
    summary = grouped_assignment_summary(rows, "accuracy")
    assert summary["primary_assignment_count"] == 4
    assert summary["secondary_pooled_count"] == 16
    assert set(summary["assignment_means"]) == {
        str(assignment) for assignment in PRODUCTION_ENDPOINTS
    }


def test_assignment_summary_rejects_a_duplicated_write() -> None:
    rows = [
        {
            "assignment_seed": assignment,
            "endpoint_seed": endpoint,
            "accuracy": 0.9,
        }
        for assignment, endpoints in PRODUCTION_ENDPOINTS.items()
        for endpoint in endpoints
    ]
    rows[-1]["endpoint_seed"] = rows[-2]["endpoint_seed"]
    with pytest.raises(ValueError, match="exactly once"):
        grouped_assignment_summary(rows, "accuracy")
