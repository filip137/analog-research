"""Deterministic tuning selection and production aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

from experiments.artifacts import atomic_write_json
from experiments.mnist_analog_relu.staged_config import (
    ADAM_LEARNING_RATE_GRID,
    ADAM_PULSE_CAP_GRID,
)


SELECTION_SCHEMA = "ebl.ibm_om_crossbar_adam_selection"
SELECTION_SCHEMA_VERSION = 1
START_STATES = (
    "hwa_healthy_p0",
    "hwa_published_fault",
    "scratch_healthy_p0",
    "scratch_published_fault",
)
SCORE = (
    "mean_final_apparent_state_validation_cross_entropy_"
    "across_four_matched_starts"
)
TIE_POLICY = "prefer_pulse_cap_128_then_lower_learning_rate"
TIE_TOLERANCE = 1e-6
PRODUCTION_ENDPOINTS = {
    2_090_501: (2_090_601, 2_090_602, 2_090_603, 2_090_604),
    2_090_502: (2_090_611, 2_090_612, 2_090_613, 2_090_614),
    2_090_503: (2_090_621, 2_090_622, 2_090_623, 2_090_624),
    2_090_504: (2_090_631, 2_090_632, 2_090_633, 2_090_634),
}


@dataclass(frozen=True)
class AdamSelection:
    learning_rate: float
    pulse_cap_per_cell: int | None
    mean_validation_cross_entropy: float


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected {label} to be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected {label} to be a finite number.")
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Expected {label} to be a lowercase SHA-256 digest.")
    return value


def _candidate_key(rate: float, cap: int | None) -> tuple[float, int | None]:
    canonical_rate = next(
        (
            item
            for item in ADAM_LEARNING_RATE_GRID
            if math.isclose(float(rate), item, rel_tol=0.0, abs_tol=1e-15)
        ),
        None,
    )
    if canonical_rate is None or cap not in ADAM_PULSE_CAP_GRID:
        raise ValueError("Expected a candidate in the frozen Adam grid.")
    return canonical_rate, cap


def select_adam_hyperparameters(
    rows: Iterable[Mapping[str, Any]],
    *,
    study_id: str,
    source_plan_sha256: str,
) -> dict[str, Any]:
    """Validate complete 6x4 tuning coverage and return its strict receipt."""

    table: dict[tuple[float, int | None], dict[str, float]] = {}
    provenance: dict[tuple[float, int | None], dict[str, str]] = {}
    origins: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Expected tuning row {index} to be a mapping.")
        key = _candidate_key(row.get("learning_rate"), row.get("pulse_cap_per_cell"))
        start = row.get("start_state")
        if start not in START_STATES:
            raise ValueError("Expected every tuning row to name one required start state.")
        score = _finite(
            row.get("final_validation_cross_entropy"),
            label="final tuning validation cross-entropy",
        )
        result_sha256 = _sha256(
            row.get("result_sha256"), label="tuning result"
        )
        device_state_sha256 = _sha256(
            row.get("origin_device_state_sha256"),
            label="tuning origin device state",
        )
        prior_origin = origins.setdefault(start, device_state_sha256)
        if prior_origin != device_state_sha256:
            raise ValueError(
                "Expected every candidate for one start state to consume the "
                "same exact origin device-state artifact."
            )
        if start in table.setdefault(key, {}):
            raise ValueError("Expected exactly one tuning row per candidate/start.")
        table[key][start] = score
        provenance.setdefault(key, {})[start] = result_sha256

    expected = {
        (rate, cap)
        for rate in ADAM_LEARNING_RATE_GRID
        for cap in ADAM_PULSE_CAP_GRID
    }
    if set(table) != expected or any(set(scores) != set(START_STATES) for scores in table.values()):
        raise ValueError("Expected complete six-candidate by four-start tuning coverage.")
    candidates = []
    for rate in ADAM_LEARNING_RATE_GRID:
        for cap in ADAM_PULSE_CAP_GRID:
            scores = table[(rate, cap)]
            candidates.append(
                {
                    "learning_rate": rate,
                    "pulse_cap_per_cell": cap,
                    "start_scores": {name: scores[name] for name in START_STATES},
                    "result_sha256_by_start": {
                        name: provenance[(rate, cap)][name] for name in START_STATES
                    },
                    "origin_device_state_sha256_by_start": {
                        name: origins[name] for name in START_STATES
                    },
                    "mean_validation_cross_entropy": fmean(scores.values()),
                }
            )
    minimum = min(item["mean_validation_cross_entropy"] for item in candidates)
    tied = [
        item
        for item in candidates
        if item["mean_validation_cross_entropy"] <= minimum + TIE_TOLERANCE
    ]
    winner = min(
        tied,
        key=lambda item: (
            0 if item["pulse_cap_per_cell"] == 128 else 1,
            item["learning_rate"],
            item["mean_validation_cross_entropy"],
        ),
    )
    return {
        "schema": SELECTION_SCHEMA,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "study_id": study_id,
        "source_plan_sha256": source_plan_sha256,
        "required_start_states": list(START_STATES),
        "origin_device_state_sha256_by_start": {
            name: origins[name] for name in START_STATES
        },
        "grid": {
            "learning_rates": list(ADAM_LEARNING_RATE_GRID),
            "pulse_caps_per_cell": list(ADAM_PULSE_CAP_GRID),
        },
        "score": SCORE,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break_policy": TIE_POLICY,
        "candidates": candidates,
        "winner": {
            "learning_rate": winner["learning_rate"],
            "pulse_cap_per_cell": winner["pulse_cap_per_cell"],
            "mean_validation_cross_entropy": winner[
                "mean_validation_cross_entropy"
            ],
        },
    }


def validate_selection_receipt(value: Any) -> AdamSelection:
    """Strictly replay a tuning receipt and return its selected pair."""

    expected_keys = {
        "schema",
        "schema_version",
        "study_id",
        "source_plan_sha256",
        "required_start_states",
        "origin_device_state_sha256_by_start",
        "grid",
        "score",
        "tie_tolerance",
        "tie_break_policy",
        "candidates",
        "winner",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("schema") != SELECTION_SCHEMA
        or value.get("schema_version") != SELECTION_SCHEMA_VERSION
        or value.get("required_start_states") != list(START_STATES)
        or value.get("grid")
        != {
            "learning_rates": list(ADAM_LEARNING_RATE_GRID),
            "pulse_caps_per_cell": list(ADAM_PULSE_CAP_GRID),
        }
        or value.get("score") != SCORE
        or value.get("tie_tolerance") != TIE_TOLERANCE
        or value.get("tie_break_policy") != TIE_POLICY
    ):
        raise ValueError("Expected an exact IBM OM Adam selection receipt.")
    plan_sha = value.get("source_plan_sha256")
    if (
        not isinstance(value.get("study_id"), str)
        or not value["study_id"]
    ):
        raise ValueError("Expected selection study and plan identity.")
    _sha256(plan_sha, label="selection source plan")
    origins = value.get("origin_device_state_sha256_by_start")
    if not isinstance(origins, Mapping) or set(origins) != set(START_STATES):
        raise ValueError("Expected one common origin digest for every start state.")
    for name in START_STATES:
        _sha256(origins[name], label=f"selection origin {name}")
    rows = []
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 6:
        raise ValueError("Expected all six candidates in the selection receipt.")
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "learning_rate",
            "pulse_cap_per_cell",
            "start_scores",
            "result_sha256_by_start",
            "origin_device_state_sha256_by_start",
            "mean_validation_cross_entropy",
        }:
            raise ValueError("Expected exact candidate fields in the selection receipt.")
        scores = candidate["start_scores"]
        results = candidate["result_sha256_by_start"]
        candidate_origins = candidate["origin_device_state_sha256_by_start"]
        if not isinstance(scores, Mapping) or set(scores) != set(START_STATES):
            raise ValueError("Expected four start scores for each candidate.")
        if not isinstance(results, Mapping) or set(results) != set(START_STATES):
            raise ValueError("Expected four result hashes for each candidate.")
        if (
            not isinstance(candidate_origins, Mapping)
            or dict(candidate_origins) != dict(origins)
        ):
            raise ValueError(
                "Expected every candidate to preserve the same exact origin "
                "digest for each start state."
            )
        observed_mean = fmean(_finite(scores[name], label="candidate score") for name in START_STATES)
        declared_mean = _finite(candidate["mean_validation_cross_entropy"], label="candidate mean")
        if not math.isclose(observed_mean, declared_mean, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("Expected each candidate mean to match its four starts.")
        for name in START_STATES:
            rows.append(
                {
                    "learning_rate": candidate["learning_rate"],
                    "pulse_cap_per_cell": candidate["pulse_cap_per_cell"],
                    "start_state": name,
                    "final_validation_cross_entropy": scores[name],
                    "result_sha256": results[name],
                    "origin_device_state_sha256": origins[name],
                }
            )
    replayed = select_adam_hyperparameters(
        rows,
        study_id=value["study_id"],
        source_plan_sha256=plan_sha,
    )
    if (
        replayed["origin_device_state_sha256_by_start"] != dict(origins)
        or replayed["candidates"] != candidates
        or replayed["winner"] != value.get("winner")
    ):
        raise ValueError("Expected the declared Adam winner to match deterministic replay.")
    winner = replayed["winner"]
    return AdamSelection(
        learning_rate=winner["learning_rate"],
        pulse_cap_per_cell=winner["pulse_cap_per_cell"],
        mean_validation_cross_entropy=winner["mean_validation_cross_entropy"],
    )


def load_selection_receipt(path: Path) -> tuple[dict[str, Any], AdamSelection]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Expected a readable strict Adam selection receipt.") from error
    selection = validate_selection_receipt(value)
    return dict(value), selection


def write_selection_receipt(path: Path, value: Mapping[str, Any]) -> Path:
    validate_selection_receipt(value)
    atomic_write_json(path, value)
    return path


def grouped_assignment_summary(rows: Iterable[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    """Average writes per assignment before returning primary uncertainty."""

    grouped: dict[int, list[float]] = {}
    observed_endpoints: dict[int, set[int]] = {}
    pooled: list[float] = []
    for row in rows:
        assignment = row.get("assignment_seed")
        endpoint = row.get("endpoint_seed")
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (assignment, endpoint)
        ):
            raise ValueError("Expected integer assignment and endpoint seeds.")
        if endpoint in observed_endpoints.setdefault(assignment, set()):
            raise ValueError("Expected each assignment/write realization exactly once.")
        observed_endpoints[assignment].add(endpoint)
        value = _finite(row.get(metric), label=metric)
        grouped.setdefault(assignment, []).append(value)
        pooled.append(value)
    expected_endpoints = {
        assignment: set(endpoints)
        for assignment, endpoints in PRODUCTION_ENDPOINTS.items()
    }
    if observed_endpoints != expected_endpoints:
        raise ValueError("Expected the exact frozen four-assignment by four-write matrix.")
    assignment_means = {str(key): fmean(values) for key, values in sorted(grouped.items())}
    primary = list(assignment_means.values())
    return {
        "metric": metric,
        "assignment_means": assignment_means,
        "primary_assignment_count": 4,
        "primary_mean": fmean(primary),
        "primary_population_sd": pstdev(primary),
        "primary_range": [min(primary), max(primary)],
        "secondary_pooled_count": len(pooled),
        "secondary_pooled_mean": fmean(pooled),
        "secondary_pooled_population_sd": pstdev(pooled),
        "secondary_pooled_range": [min(pooled), max(pooled)],
    }


__all__ = [
    "AdamSelection",
    "SELECTION_SCHEMA",
    "START_STATES",
    "PRODUCTION_ENDPOINTS",
    "grouped_assignment_summary",
    "load_selection_receipt",
    "select_adam_hyperparameters",
    "validate_selection_receipt",
    "write_selection_receipt",
]
