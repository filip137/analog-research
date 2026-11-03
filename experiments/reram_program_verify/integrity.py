"""Machine-checked integrity contract for ReRAM characterization artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from core.artifacts import atomic_write_json, sha256_file
from experiments.reram_program_verify.storage import (
    STORAGE_SCHEMA_VERSION,
    TRAJECTORY_COLUMNS,
    VERIFY_EVENT_COLUMNS,
)


def _scalar(connection: sqlite3.Connection, query: str, values: Sequence[object] = ()) -> Any:
    row = connection.execute(query, values).fetchone()
    if row is None:
        raise RuntimeError("Expected an aggregate integrity query to return one row.")
    return row[0]


def validate_trajectory_database(
    database_path: Path,
    *,
    output_path: Path,
    expected_trajectory_count: int,
    expected_condition_count: int,
    trajectories_per_condition: int,
    maximum_program_pulses: int,
    controllers: Sequence[str],
    start_protocols: Sequence[str],
    tolerance_ratios: Sequence[float],
    target_points: int,
    repeats_per_device: int,
    expected_partition_device_counts: Mapping[str, int],
    conditioning_scope: str = "per_target",
    trajectories_per_controller: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate exact replay/accounting invariants and write a fail-closed report.

    The checks intentionally operate on the persisted SQLite artifact, not on
    in-memory controller results.  A run cannot publish endpoint fits when the
    trajectory/event ledger does not satisfy this contract.
    """

    database_path = Path(database_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, expected: Any, *, passed: bool | None = None) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(observed == expected if passed is None else passed),
                "observed": observed,
                "expected": expected,
            }
        )

    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        sqlite_integrity = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
        check("sqlite_integrity", sqlite_integrity, ["ok"])

        schema_row = connection.execute(
            "SELECT json_value FROM metadata WHERE key='storage_schema'"
        ).fetchone()
        observed_schema = json.loads(schema_row[0]) if schema_row is not None else None
        check(
            "storage_schema",
            observed_schema,
            {
                "schema": "ebl.ibm_reram.trajectory_database",
                "schema_version": STORAGE_SCHEMA_VERSION,
                "trajectory_columns": list(TRAJECTORY_COLUMNS),
                "verify_event_columns": list(VERIFY_EVENT_COLUMNS),
            },
        )

        trajectory_count = int(_scalar(connection, "SELECT COUNT(*) FROM trajectories"))
        event_count = int(_scalar(connection, "SELECT COUNT(*) FROM verify_events"))
        check("trajectory_count", trajectory_count, expected_trajectory_count)

        condition_rows = list(
            connection.execute(
                """
                SELECT controller, start_protocol, tolerance_ratio,
                       target_index, COUNT(*) AS group_count
                FROM trajectories
                GROUP BY controller, start_protocol, tolerance_ratio, target_index
                ORDER BY controller, start_protocol, tolerance_ratio, target_index
                """
            )
        )
        condition = connection.execute(
            """
            SELECT COUNT(*) AS groups_found, MIN(group_count) AS minimum_count,
                   MAX(group_count) AS maximum_count
            FROM (
                SELECT COUNT(*) AS group_count
                FROM trajectories
                GROUP BY controller, start_protocol, tolerance_ratio, target_index
            )
            """
        ).fetchone()
        condition_summary = {
            "groups": int(condition["groups_found"] or 0),
            "minimum_trajectories": int(condition["minimum_count"] or 0),
            "maximum_trajectories": int(condition["maximum_count"] or 0),
        }
        if trajectories_per_controller is None:
            expected_condition_summary = {
                "groups": expected_condition_count,
                "minimum_trajectories": trajectories_per_condition,
                "maximum_trajectories": trajectories_per_condition,
            }
        else:
            normalized_controller_counts = {
                str(key): int(value)
                for key, value in trajectories_per_controller.items()
            }
            if set(normalized_controller_counts) != set(controllers) or any(
                value < 1 for value in normalized_controller_counts.values()
            ):
                raise ValueError(
                    "Expected trajectories_per_controller to map every declared "
                    "controller to a positive count."
                )
            expected_condition_summary = {
                "groups": expected_condition_count,
                "minimum_trajectories": min(
                    normalized_controller_counts.values()
                ),
                "maximum_trajectories": max(
                    normalized_controller_counts.values()
                ),
            }
        check(
            "condition_coverage",
            condition_summary,
            expected_condition_summary,
        )
        if trajectories_per_controller is not None:
            mismatched_condition_counts = [
                {
                    "controller": str(row["controller"]),
                    "start_protocol": str(row["start_protocol"]),
                    "tolerance_ratio": float(row["tolerance_ratio"]),
                    "target_index": int(row["target_index"]),
                    "observed": int(row["group_count"]),
                    "expected": normalized_controller_counts.get(
                        str(row["controller"])
                    ),
                }
                for row in condition_rows
                if int(row["group_count"])
                != normalized_controller_counts.get(str(row["controller"]))
            ]
            check(
                "controller_specific_condition_counts",
                mismatched_condition_counts,
                [],
            )

        observed_controllers = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT controller FROM trajectories ORDER BY controller"
            )
        ]
        check("controller_coverage", observed_controllers, sorted(controllers))

        observed_starts = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT start_protocol FROM trajectories ORDER BY start_protocol"
            )
        ]
        check("start_protocol_coverage", observed_starts, sorted(start_protocols))
        observed_tolerances = [
            float(row[0])
            for row in connection.execute(
                "SELECT DISTINCT tolerance_ratio FROM trajectories ORDER BY tolerance_ratio"
            )
        ]
        check(
            "tolerance_coverage",
            observed_tolerances,
            sorted(float(value) for value in tolerance_ratios),
        )
        observed_targets = [
            int(row[0])
            for row in connection.execute(
                "SELECT DISTINCT target_index FROM trajectories ORDER BY target_index"
            )
        ]
        check("target_grid_coverage", observed_targets, list(range(target_points)))

        observed_partitions = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                """
                SELECT partition_name, COUNT(DISTINCT device_id)
                FROM trajectories
                GROUP BY partition_name
                ORDER BY partition_name
                """
            )
        }
        check(
            "identity_partition_counts",
            observed_partitions,
            dict(sorted(expected_partition_device_counts.items())),
        )
        expected_device_count = sum(expected_partition_device_counts.values())
        identity_axis = connection.execute(
            """
            SELECT COUNT(DISTINCT device_id), MIN(device_id), MAX(device_id),
                   COUNT(DISTINCT repeat_id), MIN(repeat_id), MAX(repeat_id)
            FROM trajectories
            """
        ).fetchone()
        check(
            "device_and_repeat_axes",
            {
                "device_count": int(identity_axis[0]),
                "device_minimum": int(identity_axis[1]),
                "device_maximum": int(identity_axis[2]),
                "repeat_count": int(identity_axis[3]),
                "repeat_minimum": int(identity_axis[4]),
                "repeat_maximum": int(identity_axis[5]),
            },
            {
                "device_count": expected_device_count,
                "device_minimum": 0,
                "device_maximum": expected_device_count - 1,
                "repeat_count": repeats_per_device,
                "repeat_minimum": 0,
                "repeat_maximum": repeats_per_device - 1,
            },
        )

        incomplete = int(
            _scalar(
                connection,
                """
                SELECT COUNT(*) FROM trajectories
                WHERE accepted IS NULL OR initialization_failed IS NULL
                   OR nonfinite IS NULL OR budget_exhausted IS NULL
                   OR saturated IS NULL
                   OR (nonfinite = 0 AND endpoint_apparent IS NULL)
                   OR (nonfinite = 0 AND residual_apparent IS NULL)
                   OR endpoint_persistent IS NULL OR residual_persistent IS NULL
                   OR set_count IS NULL
                   OR reset_count IS NULL OR total_pulses IS NULL
                   OR verify_count IS NULL OR reversals IS NULL
                """,
            )
        )
        check("finished_trajectories", incomplete, 0)

        invalid_accounting = int(
            _scalar(
                connection,
                """
                SELECT COUNT(*) FROM trajectories
                WHERE set_count < 0 OR reset_count < 0 OR total_pulses < 0
                   OR verify_count < 0 OR reversals < 0
                   OR set_count + reset_count <> total_pulses
                   OR total_pulses > ?
                """,
                (maximum_program_pulses,),
            )
        )
        check("trajectory_pulse_accounting", invalid_accounting, 0)

        invalid_acceptance = int(
            _scalar(
                connection,
                """
                SELECT COUNT(*) FROM trajectories
                WHERE (accepted = 1 AND (
                           conditioning_success <> 1 OR initialization_failed <> 0
                           OR nonfinite <> 0
                           OR ABS(residual_apparent) > tolerance + 1e-7
                       ))
                   OR (initialization_failed <> (1 - conditioning_success))
                   OR (budget_exhausted = 1 AND total_pulses <> ?)
                   OR (conditioning_success = 1 AND accepted = 0 AND nonfinite = 0
                       AND budget_exhausted = 0)
                """,
                (maximum_program_pulses,),
            )
        )
        check("termination_and_acceptance_semantics", invalid_acceptance, 0)

        invalid_identity = int(
            _scalar(
                connection,
                """
                SELECT COUNT(*) FROM (
                    SELECT preset, device_id
                    FROM trajectories
                    GROUP BY preset, device_id
                    HAVING COUNT(DISTINCT partition_name) <> 1
                        OR COUNT(DISTINCT construction_seed) <> 1
                        OR COUNT(DISTINCT corrupt) <> 1
                        OR COUNT(DISTINCT corrupt_population) <> 1
                )
                """,
            )
        )
        check("device_identity_invariants", invalid_identity, 0)

        invalid_sampled_bounds = int(
            _scalar(
                connection,
                """
                SELECT COUNT(*) FROM (
                    SELECT preset, device_id,
                           MIN(sampled_lower_persistent) AS minimum_lower,
                           MAX(sampled_upper_persistent) AS maximum_upper,
                           COUNT(DISTINCT sampled_lower_persistent) AS lower_values,
                           COUNT(DISTINCT sampled_upper_persistent) AS upper_values,
                           MIN(corrupt) AS minimum_corrupt,
                           MAX(corrupt) AS maximum_corrupt
                    FROM trajectories
                    GROUP BY preset, device_id
                    HAVING lower_values <> 1 OR upper_values <> 1
                        OR minimum_lower IS NULL OR maximum_upper IS NULL
                        OR minimum_lower > maximum_upper
                        OR minimum_corrupt <> maximum_corrupt
                        OR (
                            minimum_corrupt = 1
                            AND minimum_lower <> maximum_upper
                        )
                        OR (
                            minimum_corrupt = 0
                            AND minimum_lower = maximum_upper
                        )
                )
                """,
            )
        )
        check("sampled_device_bound_invariants", invalid_sampled_bounds, 0)

        invalid_repeat_seed = int(
            _scalar(
                connection,
                """
                SELECT COUNT(*) FROM (
                    SELECT preset, device_id, repeat_id
                    FROM trajectories
                    GROUP BY preset, device_id, repeat_id
                    HAVING COUNT(DISTINCT repeat_seed) <> 1
                )
                """,
            )
        )
        check("repeat_seed_invariant", invalid_repeat_seed, 0)

        invalid_matched_stream = int(
            _scalar(
                connection,
                """
                SELECT COUNT(*) FROM (
                    SELECT preset, start_protocol, target_index, device_id, repeat_id
                    FROM trajectories
                    GROUP BY preset, start_protocol, target_index, device_id, repeat_id
                    HAVING COUNT(DISTINCT conditioning_seed) <> 1
                        OR COUNT(DISTINCT pulse_seed) <> 1
                )
                """,
            )
        )
        check("matched_controller_and_tolerance_streams", invalid_matched_stream, 0)

        if conditioning_scope not in {"per_target", "per_start"}:
            raise ValueError(
                "Expected conditioning_scope to equal 'per_target' or 'per_start'."
            )
        if conditioning_scope == "per_start":
            invalid_blocked_conditioning = int(
                _scalar(
                    connection,
                    """
                    SELECT COUNT(*) FROM (
                        SELECT preset, start_protocol, device_id, repeat_id
                        FROM trajectories
                        GROUP BY preset, start_protocol, device_id, repeat_id
                        HAVING COUNT(DISTINCT conditioning_seed) <> 1
                            OR COUNT(DISTINCT conditioning_success) <> 1
                            OR COUNT(DISTINCT conditioning_pulses) <> 1
                            OR COUNT(DISTINCT conditioned_apparent) <> 1
                            OR COUNT(DISTINCT conditioned_persistent) <> 1
                    )
                    """,
                )
            )
            check(
                "blocked_conditioning_reused_across_targets",
                invalid_blocked_conditioning,
                0,
            )

        overlapping_streams = int(
            _scalar(
                connection,
                "SELECT COUNT(*) FROM trajectories WHERE conditioning_seed = pulse_seed",
            )
        )
        check("conditioning_and_programming_streams_are_distinct", overlapping_streams, 0)

        stochastic_controller_seeds = int(
            _scalar(
                connection,
                "SELECT COUNT(*) FROM trajectories WHERE controller_seed IS NOT NULL",
            )
        )
        check(
            "deterministic_controller_seed_not_applicable",
            stochastic_controller_seeds,
            0,
        )

        event_aggregate = connection.execute(
            """
            SELECT COUNT(*) AS trajectory_groups,
                   SUM(CASE WHEN event_count <> verify_count THEN 1 ELSE 0 END)
                       AS count_mismatch,
                   SUM(CASE WHEN minimum_index <> 0 OR maximum_index <> event_count - 1
                            THEN 1 ELSE 0 END) AS index_mismatch,
                   SUM(CASE WHEN pulse_sum <> total_pulses THEN 1 ELSE 0 END)
                       AS pulse_mismatch,
                   SUM(CASE WHEN set_sum <> set_count OR reset_sum <> reset_count
                            THEN 1 ELSE 0 END) AS polarity_mismatch
            FROM (
                SELECT t.trajectory_id, t.verify_count, t.total_pulses,
                       t.set_count, t.reset_count,
                       COUNT(e.verify_index) AS event_count,
                       MIN(e.verify_index) AS minimum_index,
                       MAX(e.verify_index) AS maximum_index,
                       COALESCE(SUM(e.pulse_count), 0) AS pulse_sum,
                       COALESCE(SUM(CASE WHEN e.direction = 1 THEN e.pulse_count ELSE 0 END), 0)
                           AS set_sum,
                       COALESCE(SUM(CASE WHEN e.direction = -1 THEN e.pulse_count ELSE 0 END), 0)
                           AS reset_sum
                FROM trajectories AS t
                LEFT JOIN verify_events AS e ON e.trajectory_id = t.trajectory_id
                GROUP BY t.trajectory_id
            )
            """
        ).fetchone()
        event_summary = {
            "trajectory_groups": int(event_aggregate["trajectory_groups"] or 0),
            "event_count": event_count,
            "count_mismatch": int(event_aggregate["count_mismatch"] or 0),
            "index_mismatch": int(event_aggregate["index_mismatch"] or 0),
            "pulse_mismatch": int(event_aggregate["pulse_mismatch"] or 0),
            "polarity_mismatch": int(event_aggregate["polarity_mismatch"] or 0),
        }
        check(
            "event_trajectory_accounting",
            {
                "count_mismatch": event_summary["count_mismatch"],
                "index_mismatch": event_summary["index_mismatch"],
                "pulse_mismatch": event_summary["pulse_mismatch"],
                "polarity_mismatch": event_summary["polarity_mismatch"],
            },
            {
                "count_mismatch": 0,
                "index_mismatch": 0,
                "pulse_mismatch": 0,
                "polarity_mismatch": 0,
            },
        )

        invalid_event_rows = int(
            _scalar(
                connection,
                """
                SELECT COUNT(*) FROM verify_events AS e
                JOIN trajectories AS t ON t.trajectory_id = e.trajectory_id
                WHERE e.signed_batch <> e.direction * e.pulse_count
                   OR e.pulse_count < 0
                   OR e.total_pulses < 0
                   OR e.apparent_before IS NULL
                   OR e.persistent_after IS NULL
                   OR (e.apparent_after IS NULL AND t.nonfinite <> 1)
                   OR (e.verify_index = 0 AND (
                       e.direction <> 0 OR e.pulse_count <> 0 OR e.signed_batch <> 0
                       OR e.total_pulses <> 0 OR e.apparent_before <> e.apparent_after
                   ))
                   OR (e.verify_index > 0 AND (e.direction = 0 OR e.pulse_count = 0))
                """,
            )
        )
        check("verify_event_row_semantics", invalid_event_rows, 0)

        invalid_event_sequence = int(
            _scalar(
                connection,
                """
                SELECT COUNT(*) FROM (
                    SELECT trajectory_id, verify_index, apparent_before,
                           LAG(apparent_after) OVER (
                               PARTITION BY trajectory_id ORDER BY verify_index
                           ) AS prior_apparent,
                           total_pulses,
                           LAG(total_pulses) OVER (
                               PARTITION BY trajectory_id ORDER BY verify_index
                           ) AS prior_total,
                           pulse_count
                    FROM verify_events
                )
                WHERE verify_index > 0 AND (
                    apparent_before IS NOT prior_apparent
                    OR total_pulses <> prior_total + pulse_count
                )
                """,
            )
        )
        check("verify_event_sequence", invalid_event_sequence, 0)

        invalid_initial_or_final = int(
            _scalar(
                connection,
                """
                WITH event_bounds AS (
                    SELECT trajectory_id,
                           MAX(CASE WHEN verify_index = 0 THEN apparent_after END)
                               AS initial_apparent,
                           MAX(CASE WHEN verify_index = 0 THEN persistent_after END)
                               AS initial_persistent,
                           MAX(verify_index) AS final_index
                    FROM verify_events
                    GROUP BY trajectory_id
                ), final_events AS (
                    SELECT e.trajectory_id, e.apparent_after, e.persistent_after
                    FROM verify_events AS e
                    JOIN event_bounds AS b
                      ON b.trajectory_id = e.trajectory_id
                     AND b.final_index = e.verify_index
                )
                SELECT COUNT(*)
                FROM trajectories AS t
                LEFT JOIN event_bounds AS b ON b.trajectory_id = t.trajectory_id
                LEFT JOIN final_events AS f ON f.trajectory_id = t.trajectory_id
                WHERE (t.conditioning_success = 1 AND (
                           b.initial_apparent IS NULL
                           OR b.initial_apparent IS NOT t.conditioned_apparent
                           OR b.initial_persistent IS NOT t.conditioned_persistent
                           OR f.apparent_after IS NOT t.endpoint_apparent
                           OR f.persistent_after IS NOT t.endpoint_persistent
                       ))
                   OR (t.conditioning_success = 0 AND b.trajectory_id IS NOT NULL)
                """,
            )
        )
        check("initial_and_final_event_state", invalid_initial_or_final, 0)

        if "one_pulse" in controllers:
            invalid_one_pulse = int(
                _scalar(
                    connection,
                    """
                    SELECT COUNT(*) FROM verify_events AS e
                    JOIN trajectories AS t ON t.trajectory_id = e.trajectory_id
                    WHERE t.controller = 'one_pulse' AND e.verify_index > 0
                      AND e.pulse_count <> 1
                    """,
                )
            )
            check("one_pulse_controller_batches", invalid_one_pulse, 0)

        invalid_reversals = int(
            _scalar(
                connection,
                """
                WITH directed AS (
                    SELECT trajectory_id, verify_index, direction,
                           LAG(direction) OVER (
                               PARTITION BY trajectory_id ORDER BY verify_index
                           ) AS prior_direction
                    FROM verify_events
                ), counted AS (
                    SELECT trajectory_id,
                           SUM(CASE WHEN verify_index > 1
                                         AND direction <> prior_direction
                                    THEN 1 ELSE 0 END) AS observed_reversals
                    FROM directed
                    GROUP BY trajectory_id
                )
                SELECT COUNT(*) FROM trajectories AS t
                LEFT JOIN counted AS c ON c.trajectory_id = t.trajectory_id
                WHERE t.reversals <> COALESCE(c.observed_reversals, 0)
                """,
            )
        )
        check("polarity_reversal_accounting", invalid_reversals, 0)

        passed = all(bool(record["passed"]) for record in checks)
        artifact = {
            "schema": "ebl.ibm_reram.trajectory_integrity",
            "schema_version": 1,
            "database": database_path.name,
            "database_sha256": sha256_file(database_path),
            "passed": passed,
            "summary": {
                "trajectory_count": trajectory_count,
                "verify_event_count": event_count,
                "maximum_program_pulses": maximum_program_pulses,
                "conditioning_scope": conditioning_scope,
            },
            "checks": checks,
        }
    finally:
        connection.close()

    atomic_write_json(output_path, artifact)
    if not artifact["passed"]:
        failed = [record["name"] for record in checks if not record["passed"]]
        raise RuntimeError(
            "Expected the persisted ReRAM trajectory database to satisfy every "
            f"integrity invariant. Provided failures: {failed!r}. See {output_path}."
        )
    return artifact


__all__ = ["validate_trajectory_database"]
