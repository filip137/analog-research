"""ebl.lifecycle.rows — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from pathlib import Path


def _measurement_rows(measurements: list[dict], run_dir: Path) -> list[dict]:
    """Flatten already verified native measurements, locally or after collection."""
    rows = []
    for row in measurements:
        split = "test" if "test" in row["final"] else "development"
        # Every metric the family reports as a number, at clean, deployed and final state.
        metrics = sorted(k for k, v in row["final"][split].items() if isinstance(v, (int, float)))
        flat = {
            "lifecycle_id": row["lifecycle_id"],
            "technology": row["technology"],
            "hwa_arm": row["hwa_arm"],
            "array_seed": row["array_seed"],
            "case": row["case"],
            "onchip_arm": row["onchip_arm"],
            "weights": row["weights"],
            "calibration": row["calibration"],
            "evaluation": split,
            **{f"{state}_{k}": row[state][split][k] for state in ("clean", "initial", "final") for k in metrics},
            # Transfer test cohorts (another pinned corpus), as extra columns.
            **{
                f"{state}_{name}_{k}": row[state][name][k]
                for name in sorted(n for n in row["final"] if n.startswith(("transfer_", "context")))
                for state in ("clean", "initial", "final")
                for k in metrics
            },
            # Labelled task cohorts report their own (multiple-choice) metrics.
            **_task_columns(row, ("clean", "initial", "final")),
            **{k: row[k] for k in ("image_presentations", "sequence_presentations", "input_token_presentations",
                                  "scored_token_presentations", "optimizer_updates") if k in row},
            "failed_devices": row["identity"]["failed_devices"],
            "hwa_convergence_review_required": row["hwa_convergence_review_required"],
            "run_dir": str(run_dir),
        }
        for key, value in sorted(row["cost"].items()):
            flat[f"cost_{key}"] = value
        for key, value in sorted(row.get("digital_adapter", {}).items()):
            flat[f"adapter_{key}"] = value
        rows.append(flat)
    return rows


def _deployment_rows(run_dir: Path, result: dict) -> list[dict]:
    """One row per assignment and defect case of a deploy stage: clean and deployed metrics."""

    metrics = result["metrics"]
    clean = metrics["clean"]
    split = "test" if "test" in clean else "development"
    rows = []
    for label, case in metrics["cases"].items():
        deployed = case["initial"][split]
        names = sorted(k for k, v in deployed.items() if isinstance(v, (int, float)))
        rows.append(
            {
                "lifecycle_id": metrics["lifecycle_id"],
                "technology": metrics["technology"],
                "hwa_arm": metrics["stage"]["hwa_arm"],
                "array_seed": metrics["array_seed"],
                "case": label,
                **{f"defect_{k}": v for k, v in case.get("case", {}).items()},
                "evaluation": split,
                **{f"clean_{k}": clean[split][k] for k in names},
                **{f"deployed_{k}": deployed[k] for k in names},
                # Transfer test cohorts (another pinned corpus), as extra columns.
                **{
                    f"{state}_{name}_{k}": values[name][k]
                    for name in sorted(n for n in clean if n.startswith("transfer_"))
                    for state, values in (("clean", clean), ("deployed", case["initial"]))
                    for k in names
                },
                **_task_columns(None, (clean, case["initial"]), ("clean", "deployed")),
                "failed_devices": case["identity"]["failed_devices"],
                "out_of_range_devices": case["programming"].get("out_of_range_devices"),
                "hwa_convergence_review_required": metrics["hwa_convergence_review_required"],
                "run_dir": str(run_dir),
            }
        )
    return rows


def _task_columns(row: dict, states, names=None) -> dict:
    values = {state: row[state] for state in states} if names is None else dict(zip(names, states))
    final = next(reversed(values.values()))
    return {
        f"{label}_{name}_{k}": scores[name][k]
        for name in sorted(n for n in final if n.startswith("task_"))
        for label, scores in values.items()
        for k in sorted(k for k, v in final[name].items() if isinstance(v, (int, float)))
    }

