"""ebl.lifecycle.retention_rows — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations


def flat_rows(rows, run_dir):
    result = []
    for row in rows:
        flat = {key: row[key] for key in ("lifecycle_id", "hwa_arm", "source_sha256", "array_seed", "case", "model",
                "snapshot", "seconds", "baseline_seconds", "read_noise", "logical_weights", "physical_devices",
                "failed_devices", "apparent_sha256", "replay_exact", "residual_rms", "change_from_programmed_rms")}
        flat["noise_scale"] = row.get("noise_scale", 1.0)
        for key in ("spread_us", "mean_us", "mean_exponent", "spread_exponent", "perplexity_ratio"):
            if key in row:
                flat[key] = row[key]
        for state, values in (("clean", row["clean"]), ("baseline", row["baseline"]), ("retained", row["metrics"])):
            flat.update({f"{state}_{split}_{name}": value for split, metrics in values.items()
                         for name, value in metrics.items() if isinstance(value, (int, float))})
        flat.update({f"cost_{k}": v for k, v in row["cost"].items()})
        flat["run_dir"] = str(run_dir)
        result.append(flat)
    return result
