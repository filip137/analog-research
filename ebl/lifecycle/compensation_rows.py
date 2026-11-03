"""Pure serialization of compensation measurements into collection rows."""

def flat_rows(rows, run_dir):
    result = []
    for row in rows:
        flat = {k: row[k] for k in ("lifecycle_id", "hwa_arm", "source_sha256", "array_seed", "case", "model", "method",
                                    "seconds", "snapshot", "baseline_seconds", "logical_weights", "physical_devices", "failed_devices",
                                    "apparent_sha256", "state_unchanged", "replay_exact", "identity_snapshot", "read_noise")}
        flat["noise_scale"] = row.get("noise_scale", 1.0)
        for key in ("spread_us", "mean_us", "mean_exponent", "spread_exponent", "perplexity_ratio",
                    "uncompensated_perplexity_ratio"):
            if key in row:
                flat[key] = row[key]
        for state in ("baseline", "uncompensated", "compensated"):
            flat.update({f"{state}_{split}_{metric}": value for split, metrics in row[state].items()
                         for metric, value in metrics.items() if isinstance(value, (float, int))})
        for split in row["baseline"]:
            baseline, initial, final = [row[s][split]["teacher_kl"] for s in ("baseline", "uncompensated", "compensated")]
            flat[f"{split}_aging_kl_regained_percent"] = 100 * (initial - final) / (initial - baseline) if initial - baseline > 1e-6 else None
            flat[f"{split}_teacher_kl_reduction"] = initial - final
        flat.update({f"cost_{k}": v for k, v in row["cost"].items()})
        flat["run_dir"] = str(run_dir)
        result.append(flat)
    return result
