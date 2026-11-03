"""ebl.lifecycle.collect."""
from __future__ import annotations
import json
from pathlib import Path
from core.artifacts import sha256_file
from ebl.schema.lifecycle import EXPERIMENT_ID
from ebl.lifecycle.rows import _deployment_rows
import csv
from ebl.lifecycle.rows import _measurement_rows


def _collect(args) -> int:
    rows, deployments, retention, compensation, recovery, initial = [], [], [], [], [], []
    for root in args.roots:
        for result_path in sorted(Path(root).rglob("result.json")):
            result = json.loads(result_path.read_text())
            if (
                result.get("schema") != "ebl.run"
                or result.get("experiment_id") != EXPERIMENT_ID
                or result.get("status") != "complete"
            ):
                continue
            kind = result["metrics"].get("stage", {}).get("kind")
            if kind == "onchip":
                rows.extend(_rows(result_path.parent, result))
            elif kind == "deploy":
                deployments.extend(_deployment_rows(result_path.parent, result))
            elif kind == "retention":
                artifact = next(a for a in result["artifacts"] if a["kind"] == "retention_measurements")
                path = result_path.parent / artifact["path"]
                if sha256_file(path) != artifact["sha256"]:
                    raise ValueError(f"Changed retention measurements: {path}")
                from ebl.lifecycle.retention_rows import flat_rows
                retention.extend(flat_rows(json.loads(path.read_text()), result_path.parent))
            elif kind == "compensate":
                artifact = next(a for a in result["artifacts"] if a["kind"] == "compensation_measurements")
                path = result_path.parent / artifact["path"]
                if sha256_file(path) != artifact["sha256"]:
                    raise ValueError(f"Changed compensation measurements: {path}")
                from ebl.runtime.compensation import flat_rows
                compensation.extend(flat_rows(json.loads(path.read_text()), result_path.parent))
            elif kind == 'initial_programming':
                artifact = next(a for a in result['artifacts'] if a['kind'] == 'initial_programming_measurements')
                path = result_path.parent / artifact['path']
                if sha256_file(path) != artifact['sha256']:
                    raise ValueError(f'Changed initial-programming measurements: {path}')
                from ebl.lifecycle.programming_rows import flat_rows
                initial.extend(flat_rows(json.loads(path.read_text()), result_path.parent))
            elif kind == "drift_recovery":
                artifact = next(a for a in result['artifacts'] if a['kind'] == 'drift_recovery_measurements')
                path = result_path.parent / artifact['path']
                if sha256_file(path) != artifact['sha256']:
                    raise ValueError(f'Changed recovery measurements: {path}')
                from methods.maintenance.report import flat_rows
                recovery.extend(flat_rows(json.loads(path.read_text()), result_path.parent))
    args.output.mkdir(parents=True, exist_ok=True)
    duplicates = _write_rows(rows, ("lifecycle_id", "hwa_arm", "array_seed", "case", "onchip_arm"), args.output, "rows")
    deployment_duplicates = _write_rows(
        deployments, ("lifecycle_id", "hwa_arm", "array_seed", "case"), args.output, "deployments"
    )
    retention_duplicates = _write_rows(retention, ("lifecycle_id", "hwa_arm", "array_seed", "case", "model", "snapshot"),
                                       args.output, "retention") if retention else 0
    compensation_duplicates = _write_rows(compensation, ("lifecycle_id", "hwa_arm", "array_seed", "case", "model", "snapshot", "method"),
                                          args.output, "compensation") if compensation else 0
    recovery_duplicates = _write_rows(recovery, ('run_dir', 'trajectory', 'seconds', 'phase', 'split'),
                                      args.output, 'drift_recovery') if recovery else 0
    initial_duplicates = _write_rows(initial, ('source_sha256', 'model_sha256', 'cohort_sha256', 'policy', 'realization', 'seconds', 'canary'),
                                     args.output, 'initial_programming') if initial else 0
    print(json.dumps({"rows": len(rows), "duplicates": duplicates, "deployments": len(deployments),
                      "retention_snapshots": len(retention), "retention_duplicates": retention_duplicates,
                      "compensated_snapshots": len(compensation), "compensation_duplicates": compensation_duplicates,
                      "recovery_measurements": len(recovery), "recovery_duplicates": recovery_duplicates,
                      "initial_programming_snapshots": len(initial), "initial_programming_duplicates": initial_duplicates,
                      "deployment_duplicates": deployment_duplicates, "output": str(args.output)}))
    return 0


def _rows(run_dir: Path, result: dict) -> list[dict]:
    artifacts = {item["kind"]: item for item in result["artifacts"]}
    record = artifacts["measurements"]
    path = run_dir / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Expected unchanged measurements in {run_dir}.")
    return _measurement_rows(json.loads(path.read_text()), run_dir)


def _write_rows(rows: list[dict], key_fields: tuple[str, ...], output: Path, name: str) -> int:
    keys = [tuple(row[k] for k in key_fields) for row in rows]
    for row, key in zip(rows, keys):
        # Retries are not replicates: duplicated coverage is flagged, not pooled.
        row["duplicate_coverage"] = keys.count(key) > 1
    (output / f"{name}.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return sum(row["duplicate_coverage"] for row in rows)
