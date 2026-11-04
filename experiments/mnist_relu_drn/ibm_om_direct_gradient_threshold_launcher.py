"""Sequential launcher for the one-seed deployed gradient-threshold sweep."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from ebl.runtime.store import atomic_write_json, sha256_file
from ebl.lifecycle.study import load_study_record

from .ibm_om_reset_relative_recovery_launcher import (
    _completed_runs,
    _read_json,
    _run_one,
    _utc_now,
    _validate_file,
)


_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1"
STUDY_PLAN = _ROOT / "studies" / f"{STUDY_ID}.json"
CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_reset_relative_direct_gradient_threshold"
)
SOURCE_WEIGHTS = (
    _ROOT
    / "results"
    / "mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1"
    / "runs"
    / "train-quantized-qat"
    / "20260825T092317.972756Z-52034a40-acebee3d"
    / "checkpoints"
    / "weights.pt"
)
SOURCE_WEIGHTS_SHA256 = (
    "537681598b887d4d63c6429c7f7bee3a7082041215c09d6cb3c25947452f311a"
)
TEACHER_WEIGHTS = (
    _ROOT
    / "results"
    / "mnist-relu-drn-kd-exploratory-20260816"
    / "teacher_fixed_init"
    / "20260816T132557.806720Z-fbff3c26-6f6867f1"
    / "checkpoints"
    / "weights.pt"
)
TEACHER_WEIGHTS_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
DEVICE_MODEL = _ROOT / "data" / "ibm_reram_om_pv128_hwa_v1.json"
DEVICE_MODEL_SHA256 = (
    "3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3"
)
SOURCE_DEPLOYMENT = (
    _ROOT
    / "results"
    / "mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1"
    / "runs"
    / "deploy-source-development"
    / "20260825T152340.475476Z-b5f2dd50-29c0e6b0"
    / "artifacts"
    / "ibm_om_deployment.pt"
)
SOURCE_DEPLOYMENT_SHA256 = (
    "0b99f4dd73841a891993708bdace0ab508864e2e60e56620c165e3df99f006cd"
)
BASELINE_STUDY_ROOT = (
    _ROOT
    / "results"
    / "mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1"
)
CONTROL_RUNS = {
    0.1: {
        "run": BASELINE_STUDY_ROOT
        / "runs"
        / "direct-pulse-budget-0p1"
        / "20260825T153322.292937Z-8359c5d9-f0e44179",
        "result_sha256": "023fb0a198085e8663927d6db9b301f8de03677b1e06b0a798f61cc52fda0b43",
        "report_sha256": "f3ded58c814858f3bb1a6c8da282521e04ee8f3fdaebf9cba8ffacb1cc02a1c2",
    },
    1.0: {
        "run": BASELINE_STUDY_ROOT
        / "runs"
        / "direct-pulse-budget-1"
        / "20260825T153752.454492Z-65cf024a-7cd1158a",
        "result_sha256": "06d1a5d7efe0d3a3d7ee2b169bbf6b67ec7644f95744f2697165d8aa2d8f87e0",
        "report_sha256": "e9944e8ce1494e95369bc632be02a8ca60588be20d22ba59471d836290b2dbc1",
    },
    4.0: {
        "run": BASELINE_STUDY_ROOT
        / "runs"
        / "direct-pulse-budget-4"
        / "20260825T154252.550237Z-8efc9f83-c581ab36",
        "result_sha256": "7310f803832f66d7ec68f24d71a26298f7b33cf895600d62ca3940f4b626a3a8",
        "report_sha256": "b36e5d09f648ad47b12c12c629587cd826587fc5460d3dac4f8aea55c3cc23b1",
    },
}
THRESHOLD_ARMS = tuple(
    (
        f"p{percentile}-budget-{label}",
        f"p{percentile}_budget_{label}.json",
        float(percentile),
        budget,
    )
    for percentile in (90, 95, 99)
    for label, budget in (("0p1", 0.1), ("1", 1.0), ("4", 4.0))
)
HEARTBEAT_SECONDS = 15.0
_CUDA_DEVICES = re.compile(r"[0-9]+(?:,[0-9]+)*\Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run nine fixed p90/p95/p99 deployed direct-gradient gates "
            "sequentially with a durable semantic heartbeat."
        )
    )
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser


def _arm_record(study: Mapping[str, Any], arm_id: str) -> Mapping[str, Any]:
    matches = [arm for arm in study["arms"] if arm.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one prepared arm {arm_id!r}.")
    return matches[0]


def _validate_prepared_arm(
    study: Mapping[str, Any],
    *,
    arm_id: str,
    filename: str,
) -> Path:
    arm = _arm_record(study, arm_id)
    config = (CONFIG_ROOT / filename).resolve()
    records = arm.get("configs")
    if (
        arm.get("experiment_id") != "mnist_relu_drn_kd.v1"
        or arm.get("mode") != "train"
        or not isinstance(records, list)
        or len(records) != 1
        or not isinstance(records[0], Mapping)
        or Path(str(records[0].get("resolved_path"))).resolve() != config
        or records[0].get("sha256") != sha256_file(config)
    ):
        raise RuntimeError(
            f"Expected prepared arm {arm_id!r} to match {filename!r}."
        )
    return config


def _validate_controls() -> None:
    for budget, control in CONTROL_RUNS.items():
        run = Path(control["run"])
        result = run / "result.json"
        report = run / "artifacts" / "ibm_om_recovery.report.json"
        status = _read_json(run / "status.json")
        payload = _read_json(result)
        if (
            status is None
            or status.get("status") != "complete"
            or payload is None
            or payload.get("status") != "complete"
            or not result.is_file()
            or sha256_file(result) != control["result_sha256"]
            or not report.is_file()
            or sha256_file(report) != control["report_sha256"]
        ):
            raise RuntimeError(
                f"Expected the exact completed unthresholded control for budget {budget}."
            )


def _records(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = run / "metrics.jsonl"
    values = [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    initial = [value for value in values if value.get("mode") == "initialization"]
    epochs = [value for value in values if value.get("mode") == "train"]
    if len(initial) != 1 or len(epochs) != 10:
        raise RuntimeError("Expected one initialization and ten epoch records.")
    return initial[0], epochs


def _accuracy(record: Mapping[str, Any], state: str) -> float:
    return float(
        record["validation"]["recalibrated_gain"][state]["student_accuracy"]
    )


def _initial_accuracy(record: Mapping[str, Any], state: str) -> float:
    return float(record["recalibrated_gain"][state]["student_accuracy"])


def _comparison(study_dir: Path) -> Path:
    controls = {}
    for budget, control in CONTROL_RUNS.items():
        run = Path(control["run"])
        initial, epochs = _records(run)
        final = epochs[-1]
        best = max(
            epochs,
            key=lambda value: (
                _accuracy(value, "apparent"),
                -int(value["completed_epochs"]),
            ),
        )
        controls[str(budget)] = {
            "run_dir": str(run),
            "result_sha256": control["result_sha256"],
            "report_sha256": control["report_sha256"],
            "initial_apparent_accuracy": _initial_accuracy(initial, "apparent"),
            "initial_persistent_accuracy": _initial_accuracy(initial, "persistent"),
            "terminal_apparent_accuracy": _accuracy(final, "apparent"),
            "terminal_persistent_accuracy": _accuracy(final, "persistent"),
            "best_apparent_accuracy": _accuracy(best, "apparent"),
            "best_epoch": int(best["completed_epochs"]),
            "requested_pulses": int(final["recovery"]["slow"]["total_requested"]),
            "unique_cells_touched": int(
                final["recovery"]["slow"]["unique_cells_touched"]
            ),
            "logical_sign_flips": int(
                final["recovery"]["apparent_endpoint"]["logical_sign_flips"]
            ),
        }

    rows = []
    thresholds_by_percentile: dict[float, dict[str, float]] = {}
    sample_digest: dict[str, str] | None = None
    for arm_id, _filename, percentile, budget in THRESHOLD_ARMS:
        completed = _completed_runs(study_dir / "runs" / arm_id)
        if len(completed) != 1:
            raise RuntimeError(f"Expected one completed threshold run for {arm_id!r}.")
        run = completed[0]
        initial, epochs = _records(run)
        final = epochs[-1]
        best = max(
            epochs,
            key=lambda value: (
                _accuracy(value, "apparent"),
                -int(value["completed_epochs"]),
            ),
        )
        recovery = final["recovery"]
        gate = recovery["direct_gradient_gate"]
        calibration = gate["calibration"]
        probability = recovery["direct_probability_calibration"]
        thresholds = {
            str(key): float(value)
            for key, value in gate["threshold_by_parameter"].items()
        }
        digests = {
            str(key): str(value)
            for key, value in calibration["sample_digest_by_parameter"].items()
        }
        if float(gate["percentile"]) != percentile:
            raise RuntimeError("Expected the configured threshold percentile in report.")
        previous = thresholds_by_percentile.setdefault(percentile, thresholds)
        if previous != thresholds:
            raise RuntimeError("Expected exact threshold parity across budgets.")
        if sample_digest is None:
            sample_digest = digests
        elif sample_digest != digests:
            raise RuntimeError("Expected one exact calibration sample across all arms.")
        if probability["algorithm"] != (
            "frozen_unthresholded_control_scale_after_gradient_gate"
        ):
            raise RuntimeError("Expected a frozen, non-redistributed probability scale.")
        if not abs(
            float(probability["ungated_control"]["expected_full_run_pulses"])
            - float(recovery["slow_pulse_cap"])
        ) <= 1e-6:
            raise RuntimeError("Expected exact ungated control-scale parity.")
        control = controls[str(budget)]
        terminal_apparent = _accuracy(final, "apparent")
        requested = int(recovery["slow"]["total_requested"])
        sign_flips = int(
            recovery["apparent_endpoint"]["logical_sign_flips"]
        )
        rows.append(
            {
                "arm_id": arm_id,
                "run_dir": str(run),
                "result_sha256": sha256_file(run / "result.json"),
                "percentile": percentile,
                "budget_per_cell": budget,
                "initial_apparent_accuracy": _initial_accuracy(initial, "apparent"),
                "initial_persistent_accuracy": _initial_accuracy(initial, "persistent"),
                "terminal_apparent_accuracy": terminal_apparent,
                "terminal_persistent_accuracy": _accuracy(final, "persistent"),
                "terminal_apparent_change_from_initial": terminal_apparent
                - _initial_accuracy(initial, "apparent"),
                "terminal_apparent_change_from_control": terminal_apparent
                - float(control["terminal_apparent_accuracy"]),
                "best_apparent_accuracy": _accuracy(best, "apparent"),
                "best_epoch": int(best["completed_epochs"]),
                "requested_pulses": requested,
                "pulses_saved_vs_control": int(control["requested_pulses"])
                - requested,
                "state_changing_pulses": int(
                    recovery["slow"]["state_changing"]
                ),
                "unique_cells_touched": int(
                    recovery["slow"]["unique_cells_touched"]
                ),
                "logical_sign_flips": sign_flips,
                "logical_sign_flips_change_vs_control": sign_flips
                - int(control["logical_sign_flips"]),
                "threshold_by_parameter": thresholds,
                "calibration_sha256": calibration["calibration_sha256"],
                "sample_digest_by_parameter": digests,
                "gate_cumulative_by_parameter": gate[
                    "cumulative_by_parameter"
                ],
                "probability_calibration": probability,
            }
        )

    for key in ("base.dense_weight.0", "base.dense_weight.1"):
        values = [thresholds_by_percentile[value][key] for value in (90.0, 95.0, 99.0)]
        if not values[0] <= values[1] <= values[2]:
            raise RuntimeError("Expected monotone p90 <= p95 <= p99 thresholds.")
    if any(row["initial_apparent_accuracy"] != 0.9274 for row in rows):
        raise RuntimeError("Expected exact source apparent accuracy in every arm.")
    if any(row["initial_persistent_accuracy"] != 0.4584 for row in rows):
        raise RuntimeError("Expected exact source persistent accuracy in every arm.")

    path = study_dir / "analysis" / "gradient_threshold_comparison.json"
    atomic_write_json(
        path,
        {
            "schema": "ebl.ibm_om.direct_gradient_threshold_comparison",
            "schema_version": 1,
            "study_id": STUDY_ID,
            "source_deployment_sha256": SOURCE_DEPLOYMENT_SHA256,
            "generated_at": _utc_now(),
            "sample_digest_by_parameter": sample_digest,
            "thresholds_by_percentile": {
                str(percentile): thresholds
                for percentile, thresholds in sorted(
                    thresholds_by_percentile.items()
                )
            },
            "unthresholded_controls": controls,
            "threshold_arms": rows,
        },
    )
    return path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_dir = (
        args.study_dir
        if args.study_dir is not None
        else _ROOT / "results" / STUDY_ID
    ).expanduser().resolve()
    if _CUDA_DEVICES.fullmatch(args.cuda_visible_devices) is None:
        raise RuntimeError("Expected a valid CUDA device list.")
    _validate_file(SOURCE_WEIGHTS, SOURCE_WEIGHTS_SHA256, role="source weights")
    _validate_file(TEACHER_WEIGHTS, TEACHER_WEIGHTS_SHA256, role="teacher")
    _validate_file(DEVICE_MODEL, DEVICE_MODEL_SHA256, role="device model")
    _validate_file(
        SOURCE_DEPLOYMENT,
        SOURCE_DEPLOYMENT_SHA256,
        role="source deployment",
    )
    _validate_controls()
    study = load_study_record(study_dir)
    if study.get("study_id") != STUDY_ID:
        raise RuntimeError("Expected the exact prepared threshold study.")
    configs = {
        arm_id: _validate_prepared_arm(
            study,
            arm_id=arm_id,
            filename=filename,
        )
        for arm_id, filename, _percentile, _budget in THRESHOLD_ARMS
    }

    attempt_dir = (
        study_dir
        / "launcher"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "PYTHONUNBUFFERED": "1",
        }
    )
    state: dict[str, Any] = {
        "schema": "ebl.ibm_om.direct_gradient_threshold_launcher",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "running",
        "started_at": _utc_now(),
        "heartbeat_at": _utc_now(),
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "launcher_pid": os.getpid(),
        "cuda_visible_devices": args.cuda_visible_devices,
        "source_deployment": {
            "path": str(SOURCE_DEPLOYMENT),
            "sha256": SOURCE_DEPLOYMENT_SHA256,
        },
        "expected_arms": [item[0] for item in THRESHOLD_ARMS],
        "active_arm": None,
        "arms": {},
    }
    atomic_write_json(attempt_dir / "status.json", state)
    try:
        for arm_id, _filename, _percentile, _budget in THRESHOLD_ARMS:
            _run_one(
                (
                    sys.executable,
                    "-m",
                    "ebl",
                    "train",
                    "--config",
                    str(configs[arm_id]),
                    "--weights",
                    str(SOURCE_WEIGHTS),
                    "--deployment",
                    str(SOURCE_DEPLOYMENT),
                    "--teacher-weights",
                    str(TEACHER_WEIGHTS),
                    "--device-model",
                    str(DEVICE_MODEL),
                    "--output-dir",
                    str(study_dir / "runs" / arm_id),
                ),
                arm_id=arm_id,
                study_dir=study_dir,
                attempt_dir=attempt_dir,
                environment=environment,
                state=state,
            )
        comparison = _comparison(study_dir)
        state["comparison"] = {
            "path": str(comparison),
            "sha256": sha256_file(comparison),
        }
        atomic_write_json(attempt_dir / "status.json", state)
        summary_log = attempt_dir / "study-summary.log"
        with summary_log.open("w", encoding="utf-8") as output:
            summary = subprocess.run(
                (
                    sys.executable,
                    "-m",
                    "ebl",
                    "study",
                    "summarize",
                    "--study-dir",
                    str(study_dir),
                    "--verify-artifacts",
                ),
                cwd=_ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if summary.returncode != 0:
            raise RuntimeError("Expected final study artifact verification to pass.")
        state["status"] = "complete"
        state["completed_at"] = _utc_now()
        state["active_arm"] = None
        state["summary_log"] = str(summary_log)
        atomic_write_json(attempt_dir / "status.json", state)
        return 0
    except BaseException as error:
        state["status"] = "failed"
        state["active_arm"] = None
        state["failed_at"] = _utc_now()
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_write_json(attempt_dir / "status.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_ROOT",
    "CONTROL_RUNS",
    "SOURCE_DEPLOYMENT",
    "SOURCE_DEPLOYMENT_SHA256",
    "STUDY_ID",
    "STUDY_PLAN",
    "THRESHOLD_ARMS",
    "main",
]
