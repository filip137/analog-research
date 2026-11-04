"""Sequential watchdog launcher for the one-seed deployed-recovery pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from ebl.runtime.store import atomic_write_json, sha256_file
from ebl.lifecycle.study import load_study_record


_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = (
    "mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1"
)
STUDY_PLAN = _ROOT / "studies" / f"{STUDY_ID}.json"
CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_reset_relative_on_chip_recovery"
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
SOURCE_ARM = "deploy-source-development"
SOURCE_CONFIG = "development_deployment.json"
RECOVERY_ARMS = (
    ("rail-refresh-budget-0p1", "rail_refresh_budget_0p1.json"),
    ("rail-refresh-budget-1", "rail_refresh_budget_1.json"),
    ("rail-refresh-budget-4", "rail_refresh_budget_4.json"),
    ("direct-pulse-budget-0p1", "direct_pulse_budget_0p1.json"),
    ("direct-pulse-budget-1", "direct_pulse_budget_1.json"),
    ("direct-pulse-budget-4", "direct_pulse_budget_4.json"),
    ("tiki-taka-ideal-budget-0p1", "tiki_taka_ideal_budget_0p1.json"),
    ("tiki-taka-ideal-budget-1", "tiki_taka_ideal_budget_1.json"),
    ("tiki-taka-ideal-budget-4", "tiki_taka_ideal_budget_4.json"),
    ("tiki-taka-physical-budget-0p1", "tiki_taka_physical_budget_0p1.json"),
    ("tiki-taka-physical-budget-1", "tiki_taka_physical_budget_1.json"),
    ("tiki-taka-physical-budget-4", "tiki_taka_physical_budget_4.json"),
)
HEARTBEAT_SECONDS = 15.0
_CUDA_DEVICES = re.compile(r"[0-9]+(?:,[0-9]+)*\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the source deployment and twelve recovery arms sequentially "
            "with a durable semantic heartbeat."
        )
    )
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--aihwkit-python", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _validate_file(path: Path, digest: str, *, role: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(
            f"Expected the exact frozen {role} at {str(path)!r}."
        )


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
    mode: str,
) -> Path:
    arm = _arm_record(study, arm_id)
    config = (CONFIG_ROOT / filename).resolve()
    records = arm.get("configs")
    if (
        arm.get("experiment_id") != "mnist_relu_drn_kd.v1"
        or arm.get("mode") != mode
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


def _native_runs(arm_root: Path) -> list[Path]:
    if not arm_root.is_dir():
        return []
    return sorted(
        child
        for child in arm_root.iterdir()
        if child.is_dir() and (child / "status.json").is_file()
    )


def _completed_runs(arm_root: Path) -> list[Path]:
    completed = []
    for run in _native_runs(arm_root):
        status = _read_json(run / "status.json")
        result = _read_json(run / "result.json")
        if (
            status is not None
            and status.get("status") == "complete"
            and result is not None
            and result.get("status") == "complete"
        ):
            completed.append(run)
    return completed


def _latest_snapshot(arm_root: Path) -> dict[str, Any]:
    runs = _native_runs(arm_root)
    run = runs[-1] if runs else None
    status = _read_json(run / "status.json") if run is not None else None
    metrics = run / "metrics.jsonl" if run is not None else None
    return {
        "native_run_dir": str(run) if run is not None else None,
        "native_status": status.get("status") if status else None,
        "native_error": status.get("error") if status else None,
        "metrics_size_bytes": (
            metrics.stat().st_size
            if metrics is not None and metrics.is_file()
            else 0
        ),
        "metrics_modified_at": (
            datetime.fromtimestamp(
                metrics.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            if metrics is not None and metrics.is_file()
            else None
        ),
    }


def _source_deployment(run: Path) -> Path:
    result = _read_json(run / "result.json")
    artifacts = result.get("artifacts") if result is not None else None
    matches = [
        artifact
        for artifact in artifacts or ()
        if isinstance(artifact, Mapping)
        and artifact.get("kind") == "ibm_om_persistent_deployment"
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one source deployment artifact record.")
    path = (run / str(matches[0]["path"])).resolve()
    if (
        not path.is_file()
        or matches[0].get("sha256") != sha256_file(path)
    ):
        raise RuntimeError("Expected the source deployment artifact hash to match.")
    return path


def _run_one(
    command: Sequence[str],
    *,
    arm_id: str,
    study_dir: Path,
    attempt_dir: Path,
    environment: Mapping[str, str],
    state: dict[str, Any],
) -> Path:
    arm_root = study_dir / "runs" / arm_id
    completed = _completed_runs(arm_root)
    if len(completed) > 1:
        raise RuntimeError(f"Expected unambiguous completed coverage for {arm_id!r}.")
    if len(completed) == 1:
        state["arms"][arm_id] = {
            "status": "already_complete",
            "native_run_dir": str(completed[0]),
            "finished_at": _utc_now(),
        }
        atomic_write_json(attempt_dir / "status.json", state)
        return completed[0]

    stdout_path = attempt_dir / f"{arm_id}.stdout.log"
    stderr_path = attempt_dir / f"{arm_id}.stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            tuple(command),
            cwd=_ROOT,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        state["active_arm"] = arm_id
        state["arms"][arm_id] = {
            "status": "running",
            "pid": process.pid,
            "command": list(command),
            "started_at": _utc_now(),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        while process.poll() is None:
            state["heartbeat_at"] = _utc_now()
            state["arms"][arm_id].update(_latest_snapshot(arm_root))
            atomic_write_json(attempt_dir / "status.json", state)
            time.sleep(HEARTBEAT_SECONDS)
        returncode = int(process.returncode)
    state["heartbeat_at"] = _utc_now()
    state["arms"][arm_id].update(_latest_snapshot(arm_root))
    state["arms"][arm_id]["returncode"] = returncode
    state["arms"][arm_id]["finished_at"] = _utc_now()
    completed = _completed_runs(arm_root)
    if returncode != 0 or len(completed) != 1:
        state["arms"][arm_id]["status"] = "failed"
        state["status"] = "failed"
        state["active_arm"] = None
        atomic_write_json(attempt_dir / "status.json", state)
        raise RuntimeError(
            f"Expected arm {arm_id!r} to produce one completed native run; "
            f"returncode={returncode}, completed={len(completed)}."
        )
    state["arms"][arm_id]["status"] = "complete"
    state["arms"][arm_id]["native_run_dir"] = str(completed[0])
    state["active_arm"] = None
    atomic_write_json(attempt_dir / "status.json", state)
    return completed[0]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_dir = (
        args.study_dir
        if args.study_dir is not None
        else _ROOT / "results" / STUDY_ID
    ).expanduser().resolve()
    aihwkit_python = args.aihwkit_python.expanduser().resolve()
    if (
        _CUDA_DEVICES.fullmatch(args.cuda_visible_devices) is None
        or not aihwkit_python.is_file()
        or not os.access(aihwkit_python, os.X_OK)
    ):
        raise RuntimeError("Expected valid CUDA devices and AIHWKit executable.")
    _validate_file(SOURCE_WEIGHTS, SOURCE_WEIGHTS_SHA256, role="source weights")
    _validate_file(TEACHER_WEIGHTS, TEACHER_WEIGHTS_SHA256, role="teacher")
    _validate_file(DEVICE_MODEL, DEVICE_MODEL_SHA256, role="device model")
    study = load_study_record(study_dir)
    if study.get("study_id") != STUDY_ID:
        raise RuntimeError("Expected the exact prepared recovery study.")
    source_config = _validate_prepared_arm(
        study,
        arm_id=SOURCE_ARM,
        filename=SOURCE_CONFIG,
        mode="validate",
    )
    recovery_configs = {
        arm_id: _validate_prepared_arm(
            study,
            arm_id=arm_id,
            filename=filename,
            mode="train",
        )
        for arm_id, filename in RECOVERY_ARMS
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
            "EBL_AIHWKIT_PYTHON": str(aihwkit_python),
            "PYTHONUNBUFFERED": "1",
        }
    )
    state: dict[str, Any] = {
        "schema": "ebl.ibm_om.reset_relative_recovery_launcher",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "running",
        "started_at": _utc_now(),
        "heartbeat_at": _utc_now(),
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "launcher_pid": os.getpid(),
        "cuda_visible_devices": args.cuda_visible_devices,
        "aihwkit_python": str(aihwkit_python),
        "expected_arms": [SOURCE_ARM]
        + [arm_id for arm_id, _filename in RECOVERY_ARMS],
        "active_arm": None,
        "arms": {},
    }
    atomic_write_json(attempt_dir / "status.json", state)
    try:
        source_run = _run_one(
            (
                sys.executable,
                "-m",
                "ebl",
                "validate",
                "--config",
                str(source_config),
                "--weights",
                str(SOURCE_WEIGHTS),
                "--teacher-weights",
                str(TEACHER_WEIGHTS),
                "--device-model",
                str(DEVICE_MODEL),
                "--output-dir",
                str(study_dir / "runs" / SOURCE_ARM),
            ),
            arm_id=SOURCE_ARM,
            study_dir=study_dir,
            attempt_dir=attempt_dir,
            environment=environment,
            state=state,
        )
        deployment = _source_deployment(source_run)
        state["source_deployment"] = {
            "path": str(deployment),
            "sha256": sha256_file(deployment),
        }
        atomic_write_json(attempt_dir / "status.json", state)
        for arm_id, _filename in RECOVERY_ARMS:
            _run_one(
                (
                    sys.executable,
                    "-m",
                    "ebl",
                    "train",
                    "--config",
                    str(recovery_configs[arm_id]),
                    "--weights",
                    str(SOURCE_WEIGHTS),
                    "--deployment",
                    str(deployment),
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
    "DEVICE_MODEL",
    "DEVICE_MODEL_SHA256",
    "RECOVERY_ARMS",
    "SOURCE_ARM",
    "SOURCE_CONFIG",
    "SOURCE_WEIGHTS",
    "SOURCE_WEIGHTS_SHA256",
    "STUDY_ID",
    "STUDY_PLAN",
    "TEACHER_WEIGHTS",
    "TEACHER_WEIGHTS_SHA256",
    "main",
]
