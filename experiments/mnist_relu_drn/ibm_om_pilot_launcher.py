"""Persistent local launcher for the predeclared IBM OM DRN training phase.

This launcher is deliberately narrower than the study: it starts the four
training arms concurrently and leaves the sixteen predeclared held-out
validation arms untouched.  A successful launcher result therefore requires
the study summary to be ``incomplete`` with exactly those validation arms
pending; it must not mistake partial study coverage for completed evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

from ebl.runtime.store import atomic_write_json, sha256_file
from experiments.current_simulations import refresh_current_simulations_for_run
from ebl.lifecycle.study import load_study_record


_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "mnist-ibm-om-hwa-program-verify-pilot-20260822-v1"
STUDY_PLAN = _ROOT / "studies" / f"{STUDY_ID}.json"
CONFIG_ROOT = _ROOT / "examples" / "mnist_relu_drn" / "ibm_om_hwa_pilot"

TRAINING_ARM_CONFIGS = {
    "train-clean": "clean.json",
    "train-gaussian-3pct": "gaussian_3pct.json",
    "train-om-repaired": "om_repaired.json",
    "train-om-published": "om_published.json",
}
_TRAINING_LABELS = (
    "clean",
    "gaussian-3pct",
    "om-repaired",
    "om-published",
)
_HELDOUT_ASSIGNMENTS = (83101, 83102, 83103, 83104)
VALIDATION_ARM_CONFIGS = {
    f"test-{label}-{assignment}": f"heldout_assignment_{assignment}.json"
    for assignment in _HELDOUT_ASSIGNMENTS
    for label in _TRAINING_LABELS
}

BOUNDED_CHECKPOINT = (
    _ROOT
    / "results"
    / "mnist_bounded_drn_teacher_seed17_10ep_6aa32237"
    / "20260820T140222.931115Z-809aec60-8da7bc60"
    / "checkpoints"
    / "weights.pt"
)
BOUNDED_CHECKPOINT_SHA256 = (
    "f0036b36cebe970c5105c22ebe703d53fec99b7716215cca0e55ee09cfaf70cf"
)
DEVICE_MODEL = _ROOT / "data" / "ibm_reram_om_pv128_hwa_v1.json"
DEVICE_MODEL_SHA256 = (
    "3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3"
)

HEARTBEAT_SECONDS = 15.0
EXPECTED_EPOCHS_PER_ARM = 10
_CUDA_VISIBLE_DEVICES = re.compile(r"[0-9]+(?:,[0-9]+)*\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the four predeclared IBM OM DRN training arms. The "
            "sixteen held-out validation arms remain pending."
        )
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
        help=(
            "prepared study directory (defaults to the canonical results "
            "directory for this study)"
        ),
    )
    parser.add_argument(
        "--aihwkit-python",
        type=Path,
        required=True,
        help=(
            "executable for the pinned AIHWKit environment used only by the "
            "external array-population sampler"
        ),
    )
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _native_attempts(arm_root: Path) -> tuple[Path, ...]:
    if not arm_root.is_dir():
        return ()
    return tuple(sorted(arm_root.iterdir()))


def _latest_native_run(arm_root: Path) -> Path | None:
    candidates = [
        child
        for child in _native_attempts(arm_root)
        if child.is_dir() and (child / "status.json").is_file()
    ]
    return max(candidates, key=lambda path: path.name) if candidates else None


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(1 for _line in stream)


def _arm_snapshot(
    arm: str,
    *,
    study_dir: Path,
    process: subprocess.Popen[str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    run_dir = _latest_native_run(study_dir / "runs" / arm)
    status = _read_json(run_dir / "status.json") if run_dir is not None else None
    metrics = run_dir / "metrics.jsonl" if run_dir is not None else None
    weights = (
        run_dir / "checkpoints" / "weights.pt" if run_dir is not None else None
    )
    resume = (
        run_dir / "checkpoints" / "resume.pt" if run_dir is not None else None
    )
    return {
        "arm_id": arm,
        "pid": process.pid,
        "launcher_returncode": process.poll(),
        "native_run_dir": str(run_dir) if run_dir is not None else None,
        "native_status": status.get("status") if status else None,
        "native_started_at": status.get("started_at") if status else None,
        "native_finished_at": status.get("finished_at") if status else None,
        "metrics_size_bytes": (
            metrics.stat().st_size if metrics is not None and metrics.exists() else 0
        ),
        "metrics_lines": _line_count(metrics) if metrics is not None else 0,
        "weights_size_bytes": (
            weights.stat().st_size if weights is not None and weights.exists() else 0
        ),
        "resume_size_bytes": (
            resume.stat().st_size if resume is not None and resume.exists() else 0
        ),
        "stdout_size_bytes": stdout_path.stat().st_size,
        "stderr_size_bytes": stderr_path.stat().st_size,
    }


def _validate_input_file(path: Path, *, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(
            f"Expected the frozen {label} to be a file. Provided value: {str(path)!r}."
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"Expected the frozen {label} SHA-256 to be {expected_sha256!r}. "
            f"Provided value: {observed!r} at {str(path)!r}."
        )


def _validate_arm_record(
    arm: Mapping[str, Any],
    *,
    arm_id: str,
    mode: str,
    config_path: Path,
) -> None:
    if arm.get("arm_id") != arm_id:
        raise RuntimeError(f"Expected the prepared arm record for {arm_id!r}.")
    if arm.get("experiment_id") != "mnist_relu_drn_kd.v1" or arm.get("mode") != mode:
        raise RuntimeError(
            f"Expected arm {arm_id!r} to declare mnist_relu_drn_kd.v1/{mode}."
        )
    configs = arm.get("configs")
    if not isinstance(configs, list) or len(configs) != 1:
        raise RuntimeError(
            f"Expected arm {arm_id!r} to declare exactly one config."
        )
    record = configs[0]
    if not isinstance(record, Mapping):
        raise RuntimeError(f"Expected arm {arm_id!r} config to be a record.")
    resolved_path = record.get("resolved_path")
    if not isinstance(resolved_path, str) or Path(resolved_path).resolve() != config_path:
        raise RuntimeError(
            f"Expected arm {arm_id!r} to declare config {str(config_path)!r}."
        )
    if not config_path.is_file() or record.get("sha256") != sha256_file(config_path):
        raise RuntimeError(
            f"Expected arm {arm_id!r} config digest to match its prepared record."
        )


def _validate_prepared_study(
    study_dir: Path,
    *,
    checkpoint_path: Path = BOUNDED_CHECKPOINT,
    checkpoint_sha256: str = BOUNDED_CHECKPOINT_SHA256,
    device_model_path: Path = DEVICE_MODEL,
    device_model_sha256: str = DEVICE_MODEL_SHA256,
) -> Mapping[str, Any]:
    study_dir = study_dir.expanduser().resolve()
    study = load_study_record(study_dir)
    if study.get("study_id") != STUDY_ID:
        raise RuntimeError(
            f"Expected --study-dir to be the prepared {STUDY_ID!r} study."
        )

    source_plan = study.get("source_plan")
    if not isinstance(source_plan, Mapping):
        raise RuntimeError("Expected the prepared study to record its source plan.")
    if Path(str(source_plan.get("path"))).resolve() != STUDY_PLAN.resolve():
        raise RuntimeError("Expected the prepared study to name the tracked pilot plan.")
    if source_plan.get("sha256") != sha256_file(STUDY_PLAN):
        raise RuntimeError(
            "Expected the prepared study to match the current tracked pilot plan."
        )

    arms_raw = study.get("arms")
    if not isinstance(arms_raw, list):
        raise RuntimeError("Expected the prepared study to contain arm records.")
    arms = {
        str(arm.get("arm_id")): arm
        for arm in arms_raw
        if isinstance(arm, Mapping)
    }
    expected_arm_ids = set(TRAINING_ARM_CONFIGS) | set(VALIDATION_ARM_CONFIGS)
    if set(arms) != expected_arm_ids or len(arms_raw) != len(expected_arm_ids):
        raise RuntimeError(
            "Expected the prepared pilot to declare exactly the four training "
            "and sixteen held-out validation arms."
        )
    for arm_id, config_name in TRAINING_ARM_CONFIGS.items():
        _validate_arm_record(
            arms[arm_id],
            arm_id=arm_id,
            mode="train",
            config_path=(CONFIG_ROOT / config_name).resolve(),
        )
    for arm_id, config_name in VALIDATION_ARM_CONFIGS.items():
        _validate_arm_record(
            arms[arm_id],
            arm_id=arm_id,
            mode="validate",
            config_path=(CONFIG_ROOT / config_name).resolve(),
        )

    runs_root = study_dir / "runs"
    observed_arm_roots = {
        path.name for path in runs_root.iterdir() if path.is_dir()
    } if runs_root.is_dir() else set()
    if observed_arm_roots != expected_arm_ids:
        raise RuntimeError(
            "Expected runs/ to contain exactly one prepared directory per "
            "declared pilot arm."
        )
    existing_attempts = {
        arm_id: [str(path) for path in _native_attempts(runs_root / arm_id)]
        for arm_id in sorted(expected_arm_ids)
        if _native_attempts(runs_root / arm_id)
    }
    launch_root = study_dir / "launch"
    prior_launches = (
        [str(path) for path in sorted(launch_root.iterdir())]
        if launch_root.is_dir()
        else []
    )
    if existing_attempts or prior_launches:
        raise RuntimeError(
            "Expected a fresh prepared pilot with no native or launcher "
            "attempts. Existing attempts are immutable; recovery must use a "
            "separate reviewed launcher. "
            f"Provided value: native={existing_attempts!r}, launch={prior_launches!r}."
        )

    _validate_input_file(
        checkpoint_path.resolve(),
        expected_sha256=checkpoint_sha256,
        label="bounded-DRN checkpoint",
    )
    _validate_input_file(
        device_model_path.resolve(),
        expected_sha256=device_model_sha256,
        label="IBM OM device-model bundle",
    )
    return study


def _commands(
    *,
    python: Path,
    study_dir: Path,
    checkpoint_path: Path = BOUNDED_CHECKPOINT,
    device_model_path: Path = DEVICE_MODEL,
) -> dict[str, list[str]]:
    return {
        arm: [
            str(python),
            "-m",
            "ebl",
            "train",
            "--config",
            str((CONFIG_ROOT / config_name).resolve()),
            "--output-dir",
            str((study_dir / "runs" / arm).resolve()),
            "--teacher-weights",
            str(checkpoint_path.resolve()),
            "--device-model",
            str(device_model_path.resolve()),
        ]
        for arm, config_name in TRAINING_ARM_CONFIGS.items()
    }


def _validate_training_phase_summary(summary: Mapping[str, Any]) -> None:
    if (
        summary.get("study_id") != STUDY_ID
        or summary.get("state") != "incomplete"
        or summary.get("ready_for_review") is not False
        or summary.get("validation_mode") != "full_artifact_hashes"
    ):
        raise RuntimeError(
            "Expected the completed training phase to leave the full study "
            "in the artifact-verified incomplete state."
        )
    raw_arms = summary.get("arms")
    if not isinstance(raw_arms, list):
        raise RuntimeError("Expected the study summary to contain arm records.")
    arms = {
        str(arm.get("arm_id")): arm
        for arm in raw_arms
        if isinstance(arm, Mapping)
    }
    expected_ids = set(TRAINING_ARM_CONFIGS) | set(VALIDATION_ARM_CONFIGS)
    if set(arms) != expected_ids or len(raw_arms) != len(expected_ids):
        raise RuntimeError("Expected the study summary to cover every declared arm.")
    for arm_id in TRAINING_ARM_CONFIGS:
        arm = arms[arm_id]
        if not (
            arm.get("expected_runs") == 1
            and arm.get("complete") == 1
            and arm.get("running") == 0
            and arm.get("failed") == 0
            and arm.get("invalid") == 0
            and arm.get("coverage_complete") is True
        ):
            raise RuntimeError(
                f"Expected training arm {arm_id!r} to have one valid complete run."
            )
    for arm_id in VALIDATION_ARM_CONFIGS:
        arm = arms[arm_id]
        if not (
            arm.get("expected_runs") == 1
            and arm.get("complete") == 0
            and arm.get("running") == 0
            and arm.get("failed") == 0
            and arm.get("invalid") == 0
            and arm.get("coverage_complete") is False
        ):
            raise RuntimeError(
                f"Expected held-out validation arm {arm_id!r} to remain pending."
            )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw_study_dir = args.study_dir or _ROOT / "results" / STUDY_ID
    study_dir = raw_study_dir.expanduser().resolve()
    aihwkit_python = args.aihwkit_python.expanduser().resolve()
    if not aihwkit_python.is_file() or not os.access(aihwkit_python, os.X_OK):
        raise RuntimeError("Expected --aihwkit-python to be executable.")
    if _CUDA_VISIBLE_DEVICES.fullmatch(args.cuda_visible_devices) is None:
        raise RuntimeError(
            "Expected --cuda-visible-devices to be a comma-separated list "
            "of non-negative integer device indices."
        )
    study = _validate_prepared_study(study_dir)

    attempt_dir = study_dir / "launch" / _attempt_id()
    attempt_dir.mkdir(parents=True, exist_ok=False)
    launcher_python = Path(sys.executable).resolve()
    commands = _commands(python=launcher_python, study_dir=study_dir)
    input_records = {
        "bounded_drn_checkpoint": {
            "path": str(BOUNDED_CHECKPOINT.resolve()),
            "sha256": BOUNDED_CHECKPOINT_SHA256,
            "size_bytes": BOUNDED_CHECKPOINT.stat().st_size,
        },
        "ibm_om_device_model": {
            "path": str(DEVICE_MODEL.resolve()),
            "sha256": DEVICE_MODEL_SHA256,
            "size_bytes": DEVICE_MODEL.stat().st_size,
        },
    }
    contract = {
        "schema": "ebl.mnist_ibm_om_hwa.training_launcher",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "phase": "training",
        "formal_evidence": True,
        "study_json": {
            "path": str(study_dir / "study.json"),
            "sha256": sha256_file(study_dir / "study.json"),
            "source_plan_sha256": study["source_plan"]["sha256"],
        },
        "launcher_python": str(launcher_python),
        "aihwkit_python": str(aihwkit_python),
        "cuda_visible_devices": args.cuda_visible_devices,
        "concurrent_arms": True,
        "expected_training_arms": list(TRAINING_ARM_CONFIGS),
        "expected_epochs_per_arm": EXPECTED_EPOCHS_PER_ARM,
        "expected_pending_validation_arms": list(VALIDATION_ARM_CONFIGS),
        "expected_study_state_after_phase": "incomplete",
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "inputs": input_records,
        "configs": {
            arm: {
                "path": command[5],
                "sha256": sha256_file(Path(command[5])),
            }
            for arm, command in commands.items()
        },
        "commands": commands,
        "created_at": _utc_now(),
    }
    atomic_write_json(attempt_dir / "contract.json", contract)
    atomic_write_json(
        attempt_dir / "launcher_started.json",
        {
            "status": "launching",
            "started_at": _utc_now(),
            "expected_training_arms": list(TRAINING_ARM_CONFIGS),
        },
    )

    environment = os.environ.copy()
    environment["EBL_AIHWKIT_PYTHON"] = str(aihwkit_python)
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment["EBL_DEFER_CURRENT_SIMULATIONS"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    processes: dict[str, subprocess.Popen[str]] = {}
    stdout_logs: dict[str, Any] = {}
    stderr_logs: dict[str, Any] = {}
    stdout_paths: dict[str, Path] = {}
    stderr_paths: dict[str, Path] = {}
    stop_requested = False

    def request_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
        nonlocal stop_requested
        stop_requested = True
        for process in processes.values():
            _terminate_process(process)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    try:
        for arm, command in commands.items():
            stdout_path = attempt_dir / f"{arm}.stdout.log"
            stderr_path = attempt_dir / f"{arm}.stderr.log"
            stdout_paths[arm] = stdout_path
            stderr_paths[arm] = stderr_path
            stdout_log = stdout_path.open("w", encoding="utf-8", buffering=1)
            stderr_log = stderr_path.open("w", encoding="utf-8", buffering=1)
            stdout_logs[arm] = stdout_log
            stderr_logs[arm] = stderr_log
            process = subprocess.Popen(
                command,
                cwd=_ROOT,
                env=environment,
                stdout=stdout_log,
                stderr=stderr_log,
                text=True,
                start_new_session=True,
            )
            processes[arm] = process
            atomic_write_json(
                attempt_dir / f"{arm}.launch.json",
                {
                    "arm_id": arm,
                    "pid": process.pid,
                    "command": command,
                    "started_at": _utc_now(),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                },
            )
        atomic_write_json(
            attempt_dir / "processes.json",
            {
                "status": "running",
                "started_at": _utc_now(),
                "pids": {arm: process.pid for arm, process in processes.items()},
            },
        )

        while any(process.poll() is None for process in processes.values()):
            atomic_write_json(
                attempt_dir / "heartbeat.json",
                {
                    "updated_at": _utc_now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "stop_requested": stop_requested,
                    "arms": [
                        _arm_snapshot(
                            arm,
                            study_dir=study_dir,
                            process=processes[arm],
                            stdout_path=stdout_paths[arm],
                            stderr_path=stderr_paths[arm],
                        )
                        for arm in TRAINING_ARM_CONFIGS
                    ],
                },
            )
            if all(
                _latest_native_run(study_dir / "runs" / arm) is not None
                for arm in TRAINING_ARM_CONFIGS
            ):
                refresh_current_simulations_for_run(
                    repo_root=_ROOT,
                    run_dir=study_dir / "runs" / next(iter(TRAINING_ARM_CONFIGS)),
                )
            time.sleep(HEARTBEAT_SECONDS)

        returncodes = {arm: process.wait() for arm, process in processes.items()}
        atomic_write_json(
            attempt_dir / "heartbeat.json",
            {
                "updated_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "stop_requested": stop_requested,
                "arms": [
                    _arm_snapshot(
                        arm,
                        study_dir=study_dir,
                        process=processes[arm],
                        stdout_path=stdout_paths[arm],
                        stderr_path=stderr_paths[arm],
                    )
                    for arm in TRAINING_ARM_CONFIGS
                ],
            },
        )
        summarize = [
            str(launcher_python),
            "-m",
            "ebl",
            "study",
            "summarize",
            "--study-dir",
            str(study_dir),
            "--verify-artifacts",
        ]
        summary_process = subprocess.run(
            summarize,
            cwd=_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        (attempt_dir / "study-summarize.stdout.log").write_text(
            summary_process.stdout,
            encoding="utf-8",
        )
        (attempt_dir / "study-summarize.stderr.log").write_text(
            summary_process.stderr,
            encoding="utf-8",
        )
        summary = _read_json(study_dir / "analysis" / "summary.json")
        phase_error: str | None = None
        try:
            if summary_process.returncode != 0 or summary is None:
                raise RuntimeError("Study summarization did not produce a valid summary.")
            _validate_training_phase_summary(summary)
        except RuntimeError as error:
            phase_error = str(error)

        success = (
            not stop_requested
            and all(value == 0 for value in returncodes.values())
            and summary_process.returncode == 0
            and phase_error is None
        )
        atomic_write_json(
            attempt_dir / "launcher_result.json",
            {
                "status": "training_phase_complete" if success else "failed",
                "finished_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "returncodes": returncodes,
                "study_summarize_returncode": summary_process.returncode,
                "study_state": summary.get("state") if summary else None,
                "pending_validation_arms": list(VALIDATION_ARM_CONFIGS),
                "phase_validation_error": phase_error,
                "arms": [
                    _arm_snapshot(
                        arm,
                        study_dir=study_dir,
                        process=processes[arm],
                        stdout_path=stdout_paths[arm],
                        stderr_path=stderr_paths[arm],
                    )
                    for arm in TRAINING_ARM_CONFIGS
                ],
            },
        )
        refresh_current_simulations_for_run(
            repo_root=_ROOT,
            run_dir=study_dir / "runs" / next(iter(TRAINING_ARM_CONFIGS)),
        )
        return 0 if success else 1
    except BaseException as error:
        for process in processes.values():
            _terminate_process(process)
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        atomic_write_json(
            attempt_dir / "launcher_result.json",
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "returncodes": {
                    arm: process.poll() for arm, process in processes.items()
                },
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )
        refresh_current_simulations_for_run(
            repo_root=_ROOT,
            run_dir=study_dir / "runs" / next(iter(TRAINING_ARM_CONFIGS)),
        )
        raise
    finally:
        for log in stdout_logs.values():
            log.close()
        for log in stderr_logs.values():
            log.close()


if __name__ == "__main__":  # pragma: no cover - exercised as a launcher
    raise SystemExit(main())
