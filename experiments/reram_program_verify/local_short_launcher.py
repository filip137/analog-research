"""Persistent local launcher for the reviewed short ReRAM production study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

from core.artifacts import atomic_write_json, sha256_file
from experiments.current_simulations import refresh_current_simulations_for_run


_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "ibm-reram-program-verify-noise-20260821-v2"
ARM_CONFIGS = {
    "om-continuous": "production_short_om_continuous.json",
    "om-corrupt": "production_short_om_corrupt.json",
    "hfo2-continuous": "production_short_hfo2_continuous.json",
    "hfo2-corrupt": "production_short_hfo2_corrupt.json",
}
CAP128_STUDY_ID = "ibm-reram-program-verify-noise-20260822-v3"
CAP128_ARM_CONFIGS = {
    "om-continuous": "production_short_cap128_om_continuous.json",
    "om-corrupt": "production_short_cap128_om_corrupt.json",
    "hfo2-continuous": "production_short_cap128_hfo2_continuous.json",
    "hfo2-corrupt": "production_short_cap128_hfo2_corrupt.json",
}
HWA_STUDY_ID = "ibm-reram-om-hwa-device-model-20260822-v1"
HWA_ARM_CONFIGS = {
    "om-continuous": "hwa_production_cap128_om_continuous.json",
    "om-published": "hwa_production_cap128_om_corrupt.json",
}
EXPECTED_TRAJECTORIES_PER_ARM = 671_744
HWA_EXPECTED_TRAJECTORIES_PER_ARM = 201_392
HEARTBEAT_SECONDS = 15.0


@dataclass(frozen=True)
class LaunchPlan:
    study_id: str
    arm_configs: Mapping[str, str]
    maximum_program_pulses: int
    expected_trajectories_per_arm: int


LAUNCH_PLANS = {
    "v2": LaunchPlan(
        STUDY_ID,
        ARM_CONFIGS,
        512,
        EXPECTED_TRAJECTORIES_PER_ARM,
    ),
    "v3-cap128": LaunchPlan(
        CAP128_STUDY_ID,
        CAP128_ARM_CONFIGS,
        128,
        EXPECTED_TRAJECTORIES_PER_ARM,
    ),
    "hwa-om-cap128": LaunchPlan(
        HWA_STUDY_ID,
        HWA_ARM_CONFIGS,
        128,
        HWA_EXPECTED_TRAJECTORIES_PER_ARM,
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-profile",
        choices=tuple(LAUNCH_PLANS),
        default="v2",
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
    )
    parser.add_argument(
        "--aihwkit-python",
        type=Path,
        required=True,
    )
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _latest_native_run(arm_root: Path) -> Path | None:
    candidates = [
        child
        for child in arm_root.iterdir()
        if child.is_dir() and (child / "status.json").is_file()
    ] if arm_root.is_dir() else []
    return max(candidates, key=lambda path: path.name) if candidates else None


def _require_clean_source_commit(repo_root: Path = _ROOT) -> str:
    """Return the launch commit after failing closed on local source changes."""

    try:
        commit = subprocess.check_output(
            ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
            stderr=subprocess.PIPE,
        ).decode("ascii").strip()
        status = subprocess.check_output(
            (
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "-z",
            ),
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        raise RuntimeError(
            "Expected the formal four-arm launcher to resolve a clean Git "
            "source commit."
        ) from error
    if not commit or status:
        raise RuntimeError(
            "Expected the formal four-arm launcher to use a clean worktree. "
            "Commit or otherwise preserve every source change before launch."
        )
    return commit


def _arm_snapshot(
    arm: str,
    *,
    study_dir: Path,
    process: subprocess.Popen[str],
) -> dict[str, Any]:
    run_dir = _latest_native_run(study_dir / "runs" / arm)
    status = _read_json(run_dir / "status.json") if run_dir is not None else None
    database = (
        run_dir / "artifacts" / "trajectories.sqlite3"
        if run_dir is not None
        else None
    )
    metrics = run_dir / "metrics.jsonl" if run_dir is not None else None
    return {
        "arm_id": arm,
        "pid": process.pid,
        "launcher_returncode": process.poll(),
        "native_run_dir": str(run_dir) if run_dir is not None else None,
        "native_status": status.get("status") if status else None,
        "native_started_at": status.get("started_at") if status else None,
        "native_finished_at": status.get("finished_at") if status else None,
        "database_size_bytes": (
            database.stat().st_size if database is not None and database.exists() else 0
        ),
        "metrics_size_bytes": (
            metrics.stat().st_size if metrics is not None and metrics.exists() else 0
        ),
    }


def _validate_prepared_study(
    study_dir: Path,
    *,
    study_id: str = STUDY_ID,
    arm_configs: Mapping[str, str] = ARM_CONFIGS,
) -> None:
    study = _read_json(study_dir / "study.json")
    if study is None or study.get("study_id") != study_id:
        raise RuntimeError(
            f"Expected --study-dir to be the prepared {study_id!r} study."
        )
    declared = {
        str(arm.get("arm_id"))
        for arm in study.get("arms", [])
        if isinstance(arm, Mapping)
    }
    if declared != set(arm_configs):
        raise RuntimeError(
            "Expected the prepared short study to declare exactly the reviewed arms."
        )
    existing = []
    for arm in arm_configs:
        run_dir = _latest_native_run(study_dir / "runs" / arm)
        if run_dir is not None:
            existing.append(str(run_dir))
    if existing:
        raise RuntimeError(
            "Expected the formal launcher to start from a prepared "
            f"study with no native attempts. Found {existing!r}. Recovery must "
            "retain existing attempts and launch only missing/failed arms."
        )


def _commands(
    *,
    python: Path,
    study_dir: Path,
    arm_configs: Mapping[str, str] = ARM_CONFIGS,
) -> dict[str, list[str]]:
    config_root = _ROOT / "examples" / "reram_program_verify"
    return {
        arm: [
            str(python),
            "-m",
            "ebl",
            "characterize",
            "--config",
            str(config_root / config_name),
            "--output-dir",
            str(study_dir / "runs" / arm),
        ]
        for arm, config_name in arm_configs.items()
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    launch_plan = LAUNCH_PLANS[args.study_profile]
    raw_study_dir = args.study_dir or _ROOT / "results" / launch_plan.study_id
    study_dir = raw_study_dir.expanduser().resolve()
    aihwkit_python = args.aihwkit_python.expanduser().resolve()
    if not aihwkit_python.is_file() or not os.access(aihwkit_python, os.X_OK):
        raise RuntimeError("Expected --aihwkit-python to be executable.")
    source_commit = _require_clean_source_commit()
    _validate_prepared_study(
        study_dir,
        study_id=launch_plan.study_id,
        arm_configs=launch_plan.arm_configs,
    )

    attempt_dir = study_dir / "launch" / _attempt_id()
    attempt_dir.mkdir(parents=True, exist_ok=False)
    commands = _commands(
        python=Path(sys.executable).resolve(),
        study_dir=study_dir,
        arm_configs=launch_plan.arm_configs,
    )
    contract = {
        "schema": "ebl.ibm_reram.local_short_launcher",
        "schema_version": 1,
        "study_id": launch_plan.study_id,
        "launch_profile": args.study_profile,
        "maximum_program_pulses": launch_plan.maximum_program_pulses,
        "formal_evidence": True,
        "source_commit": source_commit,
        "launcher_python": str(Path(sys.executable).resolve()),
        "aihwkit_python": str(aihwkit_python),
        "cuda_visible_devices": args.cuda_visible_devices,
        "expected_arms": list(launch_plan.arm_configs),
        "expected_trajectories_per_arm": (
            launch_plan.expected_trajectories_per_arm
        ),
        "expected_total_trajectories": (
            launch_plan.expected_trajectories_per_arm
            * len(launch_plan.arm_configs)
        ),
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "sizing_evidence": {
            "host": "nom-cool-2",
            "gpu": "NVIDIA GeForce RTX 3090",
            "source_profile": "production_short",
            "five_target_elapsed_seconds": 61.57074950297829,
            "five_target_trajectories": 81_920,
            "five_target_verify_events": 2_863_651,
            "projection_method": (
                "naive trajectory-count scaling from a broader-profile gate; "
                "operational estimate only"
            ),
            "projected_seconds_per_arm": (
                61.57074950297829
                * launch_plan.expected_trajectories_per_arm
                / 81_920
            ),
            "ten_x_safety_seconds_parallel_independent_arms": (
                61.57074950297829
                * launch_plan.expected_trajectories_per_arm
                / 81_920
                * 10.0
            ),
        },
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

    environment = os.environ.copy()
    environment["EBL_AIHWKIT_PYTHON"] = str(aihwkit_python)
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment["EBL_DEFER_CURRENT_SIMULATIONS"] = "1"
    processes: dict[str, subprocess.Popen[str]] = {}
    logs: dict[str, Any] = {}
    stop_requested = False

    def request_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
        nonlocal stop_requested
        stop_requested = True
        for process in processes.values():
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    try:
        for arm, command in commands.items():
            log = (attempt_dir / f"{arm}.log").open(
                "w", encoding="utf-8", buffering=1
            )
            logs[arm] = log
            process = subprocess.Popen(
                command,
                cwd=_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes[arm] = process
            atomic_write_json(
                attempt_dir / f"{arm}.launch.json",
                {
                    "arm_id": arm,
                    "pid": process.pid,
                    "command": command,
                    "started_at": _utc_now(),
                    "log": str(attempt_dir / f"{arm}.log"),
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
                        )
                        for arm in launch_plan.arm_configs
                    ],
                },
            )
            if all(
                _latest_native_run(study_dir / "runs" / arm) is not None
                for arm in launch_plan.arm_configs
            ):
                refresh_current_simulations_for_run(
                    repo_root=_ROOT,
                    run_dir=(
                        study_dir
                        / "runs"
                        / next(iter(launch_plan.arm_configs))
                    ),
                )
            time.sleep(HEARTBEAT_SECONDS)

        returncodes = {arm: process.wait() for arm, process in processes.items()}
        summarize = [
            str(Path(sys.executable).resolve()),
            "-m",
            "ebl",
            "study",
            "summarize",
            "--study-dir",
            str(study_dir),
        ]
        summary = subprocess.run(
            summarize,
            cwd=_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        (attempt_dir / "study-summarize.log").write_text(
            summary.stdout + summary.stderr,
            encoding="utf-8",
        )
        success = all(value == 0 for value in returncodes.values()) and summary.returncode == 0
        atomic_write_json(
            attempt_dir / "launcher_result.json",
            {
                "status": "complete" if success else "failed",
                "started_monotonic_reference": started,
                "finished_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "returncodes": returncodes,
                "study_summarize_returncode": summary.returncode,
                "arms": [
                    _arm_snapshot(
                        arm,
                        study_dir=study_dir,
                        process=processes[arm],
                    )
                    for arm in launch_plan.arm_configs
                ],
            },
        )
        refresh_current_simulations_for_run(
            repo_root=_ROOT,
            run_dir=(
                study_dir / "runs" / next(iter(launch_plan.arm_configs))
            ),
        )
        return 0 if success else 1
    except BaseException as error:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
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
            run_dir=(
                study_dir / "runs" / next(iter(launch_plan.arm_configs))
            ),
        )
        raise
    finally:
        for log in logs.values():
            log.close()


if __name__ == "__main__":  # pragma: no cover - exercised as a launcher
    raise SystemExit(main())
