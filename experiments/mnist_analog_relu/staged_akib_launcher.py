"""Akib-only sequential launcher for the staged 784-256-10 OM campaign.

This module is orchestration, not a numerical execution surface.  Every
teacher and crossbar task enters through the public ``python -m ebl`` CLI and
writes into a canonical prepared-study arm.  Failed attempts are retained.
Completed attempts are reused only after their source, config, inputs,
registered artifacts, CUDA receipt, and staged physical ancestry all replay.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from functools import lru_cache
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from experiments.mnist_analog_relu.generate_staged_campaign import (
    ADAM_GRID,
    CONFIG_ROOT,
    DEVELOPMENT_PLAN_ID,
    PRODUCTION_ENDPOINTS,
    PRODUCTION_PLAN_ID,
    TEACHER_PLAN_ID,
)
from experiments.mnist_analog_relu.staged_analysis import (
    grouped_assignment_summary,
    load_selection_receipt,
    select_adam_hyperparameters,
    validate_selection_receipt,
    write_selection_receipt,
)
from experiments.mnist_analog_relu.staged_artifacts import (
    load_device_state,
    load_hwa_master,
)
from experiments.mnist_analog_relu.staged_config import (
    LiteralAdamHyperparameters,
    OnChipAdamStageSettings,
    parse_staged_crossbar_config,
)
from experiments.study_workflow import load_study_plan, load_study_record
from training.ibm_reram_hwa import load_om_array_population
from training.ibm_reram_program_verify import OM_PRESET
from training.ibm_om_standard_crossbar import tensor_sha256


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HOSTNAME = "integnano-akib"
EXPECTED_AIHWKIT_VERSION = "1.1.0"
TASK_PYTHON = Path("/home/filiposana/miniconda3/envs/py312/bin/python")
AIHWKIT_PYTHON = Path(
    "/home/filiposana/miniconda3/envs/aihwkit-sampler/bin/python"
)
MNIST_ROOT = Path("/home/filiposana/datasets/mnist")
RESULTS_ROOT = ROOT / "results"
HEARTBEAT_SECONDS = 10.0
MAXIMUM_HEARTBEAT_SECONDS = 15.0

TEACHER_CONFIG = ROOT / "examples" / "mnist_relu" / "teacher_784_256_10_cuda.json"
PLAN_PATHS = {
    TEACHER_PLAN_ID: ROOT / "studies" / f"{TEACHER_PLAN_ID}.json",
    DEVELOPMENT_PLAN_ID: ROOT / "studies" / f"{DEVELOPMENT_PLAN_ID}.json",
    PRODUCTION_PLAN_ID: ROOT / "studies" / f"{PRODUCTION_PLAN_ID}.json",
}

HWA_KIND = "crossbar_hwa_master"
DEPLOYED_KIND = "crossbar_deployed_state_bundle"
CORRUPTED_KIND = "crossbar_corrupted_state_bundle"
ADAM_FINAL_KIND = "crossbar_adam_final_state_bundle"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _require_executable(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Expected {label} to be executable: {resolved}.")
    return resolved


def _require_akib_hostname(hostname: str | None = None) -> str:
    observed = platform.node() if hostname is None else hostname
    if observed != EXPECTED_HOSTNAME:
        raise RuntimeError(
            "Expected the formal staged campaign to run only on "
            f"{EXPECTED_HOSTNAME!r}; observed hostname {observed!r}."
        )
    return observed


def _require_clean_source_commit(repo_root: Path = ROOT) -> str:
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
        raise RuntimeError("Expected a clean resolvable Git launch commit.") from error
    if not commit or status:
        raise RuntimeError(
            "Expected the staged Akib launcher to use a clean Git commit. "
            "Commit every source, generated config, plan, and test first."
        )
    return commit


@dataclass(frozen=True)
class TaskBlueprint:
    study_id: str
    arm_id: str
    label: str
    config: Path
    mode: str = "train"
    stage_kind: str | None = None
    source_kind: str | None = None
    assignment_seed: int | None = None
    endpoint_seed: int | None = None
    start_state: str | None = None
    learning_rate: float | None = None
    pulse_cap_per_cell: int | None = None
    expected_artifact_kind: str | None = None

    @property
    def output_arm(self) -> str:
        return self.arm_id


@dataclass(frozen=True)
class TaskInputs:
    teacher_weights: Path | None = None
    weights: Path | None = None
    device_state: Path | None = None
    selection_receipt: Path | None = None
    resume: Path | None = None

    def cli_pairs(self) -> tuple[tuple[str, Path], ...]:
        result: list[tuple[str, Path]] = []
        for option, value in (
            ("--weights", self.weights),
            ("--teacher-weights", self.teacher_weights),
            ("--device-state", self.device_state),
            ("--selection-receipt", self.selection_receipt),
            ("--resume", self.resume),
        ):
            if value is not None:
                result.append((option, value.expanduser().resolve()))
        return tuple(result)

    def manifest_hashes(
        self,
        *,
        task: TaskBlueprint | None = None,
        sampler: Path | None = None,
    ) -> dict[str, str]:
        records: dict[str, Path] = {}
        weights_role = (
            "weights"
            if task is not None and task.study_id == TEACHER_PLAN_ID
            else "hwa_master"
        )
        for role, value in (
            ("teacher_weights", self.teacher_weights),
            (weights_role, self.weights),
            ("origin_device_state", self.device_state),
            ("adam_selection_receipt", self.selection_receipt),
            ("adam_epoch_resume", self.resume),
            ("aihwkit_python", sampler),
        ):
            if value is not None:
                records[role] = value.expanduser().resolve()
        result: dict[str, str] = {}
        for role, path in records.items():
            if not path.is_file():
                raise FileNotFoundError(f"Expected {role} input artifact: {path}.")
            result[role] = sha256_file(path)
        return result


@dataclass(frozen=True)
class RunHandle:
    task: TaskBlueprint
    run_dir: Path
    manifest: Mapping[str, Any]
    result: Mapping[str, Any]
    reused: bool

    def artifact(self, kind: str) -> Path:
        records = [
            item
            for item in self.result.get("artifacts", [])
            if isinstance(item, Mapping) and item.get("kind") == kind
        ]
        if len(records) != 1:
            raise RuntimeError(
                f"Expected one registered {kind!r} artifact in {self.run_dir}; "
                f"found {len(records)}."
            )
        return (self.run_dir / str(records[0]["path"])).resolve()


def _load_staged_config(path: Path) -> Any:
    payload = _read_json(path)
    if payload is None:
        raise RuntimeError(f"Expected a strict staged JSON config: {path}.")
    return parse_staged_crossbar_config(payload)


def _rate_token(rate: float) -> str:
    return {3e-4: "3e4", 1e-3: "1e3", 3e-3: "3e3"}[float(rate)]


def _cap_token(cap: int | None) -> str:
    return "cap128" if cap == 128 else "uncapped"


_UNSET = object()


@lru_cache(maxsize=None)
def _find_staged_config(
    directory: Path,
    *,
    stage_kind: str,
    assignment_seed: int | None = None,
    endpoint_seed: int | None = None,
    source_kind: str | None = None,
    start_state: str | None = None,
    learning_rate: object = _UNSET,
    pulse_cap_per_cell: object = _UNSET,
) -> Path:
    """Resolve one generated config by its scientific declaration, not its name."""

    matches: list[Path] = []
    for path in sorted(directory.glob("*.json")):
        payload = _read_json(path)
        if payload is None or not isinstance(payload.get("stage"), Mapping):
            continue
        stage = payload["stage"]
        device = payload.get("device")
        if stage.get("kind") != stage_kind or not isinstance(device, Mapping):
            continue
        if assignment_seed is not None and device.get("assignment_seed") != assignment_seed:
            continue
        if endpoint_seed is not None and device.get("endpoint_seed") != endpoint_seed:
            continue
        if source_kind is not None and stage.get("source_kind") != source_kind:
            continue
        if start_state is not None and stage.get("start_state") != start_state:
            continue
        hyperparameters = stage.get("hyperparameters")
        if learning_rate is not _UNSET:
            if not isinstance(hyperparameters, Mapping) or hyperparameters.get("learning_rate") != learning_rate:
                continue
        if pulse_cap_per_cell is not _UNSET:
            if not isinstance(hyperparameters, Mapping) or hyperparameters.get("pulse_cap_per_cell") != pulse_cap_per_cell:
                continue
        matches.append(path.resolve())
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one generated config for scientific stage identity; "
            f"directory={directory}, kind={stage_kind}, source={source_kind}, "
            f"start={start_state}, assignment={assignment_seed}, endpoint={endpoint_seed}, "
            f"learning_rate={learning_rate!r}, cap={pulse_cap_per_cell!r}, "
            f"matches={matches!r}."
        )
    return matches[0]


def teacher_blueprints() -> tuple[TaskBlueprint, ...]:
    return (
        TaskBlueprint(
            study_id=TEACHER_PLAN_ID,
            arm_id="teacher",
            label="teacher.train",
            config=TEACHER_CONFIG,
            expected_artifact_kind="selected_named_weights",
        ),
        TaskBlueprint(
            study_id=TEACHER_PLAN_ID,
            arm_id="teacher-test",
            label="teacher.full-test",
            config=TEACHER_CONFIG,
            mode="validate",
        ),
    )


def development_blueprints() -> tuple[TaskBlueprint, ...]:
    tuning = CONFIG_ROOT / "tuning"
    result = [
        TaskBlueprint(
            DEVELOPMENT_PLAN_ID,
            "offchip-hwa",
            "development.offchip-hwa",
            CONFIG_ROOT / "offchip_hwa.json",
            stage_kind="offchip_hwa",
            source_kind="teacher",
            expected_artifact_kind=HWA_KIND,
        ),
        TaskBlueprint(
            DEVELOPMENT_PLAN_ID,
            "tune-hwa-deploy",
            "development.hwa.deploy",
            tuning / "deploy_hwa_master.json",
            stage_kind="deploy",
            source_kind="hwa_master",
            expected_artifact_kind=DEPLOYED_KIND,
        ),
        TaskBlueprint(
            DEVELOPMENT_PLAN_ID,
            "tune-hwa-corrupt",
            "development.hwa.corrupt",
            _find_staged_config(
                tuning,
                stage_kind="apply_corruption",
                source_kind="hwa_master",
            ),
            stage_kind="apply_corruption",
            source_kind="hwa_master",
            expected_artifact_kind=CORRUPTED_KIND,
        ),
        TaskBlueprint(
            DEVELOPMENT_PLAN_ID,
            "tune-scratch-deploy",
            "development.scratch.deploy",
            tuning / "deploy_scratch.json",
            stage_kind="deploy",
            source_kind="scratch",
            expected_artifact_kind=DEPLOYED_KIND,
        ),
        TaskBlueprint(
            DEVELOPMENT_PLAN_ID,
            "tune-scratch-corrupt",
            "development.scratch.corrupt",
            _find_staged_config(
                tuning,
                stage_kind="apply_corruption",
                source_kind="scratch",
            ),
            stage_kind="apply_corruption",
            source_kind="scratch",
            expected_artifact_kind=CORRUPTED_KIND,
        ),
    ]
    starts = (
        ("tune-adam-hwa-healthy", "hwa_healthy_p0", "hwa_master"),
        ("tune-adam-hwa-corrupt", "hwa_published_fault", "hwa_master"),
        ("tune-adam-scratch-healthy", "scratch_healthy_p0", "scratch"),
        ("tune-adam-scratch-corrupt", "scratch_published_fault", "scratch"),
    )
    for rate, cap in ADAM_GRID:
        for arm, start, source in starts:
            config = _find_staged_config(
                tuning,
                stage_kind="on_chip_adam",
                start_state=start,
                learning_rate=rate,
                pulse_cap_per_cell=cap,
            )
            result.append(
                TaskBlueprint(
                    DEVELOPMENT_PLAN_ID,
                    arm,
                    f"development.{start}.lr{_rate_token(rate)}.{_cap_token(cap)}",
                    config,
                    stage_kind="on_chip_adam",
                    source_kind=source,
                    start_state=start,
                    learning_rate=rate,
                    pulse_cap_per_cell=cap,
                    expected_artifact_kind=ADAM_FINAL_KIND,
                )
            )
    return tuple(result)


def production_blueprints() -> tuple[TaskBlueprint, ...]:
    root = CONFIG_ROOT / "production"
    result: list[TaskBlueprint] = []
    for assignment, endpoints in PRODUCTION_ENDPOINTS.items():
        for endpoint in endpoints:
            def task(
                arm: str,
                *,
                stage: str,
                source: str,
                start: str | None = None,
                kind: str,
            ) -> TaskBlueprint:
                config = _find_staged_config(
                    root,
                    stage_kind=stage,
                    assignment_seed=assignment,
                    endpoint_seed=endpoint,
                    source_kind=(source if stage in {"deploy", "apply_corruption"} else None),
                    start_state=(start if stage == "on_chip_adam" else None),
                )
                return TaskBlueprint(
                    PRODUCTION_PLAN_ID,
                    arm,
                    f"production.{arm}.a{assignment}.e{endpoint}",
                    config,
                    stage_kind=stage,
                    source_kind=source,
                    assignment_seed=assignment,
                    endpoint_seed=endpoint,
                    start_state=start,
                    expected_artifact_kind=kind,
                )

            result.extend(
                (
                    task("direct-deploy", stage="deploy", source="teacher", kind=DEPLOYED_KIND),
                    task("direct-corrupt", stage="apply_corruption", source="teacher", kind=CORRUPTED_KIND),
                    task("hwa-deploy", stage="deploy", source="hwa_master", kind=DEPLOYED_KIND),
                    task("hwa-corrupt", stage="apply_corruption", source="hwa_master", kind=CORRUPTED_KIND),
                    task("hwa-adam-healthy", stage="on_chip_adam", source="hwa_master", start="hwa_healthy_p0", kind=ADAM_FINAL_KIND),
                    task("hwa-adam-corrupt", stage="on_chip_adam", source="hwa_master", start="hwa_published_fault", kind=ADAM_FINAL_KIND),
                    task("scratch-deploy-preparation", stage="deploy", source="scratch", kind=DEPLOYED_KIND),
                    task("scratch-corrupt-preparation", stage="apply_corruption", source="scratch", kind=CORRUPTED_KIND),
                    task("scratch-adam-healthy", stage="on_chip_adam", source="scratch", start="scratch_healthy_p0", kind=ADAM_FINAL_KIND),
                    task("scratch-adam-corrupt", stage="on_chip_adam", source="scratch", start="scratch_published_fault", kind=ADAM_FINAL_KIND),
                )
            )
    if len(result) != 160:
        raise RuntimeError(f"Expected exactly 160 production native runs; found {len(result)}.")
    return tuple(result)


def _native_command(
    task: TaskBlueprint,
    *,
    task_python: Path,
    study_dir: Path,
    inputs: TaskInputs,
) -> list[str]:
    if task.mode not in {"train", "validate"}:
        raise RuntimeError("Expected only public ebl train/validate campaign tasks.")
    command = [
        str(task_python.resolve()),
        "-m",
        "ebl",
        task.mode,
        "--config",
        str(task.config.resolve()),
        "--output-dir",
        str((study_dir / "runs" / task.output_arm).resolve()),
    ]
    for option, path in inputs.cli_pairs():
        command.extend((option, str(path)))
    return command


def _validate_blueprint_config(task: TaskBlueprint) -> str:
    if not task.config.is_file():
        raise RuntimeError(f"Expected generated config for {task.label}: {task.config}.")
    digest = sha256_file(task.config)
    if task.study_id == TEACHER_PLAN_ID:
        payload = _read_json(task.config)
        if (
            payload is None
            or payload.get("experiment_id") != "mnist_relu.v2"
            or payload.get("runtime", {}).get("device") != "cuda"
            or payload.get("model", {}).get("dims") != [784, 256, 10]
            or payload.get("modes", {}).get("train", {}).get("minimum_validation_accuracy") != 0.97
        ):
            raise RuntimeError("Expected the exact strict CUDA teacher config.")
        return digest
    document = _load_staged_config(task.config)
    stage = document.stage
    if document.runtime.device != "cuda" or stage.kind != task.stage_kind:
        raise RuntimeError(f"Blueprint/config mismatch for {task.label}.")
    if task.source_kind is not None:
        source = getattr(stage, "source_kind", task.source_kind)
        if stage.kind in {"deploy", "apply_corruption"} and source != task.source_kind:
            raise RuntimeError(f"Stage source mismatch for {task.label}.")
    if stage.kind == "on_chip_adam" and getattr(stage, "start_state", None) != task.start_state:
        raise RuntimeError(f"Adam start-state mismatch for {task.label}.")
    if task.assignment_seed is not None and (
        document.device.assignment_seed != task.assignment_seed
        or document.device.endpoint_seed != task.endpoint_seed
    ):
        raise RuntimeError(f"Production seed mismatch for {task.label}.")
    if stage.kind == "on_chip_adam" and task.learning_rate is not None:
        if not isinstance(stage, OnChipAdamStageSettings) or not isinstance(
            stage.hyperparameters, LiteralAdamHyperparameters
        ):
            raise RuntimeError(f"Expected literal tuning Adam config for {task.label}.")
        if (
            stage.hyperparameters.learning_rate != task.learning_rate
            or stage.hyperparameters.pulse_cap_per_cell != task.pulse_cap_per_cell
        ):
            raise RuntimeError(f"Tuning hyperparameter mismatch for {task.label}.")
    return digest


def _validate_declared_coverage(blueprints: Iterable[TaskBlueprint]) -> None:
    blueprints = tuple(blueprints)
    if len(blueprints) != 191 or len({task.label for task in blueprints}) != 191:
        raise RuntimeError(
            "Expected 191 uniquely labelled native tasks: 2 teacher, 29 development, "
            "and 160 production."
        )
    plans = {study_id: load_study_plan(path) for study_id, path in PLAN_PATHS.items()}
    declared: dict[tuple[str, str], set[str]] = {}
    for study_id, plan in plans.items():
        if plan["study_id"] != study_id:
            raise RuntimeError(f"Expected exact study identity {study_id!r}.")
        for arm in plan["arms"]:
            declared[(study_id, arm["arm_id"])] = {
                item["sha256"] for item in arm["configs"]
            }
    observed: dict[tuple[str, str], set[str]] = {}
    observed_counts: dict[tuple[str, str], int] = {}
    for task in blueprints:
        digest = _validate_blueprint_config(task)
        key = (task.study_id, task.arm_id)
        if digest not in declared.get(key, set()):
            raise RuntimeError(f"Task {task.label!r} is not predeclared by its study arm.")
        observed.setdefault(key, set()).add(digest)
        observed_counts[key] = observed_counts.get(key, 0) + 1
    if observed != declared:
        missing = {
            f"{study}/{arm}": sorted(hashes - observed.get((study, arm), set()))
            for (study, arm), hashes in declared.items()
            if hashes != observed.get((study, arm), set())
        }
        raise RuntimeError(f"Expected exact generated-plan task coverage; mismatches={missing!r}.")
    duplicate_counts = {
        f"{study}/{arm}": (observed_counts.get((study, arm), 0), len(hashes))
        for (study, arm), hashes in declared.items()
        if observed_counts.get((study, arm), 0) != len(hashes)
    }
    if duplicate_counts:
        raise RuntimeError(
            f"Expected exactly one native task per declared config; mismatches={duplicate_counts!r}."
        )


def _manifest_input_hashes(manifest: Mapping[str, Any]) -> dict[str, str] | None:
    raw = manifest.get("inputs")
    if not isinstance(raw, list):
        return None
    result: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        role = item.get("role")
        path_value = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(role, str)
            or not role
            or role in result
            or not isinstance(path_value, str)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            return None
        path = Path(path_value)
        if not path.is_file() or sha256_file(path) != digest:
            return None
        result[role] = digest
    return result


def _canonical_manifest_command(
    *,
    task: TaskBlueprint,
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> list[str] | None:
    """Reconstruct the exact logical argv that ``ebl.cli`` records.

    The parent launcher executes ``sys.executable -m ebl``, while
    ``ebl.cli.main`` intentionally stores the stable logical entry point
    ``ebl`` in each RunStore manifest.  Input paths come from the manifest
    records whose bytes and roles are authenticated separately.
    """

    raw_inputs = manifest.get("inputs")
    if not isinstance(raw_inputs, list):
        return None
    input_paths: dict[str, Path] = {}
    for item in raw_inputs:
        if not isinstance(item, Mapping):
            return None
        role = item.get("role")
        path = item.get("path")
        if (
            not isinstance(role, str)
            or role in input_paths
            or not isinstance(path, str)
        ):
            return None
        input_paths[role] = Path(path).expanduser().resolve()

    known_roles = {
        "weights",
        "hwa_master",
        "teacher_weights",
        "origin_device_state",
        "adam_selection_receipt",
        "adam_epoch_resume",
        "aihwkit_python",
    }
    if set(input_paths) - known_roles:
        return None

    command = [
        "ebl",
        task.mode,
        "--config",
        str(task.config.resolve()),
        "--output-dir",
        str(run_dir.parent.resolve()),
    ]
    weights_role = "weights" if task.study_id == TEACHER_PLAN_ID else "hwa_master"
    for option, role in (
        ("--weights", weights_role),
        ("--teacher-weights", "teacher_weights"),
        ("--device-state", "origin_device_state"),
        ("--selection-receipt", "adam_selection_receipt"),
        ("--resume", "adam_epoch_resume"),
    ):
        path = input_paths.get(role)
        if path is not None:
            command.extend((option, str(path)))
    return command


def _registered_artifacts_intact(run_dir: Path, result: Mapping[str, Any]) -> bool:
    records = result.get("artifacts")
    if not isinstance(records, list):
        return False
    root = run_dir.resolve()
    seen: set[str] = set()
    for item in records:
        if not isinstance(item, Mapping):
            return False
        relative = item.get("path")
        if not isinstance(relative, str) or not relative or relative in seen:
            return False
        seen.add(relative)
        try:
            path = (run_dir / relative).resolve(strict=True)
            path.relative_to(root)
        except (OSError, ValueError):
            return False
        if (
            not path.is_file()
            or item.get("size_bytes") != path.stat().st_size
            or item.get("sha256") != sha256_file(path)
            or not isinstance(item.get("kind"), str)
            or not item["kind"]
        ):
            return False
    return True


def _native_run_dirs(arm_root: Path) -> list[Path]:
    if not arm_root.is_dir():
        return []
    return sorted(
        path
        for path in arm_root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )


def _active_native_pids(task: TaskBlueprint, arm_root: Path) -> list[int]:
    """Find live public CLI processes for one exact config/output pair on Linux."""

    expected_config = str(task.config.resolve()).encode()
    expected_output = str(arm_root.resolve()).encode()
    result: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():  # pragma: no cover - Akib is Linux
        return result
    for candidate in proc.iterdir():
        if not candidate.name.isdigit() or int(candidate.name) == os.getpid():
            continue
        try:
            fields = (candidate / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (
            b"-m" in fields
            and b"ebl" in fields
            and expected_config in fields
            and expected_output in fields
        ):
            result.append(int(candidate.name))
    return sorted(result)


def _base_run_matches(
    *,
    task: TaskBlueprint,
    run_dir: Path,
    source_commit: str,
    config_sha256: str,
    expected_inputs: Mapping[str, str],
    allow_resume_input: bool,
    task_python: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]] | None:
    manifest = _read_json(run_dir / "manifest.json")
    if manifest is None:
        return None
    source = manifest.get("source")
    study = manifest.get("study")
    runtime = manifest.get("runtime")
    command = manifest.get("command")
    inputs = _manifest_input_hashes(manifest)
    canonical_command = _canonical_manifest_command(
        task=task,
        run_dir=run_dir,
        manifest=manifest,
    )
    command_config: Path | None = None
    command_output: Path | None = None
    if isinstance(command, list):
        try:
            command_config = Path(command[command.index("--config") + 1]).resolve()
            command_output = Path(command[command.index("--output-dir") + 1]).resolve()
        except (ValueError, IndexError, TypeError):
            pass
    if (
        manifest.get("experiment_id")
        != (
            "mnist_relu.v2"
            if task.study_id == TEACHER_PLAN_ID
            else "mnist_ibm_om_crossbar_relu.v2"
        )
        or not isinstance(source, Mapping)
        or source.get("commit") != source_commit
        or source.get("dirty") is not False
        or not isinstance(study, Mapping)
        or study.get("study_id") != task.study_id
        or study.get("arm_id") != task.arm_id
        or study.get("source_config_sha256") != config_sha256
        or inputs is None
        or command != canonical_command
        or command_config != task.config.resolve()
        or command_output != run_dir.parent.resolve()
        or (
            task_python is not None
            and (
                not isinstance(runtime, Mapping)
                or runtime.get("executable") != str(task_python.resolve())
            )
        )
    ):
        return None
    if allow_resume_input:
        base = {key: value for key, value in inputs.items() if key != "adam_epoch_resume"}
        if base != dict(expected_inputs) or set(inputs) - set(expected_inputs) - {"adam_epoch_resume"}:
            return None
    elif inputs != dict(expected_inputs):
        return None
    return manifest, inputs


def _validate_cuda_receipt(task: TaskBlueprint, result: Mapping[str, Any]) -> None:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"Expected terminal metrics for {task.label}.")
    runtime = metrics.get(
        "runtime_device" if task.study_id == TEACHER_PLAN_ID else "runtime"
    )
    device_name_key = (
        "device_name" if task.study_id == TEACHER_PLAN_ID else "cuda_device_name"
    )
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("configured_device") != "cuda"
        or not isinstance(runtime.get("resolved_device"), str)
        or not runtime["resolved_device"].startswith("cuda")
        or runtime.get("cuda_available") is not True
        or not isinstance(runtime.get(device_name_key), str)
        or not runtime[device_name_key]
    ):
        raise RuntimeError(f"Expected an explicit resolved CUDA receipt for {task.label}.")


def _validate_complete_bundle(
    *,
    task: TaskBlueprint,
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    status = _read_json(run_dir / "status.json")
    result = _read_json(run_dir / "result.json")
    resolved_config = _read_json(run_dir / "config.resolved.json")
    config_record = manifest.get("config")
    if (
        manifest.get("schema") != "ebl.run"
        or manifest.get("schema_version") != 1
        or manifest.get("run_id") != run_dir.name
        or not isinstance(config_record, Mapping)
        or config_record.get("path") != "config.resolved.json"
        or resolved_config is None
        or config_record.get("sha256") != content_hash(resolved_config)
        or status is None
        or status.get("status") != "complete"
        or result is None
        or result.get("status") != "complete"
        or result.get("run_id") != run_dir.name
        or result.get("experiment_id") != manifest.get("experiment_id")
        or not _registered_artifacts_intact(run_dir, result)
    ):
        raise RuntimeError(f"Expected an artifact-intact complete RunStore bundle: {run_dir}.")
    _validate_cuda_receipt(task, result)
    if task.expected_artifact_kind is not None:
        kinds = [
            item.get("kind")
            for item in result.get("artifacts", [])
            if isinstance(item, Mapping)
        ]
        if kinds.count(task.expected_artifact_kind) != 1:
            raise RuntimeError(
                f"Expected one {task.expected_artifact_kind!r} artifact in {run_dir}."
            )
    return result


def _completed_run(
    *,
    task: TaskBlueprint,
    arm_root: Path,
    source_commit: str,
    config_sha256: str,
    expected_inputs: Mapping[str, str],
    logical_adam_inputs: bool = False,
    task_python: Path | None = None,
) -> RunHandle | None:
    valid: list[RunHandle] = []
    conflicting_complete: list[Path] = []
    for run_dir in _native_run_dirs(arm_root):
        status = _read_json(run_dir / "status.json")
        if status is None or status.get("status") != "complete":
            continue
        matched = _base_run_matches(
            task=task,
            run_dir=run_dir,
            source_commit=source_commit,
            config_sha256=config_sha256,
            expected_inputs=expected_inputs,
            allow_resume_input=logical_adam_inputs,
            task_python=task_python,
        )
        if matched is None:
            manifest = _read_json(run_dir / "manifest.json")
            study = manifest.get("study") if manifest is not None else None
            if isinstance(study, Mapping) and study.get("source_config_sha256") == config_sha256:
                conflicting_complete.append(run_dir)
            continue
        manifest, _inputs = matched
        result = _validate_complete_bundle(task=task, run_dir=run_dir, manifest=manifest)
        valid.append(RunHandle(task, run_dir, manifest, result, True))
    if conflicting_complete:
        raise RuntimeError(
            "Refusing to mix source commits or input ancestry for completed config "
            f"{task.label!r}: {conflicting_complete!r}."
        )
    if len(valid) > 1:
        raise RuntimeError(f"Expected one exact completed run for {task.label}; found {len(valid)}.")
    return valid[0] if valid else None


def _result_metrics(handle: RunHandle) -> Mapping[str, Any]:
    metrics = handle.result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"Expected metrics in {handle.run_dir}.")
    return metrics


_HEADLINE_METRICS = (
    "student_accuracy",
    "cross_entropy",
    "kl_teacher_student",
    "teacher_agreement",
)
_EXPECTED_SPLIT_EXAMPLES = {"validation": 5_000, "test": 10_000}


def _finite_metric(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Expected finite {label}.")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Expected finite {label}.")
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _flatten_plant_evaluation(
    value: Any,
    *,
    label: str,
    require_test: bool,
) -> dict[str, float]:
    """Validate full apparent/persistent evaluation counts and flatten metrics."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"Expected {label} evaluation mapping.")
    expected_splits = {"validation", "test"} if require_test else {"validation"}
    if set(value) != expected_splits:
        raise RuntimeError(
            f"Expected {label} splits {sorted(expected_splits)!r}; found {sorted(value)!r}."
        )
    result: dict[str, float] = {}
    for split in sorted(expected_splits):
        split_value = value.get(split)
        if not isinstance(split_value, Mapping):
            raise RuntimeError(f"Expected {label}.{split} metrics.")
        for state_name, state_key in (
            ("apparent", "apparent_forward"),
            ("persistent", "persistent_diagnostic"),
        ):
            state = split_value.get(state_key)
            if (
                not isinstance(state, Mapping)
                or state.get("examples") != _EXPECTED_SPLIT_EXAMPLES[split]
            ):
                raise RuntimeError(
                    f"Expected {label}.{split}.{state_key} to cover exactly "
                    f"{_EXPECTED_SPLIT_EXAMPLES[split]} examples."
                )
            for metric in _HEADLINE_METRICS:
                result[f"{split}.{state_name}.{metric}"] = _finite_metric(
                    state.get(metric), label=f"{label}.{split}.{state_key}.{metric}"
                )
    return result


def _flatten_effective_evaluation(
    value: Any,
    *,
    label: str,
    require_test: bool,
) -> dict[str, float]:
    """Validate a logical/source effective-q evaluation and flatten metrics."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"Expected {label} evaluation mapping.")
    expected_splits = {"validation", "test"} if require_test else {"validation"}
    if set(value) != expected_splits:
        raise RuntimeError(
            f"Expected {label} splits {sorted(expected_splits)!r}; found {sorted(value)!r}."
        )
    result: dict[str, float] = {}
    for split in sorted(expected_splits):
        row = value.get(split)
        if (
            not isinstance(row, Mapping)
            or row.get("examples") != _EXPECTED_SPLIT_EXAMPLES[split]
        ):
            raise RuntimeError(
                f"Expected {label}.{split} to cover exactly "
                f"{_EXPECTED_SPLIT_EXAMPLES[split]} examples."
            )
        for metric in _HEADLINE_METRICS:
            result[f"{split}.{metric}"] = _finite_metric(
                row.get(metric), label=f"{label}.{split}.{metric}"
            )
    return result


def _validate_hwa_evaluation(
    value: Any,
    *,
    label: str,
    evaluation_id: str,
    expected_master_sha256: str,
) -> None:
    """Authenticate one validation-only held-apparent HWA evaluation."""

    expected_keys = {
        "network_forward_state",
        "primary_state",
        "diagnostic_state_role",
        "persistent_device_state_present",
        "apparent_forward",
        "nonpersistent_support_clamped_digital_master_diagnostic",
        "held_apparent_state_receipt",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError(f"Expected exact held-apparent HWA metrics for {label}.")
    apparent = value.get("apparent_forward")
    diagnostic = value.get(
        "nonpersistent_support_clamped_digital_master_diagnostic"
    )
    receipt = value.get("held_apparent_state_receipt")
    receipt_keys = {
        "schema",
        "schema_version",
        "evaluation_id",
        "resolved_seed",
        "resampling",
        "support_clamped_digital_master_q_sha256",
        "held_apparent_q_sha256",
        "write_noise_q_sha256",
        "generator_state_before_sha256",
        "generator_state_after_sha256",
        "configured_sigma_q",
        "observed_mean_q",
        "observed_std_q",
        "observed_minimum_q",
        "observed_maximum_q",
    }
    if (
        value.get("network_forward_state") != "held_apparent_q"
        or value.get("primary_state") != "held_apparent_q"
        or value.get("diagnostic_state_role")
        != "nonpersistent_support_clamped_digital_master_q"
        or value.get("persistent_device_state_present") is not False
        or not isinstance(apparent, Mapping)
        or not isinstance(diagnostic, Mapping)
        or apparent.get("examples") != 5_000
        or diagnostic.get("examples") != 5_000
        or not isinstance(receipt, Mapping)
        or set(receipt) != receipt_keys
        or receipt.get("schema")
        != "ebl.ibm_om_crossbar_held_apparent_hwa_evaluation"
        or receipt.get("schema_version") != 1
        or receipt.get("evaluation_id") != evaluation_id
        or receipt.get("resampling")
        != "one_full_array_draw_held_across_complete_evaluation_cohort"
        or receipt.get("support_clamped_digital_master_q_sha256")
        != expected_master_sha256
    ):
        raise RuntimeError(f"Expected authenticated apparent-primary HWA metrics for {label}.")
    for state_name, state in (("apparent", apparent), ("digital_master", diagnostic)):
        for metric in _HEADLINE_METRICS:
            _finite_metric(state.get(metric), label=f"{label}.{state_name}.{metric}")
    for name in (
        "held_apparent_q_sha256",
        "write_noise_q_sha256",
        "generator_state_before_sha256",
        "generator_state_after_sha256",
    ):
        digest = receipt.get(name)
        if not _is_sha256(digest):
            raise RuntimeError(f"Expected {label} to authenticate {name}.")
    if (
        isinstance(receipt.get("resolved_seed"), bool)
        or not isinstance(receipt.get("resolved_seed"), int)
    ):
        raise RuntimeError(f"Expected {label} to record its evaluation seed.")
    sigma = _finite_metric(receipt.get("configured_sigma_q"), label=f"{label}.sigma")
    observed_std = _finite_metric(
        receipt.get("observed_std_q"), label=f"{label}.observed_std"
    )
    observed_minimum = _finite_metric(
        receipt.get("observed_minimum_q"), label=f"{label}.observed_minimum"
    )
    observed_maximum = _finite_metric(
        receipt.get("observed_maximum_q"), label=f"{label}.observed_maximum"
    )
    _finite_metric(receipt.get("observed_mean_q"), label=f"{label}.observed_mean")
    if sigma <= 0.0 or observed_std < 0.0 or observed_minimum > observed_maximum:
        raise RuntimeError(f"Expected physically valid HWA noise statistics for {label}.")


def _verify_teacher(handle: RunHandle, *, require_test: bool = False) -> None:
    metrics = _result_metrics(handle)
    if require_test:
        if (
            metrics.get("split") != "test"
            or metrics.get("examples") != 10000
            or not isinstance(metrics.get("accuracy"), (int, float))
        ):
            raise RuntimeError("Expected one complete 10,000-example teacher test evaluation.")
        return
    selected = metrics.get("selected")
    gate = metrics.get("acceptance_gate")
    if (
        not isinstance(selected, Mapping)
        or not isinstance(selected.get("accuracy"), (int, float))
        or float(selected["accuracy"]) <= 0.97
        or not isinstance(gate, Mapping)
        or gate.get("comparison_operator") != ">"
        or gate.get("minimum_validation_accuracy") != 0.97
        or gate.get("passed") is not True
    ):
        raise RuntimeError("Expected selected teacher validation accuracy strictly above 97%.")


def _verify_hwa(handle: RunHandle, *, teacher_sha256: str) -> None:
    master = load_hwa_master(handle.artifact(HWA_KIND))
    metrics = _result_metrics(handle)
    report = metrics.get("hwa")
    final = metrics.get("final_evaluation")
    if (
        master.dims != (784, 256, 10)
        or master.teacher_sha256 != teacher_sha256
        or metrics.get("network_forward_state") != "held_apparent_q"
        or metrics.get("hidden_update_state") != "not_applicable_no_physical_plant"
        or metrics.get("updated_state") != "digital_master_q"
        or metrics.get("physical_plant_present") is not False
        or metrics.get("persistent_device_state_present") is not False
        or not isinstance(report, Mapping)
        or report.get("fixed_final_epoch") != 10
        or report.get("fixed_final_test") is not None
        or report.get("evaluation_forward_policy")
        != "one_sampled_held_apparent_q_per_evaluation"
        or report.get("primary_evaluation_state") != "held_apparent_q"
        or report.get("diagnostic_evaluation_state")
        != "nonpersistent_support_clamped_digital_master_q"
        or report.get("persistent_device_state_present_during_offchip_hwa")
        is not False
        or not isinstance(final, Mapping)
        or set(final) != {"validation"}
        or final.get("validation") != report.get("fixed_final_validation")
    ):
        raise RuntimeError("Expected the exact teacher-bound 784-256-10 HWA master.")
    epochs = report.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != 10:
        raise RuntimeError("Expected exactly ten validation-only HWA epochs.")
    _validate_hwa_evaluation(
        report.get("initial_validation"),
        label="hwa.initial.validation",
        evaluation_id="initial.validation",
        expected_master_sha256=report.get("initial_realized_sha256"),
    )
    for epoch_index, epoch in enumerate(epochs, start=1):
        if (
            not isinstance(epoch, Mapping)
            or epoch.get("epoch") != epoch_index
            or epoch.get("examples") != 55_000
            or epoch.get("batches") != 3_438
            or epoch.get("test") is not None
        ):
            raise RuntimeError("Expected the exact full-MNIST validation-only HWA protocol.")
        _validate_hwa_evaluation(
            epoch.get("validation"),
            label=f"hwa.epoch-{epoch_index}.validation",
            evaluation_id=f"epoch_{epoch_index:03d}.validation",
            expected_master_sha256=epoch.get("realized_sha256"),
        )


def _verify_device_output(
    handle: RunHandle,
    *,
    teacher_sha256: str,
    expected_role: str,
    expected_source_kind: str,
    origin: Path | None = None,
    hwa_master: Path | None = None,
) -> None:
    kind = {
        "healthy_p0": DEPLOYED_KIND,
        "faulted_p0": CORRUPTED_KIND,
        "adam_final": ADAM_FINAL_KIND,
    }[expected_role]
    path = handle.artifact(kind)
    state = load_device_state(path)
    task = handle.task
    document = _load_staged_config(task.config)
    expected_assignment = document.device.assignment_seed
    expected_endpoint = document.device.endpoint_seed
    if (
        state.role != expected_role
        or state.source_kind != expected_source_kind
        or state.teacher_sha256 != teacher_sha256
        or state.assignment_seed != expected_assignment
        or state.endpoint_seed != expected_endpoint
    ):
        raise RuntimeError(f"Expected exact physical identity for {task.label}.")
    if expected_source_kind == "teacher" and expected_role == "healthy_p0":
        if state.source_artifact_sha256 != teacher_sha256:
            raise RuntimeError("Expected direct deployment to bind the exact teacher bytes.")
    if expected_source_kind == "scratch" and state.source_artifact_sha256 is not None:
        raise RuntimeError("Expected scratch initialization not to claim source weights.")
    if hwa_master is not None and state.source_artifact_sha256 != sha256_file(hwa_master):
        raise RuntimeError("Expected HWA deployment to bind the exact HWA master bytes.")
    if origin is not None:
        origin_sha = sha256_file(origin)
        origin_state = load_device_state(origin)
        if (
            state.parent_device_state_sha256 != origin_sha
            or state.healthy_p0.plant_state_sha256
            != origin_state.healthy_p0.plant_state_sha256
            or state.assignment_seed != origin_state.assignment_seed
            or state.endpoint_seed != origin_state.endpoint_seed
            or state.source_kind != origin_state.source_kind
        ):
            raise RuntimeError(f"Expected exact same-array ancestry for {task.label}.")
        if expected_role == "adam_final":
            recovery = state.recovery
            if (
                not isinstance(recovery, Mapping)
                or recovery.get("origin_device_state_sha256") != origin_sha
                or recovery.get("completed_epochs") != 10
                or recovery.get("start_state") != task.start_state
            ):
                raise RuntimeError(f"Expected exact ten-epoch Adam ancestry for {task.label}.")


def _validate_resume(
    path: Path,
    *,
    task: TaskBlueprint,
    origin: Path,
    learning_rate: float,
    pulse_cap: int | None,
    expected_selection: Mapping[str, Any],
) -> int:
    state = load_device_state(path)
    recovery = state.recovery
    origin_state = load_device_state(origin)
    origin_sha = sha256_file(origin)
    required_recovery = {
        "schema",
        "schema_version",
        "origin_device_state_sha256",
        "start_state",
        "completed_epochs",
        "total_epochs",
        "learning_rate",
        "pulse_cap_per_cell",
        "optimizer_state_dict",
        "train_generator_state",
        "epoch_reports",
        "initial",
        "selection",
    }
    if (
        state.role != "adam_final"
        or state.source_kind != origin_state.source_kind
        or state.assignment_seed != origin_state.assignment_seed
        or state.endpoint_seed != origin_state.endpoint_seed
        or state.parent_device_state_sha256 != origin_sha
        or not isinstance(recovery, Mapping)
        or set(recovery) != required_recovery
        or recovery.get("schema") != "ebl.ibm_om_crossbar_adam_recovery"
        or recovery.get("schema_version") != 1
        or recovery.get("origin_device_state_sha256") != origin_sha
        or recovery.get("start_state") != task.start_state
        or recovery.get("learning_rate") != learning_rate
        or recovery.get("pulse_cap_per_cell") != pulse_cap
        or recovery.get("total_epochs") != 10
        or recovery.get("selection") != expected_selection
        or state.dims != origin_state.dims
        or state.teacher_sha256 != origin_state.teacher_sha256
        or state.source_artifact_sha256 != origin_state.source_artifact_sha256
        or state.healthy_population.fingerprint
        != origin_state.healthy_population.fingerprint
        or state.published_population.fingerprint
        != origin_state.published_population.fingerprint
        or state.healthy_p0.plant_state_sha256
        != origin_state.healthy_p0.plant_state_sha256
        or state.current.layout != origin_state.current.layout
        or state.current.digital_scales != origin_state.current.digital_scales
        or state.current.state_kind != origin_state.current.state_kind
    ):
        raise ValueError("Expected an exact epoch-boundary resume for this Adam task.")
    completed = recovery.get("completed_epochs")
    reports = recovery.get("epoch_reports")
    generator_state = recovery.get("train_generator_state")
    optimizer = recovery.get("optimizer_state_dict")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 < completed <= 10
        or not isinstance(reports, list)
        or len(reports) != completed
        or [item.get("epoch") for item in reports if isinstance(item, Mapping)]
        != list(range(1, completed + 1))
        or not isinstance(generator_state, torch.Tensor)
        or generator_state.dtype != torch.uint8
        or generator_state.device.type != "cpu"
        or generator_state.ndim != 1
        or not isinstance(optimizer, Mapping)
        or optimizer.get("schema") != "ebl.ibm_om_crossbar_pulse_adam"
        or optimizer.get("schema_version") != 1
    ):
        raise ValueError("Expected a valid completed-epoch cursor in Adam resume.")
    final_report = reports[-1]
    if (
        not isinstance(final_report, Mapping)
        or final_report.get("apparent_sha256")
        != tensor_sha256(state.current.plant_state["apparent"])
        or final_report.get("persistent_sha256")
        != tensor_sha256(state.current.plant_state["persistent"])
    ):
        raise ValueError("Expected the Adam resume cursor to authenticate its plant state.")
    return completed


def _expected_recovery_selection(
    task: TaskBlueprint,
    selection_receipt: Path | None,
) -> dict[str, Any]:
    if task.learning_rate is not None:
        if selection_receipt is not None:
            raise RuntimeError("Tuning Adam must not consume a selection receipt.")
        return {"source": "literal_predeclared_grid"}
    if selection_receipt is None:
        raise RuntimeError("Production Adam requires its strict selection receipt.")
    value, _selection = load_selection_receipt(selection_receipt)
    return {
        "source": "strict_selection_receipt",
        "study_id": value["study_id"],
        "source_plan_sha256": value["source_plan_sha256"],
        "winner": dict(value["winner"]),
        "candidate_count": len(value["candidates"]),
        "winner_recomputed": True,
        "validated_receipt": dict(value),
        "artifact_sha256": sha256_file(selection_receipt),
    }


def _verify_adam_metrics(
    handle: RunHandle,
    *,
    origin: Path,
    learning_rate: float,
    pulse_cap: int | None,
    expected_selection: Mapping[str, Any],
) -> None:
    metrics = _result_metrics(handle)
    task = handle.task
    origin_state = load_device_state(origin)
    final_state = load_device_state(handle.artifact(ADAM_FINAL_KIND))
    recovery = final_state.recovery
    if (
        metrics.get("source_kind") != task.source_kind
        or metrics.get("start_state") != task.start_state
        or metrics.get("assignment_seed") != origin_state.assignment_seed
        or metrics.get("endpoint_seed") != origin_state.endpoint_seed
        or metrics.get("origin_device_state_sha256") != sha256_file(origin)
        or metrics.get("learning_rate") != learning_rate
        or metrics.get("pulse_cap_per_cell") != pulse_cap
        or metrics.get("selection") != expected_selection
        or not isinstance(recovery, Mapping)
        or recovery.get("selection") != expected_selection
    ):
        raise RuntimeError(f"Expected exact Adam settings and origin metrics for {task.label}.")


def _manifest_input_path(handle: RunHandle, role: str) -> Path | None:
    records = handle.manifest.get("inputs")
    if not isinstance(records, list):
        raise RuntimeError(f"Expected manifest inputs for {handle.task.label}.")
    matches = [
        item
        for item in records
        if isinstance(item, Mapping) and item.get("role") == role
    ]
    if not matches:
        return None
    if len(matches) != 1 or not isinstance(matches[0].get("path"), str):
        raise RuntimeError(f"Expected at most one manifest input role {role!r}.")
    path = Path(matches[0]["path"]).expanduser().resolve()
    if (
        not path.is_file()
        or matches[0].get("sha256") != sha256_file(path)
    ):
        raise RuntimeError(f"Expected intact manifest input role {role!r}.")
    return path


def _replay_completed_adam_ancestry(
    handle: RunHandle,
    *,
    origin: Path,
    learning_rate: float,
    pulse_cap: int | None,
    expected_selection: Mapping[str, Any],
) -> None:
    """Replay the optional resume cursor and the terminal selection ancestry."""

    resume = _manifest_input_path(handle, "adam_epoch_resume")
    metrics = _result_metrics(handle)
    expected_resume_sha = None if resume is None else sha256_file(resume)
    if metrics.get("resume_input_sha256") != expected_resume_sha:
        raise RuntimeError(
            f"Expected terminal resume digest to match the manifest for {handle.task.label}."
        )
    if resume is not None:
        _validate_resume(
            resume,
            task=handle.task,
            origin=origin,
            learning_rate=learning_rate,
            pulse_cap=pulse_cap,
            expected_selection=expected_selection,
        )
    _verify_adam_metrics(
        handle,
        origin=origin,
        learning_rate=learning_rate,
        pulse_cap=pulse_cap,
        expected_selection=expected_selection,
    )


def _latest_valid_resume(
    *,
    task: TaskBlueprint,
    arm_root: Path,
    source_commit: str,
    config_sha256: str,
    expected_base_inputs: Mapping[str, str],
    origin: Path,
    learning_rate: float,
    pulse_cap: int | None,
    task_python: Path,
    expected_selection: Mapping[str, Any],
) -> Path | None:
    candidates: list[tuple[int, str, Path]] = []
    for run_dir in _native_run_dirs(arm_root):
        matched = _base_run_matches(
            task=task,
            run_dir=run_dir,
            source_commit=source_commit,
            config_sha256=config_sha256,
            expected_inputs=expected_base_inputs,
            allow_resume_input=True,
            task_python=task_python,
        )
        if matched is None:
            continue
        for path in sorted((run_dir / "checkpoints").glob("epoch_*_resume.pt")):
            try:
                completed = _validate_resume(
                    path,
                    task=task,
                    origin=origin,
                    learning_rate=learning_rate,
                    pulse_cap=pulse_cap,
                    expected_selection=expected_selection,
                )
            except (OSError, RuntimeError, ValueError):
                continue
            candidates.append((completed, sha256_file(path), path.resolve()))
    if not candidates:
        return None
    maximum = max(item[0] for item in candidates)
    latest = [item for item in candidates if item[0] == maximum]
    if len({item[1] for item in latest}) != 1:
        raise RuntimeError(
            f"Divergent epoch-{maximum} Adam resumes exist for {task.label}; refusing ambiguity."
        )
    return latest[-1][2]


class ProgressWriter:
    """Continuously refresh launcher heartbeat, status, and progress receipt."""

    def __init__(self, attempt_dir: Path, *, source_commit: str, expected_tasks: int) -> None:
        self.attempt_dir = attempt_dir
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state: dict[str, Any] = {
            "source_commit": source_commit,
            "launcher_pid": os.getpid(),
            "hostname": platform.node(),
            "status": "starting",
            "active_task": None,
            "active_pid": None,
            "completed_or_reused_tasks": 0,
            "expected_tasks": expected_tasks,
            "last_result": None,
        }
        self._thread = threading.Thread(target=self._loop, name="campaign-heartbeat", daemon=True)

    def start(self) -> None:
        self._write()
        self._thread.start()

    def update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)
        self._write()

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _write(self) -> None:
        snapshot = self._snapshot()
        timestamp = _utc_now()
        atomic_write_json(
            self.attempt_dir / "heartbeat.json",
            {
                "schema": "ebl.ibm_om_crossbar_staged_akib.heartbeat",
                "schema_version": 1,
                "updated_at": timestamp,
                "cadence_seconds": HEARTBEAT_SECONDS,
                **snapshot,
            },
        )
        atomic_write_json(
            self.attempt_dir / "status.json",
            {
                "schema": "ebl.ibm_om_crossbar_staged_akib.status",
                "schema_version": 1,
                "updated_at": timestamp,
                **snapshot,
            },
        )
        atomic_write_json(
            self.attempt_dir / "progress_receipt.json",
            {
                "schema": "ebl.ibm_om_crossbar_staged_akib.progress_receipt",
                "schema_version": 1,
                "updated_at": timestamp,
                "heartbeat_maximum_seconds": MAXIMUM_HEARTBEAT_SECONDS,
                **snapshot,
            },
        )

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            self._write()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=MAXIMUM_HEARTBEAT_SECONDS)
        self._write()


def _run_process(
    command: Sequence[str],
    *,
    label: str,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    progress: ProgressWriter,
    stop_requested: threading.Event,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        progress.update(active_task=label, active_pid=process.pid, status="running")
        while process.poll() is None:
            if stop_requested.is_set():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=HEARTBEAT_SECONDS)
            except subprocess.TimeoutExpired:
                progress.update(active_task=label, active_pid=process.pid)
        returncode = process.wait()
    progress.update(active_pid=None)
    return returncode


def _probe_json(command: Sequence[str], *, environment: Mapping[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Prerequisite probe did not return JSON: {command!r}; stderr={completed.stderr!r}."
        ) from error
    if completed.returncode != 0 or not isinstance(value, dict):
        raise RuntimeError(f"Prerequisite probe failed: {command!r}; value={value!r}.")
    return value


def _probe_gpu_occupancy(*, environment: Mapping[str, str]) -> dict[str, Any]:
    """Fail closed when Akib already has a CUDA compute process.

    The formal campaign owns the single Akib GPU for a long sequential run.
    A successful CUDA allocation is therefore not enough: without this check a
    second campaign (or an unrelated user process) could silently share the
    device and invalidate runtime evidence.
    """

    queries = {
        "gpus": (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        "compute_processes": (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
    }
    rows: dict[str, list[str]] = {}
    for label, command in queries.items():
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=dict(environment),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise RuntimeError("Expected a working nvidia-smi Akib occupancy probe.") from error
        if completed.returncode != 0:
            raise RuntimeError(
                f"Akib nvidia-smi {label} probe failed: stderr={completed.stderr!r}."
            )
        rows[label] = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows["gpus"]:
        raise RuntimeError("Expected nvidia-smi to report at least one Akib GPU.")
    if rows["compute_processes"]:
        raise RuntimeError(
            "Refusing to launch the formal campaign on an occupied Akib GPU; "
            f"compute processes={rows['compute_processes']!r}."
        )
    return {
        "probe": "nvidia-smi",
        "gpu_rows": rows["gpus"],
        "compute_process_rows": rows["compute_processes"],
        "exclusive_at_probe": True,
    }


def _sampler_probe_request(corruption_policy: str) -> dict[str, Any]:
    if corruption_policy not in {"counterfactual_repaired", "published"}:
        raise ValueError(f"Unsupported sampler probe policy: {corruption_policy!r}.")
    return {
        "preset": OM_PRESET,
        "assignment_seed": 2_099_999,
        "corruption_policy": corruption_policy,
        "preset_default_corrupt_devices_prob": 0.0,
        "published_corrupt_devices_prob": 0.1348,
        "corrupt_devices_range": 0.01,
        "binding_keys": ["probe"],
        "binding_shapes": [[2, 2]],
        "required_aihwkit_version": EXPECTED_AIHWKIT_VERSION,
    }


def _probe_prerequisites(
    *,
    task_python: Path,
    aihwkit_python: Path,
    mnist_root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    occupancy = _probe_gpu_occupancy(environment=environment)
    cuda_code = (
        "import json,platform,sys,torch; "
        "x=torch.zeros(1,device='cuda'); "
        "print(json.dumps({'executable':str(__import__('pathlib').Path(sys.executable).resolve()),"
        "'hostname':platform.node(),'torch_version':torch.__version__,"
        "'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),"
        "'current_device':torch.cuda.current_device(),'device_name':torch.cuda.get_device_name(),"
        "'tensor_device':str(x.device)}))"
    )
    cuda = _probe_json((str(task_python), "-c", cuda_code), environment=environment)
    if (
        cuda.get("executable") != str(task_python)
        or cuda.get("hostname") != EXPECTED_HOSTNAME
        or cuda.get("cuda_available") is not True
        or not isinstance(cuda.get("device_count"), int)
        or cuda["device_count"] < 1
        or not str(cuda.get("tensor_device", "")).startswith("cuda")
    ):
        raise RuntimeError(f"Exact task-Python CUDA probe failed: {cuda!r}.")

    version_code = (
        "import aihwkit,json,pathlib,sys; "
        "from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice; "
        "d=ReRamArrayOMPresetDevice(); "
        "print(json.dumps({'executable':str(pathlib.Path(sys.executable).resolve()),"
        "'aihwkit_version':str(aihwkit.__version__),'device_class':type(d).__name__}))"
    )
    sampler_version = _probe_json(
        (str(aihwkit_python), "-c", version_code), environment=environment
    )
    if sampler_version != {
        "executable": str(aihwkit_python),
        "aihwkit_version": EXPECTED_AIHWKIT_VERSION,
        "device_class": "ReRamArrayOMPresetDevice",
    }:
        raise RuntimeError(f"Exact AIHWKit sampler probe failed: {sampler_version!r}.")

    sampling: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="ebl-om-sampler-probe-") as temporary:
        root = Path(temporary)
        for policy in ("counterfactual_repaired", "published"):
            request = _sampler_probe_request(policy)
            output = root / f"{policy}.npz"
            receipt = root / f"{policy}.json"
            command = [
                str(aihwkit_python),
                "-m",
                "experiments.reram_program_verify.hwa_population_sampler",
                "--request-json",
                json.dumps(request, allow_nan=False, sort_keys=True),
                "--output",
                str(output),
                "--receipt",
                str(receipt),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=dict(environment),
                check=False,
                capture_output=True,
                text=True,
            )
            record = _read_json(receipt)
            population = load_om_array_population(output) if output.is_file() else None
            if (
                completed.returncode != 0
                or record is None
                or record.get("python_executable") != str(aihwkit_python)
                or record.get("aihwkit_version") != EXPECTED_AIHWKIT_VERSION
                or record.get("request") != request
                or record.get("population_sha256") != sha256_file(output)
                or population is None
                or population.corruption_policy != policy
                or population.size != 4
            ):
                raise RuntimeError(
                    f"Pinned AIHWKit {policy} sampling probe failed; "
                    f"receipt={record!r}, stderr={completed.stderr!r}."
                )
            sampling[policy] = {
                "population_fingerprint": population.fingerprint,
                "population_sha256": sha256_file(output),
                "receipt_sha256": sha256_file(receipt),
            }

    mnist_code = (
        "import json,os; from torchvision.datasets import MNIST; "
        "r=os.environ['EBL_MNIST_ROOT']; "
        "a=MNIST(r,train=True,download=False); b=MNIST(r,train=False,download=False); "
        "print(json.dumps({'root':r,'train_examples':len(a),'test_examples':len(b)}))"
    )
    mnist = _probe_json((str(task_python), "-c", mnist_code), environment=environment)
    if mnist != {
        "root": str(mnist_root),
        "train_examples": 60000,
        "test_examples": 10000,
    }:
        raise RuntimeError(f"Exact offline MNIST probe failed: {mnist!r}.")
    return {
        "gpu_occupancy": occupancy,
        "cuda": cuda,
        "aihwkit": sampler_version,
        "sampling": sampling,
        "mnist": mnist,
    }


def _prepare_study(
    *,
    study_id: str,
    task_python: Path,
    results_root: Path,
    environment: Mapping[str, str],
    attempt_dir: Path,
    progress: ProgressWriter,
    stop_requested: threading.Event,
) -> Path:
    plan = PLAN_PATHS[study_id]
    command = [
        str(task_python),
        "-m",
        "ebl",
        "study",
        "prepare",
        "--plan",
        str(plan.resolve()),
        "--results-root",
        str(results_root.resolve()),
    ]
    returncode = _run_process(
        command,
        label=f"study.prepare.{study_id}",
        cwd=ROOT,
        environment=environment,
        log_path=attempt_dir / "logs" / f"study.prepare.{study_id}.log",
        progress=progress,
        stop_requested=stop_requested,
    )
    if returncode != 0:
        raise RuntimeError(f"Failed to prepare formal study {study_id!r}.")
    study_dir = (results_root / study_id).resolve()
    record = load_study_record(study_dir)
    plan_record = load_study_plan(plan)
    if (
        record["study_id"] != study_id
        or record["source_plan"]["sha256"] != plan_record["source_plan"]["sha256"]
    ):
        raise RuntimeError(f"Prepared study {study_id!r} did not match its tracked plan.")
    return study_dir


def _summarize_study(
    study_dir: Path,
    *,
    task_python: Path,
    environment: Mapping[str, str],
    attempt_dir: Path,
    progress: ProgressWriter,
    stop_requested: threading.Event,
) -> dict[str, Any]:
    command = [
        str(task_python),
        "-m",
        "ebl",
        "study",
        "summarize",
        "--study-dir",
        str(study_dir),
        "--verify-artifacts",
        "--json",
    ]
    label = f"study.summarize.{study_dir.name}"
    returncode = _run_process(
        command,
        label=label,
        cwd=ROOT,
        environment=environment,
        log_path=attempt_dir / "logs" / f"{label}.log",
        progress=progress,
        stop_requested=stop_requested,
    )
    summary = _read_json(study_dir / "analysis" / "summary.json")
    if (
        returncode != 0
        or summary is None
        or summary.get("validation_mode") != "full_artifact_hashes"
        or summary.get("ready_for_review") is not True
    ):
        raise RuntimeError(f"Artifact-verified study summary failed for {study_dir}.")
    return summary


def _run_native_task(
    task: TaskBlueprint,
    *,
    inputs: TaskInputs,
    study_dir: Path,
    task_python: Path,
    sampler: Path,
    source_commit: str,
    environment: Mapping[str, str],
    attempt_dir: Path,
    progress: ProgressWriter,
    stop_requested: threading.Event,
    output_validator: Callable[[RunHandle], None],
    adam_settings: tuple[float, int | None] | None = None,
) -> RunHandle:
    config_sha = _validate_blueprint_config(task)
    uses_sampler = task.stage_kind in {"offchip_hwa", "deploy"}
    base_inputs = inputs.manifest_hashes(
        task=task,
        sampler=sampler if uses_sampler else None,
    )
    arm_root = study_dir / "runs" / task.arm_id
    existing = _completed_run(
        task=task,
        arm_root=arm_root,
        source_commit=source_commit,
        config_sha256=config_sha,
        expected_inputs=base_inputs,
        logical_adam_inputs=task.stage_kind == "on_chip_adam",
        task_python=task_python,
    )
    if existing is not None:
        output_validator(existing)
        if adam_settings is not None and inputs.device_state is not None:
            _replay_completed_adam_ancestry(
                existing,
                origin=inputs.device_state,
                learning_rate=adam_settings[0],
                pulse_cap=adam_settings[1],
                expected_selection=_expected_recovery_selection(
                    task, inputs.selection_receipt
                ),
            )
        progress.update(last_result={"task": task.label, "status": "reused", "run_dir": str(existing.run_dir)})
        return existing

    active_pids = _active_native_pids(task, arm_root)
    if active_pids:
        raise RuntimeError(
            f"Refusing duplicate live native task {task.label!r}; active PIDs={active_pids!r}."
        )

    selected_inputs = inputs
    if task.stage_kind == "on_chip_adam":
        if inputs.device_state is None or adam_settings is None:
            raise RuntimeError("Expected Adam origin and resolved settings before launch.")
        resume = _latest_valid_resume(
            task=task,
            arm_root=arm_root,
            source_commit=source_commit,
            config_sha256=config_sha,
            expected_base_inputs=base_inputs,
            origin=inputs.device_state,
            learning_rate=adam_settings[0],
            pulse_cap=adam_settings[1],
            task_python=task_python,
            expected_selection=_expected_recovery_selection(
                task, inputs.selection_receipt
            ),
        )
        selected_inputs = TaskInputs(
            teacher_weights=inputs.teacher_weights,
            weights=inputs.weights,
            device_state=inputs.device_state,
            selection_receipt=inputs.selection_receipt,
            resume=resume,
        )
    actual_inputs = selected_inputs.manifest_hashes(
        task=task,
        sampler=sampler if uses_sampler else None,
    )
    command = _native_command(
        task,
        task_python=task_python,
        study_dir=study_dir,
        inputs=selected_inputs,
    )
    launch_receipt = {
        "schema": "ebl.ibm_om_crossbar_staged_akib.task_launch",
        "schema_version": 1,
        "task": task.label,
        "source_commit": source_commit,
        "config_sha256": config_sha,
        "input_sha256_by_role": actual_inputs,
        "command": command,
        "output_root": str(arm_root),
        "launched_at": _utc_now(),
    }
    atomic_write_json(attempt_dir / "tasks" / f"{task.label}.launch.json", launch_receipt)
    returncode = _run_process(
        command,
        label=task.label,
        cwd=ROOT,
        environment=environment,
        log_path=attempt_dir / "logs" / f"{task.label}.log",
        progress=progress,
        stop_requested=stop_requested,
    )
    completed = _completed_run(
        task=task,
        arm_root=arm_root,
        source_commit=source_commit,
        config_sha256=config_sha,
        expected_inputs=actual_inputs,
        logical_adam_inputs=False,
        task_python=task_python,
    )
    success = returncode == 0 and completed is not None and not stop_requested.is_set()
    terminal = {
        **launch_receipt,
        "status": "complete" if success else "failed",
        "returncode": returncode,
        "finished_at": _utc_now(),
        "run_dir": str(completed.run_dir) if completed is not None else None,
    }
    atomic_write_json(attempt_dir / "tasks" / f"{task.label}.result.json", terminal)
    if not success or completed is None:
        raise RuntimeError(
            f"Native task failed: {task.label}; see {attempt_dir / 'logs' / (task.label + '.log')}."
        )
    output_validator(completed)
    if adam_settings is not None and inputs.device_state is not None:
        _replay_completed_adam_ancestry(
            completed,
            origin=inputs.device_state,
            learning_rate=adam_settings[0],
            pulse_cap=adam_settings[1],
            expected_selection=_expected_recovery_selection(
                task, inputs.selection_receipt
            ),
        )
    progress.update(last_result={"task": task.label, "status": "complete", "run_dir": str(completed.run_dir)})
    return completed


def _final_validation_cross_entropy(handle: RunHandle) -> float:
    metrics = _result_metrics(handle)
    final = metrics.get("final_evaluation")
    if not isinstance(final, Mapping):
        final = metrics.get("final")
    flattened = _flatten_plant_evaluation(
        final,
        label=f"{handle.task.label}.final",
        require_test=False,
    )
    return flattened["validation.apparent.cross_entropy"]


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Expected non-negative integer {label}.")
    return value


def _tuning_diagnostic_row(
    task: TaskBlueprint,
    handle: RunHandle,
    *,
    origin: Path,
) -> dict[str, Any]:
    """Build one strict, human-auditable tuning row before selection."""

    if task.stage_kind != "on_chip_adam" or task.learning_rate is None:
        raise RuntimeError("Expected one literal-grid Adam tuning task.")
    document = _load_staged_config(task.config)
    stage = document.stage
    if not isinstance(stage, OnChipAdamStageSettings):
        raise RuntimeError("Expected on-chip Adam stage settings for tuning diagnostics.")
    metrics = _result_metrics(handle)
    final = metrics.get("final_evaluation")
    initial = metrics.get("initial")
    final_flat = _flatten_plant_evaluation(
        final,
        label=f"{task.label}.final",
        require_test=False,
    )
    initial_flat = _flatten_plant_evaluation(
        initial,
        label=f"{task.label}.initial",
        require_test=False,
    )
    epochs = metrics.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != stage.epochs:
        raise RuntimeError(f"Expected exactly {stage.epochs} tuning epochs for {task.label}.")
    for index, epoch in enumerate(epochs, start=1):
        if (
            not isinstance(epoch, Mapping)
            or epoch.get("epoch") != index
            or epoch.get("examples") != stage.repair_examples
            or epoch.get("batches") != stage.maximum_batches
            or epoch.get("test") is not None
        ):
            raise RuntimeError(
                f"Expected tuning epoch {index} to consume the exact validation-only "
                f"{stage.repair_examples}-example protocol."
            )
        _flatten_plant_evaluation(
            {"validation": epoch.get("validation")},
            label=f"{task.label}.epoch-{index}",
            require_test=False,
        )
    optimizer = metrics.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise RuntimeError(f"Expected terminal pulse/cap telemetry for {task.label}.")
    integer_fields = (
        "optimizer_steps",
        "requested_nonzero_commands",
        "commanded_pulses",
        "applied_pulses",
        "probability_clipped",
        "blocked_at_cap",
        "cells_at_cap",
        "commanded_cells",
        "changed_cells",
        "maximum_pulses_per_cell",
        "enabled_cells",
    )
    counts = {
        name: _nonnegative_int(optimizer.get(name), label=f"optimizer.{name}")
        for name in integer_fields
    }
    by_layer = optimizer.get("applied_pulses_by_layer")
    if (
        optimizer.get("pulse_cap_per_cell") != task.pulse_cap_per_cell
        or counts["optimizer_steps"] != stage.epochs * stage.maximum_batches
        or counts["applied_pulses"] != counts["commanded_pulses"]
        or counts["applied_pulses"] + counts["blocked_at_cap"]
        > counts["requested_nonzero_commands"]
        or counts["probability_clipped"] > counts["requested_nonzero_commands"]
        or not isinstance(by_layer, list)
        or len(by_layer) != 2
        or any(_nonnegative_int(item, label="optimizer.applied_pulses_by_layer") < 0 for item in by_layer)
        or sum(by_layer) != counts["applied_pulses"]
        or counts["enabled_cells"] != 203_264
        or (
            task.pulse_cap_per_cell is None
            and counts["cells_at_cap"] != 0
        )
        or (
            task.pulse_cap_per_cell is not None
            and counts["maximum_pulses_per_cell"] > task.pulse_cap_per_cell
        )
    ):
        raise RuntimeError(f"Expected self-consistent pulse/cap telemetry for {task.label}.")
    final_validation = final.get("validation") if isinstance(final, Mapping) else None
    initial_validation = initial.get("validation") if isinstance(initial, Mapping) else None
    return {
        "learning_rate": task.learning_rate,
        "pulse_cap_per_cell": task.pulse_cap_per_cell,
        "start_state": task.start_state,
        "source_kind": task.source_kind,
        "origin_device_state_sha256": sha256_file(origin),
        "result_sha256": sha256_file(handle.run_dir / "result.json"),
        "run_dir": str(handle.run_dir),
        "epoch_count": len(epochs),
        "examples_per_epoch": stage.repair_examples,
        "batches_per_epoch": stage.maximum_batches,
        "total_training_examples": stage.repair_examples * stage.epochs,
        "initial_validation": dict(initial_validation),
        "final_validation": dict(final_validation),
        "initial_flat": initial_flat,
        "final_flat": final_flat,
        "optimizer": dict(optimizer),
    }


def _write_immutable_analysis(path: Path, value: Mapping[str, Any], *, label: str) -> Path:
    if path.exists():
        existing = _read_json(path)
        if existing != dict(value):
            raise RuntimeError(f"Refusing to replace a different immutable {label}.")
        return path
    atomic_write_json(path, value)
    return path


def _write_tuning_selection(
    *,
    study_dir: Path,
    handles: Mapping[str, RunHandle],
    origins: Mapping[str, Path],
) -> Path:
    rows = []
    diagnostics = []
    for task in development_blueprints():
        if task.stage_kind != "on_chip_adam":
            continue
        handle = handles[task.label]
        origin = origins[task.start_state or ""]
        diagnostic = _tuning_diagnostic_row(task, handle, origin=origin)
        diagnostics.append(diagnostic)
        rows.append(
            {
                "learning_rate": task.learning_rate,
                "pulse_cap_per_cell": task.pulse_cap_per_cell,
                "start_state": task.start_state,
                "final_validation_cross_entropy": diagnostic["final_flat"][
                    "validation.apparent.cross_entropy"
                ],
                "result_sha256": diagnostic["result_sha256"],
                "origin_device_state_sha256": diagnostic[
                    "origin_device_state_sha256"
                ],
            }
        )
    study = load_study_record(study_dir)
    value = select_adam_hyperparameters(
        rows,
        study_id=DEVELOPMENT_PLAN_ID,
        source_plan_sha256=study["source_plan"]["sha256"],
    )
    validate_selection_receipt(value)
    diagnostic_value = {
        "schema": "ebl.ibm_om_crossbar_adam_tuning_diagnostics",
        "schema_version": 1,
        "study_id": DEVELOPMENT_PLAN_ID,
        "source_plan_sha256": study["source_plan"]["sha256"],
        "selection_score": value["score"],
        "row_count": len(diagnostics),
        "candidate_count": len(value["candidates"]),
        "required_start_states": list(value["required_start_states"]),
        "rows": diagnostics,
        "candidates": value["candidates"],
        "winner": value["winner"],
    }
    if len(diagnostics) != 24:
        raise RuntimeError("Expected all 24 tuning diagnostics before selection.")
    _write_immutable_analysis(
        study_dir / "analysis" / "adam_tuning_diagnostics.json",
        diagnostic_value,
        label="Adam tuning diagnostics",
    )
    path = study_dir / "analysis" / "adam_selection_receipt.json"
    if path.exists():
        existing = _read_json(path)
        if existing != value:
            raise RuntimeError("Refusing to replace a different immutable Adam selection receipt.")
        validate_selection_receipt(existing)
        return path
    return write_selection_receipt(path, value)


_SUMMARY_OUTCOMES = (
    "direct-deploy",
    "direct-corrupt",
    "hwa-deploy",
    "hwa-corrupt",
    "hwa-adam-healthy",
    "hwa-adam-corrupt",
    "scratch-adam-healthy",
    "scratch-adam-corrupt",
)

_PRODUCTION_ARMS = _SUMMARY_OUTCOMES + (
    "scratch-deploy-preparation",
    "scratch-corrupt-preparation",
)


def _production_index(
    handles: Iterable[RunHandle],
) -> dict[tuple[str, int, int], RunHandle]:
    values = tuple(handles)
    counts = Counter(handle.task.arm_id for handle in values)
    expected_counts = Counter({arm: 16 for arm in _PRODUCTION_ARMS})
    if len(values) != 160 or counts != expected_counts:
        raise RuntimeError(
            "Expected exact 160-run production coverage before analysis; "
            f"observed counts={dict(counts)!r}."
        )
    expected_identities = {
        (assignment, endpoint)
        for assignment, endpoints in PRODUCTION_ENDPOINTS.items()
        for endpoint in endpoints
    }
    result: dict[tuple[str, int, int], RunHandle] = {}
    for handle in values:
        task = handle.task
        if (
            task.study_id != PRODUCTION_PLAN_ID
            or task.assignment_seed is None
            or task.endpoint_seed is None
            or (task.assignment_seed, task.endpoint_seed) not in expected_identities
        ):
            raise RuntimeError(f"Unexpected production task identity: {task.label!r}.")
        key = (task.arm_id, task.assignment_seed, task.endpoint_seed)
        if key in result:
            raise RuntimeError(f"Duplicate production task identity: {key!r}.")
        result[key] = handle
    for arm in _PRODUCTION_ARMS:
        observed = {(key[1], key[2]) for key in result if key[0] == arm}
        if observed != expected_identities:
            raise RuntimeError(f"Expected exact 4x4 seed coverage for production arm {arm!r}.")
    return result


def _production_states_and_population_receipt(
    index: Mapping[tuple[str, int, int], RunHandle],
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    expected_state = {
        "direct-deploy": ("healthy_p0", "teacher"),
        "direct-corrupt": ("faulted_p0", "teacher"),
        "hwa-deploy": ("healthy_p0", "hwa_master"),
        "hwa-corrupt": ("faulted_p0", "hwa_master"),
        "hwa-adam-healthy": ("adam_final", "hwa_master"),
        "hwa-adam-corrupt": ("adam_final", "hwa_master"),
        "scratch-deploy-preparation": ("healthy_p0", "scratch"),
        "scratch-corrupt-preparation": ("faulted_p0", "scratch"),
        "scratch-adam-healthy": ("adam_final", "scratch"),
        "scratch-adam-corrupt": ("adam_final", "scratch"),
    }
    # Retain only compact authenticated metadata.  Keeping 160 self-contained
    # state bundles live at once would needlessly retain every device tensor.
    states: dict[tuple[str, int, int], dict[str, Any]] = {}
    healthy_by_assignment: dict[int, set[str]] = {}
    published_by_assignment: dict[int, set[str]] = {}
    for key, handle in index.items():
        arm, assignment, endpoint = key
        kind = handle.task.expected_artifact_kind
        if kind not in {DEPLOYED_KIND, CORRUPTED_KIND, ADAM_FINAL_KIND}:
            raise RuntimeError(f"Expected a production device-state artifact for {arm!r}.")
        artifact_path = handle.artifact(kind)
        state = load_device_state(artifact_path)
        role, source = expected_state[arm]
        if (
            state.assignment_seed != assignment
            or state.endpoint_seed != endpoint
            or state.role != role
            or state.source_kind != source
        ):
            raise RuntimeError(f"Production state identity mismatch for {handle.task.label}.")
        states[key] = {
            "parent_device_state_sha256": state.parent_device_state_sha256,
            "artifact_sha256": sha256_file(artifact_path),
        }
        healthy_by_assignment.setdefault(assignment, set()).add(
            state.healthy_population.fingerprint
        )
        published_by_assignment.setdefault(assignment, set()).add(
            state.published_population.fingerprint
        )
    receipt: dict[str, Any] = {}
    for assignment in PRODUCTION_ENDPOINTS:
        healthy = healthy_by_assignment.get(assignment, set())
        published = published_by_assignment.get(assignment, set())
        if len(healthy) != 1 or len(published) != 1:
            raise RuntimeError(
                "Expected direct, HWA, scratch, healthy, faulted, and recovered "
                f"branches to reuse one matched assignment population; assignment={assignment}, "
                f"healthy={sorted(healthy)!r}, published={sorted(published)!r}."
            )
        receipt[str(assignment)] = {
            "healthy_population_fingerprint": next(iter(healthy)),
            "published_population_fingerprint": next(iter(published)),
            "endpoint_seeds": list(PRODUCTION_ENDPOINTS[assignment]),
            "state_artifact_count": sum(key[1] == assignment for key in states),
        }

    # Replay every same-array parent edge, not merely the seed labels.
    parent_pairs = (
        ("direct-corrupt", "direct-deploy"),
        ("hwa-corrupt", "hwa-deploy"),
        ("hwa-adam-healthy", "hwa-deploy"),
        ("hwa-adam-corrupt", "hwa-corrupt"),
        ("scratch-corrupt-preparation", "scratch-deploy-preparation"),
        ("scratch-adam-healthy", "scratch-deploy-preparation"),
        ("scratch-adam-corrupt", "scratch-corrupt-preparation"),
    )
    for assignment, endpoints in PRODUCTION_ENDPOINTS.items():
        for endpoint in endpoints:
            for child_arm, parent_arm in parent_pairs:
                child = states[(child_arm, assignment, endpoint)]
                parent = states[(parent_arm, assignment, endpoint)]
                if child["parent_device_state_sha256"] != parent["artifact_sha256"]:
                    raise RuntimeError(
                        f"Expected exact {parent_arm!r}->{child_arm!r} state ancestry for "
                        f"assignment={assignment}, endpoint={endpoint}."
                    )
    return states, receipt


def _production_metric_rows(handles: Iterable[RunHandle]) -> dict[str, list[dict[str, Any]]]:
    result = {name: [] for name in _SUMMARY_OUTCOMES}
    for handle in handles:
        task = handle.task
        if task.arm_id not in result:
            continue
        metrics = _result_metrics(handle)
        final = metrics.get("final_evaluation")
        row: dict[str, Any] = {
            "assignment_seed": task.assignment_seed,
            "endpoint_seed": task.endpoint_seed,
            "run_dir": str(handle.run_dir),
            "result_sha256": sha256_file(handle.run_dir / "result.json"),
        }
        row.update(
            _flatten_plant_evaluation(
                final,
                label=f"{task.label}.final",
                require_test=True,
            )
        )
        result[task.arm_id].append(row)
    return result


def _expand_source_for_plant_contrast(source: Mapping[str, float]) -> dict[str, float]:
    return {
        f"{split}.{state}.{metric}": source[f"{split}.{metric}"]
        for split in _EXPECTED_SPLIT_EXAMPLES
        for state in ("apparent", "persistent")
        for metric in _HEADLINE_METRICS
    }


def _paired_contrast(
    *,
    name: str,
    semantic_role: str,
    pairs: Iterable[tuple[RunHandle, RunHandle, Mapping[str, float], Mapping[str, float]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for from_handle, to_handle, before, after in pairs:
        from_task, to_task = from_handle.task, to_handle.task
        if (
            from_task.assignment_seed != to_task.assignment_seed
            or from_task.endpoint_seed != to_task.endpoint_seed
            or set(before) != set(after)
        ):
            raise RuntimeError(f"Expected exact paired identities and metrics for contrast {name!r}.")
        deltas = {metric: float(after[metric] - before[metric]) for metric in before}
        if any(not math.isfinite(value) for value in deltas.values()):
            raise RuntimeError(f"Expected finite paired deltas for contrast {name!r}.")
        rows.append(
            {
                "assignment_seed": from_task.assignment_seed,
                "endpoint_seed": from_task.endpoint_seed,
                "from_arm": from_task.arm_id,
                "to_arm": to_task.arm_id,
                "from_run_dir": str(from_handle.run_dir),
                "to_run_dir": str(to_handle.run_dir),
                "from_result_sha256": sha256_file(from_handle.run_dir / "result.json"),
                "to_result_sha256": sha256_file(to_handle.run_dir / "result.json"),
                "from": dict(before),
                "to": dict(after),
                "delta_to_minus_from": deltas,
            }
        )
    if len(rows) != 16:
        raise RuntimeError(f"Expected 16 exact paired rows for contrast {name!r}.")
    metric_names = tuple(sorted(rows[0]["delta_to_minus_from"]))
    summaries = {}
    for metric in metric_names:
        flat_rows = [
            {
                "assignment_seed": row["assignment_seed"],
                "endpoint_seed": row["endpoint_seed"],
                metric: row["delta_to_minus_from"][metric],
            }
            for row in rows
        ]
        summaries[metric] = grouped_assignment_summary(flat_rows, metric)
    return {
        "name": name,
        "semantic_role": semantic_role,
        "delta_convention": "to_minus_from",
        "higher_is_better_metrics": ["student_accuracy", "teacher_agreement"],
        "lower_is_better_metrics": ["cross_entropy", "kl_teacher_student"],
        "rows": rows,
        "summaries": summaries,
    }


def _production_contrasts(
    index: Mapping[tuple[str, int, int], RunHandle],
) -> dict[str, Any]:
    identities = [
        (assignment, endpoint)
        for assignment, endpoints in PRODUCTION_ENDPOINTS.items()
        for endpoint in endpoints
    ]

    def plant(handle: RunHandle, field: str) -> dict[str, float]:
        return _flatten_plant_evaluation(
            _result_metrics(handle).get(field),
            label=f"{handle.task.label}.{field}",
            require_test=True,
        )

    def final(handle: RunHandle) -> dict[str, float]:
        return plant(handle, "final_evaluation")

    contrasts: dict[str, Any] = {}
    for arm, name in (
        ("direct-deploy", "direct-source-to-deployment"),
        ("hwa-deploy", "hwa-source-to-deployment"),
    ):
        pairs = []
        for assignment, endpoint in identities:
            handle = index[(arm, assignment, endpoint)]
            source = _flatten_effective_evaluation(
                _result_metrics(handle).get("source_evaluation"),
                label=f"{handle.task.label}.source_evaluation",
                require_test=True,
            )
            pairs.append((handle, handle, _expand_source_for_plant_contrast(source), final(handle)))
        contrasts[name] = _paired_contrast(
            name=name,
            semantic_role="source_to_deployment_loss",
            pairs=pairs,
        )

    for arm, name in (
        ("direct-corrupt", "direct-fault-damage"),
        ("hwa-corrupt", "hwa-fault-damage"),
    ):
        pairs = []
        for assignment, endpoint in identities:
            handle = index[(arm, assignment, endpoint)]
            pairs.append((handle, handle, plant(handle, "pre_fault"), final(handle)))
        contrasts[name] = _paired_contrast(
            name=name,
            semantic_role="fault_damage",
            pairs=pairs,
        )

    for arm, name, role in (
        ("hwa-adam-healthy", "hwa-healthy-recovery-gain", "same_array_recovery_gain"),
        ("hwa-adam-corrupt", "hwa-corrupt-recovery-gain", "same_array_recovery_gain"),
        ("scratch-adam-healthy", "scratch-healthy-training-gain", "from_scratch_training_gain"),
        ("scratch-adam-corrupt", "scratch-corrupt-training-gain", "from_scratch_training_gain"),
    ):
        pairs = []
        for assignment, endpoint in identities:
            handle = index[(arm, assignment, endpoint)]
            pairs.append((handle, handle, plant(handle, "initial"), final(handle)))
        contrasts[name] = _paired_contrast(name=name, semantic_role=role, pairs=pairs)

    for from_arm, to_arm, name in (
        ("direct-deploy", "hwa-deploy", "hwa-vs-direct-deployment"),
        ("direct-corrupt", "hwa-corrupt", "hwa-vs-direct-corrupted"),
    ):
        pairs = []
        for assignment, endpoint in identities:
            before_handle = index[(from_arm, assignment, endpoint)]
            after_handle = index[(to_arm, assignment, endpoint)]
            pairs.append(
                (before_handle, after_handle, final(before_handle), final(after_handle))
            )
        contrasts[name] = _paired_contrast(
            name=name,
            semantic_role="hwa_minus_direct_matched_control",
            pairs=pairs,
        )
    return contrasts


def _write_production_summary(study_dir: Path, handles: Iterable[RunHandle]) -> Path:
    handles = tuple(handles)
    index = _production_index(handles)
    _states, population_receipt = _production_states_and_population_receipt(index)
    rows = _production_metric_rows(handles)
    metric_names = (
        f"{split}.{state}.{metric}"
        for split in ("validation", "test")
        for state in ("apparent", "persistent")
        for metric in _HEADLINE_METRICS
    )
    metrics = tuple(metric_names)
    outcomes: dict[str, Any] = {}
    for outcome in _SUMMARY_OUTCOMES:
        if len(rows[outcome]) != 16:
            raise RuntimeError(f"Expected 16 production rows for {outcome!r}.")
        outcomes[outcome] = {
            "rows": rows[outcome],
            "summaries": {
                metric: grouped_assignment_summary(rows[outcome], metric)
                for metric in metrics
            },
        }
    value = {
        "schema": "ebl.ibm_om_crossbar_staged_production_summary",
        "schema_version": 2,
        "study_id": PRODUCTION_PLAN_ID,
        "primary_unit": "assignment_mean_after_averaging_four_programming_writes",
        "primary_assignment_count": 4,
        "secondary_unit": "pooled_programming_realization",
        "secondary_realization_count": 16,
        "network_forward_state": "apparent_q",
        "persistent_state_role": "secondary_diagnostic",
        "paired_delta_convention": "to_minus_from",
        "source_contrast_baseline": (
            "one logical effective-q source evaluation is paired separately with "
            "the deployed apparent primary and persistent secondary states"
        ),
        "matched_population_fingerprints_by_assignment": population_receipt,
        "outcomes": outcomes,
        "paired_contrasts": _production_contrasts(index),
    }
    path = study_dir / "analysis" / "production_assignment_and_pooled_summary.json"
    return _write_immutable_analysis(path, value, label="production summary")


@dataclass
class CampaignState:
    teacher_weights: Path | None = None
    hwa_master: Path | None = None
    tuning_origins: dict[str, Path] = field(default_factory=dict)
    production_origins: dict[tuple[str, str, int, int], Path] = field(default_factory=dict)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--task-python", type=Path, default=TASK_PYTHON)
    parser.add_argument("--aihwkit-python", type=Path, default=AIHWKIT_PYTHON)
    parser.add_argument("--mnist-root", type=Path, default=MNIST_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_akib_hostname()
    source_commit = _require_clean_source_commit()
    task_python = _require_executable(args.task_python, label="Akib CUDA task Python")
    sampler = _require_executable(args.aihwkit_python, label="AIHWKit 1.1 sampler Python")
    mnist_root = args.mnist_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    all_blueprints = (*teacher_blueprints(), *development_blueprints(), *production_blueprints())
    _validate_declared_coverage(all_blueprints)

    environment = dict(os.environ)
    environment.update(
        {
            "EBL_AIHWKIT_PYTHON": str(sampler),
            "EBL_MNIST_ROOT": str(mnist_root),
            "EBL_DEFER_CURRENT_SIMULATIONS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    attempt_dir = results_root / ".launch" / "ibm-om-crossbar-staged-v2" / _attempt_id()
    (attempt_dir / "logs").mkdir(parents=True, exist_ok=False)
    (attempt_dir / "tasks").mkdir()
    progress = ProgressWriter(
        attempt_dir,
        source_commit=source_commit,
        expected_tasks=len(all_blueprints),
    )
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    atomic_write_json(
        attempt_dir / "contract.json",
        {
            "schema": "ebl.ibm_om_crossbar_staged_akib.launch_contract",
            "schema_version": 1,
            "source_commit": source_commit,
            "source_clean": True,
            "target_hostname": EXPECTED_HOSTNAME,
            "observed_hostname": platform.node(),
            "task_python": str(task_python),
            "aihwkit_python": str(sampler),
            "mnist_root": str(mnist_root),
            "public_numerical_entrypoints": [
                f"{task_python} -m ebl train",
                f"{task_python} -m ebl validate",
            ],
            "plans": {
                study_id: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for study_id, path in PLAN_PATHS.items()
            },
            "coverage": {
                "teacher_native_tasks": len(teacher_blueprints()),
                "development_native_tasks": len(development_blueprints()),
                "production_native_tasks": len(production_blueprints()),
                "total_native_tasks": len(all_blueprints),
                "production_assignments": 4,
                "programming_writes_per_assignment": 4,
            },
            "heartbeat": {
                "cadence_seconds": HEARTBEAT_SECONDS,
                "maximum_seconds": MAXIMUM_HEARTBEAT_SECONDS,
                "paths": [
                    str(attempt_dir / "heartbeat.json"),
                    str(attempt_dir / "status.json"),
                    str(attempt_dir / "progress_receipt.json"),
                ],
            },
            "reuse_policy": (
                "same clean source commit, declared source-config hash, exact input "
                "hashes, exact task Python, intact registered artifacts, resolved CUDA, "
                "and replayed staged artifact ancestry"
            ),
            "adam_resume_policy": (
                "latest uniquely authenticated epoch boundary from the same origin, "
                "config, hyperparameters, selection receipt, and clean source commit"
            ),
            "created_at": _utc_now(),
        },
    )
    progress.start()
    started = time.monotonic()
    completed_count = 0
    failure: BaseException | None = None
    study_summaries: dict[str, Any] = {}
    selection_path: Path | None = None
    tuning_diagnostics_path: Path | None = None
    production_summary_path: Path | None = None
    try:
        prerequisites = _probe_prerequisites(
            task_python=task_python,
            aihwkit_python=sampler,
            mnist_root=mnist_root,
            environment=environment,
        )
        atomic_write_json(attempt_dir / "prerequisites.json", prerequisites)
        state = CampaignState()

        teacher_dir = _prepare_study(
            study_id=TEACHER_PLAN_ID,
            task_python=task_python,
            results_root=results_root,
            environment=environment,
            attempt_dir=attempt_dir,
            progress=progress,
            stop_requested=stop_requested,
        )
        teacher_train, teacher_test = teacher_blueprints()
        teacher_handle = _run_native_task(
            teacher_train,
            inputs=TaskInputs(),
            study_dir=teacher_dir,
            task_python=task_python,
            sampler=sampler,
            source_commit=source_commit,
            environment=environment,
            attempt_dir=attempt_dir,
            progress=progress,
            stop_requested=stop_requested,
            output_validator=lambda handle: _verify_teacher(handle),
        )
        completed_count += 1
        progress.update(completed_or_reused_tasks=completed_count)
        state.teacher_weights = teacher_handle.artifact("selected_named_weights")
        _run_native_task(
            teacher_test,
            inputs=TaskInputs(weights=state.teacher_weights),
            study_dir=teacher_dir,
            task_python=task_python,
            sampler=sampler,
            source_commit=source_commit,
            environment=environment,
            attempt_dir=attempt_dir,
            progress=progress,
            stop_requested=stop_requested,
            output_validator=lambda handle: _verify_teacher(handle, require_test=True),
        )
        completed_count += 1
        progress.update(completed_or_reused_tasks=completed_count)
        study_summaries[TEACHER_PLAN_ID] = _summarize_study(
            teacher_dir,
            task_python=task_python,
            environment=environment,
            attempt_dir=attempt_dir,
            progress=progress,
            stop_requested=stop_requested,
        )

        development_dir = _prepare_study(
            study_id=DEVELOPMENT_PLAN_ID,
            task_python=task_python,
            results_root=results_root,
            environment=environment,
            attempt_dir=attempt_dir,
            progress=progress,
            stop_requested=stop_requested,
        )
        dev_handles: dict[str, RunHandle] = {}
        for task in development_blueprints():
            if stop_requested.is_set():
                raise InterruptedError("Stop requested before the next development task.")
            if task.stage_kind == "offchip_hwa":
                inputs = TaskInputs(teacher_weights=state.teacher_weights)
                validator = lambda handle: _verify_hwa(
                    handle, teacher_sha256=sha256_file(state.teacher_weights)
                )
                adam_settings = None
            elif task.label == "development.hwa.deploy":
                inputs = TaskInputs(teacher_weights=state.teacher_weights, weights=state.hwa_master)
                validator = lambda handle: _verify_device_output(
                    handle,
                    teacher_sha256=sha256_file(state.teacher_weights),
                    expected_role="healthy_p0",
                    expected_source_kind="hwa_master",
                    hwa_master=state.hwa_master,
                )
                adam_settings = None
            elif task.label == "development.hwa.corrupt":
                origin = state.tuning_origins["hwa_healthy_p0"]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin)
                validator = lambda handle, origin=origin: _verify_device_output(
                    handle,
                    teacher_sha256=sha256_file(state.teacher_weights),
                    expected_role="faulted_p0",
                    expected_source_kind="hwa_master",
                    origin=origin,
                )
                adam_settings = None
            elif task.label == "development.scratch.deploy":
                inputs = TaskInputs(teacher_weights=state.teacher_weights)
                validator = lambda handle: _verify_device_output(
                    handle,
                    teacher_sha256=sha256_file(state.teacher_weights),
                    expected_role="healthy_p0",
                    expected_source_kind="scratch",
                )
                adam_settings = None
            elif task.label == "development.scratch.corrupt":
                origin = state.tuning_origins["scratch_healthy_p0"]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin)
                validator = lambda handle, origin=origin: _verify_device_output(
                    handle,
                    teacher_sha256=sha256_file(state.teacher_weights),
                    expected_role="faulted_p0",
                    expected_source_kind="scratch",
                    origin=origin,
                )
                adam_settings = None
            else:
                origin = state.tuning_origins[task.start_state or ""]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin)
                validator = lambda handle, origin=origin, task=task: _verify_device_output(
                    handle,
                    teacher_sha256=sha256_file(state.teacher_weights),
                    expected_role="adam_final",
                    expected_source_kind=task.source_kind or "",
                    origin=origin,
                )
                adam_settings = (task.learning_rate or 0.0, task.pulse_cap_per_cell)
            handle = _run_native_task(
                task,
                inputs=inputs,
                study_dir=development_dir,
                task_python=task_python,
                sampler=sampler,
                source_commit=source_commit,
                environment=environment,
                attempt_dir=attempt_dir,
                progress=progress,
                stop_requested=stop_requested,
                output_validator=validator,
                adam_settings=adam_settings,
            )
            dev_handles[task.label] = handle
            completed_count += 1
            if task.stage_kind == "offchip_hwa":
                state.hwa_master = handle.artifact(HWA_KIND)
            elif task.label == "development.hwa.deploy":
                state.tuning_origins["hwa_healthy_p0"] = handle.artifact(DEPLOYED_KIND)
            elif task.label == "development.hwa.corrupt":
                state.tuning_origins["hwa_published_fault"] = handle.artifact(CORRUPTED_KIND)
            elif task.label == "development.scratch.deploy":
                state.tuning_origins["scratch_healthy_p0"] = handle.artifact(DEPLOYED_KIND)
            elif task.label == "development.scratch.corrupt":
                state.tuning_origins["scratch_published_fault"] = handle.artifact(CORRUPTED_KIND)
            progress.update(completed_or_reused_tasks=completed_count)

        selection_path = _write_tuning_selection(
            study_dir=development_dir,
            handles=dev_handles,
            origins=state.tuning_origins,
        )
        tuning_diagnostics_path = (
            development_dir / "analysis" / "adam_tuning_diagnostics.json"
        )
        if not tuning_diagnostics_path.is_file():
            raise RuntimeError("Expected immutable all-24 Adam tuning diagnostics.")
        receipt_value = _read_json(selection_path)
        if receipt_value is None:
            raise RuntimeError("Expected readable strict Adam selection receipt.")
        winner = validate_selection_receipt(receipt_value)
        study_summaries[DEVELOPMENT_PLAN_ID] = _summarize_study(
            development_dir,
            task_python=task_python,
            environment=environment,
            attempt_dir=attempt_dir,
            progress=progress,
            stop_requested=stop_requested,
        )

        production_dir = _prepare_study(
            study_id=PRODUCTION_PLAN_ID,
            task_python=task_python,
            results_root=results_root,
            environment=environment,
            attempt_dir=attempt_dir,
            progress=progress,
            stop_requested=stop_requested,
        )
        production_handles: list[RunHandle] = []
        for task in production_blueprints():
            if stop_requested.is_set():
                raise InterruptedError("Stop requested before the next production task.")
            identity = (task.assignment_seed or 0, task.endpoint_seed or 0)
            if task.arm_id == "direct-deploy":
                inputs = TaskInputs(teacher_weights=state.teacher_weights)
                role, source, origin, hwa = "healthy_p0", "teacher", None, None
            elif task.arm_id == "direct-corrupt":
                origin = state.production_origins[("teacher", "healthy", *identity)]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin)
                role, source, hwa = "faulted_p0", "teacher", None
            elif task.arm_id == "hwa-deploy":
                inputs = TaskInputs(teacher_weights=state.teacher_weights, weights=state.hwa_master)
                role, source, origin, hwa = "healthy_p0", "hwa_master", None, state.hwa_master
            elif task.arm_id == "hwa-corrupt":
                origin = state.production_origins[("hwa_master", "healthy", *identity)]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin)
                role, source, hwa = "faulted_p0", "hwa_master", None
            elif task.arm_id == "hwa-adam-healthy":
                origin = state.production_origins[("hwa_master", "healthy", *identity)]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin, selection_receipt=selection_path)
                role, source, hwa = "adam_final", "hwa_master", None
            elif task.arm_id == "hwa-adam-corrupt":
                origin = state.production_origins[("hwa_master", "faulted", *identity)]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin, selection_receipt=selection_path)
                role, source, hwa = "adam_final", "hwa_master", None
            elif task.arm_id == "scratch-deploy-preparation":
                inputs = TaskInputs(teacher_weights=state.teacher_weights)
                role, source, origin, hwa = "healthy_p0", "scratch", None, None
            elif task.arm_id == "scratch-corrupt-preparation":
                origin = state.production_origins[("scratch", "healthy", *identity)]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin)
                role, source, hwa = "faulted_p0", "scratch", None
            elif task.arm_id == "scratch-adam-healthy":
                origin = state.production_origins[("scratch", "healthy", *identity)]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin, selection_receipt=selection_path)
                role, source, hwa = "adam_final", "scratch", None
            elif task.arm_id == "scratch-adam-corrupt":
                origin = state.production_origins[("scratch", "faulted", *identity)]
                inputs = TaskInputs(teacher_weights=state.teacher_weights, device_state=origin, selection_receipt=selection_path)
                role, source, hwa = "adam_final", "scratch", None
            else:  # pragma: no cover - fixed blueprint invariant
                raise RuntimeError(f"Unknown production arm {task.arm_id!r}.")
            validator = lambda handle, role=role, source=source, origin=origin, hwa=hwa: _verify_device_output(
                handle,
                teacher_sha256=sha256_file(state.teacher_weights),
                expected_role=role,
                expected_source_kind=source,
                origin=origin,
                hwa_master=hwa,
            )
            adam_settings = (
                (winner.learning_rate, winner.pulse_cap_per_cell)
                if task.stage_kind == "on_chip_adam"
                else None
            )
            handle = _run_native_task(
                task,
                inputs=inputs,
                study_dir=production_dir,
                task_python=task_python,
                sampler=sampler,
                source_commit=source_commit,
                environment=environment,
                attempt_dir=attempt_dir,
                progress=progress,
                stop_requested=stop_requested,
                output_validator=validator,
                adam_settings=adam_settings,
            )
            production_handles.append(handle)
            completed_count += 1
            if task.stage_kind == "deploy":
                state.production_origins[(source, "healthy", *identity)] = handle.artifact(DEPLOYED_KIND)
            elif task.stage_kind == "apply_corruption":
                state.production_origins[(source, "faulted", *identity)] = handle.artifact(CORRUPTED_KIND)
            progress.update(completed_or_reused_tasks=completed_count)

        production_summary_path = _write_production_summary(production_dir, production_handles)
        study_summaries[PRODUCTION_PLAN_ID] = _summarize_study(
            production_dir,
            task_python=task_python,
            environment=environment,
            attempt_dir=attempt_dir,
            progress=progress,
            stop_requested=stop_requested,
        )
    except BaseException as error:
        failure = error

    result = {
        "schema": "ebl.ibm_om_crossbar_staged_akib.launcher_result",
        "schema_version": 1,
        "status": "complete" if failure is None else "failed",
        "source_commit": source_commit,
        "hostname": platform.node(),
        "task_python": str(task_python),
        "aihwkit_python": str(sampler),
        "mnist_root": str(mnist_root),
        "completed_or_reused_tasks": completed_count,
        "expected_tasks": len(all_blueprints),
        "selection_receipt": str(selection_path) if selection_path is not None else None,
        "selection_receipt_sha256": (
            sha256_file(selection_path)
            if selection_path is not None and selection_path.is_file()
            else None
        ),
        "tuning_diagnostics": (
            str(tuning_diagnostics_path)
            if tuning_diagnostics_path is not None
            else None
        ),
        "tuning_diagnostics_sha256": (
            sha256_file(tuning_diagnostics_path)
            if tuning_diagnostics_path is not None
            and tuning_diagnostics_path.is_file()
            else None
        ),
        "production_summary": str(production_summary_path) if production_summary_path is not None else None,
        "production_summary_sha256": (
            sha256_file(production_summary_path)
            if production_summary_path is not None and production_summary_path.is_file()
            else None
        ),
        "study_summaries": study_summaries,
        "elapsed_seconds": time.monotonic() - started,
        "finished_at": _utc_now(),
        "error": None if failure is None else {"type": type(failure).__name__, "message": str(failure)},
    }
    atomic_write_json(attempt_dir / "launcher_result.json", result)
    progress.update(
        status=result["status"],
        active_task=None,
        active_pid=None,
        completed_or_reused_tasks=completed_count,
        last_result=result.get("error"),
    )
    progress.close()
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised on Akib
    raise SystemExit(main())


__all__ = [
    "EXPECTED_HOSTNAME",
    "HEARTBEAT_SECONDS",
    "MAXIMUM_HEARTBEAT_SECONDS",
    "TaskBlueprint",
    "TaskInputs",
    "development_blueprints",
    "main",
    "production_blueprints",
    "teacher_blueprints",
]
