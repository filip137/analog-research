"""Fail-closed dual-host orchestration for the exact-P0 HWA canary.

This module is an operational composition layer only.  Every numerical task
enters through ``python -m ebl train``.  Cross-commit input reuse is limited to
one tracked reference and is authenticated without weakening the repository's
generic same-commit completed-run reuse contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.mnist_analog_relu.generate_hwa_recovery_canary import (
    ASSIGNMENT_SEED,
    CHECKPOINT_POLICY,
    ENDPOINT_SEED,
    PLAN_PATH,
    REFERENCE_PATH,
    SELECTION_METRIC,
    SELECTION_OBJECTIVE_METRICS,
    SHARD_MANIFEST_PATH,
    SOURCE_COMMIT,
    STUDY_ID,
)
from experiments.mnist_analog_relu.staged_artifacts import load_device_state
from experiments.mnist_analog_relu.staged_config import parse_staged_crossbar_config
from experiments.study_workflow import register_federated_runs
from training.ibm_om_standard_crossbar import tensor_sha256


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_AKIB_HOSTNAME = "integnano-akib"
INPUT_FILENAMES = {
    "teacher_weights": "teacher_weights.pt",
    "hwa_master": "hwa_master.pt",
    "hwa_healthy_p0": "hwa_healthy_p0.pt",
    "hwa_faulted_p0": "hwa_faulted_p0.pt",
}
REPORT_KIND_BY_STAGE = {
    "on_chip_adam_diagnostic": "crossbar_adam_diagnostic_report",
    "fresh_apparent_diagnostic": "crossbar_fresh_apparent_diagnostic_report",
}
STATE_ARTIFACT_KIND = "crossbar_adam_final_state_bundle"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Expected readable JSON object: {path}.") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}.")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"Expected {label} to be a lowercase SHA-256 digest.")
    return value


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
        raise RuntimeError("Expected a clean resolvable Git execution commit.") from error
    if not commit or status:
        raise RuntimeError(
            "Canary launch requires one clean execution commit containing the runtime, "
            "generated configs, study, reference, shard manifest, and tests."
        )
    return commit


def _input_by_role(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in manifest.get("inputs", [])
        if isinstance(item, Mapping) and item.get("role") == role
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one original {role!r} input record.")
    return matches[0]


def _artifact_by_kind(result: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in result.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("kind") == kind
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one registered {kind!r} artifact.")
    return matches[0]


def _verify_original_control(
    *,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    expected_study_id: str,
    expected_arm_id: str,
    expected_study_sha256: str,
    expected_plan_sha256: str,
    expected_source_config_sha256: str,
) -> None:
    source = manifest.get("source")
    study = manifest.get("study")
    if (
        not isinstance(source, Mapping)
        or source.get("available") is not True
        or source.get("commit") != SOURCE_COMMIT
        or source.get("dirty") is not False
        or source.get("dirty_hash") is not None
        or not isinstance(study, Mapping)
        or study.get("study_id") != expected_study_id
        or study.get("arm_id") != expected_arm_id
        or study.get("study_sha256") != expected_study_sha256
        or study.get("source_plan_sha256") != expected_plan_sha256
        or study.get("source_config_sha256") != expected_source_config_sha256
        or result.get("status") != "complete"
    ):
        raise RuntimeError(
            f"Original {expected_arm_id!r} manifest/result fails the pinned provenance contract."
        )


def verify_import_bundle(
    input_dir: Path,
    *,
    reference_path: Path = REFERENCE_PATH,
) -> dict[str, Any]:
    """Authenticate the sole allowed cross-source-commit artifact import."""

    input_dir = input_dir.expanduser().resolve()
    reference_path = reference_path.expanduser().resolve()
    reference = _read_json(reference_path)
    source = reference.get("source")
    artifacts = reference.get("artifacts")
    provenance_hashes = reference.get("provenance_files")
    if (
        reference.get("schema") != "ebl.ibm_om_crossbar_hwa_p0_import_reference"
        or reference.get("schema_version") != 1
        or not isinstance(source, Mapping)
        or source.get("commit") != SOURCE_COMMIT
        or source.get("dirty") is not False
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != set(INPUT_FILENAMES)
        or not isinstance(provenance_hashes, Mapping)
    ):
        raise RuntimeError("Invalid exact-P0 import reference.")

    paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for role, filename in INPUT_FILENAMES.items():
        record = artifacts.get(role)
        path = input_dir / filename
        if not isinstance(record, Mapping) or record.get("filename") != filename:
            raise RuntimeError(f"Invalid import reference artifact record for {role}.")
        expected_sha = _require_sha(record.get("sha256"), label=f"{role} hash")
        expected_size = record.get("size_bytes")
        if (
            not path.is_file()
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            raise RuntimeError(f"Imported artifact bytes do not match {role}: {path}.")
        paths[role] = path
        artifact_hashes[role] = expected_sha

    provenance_dir = input_dir / "provenance"
    provenance: dict[str, dict[str, Any]] = {}
    for filename, expected_value in provenance_hashes.items():
        expected_sha = _require_sha(expected_value, label=f"{filename} hash")
        path = provenance_dir / filename
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"Original provenance bytes do not match: {path}.")
        provenance[filename] = _read_json(path)

    teacher_manifest = provenance["teacher_manifest.json"]
    teacher_result = provenance["teacher_result.json"]
    hwa_manifest = provenance["hwa_manifest.json"]
    hwa_result = provenance["hwa_result.json"]
    healthy_manifest = provenance["healthy_manifest.json"]
    healthy_result = provenance["healthy_result.json"]
    faulted_manifest = provenance["faulted_manifest.json"]
    faulted_result = provenance["faulted_result.json"]

    _verify_original_control(
        manifest=teacher_manifest,
        result=teacher_result,
        expected_study_id=str(source["teacher_study_id"]),
        expected_arm_id="teacher",
        expected_study_sha256=str(source["teacher_study_sha256"]),
        expected_plan_sha256=str(source["teacher_source_plan_sha256"]),
        expected_source_config_sha256=str(source["teacher_source_config_sha256"]),
    )
    for manifest, result, arm_id, config_sha in (
        (
            hwa_manifest,
            hwa_result,
            "offchip-hwa",
            source["hwa_source_config_sha256"],
        ),
        (
            healthy_manifest,
            healthy_result,
            "tune-hwa-deploy",
            source["healthy_source_config_sha256"],
        ),
        (
            faulted_manifest,
            faulted_result,
            "tune-hwa-corrupt",
            source["faulted_source_config_sha256"],
        ),
    ):
        _verify_original_control(
            manifest=manifest,
            result=result,
            expected_study_id=str(source["study_id"]),
            expected_arm_id=arm_id,
            expected_study_sha256=str(source["study_sha256"]),
            expected_plan_sha256=str(source["source_plan_sha256"]),
            expected_source_config_sha256=str(config_sha),
        )

    teacher_sha = artifact_hashes["teacher_weights"]
    hwa_sha = artifact_hashes["hwa_master"]
    healthy_sha = artifact_hashes["hwa_healthy_p0"]
    faulted_sha = artifact_hashes["hwa_faulted_p0"]
    if (
        _artifact_by_kind(teacher_result, "selected_named_weights").get("sha256")
        != teacher_sha
        or _input_by_role(hwa_manifest, "teacher_weights").get("sha256")
        != teacher_sha
        or _artifact_by_kind(hwa_result, "crossbar_hwa_master").get("sha256")
        != hwa_sha
        or _input_by_role(healthy_manifest, "teacher_weights").get("sha256")
        != teacher_sha
        or _input_by_role(healthy_manifest, "hwa_master").get("sha256") != hwa_sha
        or _artifact_by_kind(
            healthy_result, "crossbar_deployed_state_bundle"
        ).get("sha256")
        != healthy_sha
        or _input_by_role(faulted_manifest, "teacher_weights").get("sha256")
        != teacher_sha
        or _input_by_role(faulted_manifest, "origin_device_state").get("sha256")
        != healthy_sha
        or _artifact_by_kind(
            faulted_result, "crossbar_corrupted_state_bundle"
        ).get("sha256")
        != faulted_sha
    ):
        raise RuntimeError("Original manifests/results do not form the pinned artifact chain.")
    teacher_metrics = teacher_result.get("metrics")
    hwa_metrics = hwa_result.get("metrics")
    healthy_metrics = healthy_result.get("metrics")
    faulted_metrics = faulted_result.get("metrics")
    if (
        not isinstance(teacher_metrics, Mapping)
        or teacher_metrics.get("acceptance_gate", {}).get("passed") is not True
        or float(teacher_metrics.get("selected", {}).get("accuracy", 0.0)) <= 0.97
        or not isinstance(hwa_metrics, Mapping)
        or hwa_metrics.get("teacher_sha256") != teacher_sha
        or hwa_metrics.get("runtime", {}).get("resolved_device") != "cuda:0"
        or not isinstance(healthy_metrics, Mapping)
        or healthy_metrics.get("teacher_sha256") != teacher_sha
        or healthy_metrics.get("source_artifact_sha256") != hwa_sha
        or healthy_metrics.get("device_state_sha256") != healthy_sha
        or healthy_metrics.get("runtime", {}).get("resolved_device") != "cuda:0"
        or not isinstance(faulted_metrics, Mapping)
        or faulted_metrics.get("teacher_sha256") != teacher_sha
        or faulted_metrics.get("parent_device_state_sha256") != healthy_sha
        or faulted_metrics.get("device_state_sha256") != faulted_sha
        or faulted_metrics.get("runtime", {}).get("resolved_device") != "cuda:0"
    ):
        raise RuntimeError(
            "Original results fail the accepted-teacher, CUDA, or artifact-lineage gate."
        )

    healthy = load_device_state(paths["hwa_healthy_p0"])
    faulted = load_device_state(paths["hwa_faulted_p0"])
    healthy_record = artifacts["hwa_healthy_p0"]
    faulted_record = artifacts["hwa_faulted_p0"]
    if (
        healthy.role != "healthy_p0"
        or healthy.source_kind != "hwa_master"
        or healthy.assignment_seed != ASSIGNMENT_SEED
        or healthy.endpoint_seed != ENDPOINT_SEED
        or healthy.teacher_sha256 != teacher_sha
        or healthy.source_artifact_sha256 != hwa_sha
        or healthy.parent_device_state_sha256 is not None
        or faulted.role != "faulted_p0"
        or faulted.source_kind != "hwa_master"
        or faulted.assignment_seed != ASSIGNMENT_SEED
        or faulted.endpoint_seed != ENDPOINT_SEED
        or faulted.teacher_sha256 != teacher_sha
        or faulted.source_artifact_sha256 != hwa_sha
        or faulted.parent_device_state_sha256 != healthy_sha
        or faulted.healthy_p0.plant_state_sha256
        != healthy.healthy_p0.plant_state_sha256
        or tensor_sha256(healthy.current.plant_state["apparent"])
        != healthy_record["apparent_q_sha256"]
        or tensor_sha256(healthy.current.plant_state["persistent"])
        != healthy_record["persistent_q_sha256"]
        or healthy.current.plant_state_sha256
        != healthy_record["plant_state_sha256"]
        or tensor_sha256(faulted.current.plant_state["apparent"])
        != faulted_record["apparent_q_sha256"]
        or tensor_sha256(faulted.current.plant_state["persistent"])
        != faulted_record["persistent_q_sha256"]
        or faulted.current.plant_state_sha256
        != faulted_record["plant_state_sha256"]
    ):
        raise RuntimeError("Loaded P0 state ancestry or tensor hashes do not match the reference.")

    return {
        "schema": "ebl.ibm_om_crossbar_hwa_p0_import_verification",
        "schema_version": 1,
        "verified_at": _utc_now(),
        "source_commit": SOURCE_COMMIT,
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "input_dir": str(input_dir),
        "artifact_sha256_by_role": artifact_hashes,
        "provenance_sha256_by_filename": dict(provenance_hashes),
        "assignment_seed": ASSIGNMENT_SEED,
        "endpoint_seed": ENDPOINT_SEED,
    }


@dataclass(frozen=True)
class ArmTask:
    arm_id: str
    config: Path
    config_sha256: str
    stage_kind: str
    start_state: str


def _plan_config_path(plan_path: Path, value: str) -> Path:
    return (plan_path.parent / value).resolve()


def load_shard_tasks(
    shard: str,
    *,
    plan_path: Path = PLAN_PATH,
    shard_manifest_path: Path = SHARD_MANIFEST_PATH,
    reference_path: Path = REFERENCE_PATH,
) -> tuple[ArmTask, ...]:
    plan_path = plan_path.resolve()
    shard_manifest_path = shard_manifest_path.resolve()
    manifest = _read_json(shard_manifest_path)
    plan = _read_json(plan_path)
    if (
        manifest.get("schema")
        != "ebl.ibm_om_crossbar_hwa_recovery_canary_dual_host"
        or manifest.get("schema_version") != 1
        or manifest.get("study_id") != STUDY_ID
        or plan.get("study_id") != STUDY_ID
        or manifest.get("study_plan", {}).get("sha256") != sha256_file(plan_path)
        or manifest.get("source_reference", {}).get("sha256")
        != sha256_file(reference_path)
        or manifest.get("source_reference", {}).get("source_commit")
        != SOURCE_COMMIT
    ):
        raise RuntimeError("Study plan/shard manifest/reference identity mismatch.")
    shards = manifest.get("shards")
    if not isinstance(shards, Mapping) or set(shards) != {"local", "akib"}:
        raise RuntimeError("Expected exactly local and Akib shard declarations.")
    all_ids: list[str] = []
    for record in shards.values():
        if not isinstance(record, Mapping) or not isinstance(record.get("arm_ids"), list):
            raise RuntimeError("Malformed shard arm declaration.")
        all_ids.extend(record["arm_ids"])
    plan_arms = plan.get("arms")
    if (
        not isinstance(plan_arms, list)
        or len(all_ids) != len(set(all_ids))
        or set(all_ids) != {arm.get("arm_id") for arm in plan_arms if isinstance(arm, Mapping)}
    ):
        raise RuntimeError("Shard union must equal the disjoint declared study coverage.")
    shard_record = shards.get(shard)
    if not isinstance(shard_record, Mapping):
        raise RuntimeError(f"Unknown shard {shard!r}.")
    expected_hashes = shard_record.get("config_sha256_by_arm")
    if not isinstance(expected_hashes, Mapping):
        raise RuntimeError("Missing shard config hashes.")
    by_id = {
        str(arm["arm_id"]): arm for arm in plan_arms if isinstance(arm, Mapping)
    }
    tasks: list[ArmTask] = []
    for arm_id in shard_record["arm_ids"]:
        arm = by_id[arm_id]
        configs = arm.get("configs")
        if not isinstance(configs, list) or len(configs) != 1:
            raise RuntimeError(f"Expected one config for atomic arm {arm_id!r}.")
        config = _plan_config_path(plan_path, str(configs[0]))
        digest = sha256_file(config)
        if digest != expected_hashes.get(arm_id):
            raise RuntimeError(f"Config hash mismatch for shard arm {arm_id!r}.")
        payload = _read_json(config)
        parsed = parse_staged_crossbar_config(payload)
        stage = parsed.stage
        if (
            parsed.runtime.device != "cuda"
            or parsed.evaluation.profile != "diagnostic_validation_only"
            or parsed.device.assignment_seed != ASSIGNMENT_SEED
            or parsed.device.endpoint_seed != ENDPOINT_SEED
            or stage.kind not in REPORT_KIND_BY_STAGE
            or getattr(stage, "start_state", None)
            not in {"hwa_healthy_p0", "hwa_published_fault"}
        ):
            raise RuntimeError(f"Config contract mismatch for {arm_id!r}.")
        tasks.append(
            ArmTask(
                arm_id=arm_id,
                config=config,
                config_sha256=digest,
                stage_kind=stage.kind,
                start_state=stage.start_state,
            )
        )
    if len(tasks) != int(shard_record.get("expected_arms", -1)):
        raise RuntimeError("Shard task count mismatch.")
    return tuple(tasks)


def _require_shard_hostname(shard: str, hostname: str | None = None) -> str:
    observed = platform.node() if hostname is None else hostname
    if shard == "akib" and observed != EXPECTED_AKIB_HOSTNAME:
        raise RuntimeError(
            f"Akib shard requires {EXPECTED_AKIB_HOSTNAME!r}; observed {observed!r}."
        )
    if shard == "local" and observed == EXPECTED_AKIB_HOSTNAME:
        raise RuntimeError("Refusing to run the independently assigned local shard on Akib.")
    return observed


def _require_executable(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Expected executable {label}: {resolved}.")
    return resolved


def _cuda_probe(task_python: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    script = (
        "import json,torch; "
        "ok=torch.cuda.is_available(); "
        "print(json.dumps({'cuda_available':ok,'device_count':torch.cuda.device_count(),"
        "'device_name':torch.cuda.get_device_name(0) if ok else None})); "
        "raise SystemExit(0 if ok else 3)"
    )
    completed = subprocess.run(
        (str(task_python), "-c", script),
        check=False,
        capture_output=True,
        text=True,
        env=dict(environment),
    )
    try:
        value = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("CUDA probe did not emit its JSON receipt.") from error
    if completed.returncode != 0 or value.get("cuda_available") is not True:
        raise RuntimeError(f"CUDA probe failed closed: {value!r}; {completed.stderr[-1000:]}")
    return value


def _materialized_plan_hash(study_dir: Path) -> str:
    study = _read_json(study_dir / "study.json")
    for path in (
        ("source_plan", "sha256"),
        ("plan", "sha256"),
    ):
        current: Any = study
        for key in path:
            current = current.get(key) if isinstance(current, Mapping) else None
        if isinstance(current, str):
            return current
    direct = study.get("source_plan_sha256")
    if isinstance(direct, str):
        return direct
    # Current workflow materializes the plan fields plus this top-level hash.
    raise RuntimeError(f"Cannot resolve materialized source-plan hash in {study_dir}.")


def _safe_artifact_path(run_dir: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise RuntimeError("Artifact path must be a string.")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as error:
        raise RuntimeError("Artifact path escapes the native run directory.") from error
    return path


def _verify_registered_artifacts(run_dir: Path, result: Mapping[str, Any]) -> None:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(f"Completed run has no registered artifacts: {run_dir}.")
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise RuntimeError("Malformed registered artifact record.")
        path = _safe_artifact_path(run_dir, record.get("path"))
        expected_sha = _require_sha(record.get("sha256"), label="artifact hash")
        expected_size = record.get("size_bytes")
        if (
            not path.is_file()
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            raise RuntimeError(f"Registered artifact verification failed: {path}.")


def _run_dirs(arm_dir: Path) -> tuple[Path, ...]:
    if not arm_dir.is_dir():
        return ()
    return tuple(sorted(path for path in arm_dir.iterdir() if path.is_dir()))


def _validate_completed_run(
    run_dir: Path,
    *,
    task: ArmTask,
    execution_commit: str,
    teacher_sha: str,
    origin_sha: str,
    verify_artifacts: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(run_dir / "manifest.json")
    result = _read_json(run_dir / "result.json")
    source = manifest.get("source")
    study = manifest.get("study")
    if (
        result.get("status") != "complete"
        or manifest.get("experiment_id") != "mnist_ibm_om_crossbar_relu.v2"
        or not isinstance(source, Mapping)
        or source.get("commit") != execution_commit
        or source.get("dirty") is not False
        or not isinstance(study, Mapping)
        or study.get("study_id") != STUDY_ID
        or study.get("arm_id") != task.arm_id
        or study.get("source_config_sha256") != task.config_sha256
        or _input_by_role(manifest, "teacher_weights").get("sha256") != teacher_sha
        or _input_by_role(manifest, "origin_device_state").get("sha256") != origin_sha
    ):
        raise RuntimeError(f"Completed run provenance mismatch: {run_dir}.")
    metrics = result.get("metrics")
    runtime = metrics.get("runtime") if isinstance(metrics, Mapping) else None
    resolved_device = runtime.get("resolved_device") if isinstance(runtime, Mapping) else None
    if (
        not isinstance(resolved_device, str)
        or not resolved_device.startswith("cuda")
        or metrics.get("teacher_sha256") != teacher_sha
        or metrics.get("origin_device_state_sha256") != origin_sha
        or metrics.get("assignment_seed") != ASSIGNMENT_SEED
        or metrics.get("endpoint_seed") != ENDPOINT_SEED
        or metrics.get("start_state") != task.start_state
    ):
        raise RuntimeError(f"Completed run runtime/input identity mismatch: {run_dir}.")
    declared_stage = parse_staged_crossbar_config(_read_json(task.config)).stage
    report_record = _artifact_by_kind(result, REPORT_KIND_BY_STAGE[task.stage_kind])
    if task.stage_kind == "on_chip_adam_diagnostic":
        state_record = _artifact_by_kind(result, STATE_ARTIFACT_KIND)
        candidates = metrics.get("selection_candidates")
        epochs = metrics.get("epochs")
        objective = metrics.get("objective")
        if (
            objective != declared_stage.objective
            or float(metrics.get("learning_rate", float("nan")))
            != declared_stage.hyperparameters.learning_rate
            or metrics.get("pulse_cap_per_cell")
            != declared_stage.hyperparameters.pulse_cap_per_cell
        ):
            raise RuntimeError(f"Diagnostic Adam config/result mismatch: {run_dir}.")
        origin_role = (
            "hwa_healthy_p0"
            if task.start_state == "hwa_healthy_p0"
            else "hwa_faulted_p0"
        )
        origin_record = _read_json(REFERENCE_PATH)["artifacts"][origin_role]
        expected_objective_metric = SELECTION_OBJECTIVE_METRICS.get(objective)
        if (
            expected_objective_metric is None
            or metrics.get("selection_metric") != SELECTION_METRIC
            or metrics.get("selection_objective_metric")
            != expected_objective_metric
            or not isinstance(candidates, list)
            or any(not isinstance(item, Mapping) for item in candidates)
            or [item.get("epoch") for item in candidates] != [0, 1, 2, 3]
            or not isinstance(epochs, list)
            or any(not isinstance(item, Mapping) for item in epochs)
            or [item.get("epoch") for item in epochs] != [1, 2, 3]
            or metrics.get("selected_epoch") not in {0, 1, 2, 3}
            or metrics.get("non_degradation_passed") is not True
            or not isinstance(metrics.get("initial"), Mapping)
            or not isinstance(metrics.get("selected_validation"), Mapping)
            or not isinstance(metrics.get("training_final_validation"), Mapping)
            or candidates[0].get("validation")
            != metrics.get("initial", {}).get("validation")
            or any(
                candidates[index].get("validation")
                != epochs[index - 1].get("validation")
                for index in (1, 2, 3)
            )
        ):
            raise RuntimeError(f"Incomplete diagnostic Adam evidence: {run_dir}.")
        if (
            state_record.get("sha256") != metrics.get("device_state_sha256")
            or "test" in metrics["initial"]
            or any(
                not isinstance(item, Mapping)
                or item.get("epoch") != index
                or item.get("examples") != 55000
                or item.get("batches") != 3438
                or "test" not in item
                or item.get("test") is not None
                for index, item in enumerate(epochs, start=1)
            )
        ):
            raise RuntimeError(
                f"Diagnostic Adam stream/evaluation/state binding failed: {run_dir}."
            )
        for validation in (
            metrics["initial"]["validation"],
            metrics["selected_validation"],
            metrics["training_final_validation"],
            *(item["validation"] for item in epochs),
        ):
            apparent, persistent = _validation_states(validation)
            if (
                apparent.get("examples") != 5000
                or persistent.get("examples") != 5000
            ):
                raise RuntimeError(
                    f"Diagnostic Adam did not use the full validation split: {run_dir}."
                )
        epoch_three = epochs[-1]
        selected_optimizer = metrics.get("selected_optimizer")
        training_final_optimizer = metrics.get("training_final_optimizer")
        selected_epoch = int(metrics["selected_epoch"])
        selected_epoch_plant_sha = (
            origin_record["plant_state_sha256"]
            if selected_epoch == 0
            else epochs[selected_epoch - 1].get("plant_state_sha256")
        )
        selected_epoch_apparent_sha = (
            origin_record["apparent_q_sha256"]
            if selected_epoch == 0
            else epochs[selected_epoch - 1].get("apparent_sha256")
        )
        selected_epoch_persistent_sha = (
            origin_record["persistent_q_sha256"]
            if selected_epoch == 0
            else epochs[selected_epoch - 1].get("persistent_sha256")
        )
        selected_path = _safe_artifact_path(run_dir, state_record.get("path"))
        selected_state = load_device_state(selected_path)
        if (
            metrics.get("training_final_apparent_sha256")
            != epoch_three.get("apparent_sha256")
            or metrics.get("training_final_persistent_sha256")
            != epoch_three.get("persistent_sha256")
            or metrics.get("training_final_plant_state_sha256")
            != epoch_three.get("plant_state_sha256")
            or metrics.get("selected_plant_state_sha256")
            != selected_epoch_plant_sha
            or training_final_optimizer != epoch_three.get("optimizer_cumulative")
            or metrics.get("optimizer") != selected_optimizer
            or selected_state.role != "adam_final"
            or selected_state.parent_device_state_sha256 != origin_sha
            or selected_state.recovery.get("completed_epochs") != selected_epoch
            or selected_state.current.plant_state_sha256 != selected_epoch_plant_sha
            or tensor_sha256(selected_state.current.plant_state["apparent"])
            != selected_epoch_apparent_sha
            or tensor_sha256(selected_state.current.plant_state["persistent"])
            != selected_epoch_persistent_sha
            or (
                selected_epoch > 0
                and selected_optimizer
                != epochs[selected_epoch - 1].get("optimizer_cumulative")
            )
        ):
            raise RuntimeError(
                f"Diagnostic Adam final/selected state telemetry mismatch: {run_dir}."
            )
        if selected_epoch == 0:
            if not isinstance(selected_optimizer, Mapping) or any(
                selected_optimizer.get(key) not in (0, [0, 0])
                for key in (
                    "optimizer_steps",
                    "requested_nonzero_commands",
                    "commanded_pulses",
                    "applied_pulses",
                    "applied_pulses_by_layer",
                    "probability_clipped",
                    "blocked_at_cap",
                    "cells_at_cap",
                    "commanded_cells",
                    "changed_cells",
                    "maximum_pulses_per_cell",
                )
            ):
                raise RuntimeError(
                    f"Epoch-zero selection retained non-zero optimizer telemetry: {run_dir}."
                )
        objective_key = expected_objective_metric.rsplit(".", 1)[-1]
        try:
            for candidate in candidates:
                validation = candidate["validation"]["apparent_forward"]
                accuracy = float(candidate["accuracy"])
                objective_value = float(candidate["objective_value"])
                candidate_value = candidate["value"]
                if (
                    not math.isfinite(accuracy)
                    or not math.isfinite(objective_value)
                    or not math.isclose(
                        accuracy,
                        float(validation["student_accuracy"]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        objective_value,
                        float(validation[objective_key]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not isinstance(candidate_value, Mapping)
                    or float(candidate_value["student_accuracy"]) != accuracy
                    or float(candidate_value["objective_value"])
                    != objective_value
                ):
                    raise ValueError
            recomputed = min(
                candidates,
                key=lambda item: (
                    -float(item["accuracy"]),
                    float(item["objective_value"]),
                    int(item["epoch"]),
                ),
            )
            selected_value = metrics["selected_value"]
            selection_replays = (
                int(metrics["selected_epoch"]) == int(recomputed["epoch"])
                and metrics["selected_validation"] == recomputed["validation"]
                and math.isclose(
                    float(metrics["selected_accuracy"]),
                    float(recomputed["accuracy"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(metrics["selected_objective_value"]),
                    float(recomputed["objective_value"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and isinstance(selected_value, Mapping)
                and math.isclose(
                    float(selected_value["student_accuracy"]),
                    float(recomputed["accuracy"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(selected_value["objective_value"]),
                    float(recomputed["objective_value"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and float(recomputed["accuracy"]) >= float(candidates[0]["accuracy"])
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            selection_replays = False
        report_path = _safe_artifact_path(run_dir, report_record.get("path"))
        report = _read_json(report_path)
        if (
            not selection_replays
            or report.get("checkpoint_policy") != CHECKPOINT_POLICY
            or report.get("selection_metric") != SELECTION_METRIC
            or report.get("selection_objective_metric")
            != expected_objective_metric
        ):
            raise RuntimeError(
                f"Diagnostic Adam selection does not replay accuracy-first: {run_dir}."
            )
        if float(metrics["learning_rate"]) == 0.0:
            noop_summary = metrics.get("zero_learning_rate_noop_check")
            if (
                selected_epoch != 0
                or not isinstance(noop_summary, Mapping)
                or noop_summary.get("passed") is not True
                or noop_summary.get("all_epoch_checks_passed") is not True
                or noop_summary.get("origin_plant_state_sha256")
                != origin_record["plant_state_sha256"]
                or noop_summary.get("selected_plant_state_sha256")
                != origin_record["plant_state_sha256"]
                or noop_summary.get("training_final_plant_state_sha256")
                != origin_record["plant_state_sha256"]
                or noop_summary.get("selected_optimizer_steps") != 0
                or noop_summary.get("training_final_optimizer_steps") != 0
                or metrics.get("selected_plant_state_sha256")
                != origin_record["plant_state_sha256"]
                or metrics.get("training_final_plant_state_sha256")
                != origin_record["plant_state_sha256"]
            ):
                raise RuntimeError(f"LR=0 full-plant no-op summary failed: {run_dir}.")
            for item in epochs:
                cumulative = item.get("optimizer_cumulative")
                noop_epoch = item.get("zero_learning_rate_noop_check")
                if (
                    item.get("execution")
                    != "zero_learning_rate_no_gradient_no_write_control"
                    or any(
                        item.get(key) != 0
                        for key in (
                            "commanded_pulses",
                            "probability_clipped",
                            "blocked_at_cap",
                        )
                    )
                    or not isinstance(cumulative, Mapping)
                    or any(
                        cumulative.get(key) not in (0, [0, 0])
                        for key in (
                            "optimizer_steps",
                            "requested_nonzero_commands",
                            "commanded_pulses",
                            "applied_pulses",
                            "applied_pulses_by_layer",
                            "probability_clipped",
                            "blocked_at_cap",
                            "cells_at_cap",
                            "commanded_cells",
                            "changed_cells",
                            "maximum_pulses_per_cell",
                        )
                    )
                    or item.get("apparent_sha256")
                    != origin_record["apparent_q_sha256"]
                    or item.get("persistent_sha256")
                    != origin_record["persistent_q_sha256"]
                    or item.get("plant_state_sha256")
                    != origin_record["plant_state_sha256"]
                    or not isinstance(noop_epoch, Mapping)
                    or noop_epoch.get("passed") is not True
                    or noop_epoch.get("origin_plant_state_sha256")
                    != origin_record["plant_state_sha256"]
                    or noop_epoch.get("current_plant_state_sha256")
                    != origin_record["plant_state_sha256"]
                    or noop_epoch.get("optimizer_steps") != 0
                    or noop_epoch.get("requested_nonzero_commands") != 0
                    or noop_epoch.get("commanded_pulses") != 0
                ):
                    raise RuntimeError(
                        f"LR=0 arm mutated state or emitted pulse activity: {run_dir}."
                    )
            if (
                tensor_sha256(selected_state.current.plant_state["apparent"])
                != origin_record["apparent_q_sha256"]
                or tensor_sha256(selected_state.current.plant_state["persistent"])
                != origin_record["persistent_q_sha256"]
                or selected_state.current.plant_state_sha256
                != origin_record["plant_state_sha256"]
            ):
                raise RuntimeError(f"LR=0 selected bundle mutated the exact P0: {run_dir}.")
        elif metrics.get("zero_learning_rate_noop_check") is not None or any(
            item.get("zero_learning_rate_noop_check") is not None for item in epochs
        ):
            raise RuntimeError(
                f"Nonzero-LR arm emitted LR=0 no-op evidence: {run_dir}."
            )
    elif any(
        isinstance(item, Mapping) and item.get("kind") == STATE_ARTIFACT_KIND
        for item in result.get("artifacts", [])
    ):
        raise RuntimeError("Fresh-apparent diagnostic must not emit a mutable state artifact.")
    else:
        draws = metrics.get("draws")
        mutation = metrics.get("mutation_check")
        seed_derivation = metrics.get("seed_derivation")
        initial = metrics.get("initial")
        if (
            metrics.get("intervention")
            != "counterfactual_post_write_apparent_noise_redraw_without_device_write"
            or metrics.get("configured_seed") != declared_stage.seed
            or metrics.get("relative_scale") != declared_stage.relative_scale
            or not isinstance(draws, list)
            or [item.get("draw") for item in draws] != [1, 2, 3, 4]
            or not isinstance(mutation, Mapping)
            or mutation.get("passed") is not True
            or any(mutation.get(key) != 0 for key in (
                "writes_commanded",
                "persistent_updates",
                "held_apparent_updates",
            ))
            or metrics.get("matched_across_start_states") is not True
            or not isinstance(seed_derivation, Mapping)
            or seed_derivation.get("scheme")
            != "derive_seed_without_start_state_for_paired_noise_v1"
            or seed_derivation.get("excluded_pairing_dimension") != "start_state"
            or seed_derivation.get("matched_across_start_states") is not True
            or seed_derivation.get("resolved_seed") != metrics.get("resolved_seed")
            or seed_derivation.get("configured_seed") != metrics.get("configured_seed")
            or not isinstance(initial, Mapping)
            or "test" in initial
            or any(
                not isinstance(item.get("validation"), Mapping)
                or "test" in item["validation"]
                for item in draws
            )
        ):
            raise RuntimeError(f"Incomplete fresh-apparent evidence: {run_dir}.")
        report_path = _safe_artifact_path(run_dir, report_record.get("path"))
        report = _read_json(report_path)
        for key in (
            "configured_seed",
            "resolved_seed",
            "matched_across_start_states",
            "seed_derivation",
            "generator_state_before_sha256",
            "generator_state_after_sha256",
        ):
            if report.get(key) != metrics.get(key):
                raise RuntimeError(
                    f"Fresh-apparent report/result seed mismatch for {run_dir}."
                )
    if verify_artifacts:
        _verify_registered_artifacts(run_dir, result)
    return manifest, result


def _find_completed(
    arm_dir: Path,
    *,
    task: ArmTask,
    execution_commit: str,
    teacher_sha: str,
    origin_sha: str,
    verify_artifacts: bool,
) -> Path | None:
    completed: list[Path] = []
    for run_dir in _run_dirs(arm_dir):
        status_path = run_dir / "status.json"
        if status_path.is_file():
            status = _read_json(status_path).get("status")
            if status in {"running", "created"}:
                raise RuntimeError(f"Refusing duplicate launch beside active run: {run_dir}.")
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            continue
        result = _read_json(result_path)
        if result.get("status") != "complete":
            continue
        _validate_completed_run(
            run_dir,
            task=task,
            execution_commit=execution_commit,
            teacher_sha=teacher_sha,
            origin_sha=origin_sha,
            verify_artifacts=verify_artifacts,
        )
        completed.append(run_dir)
    if len(completed) > 1:
        raise RuntimeError(f"Duplicate completed coverage for {task.arm_id!r}: {completed!r}.")
    return completed[0] if completed else None


def _native_command(
    *,
    task_python: Path,
    task: ArmTask,
    study_dir: Path,
    teacher: Path,
    origin: Path,
) -> list[str]:
    return [
        str(task_python),
        "-m",
        "ebl",
        "train",
        "--config",
        str(task.config),
        "--output-dir",
        str(study_dir / "runs" / task.arm_id),
        "--teacher-weights",
        str(teacher),
        "--device-state",
        str(origin),
    ]


def _execute(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    heartbeat_path: Path,
    heartbeat: Mapping[str, Any],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            tuple(command),
            cwd=ROOT,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            atomic_write_json(
                heartbeat_path,
                {
                    **heartbeat,
                    "pid": process.pid,
                    "observed_at": _utc_now(),
                    "status": "running",
                },
            )
            time.sleep(10.0)
        if process.returncode != 0:
            raise RuntimeError(
                f"Native command exited {process.returncode}; inspect {log_path}."
            )


def run_shard(
    *,
    shard: str,
    input_dir: Path,
    results_root: Path,
    task_python: Path,
    mnist_root: Path,
    aihwkit_python: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    execution_commit = _require_clean_source_commit()
    hostname = _require_shard_hostname(shard)
    task_python = _require_executable(task_python, label="task Python")
    if aihwkit_python is not None:
        aihwkit_python = _require_executable(aihwkit_python, label="AIHWKit sampler Python")
    mnist_root = mnist_root.expanduser().resolve()
    if not mnist_root.is_dir():
        raise RuntimeError(f"Expected existing MNIST root: {mnist_root}.")
    tasks = load_shard_tasks(shard)
    imported = verify_import_bundle(input_dir)
    inputs = {
        role: input_dir.expanduser().resolve() / filename
        for role, filename in INPUT_FILENAMES.items()
    }
    environment = dict(os.environ)
    environment.update(
        {
            "EBL_MNIST_ROOT": str(mnist_root),
            "EBL_DEFER_CURRENT_SIMULATIONS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if aihwkit_python is not None:
        environment["EBL_AIHWKIT_PYTHON"] = str(aihwkit_python)
    cuda = _cuda_probe(task_python, environment)

    results_root = results_root.expanduser().resolve()
    attempt_dir = (
        results_root
        / ".launch"
        / "ibm-om-crossbar-hwa-recovery-canary"
        / f"{shard}-{_attempt_id()}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(attempt_dir / "import_verification.json", imported)
    contract = {
        "schema": "ebl.ibm_om_crossbar_hwa_recovery_canary_launch",
        "schema_version": 1,
        "created_at": _utc_now(),
        "study_id": STUDY_ID,
        "shard": shard,
        "hostname": hostname,
        "execution_source_commit": execution_commit,
        "execution_source_clean": True,
        "imported_source_commit": SOURCE_COMMIT,
        "plan_sha256": sha256_file(PLAN_PATH),
        "shard_manifest_sha256": sha256_file(SHARD_MANIFEST_PATH),
        "source_reference_sha256": sha256_file(REFERENCE_PATH),
        "artifact_sha256_by_role": imported["artifact_sha256_by_role"],
        "task_python": str(task_python),
        "aihwkit_python": str(aihwkit_python) if aihwkit_python is not None else None,
        "mnist_root": str(mnist_root),
        "cuda_probe": cuda,
        "expected_arms": len(tasks),
        "arm_ids": [task.arm_id for task in tasks],
        "dry_run": dry_run,
    }
    atomic_write_json(attempt_dir / "contract.json", contract)
    if dry_run:
        result = {**contract, "status": "dry_run_verified", "finished_at": _utc_now()}
        atomic_write_json(attempt_dir / "launcher_result.json", result)
        return result

    subprocess.run(
        (
            str(task_python),
            "-m",
            "ebl",
            "study",
            "prepare",
            "--plan",
            str(PLAN_PATH),
            "--results-root",
            str(results_root),
        ),
        cwd=ROOT,
        env=environment,
        check=True,
    )
    study_dir = results_root / STUDY_ID
    completed = 0
    reused = 0
    completed_runs: dict[str, Path] = {}
    try:
        for index, task in enumerate(tasks, start=1):
            origin_role = (
                "hwa_healthy_p0"
                if task.start_state == "hwa_healthy_p0"
                else "hwa_faulted_p0"
            )
            origin_sha = imported["artifact_sha256_by_role"][origin_role]
            arm_dir = study_dir / "runs" / task.arm_id
            existing = _find_completed(
                arm_dir,
                task=task,
                execution_commit=execution_commit,
                teacher_sha=imported["artifact_sha256_by_role"]["teacher_weights"],
                origin_sha=origin_sha,
                verify_artifacts=True,
            )
            if existing is not None:
                reused += 1
                completed_run = existing
            else:
                before = set(_run_dirs(arm_dir))
                command = _native_command(
                    task_python=task_python,
                    task=task,
                    study_dir=study_dir,
                    teacher=inputs["teacher_weights"],
                    origin=inputs[origin_role],
                )
                _execute(
                    command,
                    environment=environment,
                    log_path=attempt_dir / "logs" / f"{index:02d}-{task.arm_id}.log",
                    heartbeat_path=attempt_dir / "heartbeat.json",
                    heartbeat={
                        "study_id": STUDY_ID,
                        "shard": shard,
                        "active_arm": task.arm_id,
                        "ordinal": index,
                        "expected_arms": len(tasks),
                        "completed_arms": completed,
                    },
                )
                created = set(_run_dirs(arm_dir)) - before
                if len(created) != 1:
                    raise RuntimeError(
                        f"Expected one new native run for {task.arm_id!r}; found {created!r}."
                    )
                completed_run = created.pop()
                _validate_completed_run(
                    completed_run,
                    task=task,
                    execution_commit=execution_commit,
                    teacher_sha=imported["artifact_sha256_by_role"]["teacher_weights"],
                    origin_sha=origin_sha,
                    verify_artifacts=True,
                )
            completed_runs[task.arm_id] = completed_run
            completed += 1
            atomic_write_json(
                attempt_dir / "progress.json",
                {
                    "status": "running",
                    "study_id": STUDY_ID,
                    "shard": shard,
                    "completed_arms": completed,
                    "reused_arms": reused,
                    "expected_arms": len(tasks),
                    "last_completed_arm": task.arm_id,
                    "updated_at": _utc_now(),
                },
            )
        stream_parity = _adam_stream_sentinel(
            runs=completed_runs,
            tasks={task.arm_id: task for task in tasks},
        )
        fresh_pairing = _fresh_pairing_sentinel(
            runs=completed_runs,
            tasks={task.arm_id: task for task in tasks},
            require_pair=shard == "local",
        )
        atomic_write_json(attempt_dir / "adam_stream_parity.json", stream_parity)
        atomic_write_json(attempt_dir / "fresh_pairing.json", fresh_pairing)
    except BaseException as error:
        result = {
            **contract,
            "status": "failed",
            "completed_arms": completed,
            "reused_arms": reused,
            "error": {"type": type(error).__name__, "message": str(error)},
            "finished_at": _utc_now(),
        }
        atomic_write_json(attempt_dir / "launcher_result.json", result)
        raise

    result = {
        **contract,
        "status": "complete",
        "completed_arms": completed,
        "reused_arms": reused,
        "study_dir": str(study_dir),
        "adam_stream_parity": stream_parity,
        "fresh_pairing": fresh_pairing,
        "finished_at": _utc_now(),
    }
    atomic_write_json(attempt_dir / "launcher_result.json", result)
    atomic_write_json(
        attempt_dir / "heartbeat.json",
        {
            "status": "complete",
            "study_id": STUDY_ID,
            "shard": shard,
            "completed_arms": completed,
            "expected_arms": len(tasks),
            "observed_at": _utc_now(),
        },
    )
    return result


def _task_by_arm() -> dict[str, ArmTask]:
    tasks = (*load_shard_tasks("local"), *load_shard_tasks("akib"))
    return {task.arm_id: task for task in tasks}


def _completed_run_for_collection(
    arm_dir: Path,
    *,
    task: ArmTask,
    verify_artifacts: bool,
) -> tuple[Path, str]:
    completed: list[tuple[Path, str]] = []
    for run_dir in _run_dirs(arm_dir):
        if not (run_dir / "result.json").is_file():
            continue
        result = _read_json(run_dir / "result.json")
        if result.get("status") != "complete":
            continue
        manifest = _read_json(run_dir / "manifest.json")
        source = manifest.get("source")
        if not isinstance(source, Mapping) or source.get("dirty") is not False:
            raise RuntimeError(f"Incoming run has no clean source identity: {run_dir}.")
        commit = str(source.get("commit"))
        origin_role = (
            "hwa_healthy_p0"
            if task.start_state == "hwa_healthy_p0"
            else "hwa_faulted_p0"
        )
        reference = _read_json(REFERENCE_PATH)
        artifact_records = reference["artifacts"]
        _validate_completed_run(
            run_dir,
            task=task,
            execution_commit=commit,
            teacher_sha=artifact_records["teacher_weights"]["sha256"],
            origin_sha=artifact_records[origin_role]["sha256"],
            verify_artifacts=verify_artifacts,
        )
        completed.append((run_dir, commit))
    if len(completed) != 1:
        raise RuntimeError(
            f"Expected exactly one authenticated completion in {arm_dir}; found {len(completed)}."
        )
    return completed[0]


def _initial_validation(run_dir: Path) -> Mapping[str, Any]:
    result = _read_json(run_dir / "result.json")
    metrics = result.get("metrics")
    initial = metrics.get("initial") if isinstance(metrics, Mapping) else None
    validation = initial.get("validation") if isinstance(initial, Mapping) else None
    if not isinstance(validation, Mapping):
        raise RuntimeError(f"Expected epoch-zero validation evidence: {run_dir}.")
    return validation


def _compare_epoch0_validation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str,
    absolute_tolerance: float = 1e-5,
) -> dict[str, Any]:
    exact_keys = (
        "examples",
        "student_correct",
        "teacher_correct",
        "prediction_flips_from_teacher",
        "student_prediction_sha256",
        "teacher_prediction_sha256",
    )
    numeric_keys = (
        "student_accuracy",
        "teacher_accuracy",
        "teacher_agreement",
        "cross_entropy",
        "kl_teacher_student",
        "student_score_rms",
        "teacher_score_rms",
    )
    states: dict[str, Any] = {}
    for state_key in ("apparent_forward", "persistent_diagnostic"):
        left_state = left.get(state_key)
        right_state = right.get(state_key)
        if not isinstance(left_state, Mapping) or not isinstance(right_state, Mapping):
            raise RuntimeError(f"Missing {state_key} in epoch-zero parity sentinel {label}.")
        if any(left_state.get(key) != right_state.get(key) for key in exact_keys):
            raise RuntimeError(f"Exact prediction parity failed for {label}.{state_key}.")
        deltas: dict[str, float] = {}
        for key in numeric_keys:
            left_value = float(left_state[key])
            right_value = float(right_state[key])
            delta = right_value - left_value
            if not math.isfinite(delta) or abs(delta) > absolute_tolerance:
                raise RuntimeError(
                    f"Numeric epoch-zero parity failed for {label}.{state_key}.{key}: "
                    f"delta={delta!r}, tolerance={absolute_tolerance!r}."
                )
            deltas[key] = delta
        states[state_key] = {
            "exact_prediction_fields_match": True,
            "numeric_right_minus_left": deltas,
            "absolute_tolerance": absolute_tolerance,
        }
    return states


def _epoch0_parity_sentinel(
    *,
    canonical_runs: Mapping[str, Path],
    incoming_runs: Mapping[str, Path],
    tasks: Mapping[str, ArmTask],
    canonical_shard: str,
    incoming_shard: str,
) -> dict[str, Any]:
    """Use replicated Adam epoch-zero evaluations as the cross-host sentinel."""

    result: dict[str, Any] = {}
    for start_state in ("hwa_healthy_p0", "hwa_published_fault"):
        canonical_ids = sorted(
            arm_id
            for arm_id in canonical_runs
            if tasks[arm_id].stage_kind == "on_chip_adam_diagnostic"
            and tasks[arm_id].start_state == start_state
        )
        incoming_ids = sorted(
            arm_id
            for arm_id in incoming_runs
            if tasks[arm_id].stage_kind == "on_chip_adam_diagnostic"
            and tasks[arm_id].start_state == start_state
        )
        if not canonical_ids or not incoming_ids:
            raise RuntimeError(
                f"Both hosts must contain an Adam epoch-zero sentinel for {start_state}."
            )
        canonical_anchor = canonical_ids[0]
        incoming_anchor = incoming_ids[0]
        canonical_validation = _initial_validation(canonical_runs[canonical_anchor])
        incoming_validation = _initial_validation(incoming_runs[incoming_anchor])
        # Every config on one host must first agree with that host's anchor.
        for arm_id in canonical_ids[1:]:
            _compare_epoch0_validation(
                canonical_validation,
                _initial_validation(canonical_runs[arm_id]),
                label=f"{canonical_shard}.{start_state}.{arm_id}",
                absolute_tolerance=0.0,
            )
        for arm_id in incoming_ids[1:]:
            _compare_epoch0_validation(
                incoming_validation,
                _initial_validation(incoming_runs[arm_id]),
                label=f"{incoming_shard}.{start_state}.{arm_id}",
                absolute_tolerance=0.0,
            )
        cross_host = _compare_epoch0_validation(
            canonical_validation,
            incoming_validation,
            label=f"cross_host.{start_state}",
        )
        result[start_state] = {
            "canonical_shard": canonical_shard,
            "canonical_anchor_arm": canonical_anchor,
            "incoming_shard": incoming_shard,
            "incoming_anchor_arm": incoming_anchor,
            "canonical_internal_arms": canonical_ids,
            "incoming_internal_arms": incoming_ids,
            "cross_host": cross_host,
        }
    return {
        "schema": "ebl.ibm_om_crossbar_hwa_recovery_epoch0_host_parity",
        "schema_version": 1,
        "passed": True,
        "states": result,
    }


def _adam_stream_sentinel(
    *, runs: Mapping[str, Path], tasks: Mapping[str, ArmTask]
) -> dict[str, Any]:
    """Require every Adam arm to consume the same ordered stream per epoch."""

    adam_ids = sorted(
        arm_id
        for arm_id in runs
        if tasks[arm_id].stage_kind == "on_chip_adam_diagnostic"
    )
    if not adam_ids:
        raise RuntimeError("Expected at least one Adam arm for stream parity.")
    by_epoch: dict[str, Any] = {}
    for epoch_index in (1, 2, 3):
        reference_arm = adam_ids[0]
        reference: tuple[str, str] | None = None
        for arm_id in adam_ids:
            result = _read_json(runs[arm_id] / "result.json")
            metrics = result.get("metrics")
            epochs = metrics.get("epochs") if isinstance(metrics, Mapping) else None
            if (
                not isinstance(epochs, list)
                or [item.get("epoch") for item in epochs] != [1, 2, 3]
            ):
                raise RuntimeError(f"Missing Adam epoch stream evidence for {arm_id!r}.")
            epoch = epochs[epoch_index - 1]
            observed = (
                _require_sha(
                    epoch.get("ordered_model_inputs_sha256"),
                    label=f"{arm_id} epoch {epoch_index} input-stream hash",
                ),
                _require_sha(
                    epoch.get("ordered_labels_sha256"),
                    label=f"{arm_id} epoch {epoch_index} label-stream hash",
                ),
            )
            if reference is None:
                reference = observed
            elif observed != reference:
                raise RuntimeError(
                    f"Ordered training stream parity failed at epoch {epoch_index}: "
                    f"{reference_arm!r} != {arm_id!r}."
                )
        assert reference is not None
        by_epoch[str(epoch_index)] = {
            "reference_arm": reference_arm,
            "ordered_model_inputs_sha256": reference[0],
            "ordered_labels_sha256": reference[1],
        }
    return {
        "schema": "ebl.ibm_om_crossbar_hwa_recovery_stream_parity",
        "schema_version": 1,
        "passed": True,
        "arm_count": len(adam_ids),
        "arm_ids": adam_ids,
        "epochs": by_epoch,
    }


def _fresh_pairing_sentinel(
    *,
    runs: Mapping[str, Path],
    tasks: Mapping[str, ArmTask],
    require_pair: bool,
) -> dict[str, Any]:
    """Authenticate the matched no-write redraw stream across both P0 states."""

    fresh_ids = sorted(
        arm_id
        for arm_id in runs
        if tasks[arm_id].stage_kind == "fresh_apparent_diagnostic"
    )
    if not fresh_ids and not require_pair:
        return {
            "schema": "ebl.ibm_om_crossbar_hwa_recovery_fresh_pairing",
            "schema_version": 1,
            "applicable": False,
            "passed": True,
            "arm_count": 0,
        }
    if len(fresh_ids) != 2 or {
        tasks[arm_id].start_state for arm_id in fresh_ids
    } != {"hwa_healthy_p0", "hwa_published_fault"}:
        raise RuntimeError("Expected the matched healthy/faulted fresh-apparent pair.")
    documents = [
        _read_json(runs[arm_id] / "result.json")["metrics"] for arm_id in fresh_ids
    ]
    exact_fields = (
        "configured_seed",
        "resolved_seed",
        "seed_derivation",
        "generator_state_before_sha256",
        "generator_state_after_sha256",
    )
    if any(
        documents[0].get(field) != documents[1].get(field)
        for field in exact_fields
    ):
        raise RuntimeError("Fresh-apparent paired seed/RNG receipts differ by P0 state.")
    noise_hashes: list[list[str]] = []
    for arm_id, document in zip(fresh_ids, documents, strict=True):
        draws = document.get("draws")
        if (
            not isinstance(draws, list)
            or [item.get("draw") for item in draws] != [1, 2, 3, 4]
        ):
            raise RuntimeError(f"Malformed fresh-apparent draws for {arm_id!r}.")
        noise_hashes.append(
            [
                _require_sha(
                    item.get("noise_q_sha256"),
                    label=f"{arm_id} draw noise hash",
                )
                for item in draws
            ]
        )
    if noise_hashes[0] != noise_hashes[1]:
        raise RuntimeError("Fresh-apparent paired draws did not use identical noise tensors.")
    return {
        "schema": "ebl.ibm_om_crossbar_hwa_recovery_fresh_pairing",
        "schema_version": 1,
        "applicable": True,
        "passed": True,
        "arm_ids": fresh_ids,
        "configured_seed": documents[0]["configured_seed"],
        "resolved_seed": documents[0]["resolved_seed"],
        "seed_derivation": documents[0]["seed_derivation"],
        "generator_state_before_sha256": documents[0][
            "generator_state_before_sha256"
        ],
        "generator_state_after_sha256": documents[0][
            "generator_state_after_sha256"
        ],
        "noise_q_sha256_by_draw": noise_hashes[0],
    }


def _copy_arm_atomically(
    source_arm: Path,
    target_arm: Path,
    *,
    verify_copy: Callable[[Path], None],
) -> None:
    """Copy one arm into an empty target, verifying before atomic exposure."""

    if target_arm.exists() and any(target_arm.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty canonical arm: {target_arm}.")
    staging_root = target_arm.parent.parent / ".collection_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / f"{target_arm.name}.collect-{uuid4().hex}"
    shutil.copytree(source_arm, temporary)
    try:
        verify_copy(temporary)
        if target_arm.exists():
            target_arm.rmdir()
        temporary.replace(target_arm)
    except BaseException:
        # Keep the copied temporary tree as forensic evidence.  It is never
        # exposed as canonical coverage and a later retry receives a new name.
        raise


def _authenticate_existing_copy(
    source_arm: Path,
    target_arm: Path,
    *,
    verify_copy: Callable[[Path], None],
) -> bool:
    """Authenticate an earlier atomic copy without modifying either tree."""

    if not target_arm.exists() or not any(target_arm.iterdir()):
        return False
    source_runs = _run_dirs(source_arm)
    target_runs = _run_dirs(target_arm)
    if (
        len(source_runs) != 1
        or len(target_runs) != 1
        or source_runs[0].name != target_runs[0].name
    ):
        raise RuntimeError(
            f"Existing canonical arm is not the unique incoming run: {target_arm}."
        )
    verify_copy(target_arm)
    source_run = source_runs[0]
    target_run = target_runs[0]
    control_files = (
        "config.resolved.json",
        "manifest.json",
        "metrics.jsonl",
        "result.json",
        "status.json",
    )
    mismatched = [
        filename
        for filename in control_files
        if not (source_run / filename).is_file()
        or not (target_run / filename).is_file()
        or sha256_file(source_run / filename) != sha256_file(target_run / filename)
    ]
    if mismatched:
        raise RuntimeError(
            "Existing canonical arm does not retain byte-identical incoming "
            f"control files: arm={target_arm.name!r}, files={mismatched!r}."
        )
    # Both run bundles have already passed registered-artifact hashing through
    # ``_completed_run_for_collection``/``verify_copy``.  Equal result bytes
    # therefore bind equal artifact paths, sizes, and content hashes.
    return True


def _artifact_verified_summary(
    *,
    task_python: Path,
    study_dir: Path,
) -> dict[str, Any]:
    """Run the public summary command and require its semantic success state."""

    completed = subprocess.run(
        (
            str(task_python),
            "-m",
            "ebl",
            "study",
            "summarize",
            "--study-dir",
            str(study_dir),
            "--verify-artifacts",
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Artifact-verified workflow summary command failed: "
            f"returncode={completed.returncode}, stderr={completed.stderr[-2000:]!r}."
        )
    summary_path = study_dir / "analysis" / "summary.json"
    summary = _read_json(summary_path)
    if (
        summary.get("schema") != "ebl.study.summary"
        or summary.get("schema_version") != 1
        or summary.get("study_id") != STUDY_ID
        or summary.get("validation_mode") != "full_artifact_hashes"
        or summary.get("state") != "ready_for_review"
        or summary.get("ready_for_review") is not True
    ):
        raise RuntimeError(
            "Artifact-verified workflow summary did not reach ready_for_review: "
            f"state={summary.get('state')!r}, "
            f"ready_for_review={summary.get('ready_for_review')!r}."
        )
    return {
        "path": summary_path.relative_to(study_dir).as_posix(),
        "sha256": sha256_file(summary_path),
        "state": summary["state"],
        "ready_for_review": True,
        "validation_mode": summary["validation_mode"],
    }


def collect_shard(
    *,
    canonical_study_dir: Path,
    incoming_study_dir: Path,
    incoming_shard: str,
    task_python: Path | None = None,
) -> dict[str, Any]:
    """Atomically collect disjoint arm directories; never overwrite evidence."""

    canonical_study_dir = canonical_study_dir.expanduser().resolve()
    incoming_study_dir = incoming_study_dir.expanduser().resolve()
    if canonical_study_dir == incoming_study_dir:
        raise RuntimeError("Canonical and incoming study directories must differ.")
    expected_plan_sha = sha256_file(PLAN_PATH)
    if (
        _materialized_plan_hash(canonical_study_dir) != expected_plan_sha
        or _materialized_plan_hash(incoming_study_dir) != expected_plan_sha
    ):
        raise RuntimeError("Canonical/incoming materialized source-plan mismatch.")
    tasks = load_shard_tasks(incoming_shard)
    task_map = _task_by_arm()
    allowed = {task.arm_id for task in tasks}
    for arm_dir in (incoming_study_dir / "runs").iterdir():
        if arm_dir.is_dir() and any(arm_dir.iterdir()) and arm_dir.name not in allowed:
            raise RuntimeError(f"Incoming study contains non-shard evidence: {arm_dir}.")

    verified: list[tuple[ArmTask, Path, str]] = []
    commits: set[str] = set()
    for task in tasks:
        run_dir, commit = _completed_run_for_collection(
            incoming_study_dir / "runs" / task.arm_id,
            task=task,
            verify_artifacts=True,
        )
        verified.append((task, run_dir, commit))
        commits.add(commit)
    if len(commits) != 1:
        raise RuntimeError("Incoming shard mixes execution source commits.")
    incoming_commit = next(iter(commits))
    canonical_commits: set[str] = set()
    canonical_runs: dict[str, Path] = {}
    for arm_id, task in task_map.items():
        if arm_id in allowed:
            continue
        run_dir, commit = _completed_run_for_collection(
            canonical_study_dir / "runs" / arm_id,
            task=task,
            verify_artifacts=True,
        )
        canonical_runs[arm_id] = run_dir
        canonical_commits.add(commit)
    if canonical_commits != {incoming_commit}:
        raise RuntimeError(
            "Canonical and incoming shards must use the same clean execution commit."
        )
    canonical_shard = "local" if incoming_shard == "akib" else "akib"
    incoming_runs = {
        task.arm_id: run_dir for task, run_dir, _commit in verified
    }
    parity = _epoch0_parity_sentinel(
        canonical_runs=canonical_runs,
        incoming_runs=incoming_runs,
        tasks=task_map,
        canonical_shard=canonical_shard,
        incoming_shard=incoming_shard,
    )
    combined_runs = {**canonical_runs, **incoming_runs}
    stream_parity = _adam_stream_sentinel(runs=combined_runs, tasks=task_map)
    fresh_pairing = _fresh_pairing_sentinel(
        runs=combined_runs,
        tasks=task_map,
        require_pair=True,
    )

    copied: list[dict[str, Any]] = []
    federated_run_pairs: list[tuple[Path, Path]] = []
    for task, run_dir, commit in verified:
        source_arm = run_dir.parent
        target_arm = canonical_study_dir / "runs" / task.arm_id
        def verify_copy(
            copied_arm: Path,
            *,
            task: ArmTask = task,
            commit: str = commit,
            run_dir: Path = run_dir,
        ) -> None:
            copied_run, copied_commit = _completed_run_for_collection(
                copied_arm,
                task=task,
                verify_artifacts=True,
            )
            if copied_commit != commit or copied_run.name != run_dir.name:
                raise RuntimeError(
                    "Copied arm verification does not match incoming evidence."
                )

        if _authenticate_existing_copy(
            source_arm,
            target_arm,
            verify_copy=verify_copy,
        ):
            collection_action = "authenticated_existing"
        else:
            _copy_arm_atomically(source_arm, target_arm, verify_copy=verify_copy)
            collection_action = "copied"
        target_run = target_arm / run_dir.name
        federated_run_pairs.append((run_dir, target_run))
        copied.append(
            {
                "arm_id": task.arm_id,
                "run_id": run_dir.name,
                "execution_source_commit": commit,
                "result_sha256": sha256_file(target_run / "result.json"),
                "collection_action": collection_action,
            }
        )

    federated_receipt_path = register_federated_runs(
        canonical_study_dir,
        incoming_study_dir,
        run_pairs=federated_run_pairs,
    )

    workflow_summary = None
    if task_python is not None:
        task_python = _require_executable(task_python, label="summary Python")
        workflow_summary = _artifact_verified_summary(
            task_python=task_python,
            study_dir=canonical_study_dir,
        )

    receipt = {
        "schema": "ebl.ibm_om_crossbar_hwa_recovery_canary_collection",
        "schema_version": 2,
        "study_id": STUDY_ID,
        "incoming_shard": incoming_shard,
        "incoming_study_dir": str(incoming_study_dir),
        "canonical_study_dir": str(canonical_study_dir),
        "source_plan_sha256": expected_plan_sha,
        "execution_source_commit": incoming_commit,
        "epoch0_cross_host_parity": parity,
        "cross_host_stream_parity": stream_parity,
        "fresh_apparent_pairing": fresh_pairing,
        "federated_provenance": {
            "path": federated_receipt_path.relative_to(
                canonical_study_dir
            ).as_posix(),
            "sha256": sha256_file(federated_receipt_path),
        },
        "workflow_summary": workflow_summary,
        "copied": copied,
        "collected_at": _utc_now(),
    }
    receipt_path = (
        canonical_study_dir
        / "analysis"
        / f"collection_{incoming_shard}_{_attempt_id()}.json"
    )
    atomic_write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


METRIC_TRIPLET = ("student_accuracy", "cross_entropy", "kl_teacher_student")


def _validation_states(value: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise RuntimeError("Expected validation metrics object.")
    if isinstance(value.get("validation"), Mapping):
        value = value["validation"]
    apparent = value.get("apparent_forward")
    persistent = value.get("persistent_diagnostic")
    if not isinstance(apparent, Mapping) or not isinstance(persistent, Mapping):
        raise RuntimeError(
            "Expected paired held-apparent and persistent validation metrics."
        )
    return apparent, persistent


def _metric_triplet(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {name: float(metrics[name]) for name in METRIC_TRIPLET}
    if any(not math.isfinite(value) for value in result.values()):
        raise RuntimeError("Expected finite accuracy, cross-entropy, and KL metrics.")
    return result


def _paired_metric_triplets(value: Any) -> dict[str, dict[str, float]]:
    apparent, persistent = _validation_states(value)
    return {
        "held_apparent_primary": _metric_triplet(apparent),
        "persistent_secondary": _metric_triplet(persistent),
    }


def _paired_metric_delta(
    selected: Mapping[str, Mapping[str, float]],
    initial: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        state: {
            name: float(selected[state][name]) - float(initial[state][name])
            for name in METRIC_TRIPLET
        }
        for state in ("held_apparent_primary", "persistent_secondary")
    }


def _optimizer_pulse_telemetry(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Expected {label} optimizer pulse telemetry.")
    integer_keys = (
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
    result: dict[str, Any] = {}
    for key in integer_keys:
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RuntimeError(f"Expected non-negative {label}.{key} telemetry.")
        result[key] = item
    by_layer = value.get("applied_pulses_by_layer")
    if (
        not isinstance(by_layer, (list, tuple))
        or len(by_layer) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in by_layer)
    ):
        raise RuntimeError(f"Expected two-layer {label} pulse telemetry.")
    cap = value.get("pulse_cap_per_cell")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise RuntimeError(f"Expected finite {label} pulse cap telemetry.")
    result["applied_pulses_by_layer"] = list(by_layer)
    result["pulse_cap_per_cell"] = cap
    if (
        result["commanded_pulses"] != result["applied_pulses"]
        or sum(result["applied_pulses_by_layer"]) != result["applied_pulses"]
        or result["maximum_pulses_per_cell"] > cap
    ):
        raise RuntimeError(f"Inconsistent {label} optimizer pulse telemetry.")
    return result


def _pulse_telemetry(
    metrics: Mapping[str, Any], epochs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected = _optimizer_pulse_telemetry(
        metrics.get("selected_optimizer"), label="selected"
    )
    training_final = _optimizer_pulse_telemetry(
        metrics.get("training_final_optimizer"), label="training_final"
    )
    per_epoch: list[dict[str, Any]] = []
    for item in epochs:
        row: dict[str, Any] = {
            "epoch": int(item["epoch"]),
            "execution": str(item["execution"]),
        }
        for key in ("commanded_pulses", "probability_clipped", "blocked_at_cap"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"Expected non-negative epoch {key} telemetry.")
            row[key] = value
        row["optimizer_cumulative"] = _optimizer_pulse_telemetry(
            item.get("optimizer_cumulative"),
            label=f"epoch_{row['epoch']}_cumulative",
        )
        per_epoch.append(row)
    cap = int(metrics["pulse_cap_per_cell"])
    if (
        cap != 640
        or selected["pulse_cap_per_cell"] != cap
        or training_final["pulse_cap_per_cell"] != cap
        or any(
            item["optimizer_cumulative"]["pulse_cap_per_cell"] != cap
            for item in per_epoch
        )
    ):
        raise RuntimeError("Adam pulse-cap telemetry does not match the declared cap.")
    return {
        "pulse_cap_per_cell": cap,
        "per_epoch": per_epoch,
        "selected_checkpoint_cumulative": selected,
        "training_final_cumulative": training_final,
        "cap_binding_observed": bool(
            selected["blocked_at_cap"]
            or selected["cells_at_cap"]
            or training_final["blocked_at_cap"]
            or training_final["cells_at_cap"]
        ),
    }


def _select_best_grid_row(
    rows: Sequence[Mapping[str, Any]], *, objective: str
) -> Mapping[str, Any]:
    if not rows:
        raise RuntimeError("Expected at least one Adam row for grid selection.")
    metric = (
        "cross_entropy"
        if objective == "supervised_cross_entropy"
        else "kl_teacher_student"
    )
    return min(
        rows,
        key=lambda row: (
            -float(row["selected"]["held_apparent_primary"]["student_accuracy"]),
            float(row["selected_delta"]["held_apparent_primary"][metric]),
            float(row["learning_rate"]),
            str(row["arm_id"]),
        ),
    )


def _arm_result(study_dir: Path, task: ArmTask) -> tuple[Path, dict[str, Any]]:
    run_dir, _ = _completed_run_for_collection(
        study_dir / "runs" / task.arm_id,
        task=task,
        verify_artifacts=True,
    )
    return run_dir, _read_json(run_dir / "result.json")


def analyze_study(study_dir: Path) -> dict[str, Any]:
    """Write a compact metric-first canary comparison after merged coverage."""

    study_dir = study_dir.expanduser().resolve()
    task_map = _task_by_arm()
    shard_manifest = _read_json(SHARD_MANIFEST_PATH)
    host_by_arm = {
        arm_id: shard
        for shard, record in shard_manifest["shards"].items()
        for arm_id in record["arm_ids"]
    }
    adam_rows: list[dict[str, Any]] = []
    fresh_rows: list[dict[str, Any]] = []
    runs_by_arm: dict[str, Path] = {}
    for arm_id, task in sorted(task_map.items()):
        run_dir, result = _arm_result(study_dir, task)
        runs_by_arm[arm_id] = run_dir
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"Missing metrics for {arm_id!r}.")
        if task.stage_kind == "on_chip_adam_diagnostic":
            candidates = metrics.get("selection_candidates")
            epochs = metrics.get("epochs")
            if (
                not isinstance(candidates, list)
                or [item.get("epoch") for item in candidates] != [0, 1, 2, 3]
                or not isinstance(epochs, list)
                or [item.get("epoch") for item in epochs] != [1, 2, 3]
                or any(not isinstance(item, Mapping) for item in epochs)
                or metrics.get("non_degradation_passed") is not True
            ):
                raise RuntimeError(f"Incomplete epoch-zero selection evidence for {arm_id!r}.")
            initial = _paired_metric_triplets(metrics.get("initial"))
            selected = _paired_metric_triplets(metrics.get("selected_validation"))
            training_final = _paired_metric_triplets(
                metrics.get("training_final_validation")
            )
            selected_delta = _paired_metric_delta(selected, initial)
            if selected["held_apparent_primary"]["student_accuracy"] < initial[
                "held_apparent_primary"
            ]["student_accuracy"]:
                raise RuntimeError(
                    f"Accuracy-first checkpoint degraded below epoch zero for {arm_id!r}."
                )
            row = {
                "arm_id": arm_id,
                "host_shard": host_by_arm[arm_id],
                "start_state": metrics.get("start_state"),
                "objective": metrics.get("objective"),
                "learning_rate": float(metrics["learning_rate"]),
                "pulse_cap_per_cell": metrics.get("pulse_cap_per_cell"),
                "selected_epoch": int(metrics["selected_epoch"]),
                "selection_metric": metrics.get("selection_metric"),
                "selection_objective_metric": metrics.get(
                    "selection_objective_metric"
                ),
                "selected_accuracy": float(metrics["selected_accuracy"]),
                "selected_objective_value": float(
                    metrics["selected_objective_value"]
                ),
                "non_degradation_passed": metrics.get("non_degradation_passed"),
                "initial": initial,
                "selected": selected,
                "training_final": training_final,
                "selected_delta": selected_delta,
                "training_final_delta": _paired_metric_delta(
                    training_final, initial
                ),
                "pulse_telemetry": _pulse_telemetry(metrics, epochs),
                "run_dir": str(run_dir),
                "result_sha256": sha256_file(run_dir / "result.json"),
            }
            adam_rows.append(row)
        else:
            draws = metrics.get("draws")
            mutation = metrics.get("mutation_check")
            if (
                metrics.get("intervention")
                != "counterfactual_post_write_apparent_noise_redraw_without_device_write"
                or not isinstance(draws, list)
                or len(draws) != 4
                or not isinstance(mutation, Mapping)
                or mutation.get("passed") is not True
            ):
                raise RuntimeError(f"Invalid fresh-apparent diagnostic evidence for {arm_id!r}.")
            fresh_rows.append(
                {
                    "arm_id": arm_id,
                    "host_shard": host_by_arm[arm_id],
                    "start_state": metrics.get("start_state"),
                    "configured_seed": metrics.get("configured_seed"),
                    "resolved_seed": metrics.get("resolved_seed"),
                    "seed_derivation": metrics.get("seed_derivation"),
                    "generator_state_before_sha256": metrics.get(
                        "generator_state_before_sha256"
                    ),
                    "generator_state_after_sha256": metrics.get(
                        "generator_state_after_sha256"
                    ),
                    "initial": _paired_metric_triplets(metrics.get("initial")),
                    "draws": [
                        {
                            "draw": item["draw"],
                            "fresh_apparent_q_sha256": item[
                                "fresh_apparent_q_sha256"
                            ],
                            "noise_q_sha256": item["noise_q_sha256"],
                            "validation": _paired_metric_triplets(
                                item["validation"]
                            ),
                        }
                        for item in draws
                    ],
                    "summary": metrics.get("summary"),
                    "mutation_check": dict(mutation),
                    "run_dir": str(run_dir),
                    "result_sha256": sha256_file(run_dir / "result.json"),
                }
            )
    if len(adam_rows) != 28 or len(fresh_rows) != 2:
        raise RuntimeError("Expected complete 28-Adam plus 2-fresh canary coverage.")
    if {row["host_shard"] for row in fresh_rows} != {"local"}:
        raise RuntimeError("Matched fresh-apparent controls must both run on local CUDA.")
    stream_parity = _adam_stream_sentinel(runs=runs_by_arm, tasks=task_map)
    fresh_pairing = _fresh_pairing_sentinel(
        runs=runs_by_arm,
        tasks=task_map,
        require_pair=True,
    )

    best: list[dict[str, Any]] = []
    for start_state in ("hwa_healthy_p0", "hwa_published_fault"):
        for objective in ("supervised_cross_entropy", "teacher_kl"):
            rows = [
                row
                for row in adam_rows
                if row["start_state"] == start_state and row["objective"] == objective
            ]
            objective_metric = (
                "cross_entropy"
                if objective == "supervised_cross_entropy"
                else "kl_teacher_student"
            )
            winner = _select_best_grid_row(rows, objective=objective)
            best.append(
                {
                    "start_state": start_state,
                    "objective": objective,
                    "host_shard": winner["host_shard"],
                    "selection_metric": "student_accuracy",
                    "selection_objective_metric": objective_metric,
                    "selection_basis": (
                        "maximum_selected_held_apparent_accuracy_then_selected_minus_"
                        "epoch0_declared_objective_delta_then_lower_learning_rate;_"
                        "host_retained_as_blocking_label"
                    ),
                    "arm_id": winner["arm_id"],
                    "learning_rate": winner["learning_rate"],
                    "selected_epoch": winner["selected_epoch"],
                    "initial": winner["initial"],
                    "selected": winner["selected"],
                    "selected_delta": winner["selected_delta"],
                    "pulse_telemetry": winner["pulse_telemetry"],
                }
            )

    summary = {
        "schema": "ebl.ibm_om_crossbar_hwa_recovery_canary_summary",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "generated_at": _utc_now(),
        "source_plan_sha256": sha256_file(PLAN_PATH),
        "source_reference_sha256": sha256_file(REFERENCE_PATH),
        "imported_source_commit": SOURCE_COMMIT,
        "coverage": {
            "adam_expected": 28,
            "adam_complete": len(adam_rows),
            "fresh_expected": 2,
            "fresh_complete": len(fresh_rows),
        },
        "adam": adam_rows,
        "fresh_apparent": fresh_rows,
        "adam_stream_parity": stream_parity,
        "fresh_apparent_pairing": fresh_pairing,
        "best_by_state_and_objective": best,
    }
    analysis_dir = study_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / "hwa_recovery_canary_summary.json"
    atomic_write_json(output_path, summary)
    lines = [
        "# Exact-P0 HWA recovery canary",
        "",
        f"Imported source commit: `{SOURCE_COMMIT}`",
        "",
        "| Start | Objective | Host | Best LR | Epoch | Apparent accuracy delta (pp) | Persistent accuracy delta (pp) | Apparent KL delta | Persistent KL delta | Apparent CE delta | Persistent CE delta | Selected pulses | Cap blocks |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            "| {start} | {objective} | {host} | {lr:g} | {epoch} | "
            "{apparent_accuracy:+.3f} | {persistent_accuracy:+.3f} | "
            "{apparent_kl:+.6f} | {persistent_kl:+.6f} | "
            "{apparent_ce:+.6f} | {persistent_ce:+.6f} | {pulses} | {blocked} |".format(
                start=row["start_state"],
                objective=row["objective"],
                host=row["host_shard"],
                lr=row["learning_rate"],
                epoch=row["selected_epoch"],
                apparent_accuracy=100.0
                * row["selected_delta"]["held_apparent_primary"]["student_accuracy"],
                persistent_accuracy=100.0
                * row["selected_delta"]["persistent_secondary"]["student_accuracy"],
                apparent_kl=row["selected_delta"]["held_apparent_primary"]
                ["kl_teacher_student"],
                persistent_kl=row["selected_delta"]["persistent_secondary"]
                ["kl_teacher_student"],
                apparent_ce=row["selected_delta"]["held_apparent_primary"]
                ["cross_entropy"],
                persistent_ce=row["selected_delta"]["persistent_secondary"]
                ["cross_entropy"],
                pulses=row["pulse_telemetry"]["selected_checkpoint_cumulative"]
                ["applied_pulses"],
                blocked=row["pulse_telemetry"]["selected_checkpoint_cumulative"]
                ["blocked_at_cap"],
            )
        )
    lines.extend(
        (
            "",
            "Fresh-apparent rows are counterfactual post-write-noise redraws, not physical inference-read noise.",
            "Persistent metrics remain secondary diagnostics; every optimizer-facing forward used held apparent q.",
            "",
        )
    )
    (analysis_dir / "hwa_recovery_canary_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return {**summary, "output_path": str(output_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--input-dir", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--shard", choices=("local", "akib"), required=True)
    run.add_argument("--input-dir", type=Path, required=True)
    run.add_argument("--results-root", type=Path, default=ROOT / "results")
    run.add_argument("--task-python", type=Path, default=Path(sys.executable))
    run.add_argument("--aihwkit-python", type=Path)
    run.add_argument("--mnist-root", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")

    collect = subparsers.add_parser("collect")
    collect.add_argument("--canonical-study-dir", type=Path, required=True)
    collect.add_argument("--incoming-study-dir", type=Path, required=True)
    collect.add_argument("--incoming-shard", choices=("local", "akib"), required=True)
    collect.add_argument("--task-python", type=Path)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--study-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-inputs":
        print(json.dumps(verify_import_bundle(args.input_dir), indent=2, sort_keys=True))
    elif args.command == "run":
        print(
            json.dumps(
                run_shard(
                    shard=args.shard,
                    input_dir=args.input_dir,
                    results_root=args.results_root,
                    task_python=args.task_python,
                    mnist_root=args.mnist_root,
                    aihwkit_python=args.aihwkit_python,
                    dry_run=args.dry_run,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "collect":
        print(
            json.dumps(
                collect_shard(
                    canonical_study_dir=args.canonical_study_dir,
                    incoming_study_dir=args.incoming_study_dir,
                    incoming_shard=args.incoming_shard,
                    task_python=args.task_python,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(analyze_study(args.study_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArmTask",
    "analyze_study",
    "collect_shard",
    "load_shard_tasks",
    "main",
    "run_shard",
    "verify_import_bundle",
]
