"""Fail-closed launcher for the corrected IBM OM common-window pilot.

The two native training processes may start only after the production modifier
has sampled the declared fixed array and its authoritative target mapper has
passed the predeclared reachability gates.  The launcher never implements
dual-rail grouping itself.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.current_simulations import refresh_current_simulations_for_run
from experiments.definitions import resolve_experiment_config
from experiments.study_workflow import load_study_record
from training.ibm_om.spec import backfill_config_dict


_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = (
    "mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2"
)
STUDY_PLAN = _ROOT / "studies" / f"{STUDY_ID}.json"
CONFIG_ROOT = (
    _ROOT / "examples" / "mnist_relu_drn" / "ibm_om_common_window_hwa_pilot"
)

TRAINING_ARM_CONFIGS = {
    "train-clean-common-window": "clean.json",
    "train-exact-bounds-hwa": "exact_bounds_hwa.json",
}
POPULATION_ROLES = {
    "train-clean-common-window": ("selection",),
    "train-exact-bounds-hwa": ("training", "selection"),
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

ASSIGNMENT_SEED = 84001
TRAINING_ENDPOINT_SEED = 84002
SELECTION_ENDPOINT_SEED = 84003
CORRUPTION_POLICY = "counterfactual_repaired"
TARGET_MAPPING = "dual_rail_quad_common_window"
COMMON_WINDOW_MARGIN_FRACTION = 0.25
DUAL_RAIL_LAYOUT_BY_PARAMETER = {
    "base.dense_weight.0": "halves",
    "base.dense_weight.1": "paired",
}
MAXIMUM_EMPTY_QUADS = 20
MAXIMUM_EMPTY_QUAD_FRACTION = 0.0005
ENDPOINT_APPLICATION_POLICY = (
    "aihwkit_apparent_forward_persistent_update_state"
)
COMPACT_FALLBACK_POLICY = (
    "pulse_resolved_noncorrupt_out_of_bound_empty_quad_only"
)
EXPECTED_EMPTY_QUAD_FALLBACK_DEVICES = 8
EXPECTED_COMPACT_ENDPOINT_DEVICES = 158792
EXPECTED_TOTAL_DEVICES = 158800
NETWORK_PREFLIGHT_EXAMPLES = 1600
NETWORK_PREFLIGHT_BATCHES = 100
MINIMUM_NETWORK_PREFLIGHT_STUDENT_ACCURACY = 0.25
MINIMUM_NETWORK_PREFLIGHT_TEACHER_AGREEMENT = 0.25
EXPECTED_EPOCHS_PER_ARM = 10
HEARTBEAT_SECONDS = 15.0

_CUDA_VISIBLE_DEVICES = re.compile(r"[0-9]+(?:,[0-9]+)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight and concurrently launch the two predeclared IBM OM "
            "common-window pilot arms."
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
            "executable for the pinned AIHWKit 1.1.0 environment; it is "
            "used only by the external fixed-population sampler"
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
    config_path: Path,
) -> None:
    if (
        arm.get("arm_id") != arm_id
        or arm.get("experiment_id") != "mnist_relu_drn_kd.v1"
        or arm.get("mode") != "train"
    ):
        raise RuntimeError(
            f"Expected prepared arm {arm_id!r} to declare "
            "mnist_relu_drn_kd.v1/train."
        )
    configs = arm.get("configs")
    if not isinstance(configs, list) or len(configs) != 1:
        raise RuntimeError(f"Expected arm {arm_id!r} to declare exactly one config.")
    record = configs[0]
    if not isinstance(record, Mapping):
        raise RuntimeError(f"Expected arm {arm_id!r} config to be a record.")
    resolved = record.get("resolved_path")
    if not isinstance(resolved, str) or Path(resolved).resolve() != config_path:
        raise RuntimeError(
            f"Expected arm {arm_id!r} to declare config {str(config_path)!r}."
        )
    if not config_path.is_file() or record.get("sha256") != sha256_file(config_path):
        raise RuntimeError(
            f"Expected arm {arm_id!r} config digest to match its prepared record."
        )


def _backfilled_parameters(value: Any) -> Any:
    """Complete stored modifier parameters that predate the layer fields."""

    return backfill_config_dict(value) if isinstance(value, Mapping) else value


def _expected_modifier_parameters(*, execution: str) -> dict[str, Any]:
    return backfill_config_dict(
        {
            "execution": execution,
            "assignment_seed": ASSIGNMENT_SEED,
            "endpoint_seed": (
                TRAINING_ENDPOINT_SEED
                if execution == "compact_endpoint"
                else SELECTION_ENDPOINT_SEED
            ),
            "corruption_policy": CORRUPTION_POLICY,
            "noisy_evaluation": execution == "pulse_resolved",
            "endpoint_policy": "clip_0_1",
            "target_out_of_support": "error",
            "preset": "reram_array_om",
            "controller": "adaptive",
            "start_protocol": "lower_to_target",
            "tolerance_step_ratio": 0.5,
            "maximum_program_pulses": 128,
            "target_mapping": TARGET_MAPPING,
            "dual_rail_layout_by_parameter": DUAL_RAIL_LAYOUT_BY_PARAMETER,
            "common_window_margin_fraction": COMMON_WINDOW_MARGIN_FRACTION,
        }
    )


def _validate_config_contracts() -> None:
    specs: dict[str, Any] = {}
    for arm, filename in TRAINING_ARM_CONFIGS.items():
        definition, spec = resolve_experiment_config(CONFIG_ROOT / filename, "train")
        if definition.experiment_id != "mnist_relu_drn_kd.v1":
            raise RuntimeError(f"Expected arm {arm!r} to use the MNIST DRN runtime.")
        specs[arm] = spec

    clean = specs["train-clean-common-window"]
    exact = specs["train-exact-bounds-hwa"]
    for field in ("runtime", "data", "teacher", "model", "solver", "mapping"):
        if getattr(clean, field) != getattr(exact, field):
            raise RuntimeError(f"Expected matched pilot field {field!r} to be equal.")
    for field in (
        "num_epochs",
        "learning_rates",
        "temperature",
        "log_every",
        "max_batches",
        "max_validation_batches",
        "minimum_relative_kl_improvement",
        "selection_evaluation",
        "selection_noise_repeats",
        "update_backend",
    ):
        if getattr(clean.settings, field) != getattr(exact.settings, field):
            raise RuntimeError(
                f"Expected matched pilot training field {field!r} to be equal."
            )
    if clean.settings.num_epochs != EXPECTED_EPOCHS_PER_ARM:
        raise RuntimeError("Expected both corrected pilot arms to run ten epochs.")
    if clean.settings.weight_modifier.type != "none":
        raise RuntimeError("Expected the clean pilot arm to train without HWA.")
    if (
        exact.settings.weight_modifier.type != "ibm_reram_om_program_verify"
        or dict(exact.settings.weight_modifier.parameters)
        != _expected_modifier_parameters(execution="compact_endpoint")
    ):
        raise RuntimeError(
            "Expected the exact-bounds arm to use the declared compact IBM OM HWA."
        )
    expected_selection = _expected_modifier_parameters(execution="pulse_resolved")
    for arm, spec in specs.items():
        modifier = spec.settings.selection_weight_modifier
        if (
            modifier.type != "ibm_reram_om_program_verify"
            or dict(modifier.parameters) != expected_selection
        ):
            raise RuntimeError(
                f"Expected arm {arm!r} to use the matched repaired deployment."
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
        raise RuntimeError("Expected the prepared study to name the tracked plan.")
    if source_plan.get("sha256") != sha256_file(STUDY_PLAN):
        raise RuntimeError(
            "Expected the prepared study to match the current tracked plan."
        )

    raw_arms = study.get("arms")
    if not isinstance(raw_arms, list):
        raise RuntimeError("Expected the prepared study to contain arm records.")
    arms = {
        str(arm.get("arm_id")): arm
        for arm in raw_arms
        if isinstance(arm, Mapping)
    }
    if set(arms) != set(TRAINING_ARM_CONFIGS) or len(raw_arms) != len(arms):
        raise RuntimeError("Expected the corrected pilot to declare exactly two arms.")
    for arm, filename in TRAINING_ARM_CONFIGS.items():
        _validate_arm_record(
            arms[arm],
            arm_id=arm,
            config_path=(CONFIG_ROOT / filename).resolve(),
        )

    runs_root = study_dir / "runs"
    observed_roots = (
        {path.name for path in runs_root.iterdir() if path.is_dir()}
        if runs_root.is_dir()
        else set()
    )
    if observed_roots != set(TRAINING_ARM_CONFIGS):
        raise RuntimeError(
            "Expected runs/ to contain exactly one prepared root per pilot arm."
        )
    existing_attempts = {
        arm: [str(path) for path in _native_attempts(runs_root / arm)]
        for arm in TRAINING_ARM_CONFIGS
        if _native_attempts(runs_root / arm)
    }
    launch_root = study_dir / "launch"
    prior_launches = (
        [str(path) for path in sorted(launch_root.iterdir())]
        if launch_root.is_dir()
        else []
    )
    if existing_attempts or prior_launches:
        raise RuntimeError(
            "Expected a fresh prepared pilot with no native or launcher attempts. "
            "Preflight and native attempts are immutable; do not relaunch this "
            "study in place. "
            f"Provided value: native={existing_attempts!r}, launch={prior_launches!r}."
        )

    _validate_config_contracts()
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
            str((CONFIG_ROOT / filename).resolve()),
            "--output-dir",
            str((study_dir / "runs" / arm).resolve()),
            "--teacher-weights",
            str(checkpoint_path.resolve()),
            "--device-model",
            str(device_model_path.resolve()),
        ]
        for arm, filename in TRAINING_ARM_CONFIGS.items()
    }


@contextmanager
def _external_sampler_environment(aihwkit_python: Path) -> Iterator[None]:
    name = "EBL_AIHWKIT_PYTHON"
    previous = os.environ.get(name)
    os.environ[name] = str(aihwkit_python)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _evaluate_apparent_forward_canary(
    modifier,
    *,
    evaluator,
    stack,
    teacher,
    validation_loader,
) -> Mapping[str, Any]:
    """Evaluate once inside the already-programmed apparent endpoint context."""

    with modifier.evaluation_context():
        metrics = evaluator(
            stack,
            teacher,
            validation_loader,
            maximum_batches=NETWORK_PREFLIGHT_BATCHES,
        )
    if not isinstance(metrics, Mapping) or metrics.get(
        "examples"
    ) != NETWORK_PREFLIGHT_EXAMPLES:
        raise RuntimeError(
            "Expected the apparent-forward network canary to evaluate exactly "
            f"{NETWORK_PREFLIGHT_EXAMPLES} validation examples."
        )
    return metrics


def _run_authoritative_preflight(
    *,
    attempt_dir: Path,
    aihwkit_python: Path,
    checkpoint_path: Path = BOUNDED_CHECKPOINT,
    device_model_path: Path = DEVICE_MODEL,
) -> dict[str, Any]:
    """Invoke the production modifier and its target-mapping preflight."""

    import torch

    from experiments.mnist_relu_drn.components import build_student_stack
    from experiments.mnist_relu_drn.runtime import (
        _evaluate,
        _load_teacher,
        _model_checkpoint_metadata,
    )
    from experiments.mnist_shared import build_mnist_loaders
    from experiments.schema import to_plain_data
    from training.checkpoint import atomic_torch_save
    from training.ibm_reram_hwa import (
        IbmReramHwaConfig,
        build_ibm_reram_hwa_modifier,
    )

    config_path = (CONFIG_ROOT / "exact_bounds_hwa.json").resolve()
    _definition, spec = resolve_experiment_config(config_path, "train")
    stack = build_student_stack(spec, enable_measured=False)
    teacher, checkpoint_metadata = _load_teacher(
        checkpoint_path,
        device=stack.device,
        spec=spec,
    )
    teacher.copy_named_weights_to(stack.bundle.catalog)
    data = build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed,
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )

    preflight_dir = attempt_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=False)
    population_path = preflight_dir / "ibm_om_population.preflight.npz"
    receipt_path = preflight_dir / "ibm_om_population.preflight.receipt.json"
    parameters = dict(spec.settings.selection_weight_modifier.parameters)
    with _external_sampler_environment(aihwkit_python):
        modifier = build_ibm_reram_hwa_modifier(
            stack.bundle.catalog.trainable,
            IbmReramHwaConfig(**parameters),
            device_model_path=device_model_path,
            conductance_min=spec.model.conductance_min,
            conductance_max=spec.model.conductance_max,
            population_path=population_path,
            population_receipt_path=receipt_path,
        )
    mapped, mapping_report = modifier.preflight_target_mapping()
    mapped_cpu = mapped.detach().cpu()
    mapped_path = preflight_dir / "mapped_targets.pt"
    atomic_torch_save(mapped_cpu, mapped_path)

    compact_population_path = (
        preflight_dir / "ibm_om_population.compact_training_preflight.npz"
    )
    compact_receipt_path = (
        preflight_dir
        / "ibm_om_population.compact_training_preflight.receipt.json"
    )
    compact_parameters = dict(spec.settings.weight_modifier.parameters)
    with _external_sampler_environment(aihwkit_python):
        compact_modifier = build_ibm_reram_hwa_modifier(
            stack.bundle.catalog.trainable,
            IbmReramHwaConfig(**compact_parameters),
            device_model_path=device_model_path,
            conductance_min=spec.model.conductance_min,
            conductance_max=spec.model.conductance_max,
            population_path=compact_population_path,
            population_receipt_path=compact_receipt_path,
        )
    compact_mapped, compact_mapping_report = (
        compact_modifier.preflight_target_mapping()
    )
    if (
        compact_modifier.population_fingerprint
        != modifier.population_fingerprint
        or compact_mapping_report != mapping_report
        or not torch.equal(compact_mapped, mapped)
    ):
        raise RuntimeError(
            "Expected compact HWA and pulse-resolved selection preflights to "
            "use the exact same fixed population, mapper, and mapped targets."
        )
    with compact_modifier.training_context():
        pass
    compact_report = compact_modifier.programming_report
    compact_deployment = compact_modifier.last_deployment_bundle
    if compact_report is None or compact_deployment is None:
        raise RuntimeError(
            "Expected one real compact HWA context to produce a report and "
            "hybrid deployment bundle."
        )
    if (
        compact_report.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or compact_deployment.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or compact_report.get("population_fingerprint")
        != modifier.population_fingerprint
    ):
        raise RuntimeError(
            "Expected the compact HWA canary to preserve endpoint policy and "
            "the exact selection-population fingerprint."
        )
    compact_deployment_path = (
        preflight_dir / "compact_training_canary_deployment.pt"
    )
    atomic_torch_save(compact_deployment, compact_deployment_path)
    compact_report_path = preflight_dir / "compact_training_canary_report.json"
    atomic_write_json(compact_report_path, compact_report)
    compact_receipt = _read_json(compact_receipt_path)
    if compact_receipt is None:
        raise RuntimeError("Expected the compact HWA population receipt.")

    # This is the sole pulse-resolved selection programming context.  The
    # separate compact-HWA context above is a training-path canary.  The
    # production evaluator receives no modifier, so it reads these already-
    # applied apparent selection endpoints without reprogramming them.
    apparent_metrics = _evaluate_apparent_forward_canary(
        modifier,
        evaluator=_evaluate,
        stack=stack,
        teacher=teacher,
        validation_loader=data.validation,
    )
    programming_report = modifier.programming_report
    deployment = modifier.last_deployment_bundle
    if programming_report is None or deployment is None:
        raise RuntimeError(
            "Expected the pulse-resolved selection canary to produce a report "
            "and persistent deployment bundle."
        )
    if (
        programming_report.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or deployment.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
    ):
        raise RuntimeError(
            "Expected the pulse canary report and deployment bundle to bind "
            "the apparent-forward, persistent-update-state policy."
        )
    deployment_path = preflight_dir / "pulse_resolved_canary_deployment.pt"
    atomic_torch_save(deployment, deployment_path)
    network_metrics_path = preflight_dir / "apparent_forward_network_metrics.json"
    atomic_write_json(
        network_metrics_path,
        {
            "schema": "ebl.mnist_ibm_om_common_window.network_preflight",
            "schema_version": 1,
            "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            "forward_endpoint": "aihwkit_apparent_endpoint",
            "programming_contexts": 1,
            "cohort": {
                "split": "validation",
                "data_seed": spec.runtime.data_seed,
                "validation_points": spec.data.validation_points,
                "batch_size": spec.data.batch_size,
                "maximum_batches": NETWORK_PREFLIGHT_BATCHES,
                "examples": NETWORK_PREFLIGHT_EXAMPLES,
                "shuffle": False,
            },
            "metrics": apparent_metrics,
        },
    )
    checkpoint_metadata_canary = _model_checkpoint_metadata(
        stack,
        spec=spec,
        teacher_sha256=sha256_file(checkpoint_path),
        mapping=None,
        deployment_source=None,
        device_model_sha256=sha256_file(device_model_path),
    )
    selection_layout = checkpoint_metadata_canary[
        "selection_weight_modifier"
    ]["parameters"]["dual_rail_layout_by_parameter"]
    training_layout = checkpoint_metadata_canary["weight_modifier"][
        "parameters"
    ]["dual_rail_layout_by_parameter"]
    if type(selection_layout) is not dict or type(training_layout) is not dict:
        raise RuntimeError(
            "Expected v2 checkpoint metadata to recursively thaw both nested "
            "dual-rail layout mappings at the JSON artifact boundary."
        )
    checkpoint_metadata_path = (
        preflight_dir / "selected_checkpoint_metadata_canary.json"
    )
    atomic_write_json(checkpoint_metadata_path, checkpoint_metadata_canary)
    finite = torch.isfinite(mapped_cpu)
    outside = finite & ((mapped_cpu < 0.0) | (mapped_cpu > 1.0))
    receipt = _read_json(receipt_path)
    if receipt is None:
        raise RuntimeError("Expected the authoritative sampler receipt.")
    return {
        "schema": "ebl.mnist_ibm_om_common_window.mapping_preflight",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_at": _utc_now(),
        "authoritative_entry_point": (
            "training.ibm_reram_hwa."
            "IbmReramHwaParameterModifier.preflight_target_mapping"
        ),
        "checkpoint_metadata_schema": checkpoint_metadata.get("schema"),
        "inputs": {
            "bounded_drn_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
            },
            "ibm_om_device_model": {
                "path": str(device_model_path.resolve()),
                "sha256": sha256_file(device_model_path),
            },
        },
        "configs": {
            arm: {
                "path": str((CONFIG_ROOT / filename).resolve()),
                "sha256": sha256_file(CONFIG_ROOT / filename),
            }
            for arm, filename in TRAINING_ARM_CONFIGS.items()
        },
        "modifier_role": "selection",
        "modifier_parameters": to_plain_data(parameters),
        "compact_training_modifier_parameters": to_plain_data(
            compact_parameters
        ),
        "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
        "operational_retry": {
            "retry_of_study_id": (
                "mnist-ibm-om-common-window-hwa-program-verify-pilot-"
                "20260824-v1"
            ),
            "v1_failure_class": "checkpoint_metadata_serialization",
            "v1_training_updates": 0,
            "v2_model_contract_delta": (
                "eight_empty_quad_out_of_bound_cells_use_exact_"
                "pulse_resolved_fallback"
            ),
            "v2_prepared_before_delta": False,
            "metadata_serialization_canary": "passed",
        },
        "mapping_report": mapping_report,
        "compact_training_mapping_report": compact_mapping_report,
        "compact_training_canary_report": compact_report,
        "pulse_resolved_canary_report": programming_report,
        "network_preflight": {
            "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            "forward_endpoint": "aihwkit_apparent_endpoint",
            "persistent_endpoint_role": "hidden_update_state",
            "programming_contexts": 1,
            "cohort": {
                "split": "validation",
                "data_seed": spec.runtime.data_seed,
                "validation_points": spec.data.validation_points,
                "batch_size": spec.data.batch_size,
                "maximum_batches": NETWORK_PREFLIGHT_BATCHES,
                "examples": NETWORK_PREFLIGHT_EXAMPLES,
                "shuffle": False,
            },
            "apparent_forward_metrics": apparent_metrics,
            "ideal_mapped_network_diagnostic": {
                "status": "not_run",
                "reason": "avoids a second launcher-side weight-mutation surface",
            },
            "persistent_network_diagnostic": {
                "status": "not_run",
                "reason": (
                    "persistent endpoints are hidden update state; any later "
                    "network replay is exploratory and non-gating"
                ),
            },
        },
        "population_receipt": dict(receipt),
        "compact_training_population_receipt": dict(compact_receipt),
        "artifacts": {
            "population": {
                "path": str(population_path),
                "sha256": sha256_file(population_path),
            },
            "population_receipt": {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            },
            "mapped_targets": {
                "path": str(mapped_path),
                "sha256": sha256_file(mapped_path),
            },
            "compact_training_population": {
                "path": str(compact_population_path),
                "sha256": sha256_file(compact_population_path),
            },
            "compact_training_population_receipt": {
                "path": str(compact_receipt_path),
                "sha256": sha256_file(compact_receipt_path),
            },
            "compact_training_canary_report": {
                "path": str(compact_report_path),
                "sha256": sha256_file(compact_report_path),
                "size_bytes": compact_report_path.stat().st_size,
            },
            "compact_training_canary_deployment": {
                "path": str(compact_deployment_path),
                "sha256": sha256_file(compact_deployment_path),
                "size_bytes": compact_deployment_path.stat().st_size,
                "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            },
            "pulse_resolved_canary_deployment": {
                "path": str(deployment_path),
                "sha256": sha256_file(deployment_path),
                "size_bytes": deployment_path.stat().st_size,
                "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            },
            "apparent_forward_network_metrics": {
                "path": str(network_metrics_path),
                "sha256": sha256_file(network_metrics_path),
                "size_bytes": network_metrics_path.stat().st_size,
            },
            "selected_checkpoint_metadata_canary": {
                "path": str(checkpoint_metadata_path),
                "sha256": sha256_file(checkpoint_metadata_path),
                "size_bytes": checkpoint_metadata_path.stat().st_size,
            },
        },
        "mapped_targets": {
            "count": int(mapped_cpu.numel()),
            "finite_count": int(finite.sum().item()),
            "outside_0_1_count": int(outside.sum().item()),
            "minimum": float(mapped_cpu.min().item()),
            "maximum": float(mapped_cpu.max().item()),
        },
    }


def _validate_authoritative_mapping_report(
    report: Mapping[str, Any],
    *,
    expected_population_fingerprint: str | None = None,
) -> str:
    from training.ibm_reram_hwa import (
        IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
        validate_ibm_reram_target_mapping_preflight,
    )

    if IBM_RERAM_ENDPOINT_APPLICATION_POLICY != ENDPOINT_APPLICATION_POLICY:
        raise RuntimeError(
            "Expected the launcher endpoint policy to match the production "
            "IBM OM modifier contract."
        )
    validate_ibm_reram_target_mapping_preflight(
        report,
        maximum_empty_quads=MAXIMUM_EMPTY_QUADS,
    )
    fingerprint = report.get("population_fingerprint")
    if not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None:
        raise RuntimeError("Expected the mapper report to contain a fingerprint.")
    if (
        expected_population_fingerprint is not None
        and fingerprint != expected_population_fingerprint
    ):
        raise RuntimeError("Expected mapper population fingerprint parity.")
    if (
        report.get("target_mapping") != TARGET_MAPPING
        or report.get("corruption_policy") != CORRUPTION_POLICY
        or report.get("common_window_margin_fraction")
        != COMMON_WINDOW_MARGIN_FRACTION
        or report.get("global_target_outside_0_1") != 0
        or report.get("corrupt_quad_count") != 0
    ):
        raise RuntimeError("Expected the exact repaired common-window mapper report.")
    devices = report.get("devices")
    quads = report.get("quad_count")
    empty = report.get("common_window_empty_quad_count")
    empty_fraction = report.get("common_window_empty_fraction")
    if (
        isinstance(devices, bool)
        or not isinstance(devices, int)
        or isinstance(quads, bool)
        or not isinstance(quads, int)
        or devices != 4 * quads
        or isinstance(empty, bool)
        or not isinstance(empty, int)
        or not 0 <= empty <= MAXIMUM_EMPTY_QUADS
        or isinstance(empty_fraction, bool)
        or not isinstance(empty_fraction, (int, float))
        or not math.isfinite(float(empty_fraction))
        or float(empty_fraction) > MAXIMUM_EMPTY_QUAD_FRACTION
        or not math.isclose(
            float(empty_fraction),
            empty / quads,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("Expected the declared structural quad reachability gate.")
    raw_parameters = report.get("parameters")
    if not isinstance(raw_parameters, Mapping) or set(raw_parameters) != set(
        DUAL_RAIL_LAYOUT_BY_PARAMETER
    ):
        raise RuntimeError("Expected mapper reports for both declared parameters.")
    for key, layout in DUAL_RAIL_LAYOUT_BY_PARAMETER.items():
        parameter = raw_parameters[key]
        if not isinstance(parameter, Mapping) or parameter.get(
            "dual_rail_layout"
        ) != layout:
            raise RuntimeError("Expected the declared dual-rail parameter layouts.")
    return fingerprint


def _validate_compact_training_canary(
    report: Mapping[str, Any],
    *,
    mapping_report: Mapping[str, Any],
    expected_population_fingerprint: str,
    expected_fallback_indices_sha256: str | None = None,
) -> str:
    expected_below = mapping_report.get(
        "mapped_target_below_lower_bound_empty_quad"
    )
    expected_above = mapping_report.get(
        "mapped_target_above_upper_bound_empty_quad"
    )
    if (
        isinstance(expected_below, bool)
        or not isinstance(expected_below, int)
        or isinstance(expected_above, bool)
        or not isinstance(expected_above, int)
        or expected_below + expected_above
        != EXPECTED_EMPTY_QUAD_FALLBACK_DEVICES
    ):
        raise RuntimeError(
            "Expected the fixed seed-84001 mapper to expose exactly eight "
            "out-of-bound cells from permitted empty quads."
        )
    devices = report.get("devices")
    compact_devices = report.get("compact_endpoint_devices")
    fallback_devices = report.get("pulse_resolved_fallback_devices")
    accepted = report.get("accepted")
    success_fraction = report.get("success_fraction")
    fallback = report.get("pulse_resolved_fallback")
    if (
        report.get("execution") != "compact_endpoint"
        or report.get("execution_detail")
        != "compact_endpoint_with_exact_empty_quad_fallback"
        or report.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or report.get("endpoint_seed") != TRAINING_ENDPOINT_SEED
        or report.get("assignment_seed") != ASSIGNMENT_SEED
        or report.get("corruption_policy") != CORRUPTION_POLICY
        or report.get("population_fingerprint")
        != expected_population_fingerprint
        or report.get("device_model_sha256") != DEVICE_MODEL_SHA256
        or isinstance(devices, bool)
        or not isinstance(devices, int)
        or devices != EXPECTED_TOTAL_DEVICES
        or isinstance(compact_devices, bool)
        or not isinstance(compact_devices, int)
        or compact_devices != EXPECTED_COMPACT_ENDPOINT_DEVICES
        or isinstance(fallback_devices, bool)
        or not isinstance(fallback_devices, int)
        or fallback_devices != EXPECTED_EMPTY_QUAD_FALLBACK_DEVICES
        or compact_devices + fallback_devices != devices
        or report.get("target_below_lower_bound") != expected_below
        or report.get("target_above_upper_bound") != expected_above
        or report.get("target_inside_bounds") != compact_devices
        or report.get("corrupt") != 0
        or isinstance(accepted, bool)
        or not isinstance(accepted, int)
        or not 0 <= accepted <= devices
        or isinstance(success_fraction, bool)
        or not isinstance(success_fraction, (int, float))
        or not math.isfinite(float(success_fraction))
        or not math.isclose(
            float(success_fraction),
            accepted / devices,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or not isinstance(fallback, Mapping)
    ):
        raise RuntimeError(
            "Expected the compact preflight to partition the fixed population "
            "into fitted compact endpoints and exactly eight transparent "
            "empty-quad fallbacks."
        )

    indices = fallback.get("selection_indices")
    fallback_indices_sha256 = fallback.get("selection_indices_sha256")
    fallback_accepted = fallback.get("accepted")
    fallback_budget = fallback.get("budget_exhausted")
    fallback_success_fraction = fallback.get("success_fraction")
    fallback_unreachable = fallback.get("acceptance_window_unreachable")
    fallback_saturated = fallback.get("saturated")
    fallback_clipped = fallback.get("endpoint_clipped")
    pulse_count = fallback.get("pulse_count")
    verify_mean = fallback.get("verify_count_mean")
    reversal_mean = fallback.get("reversal_count_mean")
    if (
        fallback.get("policy") != COMPACT_FALLBACK_POLICY
        or fallback.get("devices") != EXPECTED_EMPTY_QUAD_FALLBACK_DEVICES
        or not isinstance(indices, list)
        or len(indices) != EXPECTED_EMPTY_QUAD_FALLBACK_DEVICES
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < devices
            for index in indices
        )
        or len(set(indices)) != len(indices)
        or not isinstance(fallback_indices_sha256, str)
        or _SHA256.fullmatch(fallback_indices_sha256) is None
        or (
            expected_fallback_indices_sha256 is not None
            and fallback_indices_sha256
            != expected_fallback_indices_sha256
        )
        or fallback.get("target_below_lower_bound") != expected_below
        or fallback.get("target_above_upper_bound") != expected_above
        or fallback.get("start_state") != "sampled_fully_reset_bound"
        or fallback.get("controller") != "adaptive"
        or fallback.get("maximum_program_pulses") != 128
        or fallback.get("random_stream_order")
        != "after_compact_endpoint_sampling"
        or fallback.get("nonfinite") != 0
        or isinstance(fallback_accepted, bool)
        or not isinstance(fallback_accepted, int)
        or isinstance(fallback_budget, bool)
        or not isinstance(fallback_budget, int)
        or fallback_accepted + fallback_budget
        != EXPECTED_EMPTY_QUAD_FALLBACK_DEVICES
        or isinstance(fallback_success_fraction, bool)
        or not isinstance(fallback_success_fraction, (int, float))
        or not math.isfinite(float(fallback_success_fraction))
        or not math.isclose(
            float(fallback_success_fraction),
            fallback_accepted / EXPECTED_EMPTY_QUAD_FALLBACK_DEVICES,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= EXPECTED_EMPTY_QUAD_FALLBACK_DEVICES
            for value in (
                fallback_unreachable,
                fallback_saturated,
                fallback_clipped,
            )
        )
        or not isinstance(pulse_count, Mapping)
    ):
        raise RuntimeError(
            "Expected the compact fallback to select only the authoritative "
            "out-of-bound empty-quad cells and preserve fail-closed compact "
            "sampling elsewhere."
        )

    pulse_mean = pulse_count.get("mean")
    pulse_median = pulse_count.get("median")
    pulse_maximum = pulse_count.get("maximum")
    set_mean = pulse_count.get("set_mean")
    reset_mean = pulse_count.get("reset_mean")
    if (
        not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in (
                pulse_mean,
                pulse_median,
                pulse_maximum,
                set_mean,
                reset_mean,
                verify_mean,
                reversal_mean,
            )
        )
        or not 0.0 <= float(pulse_mean) <= 128.0
        or not 0.0 <= float(pulse_median) <= 128.0
        or isinstance(pulse_maximum, bool)
        or not isinstance(pulse_maximum, int)
        or not 0 <= pulse_maximum <= 128
        or not math.isclose(
            float(set_mean) + float(reset_mean),
            float(pulse_mean),
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        or not 1.0 <= float(verify_mean) <= 129.0
        or not 0.0 <= float(reversal_mean) <= 128.0
    ):
        raise RuntimeError(
            "Expected finite compact-fallback pulse, verify, and reversal "
            "cost within the declared 128-pulse controller cap."
        )
    return fallback_indices_sha256


def _validate_preflight_report(report: Mapping[str, Any]) -> str:
    if (
        report.get("schema")
        != "ebl.mnist_ibm_om_common_window.mapping_preflight"
        or report.get("schema_version") != 1
        or report.get("study_id") != STUDY_ID
        or report.get("authoritative_entry_point")
        != (
            "training.ibm_reram_hwa."
            "IbmReramHwaParameterModifier.preflight_target_mapping"
        )
    ):
        raise RuntimeError("Expected the authoritative mapping preflight schema.")
    inputs = report.get("inputs")
    configs = report.get("configs")
    if not isinstance(inputs, Mapping) or not isinstance(configs, Mapping):
        raise RuntimeError("Expected preflight input and config provenance.")
    if (
        inputs.get("bounded_drn_checkpoint", {}).get("sha256")
        != BOUNDED_CHECKPOINT_SHA256
        or inputs.get("ibm_om_device_model", {}).get("sha256")
        != DEVICE_MODEL_SHA256
    ):
        raise RuntimeError("Expected preflight to bind the frozen input digests.")
    for arm, filename in TRAINING_ARM_CONFIGS.items():
        record = configs.get(arm)
        if not isinstance(record, Mapping) or record.get("sha256") != sha256_file(
            CONFIG_ROOT / filename
        ):
            raise RuntimeError("Expected preflight to bind both prepared configs.")
    if (
        report.get("modifier_role") != "selection"
        or _backfilled_parameters(report.get("modifier_parameters"))
        != _expected_modifier_parameters(execution="pulse_resolved")
        or _backfilled_parameters(
            report.get("compact_training_modifier_parameters")
        )
        != _expected_modifier_parameters(execution="compact_endpoint")
    ):
        raise RuntimeError(
            "Expected preflight to use the declared compact training and "
            "pulse-resolved selection modifiers."
        )
    mapping = report.get("mapping_report")
    compact_mapping = report.get("compact_training_mapping_report")
    receipt = report.get("population_receipt")
    compact_receipt = report.get("compact_training_population_receipt")
    mapped = report.get("mapped_targets")
    compact_canary = report.get("compact_training_canary_report")
    canary = report.get("pulse_resolved_canary_report")
    network_preflight = report.get("network_preflight")
    operational_retry = report.get("operational_retry")
    artifacts = report.get("artifacts")
    if not all(
        isinstance(value, Mapping)
        for value in (
            mapping,
            compact_mapping,
            receipt,
            compact_receipt,
            mapped,
            compact_canary,
            canary,
            network_preflight,
            operational_retry,
            artifacts,
        )
    ):
        raise RuntimeError("Expected complete authoritative preflight outputs.")
    fingerprint = _validate_authoritative_mapping_report(mapping)
    compact_fingerprint = _validate_authoritative_mapping_report(
        compact_mapping,
        expected_population_fingerprint=fingerprint,
    )
    if compact_mapping != mapping or compact_fingerprint != fingerprint:
        raise RuntimeError(
            "Expected compact HWA and pulse-resolved selection to share the "
            "exact authoritative mapping report."
        )
    metadata_record = artifacts.get("selected_checkpoint_metadata_canary")
    if (
        operational_retry
        != {
            "retry_of_study_id": (
                "mnist-ibm-om-common-window-hwa-program-verify-pilot-"
                "20260824-v1"
            ),
            "v1_failure_class": "checkpoint_metadata_serialization",
            "v1_training_updates": 0,
            "v2_model_contract_delta": (
                "eight_empty_quad_out_of_bound_cells_use_exact_"
                "pulse_resolved_fallback"
            ),
            "v2_prepared_before_delta": False,
            "metadata_serialization_canary": "passed",
        }
        or not isinstance(metadata_record, Mapping)
        or not isinstance(metadata_record.get("sha256"), str)
        or _SHA256.fullmatch(str(metadata_record.get("sha256"))) is None
    ):
        raise RuntimeError(
            "Expected v2 preflight to bind the passed v1 checkpoint-metadata "
            "serialization regression canary."
        )
    if (
        receipt.get("population_fingerprint") != fingerprint
        or receipt.get("aihwkit_version") != "1.1.0"
        or receipt.get("request", {}).get("assignment_seed") != ASSIGNMENT_SEED
        or receipt.get("request", {}).get("corruption_policy")
        != CORRUPTION_POLICY
        or receipt.get("request", {}).get("binding_keys")
        != list(DUAL_RAIL_LAYOUT_BY_PARAMETER)
    ):
        raise RuntimeError("Expected the pinned fixed-population sampling receipt.")
    if (
        compact_receipt.get("population_fingerprint") != fingerprint
        or compact_receipt.get("aihwkit_version") != "1.1.0"
        or compact_receipt.get("request") != receipt.get("request")
    ):
        raise RuntimeError(
            "Expected compact HWA to independently reproduce the exact "
            "preflight fixed-population fingerprint and request."
        )
    if (
        mapped.get("count") != mapping.get("devices")
        or mapped.get("finite_count") != mapped.get("count")
        or mapped.get("outside_0_1_count") != 0
        or not all(
            isinstance(mapped.get(key), (int, float))
            and not isinstance(mapped.get(key), bool)
            and math.isfinite(float(mapped[key]))
            for key in ("minimum", "maximum")
        )
        or float(mapped["minimum"]) < 0.0
        or float(mapped["maximum"]) > 1.0
    ):
        raise RuntimeError("Expected finite mapped targets inside [0, 1].")
    compact_report_record = artifacts.get("compact_training_canary_report")
    compact_deployment_record = artifacts.get(
        "compact_training_canary_deployment"
    )
    compact_population_record = artifacts.get("compact_training_population")
    compact_receipt_record = artifacts.get(
        "compact_training_population_receipt"
    )
    if (
        not all(
            isinstance(value, Mapping)
            for value in (
                compact_report_record,
                compact_deployment_record,
                compact_population_record,
                compact_receipt_record,
            )
        )
        or compact_deployment_record.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or any(
            not isinstance(record.get("sha256"), str)
            or _SHA256.fullmatch(str(record.get("sha256"))) is None
            for record in (
                compact_report_record,
                compact_deployment_record,
                compact_population_record,
                compact_receipt_record,
            )
        )
    ):
        raise RuntimeError(
            "Expected durable compact HWA population, report, and deployment "
            "preflight artifacts."
        )
    _validate_compact_training_canary(
        compact_canary,
        mapping_report=compact_mapping,
        expected_population_fingerprint=fingerprint,
    )
    canary_mapping = canary.get("target_mapping_report")
    if not isinstance(canary_mapping, Mapping):
        raise RuntimeError("Expected the pulse canary's authoritative mapper report.")
    _validate_authoritative_mapping_report(
        canary_mapping,
        expected_population_fingerprint=fingerprint,
    )
    devices = canary.get("devices")
    budget_exhausted = canary.get("budget_exhausted")
    success_fraction = canary.get("success_fraction")
    deployment_record = artifacts.get("pulse_resolved_canary_deployment")
    if (
        canary.get("execution") != "pulse_resolved"
        or report.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or canary.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or canary.get("endpoint_seed") != SELECTION_ENDPOINT_SEED
        or canary.get("assignment_seed") != ASSIGNMENT_SEED
        or canary.get("population_fingerprint") != fingerprint
        or canary.get("device_model_sha256") != DEVICE_MODEL_SHA256
        or canary.get("nonfinite") != 0
        or isinstance(devices, bool)
        or not isinstance(devices, int)
        or devices <= 0
        or isinstance(budget_exhausted, bool)
        or not isinstance(budget_exhausted, int)
        or budget_exhausted < 0
        or budget_exhausted / devices > 0.01
        or isinstance(success_fraction, bool)
        or not isinstance(success_fraction, (int, float))
        or not math.isfinite(float(success_fraction))
        or float(success_fraction) < 0.99
        or not isinstance(deployment_record, Mapping)
        or deployment_record.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or not isinstance(deployment_record.get("sha256"), str)
        or _SHA256.fullmatch(str(deployment_record.get("sha256"))) is None
    ):
        raise RuntimeError(
            "Expected the exact pulse-resolved canary to pass its controller, "
            "success, budget, and deployment-artifact gates."
        )
    cohort = network_preflight.get("cohort")
    apparent_metrics = network_preflight.get("apparent_forward_metrics")
    network_record = artifacts.get("apparent_forward_network_metrics")
    if (
        network_preflight.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or network_preflight.get("forward_endpoint")
        != "aihwkit_apparent_endpoint"
        or network_preflight.get("persistent_endpoint_role")
        != "hidden_update_state"
        or network_preflight.get("programming_contexts") != 1
        or not isinstance(cohort, Mapping)
        or cohort
        != {
            "split": "validation",
            "data_seed": 17,
            "validation_points": 5000,
            "batch_size": 16,
            "maximum_batches": NETWORK_PREFLIGHT_BATCHES,
            "examples": NETWORK_PREFLIGHT_EXAMPLES,
            "shuffle": False,
        }
        or not isinstance(apparent_metrics, Mapping)
        or apparent_metrics.get("examples") != NETWORK_PREFLIGHT_EXAMPLES
        or not all(
            isinstance(apparent_metrics.get(key), (int, float))
            and not isinstance(apparent_metrics.get(key), bool)
            and math.isfinite(float(apparent_metrics[key]))
            for key in (
                "kl_teacher_student",
                "raw_kl_teacher_student",
                "student_accuracy",
                "teacher_accuracy",
                "teacher_agreement",
                "raw_score_rms",
                "calibrated_score_rms",
                "teacher_logit_rms",
                "fixed_logit_gain",
            )
        )
        or float(apparent_metrics["student_accuracy"])
        < MINIMUM_NETWORK_PREFLIGHT_STUDENT_ACCURACY
        or float(apparent_metrics["teacher_agreement"])
        < MINIMUM_NETWORK_PREFLIGHT_TEACHER_AGREEMENT
        or not isinstance(network_record, Mapping)
        or not isinstance(network_record.get("sha256"), str)
        or _SHA256.fullmatch(str(network_record.get("sha256"))) is None
    ):
        raise RuntimeError(
            "Expected the single-context apparent-forward network canary to "
            "pass its deterministic cohort and quality gates."
        )
    return fingerprint


def _validate_population_fingerprints(
    study_dir: Path,
    *,
    expected_population_fingerprint: str,
    expected_fallback_indices_sha256: str,
) -> dict[str, Any]:
    from training.ibm_reram_hwa import load_om_array_population

    records: dict[str, Any] = {}
    for arm, roles in POPULATION_ROLES.items():
        attempts = _native_attempts(study_dir / "runs" / arm)
        if len(attempts) != 1 or not attempts[0].is_dir():
            raise RuntimeError(f"Expected one immutable native run for {arm!r}.")
        run_dir = attempts[0]
        arm_records: dict[str, Any] = {}
        for role in roles:
            prefix = run_dir / "artifacts" / f"ibm_om_population.{role}"
            population_path = Path(f"{prefix}.npz")
            receipt_path = Path(f"{prefix}.receipt.json")
            receipt = _read_json(receipt_path)
            if receipt is None:
                raise RuntimeError(f"Expected {arm!r}/{role} population receipt.")
            population = load_om_array_population(population_path)
            if (
                population.fingerprint != expected_population_fingerprint
                or receipt.get("population_fingerprint")
                != expected_population_fingerprint
                or receipt.get("population_sha256") != sha256_file(population_path)
                or receipt.get("aihwkit_version") != "1.1.0"
                or receipt.get("request", {}).get("assignment_seed")
                != ASSIGNMENT_SEED
                or receipt.get("request", {}).get("corruption_policy")
                != CORRUPTION_POLICY
            ):
                raise RuntimeError(
                    f"Expected {arm!r}/{role} to reuse the preflight population."
                )
            arm_records[role] = {
                "population_fingerprint": population.fingerprint,
                "population_sha256": sha256_file(population_path),
                "receipt_sha256": sha256_file(receipt_path),
            }

        result = _read_json(run_dir / "result.json")
        metrics = result.get("metrics") if result else None
        selection = (
            metrics.get("selection_device_programming")
            if isinstance(metrics, Mapping)
            else None
        )
        if not isinstance(selection, Mapping):
            raise RuntimeError(f"Expected {arm!r} selection programming report.")
        mapping = selection.get("target_mapping_report")
        if not isinstance(mapping, Mapping):
            raise RuntimeError(f"Expected {arm!r} selection mapper report.")
        _validate_authoritative_mapping_report(
            mapping,
            expected_population_fingerprint=expected_population_fingerprint,
        )
        if (
            selection.get("population_fingerprint")
            != expected_population_fingerprint
            or selection.get("endpoint_application_policy")
            != ENDPOINT_APPLICATION_POLICY
            or selection.get("device_model_sha256") != DEVICE_MODEL_SHA256
            or selection.get("assignment_seed") != ASSIGNMENT_SEED
            or selection.get("corruption_policy") != CORRUPTION_POLICY
            or selection.get("target_mapping") != TARGET_MAPPING
        ):
            raise RuntimeError(f"Expected {arm!r} matched selection deployment.")
        training_audit: Mapping[str, Any] | None = None
        if arm == "train-exact-bounds-hwa":
            training = metrics.get("training_device_programming")
            if not isinstance(training, Mapping):
                raise RuntimeError(
                    "Expected the exact-bounds arm's compact HWA report."
                )
            training_mapping = training.get("target_mapping_report")
            if not isinstance(training_mapping, Mapping):
                raise RuntimeError(
                    "Expected the exact-bounds arm's training mapper report."
                )
            _validate_authoritative_mapping_report(
                training_mapping,
                expected_population_fingerprint=expected_population_fingerprint,
            )
            if (
                training.get("execution") != "compact_endpoint"
                or training.get("endpoint_application_policy")
                != ENDPOINT_APPLICATION_POLICY
                or training.get("endpoint_seed") != TRAINING_ENDPOINT_SEED
                or training.get("assignment_seed") != ASSIGNMENT_SEED
                or training.get("population_fingerprint")
                != expected_population_fingerprint
                or training.get("device_model_sha256") != DEVICE_MODEL_SHA256
                or training.get("corruption_policy") != CORRUPTION_POLICY
                or training.get("target_mapping") != TARGET_MAPPING
            ):
                raise RuntimeError(
                    "Expected the exact-bounds compact HWA report to match "
                    "the preflight population and mapper."
                )
            fallback_indices_sha256 = _validate_compact_training_canary(
                training,
                mapping_report=training_mapping,
                expected_population_fingerprint=(
                    expected_population_fingerprint
                ),
                expected_fallback_indices_sha256=(
                    expected_fallback_indices_sha256
                ),
            )
            fallback = training["pulse_resolved_fallback"]
            training_audit = {
                "population_fingerprint": training.get(
                    "population_fingerprint"
                ),
                "endpoint_seed": training.get("endpoint_seed"),
                "target_mapping": training.get("target_mapping"),
                "execution_detail": training.get("execution_detail"),
                "compact_endpoint_devices": training.get(
                    "compact_endpoint_devices"
                ),
                "pulse_resolved_fallback_devices": training.get(
                    "pulse_resolved_fallback_devices"
                ),
                "fallback_policy": fallback.get("policy"),
                "fallback_indices_sha256": fallback_indices_sha256,
                "fallback_accepted": fallback.get("accepted"),
                "fallback_budget_exhausted": fallback.get(
                    "budget_exhausted"
                ),
                "fallback_nonfinite": fallback.get("nonfinite"),
                "fallback_pulse_count": fallback.get("pulse_count"),
                "fallback_verify_count_mean": fallback.get(
                    "verify_count_mean"
                ),
                "fallback_reversal_count_mean": fallback.get(
                    "reversal_count_mean"
                ),
            }
        records[arm] = {
            "run_dir": str(run_dir),
            "populations": arm_records,
            "selection_report_population_fingerprint": selection.get(
                "population_fingerprint"
            ),
            "training_report": training_audit,
        }
    return records


def _validate_completed_summary(summary: Mapping[str, Any]) -> None:
    if (
        summary.get("study_id") != STUDY_ID
        or summary.get("state") != "ready_for_review"
        or summary.get("ready_for_review") is not True
        or summary.get("validation_mode") != "full_artifact_hashes"
    ):
        raise RuntimeError(
            "Expected the corrected pilot to be artifact-verified and ready for review."
        )
    raw_arms = summary.get("arms")
    if not isinstance(raw_arms, list):
        raise RuntimeError("Expected the study summary to contain arm records.")
    arms = {
        str(arm.get("arm_id")): arm
        for arm in raw_arms
        if isinstance(arm, Mapping)
    }
    if set(arms) != set(TRAINING_ARM_CONFIGS) or len(raw_arms) != len(arms):
        raise RuntimeError("Expected the summary to cover both pilot arms.")
    for arm_id, arm in arms.items():
        if not (
            arm.get("expected_runs") == 1
            and arm.get("complete") == 1
            and arm.get("running") == 0
            and arm.get("failed") == 0
            and arm.get("invalid") == 0
            and arm.get("coverage_complete") is True
        ):
            raise RuntimeError(
                f"Expected arm {arm_id!r} to have one valid complete run."
            )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _refresh_if_started(study_dir: Path) -> None:
    for arm in TRAINING_ARM_CONFIGS:
        if _latest_native_run(study_dir / "runs" / arm) is not None:
            refresh_current_simulations_for_run(
                repo_root=_ROOT,
                run_dir=study_dir / "runs" / arm,
            )
            return


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_dir = (
        args.study_dir or _ROOT / "results" / STUDY_ID
    ).expanduser().resolve()
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
    base_contract = {
        "schema": "ebl.mnist_ibm_om_common_window.launch_contract",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "formal_evidence": True,
        "fresh_attempt_only": True,
        "operational_retry": {
            "retry_of_study_id": (
                "mnist-ibm-om-common-window-hwa-program-verify-pilot-"
                "20260824-v1"
            ),
            "v1_failure_class": "checkpoint_metadata_serialization",
            "v1_training_updates": 0,
            "v2_model_contract_delta": (
                "eight_empty_quad_out_of_bound_cells_use_exact_"
                "pulse_resolved_fallback"
            ),
            "v2_prepared_before_delta": False,
            "resume_v1_attempts": False,
        },
        "study_json": {
            "path": str(study_dir / "study.json"),
            "sha256": sha256_file(study_dir / "study.json"),
            "source_plan_sha256": study["source_plan"]["sha256"],
        },
        "launcher": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "python": str(launcher_python),
        },
        "aihwkit_python": str(aihwkit_python),
        "cuda_visible_devices": args.cuda_visible_devices,
        "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
        "preflight": {
            "required_before_native_launch": True,
            "authoritative_entry_point": (
                "training.ibm_reram_hwa."
                "IbmReramHwaParameterModifier.preflight_target_mapping"
            ),
            "maximum_empty_quads": MAXIMUM_EMPTY_QUADS,
            "maximum_empty_quad_fraction": MAXIMUM_EMPTY_QUAD_FRACTION,
            "zero_outside_nonempty_quads": True,
            "zero_corrupt_quads": True,
            "finite_targets_inside_0_1": True,
            "pulse_resolved_canary": {
                "endpoint_seed": SELECTION_ENDPOINT_SEED,
                "minimum_success_fraction": 0.99,
                "maximum_budget_exhausted_fraction": 0.01,
                "maximum_nonfinite": 0,
                "persistent_deployment_bundle_required": True,
            },
            "compact_training_canary": {
                "training_contexts": 1,
                "endpoint_seed": TRAINING_ENDPOINT_SEED,
                "execution_detail": (
                    "compact_endpoint_with_exact_empty_quad_fallback"
                ),
                "compact_endpoint_devices": EXPECTED_COMPACT_ENDPOINT_DEVICES,
                "pulse_resolved_fallback_devices": (
                    EXPECTED_EMPTY_QUAD_FALLBACK_DEVICES
                ),
                "fallback_policy": COMPACT_FALLBACK_POLICY,
                "maximum_fallback_nonfinite": 0,
                "maximum_fallback_program_pulses": 128,
                "same_population_fingerprint_and_mapper": True,
                "durable_report_and_deployment_required": True,
            },
            "apparent_forward_network_canary": {
                "programming_contexts": 1,
                "split": "validation",
                "data_seed": 17,
                "examples": NETWORK_PREFLIGHT_EXAMPLES,
                "batch_size": 16,
                "maximum_batches": NETWORK_PREFLIGHT_BATCHES,
                "minimum_student_accuracy": (
                    MINIMUM_NETWORK_PREFLIGHT_STUDENT_ACCURACY
                ),
                "minimum_teacher_agreement": (
                    MINIMUM_NETWORK_PREFLIGHT_TEACHER_AGREEMENT
                ),
            },
        },
        "inputs": {
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
        },
        "configs": {
            arm: {
                "path": str((CONFIG_ROOT / filename).resolve()),
                "sha256": sha256_file(CONFIG_ROOT / filename),
            }
            for arm, filename in TRAINING_ARM_CONFIGS.items()
        },
        "commands": commands,
        "expected_epochs_per_arm": EXPECTED_EPOCHS_PER_ARM,
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "created_at": _utc_now(),
    }
    atomic_write_json(attempt_dir / "contract.json", base_contract)
    atomic_write_json(
        attempt_dir / "launcher_started.json",
        {
            "status": "preflighting",
            "started_at": _utc_now(),
            "expected_arms": list(TRAINING_ARM_CONFIGS),
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
    started = time.monotonic()

    def request_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
        nonlocal stop_requested
        stop_requested = True
        for process in processes.values():
            _terminate_process(process)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        preflight = _run_authoritative_preflight(
            attempt_dir=attempt_dir,
            aihwkit_python=aihwkit_python,
        )
        try:
            population_fingerprint = _validate_preflight_report(preflight)
        except (RuntimeError, ValueError) as error:
            preflight["gate"] = {
                "passed": False,
                "validated_at": _utc_now(),
                "maximum_empty_quads": MAXIMUM_EMPTY_QUADS,
                "maximum_empty_quad_fraction": MAXIMUM_EMPTY_QUAD_FRACTION,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            atomic_write_json(attempt_dir / "preflight_report.json", preflight)
            raise
        compact_fallback_indices_sha256 = str(
            preflight["compact_training_canary_report"][
                "pulse_resolved_fallback"
            ]["selection_indices_sha256"]
        )
        preflight["gate"] = {
            "passed": True,
            "validated_at": _utc_now(),
            "maximum_empty_quads": MAXIMUM_EMPTY_QUADS,
            "maximum_empty_quad_fraction": MAXIMUM_EMPTY_QUAD_FRACTION,
            "compact_fallback_indices_sha256": (
                compact_fallback_indices_sha256
            ),
        }
        atomic_write_json(attempt_dir / "preflight_report.json", preflight)
        atomic_write_json(
            attempt_dir / "preflight_result.json",
            {
                "status": "passed",
                "population_fingerprint": population_fingerprint,
                "compact_fallback_indices_sha256": (
                    compact_fallback_indices_sha256
                ),
                "finished_at": _utc_now(),
            },
        )
        if stop_requested:
            raise RuntimeError(
                "Launcher stop was requested during preflight; native arms "
                "were not started."
            )
        atomic_write_json(
            attempt_dir / "execution_contract.json",
            {
                **base_contract,
                "schema": "ebl.mnist_ibm_om_common_window.execution_contract",
                "preflight_report_sha256": sha256_file(
                    attempt_dir / "preflight_report.json"
                ),
                "expected_population_fingerprint": population_fingerprint,
                "expected_compact_fallback_indices_sha256": (
                    compact_fallback_indices_sha256
                ),
                "created_at": _utc_now(),
            },
        )
        atomic_write_json(
            attempt_dir / "heartbeat.json",
            {
                "updated_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "phase": "preflight_passed",
                "population_fingerprint": population_fingerprint,
                "compact_fallback_indices_sha256": (
                    compact_fallback_indices_sha256
                ),
            },
        )

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
                    "expected_population_fingerprint": population_fingerprint,
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
                    "phase": "training",
                    "stop_requested": stop_requested,
                    "expected_population_fingerprint": population_fingerprint,
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
            _refresh_if_started(study_dir)
            time.sleep(HEARTBEAT_SECONDS)

        returncodes = {arm: process.wait() for arm, process in processes.items()}
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
        population_audit: Mapping[str, Any] | None = None
        try:
            if summary_process.returncode != 0 or summary is None:
                raise RuntimeError("Study summarization did not produce a summary.")
            _validate_completed_summary(summary)
            population_audit = _validate_population_fingerprints(
                study_dir,
                expected_population_fingerprint=population_fingerprint,
                expected_fallback_indices_sha256=(
                    compact_fallback_indices_sha256
                ),
            )
            atomic_write_json(
                attempt_dir / "population_fingerprint_audit.json",
                {
                    "status": "passed",
                    "expected_population_fingerprint": population_fingerprint,
                    "expected_compact_fallback_indices_sha256": (
                        compact_fallback_indices_sha256
                    ),
                    "arms": population_audit,
                    "validated_at": _utc_now(),
                },
            )
        except (RuntimeError, ValueError) as error:
            phase_error = str(error)

        success = (
            not stop_requested
            and all(value == 0 for value in returncodes.values())
            and summary_process.returncode == 0
            and phase_error is None
        )
        atomic_write_json(
            attempt_dir / "heartbeat.json",
            {
                "updated_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "phase": "complete" if success else "failed",
                "stop_requested": stop_requested,
                "expected_population_fingerprint": population_fingerprint,
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
        atomic_write_json(
            attempt_dir / "launcher_result.json",
            {
                "status": "complete" if success else "failed",
                "finished_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "returncodes": returncodes,
                "study_summarize_returncode": summary_process.returncode,
                "study_state": summary.get("state") if summary else None,
                "population_fingerprint": population_fingerprint,
                "population_fingerprint_audit_passed": population_audit is not None,
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
        _refresh_if_started(study_dir)
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
        preflight_result_path = attempt_dir / "preflight_result.json"
        if not preflight_result_path.exists():
            atomic_write_json(
                preflight_result_path,
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
            )
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
        _refresh_if_started(study_dir)
        raise
    finally:
        for log in stdout_logs.values():
            log.close()
        for log in stderr_logs.values():
            log.close()


if __name__ == "__main__":  # pragma: no cover - exercised as a launcher
    raise SystemExit(main())
