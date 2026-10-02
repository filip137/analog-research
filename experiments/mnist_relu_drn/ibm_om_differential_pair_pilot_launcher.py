"""Fail-closed launcher for the canonical eight-device IBM OM pilot.

The launcher first canonicalizes one already-trained ideal-differential base
checkpoint without changing any tensor, then exercises the production pair
mapper and both endpoint backends.  Native arms start only after every frozen
identity, mapping, serialization, controller, and network gate passes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
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
    "mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1"
)
STUDY_PLAN = _ROOT / "studies" / f"{STUDY_ID}.json"
CONFIG_ROOT = (
    _ROOT
    / "examples"
    / "mnist_relu_drn"
    / "ibm_om_differential_pair_hwa_pilot"
)

TRAINING_ARM_CONFIGS = {
    "train-clean-differential-pair": "clean.json",
    "train-exact-bounds-pair-hwa": "exact_bounds_hwa.json",
}
POPULATION_ROLES = {
    "train-clean-differential-pair": ("selection",),
    "train-exact-bounds-pair-hwa": ("training", "selection"),
}

SOURCE_BASE_CHECKPOINT = (
    _ROOT
    / "results"
    / "mnist-relu-drn-kd-exploratory-20260816"
    / "ideal_differential_finetune"
    / "20260816T135214.689196Z-9ef10aa2-642ca991"
    / "checkpoints"
    / "weights.pt"
)
SOURCE_BASE_CHECKPOINT_SHA256 = (
    "5a9dece30a938f00d2022de3e0aa57755e17f754c2df249507dd709bf9394fbc"
)
RELU_TEACHER_CHECKPOINT = (
    _ROOT
    / "results"
    / "mnist-relu-drn-kd-exploratory-20260816"
    / "teacher_fixed_init"
    / "20260816T132557.806720Z-fbff3c26-6f6867f1"
    / "checkpoints"
    / "weights.pt"
)
RELU_TEACHER_CHECKPOINT_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
DEVICE_MODEL = _ROOT / "data" / "ibm_reram_om_pv128_hwa_v1.json"
DEVICE_MODEL_SHA256 = (
    "3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3"
)

ASSIGNMENT_SEED = 84001
TRAINING_ENDPOINT_SEED = 84002
SELECTION_ENDPOINT_SEED = 84003
CORRUPTION_POLICY = "counterfactual_repaired"
TARGET_MAPPING = "differential_pair_common_window"
COMMON_WINDOW_MARGIN_FRACTION = 0.25
EXPECTED_BINDING_KEYS = (
    "base.conductance_plus.0",
    "base.conductance_minus.0",
    "base.conductance_plus.1",
    "base.conductance_minus.1",
)
EXPECTED_POPULATION_FINGERPRINT = (
    "d0dfae135fa7b741c46f1178e52f645da4593f7313b4e6a0a71a9b3793d1b2f0"
)
EXPECTED_TOTAL_DEVICES = 317600
EXPECTED_PAIR_COUNT = 158800
EXPECTED_EMPTY_PAIR_COUNT = 8
MAXIMUM_EMPTY_PAIR_COUNT = 20
MAXIMUM_EMPTY_PAIR_FRACTION = 0.0005
EXPECTED_COMPACT_ENDPOINT_DEVICES = 317584
EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES = 16
ENDPOINT_APPLICATION_POLICY = (
    "aihwkit_apparent_forward_persistent_update_state"
)
COMPACT_EXECUTION_DETAIL = "compact_endpoint_with_exact_empty_pair_fallback"
COMPACT_ENDPOINT_GENERATION_POLICY = (
    "compact_covered_exact_out_of_bound_empty_pair_fallback"
)
COMPACT_FALLBACK_POLICY = (
    "pulse_resolved_noncorrupt_out_of_bound_empty_pair_only"
)

NETWORK_PREFLIGHT_EXAMPLES = 1600
NETWORK_PREFLIGHT_BATCHES = 100
MINIMUM_NETWORK_PREFLIGHT_STUDENT_ACCURACY = 0.25
MINIMUM_NETWORK_PREFLIGHT_TEACHER_AGREEMENT = 0.25
CANONICAL_REPLAY_EXAMPLES = 5000
MINIMUM_CANONICAL_REPLAY_STUDENT_ACCURACY = 0.97
MINIMUM_CANONICAL_REPLAY_TEACHER_AGREEMENT = 0.99
MAXIMUM_CANONICAL_REPLAY_KL = 0.001
EXPECTED_EPOCHS_PER_ARM = 10
HEARTBEAT_SECONDS = 15.0

_CUDA_VISIBLE_DEVICES = re.compile(r"[0-9]+(?:,[0-9]+)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value!r} is not permitted.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight and concurrently launch the two predeclared IBM OM "
            "differential-pair pilot arms."
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
            "executable for the pinned AIHWKit 1.1.0 environment; used only "
            "by the external fixed-population sampler"
        ),
    )
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeError, ValueError):
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
    status = _read_json(run_dir / "status.json") if run_dir else None
    metrics = run_dir / "metrics.jsonl" if run_dir else None
    weights = run_dir / "checkpoints" / "weights.pt" if run_dir else None
    resume = run_dir / "checkpoints" / "resume.pt" if run_dir else None
    return {
        "arm_id": arm,
        "pid": process.pid,
        "launcher_returncode": process.poll(),
        "native_run_dir": str(run_dir) if run_dir else None,
        "native_status": status.get("status") if status else None,
        "metrics_size_bytes": (
            metrics.stat().st_size if metrics and metrics.exists() else 0
        ),
        "metrics_lines": _line_count(metrics) if metrics else 0,
        "weights_size_bytes": (
            weights.stat().st_size if weights and weights.exists() else 0
        ),
        "resume_size_bytes": (
            resume.stat().st_size if resume and resume.exists() else 0
        ),
        "stdout_size_bytes": stdout_path.stat().st_size,
        "stderr_size_bytes": stderr_path.stat().st_size,
    }


def _validate_input_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> None:
    if not path.is_file():
        raise RuntimeError(
            f"Expected the frozen {label} to be a file. Provided value: "
            f"{str(path)!r}."
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"Expected the frozen {label} SHA-256 to be {expected_sha256!r}. "
            f"Provided value: {observed!r} at {str(path)!r}."
        )


def _backfilled_parameters(value: Any) -> Any:
    """Complete stored modifier parameters that predate the layer fields."""

    return backfill_config_dict(value) if isinstance(value, Mapping) else value


def _expected_modifier_parameters(*, execution: str) -> dict[str, Any]:
    if execution not in {"compact_endpoint", "pulse_resolved"}:
        raise ValueError(f"Unexpected execution {execution!r}.")
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
            "dual_rail_layout_by_parameter": None,
            "common_window_margin_fraction": COMMON_WINDOW_MARGIN_FRACTION,
        }
    )


def _validate_config_contracts() -> None:
    specs: dict[str, Any] = {}
    for arm, filename in TRAINING_ARM_CONFIGS.items():
        definition, spec = resolve_experiment_config(
            CONFIG_ROOT / filename,
            "train",
        )
        if definition.experiment_id != "mnist_relu_drn_kd.v1":
            raise RuntimeError(f"Expected arm {arm!r} to use the MNIST DRN runtime.")
        specs[arm] = spec

    clean = specs["train-clean-differential-pair"]
    exact = specs["train-exact-bounds-pair-hwa"]
    for field in ("runtime", "data", "teacher", "model", "solver", "mapping"):
        if getattr(clean, field) != getattr(exact, field):
            raise RuntimeError(f"Expected matched pilot field {field!r} to be equal.")
    if (
        clean.runtime.seed != 42
        or clean.runtime.data_seed != 42
        or clean.model.encoding != "differential"
        or clean.teacher.type != "bias_free_relu"
        or clean.teacher.initialization != "signed_weight_mapping"
    ):
        raise RuntimeError(
            "Expected the canonical differential/ReLU seed-42 architecture."
        )
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
        raise RuntimeError("Expected both differential-pair arms to run ten epochs.")
    if clean.settings.weight_modifier.type != "none":
        raise RuntimeError("Expected the clean arm to train without HWA.")
    if (
        exact.settings.weight_modifier.type != "ibm_reram_om_program_verify"
        or dict(exact.settings.weight_modifier.parameters)
        != _expected_modifier_parameters(execution="compact_endpoint")
    ):
        raise RuntimeError(
            "Expected the pair-HWA arm to use the declared compact IBM OM modifier."
        )
    expected_selection = _expected_modifier_parameters(
        execution="pulse_resolved"
    )
    for arm, spec in specs.items():
        modifier = spec.settings.selection_weight_modifier
        if (
            modifier.type != "ibm_reram_om_program_verify"
            or dict(modifier.parameters) != expected_selection
        ):
            raise RuntimeError(
                f"Expected arm {arm!r} to use the matched repaired deployment."
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
        raise RuntimeError(f"Expected prepared arm {arm_id!r} to be MNIST train.")
    configs = arm.get("configs")
    if not isinstance(configs, list) or len(configs) != 1:
        raise RuntimeError(f"Expected arm {arm_id!r} to declare one config.")
    record = configs[0]
    if not isinstance(record, Mapping):
        raise RuntimeError(f"Expected arm {arm_id!r} config record.")
    resolved = record.get("resolved_path")
    if not isinstance(resolved, str) or Path(resolved).resolve() != config_path:
        raise RuntimeError(f"Expected arm {arm_id!r} config {str(config_path)!r}.")
    if record.get("sha256") != sha256_file(config_path):
        raise RuntimeError(f"Expected arm {arm_id!r} prepared config digest.")


def _validate_prepared_study(
    study_dir: Path,
    *,
    source_base_path: Path = SOURCE_BASE_CHECKPOINT,
    source_base_sha256: str = SOURCE_BASE_CHECKPOINT_SHA256,
    teacher_path: Path = RELU_TEACHER_CHECKPOINT,
    teacher_sha256: str = RELU_TEACHER_CHECKPOINT_SHA256,
    device_model_path: Path = DEVICE_MODEL,
    device_model_sha256: str = DEVICE_MODEL_SHA256,
) -> Mapping[str, Any]:
    study_dir = study_dir.expanduser().resolve()
    study = load_study_record(study_dir)
    if study.get("study_id") != STUDY_ID:
        raise RuntimeError(f"Expected prepared study {STUDY_ID!r}.")
    source_plan = study.get("source_plan")
    if (
        not isinstance(source_plan, Mapping)
        or Path(str(source_plan.get("path"))).resolve() != STUDY_PLAN.resolve()
        or source_plan.get("sha256") != sha256_file(STUDY_PLAN)
    ):
        raise RuntimeError("Expected the prepared study to match the tracked plan.")
    raw_arms = study.get("arms")
    if not isinstance(raw_arms, list):
        raise RuntimeError("Expected prepared arm records.")
    arms = {
        str(arm.get("arm_id")): arm
        for arm in raw_arms
        if isinstance(arm, Mapping)
    }
    if set(arms) != set(TRAINING_ARM_CONFIGS) or len(arms) != len(raw_arms):
        raise RuntimeError("Expected exactly the two differential-pair arms.")
    for arm, filename in TRAINING_ARM_CONFIGS.items():
        _validate_arm_record(
            arms[arm],
            arm_id=arm,
            config_path=(CONFIG_ROOT / filename).resolve(),
        )
    runs_root = study_dir / "runs"
    observed = (
        {path.name for path in runs_root.iterdir() if path.is_dir()}
        if runs_root.is_dir()
        else set()
    )
    if observed != set(TRAINING_ARM_CONFIGS):
        raise RuntimeError("Expected exactly one prepared root for each arm.")
    existing = {
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
    if existing or prior_launches:
        raise RuntimeError(
            "Expected a fresh prepared pilot with no native or launcher "
            f"attempts. Provided value: native={existing!r}, "
            f"launch={prior_launches!r}."
        )
    _validate_config_contracts()
    _validate_input_file(
        source_base_path.resolve(),
        expected_sha256=source_base_sha256,
        label="clean ideal-differential source checkpoint",
    )
    _validate_input_file(
        teacher_path.resolve(),
        expected_sha256=teacher_sha256,
        label="bias-free ReLU teacher checkpoint",
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
    canonical_checkpoint_path: Path,
    teacher_path: Path = RELU_TEACHER_CHECKPOINT,
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
            "--weights",
            str(canonical_checkpoint_path.resolve()),
            "--teacher-weights",
            str(teacher_path.resolve()),
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
    """Evaluate once inside the sole already-programmed pulse context."""

    with modifier.evaluation_context():
        metrics = evaluator(
            stack,
            teacher,
            validation_loader,
            maximum_batches=NETWORK_PREFLIGHT_BATCHES,
        )
    if not isinstance(metrics, Mapping) or metrics.get("examples") != (
        NETWORK_PREFLIGHT_EXAMPLES
    ):
        raise RuntimeError(
            "Expected exactly 1,600 examples in the apparent-forward canary."
        )
    return metrics


def _artifact_record(path: Path, *, kind: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        **extra,
    }


def _torch_payload_equal(left: Any, right: Any) -> bool:
    """Strict recursive equality for round-tripped tensor artifacts."""

    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and tuple(left) == tuple(right)
            and all(_torch_payload_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _torch_payload_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return type(left) is type(right) and left == right


def _load_torch_weights_only(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def _validate_serialized_canary_artifacts(
    *,
    compact_report: Mapping[str, Any],
    compact_report_path: Path,
    compact_deployment: Mapping[str, Any],
    compact_deployment_path: Path,
    pulse_report: Mapping[str, Any],
    pulse_report_path: Path,
    pulse_deployment: Mapping[str, Any],
    pulse_deployment_path: Path,
    network_report: Mapping[str, Any],
    network_path: Path,
) -> dict[str, Any]:
    """Read back strict JSON/torch boundaries and audit the hybrid masks."""

    import torch

    json_records = (
        (compact_report_path, compact_report),
        (pulse_report_path, pulse_report),
        (network_path, network_report),
    )
    for path, expected in json_records:
        if _read_json(path) != expected:
            raise RuntimeError(
                f"Expected exact strict JSON round trip for {path.name!r}."
            )
    loaded_compact = _load_torch_weights_only(compact_deployment_path)
    loaded_pulse = _load_torch_weights_only(pulse_deployment_path)
    if not _torch_payload_equal(loaded_compact, compact_deployment):
        raise RuntimeError("Expected exact compact deployment torch round trip.")
    if not _torch_payload_equal(loaded_pulse, pulse_deployment):
        raise RuntimeError("Expected exact pulse deployment torch round trip.")

    compact_mask = loaded_compact.get("compact_endpoint_mask")
    fallback_mask = loaded_compact.get("pulse_resolved_fallback_mask")
    fallback = loaded_compact.get("pulse_resolved_fallback")
    if (
        not isinstance(compact_mask, torch.Tensor)
        or compact_mask.dtype != torch.bool
        or tuple(compact_mask.shape) != (EXPECTED_TOTAL_DEVICES,)
        or not isinstance(fallback_mask, torch.Tensor)
        or fallback_mask.dtype != torch.bool
        or tuple(fallback_mask.shape) != (EXPECTED_TOTAL_DEVICES,)
        or bool(torch.any(compact_mask & fallback_mask))
        or not bool(torch.all(compact_mask | fallback_mask))
        or int(compact_mask.sum().item()) != EXPECTED_COMPACT_ENDPOINT_DEVICES
        or int(fallback_mask.sum().item())
        != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or not isinstance(fallback, Mapping)
    ):
        raise RuntimeError(
            "Expected compact and exact-fallback masks to partition all cells."
        )
    fallback_indices = torch.where(fallback_mask)[0].to(dtype=torch.int64)
    reported_indices = compact_report.get("pulse_resolved_fallback", {}).get(
        "selection_indices"
    )
    stored_indices = fallback.get("selection_indices")
    if (
        not isinstance(reported_indices, list)
        or fallback_indices.tolist() != reported_indices
        or not isinstance(stored_indices, torch.Tensor)
        or stored_indices.dtype != torch.int64
        or not torch.equal(fallback_indices, stored_indices)
        or fallback.get("policy") != COMPACT_FALLBACK_POLICY
        or fallback.get("selection_indices_sha256")
        != compact_report.get("pulse_resolved_fallback", {}).get(
            "selection_indices_sha256"
        )
    ):
        raise RuntimeError(
            "Expected serialized fallback indices, digest, mask, and report parity."
        )
    for deployment, label in (
        (loaded_compact, "compact"),
        (loaded_pulse, "pulse"),
    ):
        for key in ("apparent_endpoint", "persistent_endpoint"):
            value = deployment.get(key)
            if (
                not isinstance(value, torch.Tensor)
                or value.numel() != EXPECTED_TOTAL_DEVICES
                or not bool(torch.all(torch.isfinite(value)))
            ):
                raise RuntimeError(
                    f"Expected finite {label} deployment {key!r}."
                )
    return {
        "json_round_trip": "passed",
        "checkpoint_round_trip": "passed",
        "allow_nan": False,
        "pickle_unsafe_custom_objects": False,
        "compact_and_fallback_mask_partition": "passed",
        "fallback_indices_mask_report_parity": "passed",
        "finite_apparent_and_persistent_endpoints": "passed",
    }


def _canonicalize_base_checkpoint(
    *,
    preflight_dir: Path,
    spec,
    teacher,
    validation_loader,
    source_base_path: Path,
    teacher_path: Path,
) -> tuple[Any, Path, dict[str, Any]]:
    """Copy the frozen differential tensors into current metadata exactly."""

    import torch

    from experiments.mnist_relu_drn.components import build_student_stack
    from experiments.mnist_relu_drn.runtime import (
        _amplification_index_report,
        _evaluate,
        _model_checkpoint_metadata,
        _validate_checkpoint_metadata,
    )
    from training.checkpoint import load_named_weights, save_named_weights

    source_stack = build_student_stack(spec, enable_measured=False)
    loaded_source = load_named_weights(
        source_base_path,
        source_stack.bundle.catalog,
    )
    source_metadata = dict(loaded_source.metadata)
    if (
        source_metadata.get("encoding") != "differential"
        or source_metadata.get("experiment_id") != "mnist_relu_drn_kd.v1"
        or source_metadata.get("teacher_sha256")
        != RELU_TEACHER_CHECKPOINT_SHA256
        or source_metadata.get("selection_epoch") != 19
    ):
        raise RuntimeError(
            "Expected the frozen source to be the selected epoch-19 clean "
            "ideal-differential checkpoint under the frozen ReLU teacher."
        )
    fixed_gain = source_metadata.get("fixed_logit_gain")
    if (
        isinstance(fixed_gain, bool)
        or not isinstance(fixed_gain, (int, float))
        or not math.isfinite(float(fixed_gain))
        or float(fixed_gain) <= 0.0
    ):
        raise RuntimeError("Expected a finite positive source fixed logit gain.")
    source_stack.cost.gain = float(fixed_gain)
    source_tensors = {
        binding.key: binding.state.detach().cpu().clone()
        for binding in source_stack.bundle.catalog.checkpointed
    }
    if tuple(source_tensors) != EXPECTED_BINDING_KEYS:
        raise RuntimeError(
            "Expected the canonical plus/minus differential catalog order."
        )
    source_replay = _evaluate(
        source_stack,
        teacher,
        validation_loader,
        maximum_batches=None,
    )

    metadata = _model_checkpoint_metadata(
        source_stack,
        spec=spec,
        teacher_sha256=sha256_file(teacher_path),
        mapping=source_metadata.get("mapping"),
        deployment_source=None,
        device_model_sha256=None,
    )
    # The canonical object is a neutral pre-intervention base.  Keep the
    # source selection provenance without claiming that either formal arm's
    # physical modifier created these tensors.
    metadata.update(
        {
            "selection_evaluation": "clean",
            "selection_noise_repeats": 1,
            "weight_modifier": {"type": "none", "parameters": {}},
            "source_selection": {
                key: source_metadata.get(key)
                for key in (
                    "selection_epoch",
                    "selection_metric",
                    "selection_value",
                    "selection_student_accuracy",
                    "selection_teacher_agreement",
                )
            },
            "canonicalization": {
                "operation": "metadata_only_tensor_identical_copy",
                "source_path": str(source_base_path.resolve()),
                "source_sha256": sha256_file(source_base_path),
                "teacher_path": str(teacher_path.resolve()),
                "teacher_sha256": sha256_file(teacher_path),
                "teacher_mapping_invoked": False,
                "conductance_remapping_invoked": False,
                "optimizer_updates": 0,
                "retraining_epochs": 0,
            },
        }
    )
    metadata.pop("selection_weight_modifier", None)
    canonical_path = preflight_dir / "canonical_base_weights.pt"
    save_named_weights(
        canonical_path,
        source_stack.bundle.catalog,
        metadata=metadata,
    )

    canonical_stack = build_student_stack(spec, enable_measured=False)
    loaded_canonical = load_named_weights(
        canonical_path,
        canonical_stack.bundle.catalog,
    )
    _validate_checkpoint_metadata(
        loaded_canonical.metadata,
        spec=spec,
        teacher_sha256=sha256_file(teacher_path),
        expected_amplification_indices=(
            _amplification_index_report(canonical_stack)
        ),
    )
    canonical_stack.cost.gain = float(loaded_canonical.metadata["fixed_logit_gain"])
    canonical_tensors = {
        binding.key: binding.state.detach().cpu().clone()
        for binding in canonical_stack.bundle.catalog.checkpointed
    }
    unequal = [
        key
        for key in EXPECTED_BINDING_KEYS
        if not torch.equal(source_tensors[key], canonical_tensors[key])
    ]
    if unequal:
        raise RuntimeError(
            "Expected metadata-only canonicalization to preserve every "
            f"tensor exactly. Provided unequal keys: {unequal!r}."
        )
    canonical_replay = _evaluate(
        canonical_stack,
        teacher,
        validation_loader,
        maximum_batches=None,
    )
    if canonical_replay != source_replay:
        raise RuntimeError(
            "Expected exact source-versus-canonical ordered validation replay."
        )
    if (
        source_replay.get("examples") != CANONICAL_REPLAY_EXAMPLES
        or float(source_replay.get("student_accuracy", float("nan")))
        < MINIMUM_CANONICAL_REPLAY_STUDENT_ACCURACY
        or float(source_replay.get("teacher_agreement", float("nan")))
        < MINIMUM_CANONICAL_REPLAY_TEACHER_AGREEMENT
        or float(source_replay.get("kl_teacher_student", float("inf")))
        > MAXIMUM_CANONICAL_REPLAY_KL
    ):
        raise RuntimeError(
            "Expected the canonical base to retain the predeclared clean "
            "validation quality."
        )

    metadata_path = preflight_dir / "canonical_base_metadata.json"
    atomic_write_json(metadata_path, loaded_canonical.metadata)
    replay_path = preflight_dir / "canonical_base_replay.json"
    replay_report = {
        "schema": "ebl.mnist_ibm_om_differential_pair.canonical_replay",
        "schema_version": 1,
        "operation": "metadata_only_tensor_identical_copy",
        "source_checkpoint": {
            "path": str(source_base_path.resolve()),
            "sha256": sha256_file(source_base_path),
        },
        "canonical_checkpoint": {
            "path": str(canonical_path.resolve()),
            "sha256": sha256_file(canonical_path),
        },
        "teacher_checkpoint": {
            "path": str(teacher_path.resolve()),
            "sha256": sha256_file(teacher_path),
        },
        "catalog_order": list(EXPECTED_BINDING_KEYS),
        "tensor_equality": "torch.equal_all_named_tensors",
        "unequal_tensor_keys": [],
        "current_metadata_validation": "passed",
        "source_metrics": source_replay,
        "canonical_metrics": canonical_replay,
        "metrics_exactly_equal": True,
        "teacher_mapping_invoked": False,
        "conductance_remapping_invoked": False,
        "optimizer_updates": 0,
        "retraining_epochs": 0,
    }
    atomic_write_json(replay_path, replay_report)
    # Read both strict boundaries back before this checkpoint is eligible for
    # use as a native input.
    if _read_json(metadata_path) != loaded_canonical.metadata:
        raise RuntimeError("Expected strict canonical metadata JSON round trip.")
    if _read_json(replay_path) != replay_report:
        raise RuntimeError("Expected strict canonical replay JSON round trip.")
    return canonical_stack, canonical_path, {
        **replay_report,
        "artifacts": {
            "canonical_checkpoint": _artifact_record(
                canonical_path,
                kind="canonical_tensor_identical_named_weights",
            ),
            "canonical_metadata": _artifact_record(
                metadata_path,
                kind="strict_checkpoint_metadata_json",
            ),
            "canonical_replay": _artifact_record(
                replay_path,
                kind="exact_validation_replay_json",
            ),
        },
    }


def _run_authoritative_preflight(
    *,
    attempt_dir: Path,
    aihwkit_python: Path,
    source_base_path: Path = SOURCE_BASE_CHECKPOINT,
    teacher_path: Path = RELU_TEACHER_CHECKPOINT,
    device_model_path: Path = DEVICE_MODEL,
) -> dict[str, Any]:
    """Run canonicalization and the production pair-programming paths."""

    import torch

    from experiments.mnist_relu_drn.runtime import _evaluate, _load_teacher
    from experiments.mnist_shared import build_mnist_loaders
    from experiments.schema import to_plain_data
    from training.checkpoint import atomic_torch_save, load_named_weights
    from training.ibm_reram_hwa import (
        IbmReramHwaConfig,
        build_ibm_reram_hwa_modifier,
    )

    config_path = (CONFIG_ROOT / "exact_bounds_hwa.json").resolve()
    _definition, spec = resolve_experiment_config(config_path, "train")
    data = build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed,
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )
    teacher, _teacher_metadata = _load_teacher(
        teacher_path,
        device=torch.device(spec.runtime.device),
        spec=spec,
    )
    preflight_dir = attempt_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=False)
    stack, canonical_path, canonicalization = _canonicalize_base_checkpoint(
        preflight_dir=preflight_dir,
        spec=spec,
        teacher=teacher,
        validation_loader=data.validation,
        source_base_path=source_base_path,
        teacher_path=teacher_path,
    )
    # Reload through the same public codec used by native --weights startup.
    canonical_loaded = load_named_weights(canonical_path, stack.bundle.catalog)
    stack.cost.gain = float(canonical_loaded.metadata["fixed_logit_gain"])

    selection_population_path = preflight_dir / "ibm_om_population.selection.npz"
    selection_receipt_path = (
        preflight_dir / "ibm_om_population.selection.receipt.json"
    )
    selection_parameters = dict(
        spec.settings.selection_weight_modifier.parameters
    )
    with _external_sampler_environment(aihwkit_python):
        selection_modifier = build_ibm_reram_hwa_modifier(
            stack.bundle.catalog.trainable,
            IbmReramHwaConfig(**selection_parameters),
            device_model_path=device_model_path,
            conductance_min=spec.model.conductance_min,
            conductance_max=spec.model.conductance_max,
            population_path=selection_population_path,
            population_receipt_path=selection_receipt_path,
        )
    mapped, mapping_report = selection_modifier.preflight_target_mapping()
    mapped_cpu = mapped.detach().cpu()
    mapped_path = preflight_dir / "mapped_targets.pt"
    atomic_torch_save(mapped_cpu, mapped_path)

    training_population_path = preflight_dir / "ibm_om_population.training.npz"
    training_receipt_path = (
        preflight_dir / "ibm_om_population.training.receipt.json"
    )
    training_parameters = dict(spec.settings.weight_modifier.parameters)
    with _external_sampler_environment(aihwkit_python):
        training_modifier = build_ibm_reram_hwa_modifier(
            stack.bundle.catalog.trainable,
            IbmReramHwaConfig(**training_parameters),
            device_model_path=device_model_path,
            conductance_min=spec.model.conductance_min,
            conductance_max=spec.model.conductance_max,
            population_path=training_population_path,
            population_receipt_path=training_receipt_path,
        )
    training_mapped, training_mapping_report = (
        training_modifier.preflight_target_mapping()
    )
    if (
        training_modifier.population_fingerprint
        != selection_modifier.population_fingerprint
        or training_mapping_report != mapping_report
        or not torch.equal(training_mapped, mapped)
    ):
        raise RuntimeError(
            "Expected compact and pulse roles to reproduce the exact fixed "
            "population, authoritative pair mapping, and mapped tensor."
        )

    with training_modifier.training_context():
        pass
    compact_report = training_modifier.programming_report
    compact_deployment = training_modifier.last_deployment_bundle
    if compact_report is None or compact_deployment is None:
        raise RuntimeError("Expected a real compact training context artifact.")
    compact_report_path = preflight_dir / "compact_canary_report.json"
    compact_deployment_path = preflight_dir / "compact_canary_deployment.pt"
    atomic_write_json(compact_report_path, compact_report)
    atomic_torch_save(compact_deployment, compact_deployment_path)

    # This is the sole exact selection programming context. The evaluator
    # observes the already-applied apparent endpoints and does not reprogram.
    apparent_metrics = _evaluate_apparent_forward_canary(
        selection_modifier,
        evaluator=_evaluate,
        stack=stack,
        teacher=teacher,
        validation_loader=data.validation,
    )
    pulse_report = selection_modifier.programming_report
    pulse_deployment = selection_modifier.last_deployment_bundle
    if pulse_report is None or pulse_deployment is None:
        raise RuntimeError("Expected an exact pulse selection deployment artifact.")
    pulse_report_path = preflight_dir / "pulse_canary_report.json"
    pulse_deployment_path = preflight_dir / "pulse_canary_deployment.pt"
    network_path = preflight_dir / "apparent_forward_network_metrics.json"
    atomic_write_json(pulse_report_path, pulse_report)
    atomic_torch_save(pulse_deployment, pulse_deployment_path)
    network_report = {
        "schema": "ebl.mnist_ibm_om_differential_pair.network_preflight",
        "schema_version": 1,
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
        "metrics": apparent_metrics,
    }
    atomic_write_json(network_path, network_report)

    serialization_canary = _validate_serialized_canary_artifacts(
        compact_report=compact_report,
        compact_report_path=compact_report_path,
        compact_deployment=compact_deployment,
        compact_deployment_path=compact_deployment_path,
        pulse_report=pulse_report,
        pulse_report_path=pulse_report_path,
        pulse_deployment=pulse_deployment,
        pulse_deployment_path=pulse_deployment_path,
        network_report=network_report,
        network_path=network_path,
    )

    selection_receipt = _read_json(selection_receipt_path)
    training_receipt = _read_json(training_receipt_path)
    if selection_receipt is None or training_receipt is None:
        raise RuntimeError("Expected both strict fixed-population receipts.")
    finite = torch.isfinite(mapped_cpu)
    outside = finite & ((mapped_cpu < 0.0) | (mapped_cpu > 1.0))
    return {
        "schema": "ebl.mnist_ibm_om_differential_pair.mapping_preflight",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_at": _utc_now(),
        "authoritative_entry_point": (
            "training.ibm_reram_hwa."
            "IbmReramHwaParameterModifier.preflight_target_mapping"
        ),
        "inputs": {
            "source_base_checkpoint": {
                "path": str(source_base_path.resolve()),
                "sha256": sha256_file(source_base_path),
            },
            "canonical_base_checkpoint": {
                "path": str(canonical_path.resolve()),
                "sha256": sha256_file(canonical_path),
            },
            "relu_teacher_checkpoint": {
                "path": str(teacher_path.resolve()),
                "sha256": sha256_file(teacher_path),
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
        "canonicalization": canonicalization,
        "selection_modifier_parameters": to_plain_data(selection_parameters),
        "training_modifier_parameters": to_plain_data(training_parameters),
        "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
        "mapping_report": mapping_report,
        "training_mapping_report": training_mapping_report,
        "selection_population_receipt": selection_receipt,
        "training_population_receipt": training_receipt,
        "mapped_targets": {
            "count": int(mapped_cpu.numel()),
            "finite_count": int(finite.sum().item()),
            "outside_0_1_count": int(outside.sum().item()),
            "minimum": float(mapped_cpu.min().item()),
            "maximum": float(mapped_cpu.max().item()),
            "sha256": sha256_file(mapped_path),
        },
        "compact_training_canary_report": compact_report,
        "pulse_resolved_canary_report": pulse_report,
        "network_preflight": network_report,
        "serialization_canary": serialization_canary,
        "artifacts": {
            **canonicalization["artifacts"],
            "selection_population": _artifact_record(
                selection_population_path,
                kind="ibm_om_array_population",
            ),
            "selection_population_receipt": _artifact_record(
                selection_receipt_path,
                kind="ibm_om_array_population_receipt",
            ),
            "training_population": _artifact_record(
                training_population_path,
                kind="ibm_om_array_population",
            ),
            "training_population_receipt": _artifact_record(
                training_receipt_path,
                kind="ibm_om_array_population_receipt",
            ),
            "mapped_targets": _artifact_record(
                mapped_path,
                kind="authoritative_pair_mapped_targets",
            ),
            "compact_report": _artifact_record(
                compact_report_path,
                kind="compact_pair_canary_report",
            ),
            "compact_deployment": _artifact_record(
                compact_deployment_path,
                kind="compact_pair_canary_deployment",
                endpoint_application_policy=compact_deployment.get(
                    "endpoint_application_policy"
                ),
                endpoint_generation_policy=compact_deployment.get(
                    "endpoint_generation_policy"
                ),
            ),
            "pulse_report": _artifact_record(
                pulse_report_path,
                kind="pulse_pair_canary_report",
            ),
            "pulse_deployment": _artifact_record(
                pulse_deployment_path,
                kind="pulse_pair_canary_deployment",
                endpoint_application_policy=pulse_deployment.get(
                    "endpoint_application_policy"
                ),
            ),
            "network_preflight": _artifact_record(
                network_path,
                kind="single_context_apparent_forward_metrics",
            ),
        },
    }


def _finite_summary(value: Any, *, strictly_positive_minimum: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    numbers = [value.get(key) for key in ("minimum", "mean", "maximum")]
    if not all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in numbers
    ):
        return False
    minimum, mean, maximum = (float(item) for item in numbers)
    return (
        minimum <= mean <= maximum
        and (minimum > 0.0 if strictly_positive_minimum else minimum >= 0.0)
    )


def _validate_canonicalization_report(report: Mapping[str, Any]) -> str:
    source = report.get("source_checkpoint")
    canonical = report.get("canonical_checkpoint")
    teacher = report.get("teacher_checkpoint")
    metrics = report.get("canonical_metrics")
    artifacts = report.get("artifacts")
    if not all(
        isinstance(value, Mapping)
        for value in (source, canonical, teacher, metrics, artifacts)
    ):
        raise RuntimeError("Expected complete canonicalization provenance.")
    canonical_sha = canonical.get("sha256")
    if (
        report.get("schema")
        != "ebl.mnist_ibm_om_differential_pair.canonical_replay"
        or report.get("schema_version") != 1
        or report.get("operation") != "metadata_only_tensor_identical_copy"
        or source.get("sha256") != SOURCE_BASE_CHECKPOINT_SHA256
        or teacher.get("sha256") != RELU_TEACHER_CHECKPOINT_SHA256
        or not isinstance(canonical_sha, str)
        or _SHA256.fullmatch(canonical_sha) is None
        or report.get("catalog_order") != list(EXPECTED_BINDING_KEYS)
        or report.get("tensor_equality") != "torch.equal_all_named_tensors"
        or report.get("unequal_tensor_keys") != []
        or report.get("current_metadata_validation") != "passed"
        or report.get("metrics_exactly_equal") is not True
        or report.get("source_metrics") != metrics
        or report.get("teacher_mapping_invoked") is not False
        or report.get("conductance_remapping_invoked") is not False
        or report.get("optimizer_updates") != 0
        or report.get("retraining_epochs") != 0
    ):
        raise RuntimeError(
            "Expected one exact metadata-only current-schema canonicalization."
        )
    if (
        metrics.get("examples") != CANONICAL_REPLAY_EXAMPLES
        or not all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics.get(key), bool)
            and math.isfinite(float(metrics[key]))
            for key in (
                "student_accuracy",
                "teacher_agreement",
                "kl_teacher_student",
            )
        )
        or float(metrics["student_accuracy"])
        < MINIMUM_CANONICAL_REPLAY_STUDENT_ACCURACY
        or float(metrics["teacher_agreement"])
        < MINIMUM_CANONICAL_REPLAY_TEACHER_AGREEMENT
        or float(metrics["kl_teacher_student"]) > MAXIMUM_CANONICAL_REPLAY_KL
    ):
        raise RuntimeError("Expected canonical replay quality gates to pass.")
    required_artifacts = {
        "canonical_checkpoint",
        "canonical_metadata",
        "canonical_replay",
    }
    if set(artifacts) != required_artifacts or any(
        not isinstance(artifacts[key], Mapping)
        or not isinstance(artifacts[key].get("sha256"), str)
        or _SHA256.fullmatch(str(artifacts[key].get("sha256"))) is None
        for key in required_artifacts
    ):
        raise RuntimeError("Expected durable strict canonical artifacts.")
    if artifacts["canonical_checkpoint"].get("sha256") != canonical_sha:
        raise RuntimeError("Expected canonical checkpoint digest parity.")
    return canonical_sha


def _validate_authoritative_mapping_report(
    report: Mapping[str, Any],
    *,
    expected_population_fingerprint: str = EXPECTED_POPULATION_FINGERPRINT,
) -> str:
    fingerprint = report.get("population_fingerprint")
    pair_binding = report.get("differential_pair_binding")
    parameters = report.get("parameters")
    parameter_pairs = report.get("parameter_pairs")
    nonempty_span = report.get("nonempty_inner_common_span")
    mapped_support = report.get("mapped_target_support")
    if not all(
        isinstance(value, Mapping)
        for value in (
            pair_binding,
            parameters,
            parameter_pairs,
            nonempty_span,
            mapped_support,
        )
    ):
        raise RuntimeError("Expected complete authoritative pair mapping fields.")
    empty = report.get("common_window_empty_group_count")
    below_nonempty = report.get(
        "mapped_target_below_lower_bound_nonempty_group"
    )
    above_nonempty = report.get(
        "mapped_target_above_upper_bound_nonempty_group"
    )
    below_empty = report.get("mapped_target_below_lower_bound_empty_group")
    above_empty = report.get("mapped_target_above_upper_bound_empty_group")
    if (
        report.get("target_mapping") != TARGET_MAPPING
        or report.get("common_window_margin_fraction")
        != COMMON_WINDOW_MARGIN_FRACTION
        or fingerprint != expected_population_fingerprint
        or report.get("corruption_policy") != CORRUPTION_POLICY
        or report.get("devices") != EXPECTED_TOTAL_DEVICES
        or report.get("global_target_outside_0_1") != 0
        or report.get("common_window_grouping") != "differential_pair"
        or report.get("common_window_group_size") != 2
        or report.get("common_window_group_count") != EXPECTED_PAIR_COUNT
        or report.get("pair_count") != EXPECTED_PAIR_COUNT
        or empty != EXPECTED_EMPTY_PAIR_COUNT
        or report.get("common_window_empty_pair_count")
        != EXPECTED_EMPTY_PAIR_COUNT
        or report.get("corrupt_group_count") != 0
        or report.get("corrupt_pair_count") != 0
        or below_nonempty != 0
        or above_nonempty != 0
        or report.get("mapped_target_below_lower_bound_nonempty_pair") != 0
        or report.get("mapped_target_above_upper_bound_nonempty_pair") != 0
        or not isinstance(below_empty, int)
        or not isinstance(above_empty, int)
        or below_empty < 0
        or above_empty < 0
        or below_empty + above_empty != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or report.get("mapped_target_below_lower_bound_empty_pair")
        != below_empty
        or report.get("mapped_target_above_upper_bound_empty_pair")
        != above_empty
        or mapped_support.get("below_lower_bound") != below_empty
        or mapped_support.get("above_upper_bound") != above_empty
        or mapped_support.get("inside_bounds")
        != EXPECTED_COMPACT_ENDPOINT_DEVICES
        or not _finite_summary(
            report.get("raw_common_span"),
            strictly_positive_minimum=False,
        )
        or not _finite_summary(
            report.get("inner_common_span"),
            strictly_positive_minimum=False,
        )
        or not _finite_summary(
            nonempty_span,
            strictly_positive_minimum=True,
        )
        or report.get("nonempty_pair_inner_common_span") != nonempty_span
    ):
        raise RuntimeError(
            "Expected the exact canonical differential-pair mapping fixture."
        )
    empty_fraction = empty / EXPECTED_PAIR_COUNT
    if (
        empty > MAXIMUM_EMPTY_PAIR_COUNT
        or empty_fraction > MAXIMUM_EMPTY_PAIR_FRACTION
        or not math.isclose(
            float(report.get("common_window_empty_fraction", float("nan"))),
            empty_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("Expected pair structural reachability gates to pass.")
    if (
        pair_binding.get("policy")
        != "canonical_adjacent_plus_minus_same_coordinate"
        or pair_binding.get("catalog_order") != list(EXPECTED_BINDING_KEYS)
        or set(parameters) != set(EXPECTED_BINDING_KEYS)
        or set(parameter_pairs)
        != {"base.differential_pair.0", "base.differential_pair.1"}
    ):
        raise RuntimeError("Expected canonical adjacent plus/minus provenance.")
    pairs = pair_binding.get("pairs")
    if (
        not isinstance(pairs, list)
        or pairs
        != [
            {
                "pair_index": 0,
                "conductance_plus_key": EXPECTED_BINDING_KEYS[0],
                "conductance_minus_key": EXPECTED_BINDING_KEYS[1],
                "shape": [1568, 100],
            },
            {
                "pair_index": 1,
                "conductance_plus_key": EXPECTED_BINDING_KEYS[2],
                "conductance_minus_key": EXPECTED_BINDING_KEYS[3],
                "shape": [100, 20],
            },
        ]
    ):
        raise RuntimeError("Expected exact two-layer differential pair binding.")
    expected_layers = (
        {
            "pair_key": "base.differential_pair.0",
            "pair_index": 0,
            "plus_key": EXPECTED_BINDING_KEYS[0],
            "minus_key": EXPECTED_BINDING_KEYS[1],
            "shape": [1568, 100],
            "cell_count": 156800,
            "pair_devices": 313600,
            "empty_pairs": 8,
        },
        {
            "pair_key": "base.differential_pair.1",
            "pair_index": 1,
            "plus_key": EXPECTED_BINDING_KEYS[2],
            "minus_key": EXPECTED_BINDING_KEYS[3],
            "shape": [100, 20],
            "cell_count": 2000,
            "pair_devices": 4000,
            "empty_pairs": 0,
        },
    )
    pair_empty_oob = 0
    parameter_empty_oob = 0
    for layer in expected_layers:
        pair_report = parameter_pairs.get(layer["pair_key"])
        plus_report = parameters.get(layer["plus_key"])
        minus_report = parameters.get(layer["minus_key"])
        if not all(
            isinstance(value, Mapping)
            for value in (pair_report, plus_report, minus_report)
        ):
            raise RuntimeError("Expected complete per-layer pair reports.")
        if (
            pair_report.get("pair_index") != layer["pair_index"]
            or pair_report.get("conductance_plus_key") != layer["plus_key"]
            or pair_report.get("conductance_minus_key") != layer["minus_key"]
            or pair_report.get("shape") != layer["shape"]
            or pair_report.get("devices") != layer["pair_devices"]
            or pair_report.get("pair_count") != layer["cell_count"]
            or pair_report.get("common_window_empty_pair_count")
            != layer["empty_pairs"]
            or pair_report.get("corrupt_pair_count") != 0
            or pair_report.get(
                "mapped_target_below_lower_bound_nonempty_pair"
            )
            != 0
            or pair_report.get(
                "mapped_target_above_upper_bound_nonempty_pair"
            )
            != 0
            or not _finite_summary(
                pair_report.get("nonempty_inner_common_span"),
                strictly_positive_minimum=True,
            )
            or pair_report.get("nonempty_pair_inner_common_span")
            != pair_report.get("nonempty_inner_common_span")
        ):
            raise RuntimeError("Expected exact per-layer pair aggregation.")
        pair_empty_oob += int(
            pair_report.get("mapped_target_below_lower_bound_empty_pair", -1)
        ) + int(
            pair_report.get("mapped_target_above_upper_bound_empty_pair", -1)
        )
        for role, key, peer, branch_report in (
            ("conductance_plus", layer["plus_key"], layer["minus_key"], plus_report),
            ("conductance_minus", layer["minus_key"], layer["plus_key"], minus_report),
        ):
            if (
                branch_report.get("differential_role") != role
                or branch_report.get("paired_parameter") != peer
                or branch_report.get("pair_index") != layer["pair_index"]
                or branch_report.get("shape") != layer["shape"]
                or branch_report.get("devices") != layer["cell_count"]
                or branch_report.get("pair_count") != layer["cell_count"]
                or branch_report.get("common_window_empty_pair_count")
                != layer["empty_pairs"]
                or branch_report.get(
                    "mapped_target_below_lower_bound_nonempty_pair"
                )
                != 0
                or branch_report.get(
                    "mapped_target_above_upper_bound_nonempty_pair"
                )
                != 0
                or not _finite_summary(
                    branch_report.get("nonempty_inner_common_span"),
                    strictly_positive_minimum=True,
                )
                or branch_report.get("nonempty_pair_inner_common_span")
                != branch_report.get("nonempty_inner_common_span")
            ):
                raise RuntimeError(
                    f"Expected exact branch report for {key!r}."
                )
            parameter_empty_oob += int(
                branch_report.get(
                    "mapped_target_below_lower_bound_empty_pair",
                    -1,
                )
            ) + int(
                branch_report.get(
                    "mapped_target_above_upper_bound_empty_pair",
                    -1,
                )
            )
    if (
        pair_empty_oob != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or parameter_empty_oob != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
    ):
        raise RuntimeError(
            "Expected per-layer reports to partition the 16 empty-pair cells."
        )
    return str(fingerprint)


def _validate_compact_training_canary(
    report: Mapping[str, Any],
    *,
    mapping_report: Mapping[str, Any],
    expected_fallback_indices_sha256: str | None = None,
) -> str:
    fallback = report.get("pulse_resolved_fallback")
    report_mapping = report.get("target_mapping_report")
    if not isinstance(fallback, Mapping) or not isinstance(
        report_mapping,
        Mapping,
    ):
        raise RuntimeError("Expected compact pair report and mapping.")
    _validate_authoritative_mapping_report(report_mapping)
    if report_mapping != mapping_report:
        raise RuntimeError("Expected compact canary mapper-report identity.")
    indices = fallback.get("selection_indices")
    digest = fallback.get("selection_indices_sha256")
    pulse_count = fallback.get("pulse_count")
    report_below = report.get("target_below_lower_bound")
    report_above = report.get("target_above_upper_bound")
    fallback_below = fallback.get("target_below_lower_bound")
    fallback_above = fallback.get("target_above_upper_bound")
    fallback_accepted = fallback.get("accepted")
    fallback_exhausted = fallback.get("budget_exhausted")
    fallback_success = fallback.get("success_fraction")
    fallback_saturated = fallback.get("saturated")
    fallback_clipped = fallback.get("endpoint_clipped")
    compact_accepted = report.get("accepted")
    compact_success = report.get("success_fraction")
    compact_failed = report.get("failed_noncorrupt")
    integer_counts = (
        report_below,
        report_above,
        fallback_below,
        fallback_above,
        fallback_accepted,
        fallback_exhausted,
        fallback_saturated,
        fallback_clipped,
        compact_accepted,
        compact_failed,
    )
    counts_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in integer_counts
    )
    computed_digest: str | None = None
    if isinstance(indices, list) and all(
        isinstance(index, int) and not isinstance(index, bool) for index in indices
    ):
        import torch

        computed_digest = sha256(
            torch.tensor(indices, dtype=torch.int64).numpy().tobytes()
        ).hexdigest()
    if (
        report.get("execution") != "compact_endpoint"
        or report.get("execution_detail") != COMPACT_EXECUTION_DETAIL
        or report.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or report.get("endpoint_seed") != TRAINING_ENDPOINT_SEED
        or report.get("assignment_seed") != ASSIGNMENT_SEED
        or report.get("corruption_policy") != CORRUPTION_POLICY
        or report.get("population_fingerprint")
        != EXPECTED_POPULATION_FINGERPRINT
        or report.get("population_sampling_backend")
        != "external_pinned_aihwkit_python"
        or report.get("device_model_sha256") != DEVICE_MODEL_SHA256
        or report.get("target_mapping") != TARGET_MAPPING
        or report.get("devices") != EXPECTED_TOTAL_DEVICES
        or report.get("compact_endpoint_devices")
        != EXPECTED_COMPACT_ENDPOINT_DEVICES
        or report.get("pulse_resolved_fallback_devices")
        != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or report.get("corrupt") != 0
        or not counts_valid
        or report_below + report_above
        != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or report.get("target_inside_bounds")
        != EXPECTED_COMPACT_ENDPOINT_DEVICES
        or compact_accepted > EXPECTED_TOTAL_DEVICES
        or compact_failed > EXPECTED_TOTAL_DEVICES
        or compact_accepted + compact_failed != EXPECTED_TOTAL_DEVICES
        or not isinstance(compact_success, (int, float))
        or isinstance(compact_success, bool)
        or not math.isfinite(float(compact_success))
        or not 0.0 <= float(compact_success) <= 1.0
        or not math.isclose(
            float(compact_success),
            compact_accepted / EXPECTED_TOTAL_DEVICES,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or fallback.get("policy") != COMPACT_FALLBACK_POLICY
        or fallback.get("devices") != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or not isinstance(indices, list)
        or len(indices) != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or len(set(indices)) != len(indices)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < EXPECTED_TOTAL_DEVICES
            for index in indices
        )
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or digest != computed_digest
        or (
            expected_fallback_indices_sha256 is not None
            and digest != expected_fallback_indices_sha256
        )
        or fallback_below + fallback_above
        != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or fallback_accepted > EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or fallback_exhausted > EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or fallback_accepted + fallback_exhausted
        != EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or fallback_saturated > EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or fallback_clipped > EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        or not isinstance(fallback_success, (int, float))
        or isinstance(fallback_success, bool)
        or not math.isfinite(float(fallback_success))
        or not 0.0 <= float(fallback_success) <= 1.0
        or not math.isclose(
            float(fallback_success),
            fallback_accepted / EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or fallback.get("maximum_program_pulses") != 128
        or fallback.get("nonfinite") != 0
        or not isinstance(pulse_count, Mapping)
        or isinstance(pulse_count.get("maximum"), bool)
        or not isinstance(pulse_count.get("maximum"), (int, float))
        or not 0 <= float(pulse_count["maximum"]) <= 128
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in (
                pulse_count.get("mean"),
                pulse_count.get("median"),
                pulse_count.get("set_mean"),
                pulse_count.get("reset_mean"),
                fallback.get("verify_count_mean"),
                fallback.get("reversal_count_mean"),
            )
        )
        or float(pulse_count["mean"]) > 128.0
        or float(pulse_count["median"]) > 128.0
        or float(pulse_count["set_mean"]) > 128.0
        or float(pulse_count["reset_mean"]) > 128.0
        or float(fallback["verify_count_mean"]) > 129.0
        or float(fallback["reversal_count_mean"]) > 128.0
    ):
        raise RuntimeError("Expected the exact 317584-plus-16 compact partition.")
    return digest


def _validate_pulse_canary(
    report: Mapping[str, Any],
    *,
    mapping_report: Mapping[str, Any],
) -> None:
    report_mapping = report.get("target_mapping_report")
    if not isinstance(report_mapping, Mapping):
        raise RuntimeError("Expected pulse canary authoritative mapper report.")
    _validate_authoritative_mapping_report(report_mapping)
    devices = report.get("devices")
    accepted = report.get("accepted")
    exhausted = report.get("budget_exhausted")
    success = report.get("success_fraction")
    if (
        report_mapping != mapping_report
        or report.get("execution") != "pulse_resolved"
        or report.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or report.get("endpoint_seed") != SELECTION_ENDPOINT_SEED
        or report.get("assignment_seed") != ASSIGNMENT_SEED
        or report.get("corruption_policy") != CORRUPTION_POLICY
        or report.get("population_fingerprint")
        != EXPECTED_POPULATION_FINGERPRINT
        or report.get("population_sampling_backend")
        != "external_pinned_aihwkit_python"
        or report.get("device_model_sha256") != DEVICE_MODEL_SHA256
        or report.get("target_mapping") != TARGET_MAPPING
        or devices != EXPECTED_TOTAL_DEVICES
        or report.get("corrupt") != 0
        or report.get("nonfinite") != 0
        or not isinstance(accepted, int)
        or isinstance(accepted, bool)
        or not 0 <= accepted <= devices
        or not isinstance(exhausted, int)
        or isinstance(exhausted, bool)
        or exhausted < 0
        or accepted + exhausted != devices
        or exhausted / devices > 0.01
        or not isinstance(success, (int, float))
        or isinstance(success, bool)
        or not math.isfinite(float(success))
        or float(success) < 0.99
        or not math.isclose(
            float(success),
            accepted / devices,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    ):
        raise RuntimeError("Expected the exact pulse-resolved canary gates.")


def _validate_preflight_report(report: Mapping[str, Any]) -> tuple[str, str, str]:
    if (
        report.get("schema")
        != "ebl.mnist_ibm_om_differential_pair.mapping_preflight"
        or report.get("schema_version") != 1
        or report.get("study_id") != STUDY_ID
        or report.get("authoritative_entry_point")
        != (
            "training.ibm_reram_hwa."
            "IbmReramHwaParameterModifier.preflight_target_mapping"
        )
    ):
        raise RuntimeError("Expected the authoritative pair preflight schema.")
    inputs = report.get("inputs")
    configs = report.get("configs")
    canonicalization = report.get("canonicalization")
    if not all(
        isinstance(value, Mapping)
        for value in (inputs, configs, canonicalization)
    ):
        raise RuntimeError("Expected complete preflight provenance.")
    if (
        inputs.get("source_base_checkpoint", {}).get("sha256")
        != SOURCE_BASE_CHECKPOINT_SHA256
        or inputs.get("relu_teacher_checkpoint", {}).get("sha256")
        != RELU_TEACHER_CHECKPOINT_SHA256
        or inputs.get("ibm_om_device_model", {}).get("sha256")
        != DEVICE_MODEL_SHA256
    ):
        raise RuntimeError("Expected all three frozen input digests.")
    canonical_sha = _validate_canonicalization_report(canonicalization)
    if inputs.get("canonical_base_checkpoint", {}).get("sha256") != canonical_sha:
        raise RuntimeError("Expected canonical checkpoint provenance parity.")
    for arm, filename in TRAINING_ARM_CONFIGS.items():
        record = configs.get(arm)
        if not isinstance(record, Mapping) or record.get("sha256") != (
            sha256_file(CONFIG_ROOT / filename)
        ):
            raise RuntimeError("Expected both exact prepared config hashes.")
    if (
        _backfilled_parameters(report.get("selection_modifier_parameters"))
        != _expected_modifier_parameters(execution="pulse_resolved")
        or _backfilled_parameters(report.get("training_modifier_parameters"))
        != _expected_modifier_parameters(execution="compact_endpoint")
        or report.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
    ):
        raise RuntimeError("Expected the frozen pair modifier parameters.")

    mapping = report.get("mapping_report")
    training_mapping = report.get("training_mapping_report")
    selection_receipt = report.get("selection_population_receipt")
    training_receipt = report.get("training_population_receipt")
    mapped = report.get("mapped_targets")
    compact = report.get("compact_training_canary_report")
    pulse = report.get("pulse_resolved_canary_report")
    network = report.get("network_preflight")
    serialization = report.get("serialization_canary")
    artifacts = report.get("artifacts")
    if not all(
        isinstance(value, Mapping)
        for value in (
            mapping,
            training_mapping,
            selection_receipt,
            training_receipt,
            mapped,
            compact,
            pulse,
            network,
            serialization,
            artifacts,
        )
    ):
        raise RuntimeError("Expected all pair preflight outputs.")
    fingerprint = _validate_authoritative_mapping_report(mapping)
    _validate_authoritative_mapping_report(training_mapping)
    if training_mapping != mapping:
        raise RuntimeError("Expected exact compact/selection mapping parity.")
    expected_receipt_request = {
        "assignment_seed": ASSIGNMENT_SEED,
        "corruption_policy": CORRUPTION_POLICY,
        "binding_keys": list(EXPECTED_BINDING_KEYS),
    }
    for receipt in (selection_receipt, training_receipt):
        request = receipt.get("request")
        if (
            receipt.get("population_fingerprint") != fingerprint
            or receipt.get("aihwkit_version") != "1.1.0"
            or not isinstance(request, Mapping)
            or any(
                request.get(key) != value
                for key, value in expected_receipt_request.items()
            )
        ):
            raise RuntimeError("Expected pinned pair population receipts.")
    if training_receipt.get("request") != selection_receipt.get("request"):
        raise RuntimeError("Expected exact compact/selection sampler request parity.")
    if (
        mapped.get("count") != EXPECTED_TOTAL_DEVICES
        or mapped.get("finite_count") != EXPECTED_TOTAL_DEVICES
        or mapped.get("outside_0_1_count") != 0
        or not all(
            isinstance(mapped.get(key), (int, float))
            and not isinstance(mapped.get(key), bool)
            and math.isfinite(float(mapped[key]))
            for key in ("minimum", "maximum")
        )
        or float(mapped["minimum"]) < 0.0
        or float(mapped["maximum"]) > 1.0
        or not isinstance(mapped.get("sha256"), str)
        or _SHA256.fullmatch(str(mapped.get("sha256"))) is None
    ):
        raise RuntimeError("Expected finite authoritative targets inside [0, 1].")
    fallback_digest = _validate_compact_training_canary(
        compact,
        mapping_report=training_mapping,
    )
    _validate_pulse_canary(pulse, mapping_report=mapping)

    cohort = network.get("cohort")
    metrics = network.get("metrics")
    if (
        network.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or network.get("forward_endpoint") != "aihwkit_apparent_endpoint"
        or network.get("persistent_endpoint_role") != "hidden_update_state"
        or network.get("programming_contexts") != 1
        or cohort
        != {
            "split": "validation",
            "data_seed": 42,
            "validation_points": 5000,
            "batch_size": 16,
            "maximum_batches": NETWORK_PREFLIGHT_BATCHES,
            "examples": NETWORK_PREFLIGHT_EXAMPLES,
            "shuffle": False,
        }
        or not isinstance(metrics, Mapping)
        or metrics.get("examples") != NETWORK_PREFLIGHT_EXAMPLES
        or not all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics.get(key), bool)
            and math.isfinite(float(metrics[key]))
            for key in ("student_accuracy", "teacher_agreement")
        )
        or float(metrics["student_accuracy"])
        < MINIMUM_NETWORK_PREFLIGHT_STUDENT_ACCURACY
        or float(metrics["teacher_agreement"])
        < MINIMUM_NETWORK_PREFLIGHT_TEACHER_AGREEMENT
    ):
        raise RuntimeError("Expected the single-context apparent-forward gate.")
    if serialization != {
        "json_round_trip": "passed",
        "checkpoint_round_trip": "passed",
        "allow_nan": False,
        "pickle_unsafe_custom_objects": False,
        "compact_and_fallback_mask_partition": "passed",
        "fallback_indices_mask_report_parity": "passed",
        "finite_apparent_and_persistent_endpoints": "passed",
    }:
        raise RuntimeError("Expected strict JSON/checkpoint serialization gates.")
    required_artifacts = {
        "canonical_checkpoint",
        "canonical_metadata",
        "canonical_replay",
        "selection_population",
        "selection_population_receipt",
        "training_population",
        "training_population_receipt",
        "mapped_targets",
        "compact_report",
        "compact_deployment",
        "pulse_report",
        "pulse_deployment",
        "network_preflight",
    }
    if set(artifacts) != required_artifacts or any(
        not isinstance(artifacts[key], Mapping)
        or not isinstance(artifacts[key].get("sha256"), str)
        or _SHA256.fullmatch(str(artifacts[key].get("sha256"))) is None
        for key in required_artifacts
    ):
        raise RuntimeError("Expected every durable preflight artifact digest.")
    if (
        artifacts["compact_deployment"].get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or artifacts["compact_deployment"].get("endpoint_generation_policy")
        != COMPACT_ENDPOINT_GENERATION_POLICY
        or artifacts["pulse_deployment"].get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
    ):
        raise RuntimeError("Expected exact apparent/persistent deployment policy.")
    return fingerprint, fallback_digest, canonical_sha


def _validate_completed_summary(summary: Mapping[str, Any]) -> None:
    if (
        summary.get("study_id") != STUDY_ID
        or summary.get("state") != "ready_for_review"
        or summary.get("ready_for_review") is not True
        or summary.get("validation_mode") != "full_artifact_hashes"
    ):
        raise RuntimeError(
            "Expected the pair pilot to be artifact-verified and ready for review."
        )
    raw_arms = summary.get("arms")
    if not isinstance(raw_arms, list):
        raise RuntimeError("Expected study summary arm records.")
    arms = {
        str(arm.get("arm_id")): arm
        for arm in raw_arms
        if isinstance(arm, Mapping)
    }
    if set(arms) != set(TRAINING_ARM_CONFIGS) or len(arms) != len(raw_arms):
        raise RuntimeError("Expected summary coverage for exactly both arms.")
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


def _result_artifacts(
    run_dir: Path,
    result: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    raw = result.get("artifacts")
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected result artifacts in {str(run_dir)!r}.")
    records: dict[str, list[Mapping[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for record in raw:
        if not isinstance(record, Mapping):
            raise RuntimeError("Expected every result artifact to be a record.")
        kind = record.get("kind")
        relative = record.get("path")
        if (
            not isinstance(kind, str)
            or not isinstance(relative, str)
            or (kind, relative) in seen
        ):
            raise RuntimeError("Expected unique result artifact kind/path pairs.")
        path = (run_dir / relative).resolve()
        if (
            run_dir.resolve() not in path.parents
            or not path.is_file()
            or record.get("sha256") != sha256_file(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"Expected full artifact parity for {kind!r}.")
        seen.add((kind, relative))
        records.setdefault(kind, []).append(record)
    return records


def _validate_native_manifest_inputs(
    manifest: Mapping[str, Any],
    *,
    canonical_sha256: str,
) -> None:
    raw_inputs = manifest.get("inputs")
    if not isinstance(raw_inputs, list):
        raise RuntimeError("Expected native manifest input records.")
    expected = {
        "teacher_weights": RELU_TEACHER_CHECKPOINT_SHA256,
        "weights": canonical_sha256,
        "device_model": DEVICE_MODEL_SHA256,
    }
    if (
        len(raw_inputs) != len(expected)
        or any(not isinstance(record, Mapping) for record in raw_inputs)
        or len({str(record.get("role")) for record in raw_inputs})
        != len(raw_inputs)
    ):
        raise RuntimeError(
            "Expected native arms to bind canonical weights, ReLU teacher, "
            "and IBM device model exactly once."
        )
    inputs = {
        str(record.get("role")): record
        for record in raw_inputs
        if isinstance(record, Mapping)
    }
    if set(inputs) != set(expected) or any(
        inputs[role].get("sha256") != digest
        for role, digest in expected.items()
    ):
        raise RuntimeError(
            "Expected native arms to bind canonical weights, ReLU teacher, "
            "and IBM device model exactly."
        )


def _validate_native_epoch_records(
    run_dir: Path,
) -> tuple[Mapping[str, Any], ...]:
    path = run_dir / "metrics.jsonl"
    if _line_count(path) != EXPECTED_EPOCHS_PER_ARM + 1:
        raise RuntimeError("Expected initialization plus exactly ten epoch records.")
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(
                    line,
                    parse_constant=_reject_nonfinite_json_constant,
                )
            except ValueError as error:
                raise RuntimeError("Expected strict metrics JSONL.") from error
            if not isinstance(value, Mapping):
                raise RuntimeError("Expected metrics JSON objects.")
            records.append(value)
    if records[0].get("mode") != "initialization":
        raise RuntimeError("Expected one initialization record before training.")
    epochs = records[1:]
    if (
        [record.get("epoch") for record in epochs]
        != list(range(EXPECTED_EPOCHS_PER_ARM))
        or [record.get("completed_epochs") for record in epochs]
        != list(range(1, EXPECTED_EPOCHS_PER_ARM + 1))
        or [record.get("mode") for record in epochs]
        != ["train"] * EXPECTED_EPOCHS_PER_ARM
        or [record.get("global_step") for record in epochs]
        != [3438 * epoch for epoch in range(1, EXPECTED_EPOCHS_PER_ARM + 1)]
    ):
        raise RuntimeError(
            "Expected complete ordered epoch/global-step coverage 0 through 9."
        )
    return tuple(records)


def _validate_result_metric_bindings(
    *,
    result: Mapping[str, Any],
    metrics: Mapping[str, Any],
    epoch_records: tuple[Mapping[str, Any], ...],
) -> tuple[int, Mapping[str, Any]]:
    """Bind selected/last result summaries to the immutable metric records."""

    if (
        result.get("schema") != "ebl.run"
        or result.get("schema_version") != 1
        or result.get("status") != "complete"
        or result.get("experiment_id") != "mnist_relu_drn_kd.v1"
        or result.get("error") is not None
    ):
        raise RuntimeError("Expected strict complete native result schema.")
    if len(epoch_records) != EXPECTED_EPOCHS_PER_ARM + 1:
        raise RuntimeError("Expected initialization plus ten metric records.")
    initial_record = epoch_records[0]
    final_record = epoch_records[-1]
    if (
        metrics.get("initial_validation") != initial_record.get("validation")
        or metrics.get("initial_clean_validation")
        != initial_record.get("validation_clean")
        or metrics.get("last_train") != final_record.get("train")
        or metrics.get("last_validation") != final_record.get("validation")
        or metrics.get("last_clean_validation")
        != final_record.get("validation_clean")
    ):
        raise RuntimeError(
            "Expected result initial/last metrics to equal metrics.jsonl records."
        )
    selected = metrics.get("selected")
    if not isinstance(selected, Mapping):
        raise RuntimeError("Expected one selected validation result.")
    selected_epoch = selected.get("epoch")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or not -1 <= selected_epoch < EXPECTED_EPOCHS_PER_ARM
    ):
        raise RuntimeError("Expected selected epoch -1 or one of epochs 0 through 9.")
    selected_validation = dict(selected)
    selected_validation.pop("epoch")
    source_record = (
        initial_record
        if selected_epoch == -1
        else epoch_records[selected_epoch + 1]
    )
    if selected_validation != source_record.get("validation"):
        raise RuntimeError(
            "Expected selected validation to equal its exact epoch record."
        )
    validations = [
        initial_record.get("validation"),
        *(record.get("validation") for record in epoch_records[1:]),
    ]
    kl_values: list[float] = []
    for validation in validations:
        value = (
            validation.get("kl_teacher_student")
            if isinstance(validation, Mapping)
            else None
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError("Expected finite KL in every validation record.")
        kl_values.append(float(value))
    first_argmin_epoch = min(range(len(kl_values)), key=kl_values.__getitem__) - 1
    if selected_epoch != first_argmin_epoch:
        raise RuntimeError("Expected the strict first-argmin KL selected epoch.")
    return selected_epoch, selected_validation


def _validate_selected_checkpoint_and_resume(
    *,
    run_dir: Path,
    metrics: Mapping[str, Any],
    selected_epoch: int,
    selected_validation: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    """Bind selected named weights and exact epoch-boundary resume state."""

    selected_path = run_dir / "checkpoints" / "weights.pt"
    resume_path = run_dir / "checkpoints" / "resume.pt"
    selected = _load_torch_weights_only(selected_path)
    resume = _load_torch_weights_only(resume_path)
    selected_metadata = (
        selected.get("metadata") if isinstance(selected, Mapping) else None
    )
    if (
        not isinstance(selected, Mapping)
        or selected.get("schema") != "drn.named-weights"
        or selected.get("schema_version") != 1
        or not isinstance(selected_metadata, Mapping)
        or selected_metadata.get("teacher_sha256")
        != RELU_TEACHER_CHECKPOINT_SHA256
        or selected_metadata.get("encoding") != "differential"
        or selected_metadata.get("selection_metric")
        != "validation_modifier_mean.kl_teacher_student"
        or selected_metadata.get("selection_epoch") != selected_epoch
        or selected_metadata.get("selection_value")
        != selected_validation.get("kl_teacher_student")
        or selected_metadata.get("selection_student_accuracy")
        != selected_validation.get("student_accuracy")
        or selected_metadata.get("selection_teacher_agreement")
        != selected_validation.get("teacher_agreement")
    ):
        raise RuntimeError("Expected strict selected checkpoint metric provenance.")
    progress = resume.get("progress_state") if isinstance(resume, Mapping) else None
    if (
        not isinstance(resume, Mapping)
        or resume.get("schema") != "drn.epoch-boundary-resume"
        or resume.get("schema_version") != 3
        or resume.get("resume_capability") != "exact"
        or resume.get("epoch") != EXPECTED_EPOCHS_PER_ARM
        or resume.get("global_step") != 34380
        or not isinstance(progress, Mapping)
        or progress.get("selected_epoch") != selected_epoch
        or progress.get("selected_validation") != selected_validation
        or progress.get("last_train") != metrics.get("last_train")
        or progress.get("last_validation") != metrics.get("last_validation")
        or progress.get("last_clean_validation")
        != metrics.get("last_clean_validation")
        or not _torch_payload_equal(resume.get("selected_weights"), selected)
    ):
        raise RuntimeError(
            "Expected exact epoch-10 resume and selected-checkpoint parity."
        )
    return selected, sha256_file(selected_path)


def _validate_pulse_deployment_integrity(
    deployment: Mapping[str, Any],
    *,
    selection_report: Mapping[str, Any],
    selected_metadata: Mapping[str, Any],
    selected_weights_sha256: str,
    selected_epoch: int,
) -> None:
    """Validate every network-facing tensor in the selected pulse bundle."""

    import torch

    from experiments.mnist_relu_drn.ibm_om_deployment_decomposition import (
        validate_deployment_contract,
    )

    expected_shapes = (
        (1568, 100),
        (1568, 100),
        (100, 20),
        (100, 20),
    )
    if (
        deployment.get("schema")
        != "ebl.ibm_reram.om_pulse_resolved_deployment"
        or deployment.get("schema_version") != 1
        or deployment.get("endpoint_application_policy")
        != ENDPOINT_APPLICATION_POLICY
        or deployment.get("population_fingerprint")
        != EXPECTED_POPULATION_FINGERPRINT
        or deployment.get("device_model_sha256") != DEVICE_MODEL_SHA256
        or deployment.get("selected_weights_sha256")
        != selected_weights_sha256
        or deployment.get("selected_epoch") != selected_epoch
        or tuple(deployment.get("binding_keys", ())) != EXPECTED_BINDING_KEYS
        or tuple(tuple(shape) for shape in deployment.get("binding_shapes", ()))
        != expected_shapes
        or deployment.get("report") != selection_report
        or deployment.get("target_mapping_report")
        != selection_report.get("target_mapping_report")
    ):
        raise RuntimeError("Expected selected pulse deployment provenance.")
    config = deployment.get("config")
    if not isinstance(config, Mapping) or dict(config) != (
        _expected_modifier_parameters(execution="pulse_resolved")
    ):
        raise RuntimeError("Expected selected pulse deployment config parity.")
    try:
        validate_deployment_contract(
            deployment,
            expected_selected_weights_sha256=selected_weights_sha256,
            expected_binding_keys=EXPECTED_BINDING_KEYS,
            expected_binding_shapes=expected_shapes,
            configured_modifiers=(
                _expected_modifier_parameters(execution="pulse_resolved"),
            ),
            checkpoint_metadata=selected_metadata,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Expected the selected pulse deployment contract to replay exactly."
        ) from error

    floating_keys = (
        "requested_target",
        "persistent_endpoint",
        "raw_apparent_endpoint",
        "apparent_endpoint",
        "global_requested_target",
    )
    boolean_keys = (
        "accepted",
        "budget_exhausted",
        "corrupt",
        "target_below_lower_bound",
        "target_inside_bounds",
        "target_above_upper_bound",
        "acceptance_window_reachable",
        "saturated",
    )
    counter_keys = (
        "set_count",
        "reset_count",
        "total_pulses",
        "verify_count",
        "reversals",
    )
    for key in floating_keys:
        value = deployment.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != (EXPECTED_TOTAL_DEVICES,)
            or not bool(torch.all(torch.isfinite(value)))
        ):
            raise RuntimeError(f"Expected finite pulse deployment tensor {key!r}.")
    for key in boolean_keys:
        value = deployment.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or tuple(value.shape) != (EXPECTED_TOTAL_DEVICES,)
        ):
            raise RuntimeError(f"Expected boolean pulse deployment mask {key!r}.")
    for key in counter_keys:
        value = deployment.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.int64
            or tuple(value.shape) != (EXPECTED_TOTAL_DEVICES,)
            or bool(torch.any(value < 0))
        ):
            raise RuntimeError(f"Expected non-negative pulse counter {key!r}.")

    accepted = deployment["accepted"]
    exhausted = deployment["budget_exhausted"]
    corrupt = deployment["corrupt"]
    below = deployment["target_below_lower_bound"]
    inside = deployment["target_inside_bounds"]
    above = deployment["target_above_upper_bound"]
    set_count = deployment["set_count"]
    reset_count = deployment["reset_count"]
    total = deployment["total_pulses"]
    verify = deployment["verify_count"]
    reversals = deployment["reversals"]
    raw = deployment["raw_apparent_endpoint"]
    apparent = deployment["apparent_endpoint"]
    if (
        bool(torch.any(accepted & exhausted))
        or not bool(torch.all(accepted | exhausted))
        or bool(torch.any(corrupt))
        or not bool(
            torch.all(
                (
                    below.to(torch.int8)
                    + inside.to(torch.int8)
                    + above.to(torch.int8)
                )
                == 1
            )
        )
        or not bool(torch.all(set_count + reset_count == total))
        or bool(torch.any(total > 128))
        or bool(torch.any(verify < 1))
        or bool(torch.any(verify > total + 1))
        or bool(torch.any(reversals > total))
        or not torch.equal(apparent, raw.clamp(0.0, 1.0))
        or bool(torch.any(apparent < 0.0))
        or bool(torch.any(apparent > 1.0))
    ):
        raise RuntimeError("Expected exact pulse tensor partitions and caps.")
    clipped = int((raw != apparent).sum().item())
    unreachable = int((~deployment["acceptance_window_reachable"]).sum().item())
    if (
        selection_report.get("accepted") != int(accepted.sum().item())
        or selection_report.get("budget_exhausted")
        != int(exhausted.sum().item())
        or selection_report.get("corrupt") != int(corrupt.sum().item())
        or selection_report.get("target_below_lower_bound")
        != int(below.sum().item())
        or selection_report.get("target_inside_bounds")
        != int(inside.sum().item())
        or selection_report.get("target_above_upper_bound")
        != int(above.sum().item())
        or selection_report.get("acceptance_window_unreachable") != unreachable
        or selection_report.get("saturated")
        != int(deployment["saturated"].sum().item())
        or selection_report.get("endpoint_clipped") != clipped
        or selection_report.get("pulse_count", {}).get("maximum")
        != int(total.max().item())
    ):
        raise RuntimeError("Expected pulse deployment masks to equal report counts.")


def _validate_population_artifact(
    run_dir: Path,
    *,
    role: str,
) -> dict[str, Any]:
    from training.ibm_reram_hwa import load_om_array_population

    prefix = run_dir / "artifacts" / f"ibm_om_population.{role}"
    population_path = Path(f"{prefix}.npz")
    receipt_path = Path(f"{prefix}.receipt.json")
    receipt = _read_json(receipt_path)
    if receipt is None:
        raise RuntimeError(f"Expected strict {role!r} population receipt.")
    population = load_om_array_population(population_path)
    request = receipt.get("request")
    if (
        population.fingerprint != EXPECTED_POPULATION_FINGERPRINT
        or population.size != EXPECTED_TOTAL_DEVICES
        or population.binding_keys != EXPECTED_BINDING_KEYS
        or receipt.get("population_fingerprint")
        != EXPECTED_POPULATION_FINGERPRINT
        or receipt.get("population_sha256") != sha256_file(population_path)
        or receipt.get("aihwkit_version") != "1.1.0"
        or not isinstance(request, Mapping)
        or request.get("assignment_seed") != ASSIGNMENT_SEED
        or request.get("corruption_policy") != CORRUPTION_POLICY
        or request.get("binding_keys") != list(EXPECTED_BINDING_KEYS)
    ):
        raise RuntimeError(f"Expected the frozen {role!r} pair population.")
    return {
        "population_fingerprint": population.fingerprint,
        "population_sha256": sha256_file(population_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def _validate_post_run_audit(
    study_dir: Path,
    *,
    canonical_sha256: str,
    fallback_indices_sha256: str,
) -> dict[str, Any]:
    """Audit both complete native runs, populations, reports, and bundles."""

    records: dict[str, Any] = {}
    for arm, roles in POPULATION_ROLES.items():
        attempts = _native_attempts(study_dir / "runs" / arm)
        if len(attempts) != 1 or not attempts[0].is_dir():
            raise RuntimeError(f"Expected one immutable native run for {arm!r}.")
        run_dir = attempts[0]
        status = _read_json(run_dir / "status.json")
        result = _read_json(run_dir / "result.json")
        manifest = _read_json(run_dir / "manifest.json")
        if (
            status is None
            or status.get("status") != "complete"
            or result is None
            or result.get("error") is not None
            or manifest is None
        ):
            raise RuntimeError(f"Expected complete native artifacts for {arm!r}.")
        _validate_native_manifest_inputs(
            manifest,
            canonical_sha256=canonical_sha256,
        )
        epoch_records = _validate_native_epoch_records(run_dir)
        artifact_records = _result_artifacts(run_dir, result)
        expected_artifact_pairs = {
            ("selected_named_weights", "checkpoints/weights.pt"),
            ("epoch_boundary_resume", "checkpoints/resume.pt"),
            (
                "ibm_om_persistent_deployment",
                "artifacts/ibm_om_deployment.pt",
            ),
            *(
                (
                    "ibm_om_array_population",
                    f"artifacts/ibm_om_population.{role}.npz",
                )
                for role in roles
            ),
            *(
                (
                    "ibm_om_array_population_receipt",
                    f"artifacts/ibm_om_population.{role}.receipt.json",
                )
                for role in roles
            ),
        }
        if arm == "train-exact-bounds-pair-hwa":
            expected_artifact_pairs.add(
                (
                    "ibm_om_training_device_programming",
                    "artifacts/ibm_om_training_device_programming.json",
                )
            )
        observed_artifact_pairs = {
            (kind, str(record.get("path")))
            for kind, kind_records in artifact_records.items()
            for record in kind_records
        }
        if observed_artifact_pairs != expected_artifact_pairs:
            raise RuntimeError(
                f"Expected the exact native artifact set for {arm!r}."
            )
        required_kinds = {
            "selected_named_weights",
            "epoch_boundary_resume",
            "ibm_om_persistent_deployment",
            "ibm_om_array_population",
            "ibm_om_array_population_receipt",
        }
        if not required_kinds.issubset(artifact_records):
            raise RuntimeError(f"Expected checkpoint/deployment artifacts for {arm!r}.")
        expected_singleton_paths = {
            "selected_named_weights": "checkpoints/weights.pt",
            "epoch_boundary_resume": "checkpoints/resume.pt",
            "ibm_om_persistent_deployment": "artifacts/ibm_om_deployment.pt",
        }
        if any(
            len(artifact_records[kind]) != 1
            or artifact_records[kind][0].get("path") != relative
            for kind, relative in expected_singleton_paths.items()
        ):
            raise RuntimeError(
                "Expected exact canonical checkpoint/deployment paths and "
                "exact checkpoint/deployment artifact paths for "
                f"{arm!r}."
            )
        population_records = {
            role: _validate_population_artifact(run_dir, role=role)
            for role in roles
        }
        expected_population_paths = {
            f"artifacts/ibm_om_population.{role}.npz" for role in roles
        }
        expected_receipt_paths = {
            f"artifacts/ibm_om_population.{role}.receipt.json" for role in roles
        }
        if (
            {
                str(record.get("path"))
                for record in artifact_records["ibm_om_array_population"]
            }
            != expected_population_paths
            or {
                str(record.get("path"))
                for record in artifact_records[
                    "ibm_om_array_population_receipt"
                ]
            }
            != expected_receipt_paths
        ):
            raise RuntimeError(
                f"Expected exact training/selection population roles for {arm!r}."
            )

        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"Expected final metrics for {arm!r}.")
        selected_epoch, selected_validation = _validate_result_metric_bindings(
            result=result,
            metrics=metrics,
            epoch_records=epoch_records,
        )
        fingerprints = metrics.get("ibm_om_population_fingerprints")
        if (
            metrics.get("encoding") != "differential"
            or metrics.get("device_model_sha256") != DEVICE_MODEL_SHA256
            or not isinstance(fingerprints, Mapping)
            or dict(fingerprints)
            != {role: EXPECTED_POPULATION_FINGERPRINT for role in roles}
        ):
            raise RuntimeError(f"Expected differential fixed-array result for {arm!r}.")
        selection = metrics.get("selection_device_programming")
        if not isinstance(selection, Mapping):
            raise RuntimeError(f"Expected selection report for {arm!r}.")
        selection_mapping = selection.get("target_mapping_report")
        if not isinstance(selection_mapping, Mapping):
            raise RuntimeError(f"Expected selection pair mapping for {arm!r}.")
        _validate_authoritative_mapping_report(selection_mapping)
        _validate_pulse_canary(selection, mapping_report=selection_mapping)
        selected_repeats = selected_validation.get("repeat_device_programming")
        if (
            not isinstance(selected_repeats, list)
            or len(selected_repeats) != 1
            or selected_repeats[0] != selection
            or selection.get("population_receipt_sha256")
            != population_records["selection"]["receipt_sha256"]
        ):
            raise RuntimeError(
                f"Expected selected programming/report/receipt parity for {arm!r}."
            )

        selected, selected_weights_sha256 = (
            _validate_selected_checkpoint_and_resume(
                run_dir=run_dir,
                metrics=metrics,
                selected_epoch=selected_epoch,
                selected_validation=selected_validation,
            )
        )
        selected_metadata = selected.get("metadata")
        if not isinstance(selected_metadata, Mapping):
            raise RuntimeError(f"Expected selected checkpoint metadata for {arm!r}.")

        deployment_path = run_dir / "artifacts" / "ibm_om_deployment.pt"
        deployment = _load_torch_weights_only(deployment_path)
        if (
            not isinstance(deployment, Mapping)
            or deployment.get("endpoint_application_policy")
            != ENDPOINT_APPLICATION_POLICY
            or deployment.get("population_fingerprint")
            != EXPECTED_POPULATION_FINGERPRINT
            or deployment.get("device_model_sha256") != DEVICE_MODEL_SHA256
            or deployment.get("report") != selection
        ):
            raise RuntimeError(f"Expected exact persistent deployment for {arm!r}.")
        _validate_pulse_deployment_integrity(
            deployment,
            selection_report=selection,
            selected_metadata=selected_metadata,
            selected_weights_sha256=selected_weights_sha256,
            selected_epoch=selected_epoch,
        )
        training_audit: Mapping[str, Any] | None = None
        if arm == "train-exact-bounds-pair-hwa":
            if metrics.get("ibm_om_population_fingerprints", {}).get(
                "training"
            ) != EXPECTED_POPULATION_FINGERPRINT:
                raise RuntimeError("Expected HWA training population fingerprint.")
            training = metrics.get("training_device_programming")
            if not isinstance(training, Mapping):
                raise RuntimeError("Expected final compact HWA report.")
            training_mapping = training.get("target_mapping_report")
            if not isinstance(training_mapping, Mapping):
                raise RuntimeError("Expected final compact mapping report.")
            _validate_authoritative_mapping_report(training_mapping)
            digest = _validate_compact_training_canary(
                training,
                mapping_report=training_mapping,
                expected_fallback_indices_sha256=fallback_indices_sha256,
            )
            sidecar_records = artifact_records.get(
                "ibm_om_training_device_programming",
                [],
            )
            expected_sidecar_path = (
                "artifacts/ibm_om_training_device_programming.json"
            )
            if (
                len(sidecar_records) != 1
                or sidecar_records[0].get("path") != expected_sidecar_path
                or _read_json(run_dir / expected_sidecar_path) != training
                or training.get("population_receipt_sha256")
                != population_records["training"]["receipt_sha256"]
            ):
                raise RuntimeError(
                    "Expected exact compact-training sidecar and receipt parity."
                )
            training_audit = {
                "population_fingerprint": training.get("population_fingerprint"),
                "endpoint_seed": training.get("endpoint_seed"),
                "execution_detail": training.get("execution_detail"),
                "compact_endpoint_devices": training.get(
                    "compact_endpoint_devices"
                ),
                "pulse_resolved_fallback_devices": training.get(
                    "pulse_resolved_fallback_devices"
                ),
                "fallback_policy": training.get(
                    "pulse_resolved_fallback",
                    {},
                ).get("policy"),
                "fallback_indices_sha256": digest,
            }
        elif (
            metrics.get("ibm_om_population_fingerprints", {}).get("training")
            is not None
            or metrics.get("training_device_programming") is not None
            or artifact_records.get("ibm_om_training_device_programming", [])
        ):
            raise RuntimeError(
                "Expected no compact training population/report in clean arm."
            )

        selected_path = run_dir / "checkpoints" / "weights.pt"
        records[arm] = {
            "run_dir": str(run_dir.resolve()),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "status_sha256": sha256_file(run_dir / "status.json"),
            "result_sha256": sha256_file(run_dir / "result.json"),
            "metrics_sha256": sha256_file(run_dir / "metrics.jsonl"),
            "selected_checkpoint_sha256": sha256_file(selected_path),
            "deployment_sha256": sha256_file(deployment_path),
            "populations": population_records,
            "selection_population_fingerprint": selection.get(
                "population_fingerprint"
            ),
            "training_report": training_audit,
        }
    return records


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
            "Expected --cuda-visible-devices to be comma-separated "
            "non-negative integer device indices."
        )
    study = _validate_prepared_study(study_dir)

    attempt_dir = study_dir / "launch" / _attempt_id()
    attempt_dir.mkdir(parents=True, exist_ok=False)
    launcher_python = Path(sys.executable).resolve()
    base_contract = {
        "schema": "ebl.mnist_ibm_om_differential_pair.launch_contract",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "formal_evidence": True,
        "fresh_attempt_only": True,
        "within_eight_matched_claim_only": True,
        "four_device_v2_comparison": "unmatched_mechanistic_only",
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
        "inputs": {
            "source_base_checkpoint": {
                "path": str(SOURCE_BASE_CHECKPOINT.resolve()),
                "sha256": SOURCE_BASE_CHECKPOINT_SHA256,
                "size_bytes": SOURCE_BASE_CHECKPOINT.stat().st_size,
            },
            "relu_teacher_checkpoint": {
                "path": str(RELU_TEACHER_CHECKPOINT.resolve()),
                "sha256": RELU_TEACHER_CHECKPOINT_SHA256,
                "size_bytes": RELU_TEACHER_CHECKPOINT.stat().st_size,
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
        "preflight": {
            "required_before_native_launch": True,
            "canonicalization": {
                "source_sha256": SOURCE_BASE_CHECKPOINT_SHA256,
                "teacher_sha256": RELU_TEACHER_CHECKPOINT_SHA256,
                "operation": "metadata_only_tensor_identical_copy",
                "exact_ordered_replay_examples": CANONICAL_REPLAY_EXAMPLES,
            },
            "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
            "cells": EXPECTED_TOTAL_DEVICES,
            "grouping": "differential_pair",
            "group_size": 2,
            "groups": EXPECTED_PAIR_COUNT,
            "empty_groups": EXPECTED_EMPTY_PAIR_COUNT,
            "strictly_positive_nonempty_inner_span": True,
            "compact_partition": {
                "compact_endpoint_devices": EXPECTED_COMPACT_ENDPOINT_DEVICES,
                "pulse_resolved_fallback_devices": (
                    EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
                ),
                "fallback_policy": COMPACT_FALLBACK_POLICY,
                "endpoint_generation_policy": (
                    COMPACT_ENDPOINT_GENERATION_POLICY
                ),
            },
            "pulse_canary": {
                "endpoint_seed": SELECTION_ENDPOINT_SEED,
                "minimum_success_fraction": 0.99,
                "maximum_budget_exhausted_fraction": 0.01,
                "maximum_nonfinite": 0,
            },
            "network_canary": {
                "programming_contexts": 1,
                "data_seed": 42,
                "examples": NETWORK_PREFLIGHT_EXAMPLES,
                "minimum_student_accuracy": (
                    MINIMUM_NETWORK_PREFLIGHT_STUDENT_ACCURACY
                ),
                "minimum_teacher_agreement": (
                    MINIMUM_NETWORK_PREFLIGHT_TEACHER_AGREEMENT
                ),
            },
            "strict_serialization_round_trip": True,
        },
        "expected_epochs_per_arm": EXPECTED_EPOCHS_PER_ARM,
        "heartbeat_seconds": HEARTBEAT_SECONDS,
        "created_at": _utc_now(),
    }
    atomic_write_json(attempt_dir / "contract.json", base_contract)
    atomic_write_json(
        attempt_dir / "watchdog_contract.json",
        {
            "schema": "ebl.local_experiment_watchdog",
            "schema_version": 1,
            "heartbeat_path": str(attempt_dir / "heartbeat.json"),
            "heartbeat_seconds": HEARTBEAT_SECONDS,
            "progress_evidence": [
                "native status.json",
                "metrics.jsonl line and byte growth",
                "weights.pt byte growth",
                "resume.pt byte growth",
                "launcher stdout/stderr byte growth",
            ],
            "expected_arms": list(TRAINING_ARM_CONFIGS),
            "created_at": _utc_now(),
        },
    )
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
            (
                population_fingerprint,
                fallback_indices_sha256,
                canonical_sha256,
            ) = _validate_preflight_report(preflight)
        except (RuntimeError, TypeError, ValueError) as error:
            preflight["gate"] = {
                "passed": False,
                "validated_at": _utc_now(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            atomic_write_json(attempt_dir / "preflight_report.json", preflight)
            raise
        canonical_path = Path(
            str(preflight["inputs"]["canonical_base_checkpoint"]["path"])
        ).resolve()
        if (
            canonical_path.parent != (attempt_dir / "preflight").resolve()
            or not canonical_path.is_file()
            or sha256_file(canonical_path) != canonical_sha256
        ):
            raise RuntimeError(
                "Expected the eligible canonical checkpoint inside this "
                "immutable preflight attempt."
            )
        preflight["gate"] = {
            "passed": True,
            "validated_at": _utc_now(),
            "population_fingerprint": population_fingerprint,
            "fallback_indices_sha256": fallback_indices_sha256,
            "canonical_checkpoint_sha256": canonical_sha256,
        }
        atomic_write_json(attempt_dir / "preflight_report.json", preflight)
        atomic_write_json(
            attempt_dir / "preflight_result.json",
            {
                "status": "passed",
                "population_fingerprint": population_fingerprint,
                "fallback_indices_sha256": fallback_indices_sha256,
                "canonical_checkpoint_sha256": canonical_sha256,
                "finished_at": _utc_now(),
            },
        )
        if stop_requested:
            raise RuntimeError(
                "Launcher stop requested during preflight; native arms not started."
            )

        commands = _commands(
            python=launcher_python,
            study_dir=study_dir,
            canonical_checkpoint_path=canonical_path,
        )
        execution_contract = {
            **base_contract,
            "schema": "ebl.mnist_ibm_om_differential_pair.execution_contract",
            "preflight_report_sha256": sha256_file(
                attempt_dir / "preflight_report.json"
            ),
            "canonical_checkpoint": {
                "path": str(canonical_path),
                "sha256": canonical_sha256,
            },
            "expected_population_fingerprint": population_fingerprint,
            "expected_fallback_indices_sha256": fallback_indices_sha256,
            "commands": commands,
            "created_at": _utc_now(),
        }
        atomic_write_json(attempt_dir / "execution_contract.json", execution_contract)
        atomic_write_json(
            attempt_dir / "heartbeat.json",
            {
                "updated_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "phase": "preflight_passed",
                "population_fingerprint": population_fingerprint,
                "canonical_checkpoint_sha256": canonical_sha256,
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
                    "canonical_checkpoint_sha256": canonical_sha256,
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
                    "canonical_checkpoint_sha256": canonical_sha256,
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
        post_audit: Mapping[str, Any] | None = None
        try:
            if summary_process.returncode != 0 or summary is None:
                raise RuntimeError("Study summarization did not produce a summary.")
            _validate_completed_summary(summary)
            post_audit = _validate_post_run_audit(
                study_dir,
                canonical_sha256=canonical_sha256,
                fallback_indices_sha256=fallback_indices_sha256,
            )
            atomic_write_json(
                attempt_dir / "post_run_artifact_audit.json",
                {
                    "status": "passed",
                    "canonical_checkpoint_sha256": canonical_sha256,
                    "expected_population_fingerprint": population_fingerprint,
                    "expected_fallback_indices_sha256": (
                        fallback_indices_sha256
                    ),
                    "arms": post_audit,
                    "validated_at": _utc_now(),
                },
            )
        except (RuntimeError, TypeError, ValueError) as error:
            phase_error = str(error)

        success = (
            not stop_requested
            and all(value == 0 for value in returncodes.values())
            and summary_process.returncode == 0
            and phase_error is None
        )
        final_snapshots = [
            _arm_snapshot(
                arm,
                study_dir=study_dir,
                process=processes[arm],
                stdout_path=stdout_paths[arm],
                stderr_path=stderr_paths[arm],
            )
            for arm in TRAINING_ARM_CONFIGS
        ]
        atomic_write_json(
            attempt_dir / "heartbeat.json",
            {
                "updated_at": _utc_now(),
                "elapsed_seconds": time.monotonic() - started,
                "phase": "complete" if success else "failed",
                "stop_requested": stop_requested,
                "canonical_checkpoint_sha256": canonical_sha256,
                "arms": final_snapshots,
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
                "canonical_checkpoint_sha256": canonical_sha256,
                "population_fingerprint": population_fingerprint,
                "post_run_artifact_audit_passed": post_audit is not None,
                "phase_validation_error": phase_error,
                "arms": final_snapshots,
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


if __name__ == "__main__":  # pragma: no cover - launcher entry point
    raise SystemExit(main())
