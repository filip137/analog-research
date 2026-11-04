from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import experiments.mnist_relu_drn.ibm_om_differential_pair_pilot_launcher as launcher
from ebl.runtime.store import atomic_write_json, sha256_file
from experiments.mnist_relu_drn.ibm_om_differential_pair_pilot_launcher import (
    ASSIGNMENT_SEED,
    CANONICAL_REPLAY_EXAMPLES,
    COMMON_WINDOW_MARGIN_FRACTION,
    COMPACT_ENDPOINT_GENERATION_POLICY,
    COMPACT_EXECUTION_DETAIL,
    COMPACT_FALLBACK_POLICY,
    CONFIG_ROOT,
    CORRUPTION_POLICY,
    DEVICE_MODEL_SHA256,
    ENDPOINT_APPLICATION_POLICY,
    EXPECTED_BINDING_KEYS,
    EXPECTED_COMPACT_ENDPOINT_DEVICES,
    EXPECTED_EMPTY_PAIR_COUNT,
    EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES,
    EXPECTED_EPOCHS_PER_ARM,
    EXPECTED_PAIR_COUNT,
    EXPECTED_POPULATION_FINGERPRINT,
    EXPECTED_TOTAL_DEVICES,
    NETWORK_PREFLIGHT_BATCHES,
    NETWORK_PREFLIGHT_EXAMPLES,
    POPULATION_ROLES,
    RELU_TEACHER_CHECKPOINT_SHA256,
    SELECTION_ENDPOINT_SEED,
    SOURCE_BASE_CHECKPOINT_SHA256,
    STUDY_ID,
    STUDY_PLAN,
    TARGET_MAPPING,
    TRAINING_ARM_CONFIGS,
    TRAINING_ENDPOINT_SEED,
    _canonicalize_base_checkpoint,
    _commands,
    _evaluate_apparent_forward_canary,
    _expected_modifier_parameters,
    _result_artifacts,
    _torch_payload_equal,
    _validate_authoritative_mapping_report,
    _validate_canonicalization_report,
    _validate_completed_summary,
    _validate_compact_training_canary,
    _validate_config_contracts,
    _validate_native_epoch_records,
    _validate_native_manifest_inputs,
    _validate_post_run_audit,
    _validate_preflight_report,
    _validate_prepared_study,
    _validate_pulse_deployment_integrity,
    _validate_pulse_canary,
    _validate_result_metric_bindings,
    _validate_selected_checkpoint_and_resume,
    _validate_serialized_canary_artifacts,
)
from ebl.lifecycle.study import prepare_study


_SHA = "a" * 64


def _frozen_inputs(
    tmp_path: Path,
) -> tuple[Path, str, Path, str, Path, str]:
    source = tmp_path / "clean-differential.pt"
    source.write_bytes(b"clean differential fixture")
    teacher = tmp_path / "relu-teacher.pt"
    teacher.write_bytes(b"relu teacher fixture")
    device_model = tmp_path / "ibm-om-model.json"
    device_model.write_bytes(b'{"fixture":"device model"}\n')
    return (
        source,
        sha256_file(source),
        teacher,
        sha256_file(teacher),
        device_model,
        sha256_file(device_model),
    )


def _prepared_study(tmp_path: Path) -> Path:
    return prepare_study(STUDY_PLAN, tmp_path / "results")


def _summary(minimum: float, mean: float, maximum: float) -> dict[str, float]:
    return {"minimum": minimum, "mean": mean, "maximum": maximum}


def _mapping_report() -> dict[str, object]:
    shapes = ((1568, 100), (100, 20))
    parameters: dict[str, object] = {}
    parameter_pairs: dict[str, object] = {}
    for pair_index, shape in enumerate(shapes):
        plus_key = EXPECTED_BINDING_KEYS[2 * pair_index]
        minus_key = EXPECTED_BINDING_KEYS[2 * pair_index + 1]
        pair_count = shape[0] * shape[1]
        empty = EXPECTED_EMPTY_PAIR_COUNT if pair_index == 0 else 0
        pair_record = {
            "pair_index": pair_index,
            "conductance_plus_key": plus_key,
            "conductance_minus_key": minus_key,
            "shape": list(shape),
            "devices": 2 * pair_count,
            "pair_count": pair_count,
            "common_window_empty_pair_count": empty,
            "common_window_empty_fraction": empty / pair_count,
            "corrupt_pair_count": 0,
            "published_corrupt_pair_count": 0,
            "raw_common_span": _summary(0.0, 0.7, 1.0),
            "inner_common_span": _summary(0.0, 0.35, 0.5),
            "nonempty_inner_common_span": _summary(0.001, 0.35, 0.5),
            "nonempty_pair_inner_common_span": _summary(0.001, 0.35, 0.5),
            "mapped_target_below_lower_bound_nonempty_pair": 0,
            "mapped_target_above_upper_bound_nonempty_pair": 0,
            "mapped_target_below_lower_bound_empty_pair": empty,
            "mapped_target_above_upper_bound_empty_pair": empty,
        }
        parameter_pairs[f"base.differential_pair.{pair_index}"] = pair_record
        for role, key, peer in (
            ("conductance_plus", plus_key, minus_key),
            ("conductance_minus", minus_key, plus_key),
        ):
            empty_below = empty if role == "conductance_plus" else 0
            empty_above = empty if role == "conductance_minus" else 0
            parameters[key] = {
                "differential_role": role,
                "paired_parameter": peer,
                "pair_index": pair_index,
                "shape": list(shape),
                "devices": pair_count,
                "pair_count": pair_count,
                "common_window_empty_pair_count": empty,
                "raw_common_span": _summary(0.0, 0.7, 1.0),
                "inner_common_span": _summary(0.0, 0.35, 0.5),
                "nonempty_inner_common_span": _summary(0.001, 0.35, 0.5),
                "nonempty_pair_inner_common_span": _summary(0.001, 0.35, 0.5),
                "mapped_target_below_lower_bound_nonempty_pair": 0,
                "mapped_target_above_upper_bound_nonempty_pair": 0,
                "mapped_target_below_lower_bound_empty_pair": empty_below,
                "mapped_target_above_upper_bound_empty_pair": empty_above,
            }
    return {
        "target_mapping": TARGET_MAPPING,
        "common_window_margin_fraction": COMMON_WINDOW_MARGIN_FRACTION,
        "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
        "corruption_policy": CORRUPTION_POLICY,
        "devices": EXPECTED_TOTAL_DEVICES,
        "global_target_support": {
            "below_lower_bound": 137946,
            "above_upper_bound": 23,
            "inside_bounds": 179631,
        },
        "global_target_outside_0_1": 0,
        "parameters": parameters,
        "parameter_pairs": parameter_pairs,
        "common_window_grouping": "differential_pair",
        "common_window_group_size": 2,
        "common_window_group_count": EXPECTED_PAIR_COUNT,
        "common_window_empty_group_count": EXPECTED_EMPTY_PAIR_COUNT,
        "corrupt_group_count": 0,
        "published_corrupt_group_count": 1200,
        "differential_pair_binding": {
            "policy": "canonical_adjacent_plus_minus_same_coordinate",
            "catalog_order": list(EXPECTED_BINDING_KEYS),
            "pairs": [
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
            ],
        },
        "pair_count": EXPECTED_PAIR_COUNT,
        "common_window_empty_pair_count": EXPECTED_EMPTY_PAIR_COUNT,
        "common_window_empty_fraction": (
            EXPECTED_EMPTY_PAIR_COUNT / EXPECTED_PAIR_COUNT
        ),
        "corrupt_pair_count": 0,
        "published_corrupt_pair_count": 1200,
        "raw_common_span": _summary(0.0, 0.702, 1.0),
        "inner_common_span": _summary(0.0, 0.351, 0.5),
        "nonempty_inner_common_span": _summary(0.001, 0.351, 0.5),
        "nonempty_pair_inner_common_span": _summary(0.001, 0.351, 0.5),
        "mapped_target_below_lower_bound_nonempty_group": 0,
        "mapped_target_above_upper_bound_nonempty_group": 0,
        "mapped_target_below_lower_bound_empty_group": 8,
        "mapped_target_above_upper_bound_empty_group": 8,
        "mapped_target_below_lower_bound_nonempty_pair": 0,
        "mapped_target_above_upper_bound_nonempty_pair": 0,
        "mapped_target_below_lower_bound_empty_pair": 8,
        "mapped_target_above_upper_bound_empty_pair": 8,
        "mapped_target_support": {
            "below_lower_bound": 8,
            "above_upper_bound": 8,
            "inside_bounds": EXPECTED_COMPACT_ENDPOINT_DEVICES,
        },
    }


def _canonicalization_report() -> dict[str, object]:
    metrics = {
        "examples": CANONICAL_REPLAY_EXAMPLES,
        "student_accuracy": 0.9716,
        "teacher_agreement": 0.9984,
        "kl_teacher_student": 0.000685,
    }
    return {
        "schema": "ebl.mnist_ibm_om_differential_pair.canonical_replay",
        "schema_version": 1,
        "operation": "metadata_only_tensor_identical_copy",
        "source_checkpoint": {"sha256": SOURCE_BASE_CHECKPOINT_SHA256},
        "canonical_checkpoint": {"sha256": _SHA},
        "teacher_checkpoint": {"sha256": RELU_TEACHER_CHECKPOINT_SHA256},
        "catalog_order": list(EXPECTED_BINDING_KEYS),
        "tensor_equality": "torch.equal_all_named_tensors",
        "unequal_tensor_keys": [],
        "current_metadata_validation": "passed",
        "source_metrics": deepcopy(metrics),
        "canonical_metrics": metrics,
        "metrics_exactly_equal": True,
        "teacher_mapping_invoked": False,
        "conductance_remapping_invoked": False,
        "optimizer_updates": 0,
        "retraining_epochs": 0,
        "artifacts": {
            "canonical_checkpoint": {"sha256": _SHA},
            "canonical_metadata": {"sha256": "b" * 64},
            "canonical_replay": {"sha256": "c" * 64},
        },
    }


def _fallback_indices() -> list[int]:
    return list(range(EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES))


def _fallback_digest(indices: list[int] | None = None) -> str:
    values = _fallback_indices() if indices is None else indices
    return sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _compact_report(mapping: dict[str, object]) -> dict[str, object]:
    indices = _fallback_indices()
    return {
        "execution": "compact_endpoint",
        "execution_detail": COMPACT_EXECUTION_DETAIL,
        "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
        "endpoint_seed": TRAINING_ENDPOINT_SEED,
        "assignment_seed": ASSIGNMENT_SEED,
        "corruption_policy": CORRUPTION_POLICY,
        "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
        "population_sampling_backend": "external_pinned_aihwkit_python",
        "device_model_sha256": DEVICE_MODEL_SHA256,
        "target_mapping": TARGET_MAPPING,
        "devices": EXPECTED_TOTAL_DEVICES,
        "compact_endpoint_devices": EXPECTED_COMPACT_ENDPOINT_DEVICES,
        "pulse_resolved_fallback_devices": (
            EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        ),
        "accepted": EXPECTED_TOTAL_DEVICES - 100,
        "success_fraction": (EXPECTED_TOTAL_DEVICES - 100) / EXPECTED_TOTAL_DEVICES,
        "failed_noncorrupt": 100,
        "corrupt": 0,
        "target_below_lower_bound": 8,
        "target_inside_bounds": EXPECTED_COMPACT_ENDPOINT_DEVICES,
        "target_above_upper_bound": 8,
        "target_mapping_report": deepcopy(mapping),
        "pulse_resolved_fallback": {
            "policy": COMPACT_FALLBACK_POLICY,
            "devices": EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES,
            "selection_indices": indices,
            "selection_indices_sha256": _fallback_digest(indices),
            "target_below_lower_bound": 8,
            "target_above_upper_bound": 8,
            "acceptance_window_unreachable": 16,
            "start_state": "sampled_fully_reset_bound",
            "controller": "adaptive",
            "maximum_program_pulses": 128,
            "random_stream_order": "after_compact_endpoint_sampling",
            "accepted": 2,
            "success_fraction": 0.125,
            "budget_exhausted": 14,
            "nonfinite": 0,
            "pulse_count": {
                "mean": 112.0,
                "median": 128.0,
                "maximum": 128,
                "set_mean": 100.0,
                "reset_mean": 12.0,
            },
            "verify_count_mean": 113.0,
            "reversal_count_mean": 1.0,
            "saturated": 16,
            "endpoint_clipped": 0,
        },
    }


def _pulse_report(mapping: dict[str, object]) -> dict[str, object]:
    exhausted = 2000
    return {
        "execution": "pulse_resolved",
        "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
        "endpoint_seed": SELECTION_ENDPOINT_SEED,
        "assignment_seed": ASSIGNMENT_SEED,
        "corruption_policy": CORRUPTION_POLICY,
        "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
        "population_sampling_backend": "external_pinned_aihwkit_python",
        "device_model_sha256": DEVICE_MODEL_SHA256,
        "target_mapping": TARGET_MAPPING,
        "devices": EXPECTED_TOTAL_DEVICES,
        "accepted": EXPECTED_TOTAL_DEVICES - exhausted,
        "corrupt": 0,
        "nonfinite": 0,
        "budget_exhausted": exhausted,
        "target_below_lower_bound": 8,
        "target_inside_bounds": EXPECTED_COMPACT_ENDPOINT_DEVICES,
        "target_above_upper_bound": 8,
        "success_fraction": (EXPECTED_TOTAL_DEVICES - exhausted)
        / EXPECTED_TOTAL_DEVICES,
        "target_mapping_report": deepcopy(mapping),
    }


def _preflight_report() -> dict[str, object]:
    mapping = _mapping_report()
    canonicalization = _canonicalization_report()
    request = {
        "preset": "reram_array_om",
        "assignment_seed": ASSIGNMENT_SEED,
        "corruption_policy": CORRUPTION_POLICY,
        "binding_keys": list(EXPECTED_BINDING_KEYS),
        "binding_shapes": [[1568, 100], [1568, 100], [100, 20], [100, 20]],
        "required_aihwkit_version": "1.1.0",
    }
    artifact_names = {
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
    artifacts: dict[str, dict[str, object]] = {
        name: {"sha256": "f" * 64} for name in artifact_names
    }
    artifacts["canonical_checkpoint"]["sha256"] = _SHA
    artifacts["compact_deployment"].update(
        {
            "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            "endpoint_generation_policy": COMPACT_ENDPOINT_GENERATION_POLICY,
        }
    )
    artifacts["pulse_deployment"]["endpoint_application_policy"] = (
        ENDPOINT_APPLICATION_POLICY
    )
    return {
        "schema": "ebl.mnist_ibm_om_differential_pair.mapping_preflight",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "authoritative_entry_point": (
            "methods.ibm_om.modifier."
            "IbmReramHwaParameterModifier.preflight_target_mapping"
        ),
        "inputs": {
            "source_base_checkpoint": {"sha256": SOURCE_BASE_CHECKPOINT_SHA256},
            "canonical_base_checkpoint": {"sha256": _SHA},
            "relu_teacher_checkpoint": {
                "sha256": RELU_TEACHER_CHECKPOINT_SHA256
            },
            "ibm_om_device_model": {"sha256": DEVICE_MODEL_SHA256},
        },
        "configs": {
            arm: {"sha256": sha256_file(CONFIG_ROOT / filename)}
            for arm, filename in TRAINING_ARM_CONFIGS.items()
        },
        "canonicalization": canonicalization,
        "selection_modifier_parameters": _expected_modifier_parameters(
            execution="pulse_resolved"
        ),
        "training_modifier_parameters": _expected_modifier_parameters(
            execution="compact_endpoint"
        ),
        "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
        "mapping_report": mapping,
        "training_mapping_report": deepcopy(mapping),
        "selection_population_receipt": {
            "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
            "aihwkit_version": "1.1.0",
            "request": deepcopy(request),
        },
        "training_population_receipt": {
            "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
            "aihwkit_version": "1.1.0",
            "request": deepcopy(request),
        },
        "mapped_targets": {
            "count": EXPECTED_TOTAL_DEVICES,
            "finite_count": EXPECTED_TOTAL_DEVICES,
            "outside_0_1_count": 0,
            "minimum": 0.105,
            "maximum": 0.832,
            "sha256": "d" * 64,
        },
        "compact_training_canary_report": _compact_report(mapping),
        "pulse_resolved_canary_report": _pulse_report(mapping),
        "network_preflight": {
            "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            "forward_endpoint": "aihwkit_apparent_endpoint",
            "persistent_endpoint_role": "hidden_update_state",
            "programming_contexts": 1,
            "cohort": {
                "split": "validation",
                "data_seed": 42,
                "validation_points": 5000,
                "batch_size": 16,
                "maximum_batches": NETWORK_PREFLIGHT_BATCHES,
                "examples": NETWORK_PREFLIGHT_EXAMPLES,
                "shuffle": False,
            },
            "metrics": {
                "examples": NETWORK_PREFLIGHT_EXAMPLES,
                "student_accuracy": 0.45,
                "teacher_agreement": 0.5,
                "kl_teacher_student": 0.2,
            },
        },
        "serialization_canary": {
            "json_round_trip": "passed",
            "checkpoint_round_trip": "passed",
            "allow_nan": False,
            "pickle_unsafe_custom_objects": False,
            "compact_and_fallback_mask_partition": "passed",
            "fallback_indices_mask_report_parity": "passed",
            "finite_apparent_and_persistent_endpoints": "passed",
        },
        "artifacts": artifacts,
    }


def _serialized_canary_fixture(tmp_path: Path) -> dict[str, Any]:
    import torch

    mapping = _mapping_report()
    compact_report = _compact_report(mapping)
    pulse_report = _pulse_report(mapping)
    network_report = _preflight_report()["network_preflight"]
    fallback_mask = torch.zeros(EXPECTED_TOTAL_DEVICES, dtype=torch.bool)
    fallback_mask[:EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES] = True
    compact_mask = ~fallback_mask
    fallback_indices = torch.where(fallback_mask)[0].to(dtype=torch.int64)
    compact_deployment = {
        "compact_endpoint_mask": compact_mask,
        "pulse_resolved_fallback_mask": fallback_mask,
        "pulse_resolved_fallback": {
            "policy": COMPACT_FALLBACK_POLICY,
            "selection_indices": fallback_indices,
            "selection_indices_sha256": _fallback_digest(),
        },
        "apparent_endpoint": torch.linspace(
            0.0,
            1.0,
            EXPECTED_TOTAL_DEVICES,
        ),
        "persistent_endpoint": torch.linspace(
            1.0,
            0.0,
            EXPECTED_TOTAL_DEVICES,
        ),
    }
    pulse_deployment = {
        "apparent_endpoint": torch.zeros(EXPECTED_TOTAL_DEVICES),
        "persistent_endpoint": torch.ones(EXPECTED_TOTAL_DEVICES),
    }
    return {
        "compact_report": compact_report,
        "compact_report_path": tmp_path / "compact.json",
        "compact_deployment": compact_deployment,
        "compact_deployment_path": tmp_path / "compact.pt",
        "pulse_report": pulse_report,
        "pulse_report_path": tmp_path / "pulse.json",
        "pulse_deployment": pulse_deployment,
        "pulse_deployment_path": tmp_path / "pulse.pt",
        "network_report": network_report,
        "network_path": tmp_path / "network.json",
    }


def _write_serialized_canary_fixture(fixture: dict[str, Any]) -> None:
    import torch

    for record_key, path_key in (
        ("compact_report", "compact_report_path"),
        ("pulse_report", "pulse_report_path"),
        ("network_report", "network_path"),
    ):
        atomic_write_json(fixture[path_key], fixture[record_key])
    torch.save(
        fixture["compact_deployment"],
        fixture["compact_deployment_path"],
    )
    torch.save(
        fixture["pulse_deployment"],
        fixture["pulse_deployment_path"],
    )


def _write_epoch_records(run_dir: Path) -> None:
    records = [{"mode": "initialization"}]
    records.extend(
        {
            "mode": "train",
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "global_step": 3438 * (epoch + 1),
        }
        for epoch in range(EXPECTED_EPOCHS_PER_ARM)
    )
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _artifact(path: Path, *, run_dir: Path, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path.relative_to(run_dir)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _install_canonicalization_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: str | None = None,
) -> dict[str, Any]:
    import torch

    import ebl.runtime.mnist_relu_drn.components as components
    import ebl.runtime.mnist_relu_drn.runtime as runtime
    import methods.checkpoint as checkpoint

    values = {
        key: torch.tensor([float(index), float(index) + 0.25])
        for index, key in enumerate(EXPECTED_BINDING_KEYS)
    }
    source_metadata = {
        "encoding": "differential",
        "experiment_id": "mnist_relu_drn_kd.v1",
        "teacher_sha256": RELU_TEACHER_CHECKPOINT_SHA256,
        "selection_epoch": 19,
        "fixed_logit_gain": 2.5,
        "mapping": {"source": "already-trained-ideal-differential"},
        "selection_metric": "student_accuracy",
        "selection_value": 0.9716,
        "selection_student_accuracy": 0.9716,
        "selection_teacher_agreement": 0.9984,
        "selection_weight_modifier": {"type": "legacy"},
    }
    state: dict[str, Any] = {
        "builds": [],
        "saved": None,
        "metadata_validations": 0,
        "evaluations": 0,
    }

    def build_student_stack(spec: object, *, enable_measured: bool):
        bindings = [
            SimpleNamespace(key=key, state=torch.zeros_like(value))
            for key, value in values.items()
        ]
        stack = SimpleNamespace(
            bundle=SimpleNamespace(
                catalog=SimpleNamespace(checkpointed=bindings),
            ),
            cost=SimpleNamespace(gain=1.0),
        )
        state["builds"].append(stack)
        assert enable_measured is False
        return stack

    def load_named_weights(path: Path, catalog: object):
        is_canonical = path.name == "canonical_base_weights.pt"
        payload = state["saved"] if is_canonical else None
        tensors = payload["tensors"] if payload is not None else values
        for binding in catalog.checkpointed:
            binding.state.copy_(tensors[binding.key])
        if is_canonical and failure == "tensor":
            catalog.checkpointed[0].state.add_(1.0)
        metadata = payload["metadata"] if payload is not None else source_metadata
        return SimpleNamespace(metadata=deepcopy(metadata))

    def save_named_weights(
        path: Path,
        catalog: object,
        *,
        metadata: dict[str, Any],
    ) -> None:
        tensors = {
            binding.key: binding.state.detach().cpu().clone()
            for binding in catalog.checkpointed
        }
        state["saved"] = {
            "metadata": deepcopy(metadata),
            "tensors": tensors,
        }
        torch.save(
            {
                "schema": "drn.named-weights",
                "metadata": metadata,
                "tensors": tensors,
            },
            path,
        )

    def model_checkpoint_metadata(
        stack: object,
        *,
        spec: object,
        teacher_sha256: str,
        mapping: object,
        deployment_source: object,
        device_model_sha256: object,
    ) -> dict[str, Any]:
        return {
            "experiment_id": "mnist_relu_drn_kd.v1",
            "encoding": "differential",
            "teacher_sha256": teacher_sha256,
            "fixed_logit_gain": stack.cost.gain,
            "mapping": mapping,
        }

    def validate_checkpoint_metadata(*args: object, **kwargs: object) -> None:
        state["metadata_validations"] += 1
        if failure == "metadata":
            raise RuntimeError("current metadata rejected")

    exact_metrics = {
        "examples": CANONICAL_REPLAY_EXAMPLES,
        "student_accuracy": 0.9716,
        "teacher_agreement": 0.9984,
        "kl_teacher_student": 0.000685,
    }

    def evaluate(*args: object, **kwargs: object) -> dict[str, Any]:
        state["evaluations"] += 1
        metrics = deepcopy(exact_metrics)
        if failure == "replay" and state["evaluations"] == 2:
            metrics["teacher_agreement"] -= 0.0001
        if failure == "quality":
            metrics["student_accuracy"] = 0.5
        return metrics

    monkeypatch.setattr(components, "build_student_stack", build_student_stack)
    monkeypatch.setattr(checkpoint, "load_named_weights", load_named_weights)
    monkeypatch.setattr(checkpoint, "save_named_weights", save_named_weights)
    monkeypatch.setattr(
        runtime,
        "_model_checkpoint_metadata",
        model_checkpoint_metadata,
    )
    monkeypatch.setattr(
        runtime,
        "_validate_checkpoint_metadata",
        validate_checkpoint_metadata,
    )
    monkeypatch.setattr(runtime, "_amplification_index_report", lambda stack: {})
    monkeypatch.setattr(runtime, "_evaluate", evaluate)
    return state


def _write_post_run_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Path]]:
    import torch

    import ebl.runtime.mnist_relu_drn.ibm_om_deployment_decomposition as decomposition

    study_dir = tmp_path / STUDY_ID
    run_dirs: dict[str, Path] = {}

    def validate_population(run_dir: Path, *, role: str) -> dict[str, Any]:
        population = run_dir / "artifacts" / f"ibm_om_population.{role}.npz"
        receipt = (
            run_dir
            / "artifacts"
            / f"ibm_om_population.{role}.receipt.json"
        )
        assert population.is_file()
        assert receipt.is_file()
        return {
            "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
            "population_sha256": sha256_file(population),
            "receipt_sha256": sha256_file(receipt),
        }

    monkeypatch.setattr(
        launcher,
        "_validate_population_artifact",
        validate_population,
    )
    monkeypatch.setattr(
        decomposition,
        "validate_deployment_contract",
        lambda *args, **kwargs: ({}, object(), {}),
    )
    for arm, roles in POPULATION_ROLES.items():
        run_dir = study_dir / "runs" / arm / "attempt-001"
        run_dirs[arm] = run_dir
        artifacts_dir = run_dir / "artifacts"
        checkpoints_dir = run_dir / "checkpoints"
        artifacts_dir.mkdir(parents=True)
        checkpoints_dir.mkdir()

        population_artifacts: list[dict[str, Any]] = []
        receipt_digests: dict[str, str] = {}
        for role in roles:
            population_path = (
                artifacts_dir / f"ibm_om_population.{role}.npz"
            )
            population_path.write_bytes(f"{arm}:{role}:population".encode())
            receipt_path = (
                artifacts_dir / f"ibm_om_population.{role}.receipt.json"
            )
            atomic_write_json(receipt_path, {"role": role})
            receipt_digests[role] = sha256_file(receipt_path)
            population_artifacts.extend(
                [
                    _artifact(
                        population_path,
                        run_dir=run_dir,
                        kind="ibm_om_array_population",
                    ),
                    _artifact(
                        receipt_path,
                        run_dir=run_dir,
                        kind="ibm_om_array_population_receipt",
                    ),
                ]
            )

        mapping = _mapping_report()
        selection = _pulse_report(mapping)
        selection.update(
            {
                "population_receipt_sha256": receipt_digests["selection"],
                "acceptance_window_unreachable": 0,
                "saturated": 0,
                "endpoint_clipped": 0,
                "pulse_count": {
                    "maximum": 0,
                    "mean": 0.0,
                    "median": 0.0,
                    "set_mean": 0.0,
                    "reset_mean": 0.0,
                },
            }
        )
        base_result, base_metrics, epoch_records = _result_metric_fixture()
        epoch_records = tuple(deepcopy(epoch_records))
        metrics: dict[str, Any] = deepcopy(base_metrics)
        selected_epoch = int(metrics["selected"]["epoch"])
        selected_validation = deepcopy(
            epoch_records[selected_epoch + 1]["validation"]
        )
        selected_validation.update(
            {
                "selection_evaluation": "modifier_fixed_sequence_average",
                "selection_noise_repeats": 1,
                "repeat_device_programming": [selection],
            }
        )
        epoch_records[selected_epoch + 1]["validation"].update(
            selected_validation
        )
        metrics["selected"] = {"epoch": selected_epoch, **selected_validation}
        fingerprints: dict[str, str] = {
            "selection": EXPECTED_POPULATION_FINGERPRINT,
        }
        metrics.update(
            {
                "encoding": "differential",
                "device_model_sha256": DEVICE_MODEL_SHA256,
                "ibm_om_population_fingerprints": fingerprints,
                "selection_device_programming": selection,
                "training_device_programming": None,
            }
        )
        training_sidecar_path: Path | None = None
        if arm == "train-exact-bounds-pair-hwa":
            fingerprints["training"] = EXPECTED_POPULATION_FINGERPRINT
            training = _compact_report(mapping)
            training["population_receipt_sha256"] = receipt_digests["training"]
            metrics["training_device_programming"] = training
            training_sidecar_path = (
                artifacts_dir / "ibm_om_training_device_programming.json"
            )
            atomic_write_json(training_sidecar_path, training)

        selected_path = checkpoints_dir / "weights.pt"
        selected = {
            "schema": "drn.named-weights",
            "schema_version": 1,
            "metadata": {
                "teacher_sha256": RELU_TEACHER_CHECKPOINT_SHA256,
                "encoding": "differential",
                "device_model_sha256": DEVICE_MODEL_SHA256,
                "selection_metric": (
                    "validation_modifier_mean.kl_teacher_student"
                ),
                "selection_epoch": selected_epoch,
                "selection_value": selected_validation[
                    "kl_teacher_student"
                ],
                "selection_student_accuracy": selected_validation[
                    "student_accuracy"
                ],
                "selection_teacher_agreement": selected_validation[
                    "teacher_agreement"
                ],
            },
            "weights": {"w": torch.tensor([1.0, 2.0])},
        }
        torch.save(selected, selected_path)
        resume_path = checkpoints_dir / "resume.pt"
        torch.save(
            {
                "schema": "drn.epoch-boundary-resume",
                "schema_version": 3,
                "resume_capability": "exact",
                "epoch": EXPECTED_EPOCHS_PER_ARM,
                "global_step": 34380,
                "progress_state": {
                    "selected_epoch": selected_epoch,
                    "selected_validation": selected_validation,
                    "last_train": metrics["last_train"],
                    "last_validation": metrics["last_validation"],
                    "last_clean_validation": metrics[
                        "last_clean_validation"
                    ],
                },
                "selected_weights": selected,
            },
            resume_path,
        )
        deployment_path = artifacts_dir / "ibm_om_deployment.pt"
        deployment, _ = _pulse_deployment_fixture()
        deployment.update(
            {
                "report": selection,
                "target_mapping_report": mapping,
                "selected_weights_sha256": sha256_file(selected_path),
                "selected_epoch": selected_epoch,
            }
        )
        torch.save(deployment, deployment_path)

        artifact_records = [
            _artifact(
                selected_path,
                run_dir=run_dir,
                kind="selected_named_weights",
            ),
            _artifact(
                resume_path,
                run_dir=run_dir,
                kind="epoch_boundary_resume",
            ),
            _artifact(
                deployment_path,
                run_dir=run_dir,
                kind="ibm_om_persistent_deployment",
            ),
            *population_artifacts,
        ]
        if training_sidecar_path is not None:
            artifact_records.append(
                _artifact(
                    training_sidecar_path,
                    run_dir=run_dir,
                    kind="ibm_om_training_device_programming",
                )
            )

        atomic_write_json(run_dir / "status.json", {"status": "complete"})
        atomic_write_json(
            run_dir / "manifest.json",
            {
                "inputs": [
                    {
                        "role": "teacher_weights",
                        "sha256": RELU_TEACHER_CHECKPOINT_SHA256,
                    },
                    {"role": "weights", "sha256": _SHA},
                    {"role": "device_model", "sha256": DEVICE_MODEL_SHA256},
                ]
            },
        )
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in epoch_records),
            encoding="utf-8",
        )
        atomic_write_json(
            run_dir / "result.json",
            {
                **base_result,
                "metrics": metrics,
                "artifacts": artifact_records,
            },
        )
    return study_dir, run_dirs


def _rewrite_result_artifact_record(
    run_dir: Path,
    *,
    kind: str,
    path: Path,
) -> None:
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for record in result["artifacts"]:
        if record["kind"] == kind and record["path"] == str(
            path.relative_to(run_dir)
        ):
            record["sha256"] = sha256_file(path)
            record["size_bytes"] = path.stat().st_size
            break
    else:  # pragma: no cover - fixture assertion
        raise AssertionError((kind, path))
    atomic_write_json(result_path, result)


def _result_metric_fixture() -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
]:
    initial = {
        "examples": 5000,
        "student_accuracy": 0.4,
        "teacher_agreement": 0.5,
        "kl_teacher_student": 0.2,
    }
    initial_clean = {
        "examples": 5000,
        "student_accuracy": 0.97,
        "teacher_agreement": 0.99,
        "kl_teacher_student": 0.001,
    }
    records: list[dict[str, Any]] = [
        {
            "mode": "initialization",
            "validation": initial,
            "validation_clean": initial_clean,
        }
    ]
    for epoch in range(EXPECTED_EPOCHS_PER_ARM):
        records.append(
            {
                "mode": "train",
                "epoch": epoch,
                "completed_epochs": epoch + 1,
                "global_step": 3438 * (epoch + 1),
                "train": {"examples": 55000, "epoch": epoch},
                "validation": {
                    "examples": 5000,
                    "student_accuracy": 0.5 + epoch / 100,
                    "teacher_agreement": 0.6 + epoch / 100,
                    "kl_teacher_student": 0.1 - min(epoch, 4) / 1000,
                },
                "validation_clean": {
                    "examples": 5000,
                    "student_accuracy": 0.9,
                    "teacher_agreement": 0.91,
                    "kl_teacher_student": 0.02,
                },
            }
        )
    selected_epoch = 4
    selected_validation = records[selected_epoch + 1]["validation"]
    metrics = {
        "initial_validation": initial,
        "initial_clean_validation": initial_clean,
        "last_train": records[-1]["train"],
        "last_validation": records[-1]["validation"],
        "last_clean_validation": records[-1]["validation_clean"],
        "selected": {"epoch": selected_epoch, **selected_validation},
    }
    result = {
        "schema": "ebl.run",
        "schema_version": 1,
        "status": "complete",
        "experiment_id": "mnist_relu_drn_kd.v1",
        "error": None,
    }
    return result, metrics, tuple(records)


def _pulse_deployment_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    mapping = _mapping_report()
    report = _pulse_report(mapping)
    accepted_count = report["accepted"]
    accepted = torch.zeros(EXPECTED_TOTAL_DEVICES, dtype=torch.bool)
    accepted[:accepted_count] = True
    exhausted = ~accepted
    below = torch.zeros(EXPECTED_TOTAL_DEVICES, dtype=torch.bool)
    above = torch.zeros(EXPECTED_TOTAL_DEVICES, dtype=torch.bool)
    below[:8] = True
    above[8:16] = True
    inside = ~(below | above)
    zeros = torch.zeros(EXPECTED_TOTAL_DEVICES, dtype=torch.float32)
    counter = torch.zeros(EXPECTED_TOTAL_DEVICES, dtype=torch.int64)
    report.update(
        {
            "acceptance_window_unreachable": 0,
            "saturated": 0,
            "endpoint_clipped": 0,
            "pulse_count": {
                "maximum": 0,
                "mean": 0.0,
                "median": 0.0,
                "set_mean": 0.0,
                "reset_mean": 0.0,
            },
        }
    )
    return (
        {
            "schema": "ebl.ibm_reram.om_pulse_resolved_deployment",
            "schema_version": 1,
            "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
            "device_model_sha256": DEVICE_MODEL_SHA256,
            "selected_weights_sha256": _SHA,
            "selected_epoch": 9,
            "binding_keys": EXPECTED_BINDING_KEYS,
            "binding_shapes": (
                (1568, 100),
                (1568, 100),
                (100, 20),
                (100, 20),
            ),
            "config": _expected_modifier_parameters(execution="pulse_resolved"),
            "report": report,
            "target_mapping_report": mapping,
            "requested_target": zeros.clone(),
            "persistent_endpoint": zeros.clone(),
            "raw_apparent_endpoint": zeros.clone(),
            "apparent_endpoint": zeros.clone(),
            "global_requested_target": zeros.clone(),
            "accepted": accepted,
            "budget_exhausted": exhausted,
            "corrupt": torch.zeros_like(accepted),
            "target_below_lower_bound": below,
            "target_inside_bounds": inside,
            "target_above_upper_bound": above,
            "acceptance_window_reachable": torch.ones_like(accepted),
            "saturated": torch.zeros_like(accepted),
            "set_count": counter.clone(),
            "reset_count": counter.clone(),
            "total_pulses": counter.clone(),
            "verify_count": torch.ones_like(counter),
            "reversals": counter.clone(),
        },
        report,
    )


def test_commands_freeze_canonical_base_teacher_device_model_and_arm_roots(
    tmp_path: Path,
) -> None:
    study_dir = tmp_path / STUDY_ID
    canonical = tmp_path / "canonical.pt"
    teacher = tmp_path / "teacher.pt"
    model = tmp_path / "device.json"
    commands = _commands(
        python=Path("/test/python"),
        study_dir=study_dir,
        canonical_checkpoint_path=canonical,
        teacher_path=teacher,
        device_model_path=model,
    )

    assert set(commands) == set(TRAINING_ARM_CONFIGS)
    for arm, command in commands.items():
        assert command[:4] == ["/test/python", "-m", "ebl", "train"]
        assert Path(command[command.index("--config") + 1]).name == (
            TRAINING_ARM_CONFIGS[arm]
        )
        assert Path(command[command.index("--output-dir") + 1]) == (
            study_dir / "runs" / arm
        ).resolve()
        assert Path(command[command.index("--weights") + 1]) == canonical.resolve()
        assert Path(command[command.index("--teacher-weights") + 1]) == (
            teacher.resolve()
        )
        assert Path(command[command.index("--device-model") + 1]) == model.resolve()


def test_config_contracts_isolate_only_clean_vs_pair_hwa() -> None:
    _validate_config_contracts()


def test_study_plan_predeclares_internal_match_and_unmatched_four_cell_boundary() -> None:
    plan = json.loads(STUDY_PLAN.read_text(encoding="utf-8"))
    text = json.dumps(plan)

    assert plan["study_id"] == STUDY_ID
    assert {arm["arm_id"] for arm in plan["arms"]} == set(TRAINING_ARM_CONFIGS)
    assert "tensor-identical current-schema canonicalization" in plan["motivation"]
    assert "do not constitute a matched four-versus-eight experiment" in plan[
        "motivation"
    ]
    assert "cannot by itself establish that on-chip training is necessary" in text
    assert "Do not calculate a causal four-versus-eight effect" in text


def test_launcher_accepts_only_a_fresh_hash_matched_prepared_study(
    tmp_path: Path,
) -> None:
    study_dir = _prepared_study(tmp_path)
    source, source_sha, teacher, teacher_sha, model, model_sha = _frozen_inputs(
        tmp_path
    )

    study = _validate_prepared_study(
        study_dir,
        source_base_path=source,
        source_base_sha256=source_sha,
        teacher_path=teacher,
        teacher_sha256=teacher_sha,
        device_model_path=model,
        device_model_sha256=model_sha,
    )

    assert study["study_id"] == STUDY_ID
    assert len(study["arms"]) == 2


@pytest.mark.parametrize("prior_kind", ["launch", "native"])
def test_launcher_rejects_any_prior_launcher_or_native_attempt(
    tmp_path: Path,
    prior_kind: str,
) -> None:
    study_dir = _prepared_study(tmp_path)
    source, source_sha, teacher, teacher_sha, model, model_sha = _frozen_inputs(
        tmp_path
    )
    if prior_kind == "launch":
        (study_dir / "launch" / "failed-preflight").mkdir(parents=True)
    else:
        arm = next(iter(TRAINING_ARM_CONFIGS))
        (study_dir / "runs" / arm / "attempt-001").mkdir()

    with pytest.raises(RuntimeError, match="no native or launcher attempts"):
        _validate_prepared_study(
            study_dir,
            source_base_path=source,
            source_base_sha256=source_sha,
            teacher_path=teacher,
            teacher_sha256=teacher_sha,
            device_model_path=model,
            device_model_sha256=model_sha,
        )


def test_launcher_rejects_frozen_input_hash_and_prepared_config_mutations(
    tmp_path: Path,
) -> None:
    study_dir = _prepared_study(tmp_path)
    source, source_sha, teacher, teacher_sha, model, model_sha = _frozen_inputs(
        tmp_path
    )
    source.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="source checkpoint SHA-256"):
        _validate_prepared_study(
            study_dir,
            source_base_path=source,
            source_base_sha256=source_sha,
            teacher_path=teacher,
            teacher_sha256=teacher_sha,
            device_model_path=model,
            device_model_sha256=model_sha,
        )

    source.write_bytes(b"clean differential fixture")
    record_path = study_dir / "study.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["arms"][0]["configs"][0]["sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="prepared config digest"):
        _validate_prepared_study(
            study_dir,
            source_base_path=source,
            source_base_sha256=source_sha,
            teacher_path=teacher,
            teacher_sha256=teacher_sha,
            device_model_path=model,
            device_model_sha256=model_sha,
        )


def test_canonicalization_report_requires_tensor_identity_current_metadata_and_replay() -> None:
    report = _canonicalization_report()

    assert _validate_canonicalization_report(report) == _SHA

    for mutation, match in (
        (("unequal_tensor_keys", [EXPECTED_BINDING_KEYS[0]]), "metadata-only"),
        (("current_metadata_validation", "skipped"), "metadata-only"),
        (("metrics_exactly_equal", False), "metadata-only"),
        (("optimizer_updates", 1), "metadata-only"),
    ):
        invalid = deepcopy(report)
        invalid[mutation[0]] = mutation[1]
        with pytest.raises(RuntimeError, match=match):
            _validate_canonicalization_report(invalid)


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("examples", CANONICAL_REPLAY_EXAMPLES - 1),
        ("student_accuracy", 0.9699),
        ("teacher_agreement", 0.9899),
        ("kl_teacher_student", 0.001001),
        ("kl_teacher_student", float("nan")),
    ],
)
def test_canonicalization_report_fails_closed_on_replay_quality(
    metric: str,
    value: float | int,
) -> None:
    report = _canonicalization_report()
    report["canonical_metrics"][metric] = value
    report["source_metrics"][metric] = value

    with pytest.raises(RuntimeError, match="quality gates"):
        _validate_canonicalization_report(report)


def test_canonicalization_copies_every_tensor_and_replays_current_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_canonicalization_fakes(monkeypatch)
    source_path = tmp_path / "source.pt"
    teacher_path = tmp_path / "teacher.pt"
    source_path.write_bytes(b"source")
    teacher_path.write_bytes(b"teacher")
    preflight_dir = tmp_path / "preflight"
    preflight_dir.mkdir()

    stack, canonical_path, report = _canonicalize_base_checkpoint(
        preflight_dir=preflight_dir,
        spec=object(),
        teacher=object(),
        validation_loader=object(),
        source_base_path=source_path,
        teacher_path=teacher_path,
    )

    assert canonical_path.is_file()
    assert state["metadata_validations"] == 1
    assert state["evaluations"] == 2
    assert stack.cost.gain == 2.5
    assert report["unequal_tensor_keys"] == []
    assert report["metrics_exactly_equal"] is True
    assert report["current_metadata_validation"] == "passed"
    assert tuple(state["saved"]["tensors"]) == EXPECTED_BINDING_KEYS
    assert state["saved"]["metadata"]["weight_modifier"] == {
        "type": "none",
        "parameters": {},
    }
    assert "selection_weight_modifier" not in state["saved"]["metadata"]
    assert state["saved"]["metadata"]["canonicalization"] == {
        "operation": "metadata_only_tensor_identical_copy",
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "teacher_path": str(teacher_path.resolve()),
        "teacher_sha256": sha256_file(teacher_path),
        "teacher_mapping_invoked": False,
        "conductance_remapping_invoked": False,
        "optimizer_updates": 0,
        "retraining_epochs": 0,
    }


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("tensor", "preserve every tensor exactly"),
        ("metadata", "current metadata rejected"),
        ("replay", "exact source-versus-canonical"),
        ("quality", "predeclared clean validation quality"),
    ],
)
def test_canonicalization_fails_before_eligibility_on_identity_metadata_or_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    match: str,
) -> None:
    _install_canonicalization_fakes(monkeypatch, failure=failure)
    source_path = tmp_path / "source.pt"
    teacher_path = tmp_path / "teacher.pt"
    source_path.write_bytes(b"source")
    teacher_path.write_bytes(b"teacher")
    preflight_dir = tmp_path / "preflight"
    preflight_dir.mkdir()

    with pytest.raises(RuntimeError, match=match):
        _canonicalize_base_checkpoint(
            preflight_dir=preflight_dir,
            spec=object(),
            teacher=object(),
            validation_loader=object(),
            source_base_path=source_path,
            teacher_path=teacher_path,
        )


def test_authoritative_pair_mapping_requires_exact_geometry_support_and_span() -> None:
    report = _mapping_report()

    assert (
        _validate_authoritative_mapping_report(report)
        == EXPECTED_POPULATION_FINGERPRINT
    )

    mutations = (
        ("common_window_group_size", 4, "mapping fixture"),
        ("common_window_empty_pair_count", 7, "mapping fixture"),
        (
            "mapped_target_below_lower_bound_nonempty_pair",
            1,
            "mapping fixture",
        ),
        ("corrupt_pair_count", 1, "mapping fixture"),
    )
    for key, value, match in mutations:
        invalid = deepcopy(report)
        invalid[key] = value
        with pytest.raises(RuntimeError, match=match):
            _validate_authoritative_mapping_report(invalid)

    invalid = deepcopy(report)
    invalid["nonempty_inner_common_span"]["minimum"] = 0.0
    invalid["nonempty_pair_inner_common_span"]["minimum"] = 0.0
    with pytest.raises(RuntimeError, match="mapping fixture"):
        _validate_authoritative_mapping_report(invalid)

    invalid = deepcopy(report)
    invalid["differential_pair_binding"]["catalog_order"] = list(
        reversed(EXPECTED_BINDING_KEYS)
    )
    with pytest.raises(RuntimeError, match="canonical adjacent"):
        _validate_authoritative_mapping_report(invalid)


def test_compact_canary_requires_exact_partition_fallback_policy_and_digest() -> None:
    mapping = _mapping_report()
    report = _compact_report(mapping)
    digest = _fallback_digest()

    assert (
        _validate_compact_training_canary(
            report,
            mapping_report=mapping,
            expected_fallback_indices_sha256=digest,
        )
        == digest
    )

    wrong_partition = deepcopy(report)
    wrong_partition["compact_endpoint_devices"] -= 1
    with pytest.raises(RuntimeError, match="317584-plus-16"):
        _validate_compact_training_canary(
            wrong_partition,
            mapping_report=mapping,
        )

    wrong_policy = deepcopy(report)
    wrong_policy["pulse_resolved_fallback"]["policy"] = "general_extrapolation"
    with pytest.raises(RuntimeError, match="317584-plus-16"):
        _validate_compact_training_canary(wrong_policy, mapping_report=mapping)

    wrong_digest = deepcopy(report)
    wrong_digest["pulse_resolved_fallback"]["selection_indices_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="317584-plus-16"):
        _validate_compact_training_canary(
            wrong_digest,
            mapping_report=mapping,
            expected_fallback_indices_sha256=digest,
        )

    wrong_success = deepcopy(report)
    wrong_success["success_fraction"] = 0.5
    with pytest.raises(RuntimeError, match="317584-plus-16"):
        _validate_compact_training_canary(wrong_success, mapping_report=mapping)

    incomplete_fallback_partition = deepcopy(report)
    incomplete_fallback_partition["pulse_resolved_fallback"]["accepted"] = 1
    incomplete_fallback_partition["pulse_resolved_fallback"][
        "budget_exhausted"
    ] = 1
    incomplete_fallback_partition["pulse_resolved_fallback"][
        "success_fraction"
    ] = 1 / EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
    with pytest.raises(RuntimeError, match="317584-plus-16"):
        _validate_compact_training_canary(
            incomplete_fallback_partition,
            mapping_report=mapping,
        )


def test_pulse_canary_requires_full_exact_array_success_and_budget_gates() -> None:
    mapping = _mapping_report()
    report = _pulse_report(mapping)

    _validate_pulse_canary(report, mapping_report=mapping)

    low_success = deepcopy(report)
    low_success["success_fraction"] = 0.989
    with pytest.raises(RuntimeError, match="pulse-resolved canary"):
        _validate_pulse_canary(low_success, mapping_report=mapping)

    excessive_budget = deepcopy(report)
    excessive_budget["budget_exhausted"] = EXPECTED_TOTAL_DEVICES // 100 + 1
    with pytest.raises(RuntimeError, match="pulse-resolved canary"):
        _validate_pulse_canary(excessive_budget, mapping_report=mapping)

    inconsistent_success = deepcopy(report)
    inconsistent_success["success_fraction"] = 0.999
    with pytest.raises(RuntimeError, match="pulse-resolved canary"):
        _validate_pulse_canary(inconsistent_success, mapping_report=mapping)

    inconsistent_accepted = deepcopy(report)
    inconsistent_accepted["accepted"] -= 1
    with pytest.raises(RuntimeError, match="pulse-resolved canary"):
        _validate_pulse_canary(inconsistent_accepted, mapping_report=mapping)


def test_full_preflight_requires_all_frozen_pair_programming_and_artifact_gates() -> None:
    report = _preflight_report()

    assert _validate_preflight_report(report) == (
        EXPECTED_POPULATION_FINGERPRINT,
        _fallback_digest(),
        _SHA,
    )

    mutations = [
        ("training_mapping_report", "population_fingerprint", "0" * 64),
        ("mapped_targets", "finite_count", EXPECTED_TOTAL_DEVICES - 1),
        ("network_preflight", "programming_contexts", 2),
        ("serialization_canary", "json_round_trip", "declared_only"),
        ("artifacts", "pulse_report", {"sha256": "bad"}),
    ]
    for section, key, value in mutations:
        invalid = deepcopy(report)
        invalid[section][key] = value
        with pytest.raises((RuntimeError, ValueError)):
            _validate_preflight_report(invalid)


def test_full_preflight_rejects_wrong_fallback_deployment_and_endpoint_policy() -> None:
    report = _preflight_report()
    report["artifacts"]["compact_deployment"][
        "endpoint_generation_policy"
    ] = "compact_only"
    with pytest.raises(RuntimeError, match="deployment policy"):
        _validate_preflight_report(report)

    report = _preflight_report()
    report["pulse_resolved_canary_report"][
        "endpoint_application_policy"
    ] = "persistent_forward"
    with pytest.raises(RuntimeError, match="pulse-resolved canary"):
        _validate_preflight_report(report)


def test_network_canary_uses_one_already_programmed_context() -> None:
    class Modifier:
        contexts = 0
        active = False

        @contextmanager
        def evaluation_context(self):
            self.contexts += 1
            self.active = True
            try:
                yield
            finally:
                self.active = False

    modifier = Modifier()

    def evaluator(stack, teacher, loader, *, maximum_batches):
        assert modifier.active is True
        assert maximum_batches == NETWORK_PREFLIGHT_BATCHES
        return {"examples": NETWORK_PREFLIGHT_EXAMPLES}

    metrics = _evaluate_apparent_forward_canary(
        modifier,
        evaluator=evaluator,
        stack=object(),
        teacher=object(),
        validation_loader=object(),
    )

    assert metrics["examples"] == NETWORK_PREFLIGHT_EXAMPLES
    assert modifier.contexts == 1
    assert modifier.active is False


def test_torch_payload_equality_is_tensor_exact_and_type_strict() -> None:
    import torch

    payload = {
        "tensor": torch.tensor([1.0, 2.0]),
        "nested": [1, ("x", False)],
    }
    clone = {
        "tensor": payload["tensor"].clone(),
        "nested": [1, ("x", False)],
    }
    assert _torch_payload_equal(payload, clone)

    reordered = {"nested": clone["nested"], "tensor": clone["tensor"]}
    assert not _torch_payload_equal(payload, reordered)
    changed = deepcopy(clone)
    changed["tensor"][0] = 1.0001
    assert not _torch_payload_equal(payload, changed)
    assert not _torch_payload_equal([1, 2], (1, 2))
    assert not _torch_payload_equal(1, True)


def test_serialization_canary_reads_back_exact_json_torch_masks_and_endpoints(
    tmp_path: Path,
) -> None:
    fixture = _serialized_canary_fixture(tmp_path)
    _write_serialized_canary_fixture(fixture)

    assert _validate_serialized_canary_artifacts(**fixture) == {
        "json_round_trip": "passed",
        "checkpoint_round_trip": "passed",
        "allow_nan": False,
        "pickle_unsafe_custom_objects": False,
        "compact_and_fallback_mask_partition": "passed",
        "fallback_indices_mask_report_parity": "passed",
        "finite_apparent_and_persistent_endpoints": "passed",
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("mask_overlap", "partition all cells"),
        ("mask_gap", "partition all cells"),
        ("stored_indices", "indices, digest, mask"),
        ("reported_indices", "indices, digest, mask"),
        ("fallback_policy", "indices, digest, mask"),
        ("nonfinite_apparent", "finite compact deployment"),
        ("nonfinite_persistent", "finite pulse deployment"),
    ],
)
def test_serialization_canary_fails_closed_on_hybrid_or_endpoint_mutation(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    import torch

    fixture = _serialized_canary_fixture(tmp_path)
    compact = fixture["compact_deployment"]
    pulse = fixture["pulse_deployment"]
    if mutation == "mask_overlap":
        compact["compact_endpoint_mask"][0] = True
    elif mutation == "mask_gap":
        compact["pulse_resolved_fallback_mask"][0] = False
    elif mutation == "stored_indices":
        compact["pulse_resolved_fallback"]["selection_indices"] = torch.arange(
            1,
            EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES + 1,
            dtype=torch.int64,
        )
    elif mutation == "reported_indices":
        fixture["compact_report"]["pulse_resolved_fallback"][
            "selection_indices"
        ] = list(range(1, EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES + 1))
    elif mutation == "fallback_policy":
        compact["pulse_resolved_fallback"]["policy"] = "compact_all"
    elif mutation == "nonfinite_apparent":
        compact["apparent_endpoint"][0] = float("inf")
    elif mutation == "nonfinite_persistent":
        pulse["persistent_endpoint"][0] = float("inf")
    else:  # pragma: no cover - parametrization exhaustiveness
        raise AssertionError(mutation)
    _write_serialized_canary_fixture(fixture)

    with pytest.raises(RuntimeError, match=match):
        _validate_serialized_canary_artifacts(**fixture)


def test_serialization_canary_rejects_corrupt_or_nonidentical_round_trip(
    tmp_path: Path,
) -> None:
    fixture = _serialized_canary_fixture(tmp_path)
    _write_serialized_canary_fixture(fixture)
    fixture["network_path"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="strict JSON round trip"):
        _validate_serialized_canary_artifacts(**fixture)

    fixture = _serialized_canary_fixture(tmp_path / "torch")
    fixture["compact_report_path"].parent.mkdir(parents=True)
    _write_serialized_canary_fixture(fixture)
    import torch

    loaded = torch.load(
        fixture["pulse_deployment_path"],
        map_location="cpu",
        weights_only=True,
    )
    loaded["persistent_endpoint"][0] = 0.5
    torch.save(loaded, fixture["pulse_deployment_path"])
    with pytest.raises(RuntimeError, match="exact pulse deployment"):
        _validate_serialized_canary_artifacts(**fixture)


def test_completed_summary_requires_exact_artifact_verified_arm_coverage() -> None:
    summary = {
        "study_id": STUDY_ID,
        "state": "ready_for_review",
        "ready_for_review": True,
        "validation_mode": "full_artifact_hashes",
        "arms": [
            {
                "arm_id": arm,
                "expected_runs": 1,
                "complete": 1,
                "running": 0,
                "failed": 0,
                "invalid": 0,
                "coverage_complete": True,
            }
            for arm in TRAINING_ARM_CONFIGS
        ],
    }
    _validate_completed_summary(summary)

    for mutation in ("failed", "missing", "extra", "validation"):
        invalid = deepcopy(summary)
        if mutation == "failed":
            invalid["arms"][0]["failed"] = 1
        elif mutation == "missing":
            invalid["arms"].pop()
        elif mutation == "extra":
            invalid["arms"].append(
                {**invalid["arms"][0], "arm_id": "unmatched-four-cell"}
            )
        else:
            invalid["validation_mode"] = "paths_only"
        with pytest.raises(RuntimeError):
            _validate_completed_summary(invalid)


def test_native_manifest_requires_exact_three_frozen_input_roles() -> None:
    manifest = {
        "inputs": [
            {
                "role": "teacher_weights",
                "sha256": RELU_TEACHER_CHECKPOINT_SHA256,
            },
            {"role": "weights", "sha256": _SHA},
            {"role": "device_model", "sha256": DEVICE_MODEL_SHA256},
        ]
    }
    _validate_native_manifest_inputs(manifest, canonical_sha256=_SHA)

    wrong_hash = deepcopy(manifest)
    wrong_hash["inputs"][1]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bind canonical weights"):
        _validate_native_manifest_inputs(wrong_hash, canonical_sha256=_SHA)

    duplicate_role = deepcopy(manifest)
    duplicate_role["inputs"].append(deepcopy(duplicate_role["inputs"][0]))
    with pytest.raises(RuntimeError, match="bind canonical weights"):
        _validate_native_manifest_inputs(duplicate_role, canonical_sha256=_SHA)


def test_native_epoch_records_require_initialization_and_exact_ordered_ten_epochs(
    tmp_path: Path,
) -> None:
    _write_epoch_records(tmp_path)
    _validate_native_epoch_records(tmp_path)

    path = tmp_path / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[4]["epoch"] = 99
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="ordered epoch/global-step coverage"):
        _validate_native_epoch_records(tmp_path)

    _write_epoch_records(tmp_path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"epoch": 10, "completed_epochs": 11}) + "\n")
    with pytest.raises(RuntimeError, match="exactly ten"):
        _validate_native_epoch_records(tmp_path)


def test_result_artifacts_accept_role_distinguished_duplicate_kinds_and_hash_all(
    tmp_path: Path,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    first = artifacts_dir / "ibm_om_population.training.npz"
    second = artifacts_dir / "ibm_om_population.selection.npz"
    first.write_bytes(b"training population")
    second.write_bytes(b"selection population")
    result = {
        "artifacts": [
            _artifact(first, run_dir=tmp_path, kind="ibm_om_array_population"),
            _artifact(second, run_dir=tmp_path, kind="ibm_om_array_population"),
        ]
    }

    records = _result_artifacts(tmp_path, result)
    assert [record["path"] for record in records["ibm_om_array_population"]] == [
        "artifacts/ibm_om_population.training.npz",
        "artifacts/ibm_om_population.selection.npz",
    ]

    duplicate = deepcopy(result)
    duplicate["artifacts"].append(deepcopy(duplicate["artifacts"][0]))
    with pytest.raises(RuntimeError, match="kind/path pairs"):
        _result_artifacts(tmp_path, duplicate)

    wrong_hash = deepcopy(result)
    wrong_hash["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="full artifact parity"):
        _result_artifacts(tmp_path, wrong_hash)

    escaped = deepcopy(result)
    escaped["artifacts"][0]["path"] = "../outside.npz"
    (tmp_path.parent / "outside.npz").write_bytes(b"outside")
    escaped["artifacts"][0]["sha256"] = sha256_file(
        tmp_path.parent / "outside.npz"
    )
    escaped["artifacts"][0]["size_bytes"] = (
        tmp_path.parent / "outside.npz"
    ).stat().st_size
    with pytest.raises(RuntimeError, match="full artifact parity"):
        _result_artifacts(tmp_path, escaped)


def test_result_metrics_bind_selected_and_last_values_to_exact_epoch_records() -> None:
    result, metrics, records = _result_metric_fixture()

    selected_epoch, selected_validation = _validate_result_metric_bindings(
        result=result,
        metrics=metrics,
        epoch_records=records,
    )
    assert selected_epoch == 4
    assert selected_validation == records[5]["validation"]

    mutations = (
        ("status", "failed", "strict complete native result"),
        ("selected_epoch", True, "selected epoch"),
        ("selected_value", 0.0, "selected validation"),
        ("last_value", 0.0, "initial/last metrics"),
    )
    for mutation, value, match in mutations:
        invalid_result = deepcopy(result)
        invalid_metrics = deepcopy(metrics)
        if mutation == "status":
            invalid_result["status"] = value
        elif mutation == "selected_epoch":
            invalid_metrics["selected"]["epoch"] = value
        elif mutation == "selected_value":
            invalid_metrics["selected"]["student_accuracy"] = value
        else:
            invalid_metrics["last_validation"]["student_accuracy"] = value
        with pytest.raises(RuntimeError, match=match):
            _validate_result_metric_bindings(
                result=invalid_result,
                metrics=invalid_metrics,
                epoch_records=records,
            )

    wrong_best = deepcopy(metrics)
    wrong_best["selected"] = {"epoch": 3, **records[4]["validation"]}
    with pytest.raises(RuntimeError, match="strict first-argmin KL"):
        _validate_result_metric_bindings(
            result=result,
            metrics=wrong_best,
            epoch_records=records,
        )


def test_selected_checkpoint_and_resume_bind_metrics_epoch_step_and_tensors(
    tmp_path: Path,
) -> None:
    import torch

    _, metrics, records = _result_metric_fixture()
    selected_epoch = 4
    selected_validation = records[selected_epoch + 1]["validation"]
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    selected = {
        "schema": "drn.named-weights",
        "schema_version": 1,
        "metadata": {
            "teacher_sha256": RELU_TEACHER_CHECKPOINT_SHA256,
            "encoding": "differential",
            "selection_metric": "validation_modifier_mean.kl_teacher_student",
            "selection_epoch": selected_epoch,
            "selection_value": selected_validation["kl_teacher_student"],
            "selection_student_accuracy": selected_validation[
                "student_accuracy"
            ],
            "selection_teacher_agreement": selected_validation[
                "teacher_agreement"
            ],
        },
        "weights": {"w": torch.tensor([1.0, 2.0])},
    }
    torch.save(selected, checkpoints / "weights.pt")
    resume = {
        "schema": "drn.epoch-boundary-resume",
        "schema_version": 3,
        "resume_capability": "exact",
        "epoch": EXPECTED_EPOCHS_PER_ARM,
        "global_step": 34380,
        "progress_state": {
            "selected_epoch": selected_epoch,
            "selected_validation": selected_validation,
            "last_train": metrics["last_train"],
            "last_validation": metrics["last_validation"],
            "last_clean_validation": metrics["last_clean_validation"],
        },
        "selected_weights": selected,
    }
    torch.save(resume, checkpoints / "resume.pt")

    loaded, digest = _validate_selected_checkpoint_and_resume(
        run_dir=tmp_path,
        metrics=metrics,
        selected_epoch=selected_epoch,
        selected_validation=selected_validation,
    )
    assert _torch_payload_equal(loaded, selected)
    assert digest == sha256_file(checkpoints / "weights.pt")

    for mutation, match in (
        ("selection_value", "selected checkpoint metric provenance"),
        ("global_step", "epoch-10 resume"),
        ("selected_weights", "epoch-10 resume"),
    ):
        if mutation == "selection_value":
            bad_selected = deepcopy(selected)
            bad_selected["metadata"]["selection_value"] = 1.0
            torch.save(bad_selected, checkpoints / "weights.pt")
        else:
            torch.save(selected, checkpoints / "weights.pt")
            bad_resume = deepcopy(resume)
            if mutation == "global_step":
                bad_resume["global_step"] -= 1
            else:
                bad_resume["selected_weights"]["weights"]["w"][0] = 99.0
            torch.save(bad_resume, checkpoints / "resume.pt")
        with pytest.raises(RuntimeError, match=match):
            _validate_selected_checkpoint_and_resume(
                run_dir=tmp_path,
                metrics=metrics,
                selected_epoch=selected_epoch,
                selected_validation=selected_validation,
            )
        torch.save(resume, checkpoints / "resume.pt")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("selected_sha", "deployment provenance"),
        ("nonfinite", "finite pulse deployment tensor"),
        ("accepted_overlap", "pulse tensor partitions and caps"),
        ("target_gap", "pulse tensor partitions and caps"),
        ("counter_sum", "pulse tensor partitions and caps"),
        ("counter_cap", "pulse tensor partitions and caps"),
        ("report_count", "masks to equal report counts"),
    ],
)
def test_pulse_deployment_integrity_gates_provenance_endpoints_masks_and_counts(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    import ebl.runtime.mnist_relu_drn.ibm_om_deployment_decomposition as decomposition

    monkeypatch.setattr(
        decomposition,
        "validate_deployment_contract",
        lambda *args, **kwargs: ({}, object(), {}),
    )
    deployment, report = _pulse_deployment_fixture()
    if mutation == "selected_sha":
        deployment["selected_weights_sha256"] = "0" * 64
    elif mutation == "nonfinite":
        deployment["persistent_endpoint"][0] = float("nan")
    elif mutation == "accepted_overlap":
        deployment["budget_exhausted"][0] = True
    elif mutation == "target_gap":
        deployment["target_inside_bounds"][100] = False
    elif mutation == "counter_sum":
        deployment["set_count"][0] = 1
    elif mutation == "counter_cap":
        deployment["total_pulses"][0] = 129
        deployment["set_count"][0] = 129
        deployment["verify_count"][0] = 130
    elif mutation == "report_count":
        report["saturated"] = 1
    else:  # pragma: no cover - parametrization exhaustiveness
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=match):
        _validate_pulse_deployment_integrity(
            deployment,
            selection_report=report,
            selected_metadata={
                "selection_epoch": 9,
                "device_model_sha256": DEVICE_MODEL_SHA256,
            },
            selected_weights_sha256=_SHA,
            selected_epoch=9,
        )


def test_pulse_deployment_integrity_accepts_the_exact_full_array_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ebl.runtime.mnist_relu_drn.ibm_om_deployment_decomposition as decomposition

    monkeypatch.setattr(
        decomposition,
        "validate_deployment_contract",
        lambda *args, **kwargs: ({}, object(), {}),
    )
    deployment, report = _pulse_deployment_fixture()
    _validate_pulse_deployment_integrity(
        deployment,
        selection_report=report,
        selected_metadata={
            "selection_epoch": 9,
            "device_model_sha256": DEVICE_MODEL_SHA256,
        },
        selected_weights_sha256=_SHA,
        selected_epoch=9,
    )


def test_post_run_audit_rechecks_both_matched_arms_and_role_specific_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_dir, run_dirs = _write_post_run_fixture(tmp_path, monkeypatch)

    audit = _validate_post_run_audit(
        study_dir,
        canonical_sha256=_SHA,
        fallback_indices_sha256=_fallback_digest(),
    )

    assert set(audit) == set(TRAINING_ARM_CONFIGS)
    assert audit["train-clean-differential-pair"]["training_report"] is None
    assert audit["train-exact-bounds-pair-hwa"]["training_report"] == {
        "population_fingerprint": EXPECTED_POPULATION_FINGERPRINT,
        "endpoint_seed": TRAINING_ENDPOINT_SEED,
        "execution_detail": COMPACT_EXECUTION_DETAIL,
        "compact_endpoint_devices": EXPECTED_COMPACT_ENDPOINT_DEVICES,
        "pulse_resolved_fallback_devices": (
            EXPECTED_EMPTY_PAIR_FALLBACK_DEVICES
        ),
        "fallback_policy": COMPACT_FALLBACK_POLICY,
        "fallback_indices_sha256": _fallback_digest(),
    }
    for arm, run_dir in run_dirs.items():
        assert audit[arm]["run_dir"] == str(run_dir.resolve())
        assert set(audit[arm]["populations"]) == set(POPULATION_ROLES[arm])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("status", "complete native artifacts"),
        ("extra_population_role", "exact native artifact set"),
        ("artifact_path_alias", "exact native artifact set"),
        ("stale_artifact_hash", "full artifact parity"),
        ("deployment_report", "exact persistent deployment"),
        ("selected_metadata", "strict selected checkpoint"),
        ("selection_receipt", "programming/report/receipt parity"),
        ("training_sidecar", "compact-training sidecar"),
        ("fallback_digest", "317584-plus-16"),
        ("clean_training_population", "differential fixed-array result"),
    ],
)
def test_post_run_audit_fails_closed_on_native_artifact_or_semantic_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    import torch

    study_dir, run_dirs = _write_post_run_fixture(tmp_path, monkeypatch)
    clean = run_dirs["train-clean-differential-pair"]
    hwa = run_dirs["train-exact-bounds-pair-hwa"]
    if mutation == "status":
        atomic_write_json(clean / "status.json", {"status": "failed"})
    elif mutation == "extra_population_role":
        path = clean / "artifacts" / "ibm_om_population.training.npz"
        path.write_bytes(b"unexpected training role")
        result_path = clean / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["artifacts"].append(
            _artifact(path, run_dir=clean, kind="ibm_om_array_population")
        )
        atomic_write_json(result_path, result)
    elif mutation == "artifact_path_alias":
        path = clean / "artifacts" / "dummy_selected_weights.pt"
        path.write_bytes((clean / "checkpoints" / "weights.pt").read_bytes())
        result_path = clean / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for index, record in enumerate(result["artifacts"]):
            if record["kind"] == "selected_named_weights":
                result["artifacts"][index] = _artifact(
                    path,
                    run_dir=clean,
                    kind="selected_named_weights",
                )
                break
        else:  # pragma: no cover - fixture assertion
            raise AssertionError("missing selected artifact")
        atomic_write_json(result_path, result)
    elif mutation == "stale_artifact_hash":
        (clean / "checkpoints" / "resume.pt").write_bytes(b"mutated")
    elif mutation == "deployment_report":
        path = clean / "artifacts" / "ibm_om_deployment.pt"
        deployment = torch.load(path, map_location="cpu", weights_only=True)
        deployment["report"]["untracked_mutation"] = True
        torch.save(deployment, path)
        _rewrite_result_artifact_record(
            clean,
            kind="ibm_om_persistent_deployment",
            path=path,
        )
    elif mutation == "selected_metadata":
        path = clean / "checkpoints" / "weights.pt"
        selected = torch.load(path, map_location="cpu", weights_only=True)
        selected["metadata"]["teacher_sha256"] = "0" * 64
        torch.save(selected, path)
        _rewrite_result_artifact_record(
            clean,
            kind="selected_named_weights",
            path=path,
        )
    elif mutation == "selection_receipt":
        path = clean / "artifacts" / "ibm_om_population.selection.receipt.json"
        atomic_write_json(path, {"role": "selection", "mutation": True})
        _rewrite_result_artifact_record(
            clean,
            kind="ibm_om_array_population_receipt",
            path=path,
        )
    elif mutation == "training_sidecar":
        path = hwa / "artifacts" / "ibm_om_training_device_programming.json"
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        sidecar["untracked_mutation"] = True
        atomic_write_json(path, sidecar)
        _rewrite_result_artifact_record(
            hwa,
            kind="ibm_om_training_device_programming",
            path=path,
        )
    elif mutation == "fallback_digest":
        result_path = hwa / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["metrics"]["training_device_programming"][
            "pulse_resolved_fallback"
        ]["selection_indices_sha256"] = "0" * 64
        atomic_write_json(result_path, result)
    elif mutation == "clean_training_population":
        result_path = clean / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["metrics"]["ibm_om_population_fingerprints"][
            "training"
        ] = EXPECTED_POPULATION_FINGERPRINT
        atomic_write_json(result_path, result)
    else:  # pragma: no cover - parametrization exhaustiveness
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=match):
        _validate_post_run_audit(
            study_dir,
            canonical_sha256=_SHA,
            fallback_indices_sha256=_fallback_digest(),
        )
