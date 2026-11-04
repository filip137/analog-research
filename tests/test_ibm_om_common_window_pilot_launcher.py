from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from ebl.runtime.store import sha256_file
from experiments.mnist_relu_drn.ibm_om_common_window_pilot_launcher import (
    ASSIGNMENT_SEED,
    BOUNDED_CHECKPOINT_SHA256,
    COMMON_WINDOW_MARGIN_FRACTION,
    COMPACT_FALLBACK_POLICY,
    CONFIG_ROOT,
    CORRUPTION_POLICY,
    DEVICE_MODEL_SHA256,
    DUAL_RAIL_LAYOUT_BY_PARAMETER,
    ENDPOINT_APPLICATION_POLICY,
    NETWORK_PREFLIGHT_BATCHES,
    NETWORK_PREFLIGHT_EXAMPLES,
    SELECTION_ENDPOINT_SEED,
    STUDY_ID,
    STUDY_PLAN,
    TARGET_MAPPING,
    TRAINING_ENDPOINT_SEED,
    TRAINING_ARM_CONFIGS,
    _commands,
    _evaluate_apparent_forward_canary,
    _expected_modifier_parameters,
    _validate_completed_summary,
    _validate_config_contracts,
    _validate_preflight_report,
    _validate_prepared_study,
)
from ebl.lifecycle.study import prepare_study


_FINGERPRINT = "a" * 64


def _frozen_inputs(tmp_path: Path) -> tuple[Path, str, Path, str]:
    checkpoint = tmp_path / "bounded-weights.pt"
    checkpoint.write_bytes(b"bounded checkpoint fixture")
    device_model = tmp_path / "ibm-om-model.json"
    device_model.write_bytes(b'{"fixture":"device model"}\n')
    return (
        checkpoint,
        sha256(checkpoint.read_bytes()).hexdigest(),
        device_model,
        sha256(device_model.read_bytes()).hexdigest(),
    )


def _prepared_study(tmp_path: Path) -> Path:
    return prepare_study(STUDY_PLAN, tmp_path / "results")


def _mapping_report() -> dict[str, object]:
    parameter_devices = {
        "base.dense_weight.0": 156800,
        "base.dense_weight.1": 2000,
    }
    parameters = {}
    for key, layout in DUAL_RAIL_LAYOUT_BY_PARAMETER.items():
        devices = parameter_devices[key]
        quads = devices // 4
        empty = 4 if key == "base.dense_weight.0" else 0
        empty_below = 4 if empty else 0
        empty_above = 4 if empty else 0
        parameters[key] = {
            "dual_rail_layout": layout,
            "devices": devices,
            "quad_count": quads,
            "common_window_empty_quad_count": empty,
            "common_window_empty_fraction": empty / quads,
            "corrupt_quad_count": 0,
            "published_corrupt_quad_count": 0,
            "raw_common_span": {"minimum": 0.2, "mean": 0.3, "maximum": 0.4},
            "inner_common_span": {
                "minimum": 0.1,
                "mean": 0.15,
                "maximum": 0.2,
            },
            "mapped_target_below_lower_bound_nonempty_quad": 0,
            "mapped_target_above_upper_bound_nonempty_quad": 0,
            "mapped_target_below_lower_bound_empty_quad": empty_below,
            "mapped_target_above_upper_bound_empty_quad": empty_above,
        }
    return {
        "target_mapping": TARGET_MAPPING,
        "common_window_margin_fraction": COMMON_WINDOW_MARGIN_FRACTION,
        "population_fingerprint": _FINGERPRINT,
        "corruption_policy": CORRUPTION_POLICY,
        "devices": 158800,
        "global_target_support": {
            "below_lower_bound": 4,
            "above_upper_bound": 4,
            "inside_bounds": 158792,
        },
        "global_target_outside_0_1": 0,
        "parameters": parameters,
        "quad_count": 39700,
        "common_window_empty_quad_count": 4,
        "common_window_empty_fraction": 4 / 39700,
        "corrupt_quad_count": 0,
        "published_corrupt_quad_count": 0,
        "raw_common_span": {"minimum": 0.2, "mean": 0.3, "maximum": 0.4},
        "inner_common_span": {"minimum": 0.1, "mean": 0.15, "maximum": 0.2},
        "mapped_target_below_lower_bound_nonempty_quad": 0,
        "mapped_target_above_upper_bound_nonempty_quad": 0,
        "mapped_target_below_lower_bound_empty_quad": 4,
        "mapped_target_above_upper_bound_empty_quad": 4,
        "mapped_target_support": {
            "below_lower_bound": 4,
            "above_upper_bound": 4,
            "inside_bounds": 158792,
        },
    }


def _preflight_report() -> dict[str, object]:
    mapping = _mapping_report()
    apparent_metrics = {
        "examples": NETWORK_PREFLIGHT_EXAMPLES,
        "kl_teacher_student": 0.5,
        "raw_kl_teacher_student": 0.5,
        "student_accuracy": 0.3,
        "teacher_accuracy": 0.95,
        "teacher_agreement": 0.3,
        "raw_score_rms": 1.0,
        "calibrated_score_rms": 1.0,
        "teacher_logit_rms": 2.0,
        "fixed_logit_gain": 1.0,
    }
    return {
        "schema": "ebl.mnist_ibm_om_common_window.mapping_preflight",
        "schema_version": 1,
        "study_id": STUDY_ID,
        "authoritative_entry_point": (
            "methods.ibm_om.api."
            "IbmReramHwaParameterModifier.preflight_target_mapping"
        ),
        "inputs": {
            "bounded_drn_checkpoint": {"sha256": BOUNDED_CHECKPOINT_SHA256},
            "ibm_om_device_model": {"sha256": DEVICE_MODEL_SHA256},
        },
        "configs": {
            arm: {"sha256": sha256_file(CONFIG_ROOT / filename)}
            for arm, filename in TRAINING_ARM_CONFIGS.items()
        },
        "modifier_role": "selection",
        "modifier_parameters": _expected_modifier_parameters(
            execution="pulse_resolved"
        ),
        "compact_training_modifier_parameters": _expected_modifier_parameters(
            execution="compact_endpoint"
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
        "mapping_report": mapping,
        "compact_training_mapping_report": deepcopy(mapping),
        "population_receipt": {
            "population_fingerprint": _FINGERPRINT,
            "aihwkit_version": "1.1.0",
            "request": {
                "assignment_seed": ASSIGNMENT_SEED,
                "corruption_policy": CORRUPTION_POLICY,
                "binding_keys": list(DUAL_RAIL_LAYOUT_BY_PARAMETER),
            },
        },
        "compact_training_population_receipt": {
            "population_fingerprint": _FINGERPRINT,
            "aihwkit_version": "1.1.0",
            "request": {
                "assignment_seed": ASSIGNMENT_SEED,
                "corruption_policy": CORRUPTION_POLICY,
                "binding_keys": list(DUAL_RAIL_LAYOUT_BY_PARAMETER),
            },
        },
        "mapped_targets": {
            "count": 158800,
            "finite_count": 158800,
            "outside_0_1_count": 0,
            "minimum": 0.2,
            "maximum": 0.8,
        },
        "compact_training_canary_report": {
            "execution": "compact_endpoint",
            "execution_detail": (
                "compact_endpoint_with_exact_empty_quad_fallback"
            ),
            "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            "endpoint_seed": TRAINING_ENDPOINT_SEED,
            "assignment_seed": ASSIGNMENT_SEED,
            "corruption_policy": CORRUPTION_POLICY,
            "population_fingerprint": _FINGERPRINT,
            "device_model_sha256": DEVICE_MODEL_SHA256,
            "devices": 158800,
            "compact_endpoint_devices": 158792,
            "pulse_resolved_fallback_devices": 8,
            "accepted": 158800,
            "success_fraction": 1.0,
            "failed_noncorrupt": 0,
            "corrupt": 0,
            "target_below_lower_bound": 4,
            "target_inside_bounds": 158792,
            "target_above_upper_bound": 4,
            "pulse_resolved_fallback": {
                "policy": COMPACT_FALLBACK_POLICY,
                "devices": 8,
                "selection_indices": list(range(8)),
                "selection_indices_sha256": "e" * 64,
                "target_below_lower_bound": 4,
                "target_above_upper_bound": 4,
                "acceptance_window_unreachable": 2,
                "start_state": "sampled_fully_reset_bound",
                "controller": "adaptive",
                "maximum_program_pulses": 128,
                "random_stream_order": "after_compact_endpoint_sampling",
                "accepted": 8,
                "success_fraction": 1.0,
                "budget_exhausted": 0,
                "nonfinite": 0,
                "pulse_count": {
                    "mean": 14.875,
                    "median": 4.0,
                    "maximum": 95,
                    "set_mean": 10.0,
                    "reset_mean": 4.875,
                },
                "verify_count_mean": 15.875,
                "reversal_count_mean": 1.0,
                "saturated": 7,
                "endpoint_clipped": 0,
            },
        },
        "pulse_resolved_canary_report": {
            "execution": "pulse_resolved",
            "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            "endpoint_seed": SELECTION_ENDPOINT_SEED,
            "assignment_seed": ASSIGNMENT_SEED,
            "population_fingerprint": _FINGERPRINT,
            "device_model_sha256": DEVICE_MODEL_SHA256,
            "devices": 158800,
            "nonfinite": 0,
            "budget_exhausted": 0,
            "success_fraction": 1.0,
            "target_mapping_report": deepcopy(mapping),
        },
        "network_preflight": {
            "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            "forward_endpoint": "aihwkit_apparent_endpoint",
            "persistent_endpoint_role": "hidden_update_state",
            "programming_contexts": 1,
            "cohort": {
                "split": "validation",
                "data_seed": 17,
                "validation_points": 5000,
                "batch_size": 16,
                "maximum_batches": NETWORK_PREFLIGHT_BATCHES,
                "examples": NETWORK_PREFLIGHT_EXAMPLES,
                "shuffle": False,
            },
            "apparent_forward_metrics": apparent_metrics,
        },
        "artifacts": {
            "compact_training_population": {"sha256": "f" * 64},
            "compact_training_population_receipt": {"sha256": "1" * 64},
            "compact_training_canary_report": {"sha256": "2" * 64},
            "compact_training_canary_deployment": {
                "sha256": "3" * 64,
                "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            },
            "pulse_resolved_canary_deployment": {
                "sha256": "b" * 64,
                "endpoint_application_policy": ENDPOINT_APPLICATION_POLICY,
            },
            "apparent_forward_network_metrics": {
                "sha256": "c" * 64,
            },
            "selected_checkpoint_metadata_canary": {
                "sha256": "d" * 64,
            },
        },
    }


def _completed_summary() -> dict[str, object]:
    return {
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


def test_commands_freeze_two_exact_configs_inputs_and_output_roots(
    tmp_path: Path,
) -> None:
    study_dir = tmp_path / STUDY_ID
    checkpoint = tmp_path / "bounded.pt"
    device_model = tmp_path / "device.json"
    commands = _commands(
        python=Path("/test/python"),
        study_dir=study_dir,
        checkpoint_path=checkpoint,
        device_model_path=device_model,
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
        assert Path(command[command.index("--teacher-weights") + 1]) == (
            checkpoint.resolve()
        )
        assert Path(command[command.index("--device-model") + 1]) == (
            device_model.resolve()
        )


def test_config_contracts_isolate_only_clean_vs_exact_hwa() -> None:
    _validate_config_contracts()


def test_v2_is_corrected_retry_with_exact_predeclared_delta_from_v1() -> None:
    v1_path = STUDY_PLAN.with_name(
        "mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v1.json"
    )
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(STUDY_PLAN.read_text(encoding="utf-8"))

    assert STUDY_ID.endswith("-20260824-v2")
    assert v1["study_id"].endswith("-20260824-v1")
    for field in ("hypothesis", "evidence_class", "arms"):
        assert v2[field] == v1[field]
    assert v2["completion_criteria"] != v1["completion_criteria"]
    assert v2["analysis_plan"] != v1["analysis_plan"]
    assert "two fixes discovered before v2 was prepared or launched" in v2[
        "motivation"
    ]
    assert "neither arm executed a training or optimizer update" in v2[
        "motivation"
    ]
    assert "must not be described as scientifically identical" in v2[
        "motivation"
    ]
    assert any(
        "exactly eight" in criterion and "fallback" in criterion
        for criterion in v2["completion_criteria"]
    )


def test_launcher_accepts_only_fresh_prepared_corrected_pilot(
    tmp_path: Path,
) -> None:
    study_dir = _prepared_study(tmp_path)
    checkpoint, checkpoint_sha, device_model, device_model_sha = _frozen_inputs(
        tmp_path
    )

    study = _validate_prepared_study(
        study_dir,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        device_model_path=device_model,
        device_model_sha256=device_model_sha,
    )

    assert study["study_id"] == STUDY_ID
    assert len(study["arms"]) == 2


def test_launcher_rejects_prior_preflight_or_native_attempt(
    tmp_path: Path,
) -> None:
    study_dir = _prepared_study(tmp_path)
    checkpoint, checkpoint_sha, device_model, device_model_sha = _frozen_inputs(
        tmp_path
    )
    (study_dir / "launch" / "failed-preflight").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="no native or launcher attempts"):
        _validate_prepared_study(
            study_dir,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            device_model_path=device_model,
            device_model_sha256=device_model_sha,
        )


def test_preflight_requires_authoritative_mapper_and_exact_pulse_canary() -> None:
    report = _preflight_report()

    assert _validate_preflight_report(report) == _FINGERPRINT

    failed = deepcopy(report)
    failed["pulse_resolved_canary_report"]["success_fraction"] = 0.989
    with pytest.raises(RuntimeError, match="pulse-resolved canary"):
        _validate_preflight_report(failed)


def test_preflight_gates_exact_compact_empty_quad_fallback() -> None:
    wrong_partition = _preflight_report()
    wrong_partition["compact_training_canary_report"][
        "pulse_resolved_fallback_devices"
    ] = 7
    with pytest.raises(RuntimeError, match="compact preflight"):
        _validate_preflight_report(wrong_partition)

    nonfinite = _preflight_report()
    nonfinite["compact_training_canary_report"][
        "pulse_resolved_fallback"
    ]["nonfinite"] = 1
    with pytest.raises(RuntimeError, match="compact fallback"):
        _validate_preflight_report(nonfinite)

    excessive_cost = _preflight_report()
    excessive_cost["compact_training_canary_report"][
        "pulse_resolved_fallback"
    ]["pulse_count"]["maximum"] = 129
    with pytest.raises(RuntimeError, match="compact-fallback pulse"):
        _validate_preflight_report(excessive_cost)


def test_preflight_requires_single_context_apparent_forward_network_gate() -> None:
    report = _preflight_report()
    report["network_preflight"]["apparent_forward_metrics"][
        "student_accuracy"
    ] = 0.249

    with pytest.raises(RuntimeError, match="apparent-forward network canary"):
        _validate_preflight_report(report)

    wrong_policy = _preflight_report()
    wrong_policy["pulse_resolved_canary_report"][
        "endpoint_application_policy"
    ] = "persistent_forward"
    with pytest.raises(RuntimeError, match="pulse-resolved canary"):
        _validate_preflight_report(wrong_policy)


def test_network_canary_reuses_exactly_one_programmed_context() -> None:
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


def test_preflight_rejects_nonempty_outside_target_and_excess_empty_fraction() -> None:
    outside = _preflight_report()
    outside["mapping_report"][
        "mapped_target_below_lower_bound_nonempty_quad"
    ] = 1
    with pytest.raises(ValueError, match="outside physical support"):
        _validate_preflight_report(outside)

    empty = _preflight_report()
    empty["mapping_report"]["common_window_empty_quad_count"] = 1
    empty["mapping_report"]["common_window_empty_fraction"] = 0.5
    with pytest.raises(RuntimeError, match="structural quad reachability"):
        _validate_preflight_report(empty)


def test_completed_phase_requires_both_artifact_verified_arms() -> None:
    summary = _completed_summary()
    _validate_completed_summary(summary)

    summary["arms"][0]["complete"] = 0
    summary["arms"][0]["coverage_complete"] = False
    with pytest.raises(RuntimeError, match="one valid complete run"):
        _validate_completed_summary(summary)
