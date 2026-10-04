from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.mnist_analog_relu import staged_akib_launcher as launcher
from experiments.artifacts import sha256_file
from experiments.artifacts import content_hash
from experiments.mnist_analog_relu.generate_staged_campaign import (
    DEVELOPMENT_PLAN_ID,
    PRODUCTION_PLAN_ID,
    TEACHER_PLAN_ID,
)
from experiments.mnist_analog_relu.staged_akib_launcher import (
    HEARTBEAT_SECONDS,
    MAXIMUM_HEARTBEAT_SECONDS,
    RunHandle,
    TaskBlueprint,
    TaskInputs,
    _completed_run,
    _native_command,
    _production_metric_rows,
    _probe_gpu_occupancy,
    _registered_artifacts_intact,
    _replay_completed_adam_ancestry,
    _require_akib_hostname,
    _require_clean_source_commit,
    _sampler_probe_request,
    _tuning_diagnostic_row,
    _validate_declared_coverage,
    _validate_hwa_evaluation,
    _verify_adam_metrics,
    _write_production_summary,
    development_blueprints,
    production_blueprints,
    teacher_blueprints,
)
from experiments.reram_program_verify.hwa_population_sampler import (
    _request as parse_hwa_population_request,
)


def test_blueprints_exactly_cover_all_three_generated_studies() -> None:
    teacher = teacher_blueprints()
    development = development_blueprints()
    production = production_blueprints()
    _validate_declared_coverage((*teacher, *development, *production))

    assert len(teacher) == 2
    assert len(development) == 29
    assert len(production) == 160
    assert Counter(task.arm_id for task in production) == {
        "direct-deploy": 16,
        "direct-corrupt": 16,
        "hwa-deploy": 16,
        "hwa-corrupt": 16,
        "hwa-adam-healthy": 16,
        "hwa-adam-corrupt": 16,
        "scratch-deploy-preparation": 16,
        "scratch-corrupt-preparation": 16,
        "scratch-adam-healthy": 16,
        "scratch-adam-corrupt": 16,
    }
    seeds = defaultdict(set)
    for task in production:
        seeds[task.assignment_seed].add(task.endpoint_seed)
    assert len(seeds) == 4
    assert all(len(endpoints) == 4 for endpoints in seeds.values())


def test_native_commands_use_only_public_ebl_train_or_validate(tmp_path: Path) -> None:
    teacher_train, teacher_test = teacher_blueprints()
    python = Path("/test/task-python")
    teacher_weights = tmp_path / "teacher.pt"
    teacher_weights.write_bytes(b"teacher")

    train = _native_command(
        teacher_train,
        task_python=python,
        study_dir=tmp_path / TEACHER_PLAN_ID,
        inputs=TaskInputs(),
    )
    validate = _native_command(
        teacher_test,
        task_python=python,
        study_dir=tmp_path / TEACHER_PLAN_ID,
        inputs=TaskInputs(weights=teacher_weights),
    )
    assert train[:4] == [str(python.resolve()), "-m", "ebl", "train"]
    assert validate[:4] == [str(python.resolve()), "-m", "ebl", "validate"]
    assert validate[-2:] == ["--weights", str(teacher_weights.resolve())]
    assert Path(train[train.index("--output-dir") + 1]).parent.name == "runs"

    production = next(
        task for task in production_blueprints() if task.arm_id == "hwa-adam-corrupt"
    )
    state = tmp_path / "faulted.pt"
    receipt = tmp_path / "selection.json"
    state.write_bytes(b"state")
    receipt.write_text("{}", encoding="utf-8")
    command = _native_command(
        production,
        task_python=python,
        study_dir=tmp_path / PRODUCTION_PLAN_ID,
        inputs=TaskInputs(
            teacher_weights=teacher_weights,
            device_state=state,
            selection_receipt=receipt,
        ),
    )
    assert command[:4] == [str(python.resolve()), "-m", "ebl", "train"]
    assert "--device-state" in command
    assert "--selection-receipt" in command


def test_manifest_roles_distinguish_teacher_weights_from_hwa_master(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"payload")
    _teacher_train, teacher_test = teacher_blueprints()
    hwa_deploy = next(
        task for task in development_blueprints() if task.label == "development.hwa.deploy"
    )
    assert TaskInputs(weights=artifact).manifest_hashes(task=teacher_test) == {
        "weights": sha256_file(artifact)
    }
    assert TaskInputs(weights=artifact).manifest_hashes(task=hwa_deploy) == {
        "hwa_master": sha256_file(artifact)
    }


def test_canonical_runstore_commands_cover_teacher_and_all_staged_input_roles(
    tmp_path: Path,
) -> None:
    run = tmp_path / "study" / "runs" / "arm" / "run-1"
    paths = {
        role: tmp_path / f"{role}.bin"
        for role in (
            "weights",
            "hwa_master",
            "teacher_weights",
            "origin_device_state",
            "adam_selection_receipt",
            "adam_epoch_resume",
            "aihwkit_python",
        )
    }

    def manifest(*roles: str) -> dict:
        # RunStore input order is independent of the public CLI option order.
        return {
            "inputs": [
                {"role": role, "path": str(paths[role])}
                for role in reversed(roles)
            ]
        }

    def base(task: TaskBlueprint) -> list[str]:
        return [
            "ebl",
            task.mode,
            "--config",
            str(task.config.resolve()),
            "--output-dir",
            str(run.parent.resolve()),
        ]

    teacher_train, teacher_test = teacher_blueprints()
    assert launcher._canonical_manifest_command(
        task=teacher_train, run_dir=run, manifest=manifest()
    ) == base(teacher_train)
    assert launcher._canonical_manifest_command(
        task=teacher_test, run_dir=run, manifest=manifest("weights")
    ) == base(teacher_test) + ["--weights", str(paths["weights"].resolve())]

    hwa_deploy = next(
        task for task in development_blueprints() if task.label == "development.hwa.deploy"
    )
    assert launcher._canonical_manifest_command(
        task=hwa_deploy,
        run_dir=run,
        manifest=manifest("hwa_master", "teacher_weights", "aihwkit_python"),
    ) == base(hwa_deploy) + [
        "--weights",
        str(paths["hwa_master"].resolve()),
        "--teacher-weights",
        str(paths["teacher_weights"].resolve()),
    ]

    corruption = next(
        task for task in development_blueprints() if task.label == "development.hwa.corrupt"
    )
    assert launcher._canonical_manifest_command(
        task=corruption,
        run_dir=run,
        manifest=manifest("teacher_weights", "origin_device_state"),
    ) == base(corruption) + [
        "--teacher-weights",
        str(paths["teacher_weights"].resolve()),
        "--device-state",
        str(paths["origin_device_state"].resolve()),
    ]

    adam = next(
        task for task in production_blueprints() if task.arm_id == "hwa-adam-corrupt"
    )
    assert launcher._canonical_manifest_command(
        task=adam,
        run_dir=run,
        manifest=manifest(
            "teacher_weights",
            "origin_device_state",
            "adam_selection_receipt",
            "adam_epoch_resume",
        ),
    ) == base(adam) + [
        "--teacher-weights",
        str(paths["teacher_weights"].resolve()),
        "--device-state",
        str(paths["origin_device_state"].resolve()),
        "--selection-receipt",
        str(paths["adam_selection_receipt"].resolve()),
        "--resume",
        str(paths["adam_epoch_resume"].resolve()),
    ]


def test_launcher_rejects_every_non_akib_hostname() -> None:
    assert _require_akib_hostname("integnano-akib") == "integnano-akib"
    with pytest.raises(RuntimeError, match="only on"):
        _require_akib_hostname("local-workstation")


def test_launcher_requires_clean_resolvable_git_commit(monkeypatch) -> None:
    def clean(command, **_kwargs):
        if command[-2:] == ("rev-parse", "HEAD"):
            return b"a" * 40 + b"\n"
        return b""

    monkeypatch.setattr(subprocess, "check_output", clean)
    assert _require_clean_source_commit(Path("/repo")) == "a" * 40

    def dirty(command, **_kwargs):
        if command[-2:] == ("rev-parse", "HEAD"):
            return b"a" * 40 + b"\n"
        return b"?? untracked.py\0"

    monkeypatch.setattr(subprocess, "check_output", dirty)
    with pytest.raises(RuntimeError, match="clean Git commit"):
        _require_clean_source_commit(Path("/repo"))


def test_registered_artifact_verification_detects_byte_tampering(tmp_path: Path) -> None:
    run = tmp_path / "run"
    artifact = run / "artifacts" / "payload.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"original")
    result = {
        "artifacts": [
            {
                "path": "artifacts/payload.bin",
                "kind": "payload",
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        ]
    }
    assert _registered_artifacts_intact(run, result)
    artifact.write_bytes(b"tampered")
    assert not _registered_artifacts_intact(run, result)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_complete_reuse_requires_source_config_and_exact_input_hashes(tmp_path: Path) -> None:
    task = TaskBlueprint(
        study_id=TEACHER_PLAN_ID,
        arm_id="teacher-test",
        label="teacher.full-test",
        config=tmp_path / "config.json",
        mode="validate",
    )
    task.config.write_text("{}", encoding="utf-8")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    input_sha = sha256_file(weights)
    config_sha = sha256_file(task.config)
    commit = "b" * 40
    run = tmp_path / "study" / "runs" / task.arm_id / "run-1"
    _write_json(
        run / "manifest.json",
        {
            "schema": "ebl.run",
            "schema_version": 1,
            "run_id": run.name,
            "experiment_id": "mnist_relu.v2",
            "source": {"commit": commit, "dirty": False},
            "runtime": {"executable": "/test/python"},
            "command": [
                "ebl",
                "validate",
                "--config",
                str(task.config),
                "--output-dir",
                str(run.parent),
                "--weights",
                str(weights),
            ],
            "study": {
                "study_id": TEACHER_PLAN_ID,
                "arm_id": task.arm_id,
                "source_config_sha256": config_sha,
            },
            "inputs": [
                {"role": "weights", "path": str(weights), "sha256": input_sha}
            ],
            "config": {
                "path": "config.resolved.json",
                "sha256": content_hash({}),
            },
        },
    )
    _write_json(run / "config.resolved.json", {})
    _write_json(run / "status.json", {"status": "complete"})
    _write_json(
        run / "result.json",
        {
            "run_id": run.name,
            "experiment_id": "mnist_relu.v2",
            "status": "complete",
            "metrics": {
                "split": "test",
                "examples": 10000,
                "accuracy": 0.98,
                "runtime_device": {
                    "configured_device": "cuda",
                    "resolved_device": "cuda:0",
                    "cuda_available": True,
                    "device_name": "test-gpu",
                },
            },
            "artifacts": [],
        },
    )
    found = _completed_run(
        task=task,
        arm_root=run.parent,
        source_commit=commit,
        config_sha256=config_sha,
        expected_inputs={"weights": input_sha},
        task_python=Path("/test/python"),
    )
    assert found is not None and found.run_dir == run

    with pytest.raises(RuntimeError, match="mix source commits or input ancestry"):
        _completed_run(
            task=task,
            arm_root=run.parent,
            source_commit=commit,
            config_sha256=config_sha,
            expected_inputs={"weights": "c" * 64},
            task_python=Path("/test/python"),
        )

    with pytest.raises(RuntimeError, match="mix source commits or input ancestry"):
        _completed_run(
            task=task,
            arm_root=run.parent,
            source_commit="d" * 40,
            config_sha256=config_sha,
            expected_inputs={"weights": input_sha},
            task_python=Path("/test/python"),
        )


def test_progress_cadence_is_stricter_than_fifteen_seconds() -> None:
    assert 0 < HEARTBEAT_SECONDS <= MAXIMUM_HEARTBEAT_SECONDS <= 15


def test_development_grid_is_exactly_six_candidates_by_four_starts() -> None:
    adam = [task for task in development_blueprints() if task.stage_kind == "on_chip_adam"]
    assert len(adam) == 24
    assert len({(task.learning_rate, task.pulse_cap_per_cell) for task in adam}) == 6
    assert Counter(task.start_state for task in adam) == {
        "hwa_healthy_p0": 6,
        "hwa_published_fault": 6,
        "scratch_healthy_p0": 6,
        "scratch_published_fault": 6,
    }
    assert all(task.study_id == DEVELOPMENT_PLAN_ID for task in adam)


def _state_evaluation(value: float, *, include_test: bool) -> dict:
    result = {}
    splits = ("validation", "test") if include_test else ("validation",)
    for split in splits:
        row = {
            "examples": 5000 if split == "validation" else 10000,
            "student_accuracy": value,
            "cross_entropy": 1.0 - value,
            "kl_teacher_student": value / 10.0,
            "teacher_agreement": value + 0.1,
        }
        result[split] = {
            "apparent_forward": dict(row),
            "persistent_diagnostic": dict(row),
        }
    return result


def _effective_evaluation(value: float) -> dict:
    return {
        split: {
            "examples": 5000 if split == "validation" else 10000,
            "student_accuracy": value,
            "cross_entropy": 1.0 - value,
            "kl_teacher_student": value / 10.0,
            "teacher_agreement": value + 0.1,
        }
        for split in ("validation", "test")
    }


def test_hwa_validation_requires_held_apparent_primary_and_nonpersistent_diagnostic() -> None:
    metrics = {
        "examples": 5000,
        "student_accuracy": 0.9,
        "cross_entropy": 0.2,
        "kl_teacher_student": 0.1,
        "teacher_agreement": 0.91,
    }
    value = {
        "network_forward_state": "held_apparent_q",
        "primary_state": "held_apparent_q",
        "diagnostic_state_role": "nonpersistent_support_clamped_digital_master_q",
        "persistent_device_state_present": False,
        "apparent_forward": dict(metrics),
        "nonpersistent_support_clamped_digital_master_diagnostic": dict(metrics),
        "held_apparent_state_receipt": {
            "schema": "ebl.ibm_om_crossbar_held_apparent_hwa_evaluation",
            "schema_version": 1,
            "evaluation_id": "epoch_010.validation",
            "resolved_seed": 123,
            "resampling": "one_full_array_draw_held_across_complete_evaluation_cohort",
            "support_clamped_digital_master_q_sha256": "a" * 64,
            "held_apparent_q_sha256": "b" * 64,
            "write_noise_q_sha256": "c" * 64,
            "generator_state_before_sha256": "d" * 64,
            "generator_state_after_sha256": "e" * 64,
            "configured_sigma_q": 0.01,
            "observed_mean_q": 0.0,
            "observed_std_q": 0.01,
            "observed_minimum_q": -0.04,
            "observed_maximum_q": 0.04,
        },
    }
    _validate_hwa_evaluation(
        value,
        label="hwa.final",
        evaluation_id="epoch_010.validation",
        expected_master_sha256="a" * 64,
    )
    value["persistent_device_state_present"] = True
    with pytest.raises(RuntimeError, match="apparent-primary HWA"):
        _validate_hwa_evaluation(
            value,
            label="hwa.final",
            evaluation_id="epoch_010.validation",
            expected_master_sha256="a" * 64,
        )


def test_production_summary_has_exact_population_matching_and_paired_contrasts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    handles = []
    artifacts: dict[tuple[str, int, int], Path] = {}
    fake_states: dict[Path, object] = {}
    arm_offsets = {
        "direct-deploy": 0.40,
        "direct-corrupt": 0.30,
        "hwa-deploy": 0.45,
        "hwa-corrupt": 0.34,
        "hwa-adam-healthy": 0.55,
        "hwa-adam-corrupt": 0.50,
        "scratch-deploy-preparation": 0.10,
        "scratch-corrupt-preparation": 0.05,
        "scratch-adam-healthy": 0.48,
        "scratch-adam-corrupt": 0.43,
    }
    parent_arm = {
        "direct-corrupt": "direct-deploy",
        "hwa-corrupt": "hwa-deploy",
        "hwa-adam-healthy": "hwa-deploy",
        "hwa-adam-corrupt": "hwa-corrupt",
        "scratch-corrupt-preparation": "scratch-deploy-preparation",
        "scratch-adam-healthy": "scratch-deploy-preparation",
        "scratch-adam-corrupt": "scratch-corrupt-preparation",
    }
    for task in production_blueprints():
        identity = (task.arm_id, task.assignment_seed, task.endpoint_seed)
        run = tmp_path / task.label
        run.mkdir(parents=True)
        artifact = run / "state.pt"
        artifact.write_bytes(task.label.encode())
        artifacts[identity] = artifact
        parent = parent_arm.get(task.arm_id)
        parent_sha = (
            None
            if parent is None
            else sha256_file(artifacts[(parent, task.assignment_seed, task.endpoint_seed)])
        )
        expected_role = (
            "adam_final"
            if task.stage_kind == "on_chip_adam"
            else "faulted_p0"
            if task.stage_kind == "apply_corruption"
            else "healthy_p0"
        )
        fake_states[artifact.resolve()] = SimpleNamespace(
            assignment_seed=task.assignment_seed,
            endpoint_seed=task.endpoint_seed,
            parent_device_state_sha256=parent_sha,
            role=expected_role,
            source_kind=task.source_kind,
            healthy_population=SimpleNamespace(
                fingerprint=f"healthy-{task.assignment_seed}"
            ),
            published_population=SimpleNamespace(
                fingerprint=f"published-{task.assignment_seed}"
            ),
        )
        value = arm_offsets[task.arm_id]
        metrics = {"final_evaluation": _state_evaluation(value, include_test=True)}
        if task.stage_kind == "deploy":
            metrics["source_evaluation"] = _effective_evaluation(value + 0.10)
        elif task.stage_kind == "apply_corruption":
            metrics["pre_fault"] = _state_evaluation(value + 0.10, include_test=True)
        elif task.stage_kind == "on_chip_adam":
            metrics["initial"] = _state_evaluation(value - 0.10, include_test=True)
        result = {
            "metrics": metrics,
            "artifacts": [{"kind": task.expected_artifact_kind, "path": "state.pt"}],
        }
        _write_json(run / "result.json", result)
        handles.append(RunHandle(task, run, {}, result, False))
    monkeypatch.setattr(
        launcher,
        "load_device_state",
        lambda path: fake_states[Path(path).resolve()],
    )

    analysis = tmp_path / "study" / "analysis"
    analysis.mkdir(parents=True)
    path = _write_production_summary(analysis.parent, handles)
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value["outcomes"]["direct-deploy"]["summaries"][
        "test.apparent.student_accuracy"
    ]
    assert summary["primary_assignment_count"] == 4
    assert summary["secondary_pooled_count"] == 16
    assert len(summary["assignment_means"]) == 4
    assert value["schema_version"] == 2
    assert set(value["matched_population_fingerprints_by_assignment"]) == {
        "2090501",
        "2090502",
        "2090503",
        "2090504",
    }
    contrasts = value["paired_contrasts"]
    assert set(contrasts) == {
        "direct-source-to-deployment",
        "hwa-source-to-deployment",
        "direct-fault-damage",
        "hwa-fault-damage",
        "hwa-healthy-recovery-gain",
        "hwa-corrupt-recovery-gain",
        "scratch-healthy-training-gain",
        "scratch-corrupt-training-gain",
        "hwa-vs-direct-deployment",
        "hwa-vs-direct-corrupted",
    }
    hwa_delta = contrasts["hwa-vs-direct-deployment"]["summaries"][
        "test.apparent.student_accuracy"
    ]
    assert hwa_delta["primary_mean"] == pytest.approx(0.05)
    assert len(contrasts["direct-fault-damage"]["rows"]) == 16

    mismatched = next(iter(fake_states.values()))
    mismatched.healthy_population.fingerprint = "wrong-population"
    mismatch_analysis = tmp_path / "mismatched-study" / "analysis"
    mismatch_analysis.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="one matched assignment population"):
        _write_production_summary(mismatch_analysis.parent, handles)


def test_production_summary_rejects_partial_counts_and_population_mismatch(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="160-run production coverage"):
        _write_production_summary(tmp_path, [])


def test_production_metrics_reject_incomplete_validation_or_test_counts(
    tmp_path: Path,
) -> None:
    task = next(
        task for task in production_blueprints() if task.arm_id == "direct-deploy"
    )
    evaluation = _state_evaluation(0.8, include_test=True)
    evaluation["validation"]["apparent_forward"]["examples"] = 4999
    run = tmp_path / "run"
    run.mkdir()
    result = {"metrics": {"final_evaluation": evaluation}}
    _write_json(run / "result.json", result)
    with pytest.raises(RuntimeError, match="exactly 5000 examples"):
        _production_metric_rows([RunHandle(task, run, {}, result, False)])


def test_gpu_occupancy_probe_fails_closed_on_compute_process(monkeypatch) -> None:
    def run(command, **_kwargs):
        if "--query-gpu=index,name,memory.total,memory.used,utilization.gpu" in command:
            return subprocess.CompletedProcess(command, 0, "0, RTX 3080, 10240, 0, 0\n", "")
        return subprocess.CompletedProcess(command, 0, "1234, python, 9000\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="occupied Akib GPU"):
        _probe_gpu_occupancy(environment={})

    def idle(command, **_kwargs):
        if "--query-gpu=index,name,memory.total,memory.used,utilization.gpu" in command:
            return subprocess.CompletedProcess(command, 0, "0, RTX 3080, 10240, 0, 0\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", idle)
    assert _probe_gpu_occupancy(environment={})["exclusive_at_probe"] is True


def test_prerequisite_sampler_probe_requests_match_canonical_sampler_schema() -> None:
    for policy in ("counterfactual_repaired", "published"):
        request = _sampler_probe_request(policy)
        assert parse_hwa_population_request(
            json.dumps(request, allow_nan=False, sort_keys=True)
        ) == request

    with pytest.raises(ValueError, match="Unsupported sampler probe policy"):
        _sampler_probe_request("unknown")


def test_all_24_tuning_diagnostics_require_epochs_examples_states_and_pulses(
    tmp_path: Path,
) -> None:
    origins = {}
    for start in (
        "hwa_healthy_p0",
        "hwa_published_fault",
        "scratch_healthy_p0",
        "scratch_published_fault",
    ):
        path = tmp_path / f"{start}.pt"
        path.write_bytes(start.encode())
        origins[start] = path
    rows = []
    for task in development_blueprints():
        if task.stage_kind != "on_chip_adam":
            continue
        run = tmp_path / task.label
        run.mkdir(parents=True)
        evaluation = _state_evaluation(0.75, include_test=False)
        epochs = [
            {
                "epoch": epoch,
                "examples": 55000,
                "batches": 3438,
                "test": None,
                "validation": evaluation["validation"],
            }
            for epoch in range(1, 11)
        ]
        optimizer = {
            "optimizer_steps": 34380,
            "requested_nonzero_commands": 100,
            "commanded_pulses": 80,
            "applied_pulses": 80,
            "applied_pulses_by_layer": [70, 10],
            "probability_clipped": 10,
            "blocked_at_cap": 20,
            "pulse_cap_per_cell": task.pulse_cap_per_cell,
            "cells_at_cap": 0,
            "commanded_cells": 60,
            "changed_cells": 60,
            "maximum_pulses_per_cell": 1,
            "enabled_cells": 203264,
        }
        result = {
            "metrics": {
                "initial": evaluation,
                "final_evaluation": evaluation,
                "epochs": epochs,
                "optimizer": optimizer,
            }
        }
        _write_json(run / "result.json", result)
        row = _tuning_diagnostic_row(
            task,
            RunHandle(task, run, {}, result, False),
            origin=origins[task.start_state],
        )
        rows.append(row)
    assert len(rows) == 24
    assert len({(row["learning_rate"], row["pulse_cap_per_cell"], row["start_state"]) for row in rows}) == 24
    assert all(row["epoch_count"] == 10 for row in rows)
    assert all(row["total_training_examples"] == 550000 for row in rows)
    assert all(
        row["final_validation"]["persistent_diagnostic"]["examples"] == 5000
        for row in rows
    )


def test_completed_adam_replays_resume_and_terminal_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = next(
        task
        for task in development_blueprints()
        if task.stage_kind == "on_chip_adam"
    )
    origin = tmp_path / "origin.pt"
    resume = tmp_path / "resume.pt"
    final = tmp_path / "final.pt"
    for path in (origin, resume, final):
        path.write_bytes(path.name.encode())
    selection = {"source": "literal_predeclared_grid"}
    origin_state = SimpleNamespace(assignment_seed=2090402, endpoint_seed=2091402)
    final_state = SimpleNamespace(recovery={"selection": selection})
    monkeypatch.setattr(
        launcher,
        "load_device_state",
        lambda path: final_state if Path(path).resolve() == final.resolve() else origin_state,
    )
    called = []
    monkeypatch.setattr(launcher, "_validate_resume", lambda *args, **kwargs: called.append(kwargs))
    result = {
        "metrics": {
            "source_kind": task.source_kind,
            "start_state": task.start_state,
            "assignment_seed": 2090402,
            "endpoint_seed": 2091402,
            "origin_device_state_sha256": sha256_file(origin),
            "learning_rate": task.learning_rate,
            "pulse_cap_per_cell": task.pulse_cap_per_cell,
            "selection": selection,
            "resume_input_sha256": sha256_file(resume),
        },
        "artifacts": [{"kind": "crossbar_adam_final_state_bundle", "path": str(final)}],
    }
    manifest = {
        "inputs": [
            {
                "role": "adam_epoch_resume",
                "path": str(resume),
                "sha256": sha256_file(resume),
            }
        ]
    }
    # Absolute artifact paths are not native RunStore records, so anchor it via
    # a run directory and a relative sibling for this focused replay check.
    run = tmp_path / "run"
    run.mkdir()
    local_final = run / "final.pt"
    local_final.write_bytes(final.read_bytes())
    final_state_local = final_state
    monkeypatch.setattr(
        launcher,
        "load_device_state",
        lambda path: final_state_local if Path(path).resolve() == local_final.resolve() else origin_state,
    )
    result["artifacts"][0]["path"] = "final.pt"
    handle = RunHandle(task, run, manifest, result, True)
    _replay_completed_adam_ancestry(
        handle,
        origin=origin,
        learning_rate=task.learning_rate,
        pulse_cap=task.pulse_cap_per_cell,
        expected_selection=selection,
    )
    assert len(called) == 1
    final_state_local.recovery["selection"] = {"source": "wrong"}
    with pytest.raises(RuntimeError, match="exact Adam settings"):
        _verify_adam_metrics(
            handle,
            origin=origin,
            learning_rate=task.learning_rate,
            pulse_cap=task.pulse_cap_per_cell,
            expected_selection=selection,
        )
