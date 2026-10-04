from __future__ import annotations

from collections import Counter
from json import dumps, loads
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.artifacts import RunStore, sha256_file
from experiments.mnist_analog_relu.generate_hwa_recovery_canary import (
    CHECKPOINT_POLICY,
    FRESH_DRAW_SEEDS,
    LEARNING_RATES,
    OBJECTIVES,
    SOURCE_COMMIT,
    START_STATES,
    build_documents,
)
from experiments.mnist_analog_relu import hwa_recovery_canary as campaign
from experiments.mnist_analog_relu.staged_config import (
    FreshApparentDiagnosticStageSettings,
    OnChipAdamDiagnosticStageSettings,
    parse_staged_crossbar_config,
)
from experiments.study_workflow import load_study_plan, prepare_study


ROOT = Path(__file__).resolve().parents[1]


def test_generated_canary_is_complete_strict_and_byte_stable() -> None:
    configs, plan, shard_manifest = build_documents(ROOT)
    assert len(configs) == 30
    assert len(plan["arms"]) == 30
    plan_path = (
        ROOT
        / "studies/mnist-ibm-om-crossbar-hwa-recovery-canary-20260905-v1.json"
    )
    assert loads(plan_path.read_text(encoding="utf-8")) == plan
    assert load_study_plan(plan_path)["study_id"] == plan["study_id"]

    observed = Counter()
    fresh = []
    for path, expected in configs.items():
        assert loads(path.read_text(encoding="utf-8")) == expected
        parsed = parse_staged_crossbar_config(expected)
        assert parsed.runtime.device == "cuda"
        assert parsed.evaluation.profile == "diagnostic_validation_only"
        assert parsed.device.assignment_seed == 2090402
        assert parsed.device.endpoint_seed == 2091402
        if isinstance(parsed.stage, OnChipAdamDiagnosticStageSettings):
            observed[
                (
                    parsed.stage.start_state,
                    parsed.stage.objective,
                    parsed.stage.hyperparameters.learning_rate,
                )
            ] += 1
            assert parsed.stage.epochs == 3
            assert parsed.stage.hyperparameters.pulse_cap_per_cell == 640
            assert (
                parsed.stage.checkpoint_policy == CHECKPOINT_POLICY
            )
        else:
            assert isinstance(parsed.stage, FreshApparentDiagnosticStageSettings)
            fresh.append(parsed.stage.start_state)

    assert observed == Counter(
        {
            (state, objective, rate): 1
            for state in START_STATES
            for objective in OBJECTIVES
            for rate in LEARNING_RATES
        }
    )
    assert sorted(fresh) == sorted(START_STATES)
    assert len(set(FRESH_DRAW_SEEDS.values())) == 1
    assert shard_manifest == loads(
        (
            ROOT
            / "campaigns/manifests/"
            "mnist_ibm_om_crossbar_hwa_recovery_canary_dual_host.json"
        ).read_text(encoding="utf-8")
    )


def test_dual_host_split_is_disjoint_balanced_and_keeps_objective_pairs_local() -> None:
    local = campaign.load_shard_tasks("local")
    akib = campaign.load_shard_tasks("akib")
    assert len(local) == 16
    assert len(akib) == 14
    assert {task.arm_id for task in local}.isdisjoint(
        {task.arm_id for task in akib}
    )
    assert len({task.arm_id for task in (*local, *akib)}) == 30

    shard_by_arm = {
        task.arm_id: shard
        for shard, tasks in (("local", local), ("akib", akib))
        for task in tasks
    }
    for state in ("healthy", "faulted"):
        for token in (
            "0",
            "3em6",
            "1em5",
            "3em5",
            "1em4",
            "2em4",
            "3em4",
        ):
            assert (
                shard_by_arm[f"adam-{state}-ce-lr{token}"]
                == shard_by_arm[f"adam-{state}-kl-lr{token}"]
            )
    for shard, tasks in (("local", local), ("akib", akib)):
        adam_ids = [task.arm_id for task in tasks if task.arm_id.startswith("adam-")]
        assert len(adam_ids) == 14
        assert sum("-ce-" in arm_id for arm_id in adam_ids) == 7
        assert sum("-kl-" in arm_id for arm_id in adam_ids) == 7
        assert sorted(
            (
                sum("adam-healthy-" in arm_id for arm_id in adam_ids),
                sum("adam-faulted-" in arm_id for arm_id in adam_ids),
            )
        ) == [6, 8]
    assert {
        task.arm_id for task in local if task.arm_id.startswith("fresh-apparent-")
    } == {"fresh-apparent-healthy", "fresh-apparent-faulted"}
    assert not any(
        task.arm_id.startswith("fresh-apparent-") for task in akib
    )


def test_import_reference_pins_original_clean_commit_and_exact_chain() -> None:
    reference = loads(
        (
            ROOT
            / "studies/references/"
            "mnist_ibm_om_crossbar_hwa_p0_1f4ef1b8_20260905.json"
        ).read_text(encoding="utf-8")
    )
    assert reference["source"]["commit"] == SOURCE_COMMIT
    assert reference["source"]["dirty"] is False
    assert reference["artifacts"]["hwa_faulted_p0"]["parent_artifact_sha256"] == (
        reference["artifacts"]["hwa_healthy_p0"]["sha256"]
    )
    assert reference["artifacts"]["hwa_healthy_p0"]["plant_state_sha256"] == (
        "c4b6112b7b5de2dbf11ca4018d107c3965701cd4d15e7f796ac240ecf9b2e999"
    )
    assert reference["artifacts"]["hwa_faulted_p0"]["plant_state_sha256"] == (
        "dc8d3665def8f8fbee1f937761fd1563e8233a0e279ebab80135c78585443a24"
    )
    assert set(reference["provenance_files"]) == {
        "teacher_manifest.json",
        "teacher_result.json",
        "hwa_manifest.json",
        "hwa_result.json",
        "healthy_manifest.json",
        "healthy_result.json",
        "faulted_manifest.json",
        "faulted_result.json",
    }


def _fake_import_fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    input_dir = root / "inputs"
    provenance_dir = input_dir / "provenance"
    provenance_dir.mkdir(parents=True)
    hashes = {
        "teacher_weights": "1" * 64,
        "hwa_master": "2" * 64,
        "hwa_healthy_p0": "3" * 64,
        "hwa_faulted_p0": "4" * 64,
    }
    artifacts = {}
    for role, filename in campaign.INPUT_FILENAMES.items():
        (input_dir / filename).write_bytes(b"x")
        artifacts[role] = {
            "filename": filename,
            "sha256": hashes[role],
            "size_bytes": 1,
            "apparent_q_sha256": f"{role}-apparent",
            "persistent_q_sha256": f"{role}-persistent",
            "plant_state_sha256": f"{role}-plant",
        }
    artifacts["hwa_healthy_p0"].update(
        {
            "role": "healthy_p0",
            "source_kind": "hwa_master",
            "assignment_seed": 2090402,
            "endpoint_seed": 2091402,
        }
    )
    artifacts["hwa_faulted_p0"].update(
        {
            "role": "faulted_p0",
            "source_kind": "hwa_master",
            "assignment_seed": 2090402,
            "endpoint_seed": 2091402,
            "parent_artifact_sha256": hashes["hwa_healthy_p0"],
        }
    )
    common_source = {
        "available": True,
        "commit": SOURCE_COMMIT,
        "dirty": False,
        "dirty_hash": None,
    }
    source = {
        "commit": SOURCE_COMMIT,
        "dirty": False,
        "teacher_study_id": "teacher-study",
        "teacher_study_sha256": "a" * 64,
        "teacher_source_plan_sha256": "b" * 64,
        "teacher_source_config_sha256": "c" * 64,
        "study_id": "hwa-study",
        "study_sha256": "d" * 64,
        "source_plan_sha256": "e" * 64,
        "hwa_source_config_sha256": "f" * 64,
        "healthy_source_config_sha256": "c55b2dee67c826ae724b8a93063327904b164af657695adfb5c0f13a3456c85e",
        "faulted_source_config_sha256": "b3a3261011898eb1d96de704b4da9829042013e7ae5429af94c7286b8606fcb1",
    }

    def manifest(study_id: str, arm: str, study_sha: str, plan_sha: str, config_sha: str, inputs: list[dict[str, str]]) -> dict[str, object]:
        return {
            "source": common_source,
            "study": {
                "study_id": study_id,
                "arm_id": arm,
                "study_sha256": study_sha,
                "source_plan_sha256": plan_sha,
                "source_config_sha256": config_sha,
            },
            "inputs": inputs,
        }

    def result(kind: str, digest: str) -> dict[str, object]:
        return {"status": "complete", "artifacts": [{"kind": kind, "sha256": digest}]}

    documents = {
        "teacher_manifest.json": manifest(
            "teacher-study", "teacher", "a" * 64, "b" * 64, "c" * 64, []
        ),
        "teacher_result.json": result("selected_named_weights", hashes["teacher_weights"]),
        "hwa_manifest.json": manifest(
            "hwa-study",
            "offchip-hwa",
            "d" * 64,
            "e" * 64,
            "f" * 64,
            [{"role": "teacher_weights", "sha256": hashes["teacher_weights"]}],
        ),
        "hwa_result.json": result("crossbar_hwa_master", hashes["hwa_master"]),
        "healthy_manifest.json": manifest(
            "hwa-study",
            "tune-hwa-deploy",
            "d" * 64,
            "e" * 64,
            "c55b2dee67c826ae724b8a93063327904b164af657695adfb5c0f13a3456c85e",
            [
                {"role": "teacher_weights", "sha256": hashes["teacher_weights"]},
                {"role": "hwa_master", "sha256": hashes["hwa_master"]},
            ],
        ),
        "healthy_result.json": result(
            "crossbar_deployed_state_bundle", hashes["hwa_healthy_p0"]
        ),
        "faulted_manifest.json": manifest(
            "hwa-study",
            "tune-hwa-corrupt",
            "d" * 64,
            "e" * 64,
            "b3a3261011898eb1d96de704b4da9829042013e7ae5429af94c7286b8606fcb1",
            [
                {"role": "teacher_weights", "sha256": hashes["teacher_weights"]},
                {"role": "origin_device_state", "sha256": hashes["hwa_healthy_p0"]},
            ],
        ),
        "faulted_result.json": result(
            "crossbar_corrupted_state_bundle", hashes["hwa_faulted_p0"]
        ),
    }
    documents["teacher_result.json"]["metrics"] = {
        "acceptance_gate": {"passed": True},
        "selected": {"accuracy": 0.9732},
    }
    documents["hwa_result.json"]["metrics"] = {
        "teacher_sha256": hashes["teacher_weights"],
        "runtime": {"resolved_device": "cuda:0"},
    }
    documents["healthy_result.json"]["metrics"] = {
        "teacher_sha256": hashes["teacher_weights"],
        "source_artifact_sha256": hashes["hwa_master"],
        "device_state_sha256": hashes["hwa_healthy_p0"],
        "runtime": {"resolved_device": "cuda:0"},
    }
    documents["faulted_result.json"]["metrics"] = {
        "teacher_sha256": hashes["teacher_weights"],
        "parent_device_state_sha256": hashes["hwa_healthy_p0"],
        "device_state_sha256": hashes["hwa_faulted_p0"],
        "runtime": {"resolved_device": "cuda:0"},
    }
    provenance_hashes = {}
    for index, (filename, document) in enumerate(documents.items(), start=5):
        (provenance_dir / filename).write_text(dumps(document), encoding="utf-8")
        provenance_hashes[filename] = f"{index:x}" * 64
    reference = {
        "schema": "ebl.ibm_om_crossbar_hwa_p0_import_reference",
        "schema_version": 1,
        "source": source,
        "artifacts": artifacts,
        "provenance_files": provenance_hashes,
    }
    reference_path = root / "reference.json"
    reference_path.write_text(dumps(reference), encoding="utf-8")
    return input_dir, reference_path, reference


def test_cross_commit_import_authenticates_semantics_and_rejects_source_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir, reference_path, reference = _fake_import_fixture(tmp_path)
    expected_by_name = {
        record["filename"]: record["sha256"]
        for record in reference["artifacts"].values()
    }
    expected_by_name.update(reference["provenance_files"])
    monkeypatch.setattr(
        campaign,
        "sha256_file",
        lambda path: expected_by_name.get(Path(path).name, "9" * 64),
    )
    healthy = SimpleNamespace(
        role="healthy_p0",
        source_kind="hwa_master",
        assignment_seed=2090402,
        endpoint_seed=2091402,
        teacher_sha256="1" * 64,
        source_artifact_sha256="2" * 64,
        parent_device_state_sha256=None,
        healthy_p0=SimpleNamespace(plant_state_sha256="plant"),
        current=SimpleNamespace(
            plant_state_sha256="hwa_healthy_p0-plant",
            plant_state={
                "apparent": "hwa_healthy_p0-apparent",
                "persistent": "hwa_healthy_p0-persistent",
            }
        ),
    )
    faulted = SimpleNamespace(
        role="faulted_p0",
        source_kind="hwa_master",
        assignment_seed=2090402,
        endpoint_seed=2091402,
        teacher_sha256="1" * 64,
        source_artifact_sha256="2" * 64,
        parent_device_state_sha256="3" * 64,
        healthy_p0=SimpleNamespace(plant_state_sha256="plant"),
        current=SimpleNamespace(
            plant_state_sha256="hwa_faulted_p0-plant",
            plant_state={
                "apparent": "hwa_faulted_p0-apparent",
                "persistent": "hwa_faulted_p0-persistent",
            }
        ),
    )
    monkeypatch.setattr(
        campaign,
        "load_device_state",
        lambda path: healthy if Path(path).name == "hwa_healthy_p0.pt" else faulted,
    )
    monkeypatch.setattr(campaign, "tensor_sha256", lambda value: value)
    receipt = campaign.verify_import_bundle(input_dir, reference_path=reference_path)
    assert receipt["source_commit"] == SOURCE_COMMIT

    manifest_path = input_dir / "provenance/healthy_manifest.json"
    document = loads(manifest_path.read_text(encoding="utf-8"))
    document["source"]["commit"] = "0" * 40
    manifest_path.write_text(dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pinned provenance contract"):
        campaign.verify_import_bundle(input_dir, reference_path=reference_path)


def test_atomic_collection_refuses_nonempty_target_without_copying(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    (target / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        campaign._copy_arm_atomically(source, target, verify_copy=lambda _path: None)
    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (target / "new.txt").exists()


def test_collection_retry_only_accepts_byte_identical_existing_controls(
    tmp_path: Path,
) -> None:
    source_arm = tmp_path / "source" / "run-arm"
    target_arm = tmp_path / "target" / "run-arm"
    source_run = source_arm / "run-001"
    target_run = target_arm / "run-001"
    for run in (source_run, target_run):
        run.mkdir(parents=True)
        for filename in (
            "config.resolved.json",
            "manifest.json",
            "metrics.jsonl",
            "result.json",
            "status.json",
        ):
            (run / filename).write_text(f"{filename}\n", encoding="utf-8")
    verified: list[Path] = []

    assert campaign._authenticate_existing_copy(
        source_arm,
        target_arm,
        verify_copy=verified.append,
    ) is True
    assert verified == [target_arm]

    (target_run / "manifest.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="byte-identical incoming control files"):
        campaign._authenticate_existing_copy(
            source_arm,
            target_arm,
            verify_copy=lambda _path: None,
        )


def test_collect_retry_registers_federated_study_and_requires_ready_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_id = "federated-canary-smoke-v1"
    configs = {}
    for arm_id, setting in (("incoming-arm", 1), ("canonical-arm", 2)):
        config = tmp_path / f"{arm_id}.json"
        config.write_text(dumps({"scientific_setting": setting}), encoding="utf-8")
        configs[arm_id] = config
    plan = tmp_path / "plan.json"
    plan.write_text(
        dumps(
            {
                "schema_version": 1,
                "study_id": study_id,
                "title": "Federated canary smoke",
                "hypothesis": "Both declared shards complete.",
                "motivation": "Exercise immutable collection retry.",
                "evidence_class": "exploratory",
                "arms": [
                    {
                        "arm_id": arm_id,
                        "description": f"{arm_id} description.",
                        "experiment_id": "small_drn.v1",
                        "mode": "train",
                        "configs": [str(configs[arm_id])],
                    }
                    for arm_id in ("incoming-arm", "canonical-arm")
                ],
                "completion_criteria": ["Both runs are valid."],
                "analysis_plan": ["Verify exact coverage."],
            }
        ),
        encoding="utf-8",
    )
    incoming = prepare_study(plan, tmp_path / "incoming-results")
    canonical = prepare_study(plan, tmp_path / "canonical-results")
    incoming_study_path = incoming / "study.json"
    incoming_study = loads(incoming_study_path.read_text(encoding="utf-8"))
    incoming_study["prepared_at"] = "2026-09-05T00:00:00+00:00"
    incoming_study_path.write_text(
        dumps(incoming_study, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def complete(root: Path, arm_id: str, run_id: str) -> Path:
        output = root / "runs" / arm_id
        store = RunStore.create(
            output_root=output,
            experiment_id="small_drn.v1",
            resolved_config=loads(configs[arm_id].read_text(encoding="utf-8")),
            command=(
                "ebl",
                "train",
                "--config",
                str(configs[arm_id]),
                "--output-dir",
                str(output),
            ),
            repo_root=tmp_path,
            run_id=run_id,
        )
        store.append_metric({"metric": 1.0})
        artifact = store.run_dir / "artifacts" / "value.bin"
        artifact.write_bytes(b"value")
        store.complete(
            metrics={"metric": 1.0},
            artifacts=(store.artifact_record(artifact, kind="value"),),
        )
        return store.run_dir

    incoming_run = complete(incoming, "incoming-arm", "run-incoming")
    complete(canonical, "canonical-arm", "run-canonical")
    incoming_task = campaign.ArmTask(
        "incoming-arm",
        configs["incoming-arm"],
        sha256_file(configs["incoming-arm"]),
        "on_chip_adam_diagnostic",
        "hwa_healthy_p0",
    )
    canonical_task = campaign.ArmTask(
        "canonical-arm",
        configs["canonical-arm"],
        sha256_file(configs["canonical-arm"]),
        "on_chip_adam_diagnostic",
        "hwa_healthy_p0",
    )

    monkeypatch.setattr(campaign, "PLAN_PATH", plan)
    monkeypatch.setattr(campaign, "STUDY_ID", study_id)
    monkeypatch.setattr(
        campaign,
        "load_shard_tasks",
        lambda shard: (incoming_task,) if shard == "local" else (canonical_task,),
    )

    def completed_run(
        arm_dir: Path,
        *,
        task: campaign.ArmTask,
        verify_artifacts: bool,
    ) -> tuple[Path, str]:
        del task, verify_artifacts
        runs = [path for path in arm_dir.iterdir() if path.is_dir()]
        assert len(runs) == 1
        return runs[0], "a" * 40

    monkeypatch.setattr(campaign, "_completed_run_for_collection", completed_run)
    sentinel = {"schema_version": 1, "passed": True}
    monkeypatch.setattr(campaign, "_epoch0_parity_sentinel", lambda **_kwargs: sentinel)
    monkeypatch.setattr(campaign, "_adam_stream_sentinel", lambda **_kwargs: sentinel)
    monkeypatch.setattr(campaign, "_fresh_pairing_sentinel", lambda **_kwargs: sentinel)

    first = campaign.collect_shard(
        canonical_study_dir=canonical,
        incoming_study_dir=incoming,
        incoming_shard="local",
        task_python=Path(sys.executable),
    )
    target_run = canonical / "runs" / "incoming-arm" / incoming_run.name
    manifest_before_retry = (target_run / "manifest.json").read_bytes()
    assert first["copied"][0]["collection_action"] == "copied"
    assert first["workflow_summary"]["state"] == "ready_for_review"

    second = campaign.collect_shard(
        canonical_study_dir=canonical,
        incoming_study_dir=incoming,
        incoming_shard="local",
        task_python=Path(sys.executable),
    )
    assert second["copied"][0]["collection_action"] == "authenticated_existing"
    assert second["workflow_summary"]["state"] == "ready_for_review"
    assert (target_run / "manifest.json").read_bytes() == manifest_before_retry
    assert len(list((canonical / "analysis/federated_collections").glob("*.json"))) == 1


def test_collection_summary_rejects_exit_zero_invalid_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = tmp_path / "study"
    analysis = study / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "summary.json").write_text(
        dumps(
            {
                "schema": "ebl.study.summary",
                "schema_version": 1,
                "study_id": "summary-smoke-v1",
                "validation_mode": "full_artifact_hashes",
                "state": "invalid",
                "ready_for_review": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(campaign, "STUDY_ID", "summary-smoke-v1")
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="summary-smoke-v1: invalid (ready_for_review=false)\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="did not reach ready_for_review"):
        campaign._artifact_verified_summary(
            task_python=Path(sys.executable),
            study_dir=study,
        )


def _write_epoch0_result(path: Path, *, prediction_sha: str, offset: float = 0.0) -> None:
    path.mkdir(parents=True)
    state = {
        "examples": 5000,
        "student_correct": 4800,
        "teacher_correct": 4866,
        "prediction_flips_from_teacher": 100,
        "student_prediction_sha256": prediction_sha,
        "teacher_prediction_sha256": "t" * 64,
        "student_accuracy": 0.96,
        "teacher_accuracy": 0.9732,
        "teacher_agreement": 0.98,
        "cross_entropy": 0.1 + offset,
        "kl_teacher_student": 0.02 + offset,
        "student_score_rms": 8.0,
        "teacher_score_rms": 8.2,
    }
    (path / "result.json").write_text(
        dumps(
            {
                "metrics": {
                    "initial": {
                        "validation": {
                            "apparent_forward": state,
                            "persistent_diagnostic": state,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_collection_uses_replicated_epoch0_as_cross_host_parity_sentinel(
    tmp_path: Path,
) -> None:
    tasks = {}
    canonical_runs = {}
    incoming_runs = {}
    for state, token in (
        ("hwa_healthy_p0", "healthy"),
        ("hwa_published_fault", "faulted"),
    ):
        canonical_id = f"adam-{token}-ce-lr0"
        incoming_id = f"adam-{token}-kl-lr3em6"
        tasks[canonical_id] = campaign.ArmTask(
            canonical_id, Path("canonical.json"), "a" * 64, "on_chip_adam_diagnostic", state
        )
        tasks[incoming_id] = campaign.ArmTask(
            incoming_id, Path("incoming.json"), "b" * 64, "on_chip_adam_diagnostic", state
        )
        canonical_run = tmp_path / canonical_id
        incoming_run = tmp_path / incoming_id
        _write_epoch0_result(canonical_run, prediction_sha=token * 8)
        _write_epoch0_result(incoming_run, prediction_sha=token * 8, offset=5e-6)
        canonical_runs[canonical_id] = canonical_run
        incoming_runs[incoming_id] = incoming_run
    receipt = campaign._epoch0_parity_sentinel(
        canonical_runs=canonical_runs,
        incoming_runs=incoming_runs,
        tasks=tasks,
        canonical_shard="akib",
        incoming_shard="local",
    )
    assert receipt["passed"] is True

    bad = next(iter(incoming_runs.values()))
    document = loads((bad / "result.json").read_text(encoding="utf-8"))
    document["metrics"]["initial"]["validation"]["apparent_forward"][
        "student_prediction_sha256"
    ] = "different"
    (bad / "result.json").write_text(dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="prediction parity"):
        campaign._epoch0_parity_sentinel(
            canonical_runs=canonical_runs,
            incoming_runs=incoming_runs,
            tasks=tasks,
            canonical_shard="akib",
            incoming_shard="local",
        )


def test_shard_hostname_gate_is_fail_closed() -> None:
    assert campaign._require_shard_hostname("akib", "integnano-akib") == "integnano-akib"
    assert campaign._require_shard_hostname("local", "workstation") == "workstation"
    with pytest.raises(RuntimeError):
        campaign._require_shard_hostname("akib", "workstation")
    with pytest.raises(RuntimeError):
        campaign._require_shard_hostname("local", "integnano-akib")


def _grid_row(
    arm_id: str, *, accuracy: float, objective_delta: float, learning_rate: float
) -> dict[str, object]:
    return {
        "arm_id": arm_id,
        "learning_rate": learning_rate,
        "selected": {
            "held_apparent_primary": {"student_accuracy": accuracy},
        },
        "selected_delta": {
            "held_apparent_primary": {
                "cross_entropy": objective_delta,
                "kl_teacher_student": objective_delta,
            }
        },
    }


def test_grid_selection_is_accuracy_then_objective_delta_then_lower_lr() -> None:
    lower_accuracy = _grid_row(
        "lower-accuracy", accuracy=0.9700, objective_delta=-0.5, learning_rate=3e-6
    )
    higher_accuracy = _grid_row(
        "higher-accuracy", accuracy=0.9702, objective_delta=0.2, learning_rate=3e-4
    )
    assert campaign._select_best_grid_row(
        [lower_accuracy, higher_accuracy], objective="teacher_kl"
    )["arm_id"] == "higher-accuracy"

    weaker_objective = _grid_row(
        "weaker-objective", accuracy=0.971, objective_delta=-0.01, learning_rate=3e-6
    )
    stronger_objective = _grid_row(
        "stronger-objective", accuracy=0.971, objective_delta=-0.02, learning_rate=3e-4
    )
    assert campaign._select_best_grid_row(
        [weaker_objective, stronger_objective], objective="teacher_kl"
    )["arm_id"] == "stronger-objective"

    high_lr = _grid_row(
        "high-lr", accuracy=0.972, objective_delta=-0.03, learning_rate=2e-4
    )
    low_lr = _grid_row(
        "low-lr", accuracy=0.972, objective_delta=-0.03, learning_rate=1e-4
    )
    assert campaign._select_best_grid_row(
        [high_lr, low_lr], objective="supervised_cross_entropy"
    )["arm_id"] == "low-lr"


def _write_stream_result(path: Path, *, suffix: str = "") -> None:
    path.mkdir(parents=True)
    epochs = [
        {
            "epoch": epoch,
            "ordered_model_inputs_sha256": (str(epoch) * 64)[:64],
            "ordered_labels_sha256": ((str(epoch + 3) * 64)[:64] if not suffix else suffix),
        }
        for epoch in (1, 2, 3)
    ]
    (path / "result.json").write_text(
        dumps({"metrics": {"epochs": epochs}}), encoding="utf-8"
    )


def test_adam_stream_sentinel_requires_exact_cross_arm_epoch_streams(
    tmp_path: Path,
) -> None:
    tasks: dict[str, campaign.ArmTask] = {}
    runs: dict[str, Path] = {}
    for arm_id in ("adam-a", "adam-b"):
        tasks[arm_id] = campaign.ArmTask(
            arm_id,
            Path(f"{arm_id}.json"),
            "a" * 64,
            "on_chip_adam_diagnostic",
            "hwa_healthy_p0",
        )
        runs[arm_id] = tmp_path / arm_id
        _write_stream_result(runs[arm_id])
    assert campaign._adam_stream_sentinel(runs=runs, tasks=tasks)["passed"] is True

    document = loads((runs["adam-b"] / "result.json").read_text(encoding="utf-8"))
    document["metrics"]["epochs"][1]["ordered_labels_sha256"] = "f" * 64
    (runs["adam-b"] / "result.json").write_text(dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stream parity"):
        campaign._adam_stream_sentinel(runs=runs, tasks=tasks)


def _write_fresh_pair_result(path: Path, *, resolved_seed: int = 42) -> None:
    path.mkdir(parents=True)
    seed_derivation = {
        "scheme": "derive_seed_without_start_state_for_paired_noise_v1",
        "configured_seed": 2092501,
        "resolved_seed": resolved_seed,
        "excluded_pairing_dimension": "start_state",
        "matched_across_start_states": True,
    }
    metrics = {
        "configured_seed": 2092501,
        "resolved_seed": resolved_seed,
        "seed_derivation": seed_derivation,
        "generator_state_before_sha256": "a" * 64,
        "generator_state_after_sha256": "b" * 64,
        "draws": [
            {"draw": draw, "noise_q_sha256": f"{draw:x}" * 64}
            for draw in (1, 2, 3, 4)
        ],
    }
    (path / "result.json").write_text(
        dumps({"metrics": metrics}), encoding="utf-8"
    )


def test_fresh_pairing_sentinel_requires_same_seed_and_noise_stream(
    tmp_path: Path,
) -> None:
    tasks: dict[str, campaign.ArmTask] = {}
    runs: dict[str, Path] = {}
    for arm_id, state in (
        ("fresh-apparent-healthy", "hwa_healthy_p0"),
        ("fresh-apparent-faulted", "hwa_published_fault"),
    ):
        tasks[arm_id] = campaign.ArmTask(
            arm_id,
            Path(f"{arm_id}.json"),
            "a" * 64,
            "fresh_apparent_diagnostic",
            state,
        )
        runs[arm_id] = tmp_path / arm_id
        _write_fresh_pair_result(runs[arm_id])
    assert campaign._fresh_pairing_sentinel(
        runs=runs, tasks=tasks, require_pair=True
    )["passed"] is True

    document = loads(
        (runs["fresh-apparent-faulted"] / "result.json").read_text(encoding="utf-8")
    )
    document["metrics"]["resolved_seed"] = 43
    (runs["fresh-apparent-faulted"] / "result.json").write_text(
        dumps(document), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="seed/RNG"):
        campaign._fresh_pairing_sentinel(
            runs=runs, tasks=tasks, require_pair=True
        )
