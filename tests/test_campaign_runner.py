from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import campaigns.runner as runner
import pytest
from campaigns.schema import CampaignSpec, TargetSpec


def _target(tmp_path: Path) -> dict:
    return {
        "id": "target",
        "worktree": str(tmp_path / "worktree"),
        "python": str(tmp_path / "python"),
    }


def _complete_result(
    run_dir: Path,
    *,
    artifacts: list[dict] | None = None,
    **overrides,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ebl.run",
        "schema_version": 1,
        "run_id": run_dir.name,
        "experiment_id": "mnist_relu.v1",
        "status": "complete",
        "finished_at": "2026-07-27T00:00:00+00:00",
        "duration_seconds": 0.0,
        "metrics": {},
        "artifacts": artifacts or [],
        "error": None,
        **overrides,
    }
    path = run_dir / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _artifact(run_dir: Path, relative: str, *, kind: str) -> dict:
    path = run_dir / relative
    return {
        "path": relative,
        "sha256": runner.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "kind": kind,
    }


def _preflight(_target, *, allow_dirty):
    return {
        "source": {
            "commit": "abc123",
            "dirty": bool(allow_dirty),
            "dirty_hash": "dirty" if allow_dirty else None,
        },
        "description": {
            "protocol_version": 1,
            "capabilities": {
                "supported_commands": [
                    "train",
                    "linspace",
                    "validate",
                ],
                "resume": {
                    "weights": True,
                    "weights": True,
                    "full_training_state": True,
                },
            },
            "commands": {
                "train": {
                    "available": True,
                    "required_options": ["--config", "--output-dir"],
                    "exclusive_input_options": [
                        "--weights",
                        "--device-model",
                        "--resume",
                    ],
                },
                "linspace": {
                    "available": True,
                    "required_options": [
                        "--config",
                        "--output-dir",
                        "--weights",
                    ],
                },
                "validate": {
                    "available": True,
                    "required_options": [
                        "--config",
                        "--output-dir",
                        "--weights",
                    ],
                },
            },
        },
    }


def test_dry_run_plans_stage_references_without_launching(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    (tmp_path / "validate.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "dry",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "digits",
                "target": "target",
                "command": "train",
                "config": "train.json",
            },
            {
                "id": "validate",
                "case_id": "digits",
                "target": "target",
                "command": "validate",
                "config": "validate.json",
                "depends_on": ["train"],
                "inputs": {
                    "weights": {
                        "stage": "train",
                        "artifact_kind": "weights",
                    }
                },
            },
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)

    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        dry_run=True,
    )

    assert records["train"]["status"] == "dry_run"
    assert records["validate"]["status"] == "dry_run"
    assert "@stage:train:weights" in records["validate"]["command"]
    assert records["train"]["target_source"]["commit"] == "abc123"


def test_campaign_records_manifest_provenance_and_resolved_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "campaign_id": "provenance",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "case",
                "target": "target",
                "command": "train",
                "config": "train.json",
            }
        ],
    }
    manifest_path = tmp_path / "campaign.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)

    runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        manifest_path=manifest_path,
        dry_run=True,
    )
    # An exact rerun is accepted and retains the same immutable contract.
    runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        manifest_path=manifest_path,
        dry_run=True,
        resume=True,
    )

    resolved_path = (
        tmp_path
        / "outputs"
        / "provenance"
        / "campaign.resolved.json"
    )
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    contract = runner._resolved_campaign_contract(spec)
    assert resolved["source_manifest"] == {
        "path": str(manifest_path.resolve()),
        "sha256": runner.sha256_file(manifest_path),
    }
    assert resolved["resolved_contract_sha256"] == runner.content_hash(contract)
    assert {
        key: resolved[key]
        for key in ("schema_version", "campaign_id", "targets", "stages")
    } == contract


def test_campaign_refuses_changed_contract_for_existing_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "campaign_id": "immutable",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "first",
                "target": "target",
                "command": "train",
                "config": "train.json",
            }
        ],
    }
    manifest_path = tmp_path / "campaign.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "_preflight", _preflight)
    runner.run_campaign(
        CampaignSpec.parse(manifest, base_dir=tmp_path),
        output_root=tmp_path / "outputs",
        manifest_path=manifest_path,
        dry_run=True,
    )
    resolved_path = (
        tmp_path / "outputs" / "immutable" / "campaign.resolved.json"
    )
    original = resolved_path.read_bytes()

    changed = json.loads(json.dumps(manifest))
    changed["stages"][0]["case_id"] = "changed"
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identical resolved contract"):
        runner.run_campaign(
            CampaignSpec.parse(changed, base_dir=tmp_path),
            output_root=tmp_path / "outputs",
            manifest_path=manifest_path,
            dry_run=True,
        )

    assert resolved_path.read_bytes() == original


def test_failure_skips_dependents_but_not_independent_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in ("fail", "dependent", "independent"):
        (tmp_path / f"{name}.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "continue",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "fail",
                "case_id": "first",
                "target": "target",
                "command": "train",
                "config": "fail.json",
            },
            {
                "id": "dependent",
                "case_id": "first",
                "target": "target",
                "command": "validate",
                "config": "dependent.json",
                "depends_on": ["fail"],
                "inputs": {
                    "weights": {
                        "stage": "fail",
                        "artifact_kind": "weights",
                    }
                },
            },
            {
                "id": "independent",
                "case_id": "second",
                "target": "target",
                "command": "train",
                "config": "independent.json",
            },
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)

    def fake_run(command, **_kwargs):
        if any(str(argument).endswith("/fail.json") for argument in command):
            return SimpleNamespace(
                returncode=7,
                stdout="",
                stderr="intentional failure",
            )
        output_dir = Path(command[command.index("--output-dir") + 1])
        _complete_result(output_dir / "run")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
    )

    assert records["fail"]["status"] == "failed"
    assert records["dependent"]["status"] == "skipped_dependency"
    assert records["independent"]["status"] == "complete"


def test_preflight_rejects_unadvertised_stage_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "unsupported",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "case",
                "target": "target",
                "command": "train",
                "config": "train.json",
            }
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)

    def unsupported(_target, *, allow_dirty):
        result = _preflight(_target, allow_dirty=allow_dirty)
        result["description"]["capabilities"]["supported_commands"] = [
            "validate"
        ]
        return result

    monkeypatch.setattr(runner, "_preflight", unsupported)
    with pytest.raises(RuntimeError, match="advertised by its target"):
        runner.run_campaign(
            spec,
            output_root=tmp_path / "outputs",
            dry_run=True,
        )


def test_preflight_rejects_advertised_but_unavailable_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    spec = CampaignSpec.parse(
        {
            "schema_version": 1,
            "campaign_id": "unavailable",
            "targets": [_target(tmp_path)],
            "stages": [
                {
                    "id": "train",
                    "case_id": "case",
                    "target": "target",
                    "command": "train",
                    "config": "train.json",
                }
            ],
        },
        base_dir=tmp_path,
    )

    def unavailable(_target, *, allow_dirty):
        result = _preflight(_target, allow_dirty=allow_dirty)
        result["description"]["commands"]["train"]["available"] = False
        return result

    monkeypatch.setattr(runner, "_preflight", unavailable)
    with pytest.raises(RuntimeError, match="available on its target"):
        runner.run_campaign(
            spec,
            output_root=tmp_path / "outputs",
            dry_run=True,
        )


def test_device_model_remains_an_explicit_hyphenated_stage_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    (tmp_path / "model.pt").write_bytes(b"weights")
    manifest = {
        "schema_version": 1,
        "campaign_id": "base-weights",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "case",
                "target": "target",
                "command": "train",
                "config": "train.json",
                "inputs": {"device_model": {"path": "model.pt"}},
            }
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)

    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        dry_run=True,
    )

    command = records["train"]["command"]
    assert "--device-model" in command
    assert "--device_model" not in command
    assert str((tmp_path / "model.pt").resolve()) in command


def test_optional_external_inputs_are_validated_and_forwarded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in (
        "train.json",
        "validate.json",
        "devices.hdf5",
        "device-model.json",
        "teacher.pt",
        "weights.pt",
    ):
        (tmp_path / name).write_bytes(b"input")
    manifest = {
        "schema_version": 1,
        "campaign_id": "external-inputs",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "case",
                "target": "target",
                "command": "train",
                "config": "train.json",
                "inputs": {
                    "device_data": {"path": "devices.hdf5"},
                    "device_model": {"path": "device-model.json"},
                    "teacher_weights": {"path": "teacher.pt"},
                },
            },
            {
                "id": "validate",
                "case_id": "case",
                "target": "target",
                "command": "validate",
                "config": "validate.json",
                "inputs": {
                    "weights": {"path": "weights.pt"},
                    "teacher_weights": {"path": "teacher.pt"},
                },
            },
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)

    def supports_external_inputs(_target, *, allow_dirty):
        result = _preflight(_target, allow_dirty=allow_dirty)
        commands = result["description"]["commands"]
        commands["train"]["optional_input_options"] = [
            "--device-data",
            "--device-model",
            "--teacher-weights",
        ]
        commands["validate"]["optional_input_options"] = [
            "--teacher-weights"
        ]
        return result

    monkeypatch.setattr(runner, "_preflight", supports_external_inputs)
    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        dry_run=True,
    )

    train_command = records["train"]["command"]
    assert train_command[
        train_command.index("--device-data") + 1
    ] == str((tmp_path / "devices.hdf5").resolve())
    assert train_command[
        train_command.index("--teacher-weights") + 1
    ] == str((tmp_path / "teacher.pt").resolve())
    assert train_command[
        train_command.index("--device-model") + 1
    ] == str((tmp_path / "device-model.json").resolve())
    assert "--device_data" not in train_command
    assert "--teacher_weights" not in train_command

    validate_command = records["validate"]["command"]
    assert validate_command[
        validate_command.index("--teacher-weights") + 1
    ] == str((tmp_path / "teacher.pt").resolve())


@pytest.mark.parametrize(
    ("input_name", "filename"),
    (
        ("device_data", "devices.hdf5"),
        ("teacher_weights", "teacher.pt"),
    ),
)
def test_preflight_rejects_unadvertised_external_input(
    tmp_path: Path,
    monkeypatch,
    input_name: str,
    filename: str,
) -> None:
    (tmp_path / "train.json").write_text("{}", encoding="utf-8")
    (tmp_path / filename).write_bytes(b"input")
    spec = CampaignSpec.parse(
        {
            "schema_version": 1,
            "campaign_id": f"unsupported-{input_name}",
            "targets": [_target(tmp_path)],
            "stages": [
                {
                    "id": "train",
                    "case_id": "case",
                    "target": "target",
                    "command": "train",
                    "config": "train.json",
                    "inputs": {input_name: {"path": filename}},
                }
            ],
        },
        base_dir=tmp_path,
    )
    monkeypatch.setattr(runner, "_preflight", _preflight)

    with pytest.raises(RuntimeError, match="accepted by its target command"):
        runner.run_campaign(
            spec,
            output_root=tmp_path / "outputs",
            dry_run=True,
        )


def test_fail_fast_marks_later_independent_stage_without_launching(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in ("fail", "later"):
        (tmp_path / f"{name}.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "fail-fast",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "fail",
                "case_id": "first",
                "target": "target",
                "command": "train",
                "config": "fail.json",
            },
            {
                "id": "later",
                "case_id": "second",
                "target": "target",
                "command": "train",
                "config": "later.json",
            },
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)
    launches = []

    def fail(command, **_kwargs):
        launches.append(command)
        return SimpleNamespace(returncode=3, stdout="", stderr="failed")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        fail_fast=True,
    )

    assert records["fail"]["status"] == "failed"
    assert records["later"]["status"] == "skipped_fail_fast"
    assert len(launches) == 1


def test_resume_reuses_only_a_matching_complete_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "resume",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "case",
                "target": "target",
                "command": "train",
                "config": "train.json",
            }
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)
    launches = []

    def complete(command, **_kwargs):
        launches.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        _complete_result(output_dir / "run")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", complete)
    first = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
    )
    resumed = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        resume=True,
    )

    assert first["train"]["status"] == "complete"
    assert resumed["train"]["status"] == "complete"
    assert resumed["train"]["fingerprint"] == first["train"]["fingerprint"]
    assert len(launches) == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "other.run", "schema to be 'ebl.run'"),
        ("schema_version", 2, "schema_version to be 1"),
        ("status", "failed", "status to be 'complete'"),
    ],
)
def test_child_result_rejects_invalid_protocol_fields(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    output_dir = tmp_path / "runs"
    _complete_result(output_dir / "run", **{field: value})

    with pytest.raises(RuntimeError, match=message):
        runner._find_run_result(output_dir)


def test_child_result_requires_exact_keys(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    result_path = _complete_result(output_dir / "run")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="keys to be exactly"):
        runner._find_run_result(output_dir)


def test_malformed_child_result_is_recorded_as_failed_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}", encoding="utf-8")
    spec = CampaignSpec.parse(
        {
            "schema_version": 1,
            "campaign_id": "malformed-child",
            "targets": [_target(tmp_path)],
            "stages": [
                {
                    "id": "train",
                    "case_id": "case",
                    "target": "target",
                    "command": "train",
                    "config": "train.json",
                }
            ],
        },
        base_dir=tmp_path,
    )
    monkeypatch.setattr(runner, "_preflight", _preflight)

    def malformed(command, **_kwargs):
        output_dir = Path(command[command.index("--output-dir") + 1])
        run_dir = output_dir / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(
            json.dumps({"status": "complete"}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", malformed)
    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
    )

    assert records["train"]["status"] == "failed"
    assert records["train"]["run_result"] is None
    assert records["train"]["error"]["type"] == "RuntimeError"
    assert "keys to be exactly" in records["train"]["error"]["message"]


def test_child_result_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "run"
    output_dir.mkdir()
    outside = output_dir / "outside.pt"
    outside.write_bytes(b"outside")
    record = {
        "path": "../outside.pt",
        "sha256": runner.sha256_file(outside),
        "size_bytes": outside.stat().st_size,
        "kind": "weights",
    }
    _complete_result(run_dir, artifacts=[record])

    with pytest.raises(RuntimeError, match="normalized relative path"):
        runner._find_run_result(output_dir)


@pytest.mark.parametrize("duplicate", ["path", "kind"])
def test_child_result_rejects_duplicate_artifacts(
    tmp_path: Path,
    duplicate: str,
) -> None:
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    first = run_dir / "artifacts" / "first.bin"
    second = run_dir / "artifacts" / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    first_record = _artifact(run_dir, "artifacts/first.bin", kind="weights")
    second_record = _artifact(
        run_dir,
        (
            "artifacts/first.bin"
            if duplicate == "path"
            else "artifacts/second.bin"
        ),
        kind=("resume" if duplicate == "path" else "weights"),
    )
    _complete_result(
        run_dir,
        artifacts=[first_record, second_record],
    )

    with pytest.raises(RuntimeError, match=f"artifact {duplicate}s.*unique"):
        runner._find_run_result(output_dir)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (b"other!!", "content to match sha256"),
        (b"changed-size", "size to match size_bytes"),
    ],
)
def test_child_result_rejects_tampered_artifact(
    tmp_path: Path,
    replacement: bytes,
    message: str,
) -> None:
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    weights = run_dir / "checkpoints" / "weights.pt"
    weights.write_bytes(b"weights")
    record = _artifact(run_dir, "checkpoints/weights.pt", kind="weights")
    _complete_result(run_dir, artifacts=[record])
    weights.write_bytes(replacement)

    with pytest.raises(RuntimeError, match=message):
        runner._find_run_result(output_dir)


def test_upstream_artifact_is_reverified_before_use(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    weights = run_dir / "checkpoints" / "weights.pt"
    weights.write_bytes(b"weights")
    result_path = _complete_result(
        run_dir,
        artifacts=[
            _artifact(run_dir, "checkpoints/weights.pt", kind="weights")
        ],
    )
    runner._validate_run_result(result_path)
    stage_records = {
        "train": {
            "status": "complete",
            "run_result": str(result_path),
            "run_result_sha256": runner.sha256_file(result_path),
        }
    }

    weights.write_bytes(b"altered")
    with pytest.raises(RuntimeError, match="content to match sha256"):
        runner._resolved_input(
            runner.InputRef(stage="train", artifact_kind="weights"),
            stage_records=stage_records,
        )


def test_dirty_identity_hashes_untracked_source_contents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "new_module.py"
    source.write_text("VALUE = 1\n")
    target = TargetSpec(
        target_id="target",
        worktree=tmp_path,
        python=tmp_path / "python",
    )

    def check_output(command, **_kwargs):
        if command[-2:] == ("rev-parse", "HEAD"):
            return b"abc123\n"
        if "status" in command:
            return b"?? new_module.py\0"
        if "diff" in command:
            return b""
        raise AssertionError(command)

    monkeypatch.setattr(runner.subprocess, "check_output", check_output)
    first = runner._git_identity(target)
    source.write_text("VALUE = 2\n")
    second = runner._git_identity(target)

    assert first["dirty"] is True
    assert first["dirty_hash"] != second["dirty_hash"]


def test_preflight_requires_target_python_to_be_executable(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o644)
    target = TargetSpec(
        target_id="target",
        worktree=worktree,
        python=python,
    )

    with pytest.raises(ValueError, match="existing executable"):
        runner._preflight(target, allow_dirty=False)
