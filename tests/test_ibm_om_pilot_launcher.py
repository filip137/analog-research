from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from experiments.mnist_relu_drn.ibm_om_pilot_launcher import (
    STUDY_ID,
    STUDY_PLAN,
    TRAINING_ARM_CONFIGS,
    VALIDATION_ARM_CONFIGS,
    _commands,
    _validate_prepared_study,
    _validate_training_phase_summary,
)
from ebl.lifecycle.study import prepare_study


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


def _summary_arm(
    arm_id: str,
    *,
    complete: int,
    coverage_complete: bool,
) -> dict[str, object]:
    return {
        "arm_id": arm_id,
        "expected_runs": 1,
        "complete": complete,
        "running": 0,
        "failed": 0,
        "invalid": 0,
        "coverage_complete": coverage_complete,
    }


def _training_phase_summary() -> dict[str, object]:
    return {
        "study_id": STUDY_ID,
        "state": "incomplete",
        "ready_for_review": False,
        "validation_mode": "full_artifact_hashes",
        "arms": [
            *(
                _summary_arm(arm, complete=1, coverage_complete=True)
                for arm in TRAINING_ARM_CONFIGS
            ),
            *(
                _summary_arm(arm, complete=0, coverage_complete=False)
                for arm in VALIDATION_ARM_CONFIGS
            ),
        ],
    }


def test_training_launcher_commands_freeze_exact_inputs_and_output_roots(
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
    assert len(VALIDATION_ARM_CONFIGS) == 16
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


def test_training_launcher_accepts_only_the_fresh_prepared_pilot(
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
    assert len(study["arms"]) == 20


def test_training_launcher_rejects_existing_native_or_launcher_attempts(
    tmp_path: Path,
) -> None:
    study_dir = _prepared_study(tmp_path)
    checkpoint, checkpoint_sha, device_model, device_model_sha = _frozen_inputs(
        tmp_path
    )
    (study_dir / "runs" / "train-clean" / "attempt-001").mkdir()

    with pytest.raises(RuntimeError, match="no native or launcher attempts"):
        _validate_prepared_study(
            study_dir,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            device_model_path=device_model,
            device_model_sha256=device_model_sha,
        )

    (study_dir / "runs" / "train-clean" / "attempt-001").rmdir()
    (study_dir / "launch" / "attempt-001").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no native or launcher attempts"):
        _validate_prepared_study(
            study_dir,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            device_model_path=device_model,
            device_model_sha256=device_model_sha,
        )


def test_training_launcher_rejects_changed_frozen_input(tmp_path: Path) -> None:
    study_dir = _prepared_study(tmp_path)
    checkpoint, checkpoint_sha, device_model, device_model_sha = _frozen_inputs(
        tmp_path
    )
    checkpoint.write_bytes(b"post-freeze replacement")

    with pytest.raises(RuntimeError, match="bounded-DRN checkpoint SHA-256"):
        _validate_prepared_study(
            study_dir,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            device_model_path=device_model,
            device_model_sha256=device_model_sha,
        )


def test_training_phase_completion_requires_all_four_train_runs_and_no_tests() -> None:
    summary = _training_phase_summary()

    _validate_training_phase_summary(summary)

    training = next(
        arm
        for arm in summary["arms"]
        if arm["arm_id"] == "train-om-published"
    )
    training["complete"] = 0
    training["coverage_complete"] = False
    with pytest.raises(RuntimeError, match="one valid complete run"):
        _validate_training_phase_summary(summary)


def test_training_phase_completion_rejects_early_heldout_validation() -> None:
    summary = _training_phase_summary()
    heldout = next(
        arm
        for arm in summary["arms"]
        if arm["arm_id"] == "test-clean-83101"
    )
    heldout["complete"] = 1
    heldout["coverage_complete"] = True

    with pytest.raises(RuntimeError, match="to remain pending"):
        _validate_training_phase_summary(summary)
