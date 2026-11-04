from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from experiments.reram_program_verify.local_short_launcher import (
    ARM_CONFIGS,
    CAP128_ARM_CONFIGS,
    CAP128_STUDY_ID,
    EXPECTED_TRAJECTORIES_PER_ARM,
    HWA_ARM_CONFIGS,
    HWA_EXPECTED_TRAJECTORIES_PER_ARM,
    HWA_STUDY_ID,
    STUDY_ID,
    _commands,
    _require_clean_source_commit,
    _validate_prepared_study,
)
from ebl.lifecycle.study import prepare_study


_ROOT = Path(__file__).resolve().parents[1]


def test_short_launcher_commands_cover_exact_predeclared_arms(
    tmp_path: Path,
) -> None:
    study_dir = tmp_path / STUDY_ID
    commands = _commands(python=Path("/test/python"), study_dir=study_dir)

    assert set(commands) == set(ARM_CONFIGS)
    assert EXPECTED_TRAJECTORIES_PER_ARM * len(commands) == 2_686_976
    for arm, command in commands.items():
        assert command[:4] == ["/test/python", "-m", "ebl", "characterize"]
        assert Path(command[5]).name == ARM_CONFIGS[arm]
        assert Path(command[7]) == study_dir / "runs" / arm


def test_short_launcher_accepts_the_prepared_v2_study(tmp_path: Path) -> None:
    plan = _ROOT / "studies" / f"{STUDY_ID}.json"
    study_dir = prepare_study(plan, tmp_path)

    _validate_prepared_study(study_dir)


def test_short_launcher_supports_the_prepared_cap128_study(
    tmp_path: Path,
) -> None:
    plan = _ROOT / "studies" / f"{CAP128_STUDY_ID}.json"
    study_dir = prepare_study(plan, tmp_path)
    commands = _commands(
        python=Path("/test/python"),
        study_dir=study_dir,
        arm_configs=CAP128_ARM_CONFIGS,
    )

    _validate_prepared_study(
        study_dir,
        study_id=CAP128_STUDY_ID,
        arm_configs=CAP128_ARM_CONFIGS,
    )
    assert set(commands) == set(CAP128_ARM_CONFIGS)
    assert all("cap128" in Path(command[5]).name for command in commands.values())


def test_short_launcher_supports_the_focused_hwa_om_study(
    tmp_path: Path,
) -> None:
    plan = _ROOT / "studies" / f"{HWA_STUDY_ID}.json"
    study_dir = prepare_study(plan, tmp_path)
    commands = _commands(
        python=Path("/test/python"),
        study_dir=study_dir,
        arm_configs=HWA_ARM_CONFIGS,
    )

    _validate_prepared_study(
        study_dir,
        study_id=HWA_STUDY_ID,
        arm_configs=HWA_ARM_CONFIGS,
    )
    assert set(commands) == {"om-continuous", "om-published"}
    assert HWA_EXPECTED_TRAJECTORIES_PER_ARM * len(commands) == 402_784
    assert all("hwa_production_cap128" in Path(command[5]).name for command in commands.values())


def test_short_launcher_requires_a_clean_source_commit(monkeypatch) -> None:
    def clean_output(command, **kwargs):
        return b"abc123\n" if command[-2:] == ("rev-parse", "HEAD") else b""

    monkeypatch.setattr(subprocess, "check_output", clean_output)
    assert _require_clean_source_commit(Path("/repo")) == "abc123"

    def dirty_output(command, **kwargs):
        if command[-2:] == ("rev-parse", "HEAD"):
            return b"abc123\n"
        return b"?? uncommitted.py\0"

    monkeypatch.setattr(subprocess, "check_output", dirty_output)
    with pytest.raises(RuntimeError, match="clean worktree"):
        _require_clean_source_commit(Path("/repo"))
