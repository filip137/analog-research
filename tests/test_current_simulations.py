from __future__ import annotations

import json
from pathlib import Path

from experiments.artifacts import RunStore
from experiments.current_simulations import (
    ACTIVE_BEGIN,
    ACTIVE_END,
    discover_active_runs,
    refresh_current_simulations_for_run,
    render_active_section,
)


def _write_native_run(
    repo_root: Path,
    *,
    study_id: str,
    arm: str,
    run_id: str,
    status: str,
) -> Path:
    run_dir = repo_root / "results" / study_id / "runs" / arm / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "ebl.run",
                "run_id": run_id,
                "experiment_id": "mnist_relu.v1",
                "command": ["ebl", "train", "--config", "config.json"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "schema": "ebl.run",
                "run_id": run_id,
                "status": status,
                "started_at": "2026-07-30T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_ledger(repo_root: Path) -> Path:
    path = repo_root / "docs" / "current_simulations.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "# Current simulations",
                "",
                ACTIVE_BEGIN,
                "## Active",
                "",
                "Old generated content.",
                ACTIVE_END,
                "",
                "## Queued next",
                "",
                "Keep this manual study.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_discover_active_runs_groups_only_running_native_statuses(
    tmp_path: Path,
) -> None:
    _write_native_run(
        tmp_path,
        study_id="study-a",
        arm="base",
        run_id="run-001",
        status="running",
    )
    _write_native_run(
        tmp_path,
        study_id="study-a",
        arm="lora",
        run_id="run-002",
        status="running",
    )
    _write_native_run(
        tmp_path,
        study_id="study-a",
        arm="validate",
        run_id="run-004",
        status="complete",
    )
    malformed = (
        tmp_path
        / "results"
        / "study-b"
        / "runs"
        / "base"
        / "run-003"
    )
    malformed.mkdir(parents=True)
    (malformed / "status.json").write_text("{", encoding="utf-8")

    active = discover_active_runs(tmp_path)

    assert len(active) == 2
    assert {item.study_id for item in active} == {"study-a"}
    assert {item.arm for item in active} == {"base", "lora"}
    assert {item.relative_run_dir for item in active} == {
        "study-a/runs/base/run-001",
        "study-a/runs/lora/run-002",
    }
    assert {item.mode for item in active} == {"train"}
    rendered = render_active_section(active)
    assert rendered.count("### `study-a`") == 1
    assert "**Active native runs:** `2`" in rendered


def test_refresh_preserves_historical_document(tmp_path: Path) -> None:
    ledger = _write_ledger(tmp_path)
    original = ledger.read_bytes()
    run_dir = _write_native_run(tmp_path, study_id="study-a", arm="lora", run_id="run-001", status="running")
    assert not refresh_current_simulations_for_run(repo_root=tmp_path, run_dir=run_dir)
    assert ledger.read_bytes() == original


def test_refresh_ignores_runs_outside_canonical_results(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "other" / "run-001"
    outside.mkdir(parents=True)

    assert not refresh_current_simulations_for_run(
        repo_root=tmp_path,
        run_dir=outside,
    )
    assert not (tmp_path / "docs" / "current_simulations.md").exists()


def test_refresh_preserves_unmarked_manual_document(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "docs" / "current_simulations.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("# User-owned document\n", encoding="utf-8")
    run_dir = _write_native_run(
        tmp_path,
        study_id="study-a",
        arm="base",
        run_id="run-001",
        status="running",
    )

    assert not refresh_current_simulations_for_run(
        repo_root=tmp_path,
        run_dir=run_dir,
    )
    assert ledger.read_text(encoding="utf-8") == "# User-owned document\n"


def test_run_store_preserves_history_on_create_and_completion(
    tmp_path: Path,
) -> None:
    ledger = _write_ledger(tmp_path)
    original = ledger.read_text(encoding="utf-8")
    store = RunStore.create(
        output_root=tmp_path / "results" / "study-a" / "runs" / "base",
        experiment_id="mnist_relu.v1",
        resolved_config={"schema_version": 1},
        command=["ebl", "train", "--config", "config.json"],
        repo_root=tmp_path,
        run_id="run-001",
    )

    active = ledger.read_text(encoding="utf-8")
    assert active == original
    assert "Keep this manual study." in active

    store.complete(metrics={})

    terminal = ledger.read_text(encoding="utf-8")
    assert terminal == original
    assert "Keep this manual study." in terminal


def test_run_store_preserves_history_on_failure(
    tmp_path: Path,
) -> None:
    ledger = _write_ledger(tmp_path)
    original = ledger.read_text(encoding="utf-8")
    store = RunStore.create(
        output_root=tmp_path / "results" / "study-a" / "runs" / "base",
        experiment_id="mnist_relu.v1",
        resolved_config={"schema_version": 1},
        command=["ebl", "train", "--config", "config.json"],
        repo_root=tmp_path,
        run_id="run-001",
    )
    assert ledger.read_text(encoding="utf-8") == original

    store.fail(RuntimeError("numerical failure"))

    terminal = ledger.read_text(encoding="utf-8")
    assert terminal == original
    assert "Keep this manual study." in terminal


def test_run_store_can_defer_live_ledger_updates_for_atomic_multirun_launch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ledger = _write_ledger(tmp_path)
    original = ledger.read_text(encoding="utf-8")
    monkeypatch.setenv("EBL_DEFER_CURRENT_SIMULATIONS", "1")
    store = RunStore.create(
        output_root=tmp_path / "results" / "study-a" / "runs" / "base",
        experiment_id="mnist_relu.v1",
        resolved_config={"schema_version": 1},
        command=["ebl", "train", "--config", "config.json"],
        repo_root=tmp_path,
        run_id="run-001",
    )

    assert ledger.read_text(encoding="utf-8") == original
    store.complete(metrics={})
    assert ledger.read_text(encoding="utf-8") == original


def test_refresh_failure_never_fails_the_native_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import experiments.current_simulations

    def fail_refresh(**_kwargs) -> bool:
        raise OSError("read-only documentation")

    monkeypatch.setattr(
        experiments.current_simulations,
        "refresh_current_simulations_for_run",
        fail_refresh,
    )
    store = RunStore.create(
        output_root=tmp_path / "results" / "study-a" / "runs" / "base",
        experiment_id="mnist_relu.v1",
        resolved_config={"schema_version": 1},
        command=["ebl", "train", "--config", "config.json"],
        repo_root=tmp_path,
        run_id="run-001",
    )

    store.complete(metrics={})
    assert json.loads((store.run_dir / "status.json").read_text())[
        "status"
    ] == "complete"
