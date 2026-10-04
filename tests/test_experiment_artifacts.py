from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.artifacts import RunStore, content_hash


def _store(tmp_path: Path, *, run_id: str = "run-001") -> RunStore:
    return RunStore.create(
        output_root=tmp_path,
        experiment_id="mnist_relu.v1",
        resolved_config={"schema_version": 1, "value": 3},
        command=["ebl", "train", "--config", "config.json"],
        repo_root=Path(__file__).parents[1],
        run_id=run_id,
    )


def test_run_store_writes_stable_request_and_completion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_metric({"epoch": 0, "eval/clean/error": 0.25})
    weights = store.run_dir / "checkpoints" / "weights.pt"
    weights.write_bytes(b"weights")
    record = store.artifact_record(weights, kind="weights")
    result_path = store.complete(
        metrics={"eval/clean/error": 0.25},
        artifacts=[record],
    )

    manifest = json.loads((store.run_dir / "manifest.json").read_text())
    status = json.loads((store.run_dir / "status.json").read_text())
    result = json.loads(result_path.read_text())
    assert manifest["experiment_id"] == "mnist_relu.v1"
    assert manifest["config"]["sha256"] == content_hash(
        {"schema_version": 1, "value": 3}
    )
    assert status["status"] == "complete"
    assert result["artifacts"][0]["path"] == "checkpoints/weights.pt"
    assert (store.run_dir / "metrics.jsonl").read_text().count("\n") == 1


def test_run_directories_are_exclusive(tmp_path: Path) -> None:
    _store(tmp_path)
    with pytest.raises(FileExistsError):
        _store(tmp_path)


def test_artifacts_cannot_escape_run_directory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="inside the run directory"):
        store.artifact_record(outside, kind="weights")


def test_failure_is_terminal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.fail(ValueError("bad input"))
    status = json.loads((store.run_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == {
        "message": "bad input",
        "type": "ValueError",
    }
    with pytest.raises(RuntimeError, match="active run"):
        store.append_metric({"epoch": 1})
