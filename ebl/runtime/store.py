"""ebl.runtime.store — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import time
from typing import Any, Iterable, Mapping, Sequence
from core.artifacts import ArtifactRecord, RUN_SCHEMA, RUN_SCHEMA_VERSION, _git_identity, atomic_write_json, canonical_json_bytes, content_hash, runtime_identity, sha256_file


class RunStore:
    """Own one immutable run request and its mutable execution records."""

    def __init__(
        self,
        *,
        run_dir: Path,
        manifest: Mapping[str, Any],
        started_monotonic: float,
        repo_root: Path,
    ) -> None:
        self.run_dir = run_dir
        self.manifest = dict(manifest)
        self._started_monotonic = started_monotonic
        self._repo_root = repo_root
        self._finished = False

    @classmethod
    def create(
        cls,
        *,
        output_root: Path | str,
        experiment_id: str,
        resolved_config: Mapping[str, Any],
        command: Sequence[str],
        repo_root: Path | str,
        input_artifacts: Iterable[Mapping[str, Any]] = (),
        resume_capability: str = "exact",
        run_id: str | None = None,
    ) -> "RunStore":
        """Create a run directory exclusively and write its immutable request."""

        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError(
                "Expected experiment_id to be a non-empty string. "
                f"Provided value: {experiment_id!r}."
            )
        if resume_capability not in {
            "exact",
            "stateful_nondeterministic",
            "unsupported",
        }:
            raise ValueError(
                "Expected resume_capability to be 'exact', "
                "'stateful_nondeterministic', or 'unsupported'. "
                f"Provided value: {resume_capability!r}."
            )

        canonical_config = json.loads(canonical_json_bytes(resolved_config))
        config_digest = content_hash(canonical_config)
        if run_id is None:
            timestamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S.%fZ"
            )
            run_id = (
                f"{timestamp}-{config_digest[:8]}-{secrets.token_hex(4)}"
            )
        if (
            not run_id
            or run_id in {".", ".."}
            or "/" in run_id
            or os.sep in run_id
        ):
            raise ValueError(
                "Expected run_id to be one non-empty path component. "
                f"Provided value: {run_id!r}."
            )

        output_path = Path(output_root).expanduser().resolve()
        # Canonical ``results/<study-id>/runs/<arm-id>`` roots are linked to
        # their predeclared study before any run directory is created.  The
        # import stays lazy so ordinary runs retain the lightweight artifact
        # boundary and do not pay for study parsing.
        study_context = None
        if (
            output_path.parent.name == "runs"
            and (output_path.parent.parent / "study.json").is_file()
        ):
            from experiments.study_workflow import study_context_for_run

            study_context = study_context_for_run(
                output_path,
                experiment_id=experiment_id,
                command=command,
            )
        output_path.mkdir(parents=True, exist_ok=True)
        run_dir = output_path / run_id
        run_dir.mkdir(parents=False, exist_ok=False)
        (run_dir / "logs").mkdir()
        (run_dir / "checkpoints").mkdir()
        (run_dir / "artifacts").mkdir()

        now = datetime.now(timezone.utc).isoformat()
        repo_path = Path(repo_root).expanduser().resolve()
        manifest = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "created_at": now,
            "command": list(command),
            "working_directory": str(Path.cwd()),
            "config": {
                "path": "config.resolved.json",
                "sha256": config_digest,
            },
            "inputs": list(input_artifacts),
            "source": _git_identity(repo_path),
            "runtime": runtime_identity(),
            "resume_capability": resume_capability,
        }
        if study_context is not None:
            manifest["study"] = study_context
        atomic_write_json(run_dir / "config.resolved.json", canonical_config)
        atomic_write_json(run_dir / "manifest.json", manifest)
        atomic_write_json(
            run_dir / "status.json",
            {
                "schema": RUN_SCHEMA,
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": run_id,
                "status": "running",
                "started_at": now,
                "finished_at": None,
                "duration_seconds": None,
                "error": None,
            },
        )
        store = cls(
            run_dir=run_dir,
            manifest=manifest,
            started_monotonic=time.monotonic(),
            repo_root=repo_path,
        )
        return store

    def append_metric(self, record: Mapping[str, Any]) -> None:
        """Append one canonical JSON record and make it visible immediately."""

        if self._finished:
            raise RuntimeError("Expected an active run before appending metrics.")
        payload = canonical_json_bytes(dict(record)) + b"\n"
        path = self.run_dir / "metrics.jsonl"
        with path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def artifact_record(
        self,
        path: Path | str,
        *,
        kind: str,
    ) -> ArtifactRecord:
        """Validate and describe a materialized artifact below the run root."""

        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.run_dir / candidate
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(self.run_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                "Expected artifact path to be inside the run directory. "
                f"Provided value: {str(resolved)!r}."
            ) from exc
        if not resolved.is_file():
            raise ValueError(
                "Expected artifact path to name a file. "
                f"Provided value: {str(resolved)!r}."
            )
        return ArtifactRecord(
            path=relative.as_posix(),
            sha256=sha256_file(resolved),
            size_bytes=resolved.stat().st_size,
            kind=kind,
        )

    def complete(
        self,
        *,
        metrics: Mapping[str, Any],
        artifacts: Iterable[ArtifactRecord] = (),
    ) -> Path:
        """Finalize a successful run and return ``result.json``."""

        if self._finished:
            raise RuntimeError("Expected an active run before completion.")
        finished_at = datetime.now(timezone.utc).isoformat()
        duration = time.monotonic() - self._started_monotonic
        artifact_values = [asdict(record) for record in artifacts]
        result = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.manifest["run_id"],
            "experiment_id": self.manifest["experiment_id"],
            "status": "complete",
            "finished_at": finished_at,
            "duration_seconds": duration,
            "metrics": dict(metrics),
            "artifacts": artifact_values,
            "error": None,
        }
        result_path = self.run_dir / "result.json"
        atomic_write_json(result_path, result)
        atomic_write_json(
            self.run_dir / "status.json",
            {
                "schema": RUN_SCHEMA,
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": self.manifest["run_id"],
                "status": "complete",
                "started_at": self.manifest["created_at"],
                "finished_at": finished_at,
                "duration_seconds": duration,
                "error": None,
            },
        )
        self._finished = True
        return result_path

    def fail(self, error: BaseException | Mapping[str, Any]) -> Path:
        """Atomically record a failed terminal state."""

        if self._finished:
            raise RuntimeError("Expected an active run before failure.")
        if isinstance(error, BaseException):
            error_record: Mapping[str, Any] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        else:
            error_record = dict(error)
        finished_at = datetime.now(timezone.utc).isoformat()
        duration = time.monotonic() - self._started_monotonic
        status = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.manifest["run_id"],
            "status": "failed",
            "started_at": self.manifest["created_at"],
            "finished_at": finished_at,
            "duration_seconds": duration,
            "error": error_record,
        }
        path = self.run_dir / "status.json"
        atomic_write_json(path, status)
        self._finished = True
        return path
