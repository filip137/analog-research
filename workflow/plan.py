"""Expand a lifecycle into stage configs and an ``ebl campaign run`` manifest.

Planning launches nothing. The manifest chains stages by artifact kind, so the
existing campaign runner hashes, orders, reuses (``--resume``) and records
every native stage run.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments.artifacts import content_hash, sha256_file
from experiments.cifar_crossbar.prepare import SOURCES
from workflow.lifecycle import (
    STAGE_OUTPUTS,
    Execution,
    Lifecycle,
    StageConfig,
    stage_inputs,
    stage_selections,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "crossbar_lifecycle.plan.v1"


def _pretty(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def build_plan(
    lifecycle: Lifecycle,
    *,
    teacher_weights: Path,
    execution: Execution,
    python: Path,
    source: Path | None = None,
) -> dict[str, bytes]:
    """Return every planned file (relative path -> bytes)."""

    teacher_weights = teacher_weights.resolve()
    digest = sha256_file(teacher_weights)
    if digest != SOURCES[lifecycle.network.dataset][1]:
        raise ValueError(
            f"Expected the pinned public {lifecycle.network.dataset} ResNet-32 teacher. "
            f"Provided value: {teacher_weights} with SHA-256 {digest}."
        )
    files: dict[str, bytes] = {}
    stages, summary = [], []
    for selection in stage_selections(lifecycle):
        config = StageConfig(stage=selection, execution=execution, lifecycle=lifecycle).to_dict()
        name = f"stages/{selection.stage_id}.json"
        files[name] = _pretty(config)
        inputs, depends = {}, []
        for role, upstream in stage_inputs(selection).items():
            if upstream is None:
                inputs[role] = {"path": str(teacher_weights)}
            else:
                inputs[role] = {"stage": upstream[0], "artifact_kind": upstream[1]}
                depends.append(upstream[0])
        stages.append(
            {
                "id": selection.stage_id,
                "case_id": selection.case_id,
                "target": "local",
                "command": "train",
                "config": name,
                "depends_on": sorted(set(depends)),
                "inputs": inputs,
            }
        )
        summary.append(
            {
                **selection.to_dict(),
                "id": selection.stage_id,
                "config": name,
                "config_sha256": content_hash(config),
                "inputs": inputs,
                "outputs": list(STAGE_OUTPUTS[selection.kind]),
            }
        )
    files["campaign.json"] = _pretty(
        {
            "schema_version": 1,
            "campaign_id": lifecycle.lifecycle_id,
            "targets": [{"id": "local", "worktree": str(ROOT), "python": str(python)}],
            "stages": stages,
        }
    )
    files["plan.json"] = _pretty(
        {
            "schema": PLAN_SCHEMA,
            "lifecycle_id": lifecycle.lifecycle_id,
            "lifecycle_sha256": lifecycle.digest(),
            "lifecycle_source": None if source is None else str(source),
            "teacher_weights": {"path": str(teacher_weights), "sha256": digest},
            "execution": execution.to_dict(),
            "stages": summary,
        }
    )
    files["lifecycle.json"] = _pretty(lifecycle.to_dict())
    return files


def write_plan(files: dict[str, bytes], output: Path) -> str:
    """Write a plan atomically per file; refuse to mix with a different plan."""

    output = output.resolve()
    existing = sorted(
        p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
    ) if output.exists() else []
    if existing:
        if set(existing) != set(files) or any((output / k).read_bytes() != v for k, v in files.items()):
            raise ValueError(
                "Expected an empty plan directory or an identical existing plan. "
                f"Provided value: {output}."
            )
        return "unchanged"
    for name, payload in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return "written"


def default_python() -> Path:
    return Path(sys.executable)


def commands(output: Path, runs: Path) -> dict[str, str]:
    manifest = output.resolve() / "campaign.json"
    return {
        "dry_run": f"python -m ebl campaign run --manifest {manifest} --output-dir {runs} --dry-run --allow-dirty",
        "run": f"python -m ebl campaign run --manifest {manifest} --output-dir {runs} --resume",
        "collect": f"python -m workflow collect {runs} --output {runs.parent / 'collected'}",
    }


__all__ = ["build_plan", "commands", "default_python", "write_plan"]
