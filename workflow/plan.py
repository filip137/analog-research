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
from workflow.lifecycle import (
    BASE_SOURCE,
    STAGE_OUTPUTS,
    Execution,
    Lifecycle,
    StageConfig,
    stage_inputs,
    stage_selections,
)
from workflow.networks import family

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
    frozen_sources: dict[str, Path] | None = None,
    retention_sources: dict[str, Path] | None = None,
    lora_bases: dict[str, Path] | None = None,
) -> dict[str, bytes]:
    """Return every planned file (relative path -> bytes)."""

    teacher_weights = teacher_weights.resolve()
    digest = sha256_file(teacher_weights)
    if digest != family(lifecycle.network).pinned_digest(lifecycle.network):
        raise ValueError(
            f"Expected the pinned teacher checkpoint of {lifecycle.network.describe()}. "
            f"Provided value: {teacher_weights} with SHA-256 {digest}."
        )
    frozen_sources = {arm: path.resolve() for arm, path in (frozen_sources or {}).items()}
    if set(frozen_sources) - {a.arm_id for a in lifecycle.hwa.arms}:
        raise ValueError("Expected frozen sources to name declared HWA arms.")
    source_receipts = {arm: {"path": str(path), "sha256": sha256_file(path)}
                       for arm, path in frozen_sources.items()}
    retention_sources = {key: path.resolve() for key, path in (retention_sources or {}).items()}
    expected_retention = {f"{s.hwa_arm}:{s.array_seed}" for s in stage_selections(lifecycle) if s.kind == "compensate"}
    if set(retention_sources) != expected_retention:
        raise ValueError("Expected --retention-source ARM:SEED=PATH for every compensate stage, and only those stages.")
    retention_receipts = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in retention_sources.items()}
    lora_bases = {arm: path.resolve() for arm, path in (lora_bases or {}).items()}
    expected_bases = {a.arm_id: a.base_source for a in lifecycle.hwa.arms
                      if a.method == "lora" and a.arm_id not in frozen_sources}
    if set(lora_bases) != set(expected_bases):
        raise ValueError("Expected --lora-base ARM=PATH for every LoRA arm, and only those arms.")
    base_receipts = {arm: {"path": str(path), "sha256": sha256_file(path)} for arm, path in lora_bases.items()}
    for arm, receipt in base_receipts.items():
        if receipt["sha256"] != expected_bases[arm].sha256:
            raise ValueError(f"Expected the pinned LoRA base of {arm!r}. Provided value: {receipt}.")
    files: dict[str, bytes] = {}
    stages, summary = [], []
    for selection in stage_selections(lifecycle):
        if selection.kind == "hwa" and selection.hwa_arm in frozen_sources:
            continue
        config = StageConfig(stage=selection, execution=execution, lifecycle=lifecycle).to_dict()
        name = f"stages/{selection.stage_id}.json"
        files[name] = _pretty(config)
        inputs, depends = {}, []
        for role, upstream in stage_inputs(selection, lifecycle).items():
            if upstream is None:
                inputs[role] = {"path": str(teacher_weights)}
            elif upstream == BASE_SOURCE:
                inputs[role] = {"path": base_receipts[selection.hwa_arm]["path"]}
            elif role == "weights" and selection.hwa_arm in frozen_sources:
                inputs[role] = {"path": str(frozen_sources[selection.hwa_arm])}
            elif role == "device_state" and selection.kind == "compensate":
                inputs[role] = {"path": str(retention_sources[f"{selection.hwa_arm}:{selection.array_seed}"])}
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
            **({"frozen_sources": source_receipts} if source_receipts else {}),
            **({"retention_sources": retention_receipts} if retention_receipts else {}),
            **({"lora_bases": base_receipts} if base_receipts else {}),
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
