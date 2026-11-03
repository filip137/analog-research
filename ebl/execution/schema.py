"""Strict campaign manifest schema."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class CampaignManifestError(ValueError):
    """Raised when a campaign file is not strict, valid schema-versioned JSON."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Expected {name} to be an object. Provided value: {value!r}."
        )
    return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Expected {name} to be a non-empty path string. "
            f"Provided value: {value!r}."
        )
    return Path(value).expanduser()


def _keys(
    value: Mapping[str, Any],
    *,
    name: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(
            f"Expected {name} to contain {sorted(required)!r}. "
            f"Provided value: missing keys {missing!r} in {dict(value)!r}."
        )
    if unknown:
        raise ValueError(
            f"Expected {name} keys to be drawn from "
            f"{sorted(required | optional)!r}. "
            f"Provided value: unknown keys {unknown!r} in {dict(value)!r}."
        )


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(
            f"Expected {name} to be one non-empty path-safe string. "
            f"Provided value: {value!r}."
        )
    return value


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    worktree: Path
    python: Path

    @classmethod
    def parse(cls, value: Any, *, base_dir: Path) -> "TargetSpec":
        raw = _mapping(value, "campaign target")
        _keys(
            raw,
            name="campaign target",
            required={"id", "worktree", "python"},
        )
        worktree = _path(raw["worktree"], "campaign target.worktree")
        if not worktree.is_absolute():
            worktree = base_dir / worktree
        python = _path(raw["python"], "campaign target.python")
        if not python.is_absolute():
            python = worktree / python
        return cls(
            target_id=_identifier(raw["id"], "campaign target.id"),
            worktree=worktree.resolve(),
            python=python.resolve(),
        )


@dataclass(frozen=True)
class InputRef:
    path: Path | None = None
    stage: str | None = None
    artifact_kind: str | None = None

    @classmethod
    def parse(cls, value: Any, *, base_dir: Path, name: str) -> "InputRef":
        raw = _mapping(value, name)
        _keys(
            raw,
            name=name,
            required=set(),
            optional={"path", "stage", "artifact_kind"},
        )
        has_path = "path" in raw
        has_stage = "stage" in raw or "artifact_kind" in raw
        if has_path == has_stage:
            raise ValueError(
                f"Expected {name} to contain either path or both stage and "
                f"artifact_kind. Provided value: {dict(raw)!r}."
            )
        if has_path:
            path = _path(raw["path"], f"{name}.path")
            if not path.is_absolute():
                path = base_dir / path
            return cls(path=path.resolve())
        if set(raw) != {"stage", "artifact_kind"}:
            raise ValueError(
                f"Expected {name} stage reference to contain exactly stage "
                f"and artifact_kind. Provided value: {dict(raw)!r}."
            )
        return cls(
            stage=_identifier(raw["stage"], f"{name}.stage"),
            artifact_kind=_identifier(
                raw["artifact_kind"],
                f"{name}.artifact_kind",
            ),
        )


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    case_id: str
    target_id: str
    command: str
    config: Path
    depends_on: tuple[str, ...]
    inputs: Mapping[str, InputRef]

    @classmethod
    def parse(cls, value: Any, *, base_dir: Path) -> "StageSpec":
        raw = _mapping(value, "campaign stage")
        _keys(
            raw,
            name="campaign stage",
            required={"id", "case_id", "target", "command", "config"},
            optional={"depends_on", "inputs"},
        )
        command = raw["command"]
        if command not in {"train", "validate", "characterize"}:
            raise ValueError(
                "Expected campaign stage.command to be 'train', "
                f"'validate', or 'characterize'. Provided value: {command!r}."
            )
        config = _path(raw["config"], "campaign stage.config")
        if not config.is_absolute():
            config = base_dir / config
        raw_dependencies = raw.get("depends_on", ())
        if not isinstance(raw_dependencies, (list, tuple)) or any(
            not isinstance(item, str) for item in raw_dependencies
        ):
            raise ValueError(
                "Expected campaign stage.depends_on to be a list of stage "
                f"identifiers. Provided value: {raw_dependencies!r}."
            )
        raw_inputs = _mapping(
            raw.get("inputs", {}),
            "campaign stage.inputs",
        )
        unknown_inputs = sorted(
            set(raw_inputs)
            - {
                "weights",
                "resume",
                "teacher_weights",
                "device_data",
                "device_model",
                "device_state",
                "selection_receipt",
            }
        )
        if unknown_inputs:
            raise ValueError(
                "Expected campaign stage.inputs keys to be drawn from "
                "['device_data', 'device_model', 'device_state', 'resume', "
                "'selection_receipt', 'teacher_weights', 'weights']. "
                "Provided value: "
                "unknown keys "
                f"{unknown_inputs!r} in {dict(raw_inputs)!r}."
            )
        inputs = {
            key: InputRef.parse(
                item,
                base_dir=base_dir,
                name=f"campaign stage.inputs.{key}",
            )
            for key, item in raw_inputs.items()
        }
        return cls(
            stage_id=_identifier(raw["id"], "campaign stage.id"),
            case_id=_identifier(raw["case_id"], "campaign stage.case_id"),
            target_id=_identifier(raw["target"], "campaign stage.target"),
            command=command,
            config=config.resolve(),
            depends_on=tuple(raw_dependencies),
            inputs=MappingProxyType(inputs),
        )


@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str
    targets: tuple[TargetSpec, ...]
    stages: tuple[StageSpec, ...]

    @classmethod
    def parse(
        cls,
        value: Any,
        *,
        base_dir: Path,
    ) -> "CampaignSpec":
        raw = _mapping(value, "campaign manifest")
        _keys(
            raw,
            name="campaign manifest",
            required={"schema_version", "campaign_id", "targets", "stages"},
        )
        if (
            isinstance(raw["schema_version"], bool)
            or not isinstance(raw["schema_version"], int)
            or raw["schema_version"] != 1
        ):
            raise ValueError(
                "Expected campaign schema_version to be 1. "
                f"Provided value: {raw['schema_version']!r}."
            )
        if not isinstance(raw["targets"], list) or not raw["targets"]:
            raise ValueError(
                "Expected campaign targets to be a non-empty list. "
                f"Provided value: {raw['targets']!r}."
            )
        if not isinstance(raw["stages"], list) or not raw["stages"]:
            raise ValueError(
                "Expected campaign stages to be a non-empty list. "
                f"Provided value: {raw['stages']!r}."
            )

        targets = tuple(
            TargetSpec.parse(item, base_dir=base_dir)
            for item in raw["targets"]
        )
        stages = tuple(
            StageSpec.parse(item, base_dir=base_dir)
            for item in raw["stages"]
        )
        target_ids = [target.target_id for target in targets]
        stage_ids = [stage.stage_id for stage in stages]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError(
                "Expected campaign target IDs to be unique. "
                f"Provided value: {target_ids!r}."
            )
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError(
                "Expected campaign stage IDs to be unique. "
                f"Provided value: {stage_ids!r}."
            )
        target_id_set = set(target_ids)
        stage_id_set = set(stage_ids)
        for stage in stages:
            if stage.target_id not in target_id_set:
                raise ValueError(
                    "Expected campaign stage.target to name a declared "
                    f"target. Provided value: {stage.target_id!r}."
                )
            missing_dependencies = sorted(
                set(stage.depends_on) - stage_id_set
            )
            referenced = {
                input_ref.stage
                for input_ref in stage.inputs.values()
                if input_ref.stage is not None
            }
            missing_references = sorted(referenced - stage_id_set)
            if missing_dependencies or missing_references:
                missing = sorted(
                    set(missing_dependencies) | set(missing_references)
                )
                raise ValueError(
                    "Expected campaign stage dependencies to name declared "
                    f"stages. Provided value: missing {missing!r} for "
                    f"{stage.stage_id!r}."
                )
            if not referenced.issubset(set(stage.depends_on)):
                raise ValueError(
                    "Expected stage-based inputs to also appear in "
                    "depends_on. Provided value: "
                    f"stage={stage.stage_id!r}, inputs={sorted(referenced)!r}, "
                    f"depends_on={list(stage.depends_on)!r}."
                )

        _topological_order(stages)
        return cls(
            campaign_id=_identifier(
                raw["campaign_id"],
                "campaign manifest.campaign_id",
            ),
            targets=targets,
            stages=stages,
        )


def _topological_order(stages: tuple[StageSpec, ...]) -> tuple[StageSpec, ...]:
    remaining = {stage.stage_id: stage for stage in stages}
    completed: set[str] = set()
    ordered: list[StageSpec] = []
    while remaining:
        ready = [
            stage
            for stage in stages
            if stage.stage_id in remaining
            and set(stage.depends_on).issubset(completed)
        ]
        if not ready:
            raise ValueError(
                "Expected campaign dependencies to form an acyclic graph. "
                f"Provided value: unresolved stages {sorted(remaining)!r}."
            )
        for stage in ready:
            ordered.append(stage)
            completed.add(stage.stage_id)
            del remaining[stage.stage_id]
    return tuple(ordered)


def topological_stages(spec: CampaignSpec) -> tuple[StageSpec, ...]:
    return _topological_order(spec.stages)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    duplicates = []
    for key, item in pairs:
        if key in value:
            duplicates.append(key)
        value[key] = item
    if duplicates:
        raise CampaignManifestError(
            "Expected every campaign JSON object key to be unique. "
            f"Provided value: duplicate keys {sorted(set(duplicates))!r}."
        )
    return value


def _reject_non_standard_constant(value: str) -> Any:
    raise CampaignManifestError(
        "Expected campaign manifest numbers to use strict JSON syntax. "
        f"Provided value: {value!r}."
    )


def load_campaign_manifest(path: Path | str) -> CampaignSpec:
    """Read a strict JSON manifest and resolve relative paths beside it."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        payload_text = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise CampaignManifestError(
            "Expected --manifest to reference a readable JSON file. "
            f"Provided value: {str(manifest_path)!r}. {error}"
        ) from error
    try:
        payload = json.loads(
            payload_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_standard_constant,
        )
    except CampaignManifestError:
        raise
    except json.JSONDecodeError as error:
        raise CampaignManifestError(
            "Expected --manifest to contain valid strict JSON. "
            f"Provided value: {str(manifest_path)!r} "
            f"(line {error.lineno}, column {error.colno}: {error.msg})."
        ) from error
    try:
        return CampaignSpec.parse(
            payload,
            base_dir=manifest_path.parent,
        )
    except (TypeError, ValueError) as error:
        raise CampaignManifestError(str(error)) from error
