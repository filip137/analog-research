"""Lightweight study-level workflow around native :mod:`ebl` run bundles.

The numerical runners remain authoritative for science.  This module only
freezes an initial hypothesis, links native runs to a declared study, scans
their small control files, and records a reviewed interpretation.  Metrics
streams and tensor artifacts are deliberately not loaded during normal
summarization; artifact hashing is an explicit opt-in audit.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence


STUDY_SCHEMA = "ebl.study"
STUDY_SCHEMA_VERSION = 1
SUMMARY_SCHEMA = "ebl.study.summary"
FINAL_SCHEMA = "ebl.study.final"
RUN_SCHEMA = "ebl.run"
FEDERATED_COLLECTION_SCHEMA = "ebl.study.federated_collection"
FEDERATED_COLLECTION_SCHEMA_VERSION = 1
_MODES = {"train", "linspace", "validate", "characterize"}
_OUTCOMES = {"supported", "refuted", "mixed", "inconclusive"}
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FEDERATED_CONTROL_FILES = (
    "config.resolved.json",
    "manifest.json",
    "metrics.jsonl",
    "result.json",
    "status.json",
)


class StudyWorkflowError(ValueError):
    """Raised when a study contract or result bundle fails closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_write_bytes(path, payload + b"\n")


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    duplicates: list[str] = []
    for key, value in pairs:
        if key in result:
            duplicates.append(key)
        result[key] = value
    if duplicates:
        raise StudyWorkflowError(
            "Expected every JSON object key to be unique. "
            f"Provided value: duplicate keys {sorted(set(duplicates))!r}."
        )
    return result


def _reject_constant(value: str) -> Any:
    raise StudyWorkflowError(
        "Expected JSON numbers to use finite strict syntax. "
        f"Provided value: {value!r}."
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise StudyWorkflowError(
            f"Expected {label} to be a readable JSON file. "
            f"Provided value: {str(path)!r}. {error}"
        ) from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except StudyWorkflowError:
        raise
    except json.JSONDecodeError as error:
        raise StudyWorkflowError(
            f"Expected {label} to contain strict JSON. "
            f"Provided value: {str(path)!r} "
            f"(line {error.lineno}, column {error.colno}: {error.msg})."
        ) from error
    if not isinstance(value, dict):
        raise StudyWorkflowError(
            f"Expected {label} to contain a JSON object. "
            f"Provided value: {value!r}."
        )
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    label: str,
    required: set[str],
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise StudyWorkflowError(
            f"Expected {label} keys to be exactly {sorted(required)!r}. "
            f"Provided value: missing={missing!r}, unknown={unknown!r}."
        )


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudyWorkflowError(
            f"Expected {label} to be a non-empty string. "
            f"Provided value: {value!r}."
        )
    return value.strip()


def _identifier(value: Any, *, label: str) -> str:
    result = _text(value, label=label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise StudyWorkflowError(
            f"Expected {label} to match '[a-z0-9][a-z0-9._-]*'. "
            f"Provided value: {value!r}."
        )
    return result


def _text_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise StudyWorkflowError(
            f"Expected {label} to be a non-empty list of strings. "
            f"Provided value: {value!r}."
        )
    return [_text(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StudyWorkflowError(
            f"Expected {label} to be a lowercase SHA-256 digest. "
            f"Provided value: {value!r}."
        )
    return value


def load_study_plan(path: Path | str) -> dict[str, Any]:
    """Load and normalize a tracked, immutable study plan."""

    plan_path = Path(path).expanduser().resolve()
    raw = _load_json_object(plan_path, label="--plan")
    required = {
        "schema_version",
        "study_id",
        "title",
        "hypothesis",
        "motivation",
        "evidence_class",
        "arms",
        "completion_criteria",
        "analysis_plan",
    }
    _exact_keys(raw, label="study plan", required=required)
    if (
        isinstance(raw["schema_version"], bool)
        or raw["schema_version"] != STUDY_SCHEMA_VERSION
    ):
        raise StudyWorkflowError(
            "Expected study plan schema_version to be 1. "
            f"Provided value: {raw['schema_version']!r}."
        )
    if not isinstance(raw["arms"], list) or not raw["arms"]:
        raise StudyWorkflowError(
            "Expected study plan arms to be a non-empty list. "
            f"Provided value: {raw['arms']!r}."
        )

    arms: list[dict[str, Any]] = []
    arm_ids: set[str] = set()
    for index, item in enumerate(raw["arms"]):
        if not isinstance(item, dict):
            raise StudyWorkflowError(
                f"Expected study plan arms[{index}] to be an object. "
                f"Provided value: {item!r}."
            )
        _exact_keys(
            item,
            label=f"study plan arms[{index}]",
            required={
                "arm_id",
                "description",
                "experiment_id",
                "mode",
                "configs",
            },
        )
        arm_id = _identifier(item["arm_id"], label=f"study plan arms[{index}].arm_id")
        if arm_id in arm_ids:
            raise StudyWorkflowError(
                "Expected study plan arm_id values to be unique. "
                f"Provided value: {arm_id!r}."
            )
        arm_ids.add(arm_id)
        mode = item["mode"]
        if mode not in _MODES:
            raise StudyWorkflowError(
                "Expected study plan arm mode to be 'train', 'linspace', "
                f"'validate', or 'characterize'. Provided value: {mode!r}."
            )
        configs = item["configs"]
        if not isinstance(configs, list) or not configs:
            raise StudyWorkflowError(
                f"Expected study plan arms[{index}].configs to be a non-empty "
                f"list of paths. Provided value: {configs!r}."
            )
        config_records: list[dict[str, Any]] = []
        config_hashes: set[str] = set()
        for config_index, configured_path in enumerate(configs):
            config_text = _text(
                configured_path,
                label=f"study plan arms[{index}].configs[{config_index}]",
            )
            config_path = Path(config_text).expanduser()
            if not config_path.is_absolute():
                config_path = plan_path.parent / config_path
            config_path = config_path.resolve()
            if not config_path.is_file():
                raise StudyWorkflowError(
                    "Expected every declared study config to be an existing "
                    f"file. Provided value: {str(config_path)!r}."
                )
            digest = _sha256_file(config_path)
            if digest in config_hashes:
                raise StudyWorkflowError(
                    "Expected each arm config to have distinct content. "
                    f"Provided value: duplicate SHA-256 {digest!r}."
                )
            config_hashes.add(digest)
            config_records.append(
                {
                    "declared_path": config_text,
                    "resolved_path": str(config_path),
                    "sha256": digest,
                }
            )
        arms.append(
            {
                "arm_id": arm_id,
                "description": _text(
                    item["description"],
                    label=f"study plan arms[{index}].description",
                ),
                "experiment_id": _identifier(
                    item["experiment_id"],
                    label=f"study plan arms[{index}].experiment_id",
                ),
                "mode": mode,
                "configs": config_records,
            }
        )

    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": _identifier(raw["study_id"], label="study plan study_id"),
        "title": _text(raw["title"], label="study plan title"),
        "hypothesis": _text(raw["hypothesis"], label="study plan hypothesis"),
        "motivation": _text(raw["motivation"], label="study plan motivation"),
        "evidence_class": _identifier(
            raw["evidence_class"], label="study plan evidence_class"
        ),
        "arms": arms,
        "completion_criteria": _text_list(
            raw["completion_criteria"], label="study plan completion_criteria"
        ),
        "analysis_plan": _text_list(raw["analysis_plan"], label="study plan analysis_plan"),
        "source_plan": {
            "path": str(plan_path),
            "sha256": _sha256_file(plan_path),
        },
    }


def _study_readme(study: Mapping[str, Any]) -> str:
    arms = "\n".join(
        f"- `{arm['arm_id']}`: {arm['description']} "
        f"({len(arm['configs'])} declared run(s))"
        for arm in study["arms"]
    )
    criteria = "\n".join(f"- {item}" for item in study["completion_criteria"])
    analysis = "\n".join(f"- {item}" for item in study["analysis_plan"])
    return (
        f"# {study['title']}\n\n"
        f"Study ID: `{study['study_id']}`  \n"
        f"Evidence class: `{study['evidence_class']}`\n\n"
        "## Initial hypothesis\n\n"
        f"{study['hypothesis']}\n\n"
        "## Motivation\n\n"
        f"{study['motivation']}\n\n"
        "## Declared arms\n\n"
        f"{arms}\n\n"
        "## Completion criteria\n\n"
        f"{criteria}\n\n"
        "## Analysis plan\n\n"
        f"{analysis}\n\n"
        "## Workflow\n\n"
        "Write each native run below `runs/<arm-id>/` using `python -m ebl`. "
        "Then run `python -m ebl study summarize --study-dir .`. Review in "
        "the owning pilot/result note. The optional legacy `study finalize` "
        "command writes only `analysis/final.json`; historical global "
        "documents are not updated.\n"
    )


def _materialized_contract(study: Mapping[str, Any]) -> dict[str, Any]:
    """Return the scientific contract without checkout-local absolute paths."""

    return {
        key: study[key]
        for key in (
            "schema",
            "schema_version",
            "study_id",
            "title",
            "hypothesis",
            "motivation",
            "evidence_class",
            "completion_criteria",
            "analysis_plan",
        )
    } | {
        "source_plan_sha256": study["source_plan"]["sha256"],
        "arms": [
            {
                key: arm[key]
                for key in (
                    "arm_id",
                    "description",
                    "experiment_id",
                    "mode",
                )
            }
            | {
                "configs": [
                    {
                        "declared_path": config["declared_path"],
                        "sha256": config["sha256"],
                    }
                    for config in arm["configs"]
                ]
            }
            for arm in study["arms"]
        ],
    }


def prepare_study(plan_path: Path | str, results_root: Path | str) -> Path:
    """Materialize a tracked plan beneath ``results/<study-id>``."""

    plan = load_study_plan(plan_path)
    root = Path(results_root).expanduser().resolve() / plan["study_id"]
    record = {
        "schema": STUDY_SCHEMA,
        **plan,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    study_path = root / "study.json"
    if root.exists():
        if not study_path.is_file():
            raise StudyWorkflowError(
                "Expected an existing study directory to contain study.json. "
                f"Provided value: {str(root)!r}."
            )
        existing = load_study_record(root)
        if existing["source_plan"]["sha256"] != plan["source_plan"]["sha256"]:
            raise StudyWorkflowError(
                "Expected an existing study directory to retain the same "
                "source-plan hash. Provided value: "
                f"directory={str(root)!r}, requested={plan['source_plan']['sha256']!r}."
            )
        if _materialized_contract(existing) != _materialized_contract(record):
            raise StudyWorkflowError(
                "Expected an existing prepared study to match its tracked "
                "source plan exactly. "
                f"Provided value: directory={str(root)!r}."
            )
        required_directories = [root / "analysis", root / "runs"] + [
            root / "runs" / arm["arm_id"] for arm in existing["arms"]
        ]
        missing = [str(path) for path in required_directories if not path.is_dir()]
        if missing:
            raise StudyWorkflowError(
                "Expected an existing prepared study to retain its analysis "
                "and declared arm directories. "
                f"Provided value: missing={missing!r}."
            )
        return root

    root.mkdir(parents=True, exist_ok=False)
    (root / "analysis").mkdir()
    runs = root / "runs"
    runs.mkdir()
    for arm in plan["arms"]:
        (runs / arm["arm_id"]).mkdir()
    _atomic_write_json(study_path, record)
    _atomic_write_text(root / "README.md", _study_readme(record))
    return root


def _load_study_record(
    study_dir: Path | str,
    *,
    require_directory_name: bool,
) -> dict[str, Any]:
    root = Path(study_dir).expanduser().resolve()
    path = root / "study.json"
    value = _load_json_object(path, label="study.json")
    required = {
        "schema",
        "schema_version",
        "study_id",
        "title",
        "hypothesis",
        "motivation",
        "evidence_class",
        "arms",
        "completion_criteria",
        "analysis_plan",
        "source_plan",
        "prepared_at",
    }
    _exact_keys(value, label="study.json", required=required)
    if (
        value["schema"] != STUDY_SCHEMA
        or isinstance(value["schema_version"], bool)
        or value["schema_version"] != STUDY_SCHEMA_VERSION
    ):
        raise StudyWorkflowError(
            f"Expected study.json schema to be {STUDY_SCHEMA!r} version 1. "
            f"Provided value: schema={value['schema']!r}, "
            f"version={value['schema_version']!r}."
        )
    _identifier(value["study_id"], label="study.json study_id")
    if require_directory_name and root.name != value["study_id"]:
        raise StudyWorkflowError(
            "Expected the study directory name to equal study_id. "
            f"Provided value: directory={root.name!r}, study_id={value['study_id']!r}."
        )
    if not isinstance(value["arms"], list) or not value["arms"]:
        raise StudyWorkflowError(
            "Expected study.json arms to be a non-empty list. "
            f"Provided value: {value['arms']!r}."
        )
    _text(value["title"], label="study.json title")
    _text(value["hypothesis"], label="study.json hypothesis")
    _text(value["motivation"], label="study.json motivation")
    _identifier(value["evidence_class"], label="study.json evidence_class")
    _text(value["prepared_at"], label="study.json prepared_at")
    _text_list(value["completion_criteria"], label="study.json completion_criteria")
    _text_list(value["analysis_plan"], label="study.json analysis_plan")
    source_plan = value["source_plan"]
    if not isinstance(source_plan, dict):
        raise StudyWorkflowError(
            "Expected study.json source_plan to be an object. "
            f"Provided value: {source_plan!r}."
        )
    _exact_keys(
        source_plan,
        label="study.json source_plan",
        required={"path", "sha256"},
    )
    _text(source_plan["path"], label="study.json source_plan.path")
    _digest(source_plan["sha256"], label="study.json source_plan.sha256")
    arm_ids: set[str] = set()
    for index, arm in enumerate(value["arms"]):
        if not isinstance(arm, dict):
            raise StudyWorkflowError(
                f"Expected study.json arms[{index}] to be an object. "
                f"Provided value: {arm!r}."
            )
        _exact_keys(
            arm,
            label=f"study.json arms[{index}]",
            required={
                "arm_id",
                "description",
                "experiment_id",
                "mode",
                "configs",
            },
        )
        arm_id = _identifier(arm["arm_id"], label=f"study.json arms[{index}].arm_id")
        if arm_id in arm_ids:
            raise StudyWorkflowError(
                "Expected study.json arm IDs to be unique. "
                f"Provided value: {arm_id!r}."
            )
        arm_ids.add(arm_id)
        _text(arm["description"], label=f"study.json arms[{index}].description")
        _identifier(
            arm["experiment_id"],
            label=f"study.json arms[{index}].experiment_id",
        )
        if arm["mode"] not in _MODES:
            raise StudyWorkflowError(
                "Expected study.json arm mode to be train, linspace, validate, "
                f"or characterize. Provided value: {arm['mode']!r}."
            )
        if not isinstance(arm["configs"], list) or not arm["configs"]:
            raise StudyWorkflowError(
                f"Expected study.json arms[{index}].configs to be non-empty. "
                f"Provided value: {arm['configs']!r}."
            )
        hashes: set[str] = set()
        for config_index, config in enumerate(arm["configs"]):
            if not isinstance(config, dict):
                raise StudyWorkflowError(
                    "Expected materialized study configs to be objects. "
                    f"Provided value: {config!r}."
                )
            _exact_keys(
                config,
                label=f"study.json arms[{index}].configs[{config_index}]",
                required={"declared_path", "resolved_path", "sha256"},
            )
            _text(config["declared_path"], label="study config declared_path")
            _text(config["resolved_path"], label="study config resolved_path")
            digest = _digest(config["sha256"], label="study config sha256")
            if digest in hashes:
                raise StudyWorkflowError(
                    "Expected materialized arm config hashes to be unique. "
                    f"Provided value: {digest!r}."
                )
            hashes.add(digest)
    return value


def load_study_record(study_dir: Path | str) -> dict[str, Any]:
    """Load a prepared study from its canonical ``<study-id>`` directory."""

    return _load_study_record(study_dir, require_directory_name=True)


def _command_mode(command: Sequence[Any]) -> str | None:
    return next((value for value in command if value in _MODES), None)


def _command_option(command: Sequence[Any], option: str) -> str | None:
    values = list(command)
    for index, value in enumerate(values):
        if value == option:
            if index + 1 >= len(values) or not isinstance(values[index + 1], str):
                return None
            return values[index + 1]
        prefix = f"{option}="
        if isinstance(value, str) and value.startswith(prefix):
            configured = value[len(prefix) :]
            return configured or None
    return None


def study_context_for_run(
    output_root: Path | str,
    *,
    experiment_id: str,
    command: Sequence[str],
) -> dict[str, Any] | None:
    """Return fail-closed provenance for a canonical study arm output root."""

    output = Path(output_root).expanduser().resolve()
    if output.parent.name != "runs":
        return None
    study_root = output.parent.parent
    if not (study_root / "study.json").is_file():
        return None
    study = load_study_record(study_root)
    arms = {arm.get("arm_id"): arm for arm in study["arms"] if isinstance(arm, dict)}
    arm = arms.get(output.name)
    if arm is None:
        raise StudyWorkflowError(
            "Expected the study output arm to be declared in study.json. "
            f"Provided value: {output.name!r}."
        )
    if arm.get("experiment_id") != experiment_id:
        raise StudyWorkflowError(
            "Expected the run experiment_id to match the declared study arm. "
            f"Provided value: run={experiment_id!r}, "
            f"declared={arm.get('experiment_id')!r}."
        )
    mode = _command_mode(command)
    if mode != arm.get("mode"):
        raise StudyWorkflowError(
            "Expected the run mode to match the declared study arm. "
            f"Provided value: run={mode!r}, declared={arm.get('mode')!r}."
        )
    configured = _command_option(command, "--config")
    if configured is None:
        raise StudyWorkflowError(
            "Expected a workflow-managed run command to contain --config. "
            f"Provided value: {list(command)!r}."
        )
    config_path = Path(configured).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise StudyWorkflowError(
            "Expected the workflow-managed --config path to exist. "
            f"Provided value: {str(config_path)!r}."
        )
    # This is deliberately the byte hash of the declared source file.  Native
    # run manifests separately retain the canonical hash of the resolved
    # config, so both the request and the numerical settings remain auditable.
    config_sha = _sha256_file(config_path)
    declared_hashes = {
        item.get("sha256")
        for item in arm.get("configs", [])
        if isinstance(item, dict)
    }
    if config_sha not in declared_hashes:
        raise StudyWorkflowError(
            "Expected the run config content to be predeclared by the study "
            f"arm. Provided value: SHA-256 {config_sha!r}."
        )
    return {
        "study_id": study["study_id"],
        "arm_id": arm["arm_id"],
        "evidence_class": study["evidence_class"],
        "study_sha256": _sha256_file(study_root / "study.json"),
        "source_plan_sha256": study["source_plan"]["sha256"],
        "source_config_sha256": config_sha,
    }


def _read_bundle_object(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    try:
        value = _load_json_object(path, label=label)
    except StudyWorkflowError as error:
        errors.append(str(error))
        return None
    return value


def _bundle_record(
    run_dir: Path,
    *,
    study: Mapping[str, Any] | None = None,
    arm: Mapping[str, Any] | None = None,
    study_sha256: str | None = None,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest = _read_bundle_object(run_dir / "manifest.json", errors, "run manifest.json")
    status = _read_bundle_object(run_dir / "status.json", errors, "run status.json")
    run_id = run_dir.name
    state = "invalid"
    experiment_id = None
    config_sha = None
    result_sha = None
    artifact_count = 0
    terminal_metrics = None

    if manifest is not None:
        if (
            manifest.get("schema") != RUN_SCHEMA
            or isinstance(manifest.get("schema_version"), bool)
            or manifest.get("schema_version") != 1
        ):
            errors.append("Expected manifest.json to use ebl.run schema version 1.")
        if manifest.get("run_id") != run_id:
            errors.append("Expected manifest run_id to equal the run directory name.")
        experiment_id = manifest.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            errors.append("Expected manifest experiment_id to be nonempty text.")
        elif arm is not None and experiment_id != arm["experiment_id"]:
            errors.append("Expected manifest experiment_id to match the declared arm.")
        command = manifest.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
            errors.append("Expected manifest command to be a nonempty list of strings.")
        elif arm is not None and _command_mode(command) != arm["mode"]:
            errors.append("Expected manifest command mode to match the declared arm.")
        if study is not None and arm is not None:
            context = manifest.get("study")
            expected_context = {
                "study_id": study["study_id"],
                "arm_id": arm["arm_id"],
                "evidence_class": study["evidence_class"],
                "study_sha256": study_sha256,
                "source_plan_sha256": study["source_plan"]["sha256"],
            }
            if not isinstance(context, dict):
                errors.append("Expected manifest.json to contain workflow study provenance.")
            else:
                for key, expected in expected_context.items():
                    if context.get(key) != expected:
                        errors.append(f"Expected manifest study.{key} to match study.json.")
                config_sha = context.get("source_config_sha256")
                if not isinstance(config_sha, str) or _SHA256.fullmatch(config_sha) is None:
                    errors.append(
                        "Expected manifest study.source_config_sha256 to be a "
                        "lowercase SHA-256 digest."
                    )
                elif config_sha not in {
                    config["sha256"] for config in arm["configs"]
                }:
                    errors.append(
                        "Expected manifest study.source_config_sha256 to match a "
                        "config declared by the arm."
                    )
        config_record = manifest.get("config")
        if isinstance(config_record, dict):
            configured_path = config_record.get("path")
            if configured_path != "config.resolved.json":
                errors.append(
                    "Expected manifest config.path to be 'config.resolved.json'."
                )
            resolved_config = run_dir / "config.resolved.json"
            config_digest = config_record.get("sha256")
            if (
                not isinstance(config_digest, str)
                or _SHA256.fullmatch(config_digest) is None
            ):
                errors.append(
                    "Expected manifest config.sha256 to be a lowercase SHA-256 digest."
                )
            if not resolved_config.is_file():
                errors.append("Expected the resolved config named by manifest.json to exist.")
            else:
                resolved_value = _read_bundle_object(
                    resolved_config,
                    errors,
                    "resolved config",
                )
                if (
                    resolved_value is not None
                    and sha256(_canonical_json_bytes(resolved_value)).hexdigest()
                    != config_record.get("sha256")
                ):
                    errors.append(
                        "Expected resolved-config content hash to match manifest.json."
                    )
        else:
            errors.append("Expected manifest.json config to name a hashed resolved config.")

    if status is not None:
        if (
            status.get("schema") != RUN_SCHEMA
            or isinstance(status.get("schema_version"), bool)
            or status.get("schema_version") != 1
        ):
            errors.append("Expected status.json to use ebl.run schema version 1.")
        state = status.get("status", "invalid")
        if not isinstance(state, str) or state not in {"running", "complete", "failed"}:
            errors.append("Expected status.json status to be running, complete, or failed.")
        if status.get("run_id") != run_id:
            errors.append("Expected status run_id to equal the run directory name.")

    result_path = run_dir / "result.json"
    if state == "complete":
        result = _read_bundle_object(result_path, errors, "run result.json")
        if result is not None:
            result_sha = _sha256_file(result_path)
            if (
                result.get("schema") != RUN_SCHEMA
                or isinstance(result.get("schema_version"), bool)
                or result.get("schema_version") != 1
            ):
                errors.append("Expected result.json to use ebl.run schema version 1.")
            if result.get("status") != "complete" or result.get("run_id") != run_id:
                errors.append("Expected result.json to describe this completed run.")
            if result.get("experiment_id") != experiment_id:
                errors.append(
                    "Expected result.json experiment_id to match manifest.json."
                )
            metrics = result.get("metrics")
            if not isinstance(metrics, dict):
                errors.append("Expected result.json metrics to be an object.")
            else:
                terminal_metrics = metrics
            artifacts = result.get("artifacts")
            if not isinstance(artifacts, list):
                errors.append("Expected result.json artifacts to be a list.")
            else:
                artifact_count = len(artifacts)
                paths: set[str] = set()
                for index, artifact in enumerate(artifacts):
                    if not isinstance(artifact, dict):
                        errors.append(f"Expected artifact {index} to be an object.")
                        continue
                    relative = artifact.get("path")
                    kind = artifact.get("kind")
                    if (
                        not isinstance(relative, str)
                        or not relative
                        or "\\" in relative
                    ):
                        errors.append(
                            f"Expected artifact {index} path to be a non-empty "
                            "relative POSIX path."
                        )
                        continue
                    relative_path = PurePosixPath(relative)
                    if (
                        relative_path.is_absolute()
                        or relative_path.as_posix() != relative
                        or any(part in {"", ".", ".."} for part in relative_path.parts)
                    ):
                        errors.append(
                            f"Expected artifact {index} path to be normalized "
                            "and confined to the run directory."
                        )
                        continue
                    if not isinstance(kind, str) or not kind:
                        errors.append(f"Expected artifact {index} kind to be non-empty.")
                    if relative in paths:
                        errors.append("Expected artifact paths to be unique.")
                    paths.add(relative)
                    size = artifact.get("size_bytes")
                    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                        errors.append(
                            f"Expected artifact {index} size_bytes to be a non-negative integer."
                        )
                    artifact_sha = artifact.get("sha256")
                    if (
                        not isinstance(artifact_sha, str)
                        or _SHA256.fullmatch(artifact_sha) is None
                    ):
                        errors.append(
                            f"Expected artifact {index} sha256 to be a lowercase digest."
                        )
                    candidate = run_dir.joinpath(*relative_path.parts).resolve()
                    try:
                        candidate.relative_to(run_dir.resolve())
                    except ValueError:
                        errors.append(
                            f"Expected artifact {index} to remain inside the run directory."
                        )
                        continue
                    if not candidate.is_file():
                        errors.append(f"Expected artifact {index} file to exist.")
                        continue
                    if candidate.stat().st_size != size:
                        errors.append(f"Expected artifact {index} size to match result.json.")
                    if verify_artifacts and _sha256_file(candidate) != artifact_sha:
                        errors.append(f"Expected artifact {index} SHA-256 to match result.json.")
        if not (run_dir / "metrics.jsonl").is_file():
            errors.append("Expected a completed run to contain metrics.jsonl.")
    elif result_path.exists():
        errors.append("Expected non-complete run states not to contain result.json.")

    return {
        "arm_id": arm["arm_id"] if arm is not None else None,
        "run_id": run_id,
        "path": run_dir.relative_to(run_dir.parents[2]).as_posix() if study is not None else str(run_dir.resolve()),
        "status": state,
        "valid": not errors,
        "errors": errors,
        "experiment_id": experiment_id,
        "source_config_sha256": config_sha,
        "result_sha256": result_sha,
        "artifact_count": artifact_count,
        "metrics": terminal_metrics,
    }


def inspect_run(
    run_dir: Path | str, *, verify_artifacts: bool = False,
) -> dict[str, Any]:
    """Inspect one native bundle without a study plan or documentation writes.

    Validity is structural integrity, not completion, assigned case coverage or a
    scientific verdict. Existing prepared studies retain their provenance checks.
    """
    root = Path(run_dir).expanduser().resolve()
    kwargs: dict[str, Any] = {}
    if len(root.parents) >= 3 and root.parent.parent.name == "runs":
        study_root = root.parents[2]
        if (study_root / "study.json").is_file():
            study = load_study_record(study_root)
            arms = [arm for arm in study["arms"] if arm["arm_id"] == root.parent.name]
            if len(arms) != 1:
                raise StudyWorkflowError("Expected run to belong to a declared study arm.")
            kwargs = dict(study=study, arm=arms[0], study_sha256=_sha256_file(study_root / "study.json"))
    record = _bundle_record(root, verify_artifacts=verify_artifacts, **kwargs)
    record["path"] = str(root)
    record["process_complete"] = record["status"] == "complete"
    record["validation_mode"] = "full_artifact_hashes" if verify_artifacts else "metadata_only"
    return record


def _study_contract_sha256(study: Mapping[str, Any]) -> str:
    return sha256(_canonical_json_bytes(_materialized_contract(study))).hexdigest()


def _run_relative_path(study_root: Path, run_dir: Path, *, label: str) -> str:
    root = study_root.expanduser().resolve()
    run = run_dir.expanduser().resolve()
    try:
        relative = run.relative_to(root)
    except ValueError as error:
        raise StudyWorkflowError(
            f"Expected {label} to remain inside its study directory. "
            f"Provided value: {str(run)!r}."
        ) from error
    if len(relative.parts) != 3 or relative.parts[0] != "runs":
        raise StudyWorkflowError(
            f"Expected {label} to be runs/<arm-id>/<run-id>. "
            f"Provided value: {relative.as_posix()!r}."
        )
    return relative.as_posix()


def _control_sha256(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for filename in _FEDERATED_CONTROL_FILES:
        path = run_dir / filename
        if not path.is_file():
            raise StudyWorkflowError(
                "Expected every federated completed run to retain its control "
                f"file. Provided value: {str(path)!r}."
            )
        result[filename] = _sha256_file(path)
    return result


def register_federated_runs(
    canonical_study_dir: Path | str,
    source_study_dir: Path | str,
    *,
    run_pairs: Sequence[tuple[Path | str, Path | str]],
) -> Path:
    """Register immutable copied runs from an equivalent prepared study.

    ``prepare`` intentionally records checkout-local paths and a preparation
    timestamp, so independently prepared copies of one tracked plan do not
    have byte-identical ``study.json`` files.  A native run correctly binds to
    the exact source bytes it saw.  This receipt preserves those bytes and
    proves that the source and canonical materialized scientific contracts are
    identical without rewriting a completed run manifest.
    """

    canonical_root = Path(canonical_study_dir).expanduser().resolve()
    source_root = Path(source_study_dir).expanduser().resolve()
    if canonical_root == source_root:
        raise StudyWorkflowError(
            "Expected canonical and source studies to be distinct directories."
        )
    canonical = load_study_record(canonical_root)
    # A transferred source study may be unpacked beneath an arbitrary staging
    # basename (for example ``local-study``).  Its exact study.json bytes and
    # materialized contract remain authoritative; only the canonical target is
    # required to occupy a directory named after its study ID.
    source = _load_study_record(source_root, require_directory_name=False)
    if _materialized_contract(canonical) != _materialized_contract(source):
        raise StudyWorkflowError(
            "Expected federated source and canonical studies to have the same "
            "materialized scientific contract."
        )
    canonical_sha = _sha256_file(canonical_root / "study.json")
    source_path = source_root / "study.json"
    source_bytes = source_path.read_bytes()
    source_sha = sha256(source_bytes).hexdigest()
    contract_sha = _study_contract_sha256(canonical)
    archive_relative = PurePosixPath(
        "analysis",
        "federated_sources",
        source_sha,
        canonical["study_id"],
        "study.json",
    )
    archive_path = canonical_root.joinpath(*archive_relative.parts)
    if archive_path.exists():
        if not archive_path.is_file() or _sha256_file(archive_path) != source_sha:
            raise StudyWorkflowError(
                "Expected an existing federated source-study archive to retain "
                f"the exact source bytes. Provided value: {str(archive_path)!r}."
            )
    else:
        _atomic_write_bytes(archive_path, source_bytes)

    arms = {arm["arm_id"]: arm for arm in canonical["arms"]}
    registered: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    if not run_pairs:
        raise StudyWorkflowError("Expected at least one federated run pair.")
    for pair_index, pair in enumerate(run_pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise StudyWorkflowError(
                "Expected each federated run pair to contain source and canonical paths. "
                f"Provided value at index {pair_index}: {pair!r}."
            )
        source_run = Path(pair[0]).expanduser().resolve()
        canonical_run = Path(pair[1]).expanduser().resolve()
        source_relative = _run_relative_path(
            source_root, source_run, label="federated source run"
        )
        canonical_relative = _run_relative_path(
            canonical_root, canonical_run, label="federated canonical run"
        )
        if source_relative != canonical_relative:
            raise StudyWorkflowError(
                "Expected source and canonical federated runs to have the same "
                f"study-relative path. Provided value: source={source_relative!r}, "
                f"canonical={canonical_relative!r}."
            )
        if canonical_relative in seen_paths:
            raise StudyWorkflowError(
                f"Expected federated run paths to be unique: {canonical_relative!r}."
            )
        seen_paths.add(canonical_relative)
        _runs, arm_id, run_id = PurePosixPath(canonical_relative).parts
        arm = arms.get(arm_id)
        if arm is None:
            raise StudyWorkflowError(
                f"Expected federated run arm to be declared: {arm_id!r}."
            )
        source_record = _bundle_record(
            source_run,
            study=source,
            arm=arm,
            study_sha256=source_sha,
            verify_artifacts=True,
        )
        target_record = _bundle_record(
            canonical_run,
            study=source,
            arm=arm,
            study_sha256=source_sha,
            verify_artifacts=True,
        )
        for label, record in (("source", source_record), ("canonical", target_record)):
            if not record["valid"] or record["status"] != "complete":
                raise StudyWorkflowError(
                    f"Expected {label} federated run to be a valid completed bundle. "
                    f"Provided value: path={canonical_relative!r}, "
                    f"errors={record['errors']!r}."
                )
        source_controls = _control_sha256(source_run)
        canonical_controls = _control_sha256(canonical_run)
        if source_controls != canonical_controls:
            raise StudyWorkflowError(
                "Expected copied federated run control files to remain byte-identical. "
                f"Provided value: {canonical_relative!r}."
            )
        registered.append(
            {
                "arm_id": arm_id,
                "run_id": run_id,
                "path": canonical_relative,
                "control_sha256": canonical_controls,
            }
        )

    receipt = {
        "schema": FEDERATED_COLLECTION_SCHEMA,
        "schema_version": FEDERATED_COLLECTION_SCHEMA_VERSION,
        "study_id": canonical["study_id"],
        "canonical_study_sha256": canonical_sha,
        "source_study": {
            "archive_path": archive_relative.as_posix(),
            "sha256": source_sha,
            "source_plan_sha256": source["source_plan"]["sha256"],
            "materialized_contract_sha256": contract_sha,
        },
        "runs": sorted(registered, key=lambda item: item["path"]),
    }
    registration_sha = sha256(
        _canonical_json_bytes(
            {
                "source_study_sha256": source_sha,
                "runs": receipt["runs"],
            }
        )
    ).hexdigest()
    receipt_path = (
        canonical_root
        / "analysis"
        / "federated_collections"
        / f"{source_sha}-{registration_sha[:16]}.json"
    )
    if receipt_path.exists():
        existing = _load_json_object(
            receipt_path, label="existing federated collection receipt"
        )
        if existing != receipt:
            raise StudyWorkflowError(
                "Expected an existing federated collection receipt to remain "
                f"immutable. Provided value: {str(receipt_path)!r}."
            )
    else:
        _atomic_write_json(receipt_path, receipt)
    return receipt_path


def _federated_run_contexts(
    study_root: Path,
    *,
    study: Mapping[str, Any],
    canonical_study_sha256: str,
    verify_artifacts: bool,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]], list[str]]:
    directory = study_root / "analysis" / "federated_collections"
    if not directory.exists():
        return {}, [], []
    if not directory.is_dir():
        return {}, [], [
            "Expected analysis/federated_collections to be a directory."
        ]
    contexts: dict[str, dict[str, str]] = {}
    collections: list[dict[str, Any]] = []
    errors: list[str] = []
    expected_contract_sha = _study_contract_sha256(study)
    expected_plan_sha = study["source_plan"]["sha256"]
    for receipt_path in sorted(directory.iterdir()):
        receipt_relative = receipt_path.relative_to(study_root).as_posix()
        try:
            if not receipt_path.is_file() or receipt_path.suffix != ".json":
                raise StudyWorkflowError(
                    "Expected every federated collection entry to be a JSON file."
                )
            receipt = _load_json_object(
                receipt_path, label="federated collection receipt"
            )
            _exact_keys(
                receipt,
                label="federated collection receipt",
                required={
                    "schema",
                    "schema_version",
                    "study_id",
                    "canonical_study_sha256",
                    "source_study",
                    "runs",
                },
            )
            if (
                receipt["schema"] != FEDERATED_COLLECTION_SCHEMA
                or isinstance(receipt["schema_version"], bool)
                or receipt["schema_version"] != FEDERATED_COLLECTION_SCHEMA_VERSION
            ):
                raise StudyWorkflowError(
                    "Expected federated collection schema version 1."
                )
            if receipt["study_id"] != study["study_id"]:
                raise StudyWorkflowError(
                    "Expected federated collection study_id to match study.json."
                )
            if (
                _digest(
                    receipt["canonical_study_sha256"],
                    label="federated canonical study hash",
                )
                != canonical_study_sha256
            ):
                raise StudyWorkflowError(
                    "Expected federated canonical study hash to match study.json."
                )
            source_record = receipt["source_study"]
            if not isinstance(source_record, dict):
                raise StudyWorkflowError(
                    "Expected federated source_study to be an object."
                )
            _exact_keys(
                source_record,
                label="federated source_study",
                required={
                    "archive_path",
                    "sha256",
                    "source_plan_sha256",
                    "materialized_contract_sha256",
                },
            )
            source_sha = _digest(
                source_record["sha256"], label="federated source study hash"
            )
            archive_relative = PurePosixPath(
                "analysis",
                "federated_sources",
                source_sha,
                study["study_id"],
                "study.json",
            ).as_posix()
            if source_record["archive_path"] != archive_relative:
                raise StudyWorkflowError(
                    "Expected federated source archive path to be hash-addressed."
                )
            archive_path = study_root.joinpath(*PurePosixPath(archive_relative).parts)
            if not archive_path.is_file() or _sha256_file(archive_path) != source_sha:
                raise StudyWorkflowError(
                    "Expected federated source archive bytes to match their hash."
                )
            source_study = load_study_record(archive_path.parent)
            if _materialized_contract(source_study) != _materialized_contract(study):
                raise StudyWorkflowError(
                    "Expected archived source and canonical materialized contracts to match."
                )
            if (
                _digest(
                    source_record["source_plan_sha256"],
                    label="federated source-plan hash",
                )
                != expected_plan_sha
                or source_study["source_plan"]["sha256"] != expected_plan_sha
            ):
                raise StudyWorkflowError(
                    "Expected federated source-plan hashes to match study.json."
                )
            if (
                _digest(
                    source_record["materialized_contract_sha256"],
                    label="federated materialized-contract hash",
                )
                != expected_contract_sha
            ):
                raise StudyWorkflowError(
                    "Expected federated materialized-contract hash to match study.json."
                )
            run_records = receipt["runs"]
            if not isinstance(run_records, list) or not run_records:
                raise StudyWorkflowError(
                    "Expected federated collection runs to be a non-empty list."
                )
            receipt_contexts: dict[str, dict[str, str]] = {}
            for index, run_record in enumerate(run_records):
                if not isinstance(run_record, dict):
                    raise StudyWorkflowError(
                        f"Expected federated runs[{index}] to be an object."
                    )
                _exact_keys(
                    run_record,
                    label=f"federated runs[{index}]",
                    required={
                        "arm_id",
                        "run_id",
                        "path",
                        "control_sha256",
                    },
                )
                arm_id = _identifier(
                    run_record["arm_id"], label=f"federated runs[{index}].arm_id"
                )
                run_id = _text(
                    run_record["run_id"], label=f"federated runs[{index}].run_id"
                )
                if run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
                    raise StudyWorkflowError(
                        f"Expected federated runs[{index}] run_id to be one safe "
                        "path component."
                    )
                expected_path = PurePosixPath("runs", arm_id, run_id).as_posix()
                if run_record["path"] != expected_path:
                    raise StudyWorkflowError(
                        f"Expected federated runs[{index}] path to match arm/run IDs."
                    )
                if arm_id not in {arm["arm_id"] for arm in study["arms"]}:
                    raise StudyWorkflowError(
                        f"Expected federated runs[{index}] arm to be declared."
                    )
                controls = run_record["control_sha256"]
                if not isinstance(controls, dict):
                    raise StudyWorkflowError(
                        f"Expected federated runs[{index}] control hashes to be an object."
                    )
                _exact_keys(
                    controls,
                    label=f"federated runs[{index}] control hashes",
                    required=set(_FEDERATED_CONTROL_FILES),
                )
                run_dir = study_root.joinpath(*PurePosixPath(expected_path).parts)
                try:
                    run_dir.resolve().relative_to(study_root.resolve())
                except ValueError as error:
                    raise StudyWorkflowError(
                        f"Expected federated runs[{index}] to remain inside study root."
                    ) from error
                for filename in _FEDERATED_CONTROL_FILES:
                    expected_sha = _digest(
                        controls[filename],
                        label=f"federated runs[{index}] {filename} hash",
                    )
                    control_path = run_dir / filename
                    if not control_path.is_file() or (
                        (filename != "metrics.jsonl" or verify_artifacts)
                        and _sha256_file(control_path) != expected_sha
                    ):
                        raise StudyWorkflowError(
                            f"Expected federated runs[{index}] {filename} bytes "
                            "to match the collection receipt."
                        )
                manifest = _load_json_object(
                    run_dir / "manifest.json",
                    label=f"federated runs[{index}] manifest",
                )
                context = manifest.get("study")
                if (
                    not isinstance(context, dict)
                    or context.get("study_sha256") != source_sha
                    or context.get("source_plan_sha256") != expected_plan_sha
                ):
                    raise StudyWorkflowError(
                        f"Expected federated runs[{index}] to bind the archived "
                        "source study and tracked plan."
                    )
                if expected_path in receipt_contexts or expected_path in contexts:
                    raise StudyWorkflowError(
                        f"Expected one federated receipt per run: {expected_path!r}."
                    )
                receipt_contexts[expected_path] = {
                    "study_sha256": source_sha,
                    "receipt_path": receipt_relative,
                }
            contexts.update(receipt_contexts)
            collections.append(
                {
                    "receipt_path": receipt_relative,
                    "receipt_sha256": _sha256_file(receipt_path),
                    "source_study_sha256": source_sha,
                    "source_study_archive_path": archive_relative,
                    "source_plan_sha256": expected_plan_sha,
                    "materialized_contract_sha256": expected_contract_sha,
                    "run_count": len(receipt_contexts),
                    "runs": sorted(receipt_contexts),
                }
            )
        except (OSError, StudyWorkflowError) as error:
            errors.append(f"{receipt_relative}: {error}")
    return contexts, collections, errors


def _summary_report(summary: Mapping[str, Any]) -> str:
    rows = []
    for arm in summary["arms"]:
        rows.append(
            "| `{arm_id}` | {expected} | {complete} | {running} | {failed} | "
            "{invalid} | {coverage} |".format(
                arm_id=arm["arm_id"],
                expected=arm["expected_runs"],
                complete=arm["complete"],
                running=arm["running"],
                failed=arm["failed"],
                invalid=arm["invalid"],
                coverage="yes" if arm["coverage_complete"] else "no",
            )
        )
    problems = [
        f"- `{run['path']}`: " + "; ".join(run["errors"])
        for run in summary["runs"]
        if run["errors"]
    ]
    problems.extend(
        f"- `runs/{arm_id}/`: expected a declared arm directory."
        for arm_id in summary["missing_arm_directories"]
    )
    problems.extend(
        f"- `runs/{arm_id}/`: arm directory is not declared by study.json."
        for arm_id in summary["unknown_arm_directories"]
    )
    problems.extend(
        f"- `analysis/final.json`: {error}"
        for error in summary["finalization_errors"]
    )
    problems.extend(
        f"- Federated collection: {error}"
        for error in summary.get("federated_collection_errors", [])
    )
    if not problems:
        problems = ["- None."]
    return (
        f"# {summary['title']} — study status\n\n"
        f"- **Study ID:** `{summary['study_id']}`\n"
        f"- **State:** `{summary['state']}`\n"
        f"- **Ready for review:** `{str(summary['ready_for_review']).lower()}`\n"
        f"- **Validation:** `{summary['validation_mode']}`\n\n"
        "## Initial hypothesis\n\n"
        f"{summary['hypothesis']}\n\n"
        "## Coverage\n\n"
        "| Arm | Expected | Complete | Running | Failed attempts | Invalid | Covered |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        + "\n".join(rows)
        + "\n\n## Validation problems\n\n"
        + "\n".join(problems)
        + "\n"
    )


def summarize_study(
    study_dir: Path | str,
    *,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    """Scan native control files once and write a compact analysis handoff."""

    root = Path(study_dir).expanduser().resolve()
    study = load_study_record(root)
    study_sha = _sha256_file(root / "study.json")
    federated_contexts, federated_collections, federated_errors = (
        _federated_run_contexts(
            root,
            study=study,
            canonical_study_sha256=study_sha,
            verify_artifacts=verify_artifacts,
        )
    )
    runs_root = root / "runs"
    if not runs_root.is_dir():
        raise StudyWorkflowError(
            "Expected a prepared study to contain runs/. "
            f"Provided value: {str(root)!r}."
        )
    declared = {arm["arm_id"]: arm for arm in study["arms"]}
    unknown_arm_dirs = sorted(
        path.name
        for path in runs_root.iterdir()
        if path.is_dir() and path.name not in declared
    )
    missing_arm_dirs = sorted(
        arm_id for arm_id in declared if not (runs_root / arm_id).is_dir()
    )
    records: list[dict[str, Any]] = []
    arms_summary: list[dict[str, Any]] = []
    for arm_id, arm in declared.items():
        arm_root = runs_root / arm_id
        arm_records: list[dict[str, Any]] = []
        for path in (
            sorted(arm_root.iterdir()) if arm_root.is_dir() else []
        ):
            if not path.is_dir():
                continue
            relative = path.relative_to(root).as_posix()
            federated = federated_contexts.get(relative)
            record = _bundle_record(
                path,
                study=study,
                arm=arm,
                study_sha256=(
                    federated["study_sha256"] if federated is not None else study_sha
                ),
                verify_artifacts=verify_artifacts,
            )
            record["study_sha256"] = (
                federated["study_sha256"] if federated is not None else study_sha
            )
            record["federated_collection_receipt"] = (
                federated["receipt_path"] if federated is not None else None
            )
            arm_records.append(record)
        records.extend(arm_records)
        counts = Counter(record["status"] for record in arm_records if record["valid"])
        invalid = sum(not record["valid"] for record in arm_records) + int(
            arm_id in missing_arm_dirs
        )
        expected_hashes = Counter(item["sha256"] for item in arm["configs"])
        complete_hashes = Counter(
            record["source_config_sha256"]
            for record in arm_records
            if record["valid"] and record["status"] == "complete"
        )
        missing = list((expected_hashes - complete_hashes).elements())
        unexpected = list((complete_hashes - expected_hashes).elements())
        coverage = not missing and not unexpected and counts["complete"] == len(arm["configs"])
        arms_summary.append(
            {
                "arm_id": arm_id,
                "description": arm["description"],
                "expected_runs": len(arm["configs"]),
                "complete": counts["complete"],
                "running": counts["running"],
                "failed": counts["failed"],
                "invalid": invalid,
                "missing_config_sha256": sorted(missing),
                "unexpected_config_sha256": sorted(unexpected),
                "coverage_complete": coverage,
            }
        )

    final_path = root / "analysis" / "final.json"
    finalization_errors: list[str] = []
    reviewed = False
    if final_path.is_file():
        final_record = _read_bundle_object(
            final_path,
            finalization_errors,
            "analysis/final.json",
        )
        if final_record is not None:
            if (
                final_record.get("schema") != FINAL_SCHEMA
                or isinstance(final_record.get("schema_version"), bool)
                or final_record.get("schema_version") != 1
            ):
                finalization_errors.append(
                    "Expected analysis/final.json to use ebl.study.final "
                    "schema version 1."
                )
            if final_record.get("study_id") != study["study_id"]:
                finalization_errors.append(
                    "Expected analysis/final.json study_id to match study.json."
                )
            if final_record.get("study_sha256") != study_sha:
                finalization_errors.append(
                    "Expected analysis/final.json study hash to match study.json."
                )
            reviewed = not finalization_errors

    total_running = sum(arm["running"] for arm in arms_summary)
    total_invalid = (
        sum(arm["invalid"] for arm in arms_summary)
        + len(unknown_arm_dirs)
        + int(bool(finalization_errors))
        + len(federated_errors)
    )
    ready = (
        all(arm["coverage_complete"] for arm in arms_summary)
        and total_running == 0
        and total_invalid == 0
    )
    discovered = len(records) + len(unknown_arm_dirs)
    if reviewed and ready:
        state = "reviewed"
    elif total_invalid:
        state = "invalid"
    elif total_running:
        state = "running"
    elif ready:
        state = "ready_for_review"
    elif discovered == 0:
        state = "planned"
    else:
        state = "incomplete"
    summary = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_id": study["study_id"],
        "title": study["title"],
        "hypothesis": study["hypothesis"],
        "evidence_class": study["evidence_class"],
        "study_sha256": study_sha,
        "validation_mode": "full_artifact_hashes" if verify_artifacts else "metadata_only",
        "state": state,
        "ready_for_review": ready,
        "finalization_errors": finalization_errors,
        "federated_collections": federated_collections,
        "federated_collection_errors": federated_errors,
        "missing_arm_directories": missing_arm_dirs,
        "unknown_arm_directories": unknown_arm_dirs,
        "arms": arms_summary,
        "runs": records,
    }
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    _atomic_write_json(analysis / "summary.json", summary)
    _atomic_write_text(analysis / "report.md", _summary_report(summary))
    return summary


def load_review(path: Path | str) -> dict[str, Any]:
    review_path = Path(path).expanduser().resolve()
    raw = _load_json_object(review_path, label="--review")
    _exact_keys(
        raw,
        label="study review",
        required={
            "schema_version",
            "outcome",
            "final_interpretation",
            "limitations",
            "next_steps",
        },
    )
    if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
        raise StudyWorkflowError(
            "Expected study review schema_version to be 1. "
            f"Provided value: {raw['schema_version']!r}."
        )
    if raw["outcome"] not in _OUTCOMES:
        raise StudyWorkflowError(
            f"Expected study review outcome to be one of {sorted(_OUTCOMES)!r}. "
            f"Provided value: {raw['outcome']!r}."
        )
    return {
        "schema_version": 1,
        "outcome": raw["outcome"],
        "final_interpretation": _text(
            raw["final_interpretation"], label="study review final_interpretation"
        ),
        "limitations": _text(raw["limitations"], label="study review limitations"),
        "next_steps": _text_list(raw["next_steps"], label="study review next_steps"),
        "source": {"path": str(review_path), "sha256": _sha256_file(review_path)},
    }


def finalize_study(
    study_dir: Path | str,
    *,
    review_path: Path | str,
    manifest_path: Path | str | None = None,
    verify_artifacts: bool = False,
) -> Path:
    """Write a local compatibility receipt for a complete legacy study.

    manifest_path is accepted but ignored for old callers. Historical documents
    are never read or written. New pilots/results are reviewed in their notes and
    do not need this complete-coverage legacy command, including partial reviews.
    """
    root = Path(study_dir).expanduser().resolve()
    study = load_study_record(root)
    summary = summarize_study(root, verify_artifacts=verify_artifacts)
    if not summary["ready_for_review"]:
        raise StudyWorkflowError(
            "Expected study coverage to be complete and valid before "
            f"finalization. Provided value: state={summary['state']!r}. "
            "Review partial evidence in the owning pilot/result note."
        )
    review = load_review(review_path)
    final_path = root / "analysis" / "final.json"
    study_sha = _sha256_file(root / "study.json")
    review_sha = review["source"]["sha256"]
    if final_path.exists():
        existing = _load_json_object(final_path, label="existing final.json")
        if existing.get("schema") != FINAL_SCHEMA or type(existing.get("schema_version")) is not int or existing["schema_version"] not in {1, 2}:
            raise StudyWorkflowError("Expected a supported ebl.study.final receipt.")
        if existing.get("study_sha256") != study_sha or existing.get("review_sha256") != review_sha:
            raise StudyWorkflowError("Expected a finalized study to retain the same study and review hashes.")
        return final_path
    final = {
        "schema": FINAL_SCHEMA,
        "schema_version": 1,
        "study_id": study["study_id"],
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "study_sha256": study_sha,
        "review_ready_summary_sha256": _sha256_file(root / "analysis" / "summary.json"),
        "review_sha256": review_sha,
        "outcome": review["outcome"],
        "final_interpretation": review["final_interpretation"],
        "limitations": review["limitations"],
        "next_steps": review["next_steps"],
        "validation_mode": summary["validation_mode"],
    }
    _atomic_write_json(final_path, final)
    return final_path


__all__ = [
    "StudyWorkflowError",
    "finalize_study",
    "inspect_run",
    "load_review",
    "load_study_plan",
    "load_study_record",
    "prepare_study",
    "register_federated_runs",
    "study_context_for_run",
    "summarize_study",
]
