"""Best-effort rendering of currently running repository experiments.

The generated Markdown is an operational view, not a launch gate or a
scientific result registry.  Native run ``status.json`` files remain the
source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


CURRENT_SIMULATIONS_PATH = Path("docs/current_simulations.md")
RESULTS_PATH = Path("results")
RUN_SCHEMA = "ebl.run"
_RUN_MODES = {"train", "linspace", "validate"}
ACTIVE_BEGIN = "<!-- BEGIN AUTOMATIC ACTIVE SIMULATIONS -->"
ACTIVE_END = "<!-- END AUTOMATIC ACTIVE SIMULATIONS -->"


@dataclass(frozen=True)
class ActiveRun:
    """One running native experiment discovered below ``results/``."""

    study_id: str
    relative_run_dir: str
    arm: str | None
    experiment_id: str
    mode: str
    started_at: str


def _read_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_mode(command: object) -> str:
    if isinstance(command, list):
        for value in command:
            if isinstance(value, str) and value in _RUN_MODES:
                return value
    return "run"


def _arm_from_relative(relative_run_dir: Path) -> str | None:
    parts = relative_run_dir.parts
    if len(parts) >= 4 and parts[1] == "runs":
        return parts[2]
    return None


def discover_active_runs(repo_root: Path | str) -> tuple[ActiveRun, ...]:
    """Return running native runs grouped later by their first path component."""

    repository = Path(repo_root).expanduser().resolve()
    results_root = repository / RESULTS_PATH
    if not results_root.is_dir():
        return ()

    active: list[ActiveRun] = []
    for status_path in sorted(results_root.rglob("status.json")):
        status = _read_json_mapping(status_path)
        if (
            status is None
            or status.get("schema") != RUN_SCHEMA
            or status.get("status") != "running"
        ):
            continue

        run_dir = status_path.parent
        try:
            relative = run_dir.resolve().relative_to(results_root.resolve())
        except (OSError, ValueError):
            continue
        if len(relative.parts) < 2:
            continue

        manifest = _read_json_mapping(run_dir / "manifest.json")
        if manifest is None or manifest.get("schema") != RUN_SCHEMA:
            continue
        if manifest.get("run_id") != status.get("run_id"):
            continue

        experiment_id = manifest.get("experiment_id")
        started_at = status.get("started_at")
        active.append(
            ActiveRun(
                study_id=relative.parts[0],
                relative_run_dir=relative.as_posix(),
                arm=_arm_from_relative(relative),
                experiment_id=(
                    experiment_id
                    if isinstance(experiment_id, str)
                    else "unknown"
                ),
                mode=_run_mode(manifest.get("command")),
                started_at=(
                    started_at if isinstance(started_at, str) else "unknown"
                ),
            )
        )

    return tuple(
        sorted(
            active,
            key=lambda item: (
                item.study_id,
                item.relative_run_dir,
            ),
        )
    )


def _inline_code(value: str) -> str:
    sanitized = value.replace("`", "'")
    return f"`{sanitized}`"


def render_active_section(active_runs: Sequence[ActiveRun]) -> str:
    """Render the generated active section without touching manual sections."""

    lines = [
        ACTIVE_BEGIN,
        "## Active",
        "",
    ]

    if not active_runs:
        lines.extend(
            [
                "No simulations currently running.",
                "",
            ]
        )
    else:
        grouped: dict[str, list[ActiveRun]] = {}
        for run in active_runs:
            grouped.setdefault(run.study_id, []).append(run)
        for study_id, runs in grouped.items():
            lines.extend(
                [
                    f"### {_inline_code(study_id)}",
                    "",
                    "- **Status:** `running`",
                    f"- **Active native runs:** `{len(runs)}`",
                    f"- **Raw home:** {_inline_code(f'results/{study_id}/')}",
                    "- **Runs:**",
                ]
            )
            for run in runs:
                details = [
                    _inline_code(run.mode),
                    _inline_code(run.experiment_id),
                ]
                if run.arm is not None:
                    details.append(f"arm {_inline_code(run.arm)}")
                details.append(f"started {_inline_code(run.started_at)}")
                lines.append(
                    f"  - {_inline_code(run.relative_run_dir)} — "
                    + ", ".join(details)
                )
            lines.append("")

    lines.extend([ACTIVE_END, ""])
    return "\n".join(lines)


def _default_document(active_runs: Sequence[ActiveRun]) -> str:
    return "\n".join(
        [
            "# Current LoRA/HWA Simulations",
            "",
            "The `Active` section is generated automatically from native "
            "`status.json`",
            "files below `results/`. Other sections may be maintained by "
            "hand.",
            "",
            render_active_section(active_runs).rstrip(),
            "",
        ]
    )


def _replace_active_section(
    document: str,
    active_runs: Sequence[ActiveRun],
) -> str | None:
    begin = document.find(ACTIVE_BEGIN)
    end = document.find(ACTIVE_END)
    if begin < 0 or end < begin:
        return None
    end += len(ACTIVE_END)
    replacement = render_active_section(active_runs).rstrip()
    return document[:begin] + replacement + document[end:]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def refresh_current_simulations_for_run(
    *,
    repo_root: Path | str,
    run_dir: Path | str,
) -> bool:
    """Compatibility no-op: global simulation documents are historical snapshots.

    Use discover_active_runs/render_active_section for read-only legacy inspection.
    Monitoring and current handoffs belong to the owning pilot/experiment note.
    """
    return False


__all__ = [
    "ActiveRun",
    "CURRENT_SIMULATIONS_PATH",
    "RESULTS_PATH",
    "discover_active_runs",
    "refresh_current_simulations_for_run",
    "render_active_section",
]
