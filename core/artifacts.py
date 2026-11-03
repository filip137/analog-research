"""core.artifacts — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


RUN_SCHEMA = "ebl.run"


RUN_SCHEMA_VERSION = 1


_SOURCE_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON representation used for stable hashes."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    """Hash a JSON-compatible value using the run protocol encoding."""

    return sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

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
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically write human-readable, strict JSON."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_write_bytes(path, payload + b"\n")


def _run_git(repo_root: Path, *args: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ("git", "-C", str(repo_root), *args),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_identity(repo_root: Path) -> dict[str, Any]:
    commit_bytes = _run_git(repo_root, "rev-parse", "HEAD")
    status_bytes = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    diff_bytes = _run_git(repo_root, "diff", "--binary", "HEAD")
    if commit_bytes is None or status_bytes is None or diff_bytes is None:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "dirty_hash": None,
        }

    status_records = [record for record in status_bytes.split(b"\0") if record]
    untracked_hashes: dict[str, str] = {}
    for raw_record in status_records:
        if not raw_record.startswith(b"?? "):
            continue
        relative = raw_record[3:].decode("utf-8", errors="surrogateescape")
        candidate = repo_root / relative
        if (
            candidate.is_file()
            and candidate.suffix.lower() in _SOURCE_SUFFIXES
        ):
            untracked_hashes[relative] = sha256_file(candidate)

    dirty_payload = {
        "status": status_bytes.decode("utf-8", errors="surrogateescape"),
        "diff_sha256": sha256(diff_bytes).hexdigest(),
        "untracked_source_sha256": untracked_hashes,
    }
    dirty = bool(status_records)
    return {
        "available": True,
        "commit": commit_bytes.decode("ascii").strip(),
        "dirty": dirty,
        "dirty_hash": content_hash(dirty_payload) if dirty else None,
    }


def runtime_identity() -> dict[str, Any]:
    """Collect stable runtime versions without importing optional packages."""

    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "torch": None,
        "aihwkit": None,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import aihwkit

        versions["aihwkit"] = getattr(aihwkit, "__version__", "unknown")
    except ImportError:
        pass
    return {
        **versions,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


@dataclass(frozen=True)
class ArtifactRecord:
    """A finalized artifact referenced relative to its run directory."""

    path: str
    sha256: str
    size_bytes: int
    kind: str
