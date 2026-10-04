"""Pack source (not data/checkpoints/results) with a verifiable file manifest."""

from __future__ import annotations
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    names = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ]
        )
        .decode()
        .split("\0")
    )
    extensions = {
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".md",
        ".txt",
        ".sh",
        ".c",
        ".h",
        ".cpp",
        ".cu",
        ".cuh",
    }
    names = sorted(
        n
        for n in set(names)
        if n
        and Path(n).parts[0]
        not in {"artifacts", "results", "papers", "data", "simulation_results"}
        and Path(n).suffix in extensions
        and (root / n).is_file()
    )
    files = {n: sha256((root / n).read_bytes()).hexdigest() for n in names}
    identity = sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    receipt = {
        "schema": "cifar_crossbar.source_snapshot.v1",
        "identity": identity,
        "files": files,
        "base_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"]
        )
        .decode()
        .strip(),
        "branch": subprocess.check_output(
            ["git", "-C", str(root), "branch", "--show-current"]
        )
        .decode()
        .strip(),
    }
    encoded = (json.dumps(receipt, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for name in names:
            archive.add(root / name, arcname=name, recursive=False)
        info = tarfile.TarInfo("source-receipt.json")
        info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "identity": identity,
                "files": len(names),
                "archive": str(args.output),
                "bytes": args.output.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
