"""Resolve an archived CIFAR run by content hash and print a local train command.

This prepares a replay with the integrated HWA code. It never launches training
or modifies historical manifests. Frozen source receipts remain separate inputs
for auditing the exact historical implementation and runtime.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import shlex
import sys

from experiments.artifacts import content_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = Path("results/cifar-migration-20261004/artifact-inventory.jsonl.gz")
INPUT_ROLES = {
    "teacher_weights", "weights", "device_model", "device_state",
    "device_data", "resume", "selection_receipt",
}
SOURCE_ROLES = {"source_snapshot", "staged_source_snapshot"}


class ArtifactIndex:
    """Resolve only files inside the migrated root, with no old-path fallback."""

    def __init__(self, root=ROOT, inventory=None):
        self.root = Path(root).resolve()
        self.inventory = Path(inventory) if inventory else self.root / DEFAULT_INVENTORY
        self.paths = defaultdict(set)
        self.verified = {}
        opener = gzip.open if self.inventory.suffix == ".gz" else open
        with opener(self.inventory, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                self.paths[record["sha256"]].add(record["destination"])
        extra = self.inventory.parent / "additional-artifacts.json"
        if extra.exists():
            for record in json.loads(extra.read_text())["files"]:
                self.paths[record["sha256"]].add(record["destination"])

    def resolve(self, digest, original=""):
        candidates = sorted(self.paths.get(digest, ()), key=lambda p: (
            Path(p).name != Path(original).name, len(Path(p).parts), p))
        for relative in candidates:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"Artifact escapes migrated root: {relative}")
            if not path.is_file():
                continue
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if key not in self.verified:
                self.verified[key] = sha256_file(path)
            if self.verified[key] != digest:
                raise ValueError(f"Migrated artifact digest changed: {path}")
            return path
        raise FileNotFoundError(f"No retained local artifact for {original!r}, SHA-256 {digest}")


def recipe(run, output_dir, *, index, python=sys.executable, sampler=None):
    from experiments.definitions import resolve_experiment_config

    run, output_dir = Path(run).resolve(), Path(output_dir).resolve()
    if not run.is_relative_to(index.root):
        raise ValueError("Choose a run in the migrated worktree.")
    if output_dir.exists() or output_dir.is_relative_to(run):
        raise ValueError("Choose a new output directory outside the archived run.")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema") != "ebl.run" or not manifest["experiment_id"].startswith("cifar"):
        raise ValueError("Expected a native CIFAR run manifest.")
    config = (run / manifest["config"]["path"]).resolve()
    if not config.is_relative_to(run):
        raise ValueError("Resolved config escapes its run.")
    if content_hash(json.loads(config.read_text())) != manifest["config"]["sha256"]:
        raise ValueError("Historical resolved configuration changed.")
    definition, _ = resolve_experiment_config(config, "train")
    if definition.experiment_id != manifest["experiment_id"]:
        raise ValueError("Config and manifest experiment IDs disagree.")
    command = [str(python), "-m", "ebl", "train", "--config", str(config),
               "--output-dir", str(output_dir)]
    inputs, sources, seen = [], [], set()
    for artifact in manifest["inputs"]:
        role = artifact["role"]
        if role not in INPUT_ROLES | SOURCE_ROLES or role in seen:
            raise ValueError(f"Unsupported or duplicate input role: {role}")
        seen.add(role)
        path = index.resolve(artifact["sha256"], artifact["path"])
        record = {**artifact, "original_path": artifact["path"], "path": str(path)}
        if role in SOURCE_ROLES:
            sources.append(record)
        else:
            command += ["--" + role.replace("_", "-"), str(path)]
            inputs.append(record)
    environment = {"EBL_CIFAR_ROOT": str(index.root / "artifacts/cifar_inputs/data")}
    if sampler:
        environment["EBL_AIHWKIT_PYTHON"] = str(Path(sampler).resolve())
    return {
        "schema": "cifar.local_reproduction.v1", "run_id": manifest["run_id"],
        "cwd": str(index.root), "command": command, "environment": environment,
        "unset_environment": ["EBL_SOURCE_RECEIPT"], "inputs": inputs,
        "historical_source_receipts": sources, "historical_runtime": manifest["runtime"],
        "code_policy": "integrated_hwa; historical source/runtime preserved separately",
        "training_launched": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sampler", default=os.environ.get("EBL_AIHWKIT_PYTHON"))
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()
    result = recipe(args.run, args.output_dir, index=ArtifactIndex(args.root, args.inventory),
                    python=args.python, sampler=args.sampler)
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print("cd " + shlex.quote(result["cwd"]))
        print(shlex.join(["env", "-u", "EBL_SOURCE_RECEIPT",
                          *(f"{k}={v}" for k, v in result["environment"].items()),
                          *result["command"]]))


if __name__ == "__main__":
    main()
