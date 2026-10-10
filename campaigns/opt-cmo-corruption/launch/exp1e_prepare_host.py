#!/usr/bin/env python3
"""Prepare the frozen development grid and readiness smokes; launch nothing.

Run with the training environment's Python. Standard output is one JSON record
for controller-side queue preparation. Commands are public workflow planning and
campaign dry-run commands; no model fitting or GPU allocation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import exp1e_development as development
import slice_campaign
from workflow.lifecycle import parse_lifecycle
from workflow.networks.opt_mlp import PINNED

DEFAULT_BASE = Path("/home/filip/analog-research/opt-cmo-exp1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_input(path: Path, expected: str) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing required frozen input: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen input hash mismatch: {path}; expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def command(argv: list[str], *, cwd: Path, env: dict | None = None, logs: Path | None = None) -> str:
    completed = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    if logs is not None:
        logs.parent.mkdir(parents=True, exist_ok=True)
        logs.with_suffix(".stdout.log").write_text(completed.stdout)
        logs.with_suffix(".stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Preparation command failed ({completed.returncode}): {argv!r}\n{completed.stderr[-4000:]}")
    return completed.stdout


def prepare_host(base: Path = DEFAULT_BASE, *, source: Path = ROOT, python: Path | None = None) -> dict:
    base, source = base.expanduser().resolve(), source.resolve()
    python = (python or Path(sys.executable)).resolve()
    dirty = command(["git", "status", "--porcelain"], cwd=source)
    if dirty.strip():
        raise ValueError("Expected a clean committed source checkout before host preparation")
    revision = command(["git", "rev-parse", "HEAD"], cwd=source).strip()
    corpus_root = base / "inputs/corpora"
    teacher = base / "inputs/opt-125m-27dcfa74/pytorch_model.bin"
    expected = [development.lifecycle(rate, schedule) for rate in development.RATES for schedule in development.SCHEDULES]
    expected += [development.lifecycle(1e-4, schedule, smoke=True) for schedule in development.SCHEDULES]
    lifecycles = []
    for canonical in expected:
        path = source / "campaigns/opt-cmo-corruption/lifecycles" / (canonical.lifecycle_id + ".json")
        actual = parse_lifecycle(json.loads(path.read_text()))
        if actual.digest() != canonical.digest():
            raise ValueError(f"Lifecycle differs from the frozen development design: {path}")
        lifecycles.append((path, actual))
    # Validate all scientific inputs before writing plans or specs. Development
    # intentionally needs no final transfer/test corpus.
    inputs = {"teacher": checked_input(teacher, PINNED["facebook/opt-125m"]["sha256"]), "corpora": []}
    corpus_digests = {(lc.data.corpus, lc.data.corpus_sha256) for _, lc in lifecycles}
    for name, digest in sorted(corpus_digests):
        inputs["corpora"].append(checked_input(corpus_root / (name + ".pt"), digest))
    environment = dict(os.environ, EBL_CORPUS_ROOT=str(corpus_root), EBL_CAMPAIGN_STREAM="1", PYTHONUNBUFFERED="1")
    records = []
    for config, lifecycle in lifecycles:
        case_id = lifecycle.lifecycle_id
        smoke = case_id.endswith("-smoke")
        case_root = source / "results/lifecycles" / case_id
        plan_root, runs_root = case_root / "plan", case_root / "runs"
        manifest_path, spec_path = case_root / "campaign-training.json", case_root / "tmux-worker.json"
        prepare_logs = case_root / "preparation"
        plan_argv = [str(python), "-m", "workflow", "plan", str(config), "--teacher-weights", str(teacher),
                     "--output", str(plan_root), "--runs", str(runs_root), "--device", "cuda:0",
                     "--cpu-threads", "1", "--python", str(python)]
        command(plan_argv, cwd=source, env=environment, logs=prepare_logs / "plan")
        manifest, slice_receipt = slice_campaign.build_slice(plan_root / "campaign.json",
                                                           ["prepare", "devices", "hwa__standard_hwa_model"])
        slice_campaign.write_frozen(manifest_path, manifest)
        slice_campaign.write_frozen(manifest_path.with_suffix(".slice.json"), slice_receipt)
        dry_run_root = case_root / "dry-run"
        dry_argv = [str(python), "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                    "--output-dir", str(dry_run_root), "--dry-run", "--fail-fast"]
        command(dry_argv, cwd=source, env=environment, logs=prepare_logs / "dry-run")
        worker_root = base / "worker-receipts" / revision / case_id
        run_argv = [str(python), "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                    "--output-dir", str(runs_root), "--resume", "--fail-fast"]
        worker_spec = {"case_id": case_id, "state_root": str(worker_root), "cwd": str(source),
                       "argv": run_argv, "timeout_seconds": 1800 if smoke else 5400,
                       "lease_seconds": 900,
                       "env": {"EBL_CORPUS_ROOT": str(corpus_root), "EBL_CAMPAIGN_STREAM": "1",
                               "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}}
        slice_campaign.write_frozen(spec_path, worker_spec)
        native_root = runs_root / manifest["campaign_id"]
        records.append({"case_id": case_id, "smoke": smoke, "worktree": str(source), "python": str(python),
                        "config": str(config), "config_sha256": sha256(config),
                        "lifecycle_sha256": lifecycle.digest(), "case_root": str(case_root),
                        "plan_root": str(plan_root), "plan_manifest_sha256": sha256(plan_root / "campaign.json"),
                        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                        "campaign_id": manifest["campaign_id"], "campaign_result_root": str(native_root),
                        "native_run_root": str(runs_root), "queue_run_dir": str(worker_root),
                        "campaign_worker_spec": str(spec_path), "worker_spec_sha256": sha256(spec_path),
                        "timeout_seconds": worker_spec["timeout_seconds"], "lease_seconds": 900,
                        "argv": [str(python), "-m", "campaigns.tmux_worker", "bridge", "--spec", str(spec_path)],
                        "prepared_files": [{"path": str(path), "sha256": sha256(path)} for path in
                                           [config, manifest_path, spec_path] +
                                           [Path(stage["config"]) for stage in manifest["stages"]]],
                        "dry_run": {"passed": True, "output_root": str(dry_run_root),
                                    "stdout": str(prepare_logs / "dry-run.stdout.log")}})
    return {"schema": "exp1e.host_preparation.v1", "source": str(source), "source_revision": revision,
            "python": str(python), "base": str(base), "inputs": inputs, "cases": records,
            "launched": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    print(json.dumps(prepare_host(args.base), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
