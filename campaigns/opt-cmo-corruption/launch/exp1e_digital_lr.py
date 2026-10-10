"""Prepare and collect the fixed-analog, variable-digital LR development sweep.

All computation uses native lifecycle stages and the existing persistent queue.
This module only prepares exact inputs and reads collected development evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1e_development import NOTE
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from slice_campaign import build_slice, write_frozen
from workflow.lifecycle import parse_lifecycle
from workflow.networks.opt_mlp import PINNED

PHASE = "opt-cmo-exp1e-digital-lr-20261011"
PREFIX = "opt125m-cmo-l6-exp1e-digital"
RATES = {"0": 0.0, "1em5": 1e-5, "1em4": 1e-4, "1em3": 1e-3}
ARM = "standard_hwa_model"
EVIDENCE = ROOT / "results/lifecycles/opt125m-cmo-l6-exp1e-digital-lr"


def lifecycle(label, *, smoke=False):
    if label not in RATES and not (label == "analog" and smoke):
        raise ValueError("Unknown frozen digital-LR case")
    doc = json.loads((ROOT / "campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1e.json").read_text())
    arm = copy.deepcopy(next(a for a in doc["hwa"]["arms"] if a["id"] == ARM))
    if label == "analog":
        arm.update(id="standard_hwa", scope="analog", digital_learning_rate=None)
    else:
        arm["digital_learning_rate"] = RATES[label]
    doc["hwa"]["arms"] = [arm]
    doc["lifecycle_id"] = PREFIX + ("-analog" if label == "analog" else "-lr" + label) + ("-smoke" if smoke else "")
    doc["question"].update(
        text=("Readiness only. " if smoke else "") +
             "Standard 3% HWA on synthetic text: hold analog/calibration cosine LR at 5e-3 to 1e-3 and vary only the additional digital LR.",
        decision="Rank mean healthy/3%-fault development KL; within 1% prefer smaller digital LR. Compare with the existing analog and equal-rate controls; no test-driven tuning. See " + NOTE,
    )
    doc.pop("transfer_tests", None)
    doc["data"].update(evaluation="development", test_sequences=0)
    if smoke:
        doc["hwa"]["epochs"] = 2
        doc["hwa"]["lr_schedule"]["total_steps"] = 512
    return parse_lifecycle(doc)


def cases():
    return [(label, False) for label in RATES] + [("analog", True), ("0", True)]


def generate():
    output = []
    for label, smoke in cases():
        lc = lifecycle(label, smoke=smoke)
        path = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
        write_frozen(path, lc.to_dict())
        output.append({"path": str(path), "sha256": sha256(path), "lifecycle_sha256": lc.digest()})
    return output


def prepare_host(base):
    if command(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Expected committed clean worker source")
    revision = command(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    python = str(Path(sys.executable).resolve())
    teacher = base / "inputs/opt-125m-27dcfa74/pytorch_model.bin"
    corpus_root = base / "inputs/corpora"
    lc = lifecycle("0")
    inputs = [checked_input(teacher, PINNED["facebook/opt-125m"]["sha256"]),
              checked_input(corpus_root / (lc.data.corpus + ".pt"), lc.data.corpus_sha256)]
    environment = dict(os.environ, EBL_CORPUS_ROOT=str(corpus_root), EBL_CAMPAIGN_STREAM="1", PYTHONUNBUFFERED="1")
    prepared = []
    for label, smoke in cases():
        lc = lifecycle(label, smoke=smoke)
        config = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
        if parse_lifecycle(json.loads(config.read_text())).digest() != lc.digest():
            raise ValueError("Frozen lifecycle differs from the declared case")
        case_root = ROOT / "results/lifecycles" / lc.lifecycle_id
        plan = case_root / "plan"
        command([python, "-m", "workflow", "plan", str(config), "--teacher-weights", str(teacher),
                 "--output", str(plan), "--device", "cuda:0", "--cpu-threads", "1", "--python", python],
                cwd=ROOT, env=environment)
        manifest, receipt = build_slice(plan / "campaign.json", ["hwa__" + lc.hwa.arms[0].arm_id])
        manifest_path = case_root / "campaign-training.json"
        write_frozen(manifest_path, manifest)
        write_frozen(case_root / "slice.json", receipt)
        command([python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                 "--output-dir", str(case_root / "dry-run"), "--dry-run", "--fail-fast"],
                cwd=ROOT, env=environment, logs=case_root / "preparation/dry-run")
        worker_root = base / "worker-receipts" / revision / lc.lifecycle_id
        worker_path = case_root / "tmux-worker.json"
        budget, expected = (1800, 300) if smoke else (10800, 6500)
        native_root = case_root / "runs" / manifest["campaign_id"]
        worker = {"case_id": lc.lifecycle_id, "state_root": str(worker_root), "cwd": str(ROOT),
                  "timeout_seconds": budget, "lease_seconds": 900,
                  "env": {"EBL_CORPUS_ROOT": str(corpus_root), "EBL_CAMPAIGN_STREAM": "1",
                          "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"},
                  "argv": [python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                           "--output-dir", str(case_root / "runs"), "--resume", "--fail-fast"]}
        write_frozen(worker_path, worker)
        prepared.append({"id": lc.lifecycle_id, "label": label, "smoke": smoke,
            "expected_seconds": expected, "budget_seconds": budget, "lifecycle_sha256": lc.digest(),
            "dry_run_passed": True, "placement": {
                "workdir": str(ROOT), "run_dir": str(worker_root),
                "argv": [python, "-m", "campaigns.tmux_worker", "bridge", "--spec", str(worker_path)],
                "validate_argv": [python, "-m", "campaigns.lifecycle_collect", "validate", str(native_root), "--source-only"],
                "campaign_python": python, "campaign_worker_spec": str(worker_path),
                "campaign_result_root": str(native_root), "gpu_models": ["5090"],
                "prepared_files": [{"path": str(p), "sha256": sha256(p)} for p in
                    [config, manifest_path, worker_path] + [Path(s["config"]) for s in manifest["stages"]]]}})
    return {"source_revision": revision, "source": str(ROOT), "base": str(base), "inputs": inputs,
            "cases": prepared, "launched": False}


def queue(receipts, output, controller):
    hosts = {p.stem: json.loads(p.read_text()) for p in receipts}
    if set(hosts) != {"fifi", "loulou"} or len({h["source_revision"] for h in hosts.values()}) != 1:
        raise ValueError("Expected both authorized hosts at one frozen revision")
    expected = {lifecycle(label, smoke=smoke).lifecycle_id for label, smoke in cases()}
    mapped = {host: {c["id"]: c for c in receipt["cases"]} for host, receipt in hosts.items()}
    if any(set(entries) != expected for entries in mapped.values()):
        raise ValueError("Expected four candidates and two smokes")
    jobs, campaign_cases = [], []
    for case_id, c in mapped["fifi"].items():
        placements = {}
        for host in hosts:
            other = mapped[host][case_id]
            if other["lifecycle_sha256"] != c["lifecycle_sha256"] or not other["dry_run_passed"]:
                raise ValueError("Prepared scientific identities or readiness differ")
            if c["smoke"] and host != ("loulou" if c["label"] == "analog" else "fifi"):
                continue
            placements[host] = other["placement"]
        job = {"id": case_id, "project_dir": str(controller), "handoff": NOTE,
            "authorization": "User approved holding analog cosine LR fixed and varying only digital LR toward zero; four standard-3% development runs on the two previously authorized 5090s, plus bounded readiness checks.",
            "peak_memory_mib": 21500, "headroom_mib": 2200,
            "expected_seconds": c["expected_seconds"], "budget_seconds": c["budget_seconds"],
            "priority": 100 if c["smoke"] else {"0": 40, "1em4": 40, "1em5": 30, "1em3": 30}[c["label"]],
            "startup_grace_seconds": 600, "stale_seconds": 900, "recovery": False, "retry_safe": False,
            "placements": placements, "completion": {
                "argv": [sys.executable, str(controller / "campaigns/opt-cmo-corruption/launch/exp1e_digital_lr.py"),
                         "collect", "--queue-config", str(Path.home() / ".config/run-watch/queue.json"),
                         "--output", str(EVIDENCE / "collected")],
                "workdir": str(controller), "timeout_seconds": 600, "retry_safe": True}}
        def threshold(seconds):
            return {"startup_seconds": 600, "stale_seconds": 900, "expected_seconds": seconds}
        progress = threshold(c["expected_seconds"])
        progress["phases"] = {phase: threshold(seconds) for phase, seconds in
                              (("prepare", 120), ("devices", 120), ("hwa", 300 if c["smoke"] else 6500),
                               ("validation", 600), ("collection", 600))}
        campaign_cases.append({"id": case_id, "stage": "readiness" if c["smoke"] else "training",
            "dependencies": [], "required_gates": [] if c["smoke"] else ["handoff"],
            "expected_gpu_seconds": c["expected_seconds"], "expected_storage_bytes": 1500000000,
            "progress": progress})
        path = output / "jobs" / (case_id + ".json")
        write_frozen(path, job)
        jobs.append(str(path))
    manifest = {"version": 1, "id": PHASE, "poll_seconds": 300, "required_adapters": ["tmux"],
        "budgets": {"gpu_hours": 10, "storage_bytes": 16000000000, "window_seconds": 28800},
        "storage_roots": [{"target": host, "path": info["source"]} for host, info in hosts.items()] +
            [{"target": host, "path": str(Path(info["base"]) / "worker-receipts" / info["source_revision"])} for host, info in hosts.items()],
        "cases": campaign_cases}
    from campaigns.execution_service import validate_manifest
    validate_manifest(manifest)
    write_frozen(output / "campaign.json", manifest)
    return {"manifest": str(output / "campaign.json"), "jobs": jobs, "submitted": False}


def review(output):
    from workflow.runtime import validate_source_fit
    rows = []
    for item in json.loads((output / "index.json").read_text()):
        if item["job"].endswith("-smoke"):
            continue
        folder = Path(item["receipt"]).parent
        receipt = json.loads((folder / "collection.json").read_text())
        if any(sha256(folder / name) != digest for name, digest in receipt["files"].items()):
            raise ValueError("Collected compact evidence changed")
        result_path = folder / ("hwa__" + ARM) / "result.json"
        result = json.loads(result_path.read_text())["metrics"]
        lc = parse_lifecycle(json.loads(result_path.with_name("config.resolved.json").read_text())["lifecycle"])
        label = next(label for label in RATES if lifecycle(label).digest() == lc.digest())
        fit = result["fit"]
        validate_source_fit(lc, lc.hwa.arms[0], fit)
        selected = next(row for row in fit["history"] if row["epoch"] == fit["selected_epoch"])
        if RATES[label] == 0 and not fit["scope"].get("digital_unchanged"):
            raise ValueError("Zero-rate digital weights changed")
        rows.append({"job": item["job"], "digital_lr": RATES[label], "mean_kl": fit["development_kl"],
            "cases": selected["cases"], "selected_step": selected["global_step"],
            "clean_development_kl": result["clean_development"]["teacher_kl"],
            "convergence_review_required": fit["convergence_review_required"], "result": str(result_path)})
    rows.sort(key=lambda row: row["digital_lr"])
    complete = len(rows) == len(RATES) and {row["digital_lr"] for row in rows} == set(RATES.values())
    recommended = None
    if complete:
        best = min(row["mean_kl"] for row in rows)
        recommended = min((row for row in rows if row["mean_kl"] <= 1.01 * best), key=lambda row: row["digital_lr"])["digital_lr"]
    report = {"coverage_complete": complete, "cases": rows, "recommended_digital_lr_within_new_grid": recommended,
              "scientific_gate": "awaiting_scientific_review", "test_data_used": False}
    from experiments.artifacts import atomic_write_json
    atomic_write_json(output / "digital-lr-review.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate")
    host = commands.add_parser("prepare-host")
    host.add_argument("--base", type=Path, default=DEFAULT_BASE)
    q = commands.add_parser("queue")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--controller", type=Path, required=True)
    q.add_argument("receipts", nargs="+", type=Path)
    c = commands.add_parser("collect")
    c.add_argument("--queue-config", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate()
    elif args.command == "prepare-host":
        result = prepare_host(args.base)
    elif args.command == "queue":
        result = queue(args.receipts, args.output, args.controller)
    else:
        from campaigns.lifecycle_collect import collect
        result = collect(args.queue_config, args.output, source_only=True)
        result["review"] = review(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
