"""Freeze the user-selected cosine comparison and prepare native per-source jobs.

This is configuration/transport preparation only. Training, deployment and
collection use the public lifecycle/campaign CLIs and the shared dispatcher.
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
from exp1e_development import duration_lifecycle, NOTE
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from slice_campaign import build_slice, write_frozen
from ebl.schema.lifecycle import parse_lifecycle, stage_selections
from networks.opt import PINNED

ID = "opt125m-cmo-l6-exp1e"
PHASE = "opt-cmo-exp1e-cosine-20261010"
EVIDENCE = ROOT / "results/lifecycles/opt125m-cmo-l6-exp1e-cosine"


def lifecycle(scope=None):
    doc = duration_lifecycle(5e-3).to_dict()
    doc["lifecycle_id"] = ID
    doc["question"].update(
        text="Matched analog-only and whole-transformer HWA on synthetic text: standard 3%, standard 9%, and corruption-aware 3%, with user-selected cosine LR 5e-3 to 1e-3 over 25,600 updates.",
        decision=f"Compare scopes separately at every defect rate on both held-out domains; see {NOTE}.",
    )
    doc["hwa"]["lr_schedule"] = {"kind": "cosine_warmup_decay", "total_steps": 25600,
                                 "warmup_steps": 205, "min_factor": 0.2}
    base = doc["hwa"]["arms"][0]
    arms = [{"id": "pretrained", "method": "none"}]
    for name, noise, corruption in (
        ("standard_hwa", 1.0, None),
        ("standard_hwa_9pct", 3.0, None),
        ("cdt_random_3pct", 1.0, {"kind": "random", "rates_ppm": [30000]}),
    ):
        for mode in ("analog", "model"):
            arm = copy.deepcopy(base)
            arm.update(id=name + ("_model" if mode == "model" else ""), scope=mode,
                       noise_strength=noise, corruption=corruption,
                       digital_learning_rate=0.005 if mode == "model" else None)
            arms.append(arm)
    doc["hwa"]["arms"] = arms
    doc["transfer_tests"] = [
        {"label": "wikitext", "corpus": {"name": "wikitext-2-raw-v1", "sha256":
         "33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01"}, "split": "test", "sequences": 1024},
        {"label": "generated", "corpus": {"name": "wikitext2-train-synthtest", "sha256":
         "8e73b6b43aafb92b8b52e11cdd060d573331fae410350cf2427f3eeba640e224"}, "split": "test", "sequences": 384},
    ]
    if scope is not None:
        if scope not in {"analog", "model"}:
            raise ValueError("Unknown readiness scope")
        doc["lifecycle_id"] += f"-cosine-{scope}-smoke"
        doc["question"]["text"] = "Numerical readiness and deployment cost probe; not scientific comparison evidence. " + doc["question"]["text"]
        doc["hwa"]["arms"] = [a for a in arms if a.get("scope") == scope and a["id"].startswith("cdt_")]
        doc["hwa"]["epochs"] = 2
        doc["hwa"]["lr_schedule"]["total_steps"] = 512
        doc["devices"]["assignment_seeds"] = [271001]
    return parse_lifecycle(doc)


def generate():
    records = []
    for scope in (None, "analog", "model"):
        lc = lifecycle(scope)
        path = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / f"{lc.lifecycle_id}.json"
        write_frozen(path, lc.to_dict())
        records.append({"path": str(path), "sha256": sha256(path), "lifecycle_sha256": lc.digest()})
    return records


def prepare_host(base):
    revision = command(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    if command(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Worker source must be committed and clean")
    python = str(Path(sys.executable).resolve())
    teacher = base / "inputs/opt-125m-27dcfa74/pytorch_model.bin"
    corpus_root = base / "inputs/corpora"
    inputs = [checked_input(teacher, PINNED["facebook/opt-125m"]["sha256"])]
    full = lifecycle().to_dict()
    for corpus in [full["data"]["corpus"]] + [t["corpus"] for t in full["transfer_tests"]]:
        inputs.append(checked_input(corpus_root / (corpus["name"] + ".pt"), corpus["sha256"]))
    env = dict(os.environ, EBL_CORPUS_ROOT=str(corpus_root), EBL_CAMPAIGN_STREAM="1", PYTHONUNBUFFERED="1")
    cases = []
    for scope in (None, "analog", "model"):
        lc = lifecycle(scope)
        config = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / f"{lc.lifecycle_id}.json"
        if parse_lifecycle(json.loads(config.read_text())).digest() != lc.digest():
            raise ValueError("Frozen lifecycle differs from the declared design")
        plan = ROOT / "results/lifecycles" / lc.lifecycle_id / "plan"
        command([python, "-m", "ebl", "lifecycle", "plan", str(config), "--teacher-weights", str(teacher),
                 "--output", str(plan), "--device", "cuda:0", "--cpu-threads", "1", "--python", python], cwd=ROOT, env=env)
        for arm in lc.hwa.arms:
            smoke = scope is not None
            case_id = "exp1e-cosine-" + arm.arm_id + ("-smoke" if smoke else "")
            case_root = ROOT / "results/lifecycles" / lc.lifecycle_id / case_id
            selected = [s.stage_id for s in stage_selections(lc) if s.kind == "deploy" and s.hwa_arm == arm.arm_id]
            manifest, receipt = build_slice(plan / "campaign.json", selected)
            manifest_path = case_root / "campaign.json"
            write_frozen(manifest_path, manifest)
            write_frozen(case_root / "slice.json", receipt)
            command([python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                     "--output-dir", str(case_root / "dry-run"), "--dry-run", "--fail-fast"],
                    cwd=ROOT, env=env, logs=case_root / "preparation/dry-run")
            budget = 1800 if smoke else 2700 if arm.method == "none" else 10800 if arm.scope == "model" else 8100
            expected = 900 if smoke else 1500 if arm.method == "none" else 7500 if arm.scope == "model" else 5700
            worker_root = base / "worker-receipts" / revision / case_id
            worker_path = case_root / "tmux-worker.json"
            native_root = case_root / "runs" / manifest["campaign_id"]
            worker = {"case_id": case_id, "state_root": str(worker_root), "cwd": str(ROOT),
                      "timeout_seconds": budget, "lease_seconds": 900,
                      "env": {"EBL_CORPUS_ROOT": str(corpus_root), "EBL_CAMPAIGN_STREAM": "1",
                              "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"},
                      "argv": [python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                               "--output-dir", str(case_root / "runs"), "--resume", "--fail-fast"]}
            write_frozen(worker_path, worker)
            cases.append({"id": case_id, "arm": arm.arm_id, "scope": arm.scope, "smoke": smoke,
                "expected_seconds": expected, "budget_seconds": budget, "lifecycle_sha256": lc.digest(),
                "placement": {"workdir": str(ROOT), "run_dir": str(worker_root),
                    "argv": [python, "-m", "ebl.execution.tmux_worker", "bridge", "--spec", str(worker_path)],
                    "validate_argv": [python, "-m", "ebl.execution.lifecycle_collect", "validate", str(native_root)],
                    "campaign_python": python, "campaign_worker_spec": str(worker_path),
                    "campaign_result_root": str(native_root), "gpu_models": ["5090"],
                    "prepared_files": [{"path": str(p), "sha256": sha256(p)} for p in
                        [config, manifest_path, worker_path] + [Path(s["config"]) for s in manifest["stages"]]]},
                "dry_run_passed": True})
    return {"schema": "exp1e.cosine_host.v1", "source_revision": revision, "source": str(ROOT),
            "base": str(base), "inputs": inputs, "cases": cases, "launched": False}


def queue(receipts, output):
    hosts = {p.stem: json.loads(p.read_text()) for p in receipts}
    if set(hosts) != {"fifi", "loulou"} or len({h["source_revision"] for h in hosts.values()}) != 1:
        raise ValueError("Exactly the two prepared 5090 hosts at the same revision are required")
    mapped = {h: {c["id"]: c for c in v["cases"]} for h, v in hosts.items()}
    if set(mapped["fifi"]) != set(mapped["loulou"]) or len(mapped["fifi"]) != 9:
        raise ValueError("Expected seven full source jobs and two readiness jobs")
    cases, jobs = [], []
    for case_id, c in mapped["fifi"].items():
        placements = {}
        for host in hosts:
            other = mapped[host][case_id]
            if other["lifecycle_sha256"] != c["lifecycle_sha256"] or not other["dry_run_passed"]:
                raise ValueError("Prepared host science differs or dry-run failed")
            if c["smoke"] and host != ("fifi" if c["scope"] == "model" else "loulou"):
                continue
            placements[host] = other["placement"]
        job = {"id": case_id, "project_dir": str(ROOT), "handoff": NOTE,
            "authorization": "User selected cosine LR 5e-3 to 1e-3 for the whole exp-005 comparison, authorized increased GPU-hour allowance, and specified two RTX5090s; use fifi and loulou only.",
            "peak_memory_mib": 21500 if c["scope"] == "model" else 17000, "headroom_mib": 2200,
            "expected_seconds": c["expected_seconds"], "budget_seconds": c["budget_seconds"],
            "priority": 100 if c["smoke"] else 50 if c["arm"] == "pretrained" else 30 if c["scope"] == "model" else 20,
            "startup_grace_seconds": 600, "stale_seconds": 900, "recovery": False, "retry_safe": False,
            "placements": placements, "completion": {"argv": [sys.executable, str(Path(__file__).resolve()), "collect",
                "--queue-config", str(Path.home() / ".config/run-watch/queue.json"), "--output", str(EVIDENCE / "collected")],
                "workdir": str(ROOT), "timeout_seconds": 600, "retry_safe": True}}
        def threshold(seconds):
            return {"startup_seconds": 600, "stale_seconds": 900, "expected_seconds": seconds}
        progress = threshold(c["expected_seconds"])
        progress["phases"] = {"prepare": threshold(600), "devices": threshold(120),
            "hwa": threshold(600 if c["smoke"] else 120 if c["arm"] == "pretrained" else 6500 if c["scope"] == "model" else 4500),
            "deploy": threshold(300), "validation": threshold(600), "collection": threshold(600)}
        cases.append({"id": case_id, "stage": "readiness" if c["smoke"] else "deployment" if c["arm"] == "pretrained" else "training",
            "dependencies": [], "required_gates": [] if c["smoke"] else ["handoff"],
            "expected_gpu_seconds": c["expected_seconds"], "expected_storage_bytes": 2000000000 if c["smoke"] else 5000000000,
            "progress": progress})
        path = output / "jobs" / f"{case_id}.json"
        write_frozen(path, job)
        jobs.append(str(path))
    manifest = {"version": 1, "id": PHASE, "poll_seconds": 300, "required_adapters": ["tmux"],
        "budgets": {"gpu_hours": 18, "storage_bytes": 40000000000, "window_seconds": 43200},
        "storage_roots": [{"target": host, "path": info["source"]} for host, info in hosts.items()] +
            [{"target": host, "path": str(Path(info["base"]) / "worker-receipts" / info["source_revision"])} for host, info in hosts.items()],
        "cases": cases}
    from ebl.execution.service import validate_manifest
    validate_manifest(manifest)
    write_frozen(output / "campaign.json", manifest)
    return {"manifest": str(output / "campaign.json"), "jobs": jobs, "submitted": False}


def collect(queue_config, output):
    from ebl.execution.lifecycle_collect import collect as native_collect
    result = native_collect(queue_config, output)
    rows = json.loads((output / "deployments.json").read_text())
    if len(rows) == 210:
        analysis = ROOT / "campaigns/opt-cmo-corruption/series/001-hwa-robustness/analysis/exp1e_decision.py"
        text = command([sys.executable, str(analysis), str(ROOT / "results/lifecycles"),
                        "--contract", "cosine-5090-v1", "--collected", str(output / "deployments.json")], cwd=ROOT)
        review = output / "review.txt"
        review.write_text(text)
        result["paired_comparison"] = str(review)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate")
    host = commands.add_parser("prepare-host")
    host.add_argument("--base", type=Path, default=DEFAULT_BASE)
    q = commands.add_parser("queue")
    q.add_argument("--output", type=Path, required=True)
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
        result = queue(args.receipts, args.output)
    else:
        result = collect(args.queue_config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
