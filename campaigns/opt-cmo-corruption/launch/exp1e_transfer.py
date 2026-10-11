"""Deploy the two frozen digital-LR checkpoints; never execute HWA training."""
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
from slice_campaign import write_frozen
from workflow.lifecycle import parse_lifecycle, provenance, verify_provenance
from workflow.networks.opt_mlp import PINNED

PHASE = "opt-cmo-exp1e-digital-transfer-20261011"
PREFIX = "opt125m-cmo-l6-exp1e-digital-transfer"
EVIDENCE = ROOT / "results/lifecycles" / PREFIX
ARM = "standard_hwa_model"
HOSTS = {"0": "fifi", "1em5": "loulou"}
SOURCES = Path(__file__).with_name("exp1e_transfer_sources.json")


def lifecycle(label):
    if label not in HOSTS:
        raise ValueError("Only the two frozen development-selected endpoints are authorized")
    doc = json.loads((ROOT / f"campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1e-digital-lr{label}.json").read_text())
    original = parse_lifecycle(doc)
    doc["lifecycle_id"] = PREFIX + "-lr" + label
    doc["question"].update(text="Frozen-checkpoint deployment comparison: digital LR zero versus 1e-5, on five matched arrays and both held-out domains; no retraining.",
        decision="Report paired effects separately for every fault rate and domain; retain the existing 4/5 wins and 10% mean paired reduction material-effect rule. No test-driven source selection or downstream release. See " + NOTE)
    doc["transfer_tests"] = copy.deepcopy(json.loads((ROOT / "campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1e.json").read_text())["transfer_tests"])
    lc = parse_lifecycle(doc)
    verify_provenance(provenance(original, "hwa_source", hwa_arm=ARM), lc, "hwa_source", hwa_arm=ARM)
    return lc


def generate():
    records = []
    for label in HOSTS:
        lc = lifecycle(label)
        path = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
        write_frozen(path, lc.to_dict())
        records.append({"path": str(path), "lifecycle_sha256": lc.digest()})
    return records


def deployment_manifest(plan_path, source_path):
    """Replace the planned HWA dependency by one explicit immutable input."""
    raw = json.loads(plan_path.read_text())
    stages = []
    for stage in raw["stages"]:
        if stage["id"] == "hwa__" + ARM:
            continue
        stage["config"] = str((plan_path.parent / stage["config"]).resolve())
        if stage["id"].startswith("deploy__"):
            stage["inputs"]["weights"] = {"path": str(source_path)}
            stage["depends_on"] = [name for name in stage["depends_on"] if name != "hwa__" + ARM]
        if stage["id"] not in {"prepare", "devices"} and not stage["id"].startswith("deploy__"):
            raise ValueError("Only preparation and deployment are authorized")
        stages.append(stage)
    raw["stages"] = stages
    from campaigns.schema import CampaignSpec
    CampaignSpec.parse(raw, base_dir=plan_path.parent)
    return raw


def prepare_host(host, base):
    if command(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Expected a clean frozen checkout")
    label = next(label for label, target in HOSTS.items() if target == host)
    source = json.loads(SOURCES.read_text())[label]
    revision = command(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    python = str(Path(sys.executable).resolve())
    teacher = base / "inputs/opt-125m-27dcfa74/pytorch_model.bin"
    corpus_root = base / "inputs/corpora"
    lc = lifecycle(label)
    inputs = [checked_input(teacher, PINNED["facebook/opt-125m"]["sha256"]),
              checked_input(Path(source["path"]), source["sha256"]),
              checked_input(Path(source["result"]), source["result_sha256"])]
    for corpus in [lc.to_dict()["data"]["corpus"]] + [t["corpus"] for t in lc.to_dict()["transfer_tests"]]:
        inputs.append(checked_input(corpus_root / (corpus["name"] + ".pt"), corpus["sha256"]))
    result = json.loads(Path(source["result"]).read_text())
    source_lc = parse_lifecycle(json.loads(Path(source["result"]).with_name("config.resolved.json").read_text())["lifecycle"])
    verify_provenance(provenance(source_lc, "hwa_source", hwa_arm=ARM), lc, "hwa_source", hwa_arm=ARM)
    from workflow.runtime import validate_source_fit
    validate_source_fit(lc, lc.hwa.arm(ARM), result["metrics"]["fit"])
    config = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
    if parse_lifecycle(json.loads(config.read_text())).digest() != lc.digest():
        raise ValueError("Frozen deployment lifecycle changed")
    case_root = ROOT / "results/lifecycles" / lc.lifecycle_id
    plan = case_root / "plan"
    env = dict(os.environ, EBL_CORPUS_ROOT=str(corpus_root), EBL_CAMPAIGN_STREAM="1", PYTHONUNBUFFERED="1")
    command([python, "-m", "workflow", "plan", str(config), "--teacher-weights", str(teacher),
        "--output", str(plan), "--device", "cuda:0", "--cpu-threads", "1", "--python", python], cwd=ROOT, env=env)
    manifest = deployment_manifest(plan / "campaign.json", Path(source["path"]))
    manifest_path = case_root / "campaign-deployment.json"
    write_frozen(manifest_path, manifest)
    command([python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path), "--output-dir",
        str(case_root / "dry-run"), "--dry-run", "--fail-fast"], cwd=ROOT, env=env, logs=case_root / "preparation/dry-run")
    worker_root = base / "worker-receipts" / revision / lc.lifecycle_id
    native_root = case_root / "runs" / manifest["campaign_id"]
    worker_path = case_root / "tmux-worker.json"
    worker = {"case_id": lc.lifecycle_id, "state_root": str(worker_root), "cwd": str(ROOT),
        "timeout_seconds": 1200, "lease_seconds": 900,
        "env": {"EBL_CORPUS_ROOT": str(corpus_root), "EBL_CAMPAIGN_STREAM": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"},
        "argv": [python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                 "--output-dir", str(case_root / "runs"), "--resume", "--fail-fast"]}
    write_frozen(worker_path, worker)
    files = [config, manifest_path, worker_path, Path(source["path"]), Path(source["result"])] + [Path(s["config"]) for s in manifest["stages"]]
    return {"host": host, "label": label, "source_revision": revision, "source": str(ROOT), "base": str(base),
        "inputs": inputs, "id": lc.lifecycle_id, "lifecycle_sha256": lc.digest(), "source_checkpoint": source,
        "dry_run_passed": True, "stage_ids": [s["id"] for s in manifest["stages"]],
        "placement": {"workdir": str(ROOT), "run_dir": str(worker_root),
            "argv": [python, "-m", "campaigns.tmux_worker", "bridge", "--spec", str(worker_path)],
            "validate_argv": [python, "-m", "campaigns.lifecycle_collect", "validate", str(native_root), "--deployment-only"],
            "campaign_python": python, "campaign_worker_spec": str(worker_path), "campaign_result_root": str(native_root),
            "gpu_models": ["5090"], "prepared_files": [{"path": str(p), "sha256": sha256(p)} for p in files]}}


def queue(receipts, output, controller):
    hosts = {p.stem: json.loads(p.read_text()) for p in receipts}
    if set(hosts) != set(HOSTS.values()) or len({h["source_revision"] for h in hosts.values()}) != 1:
        raise ValueError("Expected both designated hosts at one frozen revision")
    cases, jobs = [], []
    for label, host in HOSTS.items():
        h = hosts[host]
        lc = lifecycle(label)
        if not h["dry_run_passed"] or h["lifecycle_sha256"] != lc.digest() or any(s.startswith("hwa__") for s in h["stage_ids"]):
            raise ValueError("Invalid frozen deployment preparation")
        job = {"id": h["id"], "project_dir": str(controller), "handoff": NOTE,
            "authorization": "User approved step 1: evaluate the two existing digital-LR checkpoints on five matched arrays, all six fault rates and both held-out domains; no training. Use fifi and loulou, one worker each.",
            "peak_memory_mib": 17000, "headroom_mib": 2000, "expected_seconds": 600, "budget_seconds": 1200,
            "priority": 50, "startup_grace_seconds": 600, "stale_seconds": 600, "recovery": False, "retry_safe": False,
            "placements": {host: h["placement"]}, "completion": {
                "argv": [sys.executable, str(controller / "campaigns/opt-cmo-corruption/launch/exp1e_transfer.py"),
                         "collect", "--queue-config", str(Path.home() / ".config/run-watch/queue.json"), "--output", str(EVIDENCE / "collected")],
                "workdir": str(controller), "timeout_seconds": 600, "retry_safe": True}}
        path = output / "jobs" / (h["id"] + ".json")
        write_frozen(path, job)
        jobs.append(str(path))
        def threshold(seconds):
            return {"expected_seconds": seconds, "stale_seconds": 600, "startup_seconds": 600}
        cases.append({"id": h["id"], "stage": "readiness" if label == "0" else "deployment", "dependencies": [],
            "required_gates": [] if label == "0" else ["baseline_deployment"], "expected_gpu_seconds": 600,
            "expected_storage_bytes": 4000000000, "progress": {**threshold(600), "phases": {
                phase: threshold(seconds) for phase, seconds in (("prepare", 120), ("devices", 120), ("deploy", 180), ("validation", 600), ("collection", 600))}}})
    manifest = {"version": 1, "id": PHASE, "poll_seconds": 300, "required_adapters": ["tmux"],
        "budgets": {"gpu_hours": 1, "storage_bytes": 10000000000, "window_seconds": 10800}, "cases": cases,
        "storage_roots": [{"target": host, "path": h["source"]} for host, h in hosts.items()] +
            [{"target": host, "path": str(Path(h["base"]) / "worker-receipts" / h["source_revision"])} for host, h in hosts.items()]}
    from campaigns.execution_service import validate_manifest
    validate_manifest(manifest)
    write_frozen(output / "campaign.json", manifest)
    return {"manifest": str(output / "campaign.json"), "jobs": jobs, "submitted": False}


def review(output):
    rows = json.loads((output / "deployments.json").read_text())
    sources = json.loads(SOURCES.read_text())
    identities = {}
    for entry in json.loads((output / "index.json").read_text()):
        label = next(label for label in HOSTS if entry["job"] == PREFIX + "-lr" + label)
        folder = Path(entry["receipt"]).parent
        receipt = json.loads(Path(entry["receipt"]).read_text())
        if any(sha256(folder / name) != digest for name, digest in receipt["files"].items()):
            raise ValueError("Collected frozen deployment evidence changed")
        result = json.loads((folder / "reused-source/result.json").read_text())
        artifact = next(a for a in result["artifacts"] if a["kind"] == "hwa_source")
        if artifact["sha256"] != sources[label]["sha256"]:
            raise ValueError("Deployment used a different selected checkpoint")
        if any(name.startswith("hwa__") for name in receipt["stages"]):
            raise ValueError("Unexpected training in a frozen deployment comparison")
        identities[label] = {}
        for name in receipt["stages"]:
            if name.startswith("deploy__"):
                metrics = json.loads((folder / name / "result.json").read_text())["metrics"]
                for case, values in metrics["cases"].items():
                    identities[label][metrics["array_seed"], case] = values["identity"]
    lc = lifecycle("0")
    grouped = {}
    for label in HOSTS:
        expected = {(seed, case.label) for seed in lc.devices.assignment_seeds for case in lc.defects.cases}
        subset = [r for r in rows if r["lifecycle_id"] == PREFIX + "-lr" + label]
        if len({(r["array_seed"], r["case"]) for r in subset}) != len(subset):
            raise ValueError("Duplicate deployment evidence")
        if subset and {(r["array_seed"], r["case"]) for r in subset} != expected:
            raise ValueError("Incomplete source deployment coverage")
        grouped[label] = {(r["array_seed"], r["case"]): r for r in subset}
    complete = all(len(v) == 30 for v in grouped.values()) and len(rows) == 60
    comparisons = []
    if complete:
        if identities["0"] != identities["1em5"]:
            raise ValueError("Comparison array identities or fault maps differ")
        for case in lc.defects.cases:
            for domain in ("wikitext", "generated"):
                key = "deployed_transfer_" + domain + "_teacher_kl"
                pairs = [{"array_seed": seed, "frozen_digital": grouped["0"][seed, case.label][key],
                          "small_digital_lr": grouped["1em5"][seed, case.label][key]} for seed in lc.devices.assignment_seeds]
                for pair in pairs:
                    pair["relative_reduction"] = 1 - pair["small_digital_lr"] / pair["frozen_digital"]
                reduction = sum(p["relative_reduction"] for p in pairs) / len(pairs)
                wins = sum(p["small_digital_lr"] < p["frozen_digital"] for p in pairs)
                comparisons.append({"case": case.label, "domain": domain, "pairs": pairs, "wins": wins,
                    "mean_frozen_digital_kl": sum(p["frozen_digital"] for p in pairs) / len(pairs),
                    "mean_small_digital_lr_kl": sum(p["small_digital_lr"] for p in pairs) / len(pairs),
                    "mean_paired_relative_reduction": reduction, "material_rule_passed": wins >= 4 and reduction >= .1})
    report = {"coverage_complete": complete, "deployment_rows": len(rows), "expected_rows": 60,
        "comparisons": comparisons, "scientific_gate": "awaiting_scientific_review", "sources_reselected": False, "training_executed": False}
    from experiments.artifacts import atomic_write_json
    atomic_write_json(output / "review.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate")
    p = commands.add_parser("prepare-host"); p.add_argument("--host", choices=HOSTS.values(), required=True); p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    q = commands.add_parser("queue"); q.add_argument("receipts", type=Path, nargs="+"); q.add_argument("--output", type=Path, required=True); q.add_argument("--controller", type=Path, required=True)
    c = commands.add_parser("collect"); c.add_argument("--queue-config", type=Path, required=True); c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate": result = generate()
    elif args.command == "prepare-host": result = prepare_host(args.host, args.base)
    elif args.command == "queue": result = queue(args.receipts, args.output, args.controller)
    else:
        from campaigns.lifecycle_collect import collect
        result = collect(args.queue_config, args.output, deployment_only=True)
        result["review"] = review(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
