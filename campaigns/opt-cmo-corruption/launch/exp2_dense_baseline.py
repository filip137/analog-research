"""Prepare and collect the dense 50M-token on-chip pilot through native stages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from exp1f_cdt_lr import checked_collection, service, transition
from slice_campaign import write_frozen
from experiments.artifacts import atomic_write_json
from workflow.lifecycle import parse_lifecycle, provenance, verify_provenance
from workflow.networks.opt_mlp import PINNED
from workflow.runtime import validate_source_fit

PHASE = "opt-cmo-exp2-dense-baseline-20261011"
PREFIX = "opt125m-cmo-l6-exp2-dense"
NOTE = "campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-008-overnight-recovery.md"
DEADLINE = "2026-10-12T07:30:00+02:00"
ARRAY = 271001
ARM = "standard_hwa_model"
LABELS = ("l1-smoke", "l6-canary", "50m")
AUTHORIZATION = ("User requested one epoch over the full synthetic 50M-token corpus, dense CMO on-chip writes without a maximum-write cap, "
                 "LR 5e-3; selected standard 3% Gaussian HWA first after being offered 3% actual corruption. "
                 "One-array exploratory baseline on the idle loulou RTX5090; digital parameters frozen. "
                 "Two bounded readiness probes; four GPU-hours and 20 GB total, ending by 07:30 Paris on October 12. "
                 "No sparse runs or further scientific grid authorized by this manifest.")


def source():
    return json.loads(Path(__file__).with_name("exp1g_four_arm_sources.json").read_text())["std3"]


def lifecycle(label):
    if label not in LABELS:
        raise ValueError("Unknown pilot case")
    doc = json.loads((ROOT / "campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1g-std3.json").read_text())
    doc["lifecycle_id"] = PREFIX + "-" + label
    doc["defects"]["cases"] = [c for c in doc["defects"]["cases"] if c["rate_ppm"] == 30000]
    doc["question"].update(text="Dense on-chip recovery from frozen standard 3% Gaussian HWA on one 3%-corrupt array; one full synthetic corpus pass.",
        decision="Report final held-state KL/perplexity versus matched initial deployment and write cost. One-array pilot, no test-selected checkpoint or general viability claim. " + NOTE)
    doc["onchip"] = {"update_law": "cmo_idealized_write_adam", "objective": "teacher_kl", "learning_rate": .005,
        "calibration_learning_rate": 0.0, "epochs": 1, "batch_size": 32, "seed": 53,
        "training_cohort": "hwa" if label == "50m" else "recovery",
        "arms": [{"id": "none", "weights": "hold", "calibration": False},
                 {"id": "onchip_weights", "weights": "learn", "calibration": False}]}
    if label == "l1-smoke":
        doc["network"]["analog_decoder_layers"] = 1
        doc["devices"]["assignment_seeds"] = [ARRAY]
        doc["data"].update(hwa_sequences=64, recovery_sequences=64, development_sequences=32)
        doc.pop("transfer_tests", None)
        doc["hwa"].update(arms=[{"id": ARM, "method": "none"}], epochs=1, sequences_per_epoch=64)
        doc["hwa"].pop("lr_schedule", None)
    if label != "50m":
        doc["question"]["text"] = "Readiness only; not scientific evidence. " + doc["question"]["text"]
    return parse_lifecycle(doc)


def generate():
    out = []
    for label in LABELS:
        lc = lifecycle(label)
        path = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
        write_frozen(path, lc.to_dict())
        out.append({"path": str(path), "sha256": lc.digest()})
    return out


def execution_manifest(plan, label):
    raw = json.loads(plan.read_text())
    selected = []
    for stage in raw["stages"]:
        if stage["id"].startswith("onchip__") and stage["id"] != f"onchip__{ARM}__{ARRAY}":
            continue
        if label != "l1-smoke" and stage["id"] == "hwa__" + ARM:
            continue
        stage["config"] = str((plan.parent / stage["config"]).resolve())
        if label != "l1-smoke":
            for role, ref in stage["inputs"].items():
                if ref.get("artifact_kind") == "hwa_source":
                    stage["inputs"][role] = {"path": source()["original_source"]}
            stage["depends_on"] = [dep for dep in stage["depends_on"] if dep != "hwa__" + ARM]
        selected.append(stage)
    raw["stages"] = selected
    from campaigns.schema import CampaignSpec
    CampaignSpec.parse(raw, base_dir=plan.parent)
    return raw


def prepare_host(base):
    if command(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Worker checkout must be clean and frozen")
    revision = command(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    python = str(Path(sys.executable).resolve())
    teacher, corpus_root = base / "inputs/opt-125m-27dcfa74/pytorch_model.bin", base / "inputs/corpora"
    inputs = [checked_input(teacher, PINNED["facebook/opt-125m"]["sha256"])]
    doc = lifecycle("50m").to_dict()
    for corpus in [doc["data"]["corpus"]] + [t["corpus"] for t in doc["transfer_tests"]]:
        inputs.append(checked_input(corpus_root / (corpus["name"] + ".pt"), corpus["sha256"]))
    ref = source()
    inputs += [checked_input(Path(ref[k]), ref[h]) for k, h in (("original_source", "sha256"), ("original_result", "result_sha256"))]
    original = parse_lifecycle(json.loads(Path(ref["original_result"]).with_name("config.resolved.json").read_text())["lifecycle"])
    for label in ("l6-canary", "50m"):
        lc = lifecycle(label)
        verify_provenance(provenance(original, "hwa_source", hwa_arm=ARM), lc, "hwa_source", hwa_arm=ARM)
        validate_source_fit(lc, lc.hwa.arm(ARM), json.loads(Path(ref["original_result"]).read_text())["metrics"]["fit"])
    env = {"EBL_CORPUS_ROOT": str(corpus_root), "EBL_CAMPAIGN_STREAM": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}
    cases = []
    for label in LABELS:
        lc = lifecycle(label)
        config = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
        if parse_lifecycle(json.loads(config.read_text())).digest() != lc.digest():
            raise ValueError("Frozen lifecycle changed")
        root = ROOT / "results/lifecycles" / lc.lifecycle_id
        command([python, "-m", "workflow", "plan", str(config), "--teacher-weights", str(teacher), "--output", str(root / "plan"),
                 "--device", "cuda:0", "--cpu-threads", "1", "--python", python], cwd=ROOT, env=dict(os.environ, **env))
        manifest = execution_manifest(root / "plan/campaign.json", label)
        manifest_path = root / "campaign.json"
        write_frozen(manifest_path, manifest)
        command([python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path), "--output-dir", str(root / "dry-run"),
                 "--dry-run", "--fail-fast"], cwd=ROOT, env=dict(os.environ, **env), logs=root / "preparation/dry-run")
        budget, expected = (12600, 7200) if label == "50m" else (900, 600)
        state_root, worker = base / "worker-receipts" / revision / lc.lifecycle_id, root / "tmux-worker.json"
        native_root = root / "runs" / manifest["campaign_id"]
        write_frozen(worker, {"case_id": lc.lifecycle_id, "state_root": str(state_root), "cwd": str(ROOT),
            "timeout_seconds": budget - 300, "lease_seconds": 900, "env": env,
            "argv": [python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                     "--output-dir", str(root / "runs"), "--resume", "--fail-fast"]})
        files = [config, manifest_path, worker] + [Path(s["config"]) for s in manifest["stages"]]
        if label != "l1-smoke":
            files += [Path(ref["original_source"]), Path(ref["original_result"])]
        cases.append({"id": lc.lifecycle_id, "label": label, "lifecycle_sha256": lc.digest(), "dry_run_passed": True,
            "budget_seconds": budget, "expected_seconds": expected, "stage_ids": [s["id"] for s in manifest["stages"]],
            "placement": {"workdir": str(ROOT), "run_dir": str(state_root), "gpu_models": ["5090"],
                "argv": [python, "-m", "campaigns.tmux_worker", "bridge", "--spec", str(worker)],
                "validate_argv": [python, "-m", "campaigns.lifecycle_collect", "validate", str(native_root),
                                  "--recovery", "--recovery-assignment", str(ARRAY)],
                "campaign_python": python, "campaign_worker_spec": str(worker), "campaign_result_root": str(native_root),
                "prepared_files": [{"path": str(p), "sha256": sha256(p)} for p in files]}})
    return {"host": "loulou", "source_revision": revision, "source": str(ROOT), "base": str(base), "inputs": inputs, "cases": cases}


def thresholds(expected):
    return {"expected_seconds": max(1, expected), "startup_seconds": 600, "stale_seconds": 900}


def queue(receipt, output, controller, evidence):
    prepared = json.loads(receipt.read_text())
    if prepared["host"] != "loulou" or {c["label"] for c in prepared["cases"]} != set(LABELS):
        raise ValueError("Incomplete authorized placement")
    cases, jobs = [], []
    for c in prepared["cases"]:
        label = c["label"]
        lc = lifecycle(label)
        if c["id"] != lc.lifecycle_id or c["lifecycle_sha256"] != lc.digest() or not c["dry_run_passed"]:
            raise ValueError("Preparation differs from frozen pilot")
        job = {"id": c["id"], "project_dir": str(controller), "handoff": NOTE, "authorization": AUTHORIZATION,
            "peak_memory_mib": 26000, "headroom_mib": 2700, "expected_seconds": c["expected_seconds"],
            "budget_seconds": c["budget_seconds"], "deadline": DEADLINE, "priority": 90,
            "startup_grace_seconds": 600, "stale_seconds": 900, "recovery": False, "retry_safe": False,
            "placements": {"loulou": c["placement"]}, "completion": {
                "argv": [sys.executable, str(controller / "campaigns/opt-cmo-corruption/launch/exp2_dense_baseline.py"), "collect",
                         "--queue-config", str(Path.home() / ".config/run-watch/queue.json"), "--evidence", str(evidence)],
                "workdir": str(controller), "timeout_seconds": 600, "retry_safe": True}}
        previous = None if label == "l1-smoke" else "l1-smoke" if label == "l6-canary" else "l6-canary"
        cases.append({"id": c["id"], "stage": "training" if label == "50m" else "readiness",
            "dependencies": [] if previous is None else [lifecycle(previous).lifecycle_id],
            "required_gates": ["handoff" if label == "50m" else "one_layer" if label == "l6-canary" else "preflight"],
            "expected_gpu_seconds": c["expected_seconds"], "expected_storage_bytes": 6000000000 if label != "l1-smoke" else 2000000000,
            "progress": {**thresholds(c["expected_seconds"]), "phases": {p: thresholds(t) for p, t in
                (("prepare", 180), ("devices", 120), ("hwa", 120), ("deploy", 300),
                 ("onchip", 6600 if label == "50m" else 300), ("validation", 300), ("collection", 300))}}})
        path = output / "jobs" / (c["id"] + ".json")
        write_frozen(path, job); jobs.append(str(path))
    manifest = {"version": 1, "id": PHASE, "poll_seconds": 300, "required_adapters": ["tmux"],
        "budgets": {"gpu_hours": 4, "storage_bytes": 20000000000, "window_seconds": 27000}, "cases": cases,
        "storage_roots": [{"target": "loulou", "path": prepared["source"]},
                          {"target": "loulou", "path": str(Path(prepared["base"]) / "worker-receipts" / prepared["source_revision"])}]}
    from campaigns.execution_service import validate_manifest
    validate_manifest(manifest)
    write_frozen(output / "campaign.json", manifest)
    return {"manifest": str(output / "campaign.json"), "jobs": jobs, "submitted": False}


def gate(queue_config, evidence, name, payload):
    from campaigns.execution_service import digest
    manifest = json.loads((evidence / "prepared/queue/campaign.json").read_text())
    path = evidence / "readiness" / (name + ".json")
    write_frozen(path, {"manifest_sha256": digest(manifest), "passed": True, **payload})
    service(queue_config, "gate", PHASE, name, path)


def measured_projection(folder):
    receipt = checked_collection(folder / "collection.json")
    stage = f"onchip__{ARM}__{ARRAY}"
    rows = json.loads((folder / stage / "measurements.json").read_text())
    learned = next(r for r in rows if r["onchip_arm"] == "onchip_weights")
    curve = learned["curve"]
    if len(curve) != 1 or learned["sequence_presentations"] != 8192:
        raise ValueError("Canary did not finish its exact 256-update training pass")
    metrics = [json.loads(line) for line in (folder / stage / "metrics.jsonl").read_text().splitlines()]
    memory = [m for m in metrics if m.get("stage") == "onchip_memory"]
    if not memory or max(m["peak_reserved_bytes"] for m in memory) > 26000 * 1024**2:
        raise ValueError("Missing measured CUDA memory or canary exceeds the prepared reservation")
    # Includes epoch-end evaluations, so this deliberately overestimates per-step cost.
    seconds_per_step = curve[0]["epoch_seconds"] / 256
    projected_training = math.ceil(seconds_per_step * 6101 * 1.35)
    projected_total = projected_training + 900
    if projected_total > 12300:
        raise ValueError("Measured full-pass projection exceeds the unchanged worker budget")
    remaining = datetime.fromisoformat(DEADLINE).timestamp() - datetime.now(timezone.utc).timestamp()
    if projected_total + 600 > remaining:
        raise ValueError("Measured full-pass projection cannot fit the overnight deadline")
    return {"canary_receipt": str(folder / "collection.json"), "seconds_per_step_including_evaluation": seconds_per_step,
            "peak_reserved_bytes": max(m["peak_reserved_bytes"] for m in memory),
            "full_training_seconds_with_margin": projected_training, "full_total_seconds_with_margin": projected_total}


def review(folder, evidence):
    receipt = checked_collection(folder / "collection.json")
    rows = json.loads((folder / "rows.json").read_text())
    hold = next(r for r in rows if r["onchip_arm"] == "none")
    learn = next(r for r in rows if r["onchip_arm"] == "onchip_weights")
    if len(rows) != 2 or learn["sequence_presentations"] != 195226:
        raise ValueError("Pilot coverage or full-corpus count differs")
    effects = {}
    for prefix in ("", "transfer_generated_", "transfer_wikitext_"):
        a, b = hold["final_" + prefix + "teacher_kl"], learn["final_" + prefix + "teacher_kl"]
        effects[prefix or "development"] = {"deployment_kl": a, "onchip_kl": b, "relative_reduction": 1 - b / a}
    report = {"coverage_complete": True, "source_sha256": source()["sha256"], "array_seed": ARRAY, "rate_ppm": 30000,
              "input_tokens": 49977856, "optimizer_steps": 6101, "learning_rate": .005, "effects": effects,
              "cost": {k: v for k, v in learn.items() if k.startswith("cost_")},
              "collection": str(folder / "collection.json"), "further_runs_dispatched": False,
              "interpretation": "One-array dense pilot versus its own initial deployment; calibration, rewrite, stronger HWA and independent-array comparisons remain required for viability."}
    atomic_write_json(evidence / "collected/review.json", report)
    return report


def collect(queue_config, evidence):
    from campaigns.lifecycle_collect import collect as native_collect
    job = os.environ["RUN_WATCH_JOB_ID"]
    label = next((label for label in LABELS if lifecycle(label).lifecycle_id == job), None)
    if label is None:
        raise ValueError("Unexpected completion job")
    started = time.monotonic()
    result = native_collect(queue_config, evidence / "collected" / label, recovery=True, recovery_assignments=[ARRAY])
    collection_seconds = time.monotonic() - started
    folder = Path(result["collected"])
    with (evidence / ".completion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if label == "l1-smoke":
            gate(queue_config, evidence, "one_layer", {"evidence": [str(folder / "collection.json")]})
            transition(queue_config, "release", lifecycle("l6-canary").lifecycle_id)
        elif label == "l6-canary":
            projection = measured_projection(folder)
            # Freeze a measured receipt once, including idempotent completion-hook replay.
            timing_path = evidence / "readiness/canary-timing.json"
            if not timing_path.exists():
                phases = {}
                for stage in checked_collection(folder / "collection.json")["stages"]:
                    record = json.loads((folder / stage / "result.json").read_text())
                    kind = record["metrics"]["stage"]["kind"]
                    phases[kind] = max(phases.get(kind, 0), record["duration_seconds"])
                write_frozen(timing_path, {"native_stage_seconds": phases, "collection_and_revalidation_seconds": collection_seconds})
            timing = json.loads(timing_path.read_text())
            calibration = {p: thresholds(math.ceil(t * 2 + 60)) for p, t in timing["native_stage_seconds"].items()}
            calibration.update(onchip=thresholds(projection["full_training_seconds_with_margin"]),
                validation=thresholds(math.ceil(timing["collection_and_revalidation_seconds"] * 2 + 60)),
                collection=thresholds(math.ceil(timing["collection_and_revalidation_seconds"] * 2 + 60)))
            gate(queue_config, evidence, "handoff", {"evidence": [str(folder / "collection.json"), str(evidence / "readiness/preflight.json")],
                **projection, "progress_calibration": {"training": thresholds(projection["full_total_seconds_with_margin"])},
                "phase_progress_calibration": calibration})
            transition(queue_config, "release", lifecycle("50m").lifecycle_id)
            result["projection"] = projection
        else:
            result["review"] = review(folder, evidence)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    p = sub.add_parser("prepare-host"); p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    q = sub.add_parser("queue"); q.add_argument("receipt", type=Path); q.add_argument("--output", type=Path, required=True)
    q.add_argument("--controller", type=Path, required=True); q.add_argument("--evidence", type=Path, required=True)
    c = sub.add_parser("collect"); c.add_argument("--queue-config", type=Path, required=True); c.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate": result = generate()
    elif args.command == "prepare-host": result = prepare_host(args.base)
    elif args.command == "queue": result = queue(args.receipt, args.output, args.controller, args.evidence)
    else: result = collect(args.queue_config, args.evidence)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
