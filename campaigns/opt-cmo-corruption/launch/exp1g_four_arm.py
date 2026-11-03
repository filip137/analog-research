"""Native four-arm HWA comparison under the existing durable campaign service."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from exp1f_cdt_lr import checked_collection, service, transition
from slice_campaign import build_slice, write_frozen
from core.artifacts import atomic_write_json
from ebl.schema.lifecycle import parse_lifecycle, provenance, verify_provenance
from networks.opt import PINNED
from ebl.schema.source_fit import validate_source_fit

PHASE = "opt-cmo-exp1g-four-arm-20261011"
PREFIX = "opt125m-cmo-l6-exp1g"
NOTE = "campaigns/opt-cmo-corruption/series/001-hwa-robustness/experiments/exp-007-corruption-rate-sweep.md"
SOURCES = Path(__file__).with_name("exp1g_four_arm_sources.json")
DEADLINE = "2026-10-12T08:00:00+02:00"
CASES = {
    "cdt5-smoke": ("fifi", "cdt_random_5pct_model", "smoke"),
    "cdt5": ("fifi", "cdt_random_5pct_model", "full"),
    "std9": ("loulou", "standard_hwa_9pct_model", "full"),
    "std3": ("riri", "standard_hwa_model", "transfer"),
    "cdt3": ("riri", "cdt_random_3pct_model", "transfer"),
}
AUTHORIZATION = "User authorized the four-arm comparison with frozen split LRs: two new fits (3% Gaussian plus 5% faults; 9% Gaussian without faults) on two RTX5090s, and evaluation of the existing 0%-fault and 3%-fault sources on a third RTX5090, across 0-5% deployed faults. Use fifi/loulou for training and riri for evaluation. Seven GPU-hour cap; one bounded readiness canary."


def lifecycle(label):
    if label not in CASES:
        raise ValueError("Unassigned scientific case")
    _, arm_id, mode = CASES[label]
    doc = json.loads((ROOT / "campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1e.json").read_text())
    template = "cdt_random_3pct_model" if label.startswith("cdt5") else arm_id
    arm = copy.deepcopy(next(a for a in doc["hwa"]["arms"] if a["id"] == template))
    arm.update(id=arm_id, digital_learning_rate=1e-5)
    if label.startswith("cdt5"):
        arm["corruption"] = {"kind": "random", "rates_ppm": [50000]}
    doc["hwa"]["arms"] = [arm]
    doc["lifecycle_id"] = PREFIX + "-" + label
    doc["question"].update(text=("Readiness only. " if mode == "smoke" else "") +
        "Four-arm whole-model HWA: standard 3%, standard 9%, and 3% Gaussian plus temporary 3% or 5% random faults; frozen split learning rates.",
        decision="Compare paired KL and perplexity per fault rate and domain; material effect requires 4/5 array wins and 10% mean paired KL reduction. No test-driven retuning. See " + NOTE)
    doc["data"].update(evaluation="development", test_sequences=0)
    if mode == "smoke":
        doc.pop("transfer_tests", None)
        doc["hwa"]["epochs"] = 2
        doc["hwa"]["lr_schedule"]["total_steps"] = 512
    return parse_lifecycle(doc)


def generate():
    rows = []
    for label in CASES:
        lc = lifecycle(label)
        path = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
        write_frozen(path, lc.to_dict())
        rows.append({"path": str(path), "lifecycle_sha256": lc.digest()})
    return rows


def deployment_manifest(plan, source, arm):
    raw = json.loads(plan.read_text())
    stages = []
    for stage in raw["stages"]:
        if stage["id"] == "hwa__" + arm:
            continue
        if stage["id"] not in {"prepare", "devices"} and not stage["id"].startswith("deploy__" + arm + "__"):
            raise ValueError("Unexpected stage in frozen-source evaluation")
        stage["config"] = str((plan.parent / stage["config"]).resolve())
        if stage["id"].startswith("deploy__"):
            stage["inputs"]["weights"] = {"path": str(source)}
            stage["depends_on"] = [s for s in stage["depends_on"] if s != "hwa__" + arm]
        stages.append(stage)
    raw["stages"] = stages
    from ebl.execution.schema import CampaignSpec
    CampaignSpec.parse(raw, base_dir=plan.parent)
    return raw


def prepare_host(host, base):
    if command(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Worker checkout must be clean and frozen")
    revision = command(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    python = str(Path(sys.executable).resolve())
    teacher, corpus_root = base / "inputs/opt-125m-27dcfa74/pytorch_model.bin", base / "inputs/corpora"
    inputs = [checked_input(teacher, PINNED["facebook/opt-125m"]["sha256"])]
    doc = lifecycle("std3").to_dict()
    for corpus in [doc["data"]["corpus"]] + [t["corpus"] for t in doc["transfer_tests"]]:
        inputs.append(checked_input(corpus_root / (corpus["name"] + ".pt"), corpus["sha256"]))
    env = {"EBL_CORPUS_ROOT": str(corpus_root), "EBL_CAMPAIGN_STREAM": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}
    refs, cases = json.loads(SOURCES.read_text()), []
    for label, (target, arm, mode) in CASES.items():
        if host != target:
            continue
        lc = lifecycle(label)
        config = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
        if parse_lifecycle(json.loads(config.read_text())).digest() != lc.digest():
            raise ValueError("Frozen lifecycle changed")
        root = ROOT / "results/lifecycles" / lc.lifecycle_id
        plan, manifest_path = root / "plan", root / "campaign.json"
        command([python, "-m", "ebl", "lifecycle", "plan", str(config), "--teacher-weights", str(teacher),
                 "--output", str(plan), "--device", "cuda:0", "--cpu-threads", "1", "--python", python], cwd=ROOT, env=dict(os.environ, **env))
        extra = []
        if mode == "transfer":
            source = refs[label]
            inputs.extend([checked_input(Path(source[k]), source[h]) for k, h in (("path", "sha256"), ("result", "result_sha256"))])
            original = parse_lifecycle(json.loads(Path(source["result"]).with_name("config.resolved.json").read_text())["lifecycle"])
            verify_provenance(provenance(original, "hwa_source", hwa_arm=arm), lc, "hwa_source", hwa_arm=arm)
            validate_source_fit(lc, lc.hwa.arm(arm), json.loads(Path(source["result"]).read_text())["metrics"]["fit"])
            manifest = deployment_manifest(plan / "campaign.json", Path(source["path"]), arm)
            extra = [Path(source["path"]), Path(source["result"])]
        else:
            selected = ["hwa__" + arm] if mode == "smoke" else ["deploy__" + arm + "__" + str(s) for s in lc.devices.assignment_seeds]
            manifest, sliced = build_slice(plan / "campaign.json", selected)
            write_frozen(root / "slice.json", sliced)
        write_frozen(manifest_path, manifest)
        command([python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                 "--output-dir", str(root / "dry-run"), "--dry-run", "--fail-fast"], cwd=ROOT, env=dict(os.environ, **env), logs=root / "preparation/dry-run")
        budget, expected = (900, 300) if mode == "smoke" else (1200, 600) if mode == "transfer" else (9000, 7000)
        state_root, worker = base / "worker-receipts" / revision / lc.lifecycle_id, root / "tmux-worker.json"
        native_root = root / "runs" / manifest["campaign_id"]
        write_frozen(worker, {"case_id": lc.lifecycle_id, "state_root": str(state_root), "cwd": str(ROOT),
            "timeout_seconds": budget, "lease_seconds": 900, "env": env,
            "argv": [python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                     "--output-dir", str(root / "runs"), "--resume", "--fail-fast"]})
        validation = [python, "-m", "ebl.execution.lifecycle_collect", "validate", str(native_root)]
        if mode != "full":
            validation += ["--source-only" if mode == "smoke" else "--deployment-only"]
        files = [config, manifest_path, worker] + extra + [Path(s["config"]) for s in manifest["stages"]]
        cases.append({"id": lc.lifecycle_id, "label": label, "mode": mode, "lifecycle_sha256": lc.digest(),
            "dry_run_passed": True, "stage_ids": [s["id"] for s in manifest["stages"]], "budget_seconds": budget, "expected_seconds": expected,
            "placement": {"workdir": str(ROOT), "run_dir": str(state_root), "gpu_models": ["5090"],
                "argv": [python, "-m", "ebl.execution.tmux_worker", "bridge", "--spec", str(worker)], "validate_argv": validation,
                "campaign_python": python, "campaign_worker_spec": str(worker), "campaign_result_root": str(native_root),
                "prepared_files": [{"path": str(p), "sha256": sha256(p)} for p in files]}})
    if not cases:
        raise ValueError("Host has no authorized cases")
    return {"host": host, "source_revision": revision, "source": str(ROOT), "base": str(base), "inputs": inputs, "cases": cases}


def queue(receipts, output, controller, evidence):
    hosts = {p.stem: json.loads(p.read_text()) for p in receipts}
    if set(hosts) != {v[0] for v in CASES.values()} or len({v["source_revision"] for v in hosts.values()}) != 1:
        raise ValueError("Expected all three designated hosts at one frozen revision")
    jobs, cases = [], []
    for host, prepared in hosts.items():
        for c in prepared["cases"]:
            label, mode = c["label"], c["mode"]
            if CASES[label][0] != host or CASES[label][2] != mode or c["lifecycle_sha256"] != lifecycle(label).digest() or not c["dry_run_passed"]:
                raise ValueError("Prepared case does not match the frozen comparison")
            if mode == "transfer" and any(s.startswith("hwa__") for s in c["stage_ids"]):
                raise ValueError("Frozen source must never be retrained")
            job = {"id": c["id"], "project_dir": str(controller), "handoff": NOTE, "authorization": AUTHORIZATION,
                "peak_memory_mib": 17000 if mode == "transfer" else 21500, "headroom_mib": 2200,
                "expected_seconds": c["expected_seconds"], "budget_seconds": c["budget_seconds"], "deadline": DEADLINE,
                "priority": 100 if mode == "smoke" else 70 if mode == "full" else 50,
                "startup_grace_seconds": 600, "stale_seconds": 900, "recovery": False, "retry_safe": False,
                "placements": {host: c["placement"]}, "completion": {
                    "argv": [sys.executable, str(controller / "campaigns/opt-cmo-corruption/launch/exp1g_four_arm.py"), "collect",
                             "--queue-config", str(Path.home() / ".config/run-watch/queue.json"), "--evidence", str(evidence)],
                    "workdir": str(controller), "timeout_seconds": 600, "retry_safe": True}}
            def threshold(seconds):
                return {"expected_seconds": seconds, "startup_seconds": 600, "stale_seconds": 900}
            cases.append({"id": c["id"], "stage": "readiness" if mode == "smoke" else "deployment" if mode == "transfer" else "training",
                "dependencies": [lifecycle("cdt5-smoke").lifecycle_id] if mode == "full" else [],
                "required_gates": ["handoff"] if mode == "full" else ["preflight"],
                "expected_gpu_seconds": c["expected_seconds"], "expected_storage_bytes": 1500000000 if mode == "smoke" else 4000000000,
                "progress": {**threshold(c["expected_seconds"]), "phases": {p: threshold(t) for p, t in
                    (("prepare", 120), ("devices", 120), ("hwa", 300 if mode == "smoke" else 6600), ("deploy", 180), ("validation", 600), ("collection", 600))}}})
            path = output / "jobs" / (c["id"] + ".json")
            write_frozen(path, job); jobs.append(str(path))
    if len(cases) != len(CASES) or {c["id"] for c in cases} != {lifecycle(l).lifecycle_id for l in CASES}:
        raise ValueError("Missing or duplicate prepared cases")
    manifest = {"version": 1, "id": PHASE, "poll_seconds": 300, "required_adapters": ["tmux"],
        "budgets": {"gpu_hours": 7, "storage_bytes": 24000000000, "window_seconds": 21600}, "cases": cases,
        "storage_roots": [{"target": h, "path": p["source"]} for h, p in hosts.items()] +
            [{"target": h, "path": str(Path(p["base"]) / "worker-receipts" / p["source_revision"])} for h, p in hosts.items()] +
            [{"target": "riri", "path": str(DEFAULT_BASE / "inputs/exp1g-frozen-sources")} ]}
    from ebl.execution.service import validate_manifest
    validate_manifest(manifest)
    write_frozen(output / "campaign.json", manifest)
    return {"manifest": str(output / "campaign.json"), "jobs": jobs, "submitted": False}


def gate(queue_config, evidence, name, payload):
    from ebl.execution.service import digest
    manifest = json.loads((evidence / "prepared/queue/campaign.json").read_text())
    path = evidence / "readiness" / (name + ".json")
    write_frozen(path, {"manifest_sha256": digest(manifest), "passed": True, **payload})
    service(queue_config, "gate", PHASE, name, path)


def release_after_canary(queue_config, evidence, folder):
    receipt = checked_collection(folder / "collection.json")
    arm = CASES["cdt5-smoke"][1]
    result = json.loads((folder / ("hwa__" + arm) / "result.json").read_text())
    lc = lifecycle("cdt5-smoke")
    validate_source_fit(lc, lc.hwa.arm(arm), result["metrics"]["fit"])
    if json.loads((evidence / "readiness/preflight.json").read_text())["passed"] is not True:
        raise ValueError("Readiness preflight did not pass")
    gate(queue_config, evidence, "handoff", {"evidence": [str(folder / "collection.json"), str(evidence / "readiness/preflight.json")],
        "canary_job": receipt["job"], "canary_attempt": receipt["attempt"]})
    for label in ("cdt5", "std9"):
        transition(queue_config, "release", lifecycle(label).lifecycle_id)
    return {"canary_passed": True, "full_runs_released": True}


def compare_rows(rows):
    expected = {(s, c.label) for s in lifecycle("std3").devices.assignment_seeds for c in lifecycle("std3").defects.cases}
    groups = {}
    for row in rows:
        label = next((l for l in CASES if l != "cdt5-smoke" and lifecycle(l).lifecycle_id == row["lifecycle_id"]), None)
        if label is None or row["hwa_arm"] != CASES[label][1]:
            raise ValueError("Unexpected scientific source")
        key = row["array_seed"], row["case"]
        if key in groups.setdefault(label, {}):
            raise ValueError("Duplicate array/case")
        groups[label][key] = row
        for domain in ("wikitext", "generated"):
            for metric in ("teacher_kl", "perplexity"):
                value = row["deployed_transfer_" + domain + "_" + metric]
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Nonfinite or nonpositive test metric")
    if any(set(group) != expected for group in groups.values()):
        raise ValueError("Partial source deployment coverage")
    comparisons, means = [], []
    for label, group in sorted(groups.items()):
        for case in lifecycle("std3").defects.cases:
            for domain in ("wikitext", "generated"):
                means.append({"label": label, "case": case.label, "domain": domain,
                    **{metric: sum(r["deployed_transfer_" + domain + "_" + metric] for (s, c), r in group.items() if c == case.label) / 5
                       for metric in ("teacher_kl", "perplexity")}})
    for control, selected in (("std3", "cdt3"), ("std3", "cdt5"), ("cdt3", "cdt5"),
                              ("std3", "std9"), ("std9", "cdt3"), ("std9", "cdt5")):
        if control not in groups or selected not in groups:
            continue
        for case in lifecycle("std3").defects.cases:
            for domain in ("wikitext", "generated"):
                key = "deployed_transfer_" + domain + "_teacher_kl"
                pairs = [{"array_seed": seed, "control": groups[control][seed, case.label][key],
                          "selected": groups[selected][seed, case.label][key]} for seed in lifecycle("std3").devices.assignment_seeds]
                for p in pairs:
                    p["relative_reduction"] = 1 - p["selected"] / p["control"]
                effect, wins = sum(p["relative_reduction"] for p in pairs) / 5, sum(p["selected"] < p["control"] for p in pairs)
                comparisons.append({"control": control, "selected": selected, "case": case.label, "domain": domain,
                    "pairs": pairs, "wins": wins, "mean_paired_relative_reduction": effect, "material_rule_passed": wins >= 4 and effect >= .1})
    return {"coverage_complete": len(groups) == 4, "deployment_rows": len(rows), "sources_present": sorted(groups),
            "means": means, "comparisons": comparisons, "scientific_gate": "awaiting_scientific_review", "further_runs_dispatched": False}


def review(evidence):
    index = evidence / "collected/main/index.json"
    entries = json.loads(index.read_text()) if index.exists() else []
    refs, rows, sources, identities = json.loads(SOURCES.read_text()), [], {}, {}
    for entry in entries:
        label = next(l for l in CASES if lifecycle(l).lifecycle_id == entry["job"])
        arm = CASES[label][1]
        folder = Path(entry["receipt"]).parent
        receipt = checked_collection(folder / "collection.json")
        source_stage = "reused-source" if CASES[label][2] == "transfer" else "hwa__" + arm
        source = json.loads((folder / source_stage / "result.json").read_text())
        original = parse_lifecycle(json.loads((folder / source_stage / "config.resolved.json").read_text())["lifecycle"])
        lc = lifecycle(label)
        verify_provenance(provenance(original, "hwa_source", hwa_arm=arm), lc, "hwa_source", hwa_arm=arm)
        fit = source["metrics"]["fit"]
        validate_source_fit(lc, lc.hwa.arm(arm), fit)
        artifact = next(a for a in source["artifacts"] if a["kind"] == "hwa_source")
        if label in refs and artifact["sha256"] != refs[label]["sha256"]:
            raise ValueError("Reused checkpoint changed")
        selected = next(r for r in fit["history"] if r["epoch"] == fit["selected_epoch"])
        sources[label] = {"source_sha256": artifact["sha256"], "mean_development_kl": fit["development_kl"],
            "selected_update": selected["global_step"], "cases": selected["cases"], "convergence_review_required": fit["convergence_review_required"]}
        identities[label] = {}
        for name in receipt["stages"]:
            if name.startswith("deploy__"):
                metrics = json.loads((folder / name / "result.json").read_text())["metrics"]
                for case, v in metrics["cases"].items():
                    identities[label][metrics["array_seed"], case] = v["identity"]
        rows.extend(json.loads((folder / "deployments.json").read_text()))
    if identities and any(v != next(iter(identities.values())) for v in identities.values()):
        raise ValueError("Physical identities are not matched across recipes")
    report = {**compare_rows(rows), "sources": sources, "physical_identities_matched": bool(identities)}
    atomic_write_json(evidence / "collected/review.json", report)
    return {k: v for k, v in report.items() if k not in {"means", "comparisons", "sources"}}


def collect(queue_config, evidence):
    from ebl.execution.lifecycle_collect import collect as native_collect
    job = os.environ["RUN_WATCH_JOB_ID"]
    label = next((l for l in CASES if lifecycle(l).lifecycle_id == job), None)
    if label is None:
        raise ValueError("Unexpected completion job")
    mode = CASES[label][2]
    result = native_collect(queue_config, evidence / "collected" / ("smoke" if mode == "smoke" else "main"),
                            source_only=mode == "smoke", deployment_only=mode == "transfer")
    with (evidence / ".completion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if mode == "smoke":
            result["advance"] = release_after_canary(queue_config, evidence, Path(result["collected"]))
        else:
            result["review"] = review(evidence)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    p = sub.add_parser("prepare-host"); p.add_argument("--host", choices=sorted({v[0] for v in CASES.values()}), required=True); p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    q = sub.add_parser("queue"); q.add_argument("receipts", nargs="+", type=Path); q.add_argument("--output", type=Path, required=True); q.add_argument("--controller", type=Path, required=True); q.add_argument("--evidence", type=Path, required=True)
    c = sub.add_parser("collect"); c.add_argument("--queue-config", type=Path, required=True); c.add_argument("--evidence", type=Path, required=True)
    r = sub.add_parser("review"); r.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate": result = generate()
    elif args.command == "prepare-host": result = prepare_host(args.host, args.base)
    elif args.command == "queue": result = queue(args.receipts, args.output, args.controller, args.evidence)
    elif args.command == "review": result = review(args.evidence)
    else: result = collect(args.queue_config, args.evidence)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
