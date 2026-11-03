"""Frozen corruption-aware digital-LR comparison using native lifecycle stages.

The shared queue owns every GPU worker. Completion hooks collect native evidence,
apply the predeclared development rule, and release only the selected deployment.
No training runtime or persistent dispatcher implementation is changed here.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from slice_campaign import build_slice, write_frozen
from core.artifacts import atomic_write_json
from ebl.schema.lifecycle import parse_lifecycle, provenance, verify_provenance
from networks.opt import PINNED
from ebl.schema.source_fit import validate_source_fit

PHASE = "opt-cmo-exp1f-cdt-lr-20261011"
PREFIX = "opt125m-cmo-l6-exp1f-cdt"
NOTE = "campaigns/opt-cmo-corruption/series/001-hwa-robustness/experiments/exp-006-corruption-aware-digital-lr.md"
ARM = "cdt_random_3pct_model"
RATES = {"1em5": 1e-5, "1em4": 1e-4}
HOSTS = {"1em5": "fifi", "1em4": "loulou"}
REFERENCES = Path(__file__).with_name("exp1f_references.json")
QUEUE = Path.home() / ".codex/skills/run-watch/scripts/gpu_queue.py"
DEADLINE = "2026-10-12T08:00:00+02:00"


def lifecycle(label, mode="training"):
    if label not in RATES or mode not in {"training", "smoke", "transfer"}:
        raise ValueError("Unknown frozen case")
    doc = json.loads((ROOT / "campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1e.json").read_text())
    arm = copy.deepcopy(next(a for a in doc["hwa"]["arms"] if a["id"] == ARM))
    arm["digital_learning_rate"] = RATES[label]
    doc["hwa"]["arms"] = [arm]
    doc["lifecycle_id"] = PREFIX + ("-transfer" if mode == "transfer" else "") + "-lr" + label + ("-smoke" if mode == "smoke" else "")
    doc["question"].update(text=("Readiness only. " if mode == "smoke" else "") +
        "Corruption-aware 3% HWA with analog cosine LR 5e-3 to 1e-3 and smaller additional digital LR.",
        decision="Select minimum mean healthy/3%-fault development KL over digital peaks 0, 1e-5, 1e-4; within 1% prefer smaller LR. Evaluate only the frozen winner. See " + NOTE)
    doc["data"].update(evaluation="development", test_sequences=0)
    if mode != "transfer":
        doc.pop("transfer_tests", None)
    if mode == "smoke":
        doc["hwa"]["epochs"] = 2
        doc["hwa"]["lr_schedule"]["total_steps"] = 512
    return parse_lifecycle(doc)


def definitions(host=None):
    cases = [(label, mode) for label in RATES for mode in ("training", "transfer")]
    cases.append(("1em5", "smoke"))
    return [(label, mode) for label, mode in cases if host is None or HOSTS[label] == host]


def generate():
    result = []
    for label, mode in definitions():
        lc = lifecycle(label, mode)
        p = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
        write_frozen(p, lc.to_dict())
        result.append({"path": str(p), "lifecycle_sha256": lc.digest()})
    return result


def deployment_manifest(plan_path, source_path):
    raw = json.loads(plan_path.read_text())
    kept = []
    for stage in raw["stages"]:
        if stage["id"] == "hwa__" + ARM:
            continue
        if stage["id"] not in {"prepare", "devices"} and not stage["id"].startswith("deploy__" + ARM + "__"):
            raise ValueError("Unexpected stage in selected deployment")
        stage["config"] = str((plan_path.parent / stage["config"]).resolve())
        if stage["id"].startswith("deploy__"):
            stage["inputs"]["weights"] = {"path": str(source_path)}
            stage["depends_on"] = [d for d in stage["depends_on"] if d != "hwa__" + ARM]
        kept.append(stage)
    raw["stages"] = kept
    from ebl.execution.schema import CampaignSpec
    CampaignSpec.parse(raw, base_dir=plan_path.parent)
    return raw


def prepare_host(host, base):
    if command(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Expected a clean frozen worker checkout")
    revision = command(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    python = str(Path(sys.executable).resolve())
    teacher = base / "inputs/opt-125m-27dcfa74/pytorch_model.bin"
    corpus_root = base / "inputs/corpora"
    lc = lifecycle("1em5", "transfer")
    inputs = [checked_input(teacher, PINNED["facebook/opt-125m"]["sha256"])]
    for c in [lc.to_dict()["data"]["corpus"]] + [t["corpus"] for t in lc.to_dict()["transfer_tests"]]:
        inputs.append(checked_input(corpus_root / (c["name"] + ".pt"), c["sha256"]))
    env = {"EBL_CORPUS_ROOT": str(corpus_root), "EBL_CAMPAIGN_STREAM": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}
    cases = []
    for label, mode in definitions(host):
        lc = lifecycle(label, mode)
        config = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lc.lifecycle_id + ".json")
        if parse_lifecycle(json.loads(config.read_text())).digest() != lc.digest():
            raise ValueError("Frozen lifecycle changed")
        case_root = ROOT / "results/lifecycles" / lc.lifecycle_id
        plan = case_root / "plan"
        command([python, "-m", "ebl", "lifecycle", "plan", str(config), "--teacher-weights", str(teacher), "--output", str(plan),
                 "--device", "cuda:0", "--cpu-threads", "1", "--python", python], cwd=ROOT, env=dict(os.environ, **env))
        manifest_path = case_root / ("campaign-deployment.json" if mode == "transfer" else "campaign-training.json")
        if mode == "transfer":
            # This path is bound to the exact collected source before dispatch.
            # A dry run validates native stage wiring without loading weights.
            manifest = deployment_manifest(plan / "campaign.json", case_root / "selected-source.pt")
            # The existing native full-plan dry run resolves its future HWA
            # artifact symbolically. The selected external path is validated by
            # a second native dry run after selection, before any deployment.
            dry_manifest = plan / "campaign.json"
            native_root = case_root / "runs" / manifest["campaign_id"]
            argv = [python, str(Path(__file__).resolve()), "execute-deployment", "--label", label,
                    "--selection", str(ROOT / "results/exp1f-selection.json")]
        else:
            manifest, sliced = build_slice(plan / "campaign.json", ["hwa__" + ARM])
            write_frozen(manifest_path, manifest)
            write_frozen(case_root / "slice.json", sliced)
            dry_manifest = manifest_path
            native_root = case_root / "runs" / manifest["campaign_id"]
            argv = [python, "-m", "ebl", "campaign", "run", "--manifest", str(manifest_path),
                    "--output-dir", str(case_root / "runs"), "--resume", "--fail-fast"]
        command([python, "-m", "ebl", "campaign", "run", "--manifest", str(dry_manifest), "--output-dir", str(case_root / "dry-run"),
                 "--dry-run", "--fail-fast"], cwd=ROOT, env=dict(os.environ, **env), logs=case_root / "preparation/dry-run")
        budget, expected = (900, 300) if mode == "smoke" else (1200, 600) if mode == "transfer" else (9000, 6600)
        worker_root = base / "worker-receipts" / revision / lc.lifecycle_id
        worker_path = case_root / "tmux-worker.json"
        write_frozen(worker_path, {"case_id": lc.lifecycle_id, "state_root": str(worker_root), "cwd": str(ROOT),
            "timeout_seconds": budget, "lease_seconds": 900, "env": env, "argv": argv})
        prepared = [config, worker_path, dry_manifest] + [Path(s["config"]) for s in manifest["stages"]]
        cases.append({"id": lc.lifecycle_id, "label": label, "mode": mode, "lifecycle_sha256": lc.digest(), "dry_run_passed": True,
            "budget_seconds": budget, "expected_seconds": expected, "placement": {"workdir": str(ROOT), "run_dir": str(worker_root),
            "argv": [python, "-m", "ebl.execution.tmux_worker", "bridge", "--spec", str(worker_path)],
            "validate_argv": [python, "-m", "ebl.execution.lifecycle_collect", "validate", str(native_root),
                              "--deployment-only" if mode == "transfer" else "--source-only"],
            "campaign_python": python, "campaign_worker_spec": str(worker_path), "campaign_result_root": str(native_root),
            "gpu_models": ["5090"], "prepared_files": [{"path": str(p), "sha256": sha256(p)} for p in prepared]}})
    return {"host": host, "source_revision": revision, "source": str(ROOT), "base": str(base), "inputs": inputs, "cases": cases}


def execute_deployment(label, selection_path):
    selection = json.loads(selection_path.read_text())
    if selection["selected_label"] != label or selection["test_data_used"] is not False:
        raise ValueError("Deployment does not match frozen development selection")
    source = selection["selected_source"]
    checked_input(Path(source["source_path"]), source["source_sha256"])
    checked_input(Path(source["native_result"]), source["result_sha256"])
    lc = lifecycle(label, "transfer")
    original = parse_lifecycle(json.loads(Path(source["native_result"]).with_name("config.resolved.json").read_text())["lifecycle"])
    verify_provenance(provenance(original, "hwa_source", hwa_arm=ARM), lc, "hwa_source", hwa_arm=ARM)
    validate_source_fit(lc, lc.hwa.arm(ARM), json.loads(Path(source["native_result"]).read_text())["metrics"]["fit"])
    case_root = ROOT / "results/lifecycles" / lc.lifecycle_id
    manifest = deployment_manifest(case_root / "plan/campaign.json", Path(source["source_path"]))
    path = case_root / "campaign-deployment.json"
    write_frozen(path, manifest)
    argv = [sys.executable, "-m", "ebl", "campaign", "run", "--manifest", str(path), "--output-dir", str(case_root / "runs")]
    subprocess.run(argv + ["--dry-run", "--fail-fast"], cwd=ROOT, check=True)
    subprocess.run(argv + ["--resume", "--fail-fast"], cwd=ROOT, check=True)
    return {"campaign_root": str(case_root / "runs" / manifest["campaign_id"])}


def queue(receipts, output, controller, evidence):
    hosts = {p.stem: json.loads(p.read_text()) for p in receipts}
    if set(hosts) != set(HOSTS.values()) or len({h["source_revision"] for h in hosts.values()}) != 1:
        raise ValueError("Expected both hosts at one frozen revision")
    jobs, cases = [], []
    for host, h in hosts.items():
        for c in h["cases"]:
            label, mode = c["label"], c["mode"]
            lc = lifecycle(label, mode)
            if HOSTS[label] != host or c["lifecycle_sha256"] != lc.digest() or not c["dry_run_passed"]:
                raise ValueError("Prepared scientific case mismatch")
            job = {"id": c["id"], "project_dir": str(controller), "handoff": NOTE,
                "authorization": "User approved two corruption-aware HWA fits with digital peaks 1e-5 and 1e-4, reuse zero control, frozen joint development selection and evaluation of only the winner on fifi/loulou; 6 GPU-hour cap. One bounded readiness canary.",
                "peak_memory_mib": 21500 if mode != "transfer" else 17000, "headroom_mib": 2200,
                "expected_seconds": c["expected_seconds"], "budget_seconds": c["budget_seconds"], "deadline": DEADLINE,
                "priority": 100 if mode == "smoke" else 50, "startup_grace_seconds": 600, "stale_seconds": 900,
                "recovery": False, "retry_safe": False, "placements": {host: c["placement"]}, "completion": {
                    "argv": [sys.executable, str(controller / "campaigns/opt-cmo-corruption/launch/exp1f_cdt_lr.py"), "collect",
                        "--queue-config", str(Path.home() / ".config/run-watch/queue.json"), "--evidence", str(evidence)],
                    "workdir": str(controller), "timeout_seconds": 600, "retry_safe": True}}
            def threshold(seconds):
                return {"expected_seconds": seconds, "startup_seconds": 600, "stale_seconds": 900}
            cases.append({"id": c["id"], "stage": "readiness" if mode == "smoke" else "deployment" if mode == "transfer" else "training",
                "dependencies": [lifecycle(l).lifecycle_id for l in RATES] if mode == "transfer" else [],
                "required_gates": [] if mode == "smoke" else ["selection_" + label] if mode == "transfer" else ["handoff"],
                "expected_gpu_seconds": c["expected_seconds"], "expected_storage_bytes": 4000000000 if mode == "transfer" else 1500000000,
                "progress": {**threshold(c["expected_seconds"]), "phases": {p: threshold(s) for p, s in
                    (("prepare", 120), ("devices", 120), ("hwa", 300 if mode == "smoke" else 6600), ("deploy", 180), ("validation", 600), ("collection", 600))}}})
            path = output / "jobs" / (c["id"] + ".json")
            write_frozen(path, job); jobs.append(str(path))
    if {c["id"] for c in cases} != {lifecycle(l, m).lifecycle_id for l, m in definitions()}:
        raise ValueError("Incomplete prepared cases")
    manifest = {"version": 1, "id": PHASE, "poll_seconds": 300, "required_adapters": ["tmux"],
        "budgets": {"gpu_hours": 6, "storage_bytes": 16000000000, "window_seconds": 21600}, "cases": cases,
        "storage_roots": [{"target": host, "path": h["source"]} for host, h in hosts.items()] +
            [{"target": host, "path": str(Path(h["base"]) / "worker-receipts" / h["source_revision"])} for host, h in hosts.items()]}
    from ebl.execution.service import validate_manifest
    validate_manifest(manifest)
    write_frozen(output / "campaign.json", manifest)
    return {"manifest": str(output / "campaign.json"), "jobs": jobs, "submitted": False}


def checked_collection(path):
    receipt = json.loads(path.read_text())
    if any(sha256(path.parent / name) != digest for name, digest in receipt["files"].items()):
        raise ValueError("Collected reference or candidate evidence changed")
    return receipt


def source_record(path, label, *, reference=False):
    receipt = checked_collection(path)
    stage = "hwa__" + ("cdt_random_3pct" if reference else ARM)
    result_path = path.parent / stage / "result.json"
    result = json.loads(result_path.read_text())
    lc = parse_lifecycle(json.loads(result_path.with_name("config.resolved.json").read_text())["lifecycle"])
    arm = lc.hwa.arm(stage.removeprefix("hwa__"))
    validate_source_fit(lc, arm, result["metrics"]["fit"])
    if not reference and lc.digest() != lifecycle(label).digest():
        raise ValueError("Unexpected candidate lifecycle")
    if reference:
        old, new = lc.to_dict(), lifecycle("1em5").to_dict()
        for section in ("network", "data", "devices", "deployment", "program_verify"):
            if old[section] != new[section]:
                raise ValueError("Reused control differs in " + section)
        if {k: v for k, v in old["hwa"].items() if k != "arms"} != {k: v for k, v in new["hwa"].items() if k != "arms"}:
            raise ValueError("Reused control has a different training protocol")
        def intervention_free(a):
            return {k: v for k, v in a.items() if k not in {"id", "scope", "digital_learning_rate"}}
        original_arm = next(a for a in old["hwa"]["arms"] if a["id"] == "cdt_random_3pct")
        if intervention_free(original_arm) != intervention_free(new["hwa"]["arms"][0]):
            raise ValueError("Reused control differs beyond the digital-update intervention")
    fit = result["metrics"]["fit"]
    selected = next(r for r in fit["history"] if r["epoch"] == fit["selected_epoch"])
    artifact = next(a for a in result["artifacts"] if a["kind"] == "hwa_source")
    return {"label": label, "digital_lr": 0.0 if reference else RATES[label], "job": receipt["job"], "target": receipt["target"],
        "attempt": receipt["attempt"], "receipt": str(path), "receipt_sha256": sha256(path), "result_sha256": sha256(result_path),
        "native_result": str(Path(receipt["stages"][stage]) / "result.json"),
        "source_path": str(Path(receipt["stages"][stage]) / artifact["path"]), "source_sha256": artifact["sha256"],
        "mean_kl": fit["development_kl"], "cases": selected["cases"], "selected_step": selected["global_step"],
        "clean_kl": result["metrics"]["clean_development"]["teacher_kl"], "convergence_review_required": fit["convergence_review_required"],
        "reused_reference": reference}


def choose(rows):
    if len(rows) != 3 or {r["label"] for r in rows} != {"0", *RATES}:
        raise ValueError("Selection requires exactly the frozen three-case grid")
    if any(not math.isfinite(r["mean_kl"]) or r["mean_kl"] < 0 for r in rows):
        raise ValueError("Invalid development selection metric")
    best = min(r["mean_kl"] for r in rows)
    return min((r for r in rows if r["mean_kl"] <= 1.01 * best), key=lambda r: r["digital_lr"])


def selection(evidence):
    refs = json.loads(REFERENCES.read_text())
    ref = refs["cdt_zero"]
    if sha256(Path(ref["receipt"])) != ref["receipt_sha256"]:
        raise ValueError("Frozen zero control receipt changed")
    rows = [source_record(Path(ref["receipt"]), "0", reference=True)]
    if rows[0]["source_sha256"] != ref["source_sha256"] or rows[0]["result_sha256"] != ref["result_sha256"]:
        raise ValueError("Frozen zero source changed")
    index = evidence / "collected/source/index.json"
    for entry in json.loads(index.read_text()) if index.exists() else []:
        for label in RATES:
            if entry["job"] == lifecycle(label).lifecycle_id:
                rows.append(source_record(Path(entry["receipt"]), label))
    rows.sort(key=lambda r: r["digital_lr"])
    if len(rows) < 3:
        return {"coverage_complete": False, "cases": rows, "test_data_used": False}
    selected = choose(rows)
    return {"coverage_complete": True, "cases": rows, "selected_label": selected["label"], "selected_source": selected,
        "rule": "minimum joint development KL; within 1% prefer smaller digital LR", "test_data_used": False}


def service(queue_config, *args):
    return command([sys.executable, "-m", "ebl.execution.service", "--queue-config", str(queue_config), *map(str, args)], cwd=ROOT)


def control(queue_config, action, job):
    return command([sys.executable, str(QUEUE), "--config", str(queue_config), action, job], cwd=ROOT)


def gate(queue_config, evidence, name, payload):
    from ebl.execution.service import digest
    manifest = json.loads((evidence / "prepared/queue/campaign.json").read_text())
    doc = {"manifest_sha256": digest(manifest), "passed": True, **payload}
    path = evidence / "readiness" / (name + ".json")
    write_frozen(path, doc)
    service(queue_config, "gate", PHASE, name, path)


def transition(queue_config, action, job):
    import sqlite3
    cfg = json.loads(queue_config.read_text())
    db = Path(cfg.get("state_dir", "~/.local/state/run-watch/queue")).expanduser() / "queue.sqlite3"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
    if row is None:
        raise ValueError("Missing predeclared job")
    if row[0] == "held":
        control(queue_config, action, job)
    elif action == "cancel" and row[0] != "cancelled":
        raise ValueError("An unselected deployment unexpectedly left its hold")


def advance(queue_config, evidence, job, collected_path):
    # One bounded completion hook owns this transition at a time. Queue admission
    # still belongs exclusively to the persistent service and its durable ledger.
    with (evidence / ".advance.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if job.endswith("-smoke"):
            receipt = checked_collection(collected_path / "collection.json")
            result = json.loads((collected_path / ("hwa__" + ARM) / "result.json").read_text())
            validate_source_fit(lifecycle("1em5", "smoke"), lifecycle("1em5", "smoke").hwa.arm(ARM), result["metrics"]["fit"])
            preflight = evidence / "readiness/preflight.json"
            if json.loads(preflight.read_text())["passed"] is not True:
                raise ValueError("Preflight did not pass")
            gate(queue_config, evidence, "handoff", {"evidence": [str(collected_path / "collection.json"), str(preflight)],
                "canary_job": job, "canary_attempt": receipt["attempt"], "canary_source_sha256": next(a["sha256"] for a in result["artifacts"] if a["kind"] == "hwa_source")})
            for label in RATES:
                transition(queue_config, "release", lifecycle(label).lifecycle_id)
            return {"canary_passed": True, "full_runs_released": True}
        chosen = selection(evidence)
        atomic_write_json(evidence / "collected/development-review.json", chosen)
        if not chosen["coverage_complete"]:
            return chosen
        path = evidence / "selection.json"
        write_frozen(path, chosen)
        label = chosen["selected_label"]
        if label != "0":
            prepared = json.loads((evidence / "prepared" / (HOSTS[label] + ".json")).read_text())
            cfg = json.loads(queue_config.read_text()); target = cfg["targets"][HOSTS[label]]
            remote_path = str(Path(prepared["source"]) / "results/exp1f-selection.json")
            code = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); s=sys.stdin.read(); p.parent.mkdir(parents=True,exist_ok=True); assert not p.exists() or p.read_text()==s; p.write_text(s) if not p.exists() else None"
            remote = shlex.join(["python3", "-c", code, remote_path])
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", *target.get("ssh_options", []), target["host"], remote],
                           input=path.read_text(), text=True, capture_output=True, check=True, timeout=60)
            gate(queue_config, evidence, "selection_" + label, {"evidence": [str(path)], "selection_sha256": sha256(path),
                "selected_source_sha256": chosen["selected_source"]["source_sha256"], "test_data_used": False})
        for candidate in RATES:
            transition(queue_config, "release" if candidate == label else "cancel", lifecycle(candidate, "transfer").lifecycle_id)
        return transfer_review(evidence, chosen)


def transfer_review(evidence, chosen):
    refs = json.loads(REFERENCES.read_text())
    reference_rows = {}
    for name in ("cosine_deployments", "standard_transfer"):
        ref = refs[name]
        if sha256(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("Frozen transfer reference changed")
        reference_rows[name] = json.loads(Path(ref["path"]).read_text())
    analog = [r for r in reference_rows["cosine_deployments"] if r["hwa_arm"] == "cdt_random_3pct"]
    path = evidence / "collected/transfer/deployments.json"
    rows = analog if chosen["selected_label"] == "0" else json.loads(path.read_text()) if path.exists() else []
    expected = {(s, c.label) for s in lifecycle("1em5").devices.assignment_seeds for c in lifecycle("1em5").defects.cases}
    if rows and (len(rows) != 30 or {(r["array_seed"], r["case"]) for r in rows} != expected):
        raise ValueError("Unexpected selected deployment coverage")
    comparisons = []
    if rows:
        if chosen["selected_label"] != "0":
            entries = json.loads((evidence / "collected/transfer/index.json").read_text())
            if len(entries) != 1 or entries[0]["job"] != lifecycle(chosen["selected_label"], "transfer").lifecycle_id:
                raise ValueError("Only the selected checkpoint may have transfer evidence")
            path = Path(entries[0]["receipt"])
            receipt = checked_collection(path)
            reused = json.loads((path.parent / "reused-source/result.json").read_text())
            if next(a["sha256"] for a in reused["artifacts"] if a["kind"] == "hwa_source") != chosen["selected_source"]["source_sha256"]:
                raise ValueError("Transfer checkpoint differs from development selection")
            def identities(collection):
                rec = checked_collection(collection)
                result = {}
                for name in rec["stages"]:
                    if name.startswith("deploy__"):
                        metrics = json.loads((collection.parent / name / "result.json").read_text())["metrics"]
                        for case, values in metrics["cases"].items():
                            result[metrics["array_seed"], case] = values["identity"]
                return result
            if identities(path) != identities(Path(refs["cdt_zero"]["receipt"])):
                raise ValueError("Selected source and frozen control use different physical identities")
        controls = {"cdt_analog": analog,
            "standard_3pct_small_digital": [r for r in reference_rows["standard_transfer"] if r["lifecycle_id"].endswith("-lr1em5")],
            "standard_9pct_analog": [r for r in reference_rows["cosine_deployments"] if r["hwa_arm"] == "standard_hwa_9pct"]}
        selected = {(r["array_seed"], r["case"]): r for r in rows}
        for name, rr in controls.items():
            if len(rr) != 30 or {(r["array_seed"], r["case"]) for r in rr} != expected:
                raise ValueError("Reference coverage mismatch")
            control_rows = {(r["array_seed"], r["case"]): r for r in rr}
            for case in lifecycle("1em5").defects.cases:
                for domain in ("wikitext", "generated"):
                    key = "deployed_transfer_" + domain + "_teacher_kl"
                    pairs = [{"array_seed": seed, "control": control_rows[seed, case.label][key], "selected": selected[seed, case.label][key]}
                             for seed in lifecycle("1em5").devices.assignment_seeds]
                    for p in pairs:
                        p["relative_reduction"] = 1 - p["selected"] / p["control"]
                    effect = sum(p["relative_reduction"] for p in pairs) / 5
                    wins = sum(p["selected"] < p["control"] for p in pairs)
                    comparisons.append({"control": name, "case": case.label, "domain": domain, "pairs": pairs,
                        "wins": wins, "mean_paired_relative_reduction": effect, "material_rule_passed": wins >= 4 and effect >= .1})
    report = {"coverage_complete": len(rows) == 30, "selected_label": chosen["selected_label"], "deployment_rows": len(rows),
        "reused_zero_deployment": chosen["selected_label"] == "0", "comparisons": comparisons,
        "scientific_gate": "awaiting_scientific_review", "further_training_dispatched": False}
    atomic_write_json(evidence / "collected/transfer-review.json", report)
    return report


def collect(queue_config, evidence):
    from ebl.execution.lifecycle_collect import collect as native_collect
    job = os.environ["RUN_WATCH_JOB_ID"]
    transfer = job in {lifecycle(l, "transfer").lifecycle_id for l in RATES}
    if job not in {lifecycle(l, m).lifecycle_id for l, m in definitions()}:
        raise ValueError("Completion hook received an unrelated job")
    result = native_collect(queue_config, evidence / "collected" / ("transfer" if transfer else "source"),
                            deployment_only=transfer, source_only=not transfer)
    if transfer:
        result["review"] = transfer_review(evidence, json.loads((evidence / "selection.json").read_text()))
    else:
        result["advance"] = advance(queue_config, evidence, job, Path(result["collected"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    p = sub.add_parser("prepare-host"); p.add_argument("--host", required=True, choices=set(HOSTS.values())); p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    q = sub.add_parser("queue"); q.add_argument("receipts", nargs="+", type=Path); q.add_argument("--output", type=Path, required=True); q.add_argument("--controller", type=Path, required=True); q.add_argument("--evidence", type=Path, required=True)
    c = sub.add_parser("collect"); c.add_argument("--queue-config", type=Path, required=True); c.add_argument("--evidence", type=Path, required=True)
    d = sub.add_parser("execute-deployment"); d.add_argument("--label", choices=RATES, required=True); d.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate": result = generate()
    elif args.command == "prepare-host": result = prepare_host(args.host, args.base)
    elif args.command == "queue": result = queue(args.receipts, args.output, args.controller, args.evidence)
    elif args.command == "execute-deployment": result = execute_deployment(args.label, args.selection)
    else: result = collect(args.queue_config, args.evidence)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
