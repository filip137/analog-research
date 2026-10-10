"""Validate exact native stages and collect compact focused-HWA evidence.

This source-only experiment produces no deployment/on-chip rows for workflow
collect. Native inspections and hashed control files are its collection record.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.artifacts import atomic_write_json
from workflow.lifecycle import parse_lifecycle
from exp1e_development import ARM, finite_metrics, lifecycle as development_lifecycle, summarize

STAGES = {"prepare": "feature_cache", "devices": "device_bundle", f"hwa__{ARM}": "hwa_source"}
CONTROL_FILES = ("result.json", "config.resolved.json", "metrics.jsonl", "manifest.json", "status.json")
EXPORT_FILES = set(CONTROL_FILES) | {"inspection.json", "artifact-inspection.json"}
ACTIVE = {"claimed", "starting", "running", "unknown"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def aggregate_records(raw):
    records = raw["stages"]
    if len(records) != 3 or {record["stage_id"] for record in records} != set(STAGES):
        raise ValueError("Expected the exact three-stage focused training slice")
    if any(record["status"] != "complete" for record in records):
        raise ValueError("Incomplete native campaign stage")
    return {record["stage_id"]: record for record in records}


def check_stage(stage_id, result, resolved):
    lc = parse_lifecycle(resolved["lifecycle"])
    selection = {"kind": "hwa" if stage_id.startswith("hwa__") else stage_id,
                 "hwa_arm": ARM if stage_id.startswith("hwa__") else None, "array_seed": None}
    metrics = result["metrics"]
    if (result.get("status") != "complete" or resolved["stage"] != selection
            or metrics.get("stage") != selection or metrics.get("stage_id") != stage_id
            or metrics.get("lifecycle_sha256") != lc.digest()):
        raise ValueError("Native stage identity or lifecycle mismatch")
    if lc.data.evaluation != "development" or lc.data.test_sequences != 0 or lc.transfer_tests is not None:
        raise ValueError("Focused stages must hold out every test cohort")
    if lc.lifecycle_id.endswith("-smoke"):
        canonical_smokes = {development_lifecycle(1e-4, schedule, smoke=True).digest() for schedule in ("linear", "step")}
        if lc.digest() not in canonical_smokes:
            raise ValueError("Readiness lifecycle is not an exact declared 512-update smoke")
    finite_metrics(metrics, stage_id)
    artifacts = [item for item in result["artifacts"] if item["kind"] == STAGES[stage_id]]
    if len(artifacts) != 1:
        raise ValueError(f"Expected exactly one {STAGES[stage_id]} artifact")
    if stage_id == "prepare":
        score = metrics.get("digital", {}).get("teacher_kl")
        if type(score) not in (int, float) or not math.isfinite(score) or score > 1e-6:
            raise ValueError("Prepare teacher-equivalence gate failed")
        corpus = metrics.get("corpus", {})
        if corpus.get("name") != lc.data.corpus or corpus.get("sha256") != lc.data.corpus_sha256:
            raise ValueError("Prepare corpus receipt differs from the lifecycle")
        cohorts = metrics.get("cohorts", {})
        if set(cohorts) != {"hwa", "training", "development"}:
            raise ValueError("Unexpected prepare cohort coverage")
        for name, cohort in cohorts.items():
            if cohort["count"] != lc.data.cohort_size(name) or cohort["sequence_length"] != lc.data.sequence_length:
                raise ValueError("Prepare cohort size or sequence length mismatch")
    elif stage_id == "devices":
        if metrics.get("technology") != "cmo":
            raise ValueError("Expected the CMO analytic device bundle")
    else:
        fit, schedule = metrics["fit"], lc.hwa.lr_schedule
        if metrics.get("test_used_for_selection") is not False or schedule is None:
            raise ValueError("Missing no-test-selection receipt or declared schedule")
        if fit["status"] not in {"complete", "rejected_nonfinite"}:
            raise ValueError("Unsupported scientific fit status")
        if lc.lifecycle_id.endswith("-smoke") and fit["status"] != "complete":
            raise ValueError("A readiness smoke must complete numerically")
        recorded = fit["lr_schedule"]
        if any(recorded.get(key) != value for key, value in schedule.to_dict().items()):
            raise ValueError("Fit schedule differs from its declared lifecycle")
        epochs = [0] + [e for e in range(1, lc.hwa.epochs + 1) if e % lc.hwa.selection_every == 0 or e == lc.hwa.epochs]
        history = fit["history"]
        if not history or [item["epoch"] for item in history] != epochs[:len(history)]:
            raise ValueError("Unexpected checkpoint-selection cadence")
        steps = recorded["steps_run"]
        if type(steps) is not int or not 0 <= steps <= schedule.total_steps:
            raise ValueError("Invalid completed-update count")
        bases = [lc.hwa.learning_rate, lc.hwa.arm(ARM).digital_learning_rate]
        if recorded.get("base_learning_rates") != bases or recorded.get("final_learning_rates") != [
            rate * schedule.factor(steps) for rate in bases
        ]:
            raise ValueError("Optimizer-group learning rates disagree with the declared schedule")
        if fit["status"] == "complete" and (steps != schedule.total_steps or fit["epochs_run"] != lc.hwa.epochs
                                             or len(history) != len(epochs) or fit.get("rejection") is not None):
            raise ValueError("Completed fit has incomplete update/history coverage")
        expected_cases = [case.label for case in lc.hwa.arm(ARM).selection_cases]
        for item in history:
            if item["global_step"] != item["epoch"] * (schedule.total_steps // lc.hwa.epochs) or item["global_step"] > steps:
                raise ValueError("Selection checkpoint/update mismatch")
            if [case["case"] for case in item["cases"]] != expected_cases:
                raise ValueError("Unexpected development selection cases")
            mean = sum(case["teacher_kl"] for case in item["cases"]) / len(expected_cases)
            if not math.isclose(mean, item["teacher_kl"], rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Invalid checkpoint score aggregation")
        selected = min(history, key=lambda item: item["teacher_kl"])
        if selected["epoch"] != fit["selected_epoch"] or selected["teacher_kl"] != fit["development_kl"]:
            raise ValueError("Invalid selected checkpoint")
    return lc, artifacts[0]


def checked_payload(payload, expected_root):
    """Check the exact exported identities before materializing any file."""
    if payload.get("schema") != "exp1e.compact_export.v1" or payload.get("campaign_root") != str(expected_root):
        raise ValueError("Compact export campaign identity mismatch")
    records = aggregate_records(json.loads(payload["aggregate"]))
    if set(payload["stages"]) != set(STAGES):
        raise ValueError("Unexpected exported stage coverage")
    identities, artifacts, manifests = set(), {}, {}
    for stage_id, stage in payload["stages"].items():
        run, files = Path(stage["native_run"]), stage["files"]
        if not run.is_absolute() or not run.is_relative_to(Path(expected_root)) or set(files) != EXPORT_FILES:
            raise ValueError("Unexpected native run location or exported control files")
        record = records[stage_id]
        if str(run / "result.json") != record["run_result"] or text_sha(files["result.json"]) != record["run_result_sha256"]:
            raise ValueError("Exported result does not match the native aggregate digest")
        result, resolved = json.loads(files["result.json"]), json.loads(files["config.resolved.json"])
        inspection = json.loads(files["inspection.json"])
        if (len(inspection) != 1 or inspection[0].get("valid") is not True
                or inspection[0].get("process_complete") is not True
                or inspection[0].get("validation_mode") != "full_artifact_hashes"
                or inspection[0].get("path") != str(run)):
            raise ValueError("Native full artifact inspection did not pass")
        receipt = json.loads(files["artifact-inspection.json"])
        if (receipt.get("schema") != "exp1e.artifact_inspection.v1"
                or receipt.get("result_sha256") != record["run_result_sha256"]
                or any(receipt.get(key) is not True for key in ("verified", "verify_artifacts", "require_complete"))
                or receipt.get("exit_code") != 0 or receipt.get("artifacts") != result["artifacts"]):
            raise ValueError("Artifact inspection receipt does not bind the exported result")
        lc, artifact = check_stage(stage_id, result, resolved)
        identities.add(lc.digest())
        artifacts[stage_id] = {"path": str(run / artifact["path"]), "sha256": artifact["sha256"]}
        manifests[stage_id] = json.loads(files["manifest.json"])
        for line in files["metrics.jsonl"].splitlines():
            if line.strip():
                finite_metrics(json.loads(line), f"{stage_id} metric log")
    if len(identities) != 1:
        raise ValueError("Focused stages used different lifecycle identities")
    teacher_inputs = []
    for stage_id, manifest in manifests.items():
        teacher_inputs += [item["sha256"] for item in manifest["inputs"] if item["role"] == "teacher_weights"]
    if len(teacher_inputs) != 3 or len(set(teacher_inputs)) != 1:
        raise ValueError("Focused stages used different teacher inputs")
    hwa_inputs = {item["role"]: item for item in manifests[f"hwa__{ARM}"]["inputs"]}
    for role, upstream in (("device_data", "prepare"), ("device_model", "devices")):
        if any(hwa_inputs.get(role, {}).get(key) != value for key, value in artifacts[upstream].items()):
            raise ValueError("HWA inputs do not match the exact exported upstream artifacts")
    return payload


def export(root):
    root = root.resolve()
    aggregate = root / "aggregate/results.json"
    records = aggregate_records(json.loads(aggregate.read_text()))
    exported = {}
    for stage_id, record in records.items():
        result_path = Path(record["run_result"])
        if not result_path.resolve().is_relative_to(root) or sha(result_path) != record["run_result_sha256"]:
            raise ValueError("Native result location or digest mismatch")
        run = result_path.parent
        inspected = subprocess.run([sys.executable, "-m", "ebl", "runs", "inspect", str(run),
            "--verify-artifacts", "--require-complete", "--json"], capture_output=True, text=True, timeout=180)
        if inspected.returncode:
            raise RuntimeError(inspected.stdout[-4000:] + inspected.stderr[-2000:])
        files = {name: (run / name).read_text() for name in CONTROL_FILES}
        if sha(result_path) != record["run_result_sha256"]:
            raise ValueError("Native result changed during inspection")
        result = json.loads(files["result.json"])
        files["inspection.json"] = inspected.stdout
        files["artifact-inspection.json"] = json.dumps({
            "schema": "exp1e.artifact_inspection.v1", "result_sha256": sha(result_path),
            "verified": True, "verify_artifacts": True, "require_complete": True,
            "exit_code": 0, "artifacts": result["artifacts"], "native_run": str(run),
        }, indent=2) + "\n"
        exported[stage_id] = {"native_run": str(run), "files": files}
    return checked_payload({"schema": "exp1e.compact_export.v1", "campaign_root": str(root),
                            "aggregate": aggregate.read_text(), "stages": exported}, root)


def exact_attempt(dbpath, job_id):
    with sqlite3.connect(f"file:{dbpath}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown queue job")
        job = dict(row)
        attempts = [dict(row) for row in db.execute("SELECT * FROM attempts WHERE job_id=?", (job_id,))]
    successful = [attempt for attempt in attempts if attempt["state"] == "complete"]
    if len(successful) != 1 or any(attempt["state"] in ACTIVE for attempt in attempts):
        raise ValueError("Collection requires one exact successful attempt and no uncertain workers")
    return job, successful[0]


@contextmanager
def collection_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def write_frozen(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to overwrite different collected evidence: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def collect(queue_config, output):
    job_id = os.environ["RUN_WATCH_JOB_ID"]
    cfg = json.loads(queue_config.read_text())
    dbpath = Path(cfg.get("state_dir", "~/.local/state/run-watch/queue")).expanduser() / "queue.sqlite3"
    job, attempt = exact_attempt(dbpath, job_id)
    spec = json.loads(job["spec"])
    placement = spec["placements"][attempt["target"]]
    target = cfg["targets"][attempt["target"]]
    remote = [placement["campaign_python"], str(Path(placement["workdir"]) /
              "campaigns/opt-cmo-corruption/launch/exp1e_collect.py"), "export", placement["campaign_result_root"]]
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", *target.get("ssh_options", []),
               target["host"], "cd " + shlex.quote(placement["workdir"]) + " && " + shlex.join(remote)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=True)
    if len(completed.stdout.encode()) > 20_000_000:
        raise ValueError("Compact collection unexpectedly exceeds 20 MB")
    payload = checked_payload(json.loads(completed.stdout), Path(placement["campaign_result_root"]))
    with collection_lock(output):
        _, current = exact_attempt(dbpath, job_id)
        if current["token"] != attempt["token"]:
            raise ValueError("Queue attempt identity changed during collection")
        if Path(job_id).name != job_id or Path(attempt["token"]).name != attempt["token"]:
            raise ValueError("Unsafe queue job or attempt identifier")
        destination = output / job_id / attempt["token"]
        identity = {"job": job_id, "attempt": attempt["token"], "target": attempt["target"]}
        write_frozen(output / job_id / "identity.json", json.dumps(identity, sort_keys=True) + "\n")
        for stage_id, stage in payload["stages"].items():
            for name, content in stage["files"].items():
                write_frozen(destination / stage_id / name, content)
        write_frozen(destination / "aggregate-results.json", payload["aggregate"])
        receipt = {**identity, "campaign_root": payload["campaign_root"],
                   "stages": {key: value["native_run"] for key, value in payload["stages"].items()},
                   "files": {name: sha(destination / name) for name in
                             ["aggregate-results.json"] + [f"{stage}/{name}" for stage in STAGES for name in EXPORT_FILES]}}
        write_frozen(destination / "collection.json", json.dumps(receipt, sort_keys=True) + "\n")
        entries, seen_jobs = [], set()
        for path in sorted(output.glob("*/*/collection.json")):
            receipt = json.loads(path.read_text())
            if receipt["job"] in seen_jobs:
                raise ValueError("Duplicate successful collection identity for one job")
            seen_jobs.add(receipt["job"])
            for name, digest in receipt["files"].items():
                if sha(path.parent / name) != digest:
                    raise ValueError("Collected evidence no longer matches its receipt")
            result = path.parent / f"hwa__{ARM}/result.json"
            resolved = json.loads((result.parent / "config.resolved.json").read_text())
            lifecycle = resolved["lifecycle"]
            if lifecycle["lifecycle_id"].endswith("-smoke"):
                continue
            config = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / (lifecycle["lifecycle_id"] + ".json")
            entries.append({"lifecycle": str(config), "result": str(result)})
        atomic_write_json(output / "index.json", entries)
        if entries:
            summarize(output / "index.json", output / "review")
    return {"collected": str(destination), "scientific_gate": "awaiting_scientific_review"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "validate"):
        exp = sub.add_parser(command)
        exp.add_argument("root", type=Path)
    comp = sub.add_parser("collect")
    comp.add_argument("--queue-config", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in {"export", "validate"}:
        result = export(args.root)
        if args.command == "validate":
            result = {"valid": True, "campaign_root": result["campaign_root"], "stages": sorted(result["stages"])}
    else:
        result = collect(args.queue_config, args.output)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
