"""Inspect and collect one complete native lifecycle source/deployment slice.

Only compact control files are copied to the controller. Native artifacts are
fully hashed on their execution host and stay there. A slice contains prepare,
devices, one declared HWA source, and every declared assignment for that source.
Explicit source-only collection instead requires exactly prepare, devices and
one source; it cannot silently accept a missing deployment from a full slice.
Explicit deployment-only collection requires prepare, devices and every declared
deployment, binding the external HWA source to its original fully inspected bundle.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import tempfile

from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from workflow.__main__ import _deployment_rows, _measurement_rows
from workflow.lifecycle import parse_lifecycle, stage_inputs, stage_selections, STAGE_OUTPUTS, provenance, verify_provenance
from workflow.runtime import validate_source_fit

CONTROL_FILES = ("result.json", "config.resolved.json", "metrics.jsonl", "manifest.json", "status.json")
EXPORT_FILES = set(CONTROL_FILES) | {"inspection.json", "stage-config.json"}
ACTIVE = {"claimed", "starting", "running", "unknown"}


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite native metric")
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)


def unique(items, key, label):
    result = {item[key]: item for item in items}
    if len(result) != len(items):
        raise ValueError(f"Duplicate {label}")
    return result


def rows_key(row):
    return row["lifecycle_id"], row["hwa_arm"], row["array_seed"], row["case"]


def checked_reused_source(exported, lifecycle, arm):
    """Bind a frozen external HWA bundle to a deployment-only campaign."""
    run, files = Path(exported["native_run"]), exported["files"]
    if not run.is_absolute() or set(files) != EXPORT_FILES:
        raise ValueError("Unexpected reused source location or control files")
    result, manifest = json.loads(files["result.json"]), json.loads(files["manifest.json"])
    config = json.loads(files["config.resolved.json"])
    source_lifecycle = parse_lifecycle(config["lifecycle"])
    metrics = result["metrics"]
    if (config != json.loads(files["stage-config.json"])
            or manifest["config"]["sha256"] != content_hash(config)
            or config["stage"] != {"kind": "hwa", "hwa_arm": arm, "array_seed": None}
            or metrics.get("stage") != config["stage"]
            or metrics.get("lifecycle_sha256") != source_lifecycle.digest()
            or result.get("status") != "complete" or metrics.get("test_used_for_selection") is not False):
        raise ValueError("Reused source identity or selection receipt mismatch")
    inspection = json.loads(files["inspection.json"])
    if (len(inspection) != 1 or inspection[0].get("valid") is not True
            or inspection[0].get("process_complete") is not True
            or inspection[0].get("validation_mode") != "full_artifact_hashes"
            or inspection[0].get("path") != str(run)
            or inspection[0].get("result_sha256") != text_sha(files["result.json"])):
        raise ValueError("Reused source full artifact inspection failed")
    verify_provenance(provenance(source_lifecycle, "hwa_source", hwa_arm=arm), lifecycle, "hwa_source", hwa_arm=arm)
    validate_source_fit(lifecycle, lifecycle.hwa.arm(arm), metrics["fit"])
    finite(metrics)
    for line in files["metrics.jsonl"].splitlines():
        if line.strip():
            finite(json.loads(line))
    artifact = unique(result["artifacts"], "kind", "reused artifact")["hwa_source"]
    relative = Path(artifact["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Reused source artifact escaped its native run")
    return {"path": str(run / relative), "sha256": artifact["sha256"]}, metrics


def checked_payload(payload, expected_root, *, source_only=False, deployment_only=False,
                    recovery=False, recovery_assignments=()):
    """Recheck identities, native lineage, coverage and collected values."""
    root = Path(expected_root)
    if sum((source_only, deployment_only, recovery)) > 1 or (recovery_assignments and not recovery):
        raise ValueError("Source-only, deployment-only and recovery are mutually exclusive")
    schema = "lifecycle.recovery_export.v1" if recovery else "lifecycle.deployment_export.v1" if deployment_only else "lifecycle.source_export.v1" if source_only else "lifecycle.compact_export.v1"
    if payload.get("schema") != schema or payload.get("campaign_root") != str(root):
        raise ValueError("Compact export campaign identity mismatch")
    contract = json.loads(payload["campaign_resolved"])
    source_manifest = json.loads(payload["source_manifest"])
    original = {key: value for key, value in contract.items()
                if key not in {"source_manifest", "resolved_contract_sha256"}}
    if (content_hash(original) != contract["resolved_contract_sha256"]
            or text_sha(payload["source_manifest"]) != contract["source_manifest"]["sha256"]
            or source_manifest["campaign_id"] != contract["campaign_id"]
            or root.name != contract["campaign_id"]):
        raise ValueError("Native campaign contract or source manifest mismatch")
    from campaigns.runner import _resolved_campaign_contract
    from campaigns.schema import CampaignSpec
    parsed = CampaignSpec.parse(source_manifest, base_dir=Path(contract["source_manifest"]["path"]).parent)
    if _resolved_campaign_contract(parsed) != original:
        raise ValueError("Source manifest does not resolve to the native campaign contract")
    planned = unique(contract["stages"], "id", "planned stage")
    records = unique(json.loads(payload["aggregate"])["stages"], "stage_id", "native stage")
    if set(records) != set(planned) or set(payload["stages"]) != set(planned):
        raise ValueError("Native stage coverage differs from its frozen manifest")
    if any(record.get("status") != "complete" for record in records.values()):
        raise ValueError("Incomplete native campaign stage")
    lifecycle = None
    selected_arm = None
    external_source = deployment_only or (recovery and "reused_source" in payload)
    artifacts, manifests, results = {}, {}, {}
    for stage_id, exported in payload["stages"].items():
        run, files = Path(exported["native_run"]), exported["files"]
        expected_files = EXPORT_FILES | ({"measurements.json"} if recovery and stage_id.startswith("onchip__") else set())
        if not run.is_absolute() or not run.is_relative_to(root) or set(files) != expected_files:
            raise ValueError("Unexpected native run location or control-file coverage")
        record = records[stage_id]
        if str(run / "result.json") != record["run_result"] or text_sha(files["result.json"]) != record["run_result_sha256"]:
            raise ValueError("Native result location or aggregate digest mismatch")
        resolved, config = json.loads(files["config.resolved.json"]), json.loads(files["stage-config.json"])
        result, manifest = json.loads(files["result.json"]), json.loads(files["manifest.json"])
        if resolved != config or manifest["config"]["sha256"] != content_hash(resolved):
            raise ValueError("Native resolved configuration differs from its exact planned configuration")
        lc = parse_lifecycle(resolved["lifecycle"])
        if lifecycle is None:
            lifecycle = lc
        if lc.digest() != lifecycle.digest():
            raise ValueError("Slice stages used different lifecycle identities")
        metrics, selection = result["metrics"], resolved["stage"]
        if (result.get("status") != "complete" or metrics.get("stage") != selection
                or metrics.get("stage_id") != stage_id or metrics.get("lifecycle_sha256") != lc.digest()):
            raise ValueError("Native stage identity or lifecycle mismatch")
        inspection = json.loads(files["inspection.json"])
        if (len(inspection) != 1 or inspection[0].get("valid") is not True
                or inspection[0].get("process_complete") is not True
                or inspection[0].get("validation_mode") != "full_artifact_hashes"
                or inspection[0].get("path") != str(run)
                or inspection[0].get("result_sha256") != record["run_result_sha256"]):
            raise ValueError("Native full artifact inspection did not bind the exact result")
        finite(metrics)
        for line in files["metrics.jsonl"].splitlines():
            if line.strip():
                finite(json.loads(line))
        kinds = unique(result["artifacts"], "kind", "artifact kind")
        kind = selection["kind"]
        if kind not in ({"prepare", "devices", "hwa", "deploy", "onchip"} if recovery else {"prepare", "devices", "hwa", "deploy"}):
            raise ValueError("Expected source and deployment stages only")
        for output in STAGE_OUTPUTS[kind]:
            if output not in kinds:
                raise ValueError(f"Missing native artifact {output}")
        artifacts[stage_id] = {}
        for name, item in kinds.items():
            path = run / item["path"]
            if Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts:
                raise ValueError("Native artifact escaped its run directory")
            artifacts[stage_id][name] = {"path": str(path), "sha256": item["sha256"]}
        if kind == "prepare":
            score = metrics.get("digital", {}).get("teacher_kl")
            if type(score) not in (int, float) or not math.isfinite(score) or score > 1e-6:
                raise ValueError("Prepare teacher-equivalence gate failed")
            if lc.network.family == "opt_mlp_suffix":
                cohorts = metrics["cohorts"]
                expected = {name: lc.data.cohort_size(name) for name in ("hwa", "training", "development")}
                if lc.data.evaluation == "test":
                    expected["test"] = lc.data.cohort_size("test")
                for test in (() if lc.transfer_tests is None else lc.transfer_tests.tests):
                    expected[test.cache_key] = min(test.sequences, lc.data.max_examples) if lc.data.max_examples else test.sequences
                    if cohorts.get(test.cache_key, {}).get("corpus", {}).get("sha256") != test.corpus_sha256:
                        raise ValueError("Transfer cohort corpus digest mismatch")
                tasks = {}
                if lc.task is not None:
                    for name, count in (("task_development", lc.task.development_examples),
                                        ("task_test", lc.task.evaluation_examples),
                                        *((("task_train", lc.task.training_examples),)
                                          if lc.task.training_examples is not None else ())):
                        tasks[name] = min(count, lc.data.max_examples) if lc.data.max_examples else count
                        if cohorts.get(name, {}).get("source", {}).get("sha256") != lc.task.choices_sha256:
                            raise ValueError("Task cohort file digest mismatch")
                if (set(cohorts) != set(expected) | set(tasks) or any(cohorts[name]["count"] != count
                        or cohorts[name]["sequence_length"] != lc.data.sequence_length for name, count in expected.items())
                        or any(cohorts[name]["count"] != count for name, count in tasks.items())):
                    raise ValueError("Prepared cohort coverage differs from the lifecycle")
                if metrics["corpus"]["sha256"] != lc.data.corpus_sha256:
                    raise ValueError("Prepared training corpus digest mismatch")
        elif kind == "devices":
            if metrics.get("technology") != lc.devices.technology:
                raise ValueError("Device technology mismatch")
            if lc.devices.technology == "om" and metrics.get("kernel_adequate") is not True:
                raise ValueError("Device kernel validation failed")
        elif kind == "hwa":
            if external_source:
                raise ValueError("Reused source must not also be trained in the recovery slice")
            if selected_arm is not None:
                raise ValueError("Expected exactly one source per slice")
            selected_arm = selection["hwa_arm"]
            if metrics.get("test_used_for_selection") is not False:
                raise ValueError("Missing no-test-selection receipt")
            validate_source_fit(lc, lc.hwa.arm(selected_arm), metrics["fit"])
        elif kind == "deploy":
            if external_source:
                if selected_arm not in {None, selection["hwa_arm"]}:
                    raise ValueError("Expected one reused HWA source per deployment slice")
                selected_arm = selection["hwa_arm"]
            expected_cases = {case.label: case.to_dict() for case in lc.defects.cases}
            if (set(metrics["cases"]) != set(expected_cases)
                    or metrics["array_seed"] != selection["array_seed"]):
                raise ValueError("Deployment assignment or defect coverage mismatch")
            splits = [lc.data.evaluation] + [test.cache_key for test in (() if lc.transfer_tests is None else lc.transfer_tests.tests)]
            sizes = {lc.data.evaluation: lc.data.cohort_size(lc.data.evaluation)}
            sizes.update({test.cache_key: min(test.sequences, lc.data.max_examples) if lc.data.max_examples else test.sequences
                          for test in (() if lc.transfer_tests is None else lc.transfer_tests.tests)})
            for label, case in metrics["cases"].items():
                if case.get("case") != expected_cases[label]:
                    raise ValueError("Deployment defect configuration mismatch")
                for scores in (metrics["clean"], case["initial"]):
                    for split in splits:
                        if split not in scores or not isinstance(scores[split].get("teacher_kl"), (int, float)):
                            raise ValueError("Missing declared deployment evaluation domain")
                        if lc.network.family == "opt_mlp_suffix" and scores[split].get("sequences") != sizes[split]:
                            raise ValueError("Deployment evaluation cohort size mismatch")
                    if lc.task is not None and any(
                            not isinstance(scores.get(name, {}).get("accuracy_norm_percent"), (int, float))
                            for name in ("task_development", "task_test")):
                        raise ValueError("Missing declared task evaluation")
        else:
            if lc.onchip is None or text_sha(files["measurements.json"]) != kinds["measurements"]["sha256"]:
                raise ValueError("On-chip measurements are missing or changed")
            measured = json.loads(files["measurements.json"])
            expected = {(c.label, a.arm_id) for c in lc.defects.cases for a in lc.onchip.arms}
            if (len(measured) != len(expected) or {(r["case"], r["onchip_arm"]) for r in measured} != expected
                    or metrics.get("exact_replays") != len(expected)):
                raise ValueError("On-chip trajectory coverage or replay gate failed")
            finite(measured)
            arms = {a.arm_id: a for a in lc.onchip.arms}
            for row in measured:
                arm = arms[row["onchip_arm"]]
                count = lc.data.cohort_size(lc.onchip.cohort_key)
                presentations = lc.onchip.epochs * count if arm.trains else 0
                if (row["lifecycle_id"] != lc.lifecycle_id or row["hwa_arm"] != selection["hwa_arm"]
                        or row["array_seed"] != selection["array_seed"] or row.get("replay_exact") is not True
                        or row["weights"] != arm.weights or row["calibration"] != arm.calibration
                        or row.get("sequence_presentations", row.get("image_presentations")) != presentations
                        or [r["epoch"] for r in row["curve"]] != list(range(1, (lc.onchip.epochs if arm.trains else 0) + 1))):
                    raise ValueError("On-chip identity, schedule or training cohort mismatch")
                state = kinds.get(f"onchip_state.{row['case']}.{arm.arm_id}", {})
                if state.get("sha256") != row["state_sha256"]:
                    raise ValueError("On-chip final state identity mismatch")
        manifests[stage_id], results[stage_id] = manifest, result
    if lifecycle is None or selected_arm is None:
        raise ValueError("Missing lifecycle source")
    if external_source:
        source, source_metrics = checked_reused_source(payload["reused_source"], lifecycle, selected_arm)
        artifacts["hwa__" + selected_arm] = {"hwa_source": source}
        for result in results.values():
            if any(result["metrics"][key] != source_metrics[key] for key in ("teacher_sha256", "mapping_sha256")):
                raise ValueError("Reused source teacher or mapping differs from deployment")
    elif "reused_source" in payload:
        raise ValueError("Reused source requires explicit deployment-only collection")
    kinds = {"prepare", "devices", "deploy"} if external_source else {"prepare", "devices", "hwa"} if source_only else {"prepare", "devices", "hwa", "deploy"}
    if recovery:
        kinds.add("onchip")
        selected_arrays = tuple(recovery_assignments) or lifecycle.devices.assignment_seeds
        if (len(set(selected_arrays)) != len(selected_arrays)
                or not set(selected_arrays) <= set(lifecycle.devices.assignment_seeds)
                or payload.get("recovery_assignments") != list(selected_arrays)):
            raise ValueError("Recovery assignment selection differs from its declared scope")
    expected_selections = {item.stage_id: item for item in stage_selections(lifecycle)
                           if item.kind in kinds
                           and item.hwa_arm in {None, selected_arm}
                           and (item.kind != "onchip" or item.array_seed in selected_arrays)}
    if set(records) != set(expected_selections):
        raise ValueError("Expected exact source-only coverage" if source_only else
                         "Expected the source and every declared deployment assignment")
    teacher_inputs = []
    for stage_id, selection in expected_selections.items():
        if results[stage_id]["metrics"]["stage"] != selection.to_dict():
            raise ValueError("Unexpected native stage selection")
        inputs = unique(manifests[stage_id]["inputs"], "role", "input role")
        expected_inputs = stage_inputs(selection, lc)
        if set(inputs) - {"staged_source_snapshot"} != set(expected_inputs):
            raise ValueError("Native input-role coverage mismatch")
        for role, upstream in expected_inputs.items():
            reference = planned[stage_id]["inputs"][role]
            if upstream is None:
                teacher_inputs.append(inputs[role]["sha256"])
                if reference["path"] != inputs[role]["path"]:
                    raise ValueError("Teacher input differs from the frozen manifest")
            else:
                expected = artifacts[upstream[0]][upstream[1]]
                if external_source and upstream[1] == "hwa_source":
                    if reference.get("path") != expected["path"] or any(inputs[role].get(key) != value for key, value in expected.items()):
                        raise ValueError("Frozen external source lineage mismatch")
                elif (reference["stage"] != upstream[0] or reference["artifact_kind"] != upstream[1]
                        or any(inputs[role].get(key) != value for key, value in expected.items())):
                    raise ValueError("Native upstream artifact lineage mismatch")
    if len(set(teacher_inputs)) != 1:
        raise ValueError("Slice stages used different teachers")
    expected_rows = []
    for stage_id, result in results.items():
        if result["metrics"]["stage"]["kind"] == "deploy":
            expected_rows.extend(_deployment_rows(Path(payload["stages"][stage_id]["native_run"]), result))
    rows = json.loads(payload["deployments_json"])
    if len({rows_key(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate deployment coverage")
    expected_rows = [{**row, "duplicate_coverage": False} for row in expected_rows]
    if sorted(rows, key=rows_key) != sorted(expected_rows, key=rows_key):
        raise ValueError("Native collected rows differ from their exact deployment results")
    csv_rows = list(csv.DictReader(io.StringIO(payload["deployments_csv"])))
    expected_csv = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    if csv_rows != expected_csv:
        raise ValueError("Native collected CSV differs from its validated JSON rows")
    if recovery:
        expected_recovery = []
        for stage_id, result in results.items():
            selection = result["metrics"]["stage"]
            if selection["kind"] != "onchip":
                continue
            deployed_id = f"deploy__{selected_arm}__{selection['array_seed']}"
            deployment = results[deployed_id]["metrics"]
            measured = json.loads(payload["stages"][stage_id]["files"]["measurements.json"])
            for row in measured:
                case = deployment["cases"][row["case"]]
                if (row["p0_sha256"] != artifacts[deployed_id]["deployment"]["sha256"]
                        or row["initial"] != case["initial"] or row["clean"] != deployment["clean"]
                        or row["identity"] != case["identity"]
                        or row["initial_apparent_sha256"] != case["programming"]["apparent_sha256"]):
                    raise ValueError("On-chip initial state differs from the matched deployment")
            expected_recovery.extend(_measurement_rows(measured, Path(payload["stages"][stage_id]["native_run"])))
        expected_recovery = [{**row, "duplicate_coverage": False} for row in expected_recovery]
        recovery_rows = json.loads(payload["rows_json"])
        key = lambda row: (*rows_key(row), row["onchip_arm"])
        if sorted(recovery_rows, key=key) != sorted(expected_recovery, key=key):
            raise ValueError("Collected recovery rows differ from native measurements")
        recovery_fields = sorted({key for row in recovery_rows for key in row})
        if list(csv.DictReader(io.StringIO(payload["rows_csv"]))) != [
                {key: "" if row.get(key) is None else str(row[key]) for key in recovery_fields}
                for row in recovery_rows]:
            raise ValueError("Recovery CSV differs from validated JSON rows")
    return payload


def export(root, *, source_only=False, deployment_only=False, recovery=False, recovery_assignments=()):
    root = root.resolve()
    contract_text = (root / "campaign.resolved.json").read_text()
    contract = json.loads(contract_text)
    manifest_path = Path(contract["source_manifest"]["path"])
    if sha256_file(manifest_path) != contract["source_manifest"]["sha256"]:
        raise ValueError("Frozen source manifest changed")
    aggregate = (root / "aggregate/results.json").read_text()
    records = unique(json.loads(aggregate)["stages"], "stage_id", "native stage")
    planned = unique(contract["stages"], "id", "planned stage")
    if set(records) != set(planned) or any(item.get("status") != "complete" for item in records.values()):
        raise ValueError("Expected complete exact native campaign coverage")
    stages = {}
    for stage_id, record in records.items():
        result_path = Path(record["run_result"])
        if not result_path.resolve().is_relative_to(root) or sha256_file(result_path) != record["run_result_sha256"]:
            raise ValueError("Native result location or digest mismatch")
        run = result_path.parent
        inspected = subprocess.run([sys.executable, "-m", "ebl", "runs", "inspect", str(run),
            "--verify-artifacts", "--require-complete", "--json"], capture_output=True, text=True, timeout=180)
        if inspected.returncode:
            raise RuntimeError(inspected.stdout[-4000:] + inspected.stderr[-2000:])
        files = {name: (run / name).read_text() for name in CONTROL_FILES}
        files["inspection.json"] = inspected.stdout
        files["stage-config.json"] = Path(planned[stage_id]["config"]).read_text()
        if recovery and stage_id.startswith("onchip__"):
            files["measurements.json"] = (run / "measurements.json").read_text()
        stages[stage_id] = {"native_run": str(run), "files": files}
    with tempfile.TemporaryDirectory(prefix="lifecycle-collect-") as folder:
        command = [sys.executable, "-m", "workflow", "collect"]
        command += [stage["native_run"] for name, stage in stages.items()
                    if source_only or name.startswith("deploy__") or (recovery and name.startswith("onchip__"))]
        completed = subprocess.run([*command, "--output", folder], capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(completed.stdout[-4000:] + completed.stderr[-2000:])
        deployments = (Path(folder) / "deployments.json").read_text()
        csv_path = Path(folder) / "deployments.csv"
        # Native collect emits no CSV when a declared source-only slice has no rows.
        csv = "" if source_only and not csv_path.exists() else csv_path.read_text()
        extra = {}
        if recovery:
            lc = parse_lifecycle(json.loads(next(iter(stages.values()))["files"]["config.resolved.json"])["lifecycle"])
            extra.update(rows_json=(Path(folder) / "rows.json").read_text(),
                         rows_csv=(Path(folder) / "rows.csv").read_text(),
                         recovery_assignments=list(tuple(recovery_assignments) or lc.devices.assignment_seeds))
    if deployment_only or (recovery and not any(name.startswith("hwa__") for name in planned)):
        sources = {stage["inputs"]["weights"].get("path") for name, stage in planned.items() if name.startswith("deploy__")}
        if len(sources) != 1 or None in sources:
            raise ValueError("Expected one frozen external HWA source")
        path = Path(next(iter(sources)))
        run = path.parent.parent
        inspected = subprocess.run([sys.executable, "-m", "ebl", "runs", "inspect", str(run),
            "--verify-artifacts", "--require-complete", "--json"], capture_output=True, text=True, timeout=180)
        if inspected.returncode:
            raise RuntimeError(inspected.stdout[-4000:] + inspected.stderr[-2000:])
        files = {name: (run / name).read_text() for name in CONTROL_FILES}
        files["inspection.json"] = inspected.stdout
        files["stage-config.json"] = files["config.resolved.json"]
        extra["reused_source"] = {"native_run": str(run), "files": files}
    return checked_payload({"schema": "lifecycle.recovery_export.v1" if recovery else "lifecycle.deployment_export.v1" if deployment_only else "lifecycle.source_export.v1" if source_only else "lifecycle.compact_export.v1", "campaign_root": str(root),
        "campaign_resolved": contract_text, "source_manifest": manifest_path.read_text(), "aggregate": aggregate,
        "stages": stages, "deployments_json": deployments, "deployments_csv": csv, **extra}, root,
        source_only=source_only, deployment_only=deployment_only, recovery=recovery, recovery_assignments=recovery_assignments)


def exact_attempt(dbpath, job_id):
    with sqlite3.connect(f"file:{dbpath}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown queue job")
        attempts = [dict(row) for row in db.execute("SELECT * FROM attempts WHERE job_id=?", (job_id,))]
        successful = [attempt for attempt in attempts if attempt["state"] == "complete"]
        if len(successful) != 1 or any(attempt["state"] in ACTIVE for attempt in attempts):
            raise ValueError("Collection requires one exact successful attempt and no uncertain workers")
        return dict(row), successful[0]


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


def collect(queue_config, output, *, source_only=False, deployment_only=False, recovery=False, recovery_assignments=()):
    job_id = os.environ["RUN_WATCH_JOB_ID"]
    cfg = json.loads(queue_config.read_text())
    dbpath = Path(cfg.get("state_dir", "~/.local/state/run-watch/queue")).expanduser() / "queue.sqlite3"
    job, attempt = exact_attempt(dbpath, job_id)
    placement = json.loads(job["spec"])["placements"][attempt["target"]]
    target = cfg["targets"][attempt["target"]]
    argv = [placement["campaign_python"], "-m", "campaigns.lifecycle_collect", "export", placement["campaign_result_root"]]
    if source_only:
        argv.append("--source-only")
    if deployment_only:
        argv.append("--deployment-only")
    if recovery:
        argv.append("--recovery")
        for seed in recovery_assignments:
            argv.extend(["--recovery-assignment", str(seed)])
    if target.get("transport") == "local":
        command, cwd = argv, placement["workdir"]
    elif target.get("transport") == "ssh":
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", *target.get("ssh_options", []),
                   target["host"], "cd " + shlex.quote(placement["workdir"]) + " && " + shlex.join(argv)]
        cwd = None
    else:
        raise ValueError("Unsupported collector transport; expected local or ssh")
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=600, check=True)
    if len(completed.stdout.encode()) > 64 * 1024 * 1024:
        raise ValueError("Compact collection unexpectedly exceeds 64 MB")
    payload = checked_payload(json.loads(completed.stdout), Path(placement["campaign_result_root"]),
                              source_only=source_only, deployment_only=deployment_only,
                              recovery=recovery, recovery_assignments=recovery_assignments)
    with collection_lock(output):
        _, current = exact_attempt(dbpath, job_id)
        if current["token"] != attempt["token"]:
            raise ValueError("Queue attempt identity changed during collection")
        if any(Path(value).name != value for value in (job_id, attempt["token"])):
            raise ValueError("Unsafe queue job or attempt identifier")
        destination = output / job_id / attempt["token"]
        identity = {"job": job_id, "attempt": attempt["token"], "target": attempt["target"]}
        write_frozen(output / job_id / "identity.json", json.dumps(identity, sort_keys=True) + "\n")
        files = {"aggregate-results.json": payload["aggregate"], "campaign.resolved.json": payload["campaign_resolved"],
                 "source-manifest.json": payload["source_manifest"], "deployments.json": payload["deployments_json"],
                 "deployments.csv": payload["deployments_csv"]}
        for stage_id, stage in payload["stages"].items():
            files.update({f"{stage_id}/{name}": content for name, content in stage["files"].items()})
        if recovery:
            files.update({"rows.json": payload["rows_json"], "rows.csv": payload["rows_csv"]})
        if "reused_source" in payload:
            files.update({f"reused-source/{name}": content for name, content in payload["reused_source"]["files"].items()})
            files["reused-source/origin.json"] = json.dumps({"native_run": payload["reused_source"]["native_run"]}, sort_keys=True) + "\n"
        for name, content in files.items():
            write_frozen(destination / name, content)
        receipt = {**identity, "campaign_root": payload["campaign_root"],
                   "stages": {name: stage["native_run"] for name, stage in payload["stages"].items()},
                   "files": {name: text_sha(content) for name, content in files.items()}}
        write_frozen(destination / "collection.json", json.dumps(receipt, sort_keys=True) + "\n")
        entries, rows, recovered, seen = [], [], [], set()
        for path in sorted(output.glob("*/*/collection.json")):
            receipt = json.loads(path.read_text())
            if receipt["job"] in seen:
                raise ValueError("Duplicate successful collection for one queue job")
            seen.add(receipt["job"])
            for name, digest in receipt["files"].items():
                if sha256_file(path.parent / name) != digest:
                    raise ValueError("Collected evidence changed after publication")
            entries.append({**{key: receipt[key] for key in ("job", "attempt", "target")}, "receipt": str(path)})
            rows.extend(json.loads((path.parent / "deployments.json").read_text()))
            if (path.parent / "rows.json").is_file():
                recovered.extend(json.loads((path.parent / "rows.json").read_text()))
        if len({rows_key(row) for row in rows}) != len(rows):
            raise ValueError("Duplicate deployment coverage across collected jobs")
        atomic_write_json(output / "index.json", entries)
        atomic_write_json(output / "deployments.json", sorted([row for row in rows
                          if not row["lifecycle_id"].endswith("-smoke")], key=rows_key))
        atomic_write_json(output / "readiness-deployments.json", sorted([row for row in rows
                          if row["lifecycle_id"].endswith("-smoke")], key=rows_key))
        if recovery:
            key = lambda row: (*rows_key(row), row["onchip_arm"])
            if len({key(row) for row in recovered}) != len(recovered):
                raise ValueError("Duplicate recovery coverage across collected jobs")
            atomic_write_json(output / "rows.json", sorted(recovered, key=key))
    return {"collected": str(destination), "deployment_rows": len(json.loads(payload["deployments_json"])),
            "scientific_gate": "awaiting_scientific_review"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "validate"):
        sub = commands.add_parser(name)
        sub.add_argument("root", type=Path)
        sub.add_argument("--source-only", action="store_true")
        sub.add_argument("--deployment-only", action="store_true")
        sub.add_argument("--recovery", action="store_true")
        sub.add_argument("--recovery-assignment", type=int, action="append", default=[])
    comp = commands.add_parser("collect")
    comp.add_argument("--queue-config", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    comp.add_argument("--source-only", action="store_true")
    comp.add_argument("--deployment-only", action="store_true")
    comp.add_argument("--recovery", action="store_true")
    comp.add_argument("--recovery-assignment", type=int, action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "collect":
        result = collect(args.queue_config, args.output, source_only=args.source_only, deployment_only=args.deployment_only,
                         recovery=args.recovery, recovery_assignments=args.recovery_assignment)
    else:
        result = export(args.root, source_only=args.source_only, deployment_only=args.deployment_only,
                        recovery=args.recovery, recovery_assignments=args.recovery_assignment)
        if args.command == "validate":
            result = {"valid": True, "campaign_root": result["campaign_root"],
                      "stages": sorted(result["stages"]), "deployment_rows": len(json.loads(result["deployments_json"]))}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
