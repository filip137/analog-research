"""Real tiny native lifecycle slices exercise lineage and collection gates."""
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from campaigns import lifecycle_collect as collector
from campaigns.runner import _campaign_contract_record
from campaigns.schema import CampaignSpec
from experiments.artifacts import sha256_file
from test_workflow_cmo import cmo_inputs, model_arm, small
from workflow.lifecycle import stage_selections, stage_inputs, StageConfig, Execution
from workflow.runtime import validate_source_fit
from workflow_helpers import run_stage

ROOT = Path(__file__).resolve().parents[1]
_CACHE = None


@pytest.fixture
def native(cmo_inputs, tmp_path):
    weights, corpus = cmo_inputs
    lc = small(**{"data.corpus": {"name": "synthetic", "sha256": corpus}, "data.sequence_length": 9,
        "data.evaluation": "development", "data.test_sequences": 0, "hwa.arms": [model_arm()],
        "hwa.batch_size": 4, "hwa.lr_schedule": {"kind": "linear_warmup_decay", "total_steps": 2, "warmup_steps": 1},
        "onchip": None, "transfer_tests": [
            {"label": label, "corpus": {"name": "synthetic", "sha256": corpus}, "split": "test", "sequences": count}
            for label, count in (("wikitext", 3), ("generated", 2))]})
    root = tmp_path / "campaign"
    plan = tmp_path / "plan"
    plan.mkdir()
    results, records, stages = {}, [], []
    for selection in stage_selections(lc):
        results[selection.stage_id] = run_stage(lc, selection, results, weights, root)
        run, result = results[selection.stage_id]
        config = plan / (selection.stage_id + ".json")
        config.write_text(json.dumps(StageConfig(selection, Execution("cpu", 1, False), lc).to_dict()))
        inputs, dependencies = {}, []
        for role, upstream in stage_inputs(selection).items():
            inputs[role] = {"path": str(weights)} if upstream is None else {"stage": upstream[0], "artifact_kind": upstream[1]}
            if upstream is not None:
                dependencies.append(upstream[0])
        stages.append({"id": selection.stage_id, "case_id": selection.case_id, "target": "local", "command": "train",
                       "config": str(config), "inputs": inputs, "depends_on": sorted(set(dependencies))})
        records.append({"stage_id": selection.stage_id, "status": "complete", "run_result": str(run / "result.json"),
                        "run_result_sha256": sha256_file(run / "result.json")})
    manifest = {"schema_version": 1, "campaign_id": "campaign",
                "targets": [{"id": "local", "worktree": str(ROOT), "python": sys.executable}], "stages": stages}
    manifest_path = plan / "campaign.json"
    manifest_path.write_text(json.dumps(manifest))
    contract = _campaign_contract_record(CampaignSpec.parse(manifest, base_dir=plan), manifest_path=manifest_path)
    (root / "campaign.resolved.json").write_text(json.dumps(contract))
    (root / "aggregate").mkdir()
    (root / "aggregate/results.json").write_text(json.dumps({"stages": records}))
    return root, lc


@pytest.fixture
def exported(request):
    global _CACHE
    if _CACHE is None:
        root, lc = request.getfixturevalue("native")
        _CACHE = root, lc, collector.export(root)
    return _CACHE


def test_real_native_full_slice_export_and_semantic_progress(exported):
    root, lc, payload = exported
    assert collector.checked_payload(payload, root) == payload
    rows = json.loads(payload["deployments_json"])
    assert len(rows) == len(lc.devices.assignment_seeds) * len(lc.defects.cases)
    assert all("deployed_transfer_wikitext_teacher_kl" in row and "deployed_transfer_generated_teacher_kl" in row for row in rows)
    for stage_id, stage in payload["stages"].items():
        if stage_id.startswith("deploy__"):
            metrics = [json.loads(line) for line in stage["files"]["metrics.jsonl"].splitlines()]
            assert metrics[0]["stage"] == "deploy_source_gate" and metrics[0]["passed"]
            assert [item["cases_complete"] for item in metrics if item["stage"] == "deploy" and "case" in item] == list(range(1, len(lc.defects.cases) + 1))


def replace_result(payload, stage_id, change):
    files = payload["stages"][stage_id]["files"]
    result = json.loads(files["result.json"])
    change(result["metrics"])
    files["result.json"] = json.dumps(result)
    digest = collector.text_sha(files["result.json"])
    aggregate = json.loads(payload["aggregate"])
    next(row for row in aggregate["stages"] if row["stage_id"] == stage_id)["run_result_sha256"] = digest
    payload["aggregate"] = json.dumps(aggregate)
    inspection = json.loads(files["inspection.json"])
    inspection[0]["result_sha256"] = digest
    files["inspection.json"] = json.dumps(inspection)


@pytest.mark.parametrize("failure", ["missing_assignment", "duplicate_stage", "source_steps", "selection", "nonfinite",
    "cohort", "domain", "case", "upstream", "inspection", "config", "contract", "rows", "csv"])
def test_compact_export_rejects_invalid_lineage_or_science(exported, failure):
    root, lc, original = exported
    payload = copy.deepcopy(original)
    source = next(name for name in payload["stages"] if name.startswith("hwa__"))
    deploy = next(name for name in payload["stages"] if name.startswith("deploy__"))
    if failure in {"missing_assignment", "duplicate_stage"}:
        aggregate = json.loads(payload["aggregate"])
        if failure == "missing_assignment":
            aggregate["stages"] = [row for row in aggregate["stages"] if row["stage_id"] != deploy]
        else:
            aggregate["stages"].append(aggregate["stages"][0])
        payload["aggregate"] = json.dumps(aggregate)
    elif failure == "source_steps":
        replace_result(payload, source, lambda metrics: metrics["fit"]["lr_schedule"].update(steps_run=1))
    elif failure == "selection":
        replace_result(payload, source, lambda metrics: metrics["fit"].update(development_kl=123))
    elif failure == "nonfinite":
        replace_result(payload, deploy, lambda metrics: metrics["clean"]["development"].update(teacher_kl=float("nan")))
    elif failure == "cohort":
        replace_result(payload, "prepare", lambda metrics: metrics["cohorts"]["transfer_generated"].update(count=99))
    elif failure == "domain":
        replace_result(payload, deploy, lambda metrics: metrics["clean"].pop("transfer_wikitext"))
    elif failure == "case":
        replace_result(payload, deploy, lambda metrics: metrics["cases"].pop(next(iter(metrics["cases"]))))
    elif failure == "upstream":
        files = payload["stages"][deploy]["files"]
        manifest = json.loads(files["manifest.json"])
        next(item for item in manifest["inputs"] if item["role"] == "weights")["sha256"] = "0" * 64
        files["manifest.json"] = json.dumps(manifest)
    elif failure == "inspection":
        files = payload["stages"][deploy]["files"]
        inspection = json.loads(files["inspection.json"])
        inspection[0]["validation_mode"] = "metadata_only"
        files["inspection.json"] = json.dumps(inspection)
    elif failure == "config":
        payload["stages"][source]["files"]["stage-config.json"] = "{}"
    elif failure == "contract":
        contract = json.loads(payload["campaign_resolved"])
        contract["campaign_id"] = "changed"
        payload["campaign_resolved"] = json.dumps(contract)
    elif failure == "csv":
        payload["deployments_csv"] = "changed\n"
    else:
        rows = json.loads(payload["deployments_json"])
        rows[0]["deployed_teacher_kl"] = 999
        payload["deployments_json"] = json.dumps(rows)
    with pytest.raises(ValueError):
        collector.checked_payload(payload, root)


def test_full_hash_inspection_detects_corrupt_source(native):
    root, _ = native
    records = json.loads((root / "aggregate/results.json").read_text())["stages"]
    source = next(row for row in records if row["stage_id"].startswith("hwa__"))
    result_path = Path(source["run_result"])
    artifact = next(item for item in json.loads(result_path.read_text())["artifacts"] if item["kind"] == "hwa_source")
    (result_path.parent / artifact["path"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        collector.export(root)


def test_source_completion_gate_accepts_pretrained_and_rejects_bad_fit(exported):
    _, lc, payload = exported
    arm = lc.hwa.arms[0]
    fit = json.loads(payload["stages"][f"hwa__{arm.arm_id}"]["files"]["result.json"])["metrics"]["fit"]
    validate_source_fit(lc, arm, fit)
    for key, value in (("status", "rejected_nonfinite"), ("epochs_run", 0), ("selected_epoch", 999)):
        changed = copy.deepcopy(fit)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_source_fit(lc, arm, changed)
    from dataclasses import replace
    pretrained = replace(arm, method="none")
    validate_source_fit(lc, pretrained, {"kind": "digital_teacher", "method": "none", "status": "complete"})


@pytest.mark.parametrize("transport", ["local", "ssh"])
def test_controller_collection_is_idempotent_and_binds_attempt(exported, tmp_path, monkeypatch, transport):
    root, _, payload = exported
    state = tmp_path / "state"
    state.mkdir()
    database = state / "queue.sqlite3"
    spec = {"placements": {"host": {"workdir": str(ROOT), "campaign_python": sys.executable, "campaign_result_root": str(root)}}}
    with sqlite3.connect(database) as db:
        db.executescript("CREATE TABLE jobs(id TEXT,spec TEXT,state TEXT); CREATE TABLE attempts(token TEXT,job_id TEXT,state TEXT,target TEXT);")
        db.execute("INSERT INTO jobs VALUES(?,?,?)", ("job", json.dumps(spec), "collecting"))
        db.execute("INSERT INTO attempts VALUES(?,?,?,?)", ("attempt", "job", "complete", "host"))
    config = tmp_path / "queue.json"
    config.write_text(json.dumps({"state_dir": str(state), "targets": {"host": {"transport": transport, "host": "host"}}}))
    monkeypatch.setenv("RUN_WATCH_JOB_ID", "job")
    calls = []
    def call(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
    monkeypatch.setattr(collector.subprocess, "run", call)
    output = tmp_path / "collected"
    result = collector.collect(config, output)
    assert collector.collect(config, output) == result
    receipt = json.loads((Path(result["collected"]) / "collection.json").read_text())
    assert all(sha256_file(Path(result["collected"]) / name) == digest for name, digest in receipt["files"].items())
    assert len(json.loads((output / "index.json").read_text())) == 1
    assert calls[0][0] == (sys.executable if transport == "local" else "ssh")
    with sqlite3.connect(database) as db:
        db.execute("INSERT INTO attempts VALUES(?,?,?,?)", ("other", "job", "running", "host"))
    with pytest.raises(ValueError, match="one exact"):
        collector.collect(config, output)
