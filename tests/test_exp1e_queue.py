"""The generated campaign must fit the installed run-watch job contract."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from ebl.execution.service import load_queue, validate_manifest

SCRIPT = Path(__file__).resolve().parents[1] / "campaigns/opt-cmo-corruption/launch/exp1e_queue.py"
SPEC = importlib.util.spec_from_file_location("exp1e_queue", SCRIPT)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


@pytest.fixture
def receipts(tmp_path):
    ids = [queue.development_case_id(rate, schedule) for rate in queue.RATES for schedule in queue.SCHEDULES]
    ids += [queue.development_case_id(1e-4, schedule) + "-smoke" for schedule in queue.SCHEDULES]
    revision = "a" * 40
    paths = []
    for host in ("fifi", "loulou", "riri", "trex"):
        base = f"/home/filip/analog-research/opt-cmo-exp1"
        source = base + "/hwa-" + revision
        cases = []
        for name in ids:
            root = source + "/results/lifecycles/" + name
            config = source + "/campaigns/opt-cmo-corruption/lifecycles/" + name + ".json"
            config_hash = hashlib.sha256(name.encode()).hexdigest()
            manifest, spec = root + "/campaign-training.json", root + "/tmux-worker.json"
            digest = hashlib.sha256((name + host).encode()).hexdigest()
            cases.append({"case_id": name, "smoke": name.endswith("-smoke"), "worktree": source,
                          "python": "/home/filip/miniconda3/envs/py312/bin/python", "config": config,
                          "config_sha256": config_hash, "lifecycle_sha256": config_hash,
                          "manifest": manifest, "manifest_sha256": digest, "worker_spec_sha256": digest,
                          "campaign_worker_spec": spec, "campaign_result_root": root + "/runs/slice-id",
                          "queue_run_dir": base + "/worker-receipts/" + revision + "/" + name,
                          "argv": ["/home/filip/miniconda3/envs/py312/bin/python", "-m", "ebl.execution.tmux_worker",
                                   "bridge", "--spec", spec],
                          "prepared_files": [{"path": config, "sha256": config_hash},
                                             {"path": manifest, "sha256": digest}, {"path": spec, "sha256": digest}],
                          "timeout_seconds": 1800 if name.endswith("-smoke") else 5400,
                          "lease_seconds": 900, "dry_run": {"passed": True}})
        path = tmp_path / (host + ".json")
        path.write_text(json.dumps({"schema": "exp1e.host_preparation.v1", "source_revision": revision,
                                   "source": source, "base": base, "launched": False, "cases": cases}))
        paths.append(path)
    return paths


def test_queue_matches_current_grid_actual_schema_storage_and_readonly_policy(tmp_path, receipts):
    scripts = Path.home() / ".codex/skills/run-watch/scripts"
    if not (scripts / "gpu_queue.py").exists():
        pytest.skip("installed run-watch contract required")
    installed = load_queue(scripts)
    output = tmp_path / "prepared"
    result = queue.prepare(receipts, output)
    manifest = validate_manifest(json.loads(Path(result["manifest"]).read_text()))
    assert len(manifest["cases"]) == len(result["jobs"]) == len(queue.RATES) * len(queue.SCHEDULES) + 2
    assert manifest["budgets"] == {"gpu_hours": 16, "storage_bytes": 30_000_000_000, "window_seconds": 43200}
    assert len(manifest["storage_roots"]) == 8
    for host in ("fifi", "loulou", "riri", "trex"):
        host_roots = [Path(root["path"]) for root in manifest["storage_roots"] if root["target"] == host]
        assert len(host_roots) == 2
        assert all(a not in b.parents and a != b for a, b in (host_roots, host_roots[::-1]))
    cfg = {"targets": {name: {} for name in ("fifi", "loulou", "riri", "trex")}}
    for path in result["jobs"]:
        job = installed.validate_job(json.loads(Path(path).read_text()), cfg)
        case = next(case for case in manifest["cases"] if case["id"] == job["id"])
        assert job["recovery"] is False
        assert job["retry_safe"] is False
        assert job["completion"]["retry_safe"] is True
        assert case["expected_storage_bytes"] == 1_500_000_000
        if job["id"].endswith("-smoke"):
            expected_host = "fifi" if "-linear-" in job["id"] else "loulou"
            assert set(job["placements"]) == {expected_host}
            assert case["stage"] == "readiness"
            assert case["required_gates"] == []
            assert job["budget_seconds"] == 1800
        else:
            assert set(job["placements"]) == set(cfg["targets"])
            assert case["stage"] == "training"
            assert case["required_gates"] == ["handoff"]
            assert job["budget_seconds"] == 5400
        for placement in job["placements"].values():
            assert placement["run_dir"] != placement["campaign_result_root"]
            assert placement["validate_argv"][-2] == "validate"
            assert placement["campaign_worker_spec"] in [item["path"] for item in placement["prepared_files"]]
    assert result["submitted"] is False
    assert queue.prepare(receipts, output) == result


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_smoke", "failed_dry_run", "missing_hash_guard", "unexpected_case"])
def test_queue_rejects_incomplete_or_unfrozen_receipts_before_writing(tmp_path, receipts, mutation):
    path = receipts[0]
    data = json.loads(path.read_text())
    if mutation == "missing":
        data["cases"].pop()
    elif mutation == "duplicate":
        data["cases"].append(data["cases"][0])
    elif mutation == "wrong_smoke":
        data["cases"][0]["smoke"] = True
    elif mutation == "failed_dry_run":
        data["cases"][0]["dry_run"]["passed"] = False
    elif mutation == "missing_hash_guard":
        data["cases"][0]["prepared_files"].pop()
    elif mutation == "unexpected_case":
        data["cases"][0]["case_id"] = "undeclared"
    path.write_text(json.dumps(data))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        queue.prepare(receipts, output)
    assert not output.exists()


def test_queue_rejects_duplicate_hosts_and_changed_prepared_files(tmp_path, receipts):
    with pytest.raises(ValueError, match="exactly once"):
        queue.prepare(receipts + [receipts[0]], tmp_path / "duplicate")
    output = tmp_path / "prepared"
    result = queue.prepare(receipts, output)
    changed = Path(result["jobs"][0])
    changed.write_text("{}\n")
    with pytest.raises(ValueError, match="frozen slice"):
        queue.prepare(receipts, output)
