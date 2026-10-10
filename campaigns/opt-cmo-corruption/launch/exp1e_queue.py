"""Write held run-watch job specifications from exact host preparation receipts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1e_development import NOTE, RATES, SCHEDULES, case_id as development_case_id
from slice_campaign import write_frozen


def prepare(receipts, output):
    output = output.expanduser().resolve()
    if len({path.stem for path in receipts}) != len(receipts):
        raise ValueError("Each prepared host receipt must occur exactly once")
    hosts = {path.stem: json.loads(path.read_text()) for path in receipts}
    if set(hosts) - {"fifi", "loulou", "riri", "trex"} or not {"fifi", "loulou"} <= set(hosts):
        raise ValueError("Expected explicitly prepared authorized RTX 5090 hosts")
    if any(host.get("schema") != "exp1e.host_preparation.v1" for host in hosts.values()):
        raise ValueError("Expected native host preparation receipts")
    revisions = {host["source_revision"] for host in hosts.values()}
    if (len(revisions) != 1 or not all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in revisions)
            or any(host.get("launched") is not False for host in hosts.values())):
        raise ValueError("Placements must share one clean prepared source")
    cases = {name: {case["case_id"]: case for case in host["cases"]} for name, host in hosts.items()}
    expected = {development_case_id(rate, schedule) for rate in RATES for schedule in SCHEDULES}
    expected.update(development_case_id(1e-4, schedule) + "-smoke" for schedule in SCHEDULES)
    first = next(iter(cases.values()))
    if any(set(items) != expected or len(hosts[name]["cases"]) != len(expected) for name, items in cases.items()):
        raise ValueError("Expected the complete development grid and both readiness smokes on every placement")
    for host_cases in cases.values():
        for case_id, case in host_cases.items():
            smoke = case_id.endswith("-smoke")
            if (case.get("smoke") is not smoke or case.get("dry_run", {}).get("passed") is not True
                    or case.get("timeout_seconds") != (1800 if smoke else 5400) or case.get("lease_seconds") != 900):
                raise ValueError("Expected matching smoke, budget, lease and successful preparation gates")
            for key in ("config_sha256", "lifecycle_sha256", "manifest_sha256", "worker_spec_sha256"):
                if not isinstance(case.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", case[key]):
                    raise ValueError(f"Missing or invalid frozen preparation digest: {key}")
            for path_key, hash_key in (("config", "config_sha256"), ("manifest", "manifest_sha256"),
                                       ("campaign_worker_spec", "worker_spec_sha256")):
                if {"path": case[path_key], "sha256": case[hash_key]} not in case["prepared_files"]:
                    raise ValueError("Admission must recheck every frozen config, manifest and worker spec")
    queue_config = Path.home() / ".config/run-watch/queue.json"
    evidence = ROOT / "results/lifecycles/opt125m-cmo-l6-exp1e-development"
    collector = ROOT / "campaigns/opt-cmo-corruption/launch/exp1e_collect.py"
    campaign_cases, jobs, files = [], [], []
    for case_id, case in sorted(first.items()):
        smoke = case["smoke"]
        budget = 1800 if smoke else 5400
        placements = {}
        for name, host_cases in cases.items():
            other = host_cases[case_id]
            if (other["lifecycle_sha256"] != case["lifecycle_sha256"]
                    or other["config_sha256"] != case["config_sha256"]):
                raise ValueError("Scientific configuration differs between hosts")
            # The two real readiness smokes establish both initial execution hosts.
            if smoke and name != ("fifi" if "-linear-" in case_id else "loulou"):
                continue
            placements[name] = {"workdir": other["worktree"], "run_dir": other["queue_run_dir"],
                "argv": other["argv"], "validate_argv": [other["python"],
                    str(Path(other["worktree"]) / "campaigns/opt-cmo-corruption/launch/exp1e_collect.py"),
                    "validate", other["campaign_result_root"]],
                "prepared_files": other["prepared_files"], "gpu_models": ["5090"],
                "campaign_python": other["python"], "campaign_worker_spec": other["campaign_worker_spec"],
                "campaign_result_root": other["campaign_result_root"]}
        if not placements:
            raise ValueError("Both fifi and loulou must be prepared for readiness")
        rate_priority = 30 if "lr1em4" in case_id else 20 if "lr1em5" in case_id else 10
        job = {"id": case_id, "project_dir": str(ROOT), "handoff": NOTE,
            "authorization": "User authorized focused standard 3% whole-transformer synthetic-text LR/decay development, all idle weekend GPUs; broader runs held for scientific review.",
            "peak_memory_mib": 30000, "headroom_mib": 1000,
            "expected_seconds": 1200 if smoke else 5400, "budget_seconds": budget,
            "priority": 100 if smoke else rate_priority + int("-linear-" in case_id),
            "startup_grace_seconds": 600, "stale_seconds": 900,
            "recovery": False, "retry_safe": False, "placements": placements,
            "completion": {"argv": [sys.executable, str(collector), "collect", "--queue-config", str(queue_config),
                "--output", str(evidence / "collected")], "workdir": str(ROOT),
                "timeout_seconds": 600, "retry_safe": True}}
        path = output / "jobs" / (case_id + ".json")
        files.append((path, job))
        jobs.append(str(path))
        campaign_cases.append({"id": case_id, "stage": "readiness" if smoke else "training",
            "dependencies": [], "required_gates": [] if smoke else ["handoff"],
            "expected_gpu_seconds": job["expected_seconds"], "expected_storage_bytes": 1500000000,
            "progress": {"startup_seconds": 600, "stale_seconds": 900,
                         "expected_seconds": job["expected_seconds"]}})
    manifest = {"version": 1, "id": "opt-cmo-exp1e-development-20261010", "poll_seconds": 300,
        "required_adapters": ["tmux"],
        "budgets": {"gpu_hours": 16, "storage_bytes": 30000000000, "window_seconds": 43200},
        "storage_roots": [{"target": name, "path": host["source"]} for name, host in hosts.items()] +
            [{"target": name, "path": str(Path(host["base"]) / "worker-receipts" / host["source_revision"])}
             for name, host in hosts.items()],
        "cases": campaign_cases}
    # Preparation is immutable and performs no submissions. Validate everything
    # before writing so a bad later host cannot leave a partly generated queue.
    from campaigns.execution_service import validate_manifest
    validate_manifest(manifest)
    for path, job in files:
        write_frozen(path, job)
    write_frozen(output / "campaign.json", manifest)
    return {"manifest": str(output / "campaign.json"), "jobs": jobs, "submitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("receipts", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(prepare(args.receipts, args.output), indent=2))


if __name__ == "__main__":
    main()
