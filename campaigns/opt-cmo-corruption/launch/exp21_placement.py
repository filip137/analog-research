"""Prepare and collect the fault-placement comparison through native lifecycle stages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import pstdev

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from slice_campaign import write_frozen
from workflow.lifecycle import parse_lifecycle, provenance, verify_provenance

PREFIX = "opt125m-cmo-l6-exp21-fault-placement"
NOTE = "campaigns/pilots/20261016-fault-placement.md"
ARRAYS = (271002, 271003, 271004)
ARM = "standard_hwa_model"
HOLD = "pair_rewrite_verify"
LEARN = "pair_rewrite_verify_recovery"
SOURCE_SHA = "86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028"
SCENARIOS = ("uniform", "array", "die", "both")


def lifecycle(scenario=None):
    path = ROOT / "campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite-std3-kinds-learn.json"
    doc = json.loads(path.read_text())
    if scenario is not None and scenario not in SCENARIOS:
        raise ValueError("Unknown placement scenario")
    doc["lifecycle_id"] = PREFIX + ("-" + scenario if scenario else "")
    doc["question"].update(
        text="At fixed fault totals, does die/array variation change verified partner rewriting or the benefit of noisy on-chip learning?",
        decision="Matched placement differences per kind/rate/domain on assignments 271002-271004; all-three sign rule and 10% rewrite-residual effect rule. " + NOTE,
    )
    cases = [{"kind": "none", "rate_ppm": 0, "die_concentration": None, "array_concentration": None}]
    for kind in ("gmin", "gmax"):
        for ppm, concentration in ((20000, 48), (40000, 95), (60000, 140)):
            for label, (die, array) in zip(SCENARIOS, ((None, None), (None, concentration), (concentration, None), (concentration, concentration))):
                if scenario is not None and label != scenario:
                    continue
                cases.append(dict(kind=kind, rate_ppm=ppm, die_concentration=die, array_concentration=array))
    doc["defects"]["cases"] = cases
    doc["onchip"].pop("case_labels", None)
    return parse_lifecycle(doc)


def generate():
    records = []
    for scenario in (None, *SCENARIOS):
        lc = lifecycle(scenario)
        path = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / f"{lc.lifecycle_id}.json"
        write_frozen(path, lc.to_dict())
        records.append({"path": str(path), "lifecycle_sha256": lc.digest()})
    return records


def prepare(host, base, scenario, *, config=None, timeout_seconds=4800):
    if command(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Expected an immutable clean worker snapshot")
    lc = config if config is not None else lifecycle(scenario)
    root = ROOT / "results/lifecycles" / lc.lifecycle_id
    cfg = ROOT / "campaigns/opt-cmo-corruption/lifecycles" / f"{lc.lifecycle_id}.json"
    if parse_lifecycle(json.loads(cfg.read_text())).digest() != lc.digest():
        raise ValueError("Changed lifecycle")
    source = base / "inputs/exp1g-frozen-sources/std3/20261011T013506.097920Z-d4e9bdc7-e50b689b/checkpoints/source.pt"
    teacher = base / "inputs/opt-125m-27dcfa74/pytorch_model.bin"
    inputs = [checked_input(source, SOURCE_SHA), checked_input(teacher,
        "2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9")]
    for corpus in [lc.data] + list(lc.transfer_tests.tests):
        inputs.append(checked_input(base / "inputs/corpora" / f"{corpus.corpus}.pt", corpus.corpus_sha256))
    old = parse_lifecycle(json.loads((source.parent.parent / "config.resolved.json").read_text())["lifecycle"])
    verify_provenance(provenance(old, "hwa_source", hwa_arm=ARM), lc, "hwa_source", hwa_arm=ARM)
    env = dict(EBL_CORPUS_ROOT=str(base / "inputs/corpora"), EBL_CAMPAIGN_STREAM="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", EBL_SOURCE_RECEIPT=str(ROOT / "source-receipt.json"))
    command([sys.executable, "-m", "workflow", "plan", str(cfg), "--teacher-weights", str(teacher),
        "--hwa-source", f"{ARM}={source}", "--output", str(root / "plan"), "--device", "cuda:0",
        "--cpu-threads", "1", "--python", sys.executable], cwd=ROOT, env=dict(os.environ, **env))
    manifest = json.loads((root / "plan/campaign.json").read_text())
    manifest["stages"] = [s for s in manifest["stages"]
        if not s["id"].startswith("onchip__") or int(s["id"].rsplit("__", 1)[1]) in ARRAYS]
    for stage in manifest["stages"]:
        stage["config"] = str(root / "plan" / stage["config"])
    if len(manifest["stages"]) != 10 or any(s["id"].startswith("hwa__") for s in manifest["stages"]):
        raise ValueError("Unexpected stage scope")
    path = root / "campaign.json"
    write_frozen(path, manifest)
    command([sys.executable, "-m", "ebl", "campaign", "run", "--manifest", str(path),
        "--output-dir", str(root / "dry-run"), "--dry-run", "--fail-fast"], cwd=ROOT,
        env=dict(os.environ, **env), logs=root / "preparation/dry-run")
    receipt = root / "worker"
    worker = root / "tmux-worker.json"
    write_frozen(worker, dict(case_id=lc.lifecycle_id, state_root=str(receipt), cwd=str(ROOT),
        timeout_seconds=timeout_seconds, lease_seconds=900, env=env,
        argv=[sys.executable, str(Path(__file__).resolve()), "execute", "--manifest", str(path), "--output", str(root / "runs")]))
    validation = [sys.executable, "-m", "campaigns.lifecycle_collect", "validate", str(root / "runs" / lc.lifecycle_id), "--recovery"]
    for array in ARRAYS:
        validation += ["--recovery-assignment", str(array)]
    files = [cfg, path, worker, ROOT / "source-receipt.json"] + [Path(s["config"]) for s in manifest["stages"]]
    result = dict(host=host, id=lc.lifecycle_id, scenario=scenario, source=str(ROOT), source_revision=command(["git", "rev-parse", "HEAD"], cwd=ROOT).strip(),
        lifecycle_sha256=lc.digest(), dry_run_passed=True, native_stages=10,
        deployment_rows=len(lc.defects.cases) * len(lc.devices.assignment_seeds),
        recovery_endpoints=len(lc.defects.cases) * len(ARRAYS) * len(lc.onchip.arms),
        placement=dict(workdir=str(ROOT), run_dir=str(receipt), gpu_models=["5090"],
            argv=[sys.executable, "-m", "campaigns.tmux_worker", "bridge", "--spec", str(worker)],
            validate_argv=validation, campaign_python=sys.executable, campaign_worker_spec=str(worker),
            campaign_result_root=str(root / "runs" / lc.lifecycle_id),
            prepared_files=[dict(path=str(p), sha256=sha256(p)) for p in files] + inputs))
    write_frozen(root / "prepared.json", result)
    return result


def execute(manifest, output):
    # The shared queue permits sharing in other projects; this experiment requires idle admission.
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if active.strip():
        raise RuntimeError("GPU is occupied; this experiment does not permit sharing")
    subprocess.run([sys.executable, "-m", "ebl", "campaign", "run", "--manifest", str(manifest),
        "--output-dir", str(output), "--resume", "--fail-fast"], cwd=ROOT, check=True)
    return {"native_campaign": str(output / json.loads(manifest.read_text())["campaign_id"])}


def summarize(rows):
    lc = lifecycle()
    expected = {(a, c.label, arm.arm_id) for a in ARRAYS for c in lc.defects.cases for arm in lc.onchip.arms}
    keyed = {}
    for row in rows:
        key = row["array_seed"], row["case"], row["onchip_arm"]
        if key in keyed:
            # Only the healthy control repeats between the four separately executable slices.
            compare = {k: v for k, v in row.items() if k.startswith(("final_", "initial_", "cost_"))}
            if row["case"] != "nominal" or compare != {k: v for k, v in keyed[key].items() if k in compare}:
                raise ValueError("Unexpected duplicate or changed healthy control")
        else:
            keyed[key] = row
    if len(rows) != 252 or set(keyed) != expected:
        raise ValueError("Incomplete or duplicate matched recovery coverage")
    effects, decisions = [], []
    for row in rows:
        if row.get("duplicate_coverage") is not False or row["lifecycle_id"] not in {PREFIX + "-" + s for s in SCENARIOS}:
            raise ValueError("Invalid native coverage or lifecycle")
    for kind in ("gmin", "gmax"):
        for ppm in (20000, 40000, 60000):
            for case in [c for c in lc.defects.cases if c.kind == kind and c.rate_ppm == ppm and (c.die_concentration or c.array_concentration)]:
                scenario = "both" if case.die_concentration and case.array_concentration else "die" if case.die_concentration else "array"
                for domain in ("transfer_wikitext", "transfer_generated"):
                    paired = []
                    for array in ARRAYS:
                        def score(label, arm, metric="teacher_kl"):
                            return keyed[array, label, arm][f"final_{domain}_{metric}"]
                        uniform = f"{kind}_{ppm}ppm"
                        healthy = score("nominal", HOLD)
                        uniform_residual = score(uniform, HOLD) - healthy
                        delta = score(case.label, HOLD) - score(uniform, HOLD)
                        uniform_benefit = score(uniform, HOLD) - score(uniform, LEARN)
                        benefit = score(case.label, HOLD) - score(case.label, LEARN)
                        record = dict(array_seed=array, kind=kind, rate_ppm=ppm, scenario=scenario, case=case.label,
                            domain=domain, rewrite_delta_kl=delta,
                            rewrite_residual_relative_change=delta / uniform_residual if uniform_residual > 1e-6 else None,
                            uniform_learning_benefit_kl=uniform_benefit, clustered_learning_benefit_kl=benefit,
                            learning_benefit_delta_kl=benefit-uniform_benefit,
                            learning_benefit_sign_changed=(benefit > 0) != (uniform_benefit > 0),
                            arm_deltas={arm: dict(teacher_kl=score(case.label, arm)-score(uniform, arm),
                                perplexity=score(case.label, arm, "perplexity")-score(uniform, arm, "perplexity"))
                                for arm in ("none", HOLD, LEARN)})
                        paired.append(record); effects.append(record)
                    changes = [r["rewrite_residual_relative_change"] for r in paired]
                    same_positive = all(r["rewrite_delta_kl"] > 0 for r in paired)
                    same_negative = all(r["rewrite_delta_kl"] < 0 for r in paired)
                    verdict = "mixed_or_small"
                    if all(v is not None for v in changes):
                        if all(v >= .10 for v in changes): verdict = "material_increase"
                        elif all(v <= -.10 for v in changes): verdict = "material_decrease"
                    decisions.append(dict(kind=kind, rate_ppm=ppm, scenario=scenario, domain=domain,
                        consistent_direction="increase" if same_positive else "decrease" if same_negative else "mixed",
                        verdict=verdict, rewrite_residual_relative_changes=changes,
                        learning_benefit_changes=[r["learning_benefit_delta_kl"] for r in paired]))
    return dict(coverage_complete=True, assignments=list(ARRAYS), starting_conditions=75, endpoints=225, executed_endpoints=252,
        statistical_significance_claim=False, effects=effects, decisions=decisions)


def collect(queue_config, evidence):
    from campaigns.lifecycle_collect import collect as native_collect
    result = native_collect(queue_config, evidence / "collected", recovery=True, recovery_assignments=ARRAYS)
    folder = Path(result["collected"])
    mechanisms = []
    measured = []
    for path in folder.glob("onchip__*/measurements.json"):
        for row in json.loads(path.read_text()):
            measured.append(row)
            case = next(c for c in lifecycle().defects.cases if c.label == row["case"])
            expected_faults = 6 * round(case.rate * 9437184)
            if row["identity"]["failed_devices"] != expected_faults or row["replay_exact"] is not True:
                raise ValueError("Changed fault total or inexact replay")
            if row["onchip_arm"] == HOLD:
                groups = row["identity"].get("groups", [])
                dies, matrices = [], {}
                for group in groups:
                    if group["failed_devices"] != round(case.rate * group["devices"]):
                        raise ValueError("Fault total differs within an MLP")
                    for die in group["dies"]:
                        size = sum(a["devices"] for a in group["arrays"] if a["die"] == die["die"])
                        dies.append(die["failed_devices"] / size)
                    for array in group["arrays"]:
                        matrices[array["matrix"]] = matrices.get(array["matrix"], 0) + array["failed_devices"]
                mechanisms.append(dict(array_seed=row["array_seed"], case=row["case"],
                    array_fault_fraction=row["identity"].get("array_fault_fraction"),
                    die_fault_fraction=dict(std=pstdev(dies), min=min(dies), max=max(dies)) if dies else None,
                    failed_by_matrix=matrices, fault_rewrite=row["fault_rewrite"]))
    if os.environ["RUN_WATCH_JOB_ID"] == PREFIX + "-uniform":
        reference = json.loads((evidence / "reference/exp016-uniform.json").read_text())
        current = {(r["array_seed"], r["case"], r["onchip_arm"]): r for r in measured}
        for old in reference["rows"]:
            row = current[old["array_seed"], old["case"], old["onchip_arm"]]
            for key in ("initial_apparent_sha256", "final_apparent_sha256", "cost"):
                if row[key] != old[key]:
                    raise ValueError(f"Uniform anchor changed from exp-016: {row['case']} {row['onchip_arm']} {key}")
            for split, metrics in old["final"].items():
                for metric in ("teacher_kl", "perplexity"):
                    before, after = metrics[metric], row["final"][split][metric]
                    if abs(after - before) > 1e-6 * max(abs(before), 1e-12):
                        raise ValueError("Uniform anchor metrics changed from exp-016")
        write_frozen(evidence / "reviews/uniform-anchor.json", dict(passed=True, endpoints=len(reference["rows"])))
    write_frozen(evidence / "reviews" / (os.environ["RUN_WATCH_JOB_ID"] + "-mechanisms.json"), mechanisms)
    from campaigns.lifecycle_collect import collection_lock
    from experiments.artifacts import atomic_write_json
    with collection_lock(evidence / "reviews"):
        rows = json.loads((evidence / "collected/rows.json").read_text())
        completed = {r["lifecycle_id"] for r in rows}
        atomic_write_json(evidence / "reviews/progress.json", dict(completed_jobs=sorted(completed),
            expected_jobs=[PREFIX + "-" + s for s in SCENARIOS], executed_endpoints=len(rows)))
        if completed == {PREFIX + "-" + s for s in SCENARIOS}:
            report = summarize(rows)
            write_frozen(evidence / "reviews/summary.json", report)
    return {"collected": result["collected"], "completed_scenarios": len(completed), "executed_endpoints": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("generate")
    p = sub.add_parser("prepare"); p.add_argument("--host", required=True); p.add_argument("--base", type=Path, default=DEFAULT_BASE); p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p = sub.add_parser("execute"); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("collect"); p.add_argument("--queue-config", type=Path, required=True); p.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "generate": result = generate()
    elif args.mode == "prepare": result = prepare(args.host, args.base, args.scenario)
    elif args.mode == "execute": result = execute(args.manifest, args.output)
    else: result = collect(args.queue_config, args.evidence)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
