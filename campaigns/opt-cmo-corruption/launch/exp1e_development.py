"""Prepare and review the focused standard-3% HWA development experiment.

Training always uses the public lifecycle/campaign CLIs. This module only writes
strict lifecycles and reads explicitly selected native result bundles.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts import atomic_write_json, sha256_file
from workflow.lifecycle import parse_lifecycle

LIFECYCLES = ROOT / "campaigns/opt-cmo-corruption/lifecycles"
BASE = LIFECYCLES / "opt125m-cmo-l6-exp1e-lrdev.json"
NOTE = "campaigns/opt-cmo-corruption/series/001-hwa-robustness/experiments/exp-005-whole-model-hwa.md"
RATES = (1e-5, 1e-4, 1e-3, 5e-3, 1e-2)
RATE_LABELS = {1e-5: "1em5", 1e-4: "1em4", 1e-3: "1em3", 5e-3: "5em3", 1e-2: "1em2"}
SCHEDULES = ("linear", "step")
DURATION_RATES = (1e-3, 5e-3)
DURATION_BOUNDARY = 12800
ARM = "standard_hwa_model"


def case_id(rate, schedule):
    if rate not in RATES or schedule not in SCHEDULES:
        raise ValueError("Expected a predeclared development grid cell")
    return f"opt125m-cmo-l6-exp1e-dev-{schedule}-lr{RATE_LABELS[rate]}"


def lifecycle(rate, schedule, *, smoke=False):
    doc = json.loads(BASE.read_text())
    doc["lifecycle_id"] = case_id(rate, schedule) + ("-smoke" if smoke else "")
    doc["question"].update(
        text=("Readiness smoke (not scientific evidence). " if smoke else "")
        + f"Standard 3% whole-transformer HWA: peak Adam LR {rate:g}, {schedule} decay, "
          "1.6% warm-up on synthetic text; compare development KL at matched update budget.",
        decision=f"Recommend a common LR/schedule for the matched scope/recipe experiment; review before release. See {NOTE}.",
    )
    # Do not construct or evaluate test cohorts during hyperparameter development.
    # The main lifecycle keeps this data section and adds BOTH final test domains
    # as transfer_tests, preserving the winning source's exact provenance.
    doc["data"]["evaluation"] = "development"
    doc["data"]["test_sequences"] = 0
    doc.pop("transfer_tests", None)
    hwa = doc["hwa"]
    hwa.update(learning_rate=rate, lr_milestones=[], lr_decay=1.0, epochs=2 if smoke else 50)
    total = hwa["epochs"] * math.ceil(hwa["sequences_per_epoch"] / hwa["batch_size"])
    sched = {"kind": "linear_warmup_decay" if schedule == "linear" else "warmup_step_decay",
             "total_steps": total, "warmup_steps": math.ceil(0.016 * total)}
    if schedule == "step":
        sched.update(milestone_step=int(0.8 * total), decay_factor=0.1)
    hwa["lr_schedule"] = sched
    hwa["selection"] = {"every_epochs": 2, "extension_epochs": 0}
    arm = copy.deepcopy(hwa["arms"][0])
    arm.update(id=ARM, digital_learning_rate=rate, scope="model")
    hwa["arms"] = [arm]
    return parse_lifecycle(doc)


def prepare(output, rates=RATES):
    output.mkdir(parents=True, exist_ok=True)
    cases = [lifecycle(r, s) for r in rates for s in SCHEDULES]
    cases += [lifecycle(1e-4, s, smoke=True) for s in SCHEDULES]
    for lc in cases:
        path = output / (lc.lifecycle_id + ".json")
        payload = lc.to_dict()
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError(f"Refusing to replace a different prepared lifecycle: {path}")
            continue  # Preserve frozen bytes as well as their scientific identity.
        atomic_write_json(path, payload)
    return [{"id": lc.lifecycle_id, "lifecycle_sha256": lc.digest(),
             "path": str(output / (lc.lifecycle_id + '.json'))} for lc in cases]


def duration_lifecycle(rate):
    if rate not in DURATION_RATES:
        raise ValueError("Expected a predeclared duration-comparison learning rate")
    doc = lifecycle(rate, "step").to_dict()
    doc["lifecycle_id"] += "-tail4pass"
    doc["question"]["text"] = (
        f"Standard 3% whole-transformer HWA duration comparison at peak Adam LR {rate:g}: "
        "extend the unchanged 12,800-update step schedule to 25,600 updates on synthetic text, "
        "retaining 205 warm-up updates and the x0.1 decay after update 10,240. "
        "Compare only the two equal-budget duration cases; do not pool them into the original LR grid."
    )
    doc["hwa"]["epochs"] = 100
    doc["hwa"]["lr_schedule"]["total_steps"] = 25600
    return parse_lifecycle(doc)


def prepare_duration(output):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for rate in DURATION_RATES:
        lc = duration_lifecycle(rate)
        path = output / f"{lc.lifecycle_id}.json"
        if path.exists():
            if json.loads(path.read_text()) != lc.to_dict():
                raise ValueError(f"Refusing to replace a different prepared lifecycle: {path}")
        else:
            atomic_write_json(path, lc.to_dict())
        records.append({"id": lc.lifecycle_id, "lifecycle_sha256": lc.digest(), "path": str(path)})
    return records


def finite_metrics(value, label):
    """Reject nonfinite numeric fields, including per-case and clean metrics."""
    if isinstance(value, dict):
        for key, item in value.items():
            finite_metrics(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_metrics(item, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f"Nonfinite metric: {label}")


def verify_source(result_path, artifact):
    """Verify local bytes, or a receipt from remote native artifact inspection.

    Large sources may remain remote. The collector must create the adjacent
    ``artifact-inspection.json`` only after ``ebl runs inspect --verify-artifacts
    --require-complete`` succeeds on that exact native run. The receipt binds
    the original result bytes and complete source artifact record. A present
    but corrupt local file is never rescued by a remote receipt.
    """
    relative = Path(artifact["path"])
    if (not isinstance(artifact.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
            or type(artifact.get("size_bytes")) is not int or artifact["size_bytes"] <= 0):
        raise ValueError("Expected a source artifact digest and positive size")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected the source artifact inside its native run")
    source = result_path.parent / relative
    if source.exists():
        if not source.resolve().is_relative_to(result_path.parent.resolve()):
            raise ValueError("Source artifact resolves outside its native run")
        if not source.is_file() or source.stat().st_size != artifact["size_bytes"] or sha256_file(source) != artifact["sha256"]:
            raise ValueError("Source artifact size or digest mismatch")
        return {"kind": "local_bytes", "sha256": artifact["sha256"]}
    receipt_path = result_path.parent / "artifact-inspection.json"
    if not receipt_path.exists():
        raise ValueError("Source artifact is absent and has no remote inspection receipt")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "exp1e.artifact_inspection.v1"
            or receipt.get("result_sha256") != sha256_file(result_path)
            or receipt.get("verified") is not True
            or receipt.get("verify_artifacts") is not True
            or receipt.get("require_complete") is not True
            or type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0
            or receipt.get("artifacts", []).count(artifact) != 1):
        raise ValueError("Missing, failed or stale remote artifact inspection receipt")
    return {"kind": "remote_inspection", "receipt": str(receipt_path.resolve()),
            "receipt_sha256": sha256_file(receipt_path)}


def read_case(config_path, result_path):
    """Read one explicitly named result and check its scientific identity."""
    lc = parse_lifecycle(json.loads(config_path.read_text()))
    canonical = {case_id(rate, schedule): lifecycle(rate, schedule) for rate in RATES for schedule in SCHEDULES}
    canonical.update({duration_lifecycle(rate).lifecycle_id: duration_lifecycle(rate) for rate in DURATION_RATES})
    if lc.lifecycle_id not in canonical or lc.digest() != canonical[lc.lifecycle_id].digest():
        raise ValueError("Result is not an exact predeclared development grid cell")
    result = json.loads(result_path.read_text())
    resolved_path = result_path.parent / "config.resolved.json"
    resolved = json.loads(resolved_path.read_text())
    observed = parse_lifecycle(resolved["lifecycle"])
    if observed.digest() != lc.digest() or resolved["stage"] != {"kind": "hwa", "hwa_arm": ARM, "array_seed": None}:
        raise ValueError(f"Result/config identity mismatch: {result_path}")
    if result.get("status") != "complete":
        raise ValueError(f"Operationally incomplete result: {result_path}")
    metrics = result["metrics"]
    if metrics.get("test_used_for_selection") is not False:
        raise ValueError("Missing explicit no-test-selection receipt")
    fit = metrics["fit"]
    if fit["status"] not in ("complete", "rejected_nonfinite"):
        raise ValueError(f"Unsupported scientific fit status: {fit['status']}")
    finite_metrics(fit, "fit")
    finite_metrics(metrics["clean_development"], "clean_development")
    finite_metrics(result["duration_seconds"], "duration_seconds")
    clean_kl = metrics["clean_development"].get("teacher_kl")
    if type(clean_kl) not in (int, float) or type(result["duration_seconds"]) not in (int, float) or result["duration_seconds"] < 0:
        raise ValueError("Expected numeric clean development KL and nonnegative run duration")
    if type(fit["convergence_review_required"]) is not bool:
        raise ValueError("Missing convergence-review receipt")
    history = fit["history"]
    if not isinstance(history, list) or not history:
        raise ValueError("Expected a nonempty development history")
    steps = lc.hwa.lr_schedule.total_steps
    steps_per_epoch = steps // lc.hwa.epochs
    recorded_schedule = fit["lr_schedule"]
    if any(recorded_schedule.get(key) != value for key, value in lc.hwa.lr_schedule.to_dict().items()):
        raise ValueError("Recorded learning-rate schedule differs from the lifecycle")
    steps_run = recorded_schedule["steps_run"]
    if type(steps_run) is not int or not 0 <= steps_run <= steps:
        raise ValueError("Invalid completed-update count")
    rate = lc.hwa.learning_rate
    if recorded_schedule.get("base_learning_rates") != [rate, rate]:
        raise ValueError("Unexpected optimizer-group base learning rates")
    actual_rates = recorded_schedule.get("final_learning_rates")
    expected_rate = rate * lc.hwa.lr_schedule.factor(steps_run)
    if not isinstance(actual_rates, list) or len(actual_rates) != 2 or any(
        not math.isclose(value, expected_rate, rel_tol=1e-12, abs_tol=0) for value in actual_rates
    ):
        raise ValueError("Final optimizer-group learning rates disagree with completed updates")
    expected_epochs = list(range(0, lc.hwa.epochs + 1, lc.hwa.selection_every))
    if expected_epochs[-1] != lc.hwa.epochs:
        expected_epochs.append(lc.hwa.epochs)
    epochs = [item["epoch"] for item in history]
    if epochs != expected_epochs[:len(epochs)]:
        raise ValueError("Incomplete, duplicated or unordered checkpoint-selection history")
    if fit["maximum_epochs"] != lc.hwa.epochs:
        raise ValueError("Unexpected training-budget extension")
    if fit["status"] == "complete":
        if (fit["epochs_run"] != lc.hwa.epochs or steps_run != steps or epochs != expected_epochs
                or fit.get("rejection") is not None):
            raise ValueError("Completed fit has incomplete training coverage or a rejection")
    elif not isinstance(fit.get("rejection"), dict) or "nonfinite" not in fit["rejection"].get("reason", ""):
        raise ValueError("Rejected-nonfinite fit lacks its explicit numerical rejection")
    case_names = [case.label for case in lc.hwa.arms[0].selection_cases]
    for item in history:
        if (type(item["epoch"]) is not int or type(item["global_step"]) is not int
                or item["global_step"] != item["epoch"] * steps_per_epoch or item["global_step"] > steps_run):
            raise ValueError("Selection history has an inconsistent optimizer step")
        if [case["case"] for case in item["cases"]] != case_names:
            raise ValueError("Unexpected development selection cases")
        scores = [case["teacher_kl"] for case in item["cases"]]
        if any(score < 0 for score in scores) or not math.isclose(
            item["teacher_kl"], sum(scores) / len(scores), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("Development score differs from the declared mean over selection cases")
    selected = next((h for h in history if h["epoch"] == fit["selected_epoch"]), None)
    if selected is None or selected["teacher_kl"] != fit["development_kl"]:
        raise ValueError("Selected checkpoint score does not match its development history")
    if selected != min(history, key=lambda item: item["teacher_kl"]):
        raise ValueError("Checkpoint did not follow minimum-development-KL selection")
    sources = [a for a in result["artifacts"] if a["kind"] == "hwa_source"]
    if len(sources) != 1:
        raise ValueError("Expected one source artifact")
    verification = verify_source(result_path, sources[0])
    score = float(fit["development_kl"])
    if score < 0:
        raise ValueError("Unexpected negative development KL")
    return {
        "id": lc.lifecycle_id, "lifecycle": str(config_path.resolve()), "lifecycle_sha256": lc.digest(),
        "comparison": "duration" if lc.lifecycle_id.endswith("-tail4pass") else "lr_grid",
        "result": str(result_path.resolve()), "result_sha256": sha256_file(result_path),
        "source": sources[0], "source_verification": verification, "fit_status": fit["status"],
        "peak_lr": lc.hwa.learning_rate, "schedule": lc.hwa.lr_schedule.kind,
        "total_steps": steps, "steps_run": steps_run, "selected_epoch": fit["selected_epoch"],
        "selected_step": selected["global_step"],
        "selected_development_kl": score, "final_development_kl": history[-1]["teacher_kl"],
        "initial_development_kl": history[0]["teacher_kl"],
        "selected_cases": selected["cases"], "history": history,
        "clean_development": metrics["clean_development"],
        "convergence_review_required": fit["convergence_review_required"],
        "duration_seconds": result["duration_seconds"],
    }


def recommend(cases):
    viable = [c for c in cases if c["fit_status"] == "complete"]
    if not viable:
        return None
    best = min(c["selected_development_kl"] for c in viable)
    tied = [c for c in viable if c["selected_development_kl"] <= best * 1.01]
    # Predeclared tie break: lower peak LR, then the user's proposed linear schedule.
    return min(tied, key=lambda c: (c["peak_lr"], c["schedule"] != "linear_warmup_decay", c["id"]))["id"]


def review_warnings(cases, recommendation):
    warnings = []
    for case in cases:
        if case["fit_status"] != "complete":
            warnings.append({"case": case["id"], "kind": "numerical_rejection",
                             "message": "Numerically rejected; excluded from recommendation."})
        if case["selected_step"] == 0:
            warnings.append({"case": case["id"], "kind": "initial_checkpoint",
                             "message": "The initial checkpoint wins: training did not improve the declared development score."})
        elif case["selected_step"] >= 0.8 * case["total_steps"]:
            warnings.append({"case": case["id"], "kind": "late_checkpoint",
                             "message": "The selected checkpoint is in the final 20% of updates; convergence is unresolved."})
    if recommendation is not None:
        winner = next(case for case in cases if case["id"] == recommendation)
        if winner["peak_lr"] in (min(RATES), max(RATES)):
            warnings.append({"case": recommendation, "kind": "grid_edge",
                             "message": "The recommended LR is at the grid boundary; review before freezing it. No automatic extension is authorized."})
    return warnings


def plot_development(cases, output):
    """Standalone scientific curves; display actual completed updates only."""
    if not cases:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    colors = dict(zip(RATES, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")))
    for case in sorted(cases, key=lambda value: (value["peak_lr"], value["schedule"])):
        lc = parse_lifecycle(json.loads(Path(case["lifecycle"]).read_text()))
        schedule = lc.hwa.lr_schedule
        is_linear = schedule.kind == "linear_warmup_decay"
        style = "-" if is_linear else "--"
        color = colors[case["peak_lr"]]
        label = f"{case['peak_lr']:g}, {'linear' if is_linear else 'step'}"
        if case["fit_status"] != "complete":
            label += " (rejected)"
        updates = range(1, case["steps_run"] + 1)
        axes[0].plot(updates, [case["peak_lr"] * schedule.factor(step) for step in updates],
                     linestyle=style, color=color, label=label)
        history = case["history"]
        axes[1].plot([item["global_step"] for item in history], [item["teacher_kl"] for item in history],
                     marker=".", linestyle=style, color=color, label=label)
        axes[1].scatter([case["selected_step"]], [case["selected_development_kl"]],
                        marker="o", facecolors="none", edgecolors=color, s=55)
    axes[0].set_ylabel("Learning rate")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set_ylabel("Mean development teacher KL")
    for axis in axes:
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Circles mark selected checkpoints; test cohorts remain held out", fontsize=10)
    duration = any(case["total_steps"] > DURATION_BOUNDARY for case in cases)
    if duration:
        for axis in axes:
            axis.axvline(DURATION_BOUNDARY, color="0.4", linewidth=1, linestyle=":")
    fig.suptitle("Standard 3% whole-model HWA " + ("duration comparison" if duration else "development")
                 + " — one training seed")
    filenames = ["development-curves.png", "development-curves.pdf"]
    try:
        for name in filenames:
            fig.savefig(output / name, dpi=180)
    finally:
        plt.close(fig)
    return filenames


def checkpoint_metrics(item):
    if item is None:
        return None
    return {"step": item["global_step"], "mean_teacher_kl": item["teacher_kl"],
            "cases": {case["case"]: case["teacher_kl"] for case in item["cases"]}}


def metric_change(before, after):
    """Signed changes are after minus before; positive relative reduction improves."""
    if before is None or after is None:
        return None
    return {"mean_teacher_kl_delta": after["mean_teacher_kl"] - before["mean_teacher_kl"],
            "case_teacher_kl_deltas": {name: after["cases"][name] - value
                                      for name, value in before["cases"].items()},
            "relative_mean_reduction": (1 - after["mean_teacher_kl"] / before["mean_teacher_kl"])
            if before["mean_teacher_kl"] else None}


def duration_review(cases, original_cases):
    """Compare only the two equal-budget extensions, with their matched prefixes."""
    expected = {duration_lifecycle(rate).lifecycle_id for rate in DURATION_RATES}
    seen = {case["id"] for case in cases}
    originals = {case["id"]: case for case in original_cases}
    reference_ids = {case_id(rate, "step") for rate in DURATION_RATES}
    missing_references = sorted(name for name in reference_ids
                                if name not in originals or originals[name]["fit_status"] != "complete")
    complete = seen == expected
    report = {"schema": "exp1e.duration_review.v1", "coverage_complete": complete,
              "expected": sorted(expected), "missing": sorted(expected - seen),
              "reference_coverage_complete": not missing_references,
              "missing_complete_references": missing_references,
              "budget_steps": 25600, "comparison_boundary_steps": DURATION_BOUNDARY,
              "primary_duration_effect": "terminal development KL at 25600 minus 12800 within each run",
              "selection_caveat": "Selected-checkpoint improvement includes more selection opportunities at the longer budget.",
              "cases": cases, "comparisons": [],
              "recommendation": recommend(cases) if complete and not missing_references else None,
              "downstream_gate": "awaiting_scientific_review", "test_used_for_selection": False,
              "warnings": review_warnings(cases, None)}
    for case in sorted(cases, key=lambda item: item["peak_lr"]):
        points = {item["global_step"]: item for item in case["history"]}
        prefix = [item for item in case["history"] if item["global_step"] <= DURATION_BOUNDARY]
        boundary = checkpoint_metrics(points.get(DURATION_BOUNDARY))
        selected_before = checkpoint_metrics(min(prefix, key=lambda item: item["teacher_kl"])) if boundary else None
        terminal = checkpoint_metrics(points.get(25600))
        selected_after = checkpoint_metrics(min(case["history"], key=lambda item: item["teacher_kl"])) if terminal else None
        late_start = checkpoint_metrics(points.get(20480))
        reference = originals.get(case_id(case["peak_lr"], "step"))
        reference_terminal = reference_selected = prefix_identical = None
        if reference is not None and reference["fit_status"] == "complete":
            reference_terminal = checkpoint_metrics(reference["history"][-1])
            reference_selected = checkpoint_metrics(min(reference["history"], key=lambda item: item["teacher_kl"]))
            if boundary is not None:
                prefix_identical = [checkpoint_metrics(item) for item in prefix] == [
                    checkpoint_metrics(item) for item in reference["history"]]
                if not prefix_identical:
                    report["warnings"].append({"case": case["id"], "kind": "prefix_replay_difference",
                                               "message": "The repeated first 12,800 updates differ from the prior run; review the matched comparison before attributing changes to duration."})
        report["comparisons"].append({
            "id": case["id"], "peak_lr": case["peak_lr"],
            "reference_id": case_id(case["peak_lr"], "step"),
            "checkpoint_12800": boundary, "prior_checkpoint_12800": reference_terminal,
            "prior_selected": reference_selected, "history_prefix_identical": prefix_identical,
            "prior_replay_change": metric_change(reference_terminal, boundary),
            "selected_through_12800": selected_before, "selected_through_25600": selected_after,
            "terminal_25600": terminal,
            "selected_change_12800_to_25600": metric_change(selected_before, selected_after),
            "terminal_change_12800_to_25600": metric_change(boundary, terminal),
            "checkpoint_20480": late_start,
            "terminal_change_20480_to_25600": metric_change(late_start, terminal),
            "selected_in_final_20_percent": case["selected_step"] >= 20480,
        })
    return report


def write_duration_review(report, output):
    output.mkdir(parents=True, exist_ok=True)
    report["plots"] = plot_development(report["cases"], output)
    atomic_write_json(output / "duration.json", report)
    lines = ["# Experiment 1e duration comparison", "",
             "The two 25,600-update step runs are ranked separately from the 12,800-update LR grid. "
             "Means equally weight the healthy and 3% random-fault development cases. Tests remain held out.", "",
             "The primary duration effect compares terminal KL at 12,800 and 25,600 updates. "
             "Selected-checkpoint improvements are separate because the longer run has more selection opportunities.", "",
             "Changes below are after minus before; negative KL changes indicate improvement.", ""]
    for comparison in report["comparisons"]:
        lines += [f"## {comparison['id']}", "",
                  f"Reference: `{comparison['reference_id']}`. Exact metric-history prefix replay: "
                  f"{comparison['history_prefix_identical']}.", "",
                  "| Checkpoint | Update | Mean KL | Healthy KL | 3% faulty KL |",
                  "|---|---:|---:|---:|---:|"]
        for key, label in (("prior_checkpoint_12800", "Prior terminal"), ("prior_selected", "Prior selected"),
                           ("checkpoint_12800", "Repeated 12,800"), ("selected_through_12800", "Selected through 12,800"),
                           ("selected_through_25600", "Selected through 25,600"), ("terminal_25600", "Terminal 25,600")):
            value = comparison[key]
            if value is not None:
                lines.append(f"| {label} | {value['step']} | {value['mean_teacher_kl']:.8g} | "
                             f"{value['cases']['nominal']:.8g} | {value['cases']['random_30000ppm']:.8g} |")
        lines += ["", "| Comparison | Mean KL change | Healthy change | Faulty change |",
                  "|---|---:|---:|---:|"]
        for key, label in (("prior_replay_change", "Repeated minus prior at 12,800"),
                           ("terminal_change_12800_to_25600", "Terminal: 12,800 to 25,600"),
                           ("selected_change_12800_to_25600", "Selected: 12,800 to 25,600"),
                           ("terminal_change_20480_to_25600", "Terminal: final 20%")):
            value = comparison[key]
            if value is not None:
                lines.append(f"| {label} | {value['mean_teacher_kl_delta']:.8g} | "
                             f"{value['case_teacher_kl_deltas']['nominal']:.8g} | "
                             f"{value['case_teacher_kl_deltas']['random_30000ppm']:.8g} |")
        lines.append("")
    lines += [f"Duration coverage: {len(report['cases'])}/{len(report['expected'])}.",
              f"Missing complete 12,800-update references: {report['missing_complete_references']}.",
              f"Recommendation: {report['recommendation'] or 'none; incomplete coverage, missing reference or no stable candidate'}.",
              "Within 1% of the best selected mean development KL, prefer the lower peak LR. "
              "Broader experiments remain held for scientific review."]
    if report["warnings"]:
        lines += ["", "Warnings:", ""]
        lines += [f"- `{item['case']}`: {item['message']}" for item in report["warnings"]]
    if report["plots"]:
        lines += ["", "![Duration LR and development KL](development-curves.png)", "",
                  "[Vector figure](development-curves.pdf). The vertical line marks 12,800 updates."]
    (output / "duration.md").write_text("\n".join(lines) + "\n")


def summarize(index, output):
    entries = json.loads(index.read_text())
    if not isinstance(entries, list):
        raise ValueError("Expected an explicit list of lifecycle/result paths")
    all_cases = [read_case(Path(e["lifecycle"]), Path(e["result"])) for e in entries]
    expected = {case_id(r, s) for r in RATES for s in SCHEDULES}
    all_ids = [c["id"] for c in all_cases]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate development case; select the intended attempt explicitly")
    duration_expected = {duration_lifecycle(rate).lifecycle_id for rate in DURATION_RATES}
    if set(all_ids) - expected - duration_expected:
        raise ValueError("Unexpected development case")
    cases = [case for case in all_cases if case["id"] in expected]
    duration_cases = [case for case in all_cases if case["id"] in duration_expected]
    seen = {case["id"] for case in cases}
    complete = set(seen) == expected
    report = {"schema": "exp1e.development_review.v1", "coverage_complete": complete,
              "expected": sorted(expected), "missing": sorted(expected - set(seen)), "cases": cases,
              "recommendation": recommend(cases) if complete else None,
              "downstream_gate": "awaiting_scientific_review", "test_used_for_selection": False,
              "input_index_sha256": sha256_file(index)}
    report["warnings"] = review_warnings(cases, report["recommendation"])
    output.mkdir(parents=True, exist_ok=True)
    report["plots"] = plot_development(cases, output)
    report["duration_comparison"] = duration_review(duration_cases, cases)
    write_duration_review(report["duration_comparison"], output / "duration")
    atomic_write_json(output / "development.json", report)
    lines = ["# Experiment 1e development", "", "Exploratory, one training seed; tests are held out.", "",
             "| Case | Peak LR | Best dev KL | Final dev KL | Selected step | Status |",
             "|---|---:|---:|---:|---:|---|"]
    for c in sorted(cases, key=lambda c: (c["peak_lr"], c["schedule"])):
        lines.append(f"| {c['id']} | {c['peak_lr']:g} | {c['selected_development_kl']:.6g} | "
                     f"{c['final_development_kl']:.6g} | {c['selected_step']} | {c['fit_status']} |")
    lines += ["", f"Coverage: {len(cases)}/{len(expected)}.",
              f"Recommendation: {report['recommendation'] or 'none; incomplete coverage or no stable candidate'}.",
              "Broader experiments remain held for scientific review."]
    lines += ["", "[Separate duration comparison](duration/duration.md): "
              f"{len(duration_cases)}/2 cases. These runs do not enter the original LR ranking."]
    if report["warnings"]:
        lines += ["", "Warnings:", ""]
        lines += [f"- `{item['case']}`: {item['message']}" for item in report["warnings"]]
    if report["plots"]:
        lines += ["", "![Learning rates and development KL by optimizer update](development-curves.png)",
                  "", "[Vector figure](development-curves.pdf)"]
    (output / "development.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, default=LIFECYCLES)
    p = sub.add_parser("prepare-duration")
    p.add_argument("--output", type=Path, default=LIFECYCLES)
    s = sub.add_parser("summarize")
    s.add_argument("--index", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "prepare-duration":
        result = prepare_duration(args.output)
    else:
        result = summarize(args.index, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command.startswith("prepare"):
        return 0
    duration = result["duration_comparison"]
    ready = result["coverage_complete"] and result["recommendation"] is not None
    if duration["cases"]:
        ready = ready and duration["coverage_complete"] and duration["recommendation"] is not None
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
