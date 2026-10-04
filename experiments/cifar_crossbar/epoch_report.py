"""Validated paired summaries and static scientific plots of recovery budgets."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.analysis import records

MILESTONES = (1, 2, 5, 10, 20)
METHODS = ("calibration", "rewrite", "onchip_weights", "onchip_calibration")
LABELS = {"digital": "Digital source", "standard_hwa": "Standard HWA",
          "noise_hwa": "Noise-selected HWA", "cdt_open": "CDT open",
          "cdt_gmax": "CDT Gmax", "cdt_random": "CDT random",
          "calibration": "Calibration", "rewrite": "Rewrite + calibration",
          "onchip_weights": "Weight recovery", "onchip_calibration": "Joint recovery"}


def csv_write(path, rows):
    if not rows:
        raise ValueError("Empty report table.")
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten(runs):
    rows, development, policies = [], [], {}
    expected_pairs = set()
    for run in runs:
        spec, value = run["config"], run["metrics"]
        dataset = spec["dataset"]
        if value["smoke_only"] or value["epochs"] != 20 or value["milestones"] != list(MILESTONES):
            raise ValueError("Expected complete 20-pass evidence, not a canary.")
        if value["recovery_images"] != 5000 or value["evaluation_images"] != 10000 or value["development_images"] != 1000:
            raise ValueError("Incomplete cohort coverage.")
        if spec["stage"] == "tune":
            if value["completed_trajectories"] != 48:
                raise ValueError("Missing development schedule trials.")
            policies[dataset] = value["selected"]
            continue
        if value["completed_controls"] != 90:
            raise ValueError("Missing final controls.")
        grouped = defaultdict(list)
        for record in value["measurements"]:
            grouped[(record["source"], record["case"])].append(record)
        if len(grouped) != 18:
            raise ValueError("Expected 18 source/case combinations.")
        for (source, case), controls in grouped.items():
            pairing = (dataset, source, case, spec["array_seed"])
            if pairing in expected_pairs:
                raise ValueError("Duplicate physical-array/source/case run.")
            expected_pairs.add(pairing)
            if {r["method"] for r in controls} != {"none", *METHODS} or len(controls) != 5:
                raise ValueError("Incomplete paired controls.")
            for field in ("p0_sha256", "initial_apparent_sha256"):
                if len({r[field] for r in controls}) != 1:
                    raise ValueError(f"Unpaired {field}.")
            if len({r["faults"]["identity_sha256"] for r in controls}) != 1:
                raise ValueError("Unpaired fixed failures.")
            for record in controls:
                base = dict(dataset=dataset, source=source, case=case,
                            fault_kind=record["fault_kind"], fault_rate_ppm=record["fault_rate_ppm"],
                            array_seed=record["array_seed"], method=record["method"],
                            fault_identity_sha256=record["faults"]["identity_sha256"],
                            p0_sha256=record["p0_sha256"])
                initial = record["initial_test"]
                rows.append({**base, "epoch": 0, "teacher_kl": initial["teacher_kl"],
                             "accuracy_percent": initial["accuracy_percent"],
                             "image_presentations": 0, "array_reprogram_calls": 0,
                             "schedule": record.get("schedule", "none")})
                if record["method"] == "none":
                    if record["curve"]:
                        raise ValueError("No-update baseline unexpectedly trained.")
                    continue
                if [x["epoch"] for x in record["curve"]] != list(range(1, 21)):
                    raise ValueError("Incomplete learning curve.")
                for point in record["curve"]:
                    e = point["epoch"]
                    expected_writes = 0 if record["method"] == "calibration" else 79 * e
                    if point["cost"]["array_reprogram_calls"] != expected_writes or point["image_presentations"] != e * 5000:
                        raise ValueError("Unequal or incomplete recovery budget.")
                    development.append({**base, "epoch": e, "teacher_kl": point["development"]["teacher_kl"],
                                        "accuracy_percent": point["development"]["accuracy_percent"],
                                        "teacher_kl_train_mean": point["teacher_kl_train_mean"]})
                    if e in MILESTONES:
                        test = point["test"]
                        if test["examples"] != 10000:
                            raise ValueError("Incomplete final test cohort.")
                        rows.append({**base, "epoch": e, "teacher_kl": test["teacher_kl"],
                                     "accuracy_percent": test["accuracy_percent"],
                                     "image_presentations": point["image_presentations"],
                                     "array_reprogram_calls": point["cost"]["array_reprogram_calls"],
                                     "schedule": record["schedule"]})
    for r in rows:
        if r["method"] != "none" and r["schedule"] != policies[r["dataset"]][r["method"]]["schedule"]:
            raise ValueError("Final schedule differs from development selection.")
    for ds in policies:
        if sum(p[0] == ds for p in expected_pairs) != 54:
            raise ValueError("Missing fresh final arrays.")
    return rows, development, policies


def summarize(rows):
    groups = defaultdict(list)
    keys = ("dataset", "source", "case", "method", "epoch")
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    result = []
    for key, group in sorted(groups.items()):
        if {r["array_seed"] for r in group} != {101001, 101002, 101003} or len(group) != 3:
            raise ValueError("Expected three distinct fresh matched arrays.")
        row = dict(zip(keys, key))
        row["arrays"] = 3
        for metric in ("teacher_kl", "accuracy_percent"):
            values = [r[metric] for r in group]
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values, ddof=1))
        result.append(row)
    return result


def paired(rows):
    lookup = {(r["dataset"], r["source"], r["case"], r["method"], r["epoch"], r["array_seed"]): r for r in rows}
    grouped = defaultdict(list)
    for r in rows:
        if r["epoch"] == 0 or r["method"] == "none":
            continue
        comparisons = [("same_method_epoch1", r["method"], 1)]
        if r["method"] in ("onchip_weights", "onchip_calibration"):
            comparisons += [("calibration_same_budget", "calibration", r["epoch"]),
                            ("rewrite_same_budget", "rewrite", r["epoch"])]
        for label, method, epoch in comparisons:
            control = lookup[(r["dataset"], r["source"], r["case"], method, epoch, r["array_seed"])]
            if control["p0_sha256"] != r["p0_sha256"]:
                raise ValueError("Unpaired comparison.")
            key = (r["dataset"], r["source"], r["case"], r["method"], r["epoch"], label)
            grouped[key].append((control["teacher_kl"] - r["teacher_kl"], r["accuracy_percent"] - control["accuracy_percent"]))
    result = []
    for key, group in sorted(grouped.items()):
        if len(group) != 3:
            raise ValueError("Missing paired array comparison.")
        a = np.asarray(group)
        result.append({**dict(zip(("dataset", "source", "case", "method", "epoch", "comparison"), key)),
                       "arrays": 3, "kl_reduction_mean": float(a[:, 0].mean()),
                       "kl_reduction_std": float(a[:, 0].std(ddof=1)),
                       "accuracy_gain_pp_mean": float(a[:, 1].mean()),
                       "accuracy_gain_pp_std": float(a[:, 1].std(ddof=1)),
                       "arrays_kl_improved": int((a[:, 0] > 0).sum()),
                       "arrays_accuracy_improved": int((a[:, 1] > 0).sum()),
                       "arrays_both_improved": int(((a[:, 0] > 0) & (a[:, 1] > 0)).sum())})
    return result


def cross_source_pairs(rows):
    """Retain all HWA-control contrasts; do not select a source on final tests."""
    lookup = {(r["dataset"], r["case"], r["epoch"], r["source"], r["method"], r["array_seed"]): r for r in rows}
    applicable = defaultdict(set)
    for r in rows:
        if r["method"] == "none" and r["source"] != "digital":
            applicable[(r["dataset"], r["case"])].add(r["source"])
    grouped = defaultdict(list)
    for r in rows:
        if r["method"] not in ("onchip_weights", "onchip_calibration") or not r["epoch"]:
            continue
        for source in sorted(applicable[(r["dataset"], r["case"])]):
            for method in ("none", "calibration", "rewrite"):
                control_epoch = 0 if method == "none" else r["epoch"]
                control = lookup[(r["dataset"], r["case"], control_epoch, source, method, r["array_seed"])]
                if r["fault_identity_sha256"] != control["fault_identity_sha256"]:
                    raise ValueError("Cross-source physical failure identity mismatch.")
                key = (r["dataset"], r["case"], r["epoch"], r["source"], r["method"], source, method)
                grouped[key].append((control["teacher_kl"] - r["teacher_kl"], r["accuracy_percent"] - control["accuracy_percent"]))
    result = []
    for key, group in sorted(grouped.items()):
        if len(group) != 3:
            raise ValueError("Missing cross-source paired array.")
        a = np.asarray(group)
        result.append({**dict(zip(("dataset", "case", "epoch", "source", "method", "control_source", "control_method"), key)),
                       "arrays": 3, "kl_reduction_mean": float(a[:, 0].mean()),
                       "kl_reduction_std": float(a[:, 0].std(ddof=1)),
                       "accuracy_gain_pp_mean": float(a[:, 1].mean()),
                       "accuracy_gain_pp_std": float(a[:, 1].std(ddof=1)),
                       "arrays_both_improved": int(((a[:, 0] > 0) & (a[:, 1] > 0)).sum())})
    return result


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ("#3069a6", "#999999", "#c87725", "#20875c")))
    sources = ("digital", "standard_hwa", "noise_hwa", "cdt_open", "cdt_gmax", "cdt_random")
    for ds in sorted({r["dataset"] for r in summary}):
        for case in ("nominal", "open_10000ppm", "gmax_10000ppm", "random_10000ppm"):
            names = [s for s in sources if any(r["dataset"] == ds and r["case"] == case and r["source"] == s for r in summary)]
            fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 6.3), squeeze=False)
            for col, source in enumerate(names):
                for method in METHODS:
                    curve = sorted([r for r in summary if (r["dataset"], r["case"], r["source"], r["method"]) == (ds, case, source, method)], key=lambda r: r["epoch"])
                    x = np.array([r["epoch"] for r in curve])
                    for row, metric in enumerate(("teacher_kl", "accuracy_percent")):
                        y = np.array([r[metric + "_mean"] for r in curve])
                        sd = np.array([r[metric + "_std"] for r in curve])
                        ax = axes[row, col]
                        ax.plot(x, y, "o-", color=colors[method], markersize=3, label=LABELS[method])
                        ax.fill_between(x, np.maximum(y-sd, 1e-8) if row == 0 else y-sd, y+sd, color=colors[method], alpha=.1)
                axes[0, col].set_title(LABELS[source])
                axes[0, col].set_yscale("log")
                for row in range(2):
                    axes[row, col].grid(alpha=.2)
                    axes[row, col].set_xticks((0, 1, 5, 10, 20))
                    axes[row, col].set_xlabel("Passes over 5,000 images")
                axes[1, col].axhline(93.53 if ds == "cifar10" else 70.16, color="#444444", linewidth=.8, linestyle=":")
            axes[0, 0].set_ylabel("Teacher KL (nats; log scale)")
            axes[1, 0].set_ylabel("Test accuracy (%)")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
            fig.suptitle(f"{ds.upper()} — {case.replace('_10000ppm', ' failures (1% per device)')}\nMean ± sample SD across 3 fixed arrays; dotted line: original digital accuracy", fontsize=12)
            fig.tight_layout(rect=(0, .06, 1, 1))
            for ext in ("png", "pdf"):
                fig.savefig(output / f"{ds}_{case}.{ext}", dpi=180, bbox_inches="tight")
            plt.close(fig)


def report(executions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs, provenance = [], []
    for path in executions:
        campaign, collected = records(path)
        if campaign["phase"] != "recovery_epochs":
            raise ValueError("Expected the declared recovery-epochs campaign.")
        runs.extend(collected)
        provenance.append({"execution": str(Path(path).resolve()), "sha256": sha256_file(Path(path))})
    rows, development, policies = flatten(runs)
    summary, comparisons = summarize(rows), paired(rows)
    cross_source = cross_source_pairs(rows)
    csv_write(output / "per_array.csv", rows)
    csv_write(output / "development_curves.csv", development)
    csv_write(output / "summary.csv", summary)
    csv_write(output / "paired_comparisons.csv", comparisons)
    csv_write(output / "cross_source_comparisons.csv", cross_source)
    selected_rows = [r for r in rows if (r["method"] == "none" or
                     r["epoch"] == policies[r["dataset"]][r["method"]]["selected_epoch"])]
    csv_write(output / "development_selected_endpoints.csv", selected_rows)
    counts = {"final_controls": sum(r["epoch"] == 0 for r in rows),
              "post_adaptation_test_endpoints": sum(r["epoch"] > 0 for r in rows),
              "final_learning_trajectories": sum(r["epoch"] == 20 for r in rows)}
    atomic_write_json(output / "results.json", {"counts": counts, "policies": policies,
                      "summary": summary, "paired_comparisons": comparisons,
                      "cross_source_comparisons": cross_source,
                      "provenance": provenance, "analysis_code_sha256": sha256_file(Path(__file__))})
    lookup = {(r["dataset"], r["source"], r["case"], r["method"], r["epoch"]): r for r in summary}
    def value(ds, source, case, method, epoch):
        r = lookup[(ds, source, case, method, epoch)]
        return f"{r['teacher_kl_mean']:.4f} / {r['accuracy_percent_mean']:.2f}%"
    lines = ["# CIFAR recovery over 20 passes", "", "Exploratory Gaussian-endpoint reprogramming; digital Adam/backprop. Each pass uses the same 5,000 images. All tables show teacher KL (nats, T=1) / test accuracy. Full test: 10,000 images; means over 3 fresh arrays. No test-based schedule or epoch selection.", "", f"Verified counts: {counts}.", ""]
    for ds in sorted(policies):
        lines += [f"## {ds.upper()}", "", "Development-selected policies:", ""]
        for method, policy in policies[ds].items():
            lines.append(f"- {LABELS[method]}: {policy['schedule']}; development-selected reporting pass {policy['selected_epoch']}.")
        for case in ("nominal", "open_10000ppm", "gmax_10000ppm", "random_10000ppm"):
            lines += ["", f"### {case}", "", "| Source | Deployment | Joint, pass 1 | Calibration, pass 20 | Weights, pass 20 | Joint, pass 20 |", "|---|---:|---:|---:|---:|---:|"]
            for source in LABELS:
                if (ds, source, case, "none", 0) not in lookup:
                    continue
                cells = [value(ds, source, case, m, e) for m, e in (("none", 0), ("onchip_calibration", 1), ("calibration", 20), ("onchip_weights", 20), ("onchip_calibration", 20))]
                lines.append("| " + " | ".join([LABELS[source], *cells]) + " |")
            lines += ["", f"![KL and accuracy curves]({ds}_{case}.png)"]
    lines += ["", "## Scope", "", "The HWA/CDT sources are the previously selected checkpoints; several retained late-improvement flags. HWA itself was not retrained in this extension. Three-array spread does not include training-seed uncertainty. Compact checkpoints retain physical faults in P0, changing states, RNG and Adam moments. Rewrites redraw healthy programming noise. No incremental PCM pulse law, latency or energy is inferred.", "", "All paired comparisons, including losses and unchanged-target rewrite controls, are retained in paired_comparisons.csv. All-array values and uncertainty are in per_array.csv and summary.csv. Scientific review/finalization is separate from numerical completion."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    plots(summary, output)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executions", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(report(args.executions, args.output)))


if __name__ == "__main__":
    main()
