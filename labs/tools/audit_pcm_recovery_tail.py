"""Read-only audit of late PCM recovery progress, using frozen raw receipts."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import statistics

import plot_cifar_kl_fault_curves as base

GROUP = ("dataset", "depth", "source", "case", "method")
METRICS = ("train_kl", "development_kl", "test_kl", "development_accuracy_percent", "test_accuracy_percent")
METHODS = ("calibration", "rewrite", "onchip_weights", "onchip_calibration")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / "artifacts/cifar_kl_fault_curves_20260922/pcm_recovery_tail"
    manifest_path = root / "results/cifar-sweep-analysis/results.json"
    manifest = json.loads(manifest_path.read_text())
    csv_path = root / "results/cifar-sweep-analysis/per_array.csv"
    csv_rows = base.read_csv(csv_path)
    reference = {tuple(r[k] for k in GROUP) + (int(r["epoch"]), str(r["array_seed"])): r
                 for r in csv_rows if r["backend"] == "pcm" and r["method"] != "none"}
    raw, provenance = [], []
    for item in manifest["provenance"]:
        path = Path(item["path"])
        if "-pcm-" not in str(path) or "/runs/screen" not in str(path):
            continue
        base.require(base.sha256(path) == item["sha256"], f"Raw receipt hash mismatch: {path}")
        receipt = json.loads(path.read_text())
        m = receipt["metrics"]
        base.require(receipt["status"] == "complete" and m["backend"] == "pcm" and m["epochs"] == 5
                     and m["recovery_images"] == 45000 and not m["smoke_only"], f"Unexpected protocol: {path}")
        depth = re.search(r"-conv([48])-", str(path)).group(1)
        provenance.append(item)
        for row in m["measurements"]:
            if row["method"] == "none":
                continue
            base.require([r["epoch"] for r in row["curve"]] == list(range(1, 6)), "Incomplete epoch coverage.")
            for point in row["curve"]:
                epoch = point["epoch"]
                base.require(point["image_presentations"] == epoch * 45000, "Wrong image-presentation budget.")
                base.require(point["cost"]["array_reprogram_calls"] == (0 if row["method"] == "calibration" else 704 * epoch), "Wrong PCM write count.")
                base.require(point["test"]["examples"] == 10000 and point["development"]["examples"] == 1000, "Wrong evaluation cohort.")
                values = dict(dataset=m["dataset"], depth=depth, source=row["source"], case=row["case"],
                              method=row["method"], epoch=epoch, array_seed=str(row["array_seed"]),
                              train_kl=point["teacher_kl_train_mean"], development_kl=point["development"]["teacher_kl"],
                              test_kl=point["test"]["teacher_kl"],
                              development_accuracy_percent=point["development"]["accuracy_percent"],
                              test_accuracy_percent=point["test"]["accuracy_percent"])
                expected = reference[tuple(values[k] for k in GROUP) + (epoch, values["array_seed"])]
                for a, b in (("test_kl", "teacher_kl"), ("test_accuracy_percent", "accuracy_percent")):
                    base.require(math.isclose(values[a], float(expected[b]), abs_tol=1e-12, rel_tol=1e-12), "Raw receipt disagrees with plotted CSV.")
                base.require(all(math.isfinite(values[k]) for k in METRICS), "Nonfinite trajectory.")
                raw.append(values)
    base.require(len(provenance) == 12 and len(raw) == 12960, "Unexpected PCM coverage.")
    index = {tuple(r[k] for k in GROUP) + (r["epoch"], r["array_seed"]): r for r in raw}
    base.require(len(index) == len(raw), "Duplicate epoch observations.")
    groups = defaultdict(list)
    for r in raw:
        groups[tuple(r[k] for k in GROUP) + (r["epoch"],)].append(r)
    summaries = []
    for key, observations in sorted(groups.items()):
        base.require(len(observations) == 3 and {r["array_seed"] for r in observations} == base.SEEDS, "Unpaired array coverage.")
        record = dict(zip((*GROUP, "epoch"), key))
        for metric in METRICS:
            values = [r[metric] for r in observations]
            record[metric + "_mean"] = statistics.mean(values)
            record[metric + "_sd"] = statistics.stdev(values)
        summaries.append(record)
    summary = {tuple(r[k] for k in GROUP) + (r["epoch"],): r for r in summaries}
    tails, paired = [], []
    for key in sorted({k[:5] for k in groups}):
        record = dict(zip(GROUP, key))
        record["source_family"] = "cdt" if record["source"].startswith("cdt_") else record["source"]
        for metric in METRICS:
            for epoch in (1, 3, 4, 5):
                record[f"{metric}_epoch{epoch}"] = summary[key + (epoch,)][metric + "_mean"]
            sign = -1 if "accuracy" in metric else 1
            for start in (3, 4):
                changes = [sign * (index[key + (start, seed)][metric] - index[key + (5, seed)][metric]) for seed in sorted(base.SEEDS)]
                name = f"{metric}_gain_{start}_to_5"
                record[name + "_mean"] = statistics.mean(changes)
                record[name + "_sd"] = statistics.stdev(changes)
                record[name + "_positive_arrays"] = sum(v > 0 for v in changes)
        record["test_kl_monotone_3_4_5"] = record["test_kl_epoch3"] > record["test_kl_epoch4"] > record["test_kl_epoch5"]
        record["development_kl_monotone_3_4_5"] = record["development_kl_epoch3"] > record["development_kl_epoch4"] > record["development_kl_epoch5"]
        record["test_kl_reduction_percent_3_to_5"] = 100 * record["test_kl_gain_3_to_5_mean"] / record["test_kl_epoch3"]
        tails.append(record)
        for seed in sorted(base.SEEDS):
            observation = dict(zip(GROUP, key), array_seed=seed)
            for metric in METRICS:
                sign = -1 if "accuracy" in metric else 1
                for start in (3, 4):
                    observation[f"{metric}_gain_{start}_to_5"] = sign * (index[key + (start, seed)][metric] - index[key + (5, seed)][metric])
            paired.append(observation)

    advantage = []
    for key in sorted({k[:4] for k in groups}):
        for control in ("calibration", "rewrite"):
            for method in ("onchip_weights", "onchip_calibration"):
                rec = dict(zip(GROUP[:4], key), control=control, recovery=method)
                for epoch in (3, 4, 5):
                    for metric in ("test_kl", "test_accuracy_percent"):
                        sign = -1 if "accuracy" in metric else 1
                        values = [sign * (index[key + (control, epoch, seed)][metric] - index[key + (method, epoch, seed)][metric]) for seed in sorted(base.SEEDS)]
                        rec[f"{metric}_advantage_epoch{epoch}"] = statistics.mean(values)
                for metric in ("test_kl", "test_accuracy_percent"):
                    rec[f"{metric}_advantage_change_4_to_5"] = rec[f"{metric}_advantage_epoch5"] - rec[f"{metric}_advantage_epoch4"]
                advantage.append(rec)
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("epoch_per_array.csv", raw), ("epoch_summary.csv", summaries),
                           ("tail_summary.csv", tails), ("tail_per_array.csv", paired), ("recovery_advantage.csv", advantage)):
        base.write_csv(out / filename, rows)
    counts = []
    for dataset in base.DATASETS:
        for family in ("digital", "standard_hwa", "noise_hwa", "cdt"):
            rows = [r for r in tails if r["dataset"] == dataset and r["source_family"] == family and r["method"] == "onchip_calibration" and r["case"] != "nominal"]
            counts.append(dict(dataset=dataset, source_family=family, conditions=len(rows),
                test_kl_improves_4_to_5=sum(r["test_kl_gain_4_to_5_mean"] > 0 for r in rows),
                test_kl_improves_3_to_5=sum(r["test_kl_gain_3_to_5_mean"] > 0 for r in rows),
                development_kl_improves_4_to_5=sum(r["development_kl_gain_4_to_5_mean"] > 0 for r in rows),
                train_kl_improves_4_to_5=sum(r["train_kl_gain_4_to_5_mean"] > 0 for r in rows),
                all_arrays_test_kl_improve_4_to_5=sum(r["test_kl_gain_4_to_5_positive_arrays"] == 3 for r in rows),
                both_dev_and_test_improve_4_to_5=sum(r["test_kl_gain_4_to_5_mean"] > 0 and r["development_kl_gain_4_to_5_mean"] > 0 for r in rows)))
    base.write_csv(out / "tail_counts.csv", counts)
    base.plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    figures = []
    with ExitStack() as stack:
        pdfs = {ds: stack.enter_context(base.PdfPages(out / f"{ds}_pcm_recovery_epochs.pdf")) for ds in base.DATASETS}
        for dataset in base.DATASETS:
            for depth in ("4", "8"):
                fig, axes = base.plt.subplots(2, 3, figsize=(13.8, 8.2), sharex=True)
                fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.15, hspace=.26, wspace=.24)
                for col, (fault, title) in enumerate(base.FAULTS.items()):
                    for family in ("digital", "noise_hwa", "cdt"):
                        source = "cdt_" + fault if family == "cdt" else family
                        label, color, _marker, _style = base.SOURCES[family]
                        for method, style, marker in (("calibration", "--", "s"), ("onchip_calibration", "-", "o")):
                            rows = [summary[dataset, depth, source, fault + "_50000ppm", method, e] for e in range(1, 6)]
                            for row, metric in enumerate(("test_kl", "test_accuracy_percent")):
                                ax = axes[row, col]
                                ax.errorbar(range(1, 6), [r[metric + "_mean"] for r in rows],
                                    yerr=[r[metric + "_sd"] for r in rows], color=color, linestyle=style, marker=marker,
                                    linewidth=1.6, markersize=4, capsize=2, label=label + (" + calibration" if method == "calibration" else " + weight recovery + calibration"))
                    axes[0, col].set_title(title, fontsize=12)
                    axes[0, col].set_ylabel("Test teacher KL (nats)")
                    axes[1, col].set_ylabel("Test accuracy (%)")
                    axes[1, col].set_xlabel("Full recovery epochs")
                    for ax in axes[:, col]:
                        ax.set_xticks(range(1, 6)); ax.grid(alpha=.18)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="center", bbox_to_anchor=(.53, .885), ncol=3, frameon=False, fontsize=9)
                fig.suptitle(f"{base.DATASETS[dataset]} · PCM · 5% faults · {depth} analog convolutions + classifier", fontsize=16, y=.975)
                fig.text(.53, .057, "Solid: weights + calibration. Dashed: calibration only. Mean ± sample SD over three arrays; each fault panel has its own vertical range.", ha="center", fontsize=9)
                fig.text(.53, .028, "Fixed five-epoch budget; no convergence stopping rule. Full coverage of all rates, controls, training and development metrics is in the CSVs.", ha="center", fontsize=8.7)
                stem = f"{dataset}_pcm_conv{depth}_recovery_epochs_5pct"
                fig.savefig(out / f"{stem}.jpg", dpi=200, facecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
                fig.savefig(out / f"{stem}.png", dpi=150, facecolor="white")
                pdfs[dataset].savefig(fig); base.plt.close(fig); figures.append(stem)
    report = ["# PCM recovery: progress near the five-epoch limit", "",
        "Audited all twelve frozen PCM screen receipts against their recorded hashes and the existing per-array test CSV. Both datasets, four/eight analog convolutions, all six sources, nominal/1/2/3/5% cases and all four adapting controls are covered. No new training or endpoint selection was performed.",
        "Each run stopped at its fixed budget of five full 45,000-image epochs (3,520 minibatches). There was no convergence-based stopping condition. PCM writing arms redraw Gaussian programming endpoints after every minibatch; the fixed faults remain. Adam uses constant weight LR 1e-4 and calibration LR 3e-4/1e-3 for CIFAR-10/100.",
        "## Late test-KL trends for joint recovery", "",
        "Each denominator is 24 source/fault/rate/depth conditions: three fault types × four nonzero rates × two depths. These grid conditions and epochs are correlated, not independent scientific replications. Counts describe the observed grid, not significance or guaranteed convergence. CIFAR-100 PCM stuck-low CDT selected the epoch-0 teacher, so those trajectories coincide with No HWA and do not supply additional independent evidence.",
        table(["Dataset", "Source", "Mean KL falls, 4→5", "Mean KL falls, 3→5", "All three arrays improve, 4→5", "Dev and test both improve, 4→5"],
              [[base.DATASETS[r["dataset"]], base.SOURCES[r["source_family"]][0], f'{r["test_kl_improves_4_to_5"]}/24', f'{r["test_kl_improves_3_to_5"]}/24', f'{r["all_arrays_test_kl_improve_4_to_5"]}/24', f'{r["both_dev_and_test_improve_4_to_5"]}/24'] for r in counts]),
        "", "## Eight-convolution, 5% examples", ""]
    for dataset in base.DATASETS:
        rows = [r for r in tails if r["dataset"] == dataset and r["depth"] == "8" and r["method"] == "onchip_calibration" and r["case"].endswith("50000ppm") and r["source"] != "standard_hwa"]
        report.extend(["### " + base.DATASETS[dataset], "", table(["Fault", "Source", "Train KL 4→5", "Dev KL 4→5", "Test KL 3→4→5", "Test accuracy 4→5 (%)"],
            [[base.FAULTS[r["case"].split("_")[0]], base.SOURCES[r["source_family"]][0],
              f'{r["train_kl_epoch4"]:.4f} → {r["train_kl_epoch5"]:.4f}', f'{r["development_kl_epoch4"]:.4f} → {r["development_kl_epoch5"]:.4f}',
              f'{r["test_kl_epoch3"]:.4f} → {r["test_kl_epoch4"]:.4f} → {r["test_kl_epoch5"]:.4f}',
              f'{r["test_accuracy_percent_epoch4"]:.2f} → {r["test_accuracy_percent_epoch5"]:.2f}'] for r in rows]), ""])
    report.extend(["## Interpretation and next diagnostics", "",
        "CIFAR-100 shows continuing recovery progress, including sustained train/development/test KL improvement in the difficult 5% stuck-high and random-stuck cases. These five-epoch endpoints should not be described as converged performance ceilings. Gain relative to calibration also grows in the severe stuck-high examples; recovery_advantage.csv separates that from continued calibration progress.",
        "CIFAR-10 is mixed. At 5% stuck-high without HWA there is clear continuing improvement. After stronger HWA, test KL can flatten or rise while the training objective keeps falling. At 5% random faults after corruption-aware HWA, test KL rises through all five saved epochs although training KL decreases. This is consistent with overfitting or maladaptation, with stochastic endpoint noise also present; the current observations do not uniquely identify the cause. Additional epochs alone are not a reliable remedy for these cases.",
        "Training KL is the online mean over changing model states during an epoch. Development/test metrics evaluate the final programmed state. Their absolute levels are not directly comparable as a fixed-checkpoint generalization gap; the audit compares their temporal trends. Development has 1,000 adaptation-held-out images and test has 10,000. Programming realizations vary over epochs, so an isolated positive or negative last step is not a convergence test.",
        "A useful next experiment is a separately declared extension of the difficult CIFAR-100 PCM cases to 10/20 full epochs, with matched calibration and rewrite controls and development-controlled LR/stopping choices. CIFAR-10 would benefit from evaluating LR decay, development stopping and weight-only versus joint updates before assuming a longer constant-LR run will help. No extensions or hyperparameter changes were launched here.",
        "The older 20-pass PCM experiment used 5,000 images/pass (100,000 presentations total), four convolutions and 1% faults. The current five-full-epoch sweep already uses 225,000 presentations. Those historical passes are not evidence that these current trajectories have completed twenty full epochs.",
        "Input provenance and coverage are in validation.json. CSVs preserve raw epoch data, mean/sample SD, paired late changes and advantages over calibration/rewrite. This is an analysis of the saved trajectories; frozen scientific selections and pending study review are unchanged."])
    (out / "report.md").write_text("\n\n".join(report) + "\n")
    import markdown
    html = '<!doctype html><html><head><meta charset="utf-8"><title>PCM recovery tail</title><style>body{font:15px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}a{color:#2166ac}</style></head><body>'
    html += markdown.markdown((out / "report.md").read_text(), extensions=["tables"])
    for stem in figures:
        html += f'<p><a href="{stem}.jpg">JPG</a></p><img src="{stem}.png" alt="Recovery epoch trajectories">'
    (out / "index.html").write_text(html + "</body></html>")
    (out / "validation.json").write_text(json.dumps(dict(status="passed", receipt_count=len(provenance),
        verified_epoch_array_records=len(raw), mean_epoch_endpoints=len(summaries), trajectory_conditions=len(tails),
        paired_tail_records=len(paired), receipts=provenance,
        inputs={str(p): base.sha256(p) for p in (manifest_path, csv_path, Path(__file__))}, figures=figures), indent=2) + "\n")
    print(json.dumps({"output": str(out), "counts": counts, "verified_epoch_records": len(raw)}, indent=2))


if __name__ == "__main__":
    main()
