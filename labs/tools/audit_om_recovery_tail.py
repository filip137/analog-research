"""Audit saved OM recovery tails; keep capped, tuned and open-loop studies separate."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

import plot_cifar_kl_fault_curves as base
from audit_pcm_recovery_tail import table

SEEDS = sorted(base.SEEDS)
GROUP = ("dataset", "depth", "source", "case")
FIELDS = ("train_kl", "development_kl", "test_kl", "test_accuracy_percent")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / "artifacts/cifar_kl_fault_curves_20260922/om_recovery_tail"
    manifest_path = root / "results/cifar-sweep-analysis/results.json"
    manifest = json.loads(manifest_path.read_text())
    reference_path = root / "results/cifar-sweep-analysis/per_array.csv"
    reference = {tuple(r[k] for k in GROUP) + (r["method"], int(r["epoch"]), r["array_seed"]): r
                 for r in base.read_csv(reference_path) if r["backend"] == "om"}
    records, provenance = [], []
    for item in manifest["provenance"]:
        path = Path(item["path"])
        if "-om-" not in str(path) or "/runs/screen" not in str(path):
            continue
        base.require(base.sha256(path) == item["sha256"], f"Receipt hash mismatch: {path}")
        receipt = json.loads(path.read_text()); m = receipt["metrics"]
        base.require(receipt["status"] == "complete" and m["backend"] == "om" and m["epochs"] == 5
                     and m["recovery_images"] == 45000 and not m["smoke_only"], "Unexpected OM protocol.")
        depth = re.search(r"-conv([48])-", str(path)).group(1)
        provenance.append(item)
        for row in m["measurements"]:
            if row["method"] not in ("calibration", "onchip_calibration"):
                continue
            base.require([p["epoch"] for p in row["curve"]] == [1, 2, 3, 4, 5], "Missing recovery epoch.")
            for p in row["curve"]:
                cost = p["cost"]
                base.require(p["image_presentations"] == p["epoch"] * 45000 and p["test"]["examples"] == 10000
                             and p["development"]["examples"] == 1000, "Wrong cohort or budget.")
                base.require(cost["max_recovery_cell_pulses"] <= 640, "Original OM cap exceeded.")
                if row["method"] == "calibration":
                    base.require(cost["physical_pulses"] == 0, "Calibration changed physical weights.")
                r = dict(dataset=m["dataset"], depth=depth, source=row["source"], case=row["case"],
                         method=row["method"], epoch=p["epoch"], array_seed=str(row["array_seed"]),
                         train_kl=p["teacher_kl_train_mean"], development_kl=p["development"]["teacher_kl"],
                         test_kl=p["test"]["teacher_kl"], test_accuracy_percent=p["test"]["accuracy_percent"],
                         active_cells=row["identity"]["physical_devices"], physical_pulses=cost["physical_pulses"],
                         capped_cells=cost["capped_cells"], max_recovery_cell_pulses=cost["max_recovery_cell_pulses"])
                expected = reference[tuple(r[k] for k in GROUP) + (r["method"], r["epoch"], r["array_seed"])]
                for a, b in (("test_kl", "teacher_kl"), ("test_accuracy_percent", "accuracy_percent")):
                    base.require(math.isclose(r[a], float(expected[b]), abs_tol=1e-12, rel_tol=1e-12), "Receipt disagrees with original per-array CSV.")
                base.require(all(math.isfinite(r[k]) for k in FIELDS), "Nonfinite recovery metrics.")
                records.append(r)
    base.require(len(provenance) == 12 and len(records) == 6480, "Incomplete historical OM audit.")
    index = {tuple(r[k] for k in GROUP) + (r["method"], r["epoch"], r["array_seed"]): r for r in records}
    base.require(len(index) == len(records), "Duplicate raw observations.")
    grouped = defaultdict(list)
    for r in records:
        grouped[tuple(r[k] for k in GROUP) + (r["method"], r["epoch"])].append(r)
    means = []
    for key, rows in sorted(grouped.items()):
        base.require(len(rows) == 3 and {r["array_seed"] for r in rows} == base.SEEDS, "Unmatched arrays.")
        result = dict(zip((*GROUP, "method", "epoch"), key))
        for metric in FIELDS:
            values = [r[metric] for r in rows]
            result.update({metric + "_mean": statistics.mean(values), metric + "_sd": statistics.stdev(values),
                           metric + "_min": min(values), metric + "_max": max(values)})
        means.append(result)
    summary = {tuple(r[k] for k in GROUP) + (r["method"], r["epoch"]): r for r in means}
    tails = []
    for key in sorted({tuple(r[k] for k in GROUP) for r in records}):
        r = dict(zip(GROUP, key)); r["source_family"] = "cdt" if r["source"].startswith("cdt_") else r["source"]
        for metric in FIELDS:
            sign = -1 if "accuracy" in metric else 1
            for epoch in (3, 4, 5):
                r[f"{metric}_epoch{epoch}"] = summary[key + ("onchip_calibration", epoch)][metric + "_mean"]
            for start in (3, 4):
                gains = [sign * (index[key + ("onchip_calibration", start, s)][metric] - index[key + ("onchip_calibration", 5, s)][metric]) for s in SEEDS]
                prefix = f"{metric}_gain_{start}_to_5"
                r.update({prefix + "_mean": statistics.mean(gains), prefix + "_sd": statistics.stdev(gains), prefix + "_positive_arrays": sum(v > 0 for v in gains)})
        for epoch in (4, 5):
            cal, rec = (summary[key + (m, epoch)] for m in ("calibration", "onchip_calibration"))
            r[f"kl_advantage_over_calibration_epoch{epoch}"] = cal["test_kl_mean"] - rec["test_kl_mean"]
            r[f"accuracy_advantage_over_calibration_epoch{epoch}"] = rec["test_accuracy_percent_mean"] - cal["test_accuracy_percent_mean"]
        for metric in ("kl", "accuracy"):
            r[f"{metric}_advantage_change_4_to_5"] = r[f"{metric}_advantage_over_calibration_epoch5"] - r[f"{metric}_advantage_over_calibration_epoch4"]
        last = [index[key + ("onchip_calibration", 5, s)] for s in SEEDS]
        r["capped_cells_mean"] = statistics.mean(x["capped_cells"] for x in last)
        r["capped_cells_percent_max"] = max(100 * x["capped_cells"] / x["active_cells"] for x in last)
        r["last_epoch_pulses_mean"] = statistics.mean(index[key + ("onchip_calibration", 5, s)]["physical_pulses"] - index[key + ("onchip_calibration", 4, s)]["physical_pulses"] for s in SEEDS)
        tails.append(r)

    followups, followup_inputs = [], []
    for family, path in (("open_loop_saved_p0", root / "results/cifar-om-openloop-analysis/curves.csv"),
                         ("closed_loop_uncapped", root / "results/cifar-om-closedloop-lr-analysis/curves.csv")):
        followup_inputs.append(path); groups = defaultdict(dict)
        for row in base.read_csv(path):
            if row["method"] != "onchip_calibration":
                continue
            if family == "open_loop_saved_p0" and row["initialization"] != "saved_p0":
                continue
            if family == "closed_loop_uncapped" and (row["phase"] != "confirmation" or row["split"] != "test"):
                continue
            key = (row["dataset"], row.get("depth", row.get("convolutions")), row["source"], row["case"], row.get("learning_rate", "0.0001"))
            groups[key][int(row["epoch"]), row["array_seed"]] = row
        for key, group in sorted(groups.items()):
            base.require(set(group) == {(e, s) for e in range(1, 6) for s in SEEDS}, "Incomplete follow-up trajectory.")
            result = dict(zip((*GROUP, "learning_rate"), key), study=family)
            result["source_family"] = "cdt" if result["source"].startswith("cdt_") else result["source"]
            for metric in ("teacher_kl", "accuracy_percent"):
                for epoch in (3, 4, 5):
                    result[f"{metric}_epoch{epoch}"] = statistics.mean(float(group[epoch, s][metric]) for s in SEEDS)
                sign = -1 if metric == "accuracy_percent" else 1
                diffs = [sign * (float(group[4, s][metric]) - float(group[5, s][metric])) for s in SEEDS]
                result[metric + "_gain_4_to_5"] = statistics.mean(diffs)
                result[metric + "_gain_4_to_5_positive_arrays"] = sum(v > 0 for v in diffs)
            followups.append(result)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("epoch_per_array.csv", records), ("epoch_summary.csv", means),
                       ("tail_summary.csv", tails), ("separate_followup_tails.csv", followups)):
        base.write_csv(out / name, rows)
    counts = []
    for ds in base.DATASETS:
        for family in ("digital", "standard_hwa", "noise_hwa", "cdt"):
            rows = [r for r in tails if r["dataset"] == ds and r["source_family"] == family and r["case"] != "nominal"]
            counts.append(dict(dataset=ds, source_family=family, conditions=len(rows),
                test_kl_improves_4_to_5=sum(r["test_kl_gain_4_to_5_mean"] > 0 for r in rows),
                all_arrays_test_kl_improve_4_to_5=sum(r["test_kl_gain_4_to_5_positive_arrays"] == 3 for r in rows),
                both_dev_and_test_improve_4_to_5=sum(r["test_kl_gain_4_to_5_mean"] > 0 and r["development_kl_gain_4_to_5_mean"] > 0 for r in rows),
                kl_advantage_over_calibration_grows=sum(r["kl_advantage_change_4_to_5"] > 0 for r in rows),
                max_capped_cell_percent=max(r["capped_cells_percent_max"] for r in rows)))
    base.write_csv(out / "tail_counts.csv", counts)
    figures = []
    base.plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    for ds in base.DATASETS:
        fig, axes = base.plt.subplots(2, 3, figsize=(13.8, 8.2), sharex=True)
        fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.15, hspace=.26, wspace=.24)
        for col, (fault, label) in enumerate(base.FAULTS.items()):
            for family in ("digital", "noise_hwa", "cdt"):
                source = "cdt_" + fault if family == "cdt" else family
                title, color, _m, _s = base.SOURCES[family]
                for method, style, marker in (("calibration", "--", "s"), ("onchip_calibration", "-", "o")):
                    rows = [summary[ds, "8", source, fault + "_50000ppm", method, e] for e in range(1, 6)]
                    for row, metric in enumerate(("test_kl", "test_accuracy_percent")):
                        vals = [r[metric + "_mean"] for r in rows]
                        errors = [[r[metric + "_mean"] - r[metric + "_min"] for r in rows], [r[metric + "_max"] - r[metric + "_mean"] for r in rows]]
                        axes[row, col].errorbar(range(1, 6), vals, yerr=errors, color=color, linestyle=style, marker=marker,
                            linewidth=1.6, markersize=4, capsize=2, label=title + (" + calibration" if method == "calibration" else " + weight recovery + calibration"))
            axes[0, col].set_title(label); axes[0, col].set_yscale("log")
            axes[0, col].set_ylabel("Test teacher KL (nats, log)")
            axes[1, col].set_ylabel("Test accuracy (%)"); axes[1, col].set_xlabel("Full recovery epochs")
            for ax in axes[:, col]:
                ax.set_xticks(range(1, 6)); ax.grid(alpha=.18)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(.53, .885), ncol=3, frameon=False, fontsize=9)
        fig.suptitle(f"{base.DATASETS[ds]} · OM · 5% faults · 8 analog convolutions + classifier", fontsize=16, y=.975)
        fig.text(.53, .057, "Historical closed-loop LR 1e-4, cap 640 pulses/cell. Solid: weights + calibration. Dashed: calibration only.", ha="center", fontsize=9)
        fig.text(.53, .028, "Curves: three-array means; bars: observed array min/max. Each fault has its own vertical range. All depths and rates are in the CSV audit.", ha="center", fontsize=8.7)
        stem = f"{ds}_om_conv8_recovery_epochs_5pct"; figures.append(stem)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{stem}.{ext}", dpi=150, facecolor="white")
        fig.savefig(out / f"{stem}.jpg", dpi=200, facecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
        base.plt.close(fig)
    report = ["# OM recovery progress near the five-epoch limit", "",
        "Primary audit: original capped closed-loop sweep, joint recovery versus matched calibration. Twelve native screen receipts were hash-verified and 6,480 saved epoch-array observations checked against the original per-array test export. Both datasets, both analog depths, all sources and all rates are covered. No new training or test-based selection was performed.",
        "## Historical sweep: last-epoch trends", "",
        "Each denominator is 24 conditions: three faults × four nonzero rates × two depths. Counts are descriptive; these conditions are not independent repetitions. Sources retain their original selection/convergence limitations.",
        table(["Dataset", "Source", "Test KL improves 4→5", "All arrays improve", "Dev and test both improve", "KL advantage over calibration grows"],
              [[base.DATASETS[r["dataset"]], base.SOURCES[r["source_family"]][0], f'{r["test_kl_improves_4_to_5"]}/24', f'{r["all_arrays_test_kl_improve_4_to_5"]}/24', f'{r["both_dev_and_test_improve_4_to_5"]}/24', f'{r["kl_advantage_over_calibration_grows"]}/24'] for r in counts]),
        "", "## Eight-convolution, 5% stuck-high examples", "",
        table(["Dataset", "Source", "Test KL 4→5", "Accuracy 4→5 (%)", "KL advantage over calibration 4→5", "Largest capped fraction (%)"],
              [[base.DATASETS[r["dataset"]], base.SOURCES[r["source_family"]][0], f'{r["test_kl_epoch4"]:.4f} → {r["test_kl_epoch5"]:.4f}', f'{r["test_accuracy_percent_epoch4"]:.2f} → {r["test_accuracy_percent_epoch5"]:.2f}', f'{r["kl_advantage_over_calibration_epoch4"]:.4f} → {r["kl_advantage_over_calibration_epoch5"]:.4f}', f'{r["capped_cells_percent_max"]:.5f}']
               for r in tails if r["depth"] == "8" and r["case"] == "gmax_50000ppm" and r["source"] != "standard_hwa"]),
        "", "The original runs stopped at a fixed five-full-epoch budget, not a convergence condition. For CIFAR-100, training, development and test KL fell during the final epoch on every array in the nonzero-fault grid for the three requested source families. CIFAR-10 is more mixed, particularly noisy-HWA random faults.",
        "Continued improvement is distinct from beating calibration. The CIFAR-100/no-HWA/eight-convolution/5% stuck-high recovery endpoint is still worse than calibration and near chance, despite its large final-epoch KL reduction. On CIFAR-10 noisy-HWA/5% stuck-high, the recovering model improves, but its KL advantage over calibration shrinks during the final epoch and its final accuracy is slightly worse than calibration.",
        "The 640-pulse cap was not broadly exhausted. Among nonzero-fault conditions in the requested three source families, the largest capped fraction in any array is below 0.023%; physical pulses continue in epoch five. This does not rule out an important contribution from a small set of capped cells. Cost columns count the modeled abstract active OM cells and actual commanded pulse events; they are not measured physical-chip energy.",
        "## Separate follow-ups", "",
        "The CSV separate_followup_tails.csv keeps the uncapped LR confirmation runs and saved-P0 open-loop runs separate. Fresh RESET initialization is excluded from these matched-deployment comparisons. Follow-up endpoints come from their existing per-array analysis exports, whose hashes are recorded below; historical sweep receipt verification is separate."]
    for ds in base.DATASETS:
        for study, lr in (("closed_loop_uncapped", "0.001"), ("open_loop_saved_p0", "0.0001")):
            rows = [r for r in followups if r["dataset"] == ds and r["study"] == study and r["learning_rate"] == lr and r["case"] != "nominal"
                    and (study != "open_loop_saved_p0" or r["source"] != "standard_hwa")]
            improved = sum(r["teacher_kl_gain_4_to_5"] > 0 for r in rows)
            report.append(f"{base.DATASETS[ds]} / {study} / LR {lr}: mean test KL improves from epoch 4 to 5 in {improved}/{len(rows)} nonzero-fault conditions.")
    report.extend(["The selected uncapped closed-loop LR 1e-3 grid covers only digital and standard-HWA sources, nominal/3%/5% stuck-high and four/eight convolutions. Both datasets retain late improvements in all eight nonzero-fault mean trajectories. This does not establish the best LR for noisy or corruption-aware HWA. Historical versus new uncapped 1e-4 endpoints also include the previously documented backend-execution caveat; differences cannot all be attributed to cap removal.",
        "Open-loop behavior is substantially different: many conditions are flat or worsening by the last epoch, even though some severe stuck-high cases still improve. A blanket extension across controllers is not justified by the closed-loop trend.",
        "A useful next experiment would extend difficult closed-loop cases to 10/20 full epochs with matched calibration controls and development-controlled schedules. Controller/LR choice, physical write costs and poor final accuracy must remain visible. Three arrays do not establish robustness across HWA training seeds. No endpoint is selected retrospectively from these test trajectories.",
        "Training KL is an online epoch mean over changing model states; development/test KL evaluates the final state. Compare their temporal trends rather than treating their absolute levels as an exact fixed-checkpoint generalization gap. All test KL uses the original teacher, natural logs and temperature 1. Scientific review and finalization are unchanged."])
    (out / "report.md").write_text("\n\n".join(report) + "\n")
    import markdown
    doc = '<!doctype html><html><head><meta charset="utf-8"><title>OM recovery tail</title><style>body{font:15px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}a{color:#2166ac}</style></head><body>'
    doc += markdown.markdown((out / "report.md").read_text(), extensions=["tables"])
    for stem in figures:
        doc += f'<p><a href="{stem}.jpg">JPG</a> · <a href="{stem}.pdf">PDF</a></p><img src="{stem}.png" alt="OM recovery epoch curves">'
    (out / "index.html").write_text(doc + "</body></html>")
    inputs = [manifest_path, reference_path, *followup_inputs, Path(__file__)]
    (out / "validation.json").write_text(json.dumps(dict(status="passed", receipt_count=len(provenance), verified_epoch_records=len(records),
        trajectory_conditions=len(tails), followup_conditions=len(followups), receipts=provenance,
        inputs={str(p): base.sha256(p) for p in inputs}, figures=figures), indent=2) + "\n")
    print(json.dumps({"output": str(out), "counts": counts, "followup_conditions": len(followups)}, indent=2))


if __name__ == "__main__":
    main()
