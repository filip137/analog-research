"""Read-only scientific comparison of the completed CIFAR fault sweep.

All endpoints and source choices are frozen by the historical protocol.
Accuracy tolerances below are descriptive sensitivity analyses, not new
checkpoint-selection rules or retrospective significance criteria.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import markdown

DATASETS = ("cifar10", "cifar100")
KINDS = {"open": "Stuck-low", "random": "Random-stuck", "gmax": "Stuck-high"}
HARDWARE = (("pcm", "4"), ("pcm", "8"), ("om", "4"), ("om", "8"))
RATES = (1, 2, 3, 5)
SOURCE_LABELS = {"digital": "No HWA", "standard_hwa": "Normal HWA", "noise_hwa": "Noise HWA", "cdt": "Corruption-aware HWA"}
METHODS = ("none", "calibration", "rewrite", "onchip_weights", "onchip_calibration")
COLORS = {"standard_hwa": "#5b6573", "noise_hwa": "#147d92", "cdt": "#cb692b"}

INTERPRETATION = {
    ("cifar10", "open"): [
        "PCM is already robust: normal HWA plus calibration stays within 0.38 accuracy points of the clean teacher at every measured rate and both depths. Even direct deployment remains within one point at 5%. Corruption-aware training adds little, and PCM recovery often worsens teacher KL.",
        "OM is different. With four convolutions, normal HWA plus calibration stays within one point through 3%; at 5%, corruption-aware HWA gives 92.71 / 0.0694. With eight convolutions, noise HWA is stronger than the fault-specific source at every measured rate. At 5%, noise HWA plus calibration is 90.47 / 0.1883 and joint recovery is 90.80 / 0.1794: a modest improvement, with a substantial remaining clean-model gap.",
    ],
    ("cifar10", "random"): [
        "Four-convolution PCM and OM need little beyond normal HWA plus calibration across 1/2/3/5%; no-HWA plus calibration also remains within one accuracy point. At eight convolutions, stronger offline robustness matters at higher rates: noise HWA or corruption-aware HWA preserves more accuracy than normal HWA.",
        "Fault-specific training is not uniformly the strongest option. At 5%, PCM/eight-convolution corruption-aware calibration reaches 91.98 / 0.1318, compared with noise HWA at 91.65 / 0.1497. On OM/eight, noise HWA is stronger: 91.58 / 0.1379 versus fault-specific HWA at 91.05 / 0.1551. Recovery after the stronger sources has small accuracy effects; PCM fault-specific recovery at 5% raises accuracy to 92.18 but worsens KL to 0.1505.",
    ],
    ("cifar10", "gmax"): [
        "Normal HWA plus calibration is near clean for four-convolution deployments at 1–2%, but loses ground at 3–5%. At 5%, corruption-aware calibration restores PCM/four to 92.69 / 0.0692 and OM/four to 92.64 / 0.0763, both within one accuracy point of clean. Their corresponding joint recoveries are 92.78 / 0.0600 and 92.66 / 0.0751; the incremental accuracy gains are small.",
        "Eight-convolution cases remain difficult. At 5%, PCM corruption-aware calibration gives 90.42 / 0.1878 and recovery 90.82 / 0.1812. OM fault-specific calibration is weaker than noise HWA (84.18 / 0.3683 versus 86.80 / 0.3643). Its recovered endpoint is 87.76 / 0.2573: a 0.96-point advantage over the noise-HWA calibration control, rather than the much larger within-source gain. At 1–3%, OM noise HWA plus calibration already beats fault-specific joint recovery in both metrics.",
    ],
    ("cifar100", "open"): [
        "PCM remains the mild case. Normal HWA plus calibration is within two accuracy points of clean at all measured rates with four convolutions and through 3% with eight. Noise HWA reaches 69.14 / 0.2046 and 69.43 / 0.3802 at 5% for four/eight convolutions. Its better accuracy comes with worse teacher KL than normal HWA (68.28 / 0.1409 and 67.44 / 0.2737). The selected PCM stuck-low CDT checkpoints are the original teacher (epoch zero), so they provide no learned source improvement.",
        "OM benefits substantially from weight recovery at every measured rate and both depths. At 5%, noise HWA plus calibration changes from 57.19 / 0.8837 to 62.15 / 0.6617 with four convolutions, and from 47.56 / 1.3766 to 54.49 / 1.1075 with eight. These same-source improvements beat both calibration and rewrite controls on all three arrays. Fault-specific HWA is weaker than noise HWA here; recovery does not restore near-clean accuracy.",
    ],
    ("cifar100", "random"): [
        "Normal HWA plus calibration loses more than two accuracy points even at 1% in all four hardware/depth configurations. Noise HWA is a useful offline improvement for PCM: at 1%, four/eight convolutions give 68.89 / 0.1987 and 69.23 / 0.3902. At higher rates, noise-HWA recovery adds increasing accuracy and improves KL. Fault-specific HWA is not consistently better; at 5% its small PCM accuracy advantage over noise HWA does not improve KL.",
        "On OM, normal HWA is usually stronger than the fault-specific source, particularly at 1–3%. Physical-weight recovery is useful without first moving to CDT. At 5%, normal-HWA recovery gives 65.30 / 0.4210 (four convolutions) and 60.58 / 0.7689 (eight), compared with calibration alone at 60.91 / 0.6296 and 51.55 / 1.1689. Both accuracy and KL improve against calibration and rewrite on all three arrays. Severe cases still miss the clean-model accuracy substantially.",
    ],
    ("cifar100", "gmax"): [
        "This is the strongest recovery case. No calibrated HWA source is within two accuracy points of clean at any measured rate, device, or depth. Noise HWA strongly improves PCM over normal HWA. Fault-specific PCM HWA becomes competitive at the higher rates, but the eight-convolution noise-HWA recovery remains stronger than CDT recovery at every measured rate.",
        "At 5%, matched noise-HWA calibration to joint recovery improves PCM/four from 56.65 / 0.9010 to 63.90 / 0.5642, PCM/eight from 48.16 / 1.3738 to 59.65 / 0.8775, OM/four from 56.62 / 0.9024 to 60.00 / 0.7778, and OM/eight from 42.18 / 1.5755 to 46.76 / 1.4063. All four comparisons improve both metrics against calibration and rewrite on every array. CDT recovery reaches 60.33 / 0.7478 and 47.98 / 1.3690 on OM/four and OM/eight; compare these with the stronger noise-HWA controls rather than only the weak CDT starts. The residual error remains large.",
    ],
}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair(row):
    return f"{float(row['accuracy_percent_mean']):.2f} / {float(row['teacher_kl_mean']):.4f}"


def name(dataset):
    return dataset.upper().replace("CIFAR", "CIFAR-")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / "artifacts/cifar_fault_regime_analysis_20260920"
    out.mkdir(parents=True, exist_ok=True)
    paths = {n: root / "results/cifar-sweep-analysis" / n for n in ("summary.csv", "per_array.csv", "clean_sources.csv", "paired_summary.csv")}
    summary = read_csv(paths["summary.csv"])
    keys = ("dataset", "backend", "depth", "source", "case", "method", "epoch")
    index = {tuple(r[k] for k in keys): r for r in summary}
    assert len(index) == len(summary)
    raw = defaultdict(list)
    for row in read_csv(paths["per_array.csv"]):
        raw[tuple(row[k] for k in keys)].append(row)
    clean = {}
    for dataset in DATASETS:
        teachers = [r for r in read_csv(paths["clean_sources.csv"]) if r["dataset"] == dataset and r["source"] == "digital"]
        assert len({r["accuracy_percent"] for r in teachers}) == 1
        clean[dataset] = float(teachers[0]["accuracy_percent"])

    def get(dataset, backend, depth, source, kind, rate, method="calibration"):
        case = f"{kind}_{rate * 10000}ppm" if rate else "nominal"
        return index[(dataset, backend, depth, source, case, method, "0" if method == "none" else "5")]

    endpoints, gains, acceptance, comparisons, ranking = [], [], [], [], []
    verified = set()
    for dataset in DATASETS:
        for backend, depth in HARDWARE:
            for rate in RATES:
                faults = [(kind, get(dataset, backend, depth, "digital", kind, rate, "none")) for kind in KINDS]
                acc_order = [k for k, r in sorted(faults, key=lambda item: float(item[1]["accuracy_percent_mean"]))]
                kl_order = [k for k, r in sorted(faults, key=lambda item: float(item[1]["teacher_kl_mean"]), reverse=True)]
                ranking.append({"dataset": dataset, "backend": backend, "depth": depth, "rate_percent": rate,
                                "damage_order_accuracy": ">".join(acc_order), "damage_order_teacher_kl": ">".join(kl_order)})
            for kind in KINDS:
                for rate in (0, *RATES):
                    for source in ("digital", "standard_hwa", "noise_hwa", "cdt_" + kind):
                        for method in METHODS:
                            row = get(dataset, backend, depth, source, kind, rate, method)
                            key = tuple(row[k] for k in keys)
                            observations = raw[key]
                            assert len(observations) == 3
                            assert {r["array_seed"] for r in observations} == {"251001", "251002", "251003"}
                            for metric in ("accuracy_percent", "teacher_kl"):
                                values = [float(r[metric]) for r in observations]
                                for stat, fn in (("mean", statistics.mean), ("sd", statistics.stdev)):
                                    actual = float(row[metric + "_" + stat])
                                    assert abs(actual - fn(values)) < 1e-8 * max(1, abs(actual))
                            verified.add(key)
                            endpoints.append(dict(row, fault_kind=kind, fault_rate_percent=rate,
                                                  clean_accuracy_percent=clean[dataset],
                                                  accuracy_loss_from_clean_pp=clean[dataset] - float(row["accuracy_percent_mean"])))
                            if rate and method in ("none", "calibration", "onchip_calibration"):
                                for tolerance in (1, 2):
                                    acceptance.append({"dataset": dataset, "backend": backend, "depth": depth, "fault_kind": kind,
                                                       "fault_rate_percent": rate, "source": source, "method": method,
                                                       "accuracy_tolerance_pp": tolerance, "accuracy_target_percent": clean[dataset] - tolerance,
                                                       "mean_meets_target": float(row["accuracy_percent_mean"]) >= clean[dataset] - tolerance,
                                                       "arrays_meeting_target": sum(float(r["accuracy_percent"]) >= clean[dataset] - tolerance for r in observations),
                                                       "accuracy_percent_mean": row["accuracy_percent_mean"], "teacher_kl_mean": row["teacher_kl_mean"]})
                        if rate:
                            joint = get(dataset, backend, depth, source, kind, rate, "onchip_calibration")
                            joint_raw = {r["array_seed"]: r for r in raw[tuple(joint[k] for k in keys)]}
                            for control in ("calibration", "rewrite"):
                                reference = get(dataset, backend, depth, source, kind, rate, control)
                                reference_raw = {r["array_seed"]: r for r in raw[tuple(reference[k] for k in keys)]}
                                da = [float(joint_raw[s]["accuracy_percent"]) - float(reference_raw[s]["accuracy_percent"]) for s in joint_raw]
                                dk = [float(reference_raw[s]["teacher_kl"]) - float(joint_raw[s]["teacher_kl"]) for s in joint_raw]
                                gains.append({"dataset": dataset, "backend": backend, "depth": depth, "fault_kind": kind,
                                              "fault_rate_percent": rate, "source": source, "control": control,
                                              "accuracy_gain_pp_mean": statistics.mean(da), "accuracy_gain_pp_sd": statistics.stdev(da),
                                              "teacher_kl_reduction_mean": statistics.mean(dk), "teacher_kl_reduction_sd": statistics.stdev(dk),
                                              "accuracy_positive_arrays": sum(d > 0 for d in da), "kl_positive_arrays": sum(d > 0 for d in dk),
                                              "both_positive_arrays": sum(a > 0 and k > 0 for a, k in zip(da, dk)),
                                              "control_accuracy_percent_mean": reference["accuracy_percent_mean"], "control_teacher_kl_mean": reference["teacher_kl_mean"],
                                              "recovered_accuracy_percent_mean": joint["accuracy_percent_mean"], "recovered_teacher_kl_mean": joint["teacher_kl_mean"]})
                    if rate:
                        cdt = get(dataset, backend, depth, "cdt_" + kind, kind, rate)
                        for source in ("standard_hwa", "noise_hwa"):
                            ref = get(dataset, backend, depth, source, kind, rate)
                            comparisons.append({"dataset": dataset, "backend": backend, "depth": depth, "fault_kind": kind,
                                                "fault_rate_percent": rate, "control_source": source,
                                                "cdt_accuracy_gain_pp": float(cdt["accuracy_percent_mean"]) - float(ref["accuracy_percent_mean"]),
                                                "cdt_teacher_kl_reduction": float(ref["teacher_kl_mean"]) - float(cdt["teacher_kl_mean"]),
                                                "cdt_accuracy_percent_mean": cdt["accuracy_percent_mean"], "cdt_teacher_kl_mean": cdt["teacher_kl_mean"],
                                                "control_accuracy_percent_mean": ref["accuracy_percent_mean"], "control_teacher_kl_mean": ref["teacher_kl_mean"]})
    assert len(endpoints) == 2400 and len(gains) == 768 and len(acceptance) == 2304
    assert all(r["damage_order_accuracy"] == ("gmax>random>open" if r["backend"] == "pcm" else "gmax>open>random") for r in ranking)
    assert all(r["damage_order_teacher_kl"].startswith("gmax>") for r in ranking)
    for filename, rows in (("endpoints.csv", endpoints), ("paired_recovery_gains.csv", gains), ("accuracy_target_sensitivity.csv", acceptance),
                           ("cdt_vs_generic_hwa.csv", comparisons), ("direct_fault_ranking.csv", ranking)):
        write_csv(out / filename, rows)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    with PdfPages(out / "all_fault_comparisons.pdf") as pdf:
        for dataset in DATASETS:
            for kind, label in KINDS.items():
                fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
                for col, (backend, depth) in enumerate(HARDWARE):
                    for source in ("standard_hwa", "noise_hwa", "cdt_" + kind):
                        family = "cdt" if source.startswith("cdt_") else source
                        for method, style, marker in (("calibration", "-", "o"), ("onchip_calibration", "--", "s")):
                            vals = [get(dataset, backend, depth, source, kind, r, method) for r in RATES]
                            label_line = SOURCE_LABELS[family] + (" + calibration" if method == "calibration" else " + recovery + calibration")
                            for row, metric in enumerate(("accuracy_percent", "teacher_kl")):
                                axes[row, col].errorbar(RATES, [float(v[metric + "_mean"]) for v in vals],
                                                       yerr=[float(v[metric + "_sd"]) for v in vals],
                                                       color=COLORS[family], linestyle=style, marker=marker, markersize=4,
                                                       capsize=2, linewidth=1.6, label=label_line)
                    axes[0, col].axhline(clean[dataset], color="#8b9398", lw=1, linestyle=":")
                    axes[0, col].set_title(f"{backend.upper()} · {depth} analog convolutions")
                    axes[0, col].set_ylabel("Test accuracy (%)")
                    axes[1, col].set_ylabel("Teacher KL (nats, log scale)")
                    axes[1, col].set_yscale("log")
                    for ax in axes[:, col]:
                        ax.set_xticks(RATES)
                        ax.set_xlabel("Stuck-device rate (%)")
                        ax.grid(alpha=.18)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .026), frameon=False)
                fig.suptitle(f"{name(dataset)} · {label}: matched calibration and weight recovery", fontsize=15, y=.98)
                fig.text(.5, .004, "Mean ± sample SD over three arrays; fixed five-epoch budget. Dotted line: clean accuracy. 4% was not measured. Classifier is also analog.", ha="center", fontsize=9)
                fig.tight_layout(rect=(0, .125, 1, .945))
                stem = f"{dataset}_{kind}"
                fig.savefig(out / f"{stem}.png", dpi=150)
                fig.savefig(out / f"{stem}.svg")
                pdf.savefig(fig)
                plt.close(fig)

    common = [
        "All numeric pairs are **test accuracy (%) / KL(original teacher || student)** in nats at T=1, averaged over three independently drawn modeled arrays. Accuracy increases are good; KL reductions are good. Depth is the number of analog suffix convolutions; the classifier is also analog. Rates are 1/2/3/5%; **4% was not run**.",
        "Normal HWA uses the selected nominal-noise source. Noise HWA is a separate source trained with stronger healthy programming noise and selected on a mixed stress set. Corruption-aware HWA (CDT) trains with the named stuck-device type at mixed rates. Calibration adapts digital gains, suffix BN affine parameters and classifier bias for five full 45,000-image epochs, with physical weights fixed. Joint recovery adds physical-weight updates under the same calibration budget.",
        "The one- and two-percentage-point accuracy bands below are transparent, retrospective sensitivity guides. They are **not predeclared success tests**, do not impose a KL cutoff, and do not select checkpoints or a deployment policy. A target crossed only by the mean may not be met by every array; the CSV includes the per-array pass count. A useful recovery gain does not necessarily restore acceptable accuracy.",
        "Matched recovery gains are computed against both digital calibration and frozen-target rewrite plus calibration on the same source and array. Cross-source comparisons describe complete pipelines and do not isolate a controller effect. These exploratory runs use one training seed per HWA source; three arrays do not establish training-seed robustness or statistical significance. 31/40 selected HWA fits retain a late-convergence flag. PCM recovery uses 3,520 full Gaussian endpoint reprograms; OM uses the historical closed-loop LR 1e-4 and 640-pulse/cell recovery cap. Separate LR/open-loop studies are not pooled into these tables.",
    ]
    css = "body{font:15px/1.55 system-ui,sans-serif;color:#203039;max-width:1500px;margin:32px auto;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{padding:8px;border-bottom:1px solid #dce3e7;text-align:right}th{background:#eef3f5}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}h2{margin-top:44px}h3{margin-top:28px}p{max-width:1150px}a{color:#12697c}@media print{body{font-size:11px}table{font-size:9px}img{break-inside:avoid}}"
    for dataset in DATASETS:
        parts = [f"# {name(dataset)}: fault damage, HWA and recovery", f"Clean digital teacher: **{clean[dataset]:.2f}% / approximately 0 KL**.", *common]
        parts += ["## Which corruption does most damage?",
                  "At direct deployment, stuck-high is worst in accuracy and KL in every matched device/depth/rate case. The accuracy-loss ordering is stuck-high > random-stuck > stuck-low for PCM, and stuck-high > stuck-low > random-stuck for OM. The latter two faults can swap order in teacher KL: CIFAR-100 OM/eight at 5% has lower stuck-low accuracy but higher random-stuck KL. These rankings concern the unadapted digital source; adaptation can change the ordering. The following table shows the 5% endpoint."]
        parts.append(table(["Fault", "PCM · 4 convs", "PCM · 8 convs", "OM · 4 convs", "OM · 8 convs"],
                           [[label] + [pair(get(dataset, dev, dep, "digital", kind, 5, "none")) for dev, dep in HARDWARE] for kind, label in KINDS.items()]))
        parts.append("The definitions differ across devices: PCM stuck-low removes a differential-bank conductance, whereas OM stuck-low fixes its active state at its native lower bound while retaining the intrinsic reference. The same percentage is not an identical logical-weight perturbation across PCM and OM.")
        parts += ["## HWA before calibration versus HWA plus calibration",
                  "This distinction matters: HWA is offline source training, while calibration is extra adaptation after deployment. The following counts show how many of the 16 device/depth/rate conditions per fault are within one accuracy point of clean, before / after calibration. These are accuracy-only screening counts; every corresponding accuracy and KL endpoint is retained in the CSV. The detailed analysis below always labels calibrated baselines explicitly."]
        calibration_counts = []
        for kind, label in KINDS.items():
            cells = [label]
            for source in ("standard_hwa", "noise_hwa", "cdt_" + kind):
                counts = [sum(float(get(dataset, dev, dep, source, kind, rate, method)["accuracy_percent_mean"]) >= clean[dataset] - 1
                              for dev, dep in HARDWARE for rate in RATES) for method in ("none", "calibration")]
                cells.append(f"{counts[0]}/16 → {counts[1]}/16")
            calibration_counts.append(cells)
        parts.append(table(["Fault", "Normal HWA", "Noise HWA", "Corruption-aware HWA"], calibration_counts))
        for kind, label in KINDS.items():
            parts += ["## " + label, *INTERPRETATION[(dataset, kind)],
                      f"![{name(dataset)} {label}: accuracy and KL across rates]({dataset}_{kind}.svg)",
                      "### When is calibrated HWA near clean?",
                      "Entries list measured fault rates meeting the mean-accuracy target. Each entry is **within 1 pp / within 2 pp** of clean. A dash means none. Consult the endpoint table for the accompanying KL."]
            band_rows = []
            for dev, dep in HARDWARE:
                cells = [f"{dev.upper()} · {dep} convs"]
                for source in ("standard_hwa", "noise_hwa", "cdt_" + kind):
                    passes = []
                    for tol in (1, 2):
                        rs = [str(r) + "%" for r in RATES if float(get(dataset, dev, dep, source, kind, r)["accuracy_percent_mean"]) >= clean[dataset] - tol]
                        passes.append(", ".join(rs) if rs else "—")
                    cells.append(" / ".join(passes))
                band_rows.append(cells)
            parts.append(table(["Hardware", "Normal HWA + calibration", "Noise HWA + calibration", "CDT + calibration"], band_rows))
            parts += ["### All measured rates: before and after physical-weight recovery",
                      "Cal = digital calibration. Rec = physical-weight recovery plus digital calibration. All adaptation endpoints use epoch five; sources were frozen from development selection."]
            parts.append(table(["Hardware", "Rate", "Normal + cal", "Normal + rec", "Noise + cal", "Noise + rec", "CDT + cal", "CDT + rec"],
                               [[f"{dev.upper()} · {dep}", f"{r}%"] + [pair(get(dataset, dev, dep, src, kind, r, method))
                                for src in ("standard_hwa", "noise_hwa", "cdt_" + kind) for method in ("calibration", "onchip_calibration")]
                                for dev, dep in HARDWARE for r in RATES]))
            parts += ["### Five-percent paired recovery gains against both controls",
                      "Each gain is **accuracy percentage points / KL reduction**; positive is favorable. The counts show how many of the three arrays improved both metrics. These counts are descriptive, not significance tests."]
            gain_rows = []
            for dev, dep in HARDWARE:
                for src in ("standard_hwa", "noise_hwa", "cdt_" + kind):
                    rows = [g for g in gains if (g["dataset"], g["backend"], g["depth"], g["fault_kind"], g["fault_rate_percent"], g["source"]) == (dataset, dev, dep, kind, 5, src)]
                    cells = [f"{dev.upper()} · {dep}", SOURCE_LABELS["cdt" if src.startswith("cdt_") else src]]
                    for control in ("calibration", "rewrite"):
                        g = next(g for g in rows if g["control"] == control)
                        cells += [f"{g['accuracy_gain_pp_mean']:+.2f} / {g['teacher_kl_reduction_mean']:+.4f}", str(g["both_positive_arrays"]) + "/3"]
                    gain_rows.append(cells)
            parts.append(table(["Hardware", "Source", "Gain vs calibration", "Both improve", "Gain vs rewrite + calibration", "Both improve"], gain_rows))
        parts += ["## Evidence and status",
                  "The complete sweep's eight V2 studies passed full artifact verification and were ready for review in the saved completion audit. This report is the assistant's requested analysis; it does not finalize the user's scientific review or alter managed manifests. No new run or checkpoint selection was performed.",
                  "Files: [all endpoints and SDs](endpoints.csv), [paired recovery gains](paired_recovery_gains.csv), [CDT versus generic HWA](cdt_vs_generic_hwa.csv), [accuracy-target sensitivity including array counts](accuracy_target_sensitivity.csv), [direct-deployment fault rankings](direct_fault_ranking.csv), [six-page figure PDF](all_fault_comparisons.pdf)."]
        report = "\n\n".join(parts) + "\n"
        (out / f"{dataset}_analysis.md").write_text(report)
        html = markdown.markdown(report, extensions=["tables", "toc"])
        (out / f"{dataset}_analysis.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name(dataset)} fault analysis</title><style>{css}</style><body>{html}</body></html>')
    manifest = {"analysis": "Descriptive analysis; original source selection and checkpoints unchanged", "measured_fault_rates_percent": list(RATES),
                "missing_rate_percent": 4, "fault_conditions": 96, "endpoint_rows_including_nominal_references": len(endpoints),
                "distinct_verified_endpoint_keys": len(verified), "paired_comparisons": len(gains), "accuracy_sensitivity_rows": len(acceptance),
                "direct_ranking_comparisons": len(ranking), "accuracy_and_kl_fault_order_disagreements": [r for r in ranking if r["damage_order_accuracy"] != r["damage_order_teacher_kl"]],
                "clean_accuracy_percent": clean, "sources": {}}
    for filename, path in paths.items():
        manifest["sources"][filename] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(out)


if __name__ == "__main__":
    main()
