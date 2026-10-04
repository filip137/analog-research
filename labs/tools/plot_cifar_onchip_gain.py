"""Plot paired five-epoch on-chip recovery gains over calibration-only controls."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

import plot_cifar_kl_fault_curves as base


FAMILIES = ("digital", "noise_hwa", "cdt")
METRICS = ("kl_gain_nats", "kl_gain_percent", "accuracy_gain_pp")
GAIN_KEY = ("dataset", "backend", "depth", "fault_kind", "source_family", "fault_rate_percent")


def compute_gains(index, raw, source_audit):
    metadata = {(r["dataset"], r["backend"], r["depth"], r["source"]): r for r in source_audit}
    means, pairs, unique = [], [], set()
    for dataset in base.DATASETS:
        for backend, depth in base.HARDWARE:
            for fault in base.FAULTS:
                for family in FAMILIES:
                    for rate in base.RATES:
                        keys = {m: base.key_for(dataset, backend, depth, family, fault, rate, m)
                                for m in ("calibration", "onchip_calibration")}
                        matched = {m: {r["array_seed"]: r for r in raw[k]} for m, k in keys.items()}
                        base.require(all(set(v) == base.SEEDS for v in matched.values()), "Unmatched array identities.")
                        source = keys["calibration"][3]
                        record = dict(dataset=dataset, backend=backend, depth=depth, fault_kind=fault,
                                      source_family=family, source=source, fault_rate_percent=rate,
                                      fault_rate_fraction=rate / 100, case=keys["calibration"][4],
                                      control="calibration", recovery="onchip_calibration", epoch=5)
                        observations = []
                        for seed in sorted(base.SEEDS):
                            cal, rec = (matched[m][seed] for m in ("calibration", "onchip_calibration"))
                            cal_kl, rec_kl = float(cal["teacher_kl"]), float(rec["teacher_kl"])
                            cal_acc, rec_acc = float(cal["accuracy_percent"]), float(rec["accuracy_percent"])
                            base.require(cal_kl > 0, "Relative gain needs a positive calibration KL.")
                            observations.append(dict(record, array_seed=seed,
                                calibration_teacher_kl=cal_kl, recovery_teacher_kl=rec_kl,
                                calibration_accuracy_percent=cal_acc, recovery_accuracy_percent=rec_acc,
                                kl_gain_nats=cal_kl - rec_kl, kl_gain_percent=100 * (cal_kl - rec_kl) / cal_kl,
                                accuracy_gain_pp=rec_acc - cal_acc))
                        result = dict(record)
                        for metric in METRICS:
                            values = [r[metric] for r in observations]
                            base.require(all(math.isfinite(v) for v in values), f"Nonfinite gains: {record}")
                            result[metric + "_mean"] = statistics.mean(values)
                            result[metric + "_sd"] = statistics.stdev(values)
                            result[metric + "_positive_arrays"] = sum(v > 0 for v in values)
                        for metric in ("teacher_kl", "accuracy_percent"):
                            for method, label in (("calibration", "calibration"), ("onchip_calibration", "recovery")):
                                result[f"{label}_{metric}_mean"] = float(index[keys[method]][metric + "_mean"])
                        for metric, expected in (
                            ("kl_gain_nats", result["calibration_teacher_kl_mean"] - result["recovery_teacher_kl_mean"]),
                            ("accuracy_gain_pp", result["recovery_accuracy_percent_mean"] - result["calibration_accuracy_percent_mean"]),
                        ):
                            base.require(math.isclose(result[metric + "_mean"], expected, abs_tol=1e-10, rel_tol=1e-10),
                                         f"Paired gain disagrees with frozen means: {record}")
                        selected = metadata[dataset, backend, depth, source]
                        result["selected_source_epoch"] = selected["selected_epoch"]
                        result["source_convergence_review_required"] = selected["convergence_review_required"]
                        result["both_metrics_positive_arrays"] = sum(r["kl_gain_nats"] > 0 and r["accuracy_gain_pp"] > 0 for r in observations)
                        means.append(result)
                        pairs.extend(observations)
                        unique.add(keys["calibration"])
    base.require(len(means) == 360 and len(pairs) == 1080 and len(unique) == 328, "Unexpected gain coverage.")
    return means, pairs, unique


def make_figure(backend, depth, view, means, pairs):
    metric, ylabel = {
        "absolute": ("kl_gain_nats", "KL reduction (nats)"),
        "relative": ("kl_gain_percent", "KL reduction (%)"),
        "accuracy": ("accuracy_gain_pp", "Accuracy gain (percentage points)"),
    }[view]
    fig, axes = base.plt.subplots(2, 3, figsize=(13.8, 8.2), sharex=True, sharey="row")
    fig.subplots_adjust(left=.105, right=.98, bottom=.155, top=.785, hspace=.30, wspace=.12)
    for row, dataset in enumerate(base.DATASETS):
        selected = [r for r in means if (r["dataset"], r["backend"], r["depth"]) == (dataset, backend, depth)]
        lookup = {(r["fault_kind"], r["source_family"], r["fault_rate_percent"]): r for r in selected}
        per_array = base.defaultdict(list)
        for r in pairs:
            if (r["dataset"], r["backend"], r["depth"]) == (dataset, backend, depth):
                per_array[r["fault_kind"], r["source_family"], r["fault_rate_percent"]].append(r)
        bounds = [0.0]
        for r in selected:
            bounds.extend((r[metric + "_mean"] - r[metric + "_sd"], r[metric + "_mean"] + r[metric + "_sd"]))
        bounds.extend(r[metric] for group in per_array.values() for r in group)
        lo, hi = min(bounds), max(bounds)
        pad = max((hi - lo) * .12, 1e-5)
        limits = (lo - pad, hi + pad)
        for col, (fault, fault_label) in enumerate(base.FAULTS.items()):
            ax = axes[row, col]
            ax.set_ylim(*limits)
            ax.axhspan(limits[0], 0, color="#c54e4e", alpha=.045, linewidth=0)
            ax.axhline(0, color="#40464b", linestyle="--", linewidth=1, zorder=1)
            for family in FAMILIES:
                label, color, marker, style = base.SOURCES[family]
                records = [lookup[fault, family, rate] for rate in base.RATES]
                for rate in base.RATES:
                    values = [r[metric] for r in per_array[fault, family, rate]]
                    ax.scatter([rate] * 3, values, color=color, s=13, alpha=.35, linewidths=0, zorder=2)
                ax.errorbar(base.RATES, [r[metric + "_mean"] for r in records],
                            yerr=[r[metric + "_sd"] for r in records], color=color,
                            marker=marker, linestyle=style, markerfacecolor="white", markersize=5,
                            capsize=2, elinewidth=.9, linewidth=1.8, label=label, zorder=3)
            ax.set_xticks(base.RATES)
            ax.set_xlim(-.18, 5.18)
            ax.grid(alpha=.19, linewidth=.65)
            ax.tick_params(labelsize=9)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            if col == 0:
                ax.set_ylabel(ylabel, labelpad=8)
            if row == 0:
                ax.set_title(f"({chr(97 + col)})  {fault_label}", fontsize=12, pad=12)
            else:
                ax.set_xlabel("Corrupt-device ratio (%)", labelpad=7)
        box = axes[row, 0].get_position()
        fig.text(.022, (box.y0 + box.y1) / 2, base.DATASETS[dataset], rotation=90,
                 ha="center", va="center", fontsize=12, fontweight="medium")
    fig.suptitle(f"{backend.upper()}  |  Gain from on-chip weight recovery", fontsize=17, x=.53, y=.979)
    fig.text(.53, .938, f"{backend.upper()} · last {depth} convolutions + classifier analog · matched five-epoch controls", ha="center", fontsize=11, color="#48535e")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(.53, .893), ncol=3, frameon=False, fontsize=11, columnspacing=2.5)
    formula = "KL gain = KL(calibration only) − KL(weights + calibration)"
    if view == "relative":
        formula = "KL gain (%) = 100 × [KL(calibration only) − KL(weights + calibration)] / KL(calibration only), per array"
    if view == "accuracy":
        formula = "Accuracy gain = accuracy(weights + calibration) − accuracy(calibration only)"
    fig.text(.53, .848, formula, ha="center", fontsize=10)
    fig.text(.53, .070, "Above zero: recovery helped. Below zero: recovery hurt. Curves and error bars: paired mean ± sample SD across three arrays.", ha="center", fontsize=9)
    fig.text(.53, .043, "Accuracy gain = accuracy(weights + calibration) − accuracy(calibration only). Dots: individual paired gains. 4% was not run.", ha="center", fontsize=8.7, color="#48535e")
    note = "Both arms start from the same source-specific deployment; these are alternative five-epoch arms, not sequential training."
    if backend == "pcm":
        note = "CIFAR-100 PCM stuck-low corruption-aware selection returned the epoch-0 teacher; it overlaps No HWA."
    fig.text(.53, .016, note, ha="center", fontsize=8.7, color="#48535e")
    return fig


def write_gallery(out):
    content = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>On-chip recovery gains</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#21303b;max-width:1450px;margin:30px auto;padding:0 22px}h1{font-size:28px}a{color:#2166ac}p{max-width:1100px}.controls{display:flex;gap:20px;flex-wrap:wrap;background:#edf2f5;padding:18px;border-radius:8px}label{display:flex;flex-direction:column;font-size:13px;gap:5px}select{font:16px system-ui;padding:8px;background:white;border:1px solid #afbdc7;border-radius:5px}img{width:100%;height:auto}code{font-size:14px}</style></head><body>
<h1>Gain from on-chip weight recovery</h1>
<p>Compare <strong>no HWA, noisy HWA and corruption-aware HWA</strong>. The reference is five epochs of calibration alone; the recovered arm uses five epochs of physical-weight updates plus calibration, starting from the same deployment. Positive values mean improvement.</p>
<div class="controls"><label>Device model<select id="backend"><option value="pcm">PCM</option><option value="om">IBM OM</option></select></label><label>Analog convolutions + classifier<select id="depth"><option value="8">8 convolutions</option><option value="4">4 convolutions</option></select></label><label>Vertical axis<select id="view"><option value="absolute">KL reduction · absolute (nats)</option><option value="relative">KL reduction · relative (%)</option><option value="accuracy">Accuracy gain (percentage points)</option></select></label></div>
<p><a id="jpg">JPG</a> · <a id="png">PNG</a> · <a id="svg">Editable SVG</a> · <a id="pdf">PDF for selected view</a></p><img id="plot" alt="Paired recovery gains for CIFAR-10 and CIFAR-100 versus fault ratio, across three fault types">
<p><strong>Top row: CIFAR-10. Bottom row: CIFAR-100.</strong> Select KL reduction or accuracy gain using the vertical-axis selector. KL gain is KL(calibration) − KL(recovery), or that difference divided by calibration KL × 100 for each array. Accuracy gain is accuracy(recovery) − accuracy(calibration), in percentage points. Relative curves average the three per-array percentages, not a ratio of pooled means. A small baseline KL can produce a large negative percentage for a small absolute increase.</p>
<p>Lines show paired means, bars show sample SD of the paired gains, and dots show individual arrays. This is hardware variability across three arrays, not uncertainty across HWA training seeds. Zero is no change; negative results are retained. Fault panels within each row share a vertical range. Rates are 0/1/2/3/5%; 4% was not measured.</p>
<p>For CIFAR-100 PCM stuck-low faults, the selected corruption-aware source is the epoch-0 teacher, so it overlaps No HWA. Sources and endpoints retain the original development selection and convergence flags. OM uses the historical closed-loop recovery settings; separate learning-rate and open-loop follow-ups are excluded. These gains measure the full weight-update intervention over calibration; they do not separately isolate learning from rewrite/programming effects.</p>
<p><a href="gain_summary.csv">Means, paired SD and endpoints (CSV)</a> · <a href="gain_per_array.csv">Paired array data (CSV)</a> · <a href="jpg_full_paths.txt">All JPG full paths</a> · <a href="README.md">Methods and reproduction</a> · <a href="validation.json">Validation</a> · <a href="../index.html">Original KL gallery</a></p>
<script>function update(){const b=document.getElementById('backend').value,n=document.getElementById('depth').value,v=document.getElementById('view').value;const s=b+'_conv'+n+'_gain_'+v;document.getElementById('plot').src=s+'.png';for(const e of ['jpg','png','svg'])document.getElementById(e).href=s+'.'+e;document.getElementById('pdf').href=s+'.pdf';document.getElementById('plot').alt=b.toUpperCase()+' · '+n+' analog convolutions · '+v+' gain · CIFAR-10 and CIFAR-100';}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));update();</script></body></html>'''
    (out / "index.html").write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    inputs = args.input_dir or root / "results/cifar-sweep-analysis"
    audit_path = args.source_audit or root / "artifacts/cifar_recovery_full_results_20260920/source_fit_selection.csv"
    out = (args.output or root / "artifacts/cifar_kl_fault_curves_20260922/onchip_gain").resolve()
    paths = {name: inputs / name for name in ("summary.csv", "per_array.csv", "clean_sources.csv")}
    index, raw, _clean, _endpoints, _observations, _verified = base.load_data(paths)
    means, pairs, unique = compute_gains(index, raw, base.read_csv(audit_path))
    out.mkdir(parents=True, exist_ok=True)
    base.write_csv(out / "gain_summary.csv", means)
    base.write_csv(out / "gain_per_array.csv", pairs)
    base.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    figures = []
    for backend, depth in base.HARDWARE:
        for view in ("absolute", "relative", "accuracy"):
            fig = make_figure(backend, depth, view, means, pairs)
            stem = f"{backend}_conv{depth}_gain_{view}"
            fig.savefig(out / f"{stem}.png", dpi=150, facecolor="white")
            fig.savefig(out / f"{stem}.jpg", dpi=200, facecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
            fig.savefig(out / f"{stem}.svg", facecolor="white")
            fig.savefig(out / f"{stem}.pdf", facecolor="white")
            base.plt.close(fig)
            figures.append(stem)
            print(stem, flush=True)
    write_gallery(out)
    for script in (Path(__file__), Path(base.__file__)):
        destination = out / script.name
        if script.resolve() != destination.resolve():
            shutil.copyfile(script, destination)
    (out / "jpg_full_paths.txt").write_text("\n".join(str(out / f"{s}.jpg") for s in figures) + "\n")
    paths["source_fit_selection.csv"] = audit_path
    validation = dict(status="passed", plotted_gain_points=len(means), unique_gain_points=len(unique),
        plotted_paired_observations=len(pairs), unique_paired_observations=3 * len(unique),
        arrays=sorted(base.SEEDS), rates_percent=base.RATES, missing_rate_percent=[4],
        control="calibration at epoch 5", recovery="onchip_calibration at epoch 5",
        sample_sd="Computed from the three matched per-array differences, not independent endpoint SDs",
        relative_gain="Mean of 100 * (calibration_KL - recovery_KL) / calibration_KL, computed per array",
        checks=["Frozen endpoint means and SD checked against arrays", "Full matched array coverage",
                "Finite gains with positive percentage denominators", "Mean paired differences agree with endpoint differences"],
        input_files={name: dict(path=str(p.resolve()), sha256=base.sha256(p)) for name, p in paths.items()},
        generator_sha256=base.sha256(Path(__file__)), helper_sha256=base.sha256(Path(base.__file__)), figures=figures)
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (out / "README.md").write_text('''# Gain from on-chip weight recovery

Open `index.html`. Select PCM or OM, four or eight analog convolutions, and
absolute KL reduction, percentage KL reduction, or accuracy gain. Each figure
shows CIFAR-10 in the top row and CIFAR-100 in the bottom row, with three fault
columns. Download links follow the selected view in PDF, PNG, JPG and SVG.
Previously exported dataset-specific figures remain available by their old paths.

Three curves: No HWA, Noisy HWA and fault-specific Corruption-aware HWA.
Fault columns: stuck-low/open, random-stuck, stuck-high. Rates: 0/1/2/3/5%.
The classifier is analog along with the stated four/eight suffix convolutions.

The control is five full epochs of digital calibration alone. The recovery arm
is five full epochs of physical-weight recovery plus digital calibration. Both
start from the same source-specific deployment; recovery is not performed after
five prior calibration epochs. Each epoch uses all 45,000 adaptation images.
BN running statistics and the digital prefix remain frozen. All endpoints are
scored on 10,000 test images. Array identities are 251001, 251002 and 251003.

For EACH matched array:
- KL gain (nats) = KL(calibration only) - KL(weights + calibration).
- KL gain (%) = 100 * KL gain (nats) / KL(calibration only).
- Accuracy gain (percentage points) = accuracy(weights + calibration) - accuracy(calibration only).

Positive means improvement; negative means degradation. KL is measured relative
to the original digital teacher, natural logs, temperature 1. Curves average the
three paired gains and error bars show their sample SD; faint dots show arrays.
Percentage curves average per-array ratios, not a ratio of average endpoints.
Small baseline KL may yield large negative percentages even for small absolute
degradation. No clipping, pseudocount, log transform or significance claim is used.
Axes share a range across faults within each dataset row. The zero line is shown.

This estimates the incremental effect of enabling the existing physical-weight
update procedure alongside calibration. It includes its programming/rewrite effects,
and does not by itself isolate a pure learning effect from matched frozen-target
rewriting. No benefit of calibration alone is credited as the weight-recovery gain.

PCM uses Gaussian endpoint reprogramming per minibatch. OM uses historical
closed-loop pulse Adam with weight LR 1e-4 and 640-pulse/cell recovery cap. Separate
OM LR/open-loop follow-ups are excluded. The source fits retain their frozen
development selections and late-convergence flags. There is one source-training
seed, so the three arrays do not establish robustness across training seeds.
CIFAR-100 PCM stuck-low CDT selected the epoch-0 teacher and overlaps No HWA.

CSV files contain means, paired SD, signs, baseline/recovered KL and accuracy,
and source-selection metadata. There are 360 plotted comparisons (328 unique)
and 1,080 paired observations (984 unique). Generic nominal cases recur in
multiple fault panels and must not be treated as independent repetitions.
No new training was run, and this export does not finalize the study review.

Reproduce from this folder inside the complete gallery bundle:

```sh
python plot_cifar_onchip_gain.py --input-dir ../inputs --source-audit ../inputs/source_fit_selection.csv --output regenerated
```
''')
    print(json.dumps({"output": str(out), "plots": len(figures), "unique_paired_comparisons": len(unique)}))


if __name__ == "__main__":
    main()
