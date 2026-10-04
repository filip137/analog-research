"""Retrospective best observed calibrated source and its matched recovery.

Ranking uses mean teacher KL over the three fixed test arrays, independently at
each plotted condition. It summarizes frozen results; it is not checkpoint
selection or a prospectively validated deployment policy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

import plot_cifar_kl_fault_curves as base


FAMILIES = tuple(base.SOURCES)
STAGES = ("calibration", "matched_recovery", "best_recovery")
VIEWS = {"kl_log": ("teacher_kl", "log"), "kl_linear": ("teacher_kl", "linear"),
         "accuracy": ("accuracy_percent", "linear")}


def summarize(index, raw, audit):
    chosen, paired, candidates = [], [], []
    audit_index = {(r["dataset"], r["backend"], r["depth"], r["source"]): r for r in audit}
    for dataset in base.DATASETS:
        for backend, depth in base.HARDWARE:
            for fault in base.FAULTS:
                for rate in base.RATES:
                    context = dict(dataset=dataset, backend=backend, depth=depth, fault_kind=fault,
                                   fault_rate_percent=rate, fault_rate_fraction=rate / 100)
                    keys = {(family, method): base.key_for(dataset, backend, depth, family, fault, rate, method)
                            for family in FAMILIES for method in ("calibration", "onchip_calibration")}
                    def score(family, method):
                        return float(index[keys[family, method]]["teacher_kl_mean"])
                    best_cal = min(FAMILIES, key=lambda family: score(family, "calibration"))
                    best_rec = min(FAMILIES, key=lambda family: score(family, "onchip_calibration"))
                    for family in FAMILIES:
                        for method in ("calibration", "onchip_calibration"):
                            endpoint = index[keys[family, method]]
                            candidates.append(dict(endpoint, **context, source_family=family,
                                is_calibration_winner=method == "calibration" and family == best_cal,
                                is_recovery_winner=method == "onchip_calibration" and family == best_rec))
                    stage_keys = {"calibration": keys[best_cal, "calibration"],
                                  "matched_recovery": keys[best_cal, "onchip_calibration"],
                                  "best_recovery": keys[best_rec, "onchip_calibration"]}
                    record = dict(context, calibration_family=best_cal, recovery_family=best_rec,
                                  source_switch=best_cal != best_rec)
                    stage_arrays = {}
                    for stage, key in stage_keys.items():
                        endpoint = index[key]
                        record[stage + "_source"] = endpoint["source"]
                        for metric in ("teacher_kl", "accuracy_percent"):
                            for stat in ("mean", "sd"):
                                record[f"{stage}_{metric}_{stat}"] = float(endpoint[f"{metric}_{stat}"])
                        stage_arrays[stage] = {r["array_seed"]: r for r in raw[key]}
                        base.require(set(stage_arrays[stage]) == base.SEEDS, "Expected matched declared arrays.")
                        metadata = audit_index[dataset, backend, depth, endpoint["source"]]
                        record[stage + "_selected_source_epoch"] = metadata["selected_epoch"]
                        record[stage + "_convergence_review_required"] = metadata["convergence_review_required"]
                    for stage, winner, method in (("calibration", best_cal, "calibration"), ("best_recovery", best_rec, "onchip_calibration")):
                        winning_score = score(winner, method)
                        ties = [f for f in FAMILIES if score(f, method) == winning_score]
                        record[stage + "_tied_families"] = "|".join(ties)
                        ordered = sorted(score(f, method) for f in FAMILIES)
                        record[stage + "_runner_up_kl_margin"] = ordered[1] - ordered[0]
                        base.require(winning_score <= min(score(f, method) for f in FAMILIES), "Incorrect KL winner.")
                    observations = []
                    for seed in sorted(base.SEEDS):
                        row = dict(context, array_seed=seed, calibration_family=best_cal, recovery_family=best_rec)
                        for stage in STAGES:
                            for metric in ("teacher_kl", "accuracy_percent"):
                                row[stage + "_" + metric] = float(stage_arrays[stage][seed][metric])
                        for label, stage in (("matched", "matched_recovery"), ("best_available", "best_recovery")):
                            delta = row["calibration_teacher_kl"] - row[stage + "_teacher_kl"]
                            row[label + "_kl_gain_nats"] = delta
                            row[label + "_kl_gain_percent"] = 100 * delta / row["calibration_teacher_kl"]
                            row[label + "_accuracy_gain_pp"] = row[stage + "_accuracy_percent"] - row["calibration_accuracy_percent"]
                        observations.append(row)
                    for label in ("matched", "best_available"):
                        for metric in ("kl_gain_nats", "kl_gain_percent", "accuracy_gain_pp"):
                            values = [r[label + "_" + metric] for r in observations]
                            base.require(all(math.isfinite(v) for v in values), "Expected finite gains.")
                            record[f"{label}_{metric}_mean"] = statistics.mean(values)
                            record[f"{label}_{metric}_sd"] = statistics.stdev(values)
                        record[label + "_both_metrics_positive_arrays"] = sum(
                            r[label + "_kl_gain_nats"] > 0 and r[label + "_accuracy_gain_pp"] > 0 for r in observations)
                    base.require(record["best_recovery_teacher_kl_mean"] <= record["matched_recovery_teacher_kl_mean"],
                                 "Best recovered KL cannot exceed matched recovered KL.")
                    for label, stage in (("matched", "matched_recovery"), ("best_available", "best_recovery")):
                        expected = record["calibration_teacher_kl_mean"] - record[stage + "_teacher_kl_mean"]
                        base.require(math.isclose(record[label + "_kl_gain_nats_mean"], expected, rel_tol=1e-10, abs_tol=1e-10),
                                     "Paired KL gain disagrees with endpoint means.")
                    chosen.append(record)
                    paired.extend(observations)
    base.require(len(chosen) == 120 and len(paired) == 360 and len(candidates) == 960, "Unexpected coverage.")
    return chosen, paired, candidates


def figure(dataset, view, records, clean, *, device=None, analog_depth=None):
    metric, scale = VIEWS[view]
    rows = ([(d, device, depth) for d in ([dataset] if dataset else base.DATASETS) for depth in ("4", "8")]
            if device else [(dataset, backend, depth) for backend, depth in base.HARDWARE])
    title = (f"{base.DATASETS[dataset]} · {device.upper()}" if device and dataset
             else device.upper() if device else base.DATASETS[dataset])
    if analog_depth is not None:
        rows = [row for row in rows if row[2] == analog_depth]
        title += f" · {analog_depth} analog convolutions"
    compact = len(rows) == 2
    fig, axes = base.plt.subplots(len(rows), 3, figsize=(15.4, 8.8 if compact else 14.6), sharex=True, sharey="row")
    fig.subplots_adjust(left=.105, right=.985, top=.75 if compact else .828,
                        bottom=.18 if compact else .13, hspace=.57, wspace=.14)
    for row, (row_dataset, backend, depth) in enumerate(rows):
        subset = [r for r in records if (r["dataset"], r["backend"], r["depth"]) == (row_dataset, backend, depth)]
        bounds = [r[f"{stage}_{metric}_mean"] + sign * r[f"{stage}_{metric}_sd"]
                  for r in subset for stage in STAGES for sign in (-1, 1)]
        if metric == "accuracy_percent":
            bounds.append(clean[row_dataset])
        low, high = min(bounds), max(bounds)
        if scale == "log":
            base.require(low > 0, "Log bounds must remain positive without clipping.")
            pad = max(.08, (math.log10(high) - math.log10(low)) * .12)
            limits = 10 ** (math.log10(low) - pad), 10 ** (math.log10(high) + pad)
        else:
            pad = max((high - low) * .13, .002 if metric == "teacher_kl" else .3)
            limits = (0, high + pad) if metric == "teacher_kl" else (max(0, low - pad), min(100, high + pad))
        for col, (fault, fault_label) in enumerate(base.FAULTS.items()):
            ax = axes[row, col]
            points = sorted((r for r in subset if r["fault_kind"] == fault), key=lambda r: r["fault_rate_percent"])
            ax.set_yscale(scale)
            ax.set_ylim(*limits)
            ax.set_xlim(-.25, 5.25)
            ax.set_xticks(base.RATES)
            ax.tick_params(labelbottom=True, labelsize=9)
            ax.grid(alpha=.17, linewidth=.65)
            if scale == "log":
                ax.yaxis.set_major_locator(base.LogLocator(base=10, numticks=5))
                ax.yaxis.set_minor_formatter(base.NullFormatter())
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            for stage, style in (("calibration", "--"), ("matched_recovery", "-")):
                ax.plot(base.RATES, [r[f"{stage}_{metric}_mean"] for r in points],
                        color="#9ba4aa", linestyle=style, linewidth=1.2, alpha=.85, zorder=1)
            for r in points:
                rate = r["fault_rate_percent"]
                color = base.SOURCES[r["calibration_family"]][1]
                baseline, recovered = (r[f"{stage}_{metric}_mean"] for stage in ("calibration", "matched_recovery"))
                ax.vlines(rate, min(baseline, recovered), max(baseline, recovered), color=color, alpha=.6, linewidth=1.2)
                for stage, marker, face in (("calibration", "o", "white"), ("matched_recovery", "s", color)):
                    ax.errorbar(rate, r[f"{stage}_{metric}_mean"], yerr=r[f"{stage}_{metric}_sd"],
                                marker=marker, markersize=6, color=color, markerfacecolor=face, markeredgewidth=1.1,
                                capsize=1.8, linewidth=.8, elinewidth=.8, zorder=3)
                if r["source_switch"]:
                    other_color = base.SOURCES[r["recovery_family"]][1]
                    ax.errorbar(rate, r[f"best_recovery_{metric}_mean"], yerr=r[f"best_recovery_{metric}_sd"],
                                marker="D", markersize=5.5, color=other_color, markerfacecolor="white", markeredgewidth=1.4,
                                capsize=1.8, elinewidth=.8, zorder=4)
                gain = r["matched_accuracy_gain_pp_mean"] if metric == "accuracy_percent" else r["matched_kl_gain_percent_mean"]
                annotation = f"{gain:+.1f}" + ("" if metric == "accuracy_percent" else "%")
                ax.text(rate, -.225, annotation, transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=8.6, color="#08766e" if gain >= 0 else "#b33e38")
            if metric == "accuracy_percent":
                ax.axhline(clean[row_dataset], color="#88949b", linestyle=(0, (2, 3)), linewidth=.8)
            if col == 0:
                ax.set_ylabel("Teacher KL (nats" + (", log scale)" if scale == "log" else ")")
                              if metric == "teacher_kl" else "Test accuracy (%)", labelpad=8, fontsize=10)
                ax.text(-.43, -.225, "gain", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=8.6, color="#59646b")
            if row == 0:
                ax.set_title(fault_label, fontsize=12.5, pad=13)
            if row == len(rows) - 1:
                ax.set_xlabel("Corrupt-device ratio (%)", labelpad=28, fontsize=10)
        box = axes[row, 0].get_position()
        row_label = base.DATASETS[row_dataset] if device else backend.upper()
        fig.text(.022, (box.y0 + box.y1) / 2, f"{row_label} · {depth} analog convolutions", rotation=90,
                 ha="center", va="center", fontsize=11, fontweight="medium")
    fig.suptitle(f"{title}  |  Best calibrated method and on-chip recovery", fontsize=18, x=.55, y=.976)
    fig.text(.55, .927 if compact else .949, "Source chosen by lowest observed mean teacher KL at each corruption rate; the classifier is also analog", ha="center", fontsize=11, color="#48535e")
    methods = [base.Line2D([], [], color=color, marker="o", linestyle="none", markersize=7, label=label)
               for label, color, _marker, _style in base.SOURCES.values()]
    fig.legend(handles=methods, loc="center", bbox_to_anchor=(.55, .88 if compact else .916), ncol=4, frameon=False, fontsize=10.5, columnspacing=2)
    stages = [base.Line2D([], [], color="#48535e", marker="o", markerfacecolor="white", linestyle="--", label="Best source + calibration"),
              base.Line2D([], [], color="#48535e", marker="s", linestyle="-", label="Same source + weights + calibration"),
              base.Line2D([], [], color="#48535e", marker="D", markerfacecolor="white", linestyle="none", label="Different source wins after recovery")]
    fig.legend(handles=stages, loc="center", bbox_to_anchor=(.55, .835 if compact else .882), ncol=3, frameon=False, fontsize=10, columnspacing=1.8)
    fig.text(.55, .059, "Gain below each rate: " + ("paired accuracy change (percentage points)." if metric == "accuracy_percent" else "paired KL reduction (%): 100 × [KL(calibration) − KL(same-source recovery)] / KL(calibration)."),
             ha="center", fontsize=9.1)
    fig.text(.55, .039, "Means ± sample SD over three arrays; positive gain helps. Both arms use five epochs from the same initial deployment. 4% was not run.", ha="center", fontsize=9, color="#48535e")
    fig.text(.55, .019, "Best observed methods are ranked retrospectively on test results, not a validated selection policy. Different-source diamonds also change the offline source.", ha="center", fontsize=8.8, color="#48535e")
    return fig


def gallery(out, records, group_by="dataset"):
    # A compact exact-value table complements the overview without more figures.
    tables = {}
    abbreviations = {k: v[0] for k, v in base.SOURCES.items()}
    groups = ("om", "pcm") if group_by == "device" else base.DATASETS
    for group in groups:
        table = '<table><thead><tr><th>Device / depth</th><th>Fault</th><th>Rate</th><th>Best calibrated source</th><th>Cal. KL / accuracy</th><th>Same-source recovery KL / accuracy</th><th>Paired ΔKL</th><th>Best recovered source</th><th>Best recovered KL / accuracy</th></tr></thead><tbody>'
        if group_by == "device":
            table = table.replace("Device / depth", "Dataset / depth")
        for r in records:
            if r["backend" if group_by == "device" else "dataset"] != group:
                continue
            row_label = base.DATASETS[r["dataset"]] if group_by == "device" else r["backend"].upper()
            table += f'<tr data-depth="{r["depth"]}"><td>{row_label} / {r["depth"]}</td><td>{base.FAULTS[r["fault_kind"]]}</td><td>{r["fault_rate_percent"]}%</td><td>{abbreviations[r["calibration_family"]]}</td>'
            for stage in ("calibration", "matched_recovery"):
                table += f'<td>{r[stage + "_teacher_kl_mean"]:.5g} / {r[stage + "_accuracy_percent_mean"]:.2f}%</td>'
            table += f'<td>{r["matched_kl_gain_nats_mean"]:+.5g}</td><td>{abbreviations[r["recovery_family"]]}</td><td>{r["best_recovery_teacher_kl_mean"]:.5g} / {r["best_recovery_accuracy_percent_mean"]:.2f}%</td></tr>'
        tables[group] = table + '</tbody></table>'
    doc = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Best calibrated method and recovery</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1600px;margin:30px auto;padding:0 22px;color:#21303b}h1{font-size:28px}p{max-width:1200px}a{color:#2166ac}.controls{display:flex;flex-wrap:wrap;gap:24px;background:#edf2f5;padding:16px;border-radius:8px}label{display:flex;flex-direction:column;gap:5px;font-size:13px}select{font:16px system-ui;padding:8px;background:white;border:1px solid #afbdc7;border-radius:5px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:12px;width:100%}td,th{padding:8px;border-bottom:1px solid #d8e0e5;text-align:right}th{background:#edf2f5}td:nth-child(-n+4){text-align:left}.scroll{overflow-x:auto}</style></head><body>
<h1>Best calibrated method and on-chip recovery</h1><p>One overview per dataset. For every fault, hardware configuration and corruption rate, choose the source with the <strong>lowest mean teacher KL after digital calibration</strong>. Trace that same source through weight recovery; a diamond shows when a different source performs better after recovery.</p>
<div class="controls"><label>Dataset<select id="dataset"><option value="cifar10">CIFAR-10</option><option value="cifar100">CIFAR-100</option></select></label><label>Vertical axis<select id="view"><option value="kl_log">Teacher KL · logarithmic</option><option value="kl_linear">Teacher KL · linear</option><option value="accuracy">Accuracy of the KL-selected sources</option></select></label></div>
<p><a id="jpg">JPG</a> · <a id="png">PNG</a> · <a id="svg">Editable SVG</a> · <a id="pdf">Dataset PDF: main KL figure + companion accuracy</a></p><img id="plot" alt="Best calibrated source and its recovery at each fault ratio, with hardware configurations in rows and faults in columns">
<p><strong>Colour identifies the source:</strong> no HWA, normal HWA, noisy HWA, corruption-aware HWA. Open circles are the best calibrated source; filled squares are that same source after recovery. Open diamonds appear if another source has the lowest recovered KL. Lines join the first two series. Read diamond positions against the same vertical axis.</p>
<p><strong>Gain numbers below each panel</strong> compare calibration and weight recovery of the same source. KL numbers are means of paired percentage reductions; accuracy numbers are percentage-point changes. Positive helps and negative hurts. The CSV separately reports improvement from comparing the best source before recovery with the best source after recovery; that may include an offline source change.</p>
<p>Source winners are chosen using means over the same three arrays, not independently per array. Exact ties use source order: no HWA, normal HWA, noisy HWA, corruption-aware HWA; all ties and runner-up margins are recorded. This is a retrospective description of test results, not a validated rule for selecting deployments. Error bars show sample SD across arrays and do not account for source selection or training-seed uncertainty.</p>
<p>Both adaptation arms use five full epochs from the same source-specific initial deployment. All faults and original source-selection/convergence flags are retained. 0% has healthy programming error; 4% is unmeasured. CIFAR-100 PCM stuck-low corruption-aware selection returned the original teacher. Separate OM LR/open-loop follow-ups are excluded.</p>
<details><summary>Exact values for all rates</summary><div class="scroll" id="table"></div></details>
<p><a href="best_method_summary.csv">Selected means, gains, source ties and flags</a> · <a href="paired_arrays.csv">Matched arrays</a> · <a href="candidate_endpoints.csv">All candidate endpoints</a> · <a href="jpg_full_paths.txt">JPG full paths</a> · <a href="README.md">Reproduction notes</a> · <a href="../index.html">Original gallery</a></p>
<script>const tables=TABLES;function update(){const d=document.getElementById('dataset').value,v=document.getElementById('view').value,s=d+'_best_method_'+v;document.getElementById('plot').src=s+'.png';for(const e of ['jpg','png','svg'])document.getElementById(e).href=s+'.'+e;document.getElementById('pdf').href=d+'_best_method.pdf';document.getElementById('table').innerHTML=tables[d];}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));update();</script></body></html>'''
    if group_by == "device":
        doc = doc.replace("</h1>", '</h1><p><a href="../explorer.html"><strong>Unified explorer: choose dataset, device, accuracy or KL</strong></a></p>')
        doc = doc.replace("One overview per dataset.", "Separate OM and PCM overviews, each containing both datasets. Select 4 or 8 analog convolutions, or show both depths together.")
        doc = doc.replace('<label>Dataset<select id="dataset"><option value="cifar10">CIFAR-10</option><option value="cifar100">CIFAR-100</option>',
                          '<label>Device<select id="dataset"><option value="om">OM</option><option value="pcm">PCM</option>')
        doc = doc.replace('<label>Vertical axis', '<label>Analog convolutions<select id="depth"><option value="4">4 analog convolutions</option><option value="8">8 analog convolutions</option><option value="all">Both depths</option></select></label><label>Vertical axis')
        doc = doc.replace("s=d+'_best_method_'+v", "depth=document.getElementById('depth').value,prefix=d+(depth==='all'?'':'_conv'+depth),s=prefix+'_best_method_'+v")
        doc = doc.replace("href=d+'_best_method.pdf'", "href=prefix+'_best_method.pdf'")
        doc = doc.replace("innerHTML=tables[d];", "innerHTML=tables[d];document.querySelectorAll('#table tbody tr').forEach(row=>row.hidden=depth!=='all'&&row.dataset.depth!==depth);document.getElementById('plot').alt=d.toUpperCase()+' best calibrated source and recovery, '+(depth==='all'?'both depths':depth+' analog convolutions')+', both datasets';")
        doc = doc.replace("Dataset PDF:", "Selected device/depth PDF:")
        doc = doc.replace("with hardware configurations in rows", "with datasets and analog depths in rows")
        doc = doc.replace('href="../index.html">Original gallery', 'href="../index.html">Dataset overviews')
    (out / "index.html").write_text(doc.replace("TABLES", json.dumps(tables)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--group-by", choices=("dataset", "device"), default="dataset")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    inputs = args.input_dir or root / "results/cifar-sweep-analysis"
    audit_path = args.source_audit or root / "artifacts/cifar_recovery_full_results_20260920/source_fit_selection.csv"
    default_out = root / "artifacts/cifar_kl_fault_curves_20260922/best_method"
    if args.group_by == "device":
        default_out = default_out / "by_device"
    out = (args.output or default_out).resolve()
    paths = {n: inputs / n for n in ("summary.csv", "per_array.csv", "clean_sources.csv")}
    index, raw, clean, _endpoints, _observations, _verified = base.load_data(paths)
    records, paired, candidates = summarize(index, raw, base.read_csv(audit_path))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("best_method_summary.csv", records), ("paired_arrays.csv", paired), ("candidate_endpoints.csv", candidates)):
        base.write_csv(out / name, rows)
    base.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    figures = []
    groups = ("om", "pcm") if args.group_by == "device" else base.DATASETS
    for group in groups:
        for analog_depth in ((None, "4", "8") if args.group_by == "device" else (None,)):
            prefix = group + (f"_conv{analog_depth}" if analog_depth else "")
            with base.PdfPages(out / f"{prefix}_best_method.pdf") as pdf:
                for view in VIEWS:
                    fig = figure(None if args.group_by == "device" else group, view, records, clean,
                                 device=group if args.group_by == "device" else None, analog_depth=analog_depth)
                    stem = f"{prefix}_best_method_{view}"
                    fig.savefig(out / f"{stem}.png", dpi=150, facecolor="white")
                    fig.savefig(out / f"{stem}.jpg", dpi=200, facecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
                    fig.savefig(out / f"{stem}.svg", facecolor="white")
                    if view != "kl_linear":
                        pdf.savefig(fig)
                    base.plt.close(fig)
                    figures.append(stem)
                    print(stem, flush=True)
    gallery(out, records, args.group_by)
    for script in (Path(__file__), Path(base.__file__)):
        if script.resolve() != (out / script.name).resolve():
            shutil.copyfile(script, out / script.name)
    paths["source_fit_selection.csv"] = audit_path
    validation = dict(status="passed", group_by=args.group_by, selection_metric="Lowest mean teacher KL on three fixed test arrays, retrospective",
        ranking_unit="One common source per dataset/device/depth/fault/rate; not per-array best selection",
        plotted_conditions=len(records), paired_array_rows=len(paired), candidate_rows=len(candidates),
        source_switches=sum(r["source_switch"] for r in records), arrays=sorted(base.SEEDS), rates_percent=base.RATES,
        checks=["Complete candidate grid", "Original endpoint means and SD recomputed from raw arrays",
                "Selected sources minimize mean KL", "Exact ties and runner-up margins recorded",
                "Paired gains agree with endpoint differences", "Best recovery no worse in mean KL than matched recovery"],
        input_files={name: dict(path=str(p.resolve()), sha256=base.sha256(p)) for name, p in paths.items()},
        generator_sha256=base.sha256(Path(__file__)), helper_sha256=base.sha256(Path(base.__file__)), figures=figures)
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (out / "jpg_full_paths.txt").write_text("\n".join(str(out / (s + ".jpg")) for s in figures) + "\n")
    (out / "README.md").write_text('''# Best calibrated source and on-chip recovery

Two main overview figures, one per dataset: `cifar10_best_method_kl_log` and
`cifar100_best_method_kl_log`. Each has four hardware/depth rows and three fault
columns, with rates 0/1/2/3/5%. A linear KL alternative and companion accuracy
figure use exactly the same KL-selected sources. JPG, PNG, SVG and dataset PDFs
are included. Each PDF contains the main KL overview and companion accuracy.

At each condition choose the source with lowest mean teacher KL after five epochs
of calibration. Candidates: no HWA, normal HWA, noisy HWA, and the matching
fault-specific corruption-aware source. Ranking uses the common mean across the
three fixed test arrays, not different winners for each array. Exact ties keep
the first source in that order, and all tied sources and runner-up margins are
in the CSV. A numerical winner need not be meaningfully or significantly better.
These are retrospective observed best results, not a prospective deployment
policy or a change to the frozen checkpoint-selection protocol.

Open circles: winning calibrated source. Filled squares: matched five-epoch
weights-plus-calibration endpoint from the SAME source. Their colours identify
the source. If a different source has the lowest recovered mean KL, an open
diamond marks its recovered endpoint in that source's colour. A diamond can
have worse accuracy because all choices use KL. Grey lines join circles and
squares, not the diamond alternatives. Both arms start from the source-specific
initial deployment; these are not sequential five-epoch stages.

Numbers below each rate are the mean paired same-source gain. On KL figures:
100 * [KL(calibration) - KL(matched recovery)] / KL(calibration), per array and
then averaged. On accuracy figures: recovered minus calibrated accuracy, in
percentage points. Positive helps, negative hurts. KL is teacher-to-student,
natural logs, temperature 1. Error bars are endpoint sample SD across the three
arrays, not selection-adjusted confidence intervals. Selection uncertainty and
source-training-seed variation are not estimated. Very small baseline KL can
produce large percentage degradation for small absolute changes.

The CSV ALSO reports best-available gains: best calibrated endpoint versus best
recovered endpoint. These comparisons can change the offline source and do not
isolate the effect of on-chip training alone. Both gain definitions include
paired array means and SD; baseline/recovered KL and accuracy are retained.

All measured cases are preserved, including negative gains and source switches.
There are 120 plotted conditions, 360 paired array rows, 960 candidate endpoint
rows. Nominal cases can repeat across fault columns and are not independent
repetitions. All 10,000 test images are used. Three arrays are hardware variation,
not three training seeds. Preserve all original source convergence flags.

The digital prefix and BN running statistics remain frozen; calibration adjusts
gains, suffix BN affine parameters and classifier bias. PCM uses Gaussian
endpoint reprogramming per minibatch; OM uses the historical closed-loop writer,
LR 1e-4 and 640-pulse/cell recovery cap. No later OM LR/open-loop results are mixed
in. CIFAR-100 PCM stuck-low CDT selected the epoch-0 teacher. No new training or
study finalization is performed by this descriptive export.

Reproduce from this folder within the complete gallery bundle:

```sh
python plot_cifar_best_method.py --input-dir ../inputs --source-audit ../inputs/source_fit_selection.csv --output regenerated
```
''')
    if args.group_by == "device":
        readme = out / "README.md"
        text = readme.read_text().replace(
            "Two main overview figures, one per dataset: `cifar10_best_method_kl_log` and\n`cifar100_best_method_kl_log`. Each has four hardware/depth rows and three fault\ncolumns, with rates 0/1/2/3/5%.",
            "Two main overview figures, one per device: `om_best_method_kl_log` and\n`pcm_best_method_kl_log`. Each has four dataset/depth rows: CIFAR-10/4, CIFAR-10/8,\nCIFAR-100/4, CIFAR-100/8, and three fault columns, with rates 0/1/2/3/5%.\nThe OM figure contains only OM results; the PCM figure contains only PCM results.")
        text = text.replace("dataset PDFs", "device PDFs")
        text = text.replace("python plot_cifar_best_method.py --input-dir ../inputs --source-audit ../inputs/source_fit_selection.csv --output regenerated",
                            "python plot_cifar_best_method.py --group-by device --input-dir ../../inputs --source-audit ../../inputs/source_fit_selection.csv --output regenerated")
        text += "\nThe single `index.html` has an analog-convolution selector (4, 8, or both).\nDepth-specific figures use `<device>_conv<depth>_best_method_<view>` and\nPDFs use `<device>_conv<depth>_best_method.pdf`. The selector updates the\nfigure, download links, and exact-value table together; CSVs retain all depths.\n"
        readme.write_text(text)
    print(json.dumps({"output": str(out), "conditions": len(records), "source_switches": validation["source_switches"]}))


if __name__ == "__main__":
    main()
