"""Plot frozen CIFAR fault-sweep endpoints; no fitting or checkpoint selection.

Curves show arithmetic means and faint points show all three modeled arrays.
Every plotted endpoint is checked against the underlying per-array CSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import statistics
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


DATASETS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
FAULTS = {"open": "Stuck-low / open", "random": "Random-stuck", "gmax": "Stuck-high"}
HARDWARE = (("pcm", "4"), ("pcm", "8"), ("om", "4"), ("om", "8"))
RATES = (0, 1, 2, 3, 5)
SOURCES = {
    "digital": ("No HWA", "#4b5563", "o", "-"),
    "standard_hwa": ("Normal HWA", "#2166ac", "s", "--"),
    "noise_hwa": ("Noisy HWA", "#098779", "^", "-."),
    "cdt": ("Corruption-aware HWA", "#cb5221", "D", ":"),
}
METHODS = {
    "none": "Direct deployment",
    "calibration": "Digital calibration · 5 epochs",
    "onchip_calibration": "Weights + calibration · 5 epochs",
}
KEYS = ("dataset", "backend", "depth", "source", "case", "method", "epoch")
SEEDS = {"251001", "251002", "251003"}
VIEWS = {
    "kl_log": ("teacher_kl", "log", "Teacher KL (nats, log scale)"),
    "kl_linear": ("teacher_kl", "linear", "Teacher KL (nats)"),
    "accuracy": ("accuracy_percent", "linear", "Test accuracy (%)"),
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key_for(dataset, backend, depth, family, fault, rate, method):
    source = "cdt_" + fault if family == "cdt" else family
    case = f"{fault}_{rate * 10000}ppm" if rate else "nominal"
    return dataset, backend, depth, source, case, method, "0" if method == "none" else "5"


def load_data(paths):
    summaries = read_csv(paths["summary.csv"])
    index = {tuple(row[k] for k in KEYS): row for row in summaries}
    require(len(index) == len(summaries), "Expected unique summary endpoint keys.")
    raw = defaultdict(list)
    for row in read_csv(paths["per_array.csv"]):
        raw[tuple(row[k] for k in KEYS)].append(row)
    clean = {}
    for dataset in DATASETS:
        teachers = [r for r in read_csv(paths["clean_sources.csv"])
                    if r["dataset"] == dataset and r["source"] == "digital"]
        values = {float(r["accuracy_percent"]) for r in teachers}
        require(len(values) == 1, f"Expected one clean teacher accuracy for {dataset}: {values}")
        clean[dataset] = values.pop()

    endpoints, observations, verified = [], [], set()
    for dataset in DATASETS:
        for backend, depth in HARDWARE:
            for fault in FAULTS:
                for family in SOURCES:
                    for rate in RATES:
                        for method in METHODS:
                            key = key_for(dataset, backend, depth, family, fault, rate, method)
                            require(key in index, f"Expected measured endpoint, missing: {key}")
                            row, samples = index[key], raw[key]
                            require(len(samples) == 3 and {s["array_seed"] for s in samples} == SEEDS,
                                    f"Expected exactly three declared arrays: {key}")
                            require(all(int(s["examples"]) == 10000 for s in samples),
                                    f"Expected 10,000 test examples: {key}")
                            for metric in ("teacher_kl", "accuracy_percent"):
                                values = [float(s[metric]) for s in samples]
                                require(all(math.isfinite(v) for v in values), f"Nonfinite metric: {key}")
                                if metric == "teacher_kl":
                                    require(min(values) > 0, f"Log plot requires positive deployed KL: {key}")
                                else:
                                    require(all(0 <= v <= 100 for v in values), f"Invalid accuracy: {key}")
                                for suffix, fn in (("mean", statistics.mean), ("sd", statistics.stdev)):
                                    reported = float(row[f"{metric}_{suffix}"])
                                    require(math.isclose(reported, fn(values), rel_tol=1e-8, abs_tol=1e-8),
                                            f"Summary disagrees with raw arrays: {key}, {metric}, {suffix}")
                            metadata = dict(fault_kind=fault, fault_rate_percent=rate,
                                            fault_rate_fraction=rate / 100, source_family=family,
                                            clean_teacher_accuracy_percent=clean[dataset])
                            endpoints.append(dict(row, **metadata))
                            observations.extend(dict(sample, **metadata) for sample in samples)
                            verified.add(key)
    require(len(endpoints) == 1440 and len(verified) == 1296, "Unexpected endpoint coverage.")
    return index, raw, clean, endpoints, observations, verified


def make_figure(dataset, backend, depth, view, index, raw, clean):
    metric, scale, ylabel = VIEWS[view]
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 10.4), sharex=True, sharey="row")
    fig.subplots_adjust(left=.115, right=.983, top=.85, bottom=.10, hspace=.22, wspace=.10)
    for row_index, (method, method_label) in enumerate(METHODS.items()):
        for col, (fault, fault_label) in enumerate(FAULTS.items()):
            ax = axes[row_index, col]
            for family, (label, color, marker, linestyle) in SOURCES.items():
                keys = [key_for(dataset, backend, depth, family, fault, rate, method) for rate in RATES]
                means = [float(index[k][metric + "_mean"]) for k in keys]
                for rate, key in zip(RATES, keys):
                    vals = [float(r[metric]) for r in raw[key]]
                    ax.vlines(rate, min(vals), max(vals), color=color, alpha=.27, linewidth=1.1)
                    ax.scatter([rate] * 3, vals, color=color, s=16, alpha=.35,
                               linewidths=0, zorder=2)
                ax.plot(RATES, means, color=color, marker=marker, linestyle=linestyle,
                        markersize=5, markerfacecolor="white", markeredgewidth=1.2,
                        linewidth=1.8, label=label, zorder=3)
            if view == "accuracy":
                ax.axhline(clean[dataset], color="#87929a", linestyle=(0, (2, 3)), linewidth=1)
            ax.set_yscale(scale)
            if scale == "log":
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            ax.set_xticks(RATES)
            ax.set_xlim(-.18, 5.18)
            ax.grid(axis="both", which="major", alpha=.19, linewidth=.65)
            ax.tick_params(labelsize=9)
            if row_index == 0:
                ax.set_title(f"({chr(97 + col)})  {fault_label}", fontsize=12, pad=12)
            if row_index == 2:
                ax.set_xlabel("Corrupt-device ratio (%)", labelpad=7)
            if col == 0:
                ax.set_ylabel(ylabel, labelpad=8)
        box = axes[row_index, 0].get_position()
        fig.text(.024, (box.y0 + box.y1) / 2, method_label, rotation=90,
                 ha="center", va="center", fontsize=11, fontweight="medium")

    # Equal scales across faults; calibration and joint recovery also share limits.
    for row_group in ((0,), (1, 2)):
        methods = [list(METHODS)[r] for r in row_group]
        values = [float(s[metric])
                  for fault in FAULTS for family in SOURCES for rate in RATES for method in methods
                  for s in raw[key_for(dataset, backend, depth, family, fault, rate, method)]]
        if view == "accuracy":
            limits = 0, 100
        elif scale == "log":
            low, high = math.log10(min(values)), math.log10(max(values))
            padding = max(.08, .08 * (high - low))
            limits = 10 ** (low - padding), 10 ** (high + padding)
        else:
            limits = 0, max(values) * 1.10
        for r in row_group:
            axes[r, 0].set_ylim(*limits)

    title_metric = "Teacher KL versus corrupt-device ratio" if metric == "teacher_kl" else "Accuracy versus corrupt-device ratio"
    fig.suptitle(f"{DATASETS[dataset]}  |  {title_metric}", fontsize=17, x=.54, y=.977)
    fig.text(.54, .941, f"{backend.upper()} · last {depth} convolutions + classifier analog · frozen digital prefix",
             ha="center", fontsize=11, color="#48535e")
    handles = [Line2D([], [], color=c, marker=m, linestyle=s, markerfacecolor="white", linewidth=1.8, label=l)
               for l, c, m, s in SOURCES.values()]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.54, .904), ncol=4,
               frameon=False, fontsize=10, columnspacing=2)
    footnote = "Curves: arithmetic means; faint dots and vertical ranges: three modeled arrays. Lines connect measured rates; 4% was not run."
    if metric == "teacher_kl":
        second = "KL(original digital teacher || student), T = 1; lower is better. 0% means a fault-free modeled array, including programming error."
    else:
        second = f"Higher is better. Dotted reference: clean digital teacher ({clean[dataset]:.2f}%). 0% retains modeled programming error."
    if dataset == "cifar100" and backend == "pcm":
        second += "\nStuck-low corruption-aware selection returned the epoch-0 teacher; its curve overlaps No HWA."
    fig.text(.54, .048, footnote, ha="center", fontsize=8.6, color="#48535e")
    fig.text(.54, .022, second, ha="center", va="center", fontsize=8.4, color="#48535e")
    return fig


def write_gallery(out, endpoints):
    table_data = {}
    for dataset in DATASETS:
        for backend, depth in HARDWARE:
            rows = [r for r in endpoints if r["dataset"] == dataset and r["backend"] == backend
                    and r["depth"] == depth and r["fault_rate_percent"] == 5]
            rows_index = {(r["fault_kind"], r["source_family"], r["method"]): r for r in rows}
            table = '<table><caption>Measured endpoints at 5% faults: accuracy (%) / teacher KL (nats)</caption><thead><tr><th>Fault</th><th>Training</th>'
            table += ''.join(f'<th>{html.escape(m)}</th>' for m in METHODS.values()) + '</tr></thead><tbody>'
            for fault, fault_label in FAULTS.items():
                for family, (label, *_rest) in SOURCES.items():
                    table += f'<tr><td>{fault_label}</td><td>{label}</td>'
                    for method in METHODS:
                        r = rows_index[fault, family, method]
                        table += f'<td>{float(r["accuracy_percent_mean"]):.2f} / {float(r["teacher_kl_mean"]):.5g}</td>'
                    table += '</tr>'
            table_data[f"{dataset}_{backend}_conv{depth}"] = table + '</tbody></table>'
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CIFAR corruption curves</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#21303b;max-width:1450px;margin:30px auto;padding:0 22px}
h1{font-size:28px;margin-bottom:4px}p{max-width:1100px}a{color:#2166ac}label{display:flex;flex-direction:column;font-size:13px;gap:5px}
.controls{display:flex;gap:20px;flex-wrap:wrap;background:#edf2f5;padding:18px;border-radius:8px;margin:24px 0}
select{font:16px system-ui;padding:8px;border:1px solid #afbdc7;border-radius:5px;background:white}img{width:100%;height:auto}
table{border-collapse:collapse;font-size:14px;width:100%;margin:25px 0}caption{text-align:left;font-weight:600;padding:12px 0}
td,th{padding:10px;border-bottom:1px solid #d8e0e5;text-align:right}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}
th{background:#edf2f5}details{margin:24px 0}.table-wrap{overflow-x:auto}</style></head><body>
<h1>CIFAR corruption curves</h1><p>How teacher KL changes with the fraction of permanently faulty devices. Compare training methods separately for each fault, device model and analog suffix depth.</p>
<div class="controls"><label>Dataset<select id="dataset"><option value="cifar10">CIFAR-10</option><option value="cifar100">CIFAR-100</option></select></label>
<label>Device model<select id="backend"><option value="pcm">PCM</option><option value="om">IBM OM</option></select></label>
<label>Analog convolutions + classifier<select id="depth"><option value="8">8 convolutions</option><option value="4">4 convolutions</option></select></label>
<label>Vertical axis<select id="view"><option value="kl_log">Teacher KL · log scale</option><option value="kl_linear">Teacher KL · linear scale</option><option value="accuracy">Accuracy · linear scale</option></select></label></div>
<p><a id="svg">Download this figure as SVG</a> · <a id="png">PNG</a> · <a id="jpg">JPG</a> · <a id="pdf">All four configurations for this dataset (PDF)</a></p>
<img id="plot" width="1932" height="1456" alt="Nine-panel comparison of training methods across three faults and three adaptation methods">
<p><strong>Reading the plots:</strong> lower KL is better. Lines show means; faint dots and vertical ranges show the three modeled arrays. Each row compares the same four source-training methods. Calibration and weight recovery each start from the corresponding original deployment; the bottom row is not a continuation of the middle row.</p>
<p>0% means no permanent faults, with ordinary modeled programming error still present. Measured rates are 0%, 1%, 2%, 3%, and 5%; there is no 4% observation. Calibration and recovery rows share a vertical range; direct deployment has its own range.</p>
<div id="table" class="table-wrap"></div>
<details><summary>Training methods and experimental scope</summary>
<p><strong>Normal HWA:</strong> nominal programming-noise training. <strong>Noisy HWA:</strong> stronger healthy programming-noise training, selected on the declared development stress set. <strong>Corruption-aware HWA:</strong> fault-specific training with resampled 2%, 3%, and 5% fault rates. <strong>No HWA:</strong> original digital teacher weights deployed on the modeled hardware.</p>
<p><strong>Calibration:</strong> digital gains, suffix batch-normalization affine parameters and classifier bias; running statistics and physical weights stay fixed. <strong>Weights + calibration:</strong> adds physical-weight updates during the same five-epoch budget. Both use all 45,000 adaptation images per epoch. These three arrays quantify modeled hardware variability; there is only one HWA training seed.</p>
<p>The selected CIFAR-100 PCM stuck-low corruption-aware checkpoints are the original epoch-0 teacher, so those curves overlap No HWA. Of the 40 selected HWA sources, 31 retain late-convergence review flags. These are frozen exploratory results.</p>
<p>PCM recovery reprograms Gaussian endpoints each minibatch. OM uses the historical closed-loop pulse Adam, weight learning rate 1e-4 and 640-pulse/cell recovery cap. Later OM learning-rate and open-loop studies are separate. Fault ratios count PCM bank devices or OM abstract active cells, respectively.</p>
<p>Inspired by Figure 5 of Li et al., APL Machine Learning 1, 016104 (2023), DOI 10.1063/5.0131797. That figure plots classification error and varies the corruption fraction during training. These figures plot your teacher KL and compare your frozen training methods. They do not reproduce the paper's curves or establish typical physical fault rates.</p></details>
<p><a href="plot_endpoints.csv">All plotted means, sample SD and accuracy (CSV)</a> · <a href="plot_arrays.csv">Individual arrays (CSV)</a> · <a href="README.md">Protocol and reproduction notes</a> · <a href="validation.json">Data validation</a></p>
<script>const tables=TABLE_DATA;
function update(){const ds=document.getElementById('dataset').value,b=document.getElementById('backend').value,d=document.getElementById('depth').value,v=document.getElementById('view').value;
const stem=ds+'_'+b+'_conv'+d;document.getElementById('plot').src=stem+'_'+v+'.png';document.getElementById('svg').href=stem+'_'+v+'.svg';document.getElementById('png').href=stem+'_'+v+'.png';document.getElementById('jpg').href=stem+'_'+v+'.jpg';document.getElementById('pdf').href=ds+'_'+v+'.pdf';document.getElementById('table').innerHTML=tables[stem];}
document.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));update();</script></body></html>'''
    (out / "index.html").write_text(document.replace("TABLE_DATA", json.dumps(table_data)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = args.input_dir or root / "results/cifar-sweep-analysis"
    out = (args.output or root / "artifacts/cifar_kl_fault_curves_20260922").resolve()
    paths = {name: inputs / name for name in ("summary.csv", "per_array.csv", "clean_sources.csv")}
    paths["source_fit_selection.csv"] = args.source_audit or root / "artifacts/cifar_recovery_full_results_20260920/source_fit_selection.csv"
    index, raw, clean, endpoints, observations, verified = load_data(paths)
    audit = read_csv(paths["source_fit_selection.csv"])
    source_metadata = {(r["dataset"], r["backend"], r["depth"], r["source"]): r for r in audit}
    for row in endpoints:
        selected = source_metadata[row["dataset"], row["backend"], row["depth"], row["source"]]
        row.update(selected_source_epoch=selected["selected_epoch"],
                   source_convergence_review_required=selected["convergence_review_required"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs").mkdir(exist_ok=True)
    for name, path in paths.items():
        shutil.copyfile(path, out / "inputs" / name)
    shutil.copyfile(Path(__file__), out / "reproduce.py")
    write_csv(out / "plot_endpoints.csv", endpoints)
    write_csv(out / "plot_arrays.csv", observations)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    figures = []
    with ExitStack() as stack:
        pdfs = {(dataset, view): stack.enter_context(PdfPages(out / f"{dataset}_{view}.pdf"))
                for dataset in DATASETS for view in VIEWS}
        for dataset in DATASETS:
            for backend, depth in HARDWARE:
                for view in VIEWS:
                    fig = make_figure(dataset, backend, depth, view, index, raw, clean)
                    stem = f"{dataset}_{backend}_conv{depth}_{view}"
                    fig.savefig(out / f"{stem}.png", dpi=150, facecolor="white")
                    fig.savefig(out / f"{stem}.jpg", dpi=200, facecolor="white",
                                pil_kwargs={"quality": 95, "subsampling": 0})
                    fig.savefig(out / f"{stem}.svg", facecolor="white")
                    pdfs[dataset, view].savefig(fig)
                    plt.close(fig)
                    figures.append(stem)
                    print(f"Rendered {stem}", flush=True)
    write_gallery(out, endpoints)
    readme = """# CIFAR teacher KL versus corrupt-device ratio

Open `index.html` to choose dataset, device model, analog depth, and KL/accuracy view.
Each dataset PDF has four pages: PCM/4, PCM/8, OM/4, OM/8 analog convolutions,
always with an analog classifier and a frozen digital prefix. Individual figures
are available as PNG, JPG and editable vector SVG. The KL PDFs have log and linear versions;
the accuracy PDFs provide the companion task-performance metric.

Columns separate stuck-low/open, random-stuck and stuck-high faults. Rows compare
direct deployment, five epochs of digital calibration, and five epochs of physical
weight recovery plus calibration. Each row has No HWA, Normal HWA, Noisy HWA and
fault-specific Corruption-aware HWA curves. Calibration and joint recovery are
independent alternatives from the same source-specific initial deployment, not
successive five-epoch phases. Both have 45,000 adaptation images per epoch.

The metric is mean KL(original digital teacher || deployed student), natural-log
units (nats), temperature 1, on all 10,000 test images. Lower is better. Curves
are arithmetic means over three arrays; faint dots show all arrays and vertical
segments show their min/max range, not SD or confidence intervals. CSV files also
contain sample SD and accuracy. Arrays are 251001, 251002 and 251003; they are not
independent training seeds. Log plots use the same arithmetic means as linear plots.
All deployed KL values are positive; no log offset or clipping is used.

Rates are the declared fault probabilities: 0, 1, 2, 3 and 5 percent. They count
PCM differential-bank devices or OM abstract active cells, respectively. Masks
are nested across rates, and paired across sources and fault types. 4% was not
measured: connecting segments are visual guides, not additional observations.
0% still includes healthy programming error; it is not the clean digital network.
The clean teacher has KL=0 against itself and is not drawn on a logarithmic axis.
Accuracy figures include its reference accuracy.

Axes have common limits across the three faults within each row, and calibration
and recovery also share limits. Direct deployment has a separate vertical range.
The log view makes the large direct-deployment outliers visible without obscuring
the other curves; linear views retain all the same endpoints.

Normal HWA uses nominal healthy programming noise. Noisy HWA uses inflated healthy
programming noise and the original development stress selection. Corruption-aware
HWA adds fault-specific corruption resampled at 2/3/5% during training. The selected
CIFAR-100 PCM stuck-low CDT source is the epoch-0 teacher at both depths, so it
overlaps No HWA. 31/40 selected HWA fits retain late-convergence review flags. The
original selections are retained, including poor or degraded outcomes.

Calibration adjusts digital gains, suffix BN affine parameters and classifier bias;
BN running statistics are frozen. PCM weight recovery performs full Gaussian
endpoint reprogramming per minibatch. OM uses the historical closed-loop pulse Adam,
weight LR 1e-4 and 640-pulse/cell recovery cap. Separate OM learning-rate and open-loop
follow-ups are not mixed into this sweep. These are exploratory model-based results,
with one source-training seed, not a final demonstration of optimal HWA or a general
requirement for on-chip training.

This is inspired by Figure 5 of Li et al., *Impact of analog memory device failure
on in-memory computing inference accuracy*, APL Machine Learning 1, 016104 (2023),
DOI 10.1063/5.0131797. The paper plots classification error and varies the corruption
fraction used during training. Here the vertical axis is your teacher KL and the
curves compare your available training methods. The paper's curves and fault-rate
regime are not reproduced. No new training was performed for these figures.

`plot_endpoints.csv` has 1,440 plotted means, representing 1,296 distinct endpoints.
The generic sources' nominal points recur across fault panels; they are explicitly
marked rate=0 and must not be treated as extra independent observations.
`plot_arrays.csv` has 4,320 corresponding plotted array observations (3,888 unique).
`validation.json` records coverage, checks and exact input hashes.

Reproduce from this portable folder with Python, matplotlib and its dependencies:

```sh
python reproduce.py --input-dir inputs --source-audit inputs/source_fit_selection.csv --output regenerated
```

Original frozen experiment contract: `docs/cifar_fault_rate_sweep.md` in the
`cifar-resnet-suffix-recovery` worktree. This export is a descriptive visualization;
it neither changes study selection nor finalizes the pending scientific review.
"""
    (out / "README.md").write_text(readme)
    validation = {
        "status": "passed", "input_files": {name: {"path": str(p.resolve()), "sha256": sha256(p)} for name, p in paths.items()},
        "generator_sha256": sha256(Path(__file__)), "matplotlib_version": matplotlib.__version__,
        "plotted_mean_points": len(endpoints), "unique_mean_endpoints": len(verified),
        "plotted_array_points": len(observations), "unique_array_endpoints": 3 * len(verified),
        "arrays": sorted(SEEDS), "test_images_per_array": 10000, "rates_percent": RATES,
        "missing_rate_percent": [4], "checks": ["Complete declared plotting grid", "Unique summary keys",
        "Three exact declared arrays per endpoint", "10,000 test examples per array", "Positive finite deployed KL",
        "Finite accuracy in [0,100]", "Means and sample SD recomputed from arrays", "Frozen source selections retained"],
        "clean_teacher_accuracy_percent": clean, "figures": figures,
        "source_review_flags": sum(r["convergence_review_required"] == "True" for r in audit),
        "epoch_zero_cdt": [{k: r[k] for k in ("dataset", "backend", "depth", "source", "selected_epoch")}
                           for r in audit if r["source"].startswith("cdt_") and r["selected_epoch"] == "0"],
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"output": str(out), "verified_endpoints": len(verified), "figure_count": len(figures)}))


if __name__ == "__main__":
    main()
