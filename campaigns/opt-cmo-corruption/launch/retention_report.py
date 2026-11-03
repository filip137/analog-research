"""Plot the fixed retention pilot from native collected measurements only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCES = {
    "standard_hwa_model": ("std3", "HWA 3%", "#0072B2"),
    "standard_hwa_9pct_model": ("std9", "HWA 9%", "#D55E00"),
    "cdt_random_3pct_model": ("cdt3", "Fault-aware HWA 3%", "#009E73"),
}
MODELS = {"reram_cmo": "CMO ReRAM", "pcm": "PCM", "reram_wan2022": "Wan 2022 ReRAM"}
CASES = {"nominal": "Healthy array", "random_30000ppm": "3% permanent random faults"}
TIMES = (None, 1, 86400, 172800)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.collected / "retention.json").read_text())
    assert len(rows) == 72, "Expected the declared 72 rows, excluding the readiness probe."
    index = {(r["hwa_arm"], r["model"], r["case"], r["seconds"]): r for r in rows}
    expected = {(s, m, c, t) for s in SOURCES for m in MODELS for c in CASES for t in TIMES}
    assert set(index) == expected and len(index) == len(rows)
    assert all(r["replay_exact"] and not r["duplicate_coverage"] for r in rows)
    assert all(r["array_seed"] == 271002 and r["logical_weights"] == 28311552 for r in rows)
    assert all(not r["read_noise"] and r["cost_retention_device_writes"] == 0 for r in rows)
    for s in SOURCES:
        for c in CASES:
            baseline, first = (index[s, "reram_wan2022", c, t] for t in (None, 1))
            assert baseline["apparent_sha256"] == first["apparent_sha256"]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for s, (label, _, _) in SOURCES.items():
        for m in MODELS:
            for c in CASES:
                base, final = (index[s, m, c, t] for t in (None, 172800))
                item = {"source": label, "model": m, "case": c,
                        "baseline_seconds": base["baseline_seconds"],
                        "baseline_residual_rms": base["residual_rms"],
                        "two_day_residual_rms": final["residual_rms"]}
                for split in ("development", "transfer_wikitext", "transfer_generated"):
                    for metric in ("teacher_kl", "perplexity"):
                        key = f"retained_{split}_{metric}"
                        item[f"{split}_{metric}_clean"] = base[f"clean_{split}_{metric}"]
                        item[f"{split}_{metric}_baseline"] = base[key]
                        item[f"{split}_{metric}_two_days"] = final[key]
                        item[f"{split}_{metric}_delta"] = final[key] - base[key]
                summary.append(item)
    (args.output / "two_day_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.output / "two_day_summary.csv", summary)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180})
    figures = []
    for split, cohort in (("transfer_wikitext", "WikiText · 1,024 sequences"),
                          ("transfer_generated", "Generated text · 384 sequences")):
        for metric, ylabel in (("teacher_kl", "Teacher KL (nats/token)"), ("perplexity", "Perplexity")):
            key = f"retained_{split}_{metric}"
            fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharey=True, layout="constrained")
            for row_id, (case, case_name) in enumerate(CASES.items()):
                for col_id, (model, model_name) in enumerate(MODELS.items()):
                    ax = axes[row_id, col_id]
                    for source, (_, source_name, color) in SOURCES.items():
                        values = [index[source, model, case, t][key] for t in TIMES]
                        ax.plot(range(4), values, marker="o", markersize=4.5, color=color,
                                linewidth=1.6, linestyle="--", label=source_name)
                    ax.set_yscale("log")
                    ax.set_xticks(range(4), ["Baseline*" if model == "reram_wan2022" else "Baseline", "1 s", "1 day", "2 days"])
                    ax.set_title(f"{model_name}\n{case_name}", fontsize=10)
                    ax.grid(axis="y", which="both", alpha=0.18)
                    if col_id == 0:
                        ax.set_ylabel(ylabel + " · log scale")
            axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)
            fig.suptitle(f"Frozen HWA OPT-125M · six analog MLPs\n{cohort}", fontsize=14)
            fig.supxlabel("Age snapshots (categorical spacing). *Wan baseline reuses its 1 s realization.\n"
                          "One array assignment; no compensation; separate PCM/CMO read noise off.\n"
                          "Dashed lines guide comparison; ReRAM snapshots are not measured device trajectories.", fontsize=9)
            stem = f"{split}_{metric}"
            for extension in ("png", "pdf"):
                fig.savefig(args.output / f"{stem}.{extension}")
            plt.close(fig)
            figures.append(stem)
    (args.output / "report_receipt.json").write_text(json.dumps({
        "coverage": 72, "independent_array_assignments": 1, "analog_decoder_layers": 6,
        "logical_weights": 28311552, "physical_devices": 56623104,
        "figures": figures, "input": str(args.collected / "retention.json"),
        "input_sha256": hashlib.sha256((args.collected / "retention.json").read_bytes()).hexdigest(),
        "report_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["Model-based exploratory comparison", "CMO-trained weights remapped to each technology",
                   "No confidence intervals from a single assignment", "Wan includes total endpoint error"],
    }, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary), "figures": figures, "output": str(args.output)}))


if __name__ == "__main__":
    main()
