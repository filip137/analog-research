"""Render standalone scientific figures from completed, immutable EBL bundles."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


def render(runs, output, label):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    sources = []
    for run in map(Path, runs):
        native = json.loads((run/"result.json").read_text())
        if native["status"] != "complete":
            raise ValueError(f"Incomplete run: {run}")
        path = run/"artifacts/summary.json"
        case, = json.loads(path.read_text())["cases"]
        summaries.append((run, case))
        sources.append({"run": str(run.resolve()), "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    names = {"mixed_relu": "One active ReLU", "active_relu": "Two active ReLUs", "softmax_only": "Softmax alone"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, len(summaries), figsize=(4.3*len(summaries), 6.4), squeeze=False,
                              layout="constrained")
    for column, (run, case) in enumerate(summaries):
        directory = run/"artifacts"/case["case"]
        for start, color in enumerate(("#0072B2", "#009E73", "#E69F00")):
            trace = np.load(directory/f"numerical-start-{start}.npz")
            gap = trace["energy"]-trace["energy"][-1]
            select = trace["time"] <= .025
            axes[0, column].semilogy(trace["time"][select]*1000, np.maximum(gap[select], 1e-14),
                                    color=color, label=f"Start {start+1}")
        axes[0, column].set(title=names.get(case["case"], case["case"]), xlabel="Time (ms)",
                            ylabel="Energy above equilibrium", ylim=(1e-14, 10))
        rows = sorted(case["metrics"]["beta_curve"], key=lambda row: row["beta"])
        for kind, color, marker, caption in (("centered", "#0072B2", "o", "Centered EqProp"),
                                             ("one_sided", "#E69F00", "s", "One-sided EqProp")):
            errors = [max(item["relative_l2"] or 0 for item in row[kind].values()) for row in rows]
            axes[1, column].loglog([row["beta"] for row in rows], errors, marker=marker, color=color, label=caption)
        spice = max(item["relative_l2"] or 0 for item in case["metrics"]["spice_gradient"].values())
        axes[1, column].scatter([rows[0]["beta"]], [spice], marker="D", s=80, facecolors="none",
                               edgecolors="#CC3311", linewidth=1.6, label="SPICE centered EqProp", zorder=5)
        axes[1, column].set(xlabel="Nudge magnitude beta", ylabel="Worst gradient relative L2 error")
        axes[1, column].set_xticks([row["beta"] for row in rows], [f"{row['beta']:g}" for row in rows])
        axes[1, column].xaxis.set_minor_formatter(NullFormatter())
        for ax in axes[:, column]:
            ax.grid(alpha=.16, which="both")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Reciprocal analog attention — {label}\nIdeal components; gradient errors compared with implicit equilibrium differentiation", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"energy-and-eqprop.{suffix}", dpi=180)
    plt.close(fig)
    (output/"figure-provenance.json").write_text(json.dumps({"label": label, "sources": sources,
        "matplotlib": matplotlib.__version__, "numpy": np.__version__}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    render(args.runs, args.output_dir, args.label)
