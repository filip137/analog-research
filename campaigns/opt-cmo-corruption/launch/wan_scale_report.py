"""Report the Wan-2022 endpoint noise-scale stress pilot from collected native rows."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCES = {"standard_hwa_model": "std3", "standard_hwa_9pct_model": "std9",
           "cdt_random_3pct_model": "cdt3"}
SCALES = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
METHODS = {"global": "Global gain", "row_column": "Row + column LS"}
SERIES = {"none": ("No calibration", "#2a78d6"), "global": ("Global gain", "#eb6834"),
          "row_column": ("Row + column LS", "#1baf7a")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TEACHER_PPL = 46.60196571170268  # FP32 teacher, same WikiText cohort (exp-013 teacher reference).
TWO_DAYS = 172800.0


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first(ks, test):
    return next((k for k in ks if test(k)), None)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("retention", type=Path, help="collected retention directory")
    p.add_argument("compensation", type=Path, help="collected compensation directory")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ret = json.loads((args.retention / "retention.json").read_text())
    comp = json.loads((args.compensation / "compensation.json").read_text())
    assert len(ret) == 24 and len(comp) == 42
    for r in ret + comp:
        assert (r["case"] == "nominal" and r["model"] == "reram_wan2022" and r["array_seed"] == 271002
                and r["failed_devices"] == 0 and not r["read_noise"] and not r["duplicate_coverage"] and r["replay_exact"])
    assert all(r["cost_retention_device_writes"] == 0 and r["cost_retention_verify_reads"] == 0 for r in ret)
    assert all(r["state_unchanged"] and r["cost_device_writes"] == 0 and r["cost_verify_reads"] == 0 for r in comp)
    rindex = {(r["hwa_arm"], r["seconds"], r["noise_scale"]): r for r in ret}
    cindex = {(r["hwa_arm"], r["noise_scale"], r["method"]): r for r in comp}
    assert set(rindex) == {(a, None, 1.0) for a in SOURCES} | {(a, TWO_DAYS, k) for a in SOURCES for k in SCALES}
    assert set(cindex) == {(a, k, m) for a in SOURCES for k in SCALES for m in METHODS}

    rows, markers = [], {}
    for arm, label in SOURCES.items():
        prog = rindex[arm, None, 1.0]
        clean = prog["clean_transfer_wikitext_teacher_kl"]
        base_ppl = prog["retained_transfer_wikitext_perplexity"]
        states = [("programmed", None, prog)] + [(f"x{k:g}", k, rindex[arm, TWO_DAYS, k]) for k in SCALES]
        for name, k, r in states:
            kl = r["retained_transfer_wikitext_teacher_kl"]
            row = {"source": label, "snapshot": name, "noise_scale": k, "weight_error_rms": r["residual_rms"],
                   "change_from_programmed_rms": r["change_from_programmed_rms"],
                   "clean_kl": clean, "kl": kl, "kl_increase_over_clean": kl - clean,
                   "curvature": (kl - clean) / r["residual_rms"] ** 2,
                   "perplexity": r["retained_transfer_wikitext_perplexity"],
                   "perplexity_ratio_to_programmed": r["retained_transfer_wikitext_perplexity"] / base_ppl,
                   "accuracy_percent": r["retained_transfer_wikitext_next_token_accuracy_percent"],
                   "generated_kl": r["retained_transfer_generated_teacher_kl"],
                   "generated_perplexity": r["retained_transfer_generated_perplexity"]}
            for method in METHODS:
                c = cindex.get((arm, k, method))
                for metric in ("teacher_kl", "perplexity", "next_token_accuracy_percent"):
                    row[f"{method}_{metric}"] = None if c is None else c[f"compensated_transfer_wikitext_{metric}"]
                row[f"{method}_kl_regained_percent"] = None if c is None else c["transfer_wikitext_aging_kl_regained_percent"]
                row[f"{method}_generated_kl"] = None if c is None else c["compensated_transfer_generated_teacher_kl"]
            rows.append(row)
        scaled = [r for r in rows if r["source"] == label and r["noise_scale"] is not None]
        c1 = scaled[0]["curvature"]
        by_k = {r["noise_scale"]: r for r in scaled}
        markers[label] = {
            "curvature_at_k1": c1,
            "quadratic_regime_ends_at_k": first(SCALES, lambda k: by_k[k]["curvature"] > 1.5 * c1),
            "collapse_at_k": first(SCALES, lambda k: by_k[k]["perplexity_ratio_to_programmed"] >= 2),
            "programmed_perplexity": base_ppl, "programmed_kl": prog["retained_transfer_wikitext_teacher_kl"]}
    for method in METHODS:
        markers[f"{method}_material_at_k"] = [k for k in SCALES if all(
            (r := next(x for x in rows if x["source"] == s and x["noise_scale"] == k))[f"{method}_kl_regained_percent"] is not None
            and r[f"{method}_kl_regained_percent"] >= 25 for s in SOURCES.values())]

    args.output.mkdir(parents=True, exist_ok=True)
    csv_write(args.output / "wan_scale_summary.csv", rows)
    (args.output / "wan_scale_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "wan_scale_markers.json").write_text(json.dumps(markers, indent=2) + "\n")

    for metric, ylabel, filename, reference in (
            ("kl", "WikiText teacher KL (nats/token)", "wan_scale_teacher_kl", None),
            ("perplexity", "WikiText perplexity", "wan_scale_perplexity", TEACHER_PPL)):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
        for ax, (arm, label) in zip(axes, SOURCES.items()):
            data = [r for r in rows if r["source"] == label]
            prog = data[0]
            scaled = data[1:]
            ks = [r["noise_scale"] for r in scaled]
            for key, (name, color) in SERIES.items():
                column = metric if key == "none" else f"{key}_{'teacher_kl' if metric == 'kl' else 'perplexity'}"
                ax.plot(ks, [r[column] for r in scaled], color=color, linewidth=2, marker="o", markersize=4.5,
                        markeredgecolor="white", markeredgewidth=1, label=name, zorder=3)
            ax.axhline(prog[metric], color=MUTED, linewidth=1, linestyle="--", zorder=2)
            ax.annotate("programmed", (ks[-1], prog[metric]), xytext=(0, -3), textcoords="offset points",
                        ha="right", va="top", fontsize=7.5, color=MUTED)
            if reference is not None:
                ax.axhline(reference, color=MUTED, linewidth=1, linestyle=":", zorder=2)
                ax.annotate("FP32 teacher", (ks[0], reference), xytext=(0, -3), textcoords="offset points",
                            ha="left", va="top", fontsize=7.5, color=MUTED)
            ax.set_yscale("log")
            ax.set_xscale("log", base=2)
            ax.set_xticks(SCALES, [f"{k:g}" for k in SCALES])
            ax.minorticks_off()
            ax.set_title(label, fontsize=10, color=INK, loc="left")
            ax.set_xlabel("Wan two-day σ multiplier k", fontsize=8.5, color=INK)
            style(ax)
        axes[0].set_ylabel(ylabel, fontsize=8.5, color=INK)
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        save(fig, args.output / filename)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, label in zip(axes, SOURCES.values()):
        data = [r for r in rows if r["source"] == label]
        x = [r["weight_error_rms"] ** 2 for r in data]
        y = [r["kl_increase_over_clean"] for r in data]
        c1 = markers[label]["curvature_at_k1"]
        span = [min(x) * 0.8, max(x) * 1.2]
        ax.plot(span, [c1 * v for v in span], color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.text(0.03, 0.97, f"dashed: ΔKL = {c1:.1f}·σ² (k = 1 curvature)", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color=MUTED)
        ax.plot(x, y, color=SERIES["none"][1], linewidth=2, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=1, zorder=3)
        for r, xv, yv in zip(data, x, y):
            if r["snapshot"] in ("programmed", "x1", "x2", "x4", "x8"):
                ax.annotate(r["snapshot"].replace("x", "k=") if r["noise_scale"] else "prog.", (xv, yv),
                            xytext=(4, -9), textcoords="offset points", fontsize=7, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("σ² (RMS weight error vs target, squared)", fontsize=8.5, color=INK)
        style(ax)
    axes[0].set_ylabel("Teacher KL increase over clean source", fontsize=8.5, color=INK)
    save(fig, args.output / "wan_scale_curvature")
    print(json.dumps(markers, indent=2))


if __name__ == "__main__":
    main()
