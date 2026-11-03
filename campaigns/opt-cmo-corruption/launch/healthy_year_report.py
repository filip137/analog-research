"""Report the predeclared healthy-array one-year extrapolation and prior ages."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCES = {"standard_hwa_model": "std3", "standard_hwa_9pct_model": "std9",
           "cdt_random_3pct_model": "cdt3"}
MODELS = {"pcm": "PCM", "reram_cmo": "CMO ReRAM"}
METHODS = {"global": "Global", "tile_global": "Tile", "row": "Row", "column": "Column",
           "row_column": "Row + column", "global_shift": "Global + shift",
           "global_moments": "Global + moments"}
YEAR = 31536000


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("collected", type=Path)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = args.collected / "compensation.json"
    rows = json.loads(source.read_text())
    old = json.loads(args.previous.read_text())
    expected = {(s, m, a) for s in SOURCES for m in MODELS for a in METHODS}
    index = {(r["hwa_arm"], r["model"], r["method"]): r for r in rows}
    assert len(rows) == len(index) == 42 and set(index) == expected
    assert all(r["case"] == "nominal" and r["failed_devices"] == 0 and r["seconds"] == YEAR
               and r["array_seed"] == 271002 and not r["read_noise"] and not r["duplicate_coverage"]
               and r["replay_exact"] and r["state_unchanged"] and r["cost_device_writes"] == 0
               and r["cost_verify_reads"] == 0 for r in rows)
    prior = [r for r in old if r["case"] == "nominal" and r["model"] in MODELS]
    prior_index = {(r["hwa_arm"], r["model"], r["seconds"], r["method"]): r for r in prior}
    for r in rows:
        ref = prior_index[r["hwa_arm"], r["model"], 172800, r["method"]]
        assert r["source_sha256"] == ref["source_sha256"]
        assert all(v == ref[k] for k, v in r.items() if k.startswith("baseline_"))
    args.output.mkdir(parents=True, exist_ok=True)
    full = [dict(source=SOURCES[r["hwa_arm"]], **r) for r in rows]
    csv_write(args.output / "one_year_recovery.csv", full)
    (args.output / "one_year_recovery.json").write_text(json.dumps(full, indent=2) + "\n")
    summary = []
    for model in MODELS:
        for arm, label in SOURCES.items():
            r = index[arm, model, "global"]
            s = {"model": model, "source": label}
            for metric in ("teacher_kl", "perplexity", "next_token_accuracy_percent"):
                key = f"transfer_wikitext_{metric}"
                for state in ("baseline", "uncompensated"):
                    s[f"{state}_{metric}"] = r[f"{state}_{key}"]
                for method in METHODS:
                    s[f"{method}_{metric}"] = index[arm, model, method][f"compensated_{key}"]
            for method in METHODS:
                s[f"{method}_aging_kl_regained_percent"] = index[arm, model, method]["transfer_wikitext_aging_kl_regained_percent"]
            summary.append(s)
    csv_write(args.output / "one_year_wikitext_summary.csv", summary)
    (args.output / "one_year_wikitext_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    combined = prior + rows
    merged = {(r["hwa_arm"], r["model"], r["seconds"], r["method"]): r for r in combined}
    assert len(merged) == 168
    (args.output / "healthy_all_ages.json").write_text(json.dumps(combined, indent=2) + "\n")
    ages = (1, 86400, 172800, YEAR)
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 180})
    figures = []
    for metric, title in (("teacher_kl", "Teacher KL (nats/token)"), ("perplexity", "Perplexity")):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8))
        for iy, (model, model_label) in enumerate(MODELS.items()):
            for ix, (arm, label) in enumerate(SOURCES.items()):
                ax = axes[iy, ix]
                key = f"transfer_wikitext_{metric}"
                r = index[arm, model, "global"]
                ax.axhline(r[f"baseline_{key}"], color="gray", linestyle=":", label="Programmed baseline")
                ax.plot(ages, [merged[arm, model, t, "global"][f"uncompensated_{key}"] for t in ages],
                        color="black", marker="o", linestyle="--", label="Uncompensated")
                for method, color in (("global", "#0072B2"), ("row_column", "#D55E00"), ("global_moments", "#009E73")):
                    ax.plot(ages, [merged[arm, model, t, method][f"compensated_{key}"] for t in ages],
                            color=color, marker=".", linestyle="--", label=METHODS[method])
                ax.set_xscale("log")
                if metric == "teacher_kl":
                    ax.set_yscale("log")
                ax.set_xticks((1, 86400, YEAR), ("1 s", "1 day", "1 year"))
                ax.set_title(f"{model_label} · {label}")
                ax.grid(alpha=.2)
                if ix == 0:
                    ax.set_ylabel(title)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=5, frameon=False)
        fig.suptitle(f"Healthy arrays · one-year model extrapolation · {title}")
        fig.text(.5, .02, "OPT-125M, six analog MLPs; one assignment; read noise off; zero adaptation writes.\n"
                 "Points are evaluated ages; dashed lines guide the eye. Wan has no one-year fit.",
                 fontsize=9, ha="center", va="bottom")
        fig.tight_layout(rect=(0, .14, 1, .95))
        name = f"healthy_{metric}_over_time"
        for ext in ("png", "pdf"):
            fig.savefig(args.output / f"{name}.{ext}")
        plt.close(fig)
        figures.append(name)
    receipt = {"one_year_rows": len(rows), "all_age_rows": len(combined), "figures": figures,
               "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "prior_sha256": hashlib.sha256(args.previous.read_bytes()).hexdigest(),
               "report_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "assignments": 1, "models_unsupported_at_one_year": ["reram_wan2022"],
               "selection": "All predeclared methods reported; no held-out selection."}
    (args.output / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    for row in summary:
        print(json.dumps({k: v for k, v in row.items() if k in ("model", "source")
                          or k.startswith(("baseline_", "uncompensated_", "global_", "row_column_"))}))


if __name__ == "__main__":
    main()
