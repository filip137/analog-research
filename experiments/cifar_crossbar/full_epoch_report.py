"""Coverage-checked descriptive report for the full-cohort PCM/OM comparison."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from experiments.artifacts import atomic_write_json, sha256_file

METHODS = ("none", "calibration", "rewrite", "onchip_weights", "onchip_calibration")
CASES = ("nominal", "open_10000ppm", "gmax_10000ppm", "random_10000ppm")
SOURCES = ("digital", "standard_hwa")
LABELS = dict(none="deployment", calibration="calibration", rewrite="rewrite + calibration",
              onchip_weights="weight recovery", onchip_calibration="joint recovery")


def case_label(case, backend):
    if case == "nominal":
        return case
    kind = case.split("_")[0]
    labels = {"open": "stuck-low", "gmax": "stuck-high", "random": "random-stuck"} if backend == "om" else {"open": "open", "gmax": "Gmax", "random": "random-stuck"}
    return labels[kind] + " 1%"


def validate(result):
    if result["smoke_only"] or result["recovery_images"] != 45000 or result["epochs"] != 5:
        raise ValueError("Expected full five-epoch coverage.")
    rows = result["measurements"]
    keys = {(r["source"], r["case"], r["method"]) for r in rows}
    if len(rows) != 40 or keys != {(s, c, m) for s in SOURCES for c in CASES for m in METHODS}:
        raise ValueError("Missing or duplicate controls.")
    for source in SOURCES:
        for case in CASES:
            paired = [r for r in rows if r["source"] == source and r["case"] == case]
            for key in ("p0_sha256", "initial_apparent_sha256", "identity", "initial_test"):
                if len({json.dumps(r[key], sort_keys=True) for r in paired}) != 1:
                    raise ValueError(f"Unpaired {key}.")
            for r in paired:
                if r["initial_test"]["examples"] != 10000:
                    raise ValueError("Incomplete deployment test.")
                if r["method"] == "none":
                    if r["curve"]:
                        raise ValueError("No-recovery arm changed.")
                    continue
                if [x["epoch"] for x in r["curve"]] != [1, 2, 3, 4, 5]:
                    raise ValueError("Incomplete epochs.")
                for point in r["curve"]:
                    e, cost = point["epoch"], point["cost"]
                    if point["image_presentations"] != 45000 * e or point["test"]["examples"] != 10000:
                        raise ValueError("Wrong data budget.")
                    if result["backend"] == "pcm":
                        if cost["array_reprogram_calls"] != (0 if r["method"] == "calibration" else 704 * e):
                            raise ValueError("Unmatched PCM programming budget.")
                    elif cost["max_recovery_cell_pulses"] > min(640, 704 * e):
                        raise ValueError("OM pulse budget exceeded.")
    return rows


def csv_file(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        if len(values) != 3 or len({v["array_seed"] for v in values}) != 3:
            raise ValueError("Expected exactly three independent arrays per comparison.")
        item = dict(zip(keys, key))
        for metric in metrics:
            numbers = [v[metric] for v in values]
            item[metric + "_mean"] = statistics.mean(numbers)
            item[metric + "_sd"] = statistics.stdev(numbers)
        result.append(item)
    return result


def report(root, output, backends=("pcm", "om")):
    root, output = Path(root), Path(output)
    if not backends or len(set(backends)) != len(backends) or not set(backends) <= {"pcm", "om"}:
        raise ValueError("Expected distinct declared device backends.")
    output.mkdir(parents=True, exist_ok=True)
    flat, paired, provenance, hwa, costs, cross_source = [], [], [], [], [], []
    clean_sources = {}
    for dataset in ("cifar10", "cifar100"):
        for backend in backends:
            study = root / "results" / f"{dataset}-{backend}-five-full-epochs-v1"
            execution = json.loads((study / "campaign-execution.json").read_text())
            if len(execution["completed"]) != 5 or execution["active"] is not None:
                raise ValueError("Incomplete campaign.")
            for node, entry in execution["completed"].items():
                # Preserve receipts; relocate only when reading the collected mirror.
                relative = Path(entry["run_dir"]).relative_to("/home/filip/cifar_full_epoch_sources/v2")
                run = root / relative
                path = run / "result.json"
                if sha256_file(path) != entry["result_sha256"]:
                    raise ValueError("Native result digest changed.")
                value = json.loads(path.read_text())["metrics"]
                provenance.append({"dataset": dataset, "backend": backend, "node": node, "path": str(path), "sha256": sha256_file(path)})
                if value["stage"] == "sources":
                    hwa.append({"dataset": dataset, "backend": backend, **value["sources"]["standard_hwa"]})
                if value["stage"] != "screen":
                    continue
                rows = validate(value)
                if {r["array_seed"] for r in rows} != {151000 + int(node[-1])}:
                    raise ValueError("Wrong confirmation-array namespace.")
                by_key = {(r["source"], r["case"], r["method"]): r for r in rows}
                for r in rows:
                    base = {k: r[k] for k in ("source", "case", "array_seed", "method")}
                    base.update(dataset=dataset, backend=backend)
                    source_key = f"{dataset}/{backend}/{r['source']}"
                    if source_key in clean_sources and clean_sources[source_key] != r["clean"]:
                        raise ValueError("Off-chip source metrics changed between cases/arrays.")
                    clean_sources[source_key] = r["clean"]
                    if r["curve"]:
                        final_cost = r["curve"][-1]["cost"]
                        cost_row = {**base, **final_cost}
                        if backend == "om":
                            cost_row["capped_fraction"] = final_cost["capped_cells"] / r["identity"]["physical_devices"]
                        costs.append(cost_row)
                    points = [{"epoch": 0, "test": r["initial_test"], "cost": {}, "image_presentations": 0}] + r["curve"]
                    for point in points:
                        flat.append({**base, "epoch": point["epoch"], "image_presentations": point["image_presentations"],
                                     "clean_source_accuracy_percent": r["clean"]["accuracy_percent"],
                                     "clean_source_teacher_kl": r["clean"]["teacher_kl"],
                                     "deployment_accuracy_drop_pp": r["clean"]["accuracy_percent"] - r["initial_test"]["accuracy_percent"],
                                     "deployment_kl_increase": r["initial_test"]["teacher_kl"] - r["clean"]["teacher_kl"],
                                     **{k: point["test"][k] for k in ("teacher_kl", "accuracy_percent", "teacher_agreement_percent")},
                                     **point["cost"],
                                     **({"persistent_kl": point["persistent_test"]["teacher_kl"], "persistent_accuracy": point["persistent_test"]["accuracy_percent"]} if "persistent_test" in point else {})})
                    if r["method"] not in ("onchip_weights", "onchip_calibration"):
                        continue
                    for control in ("none", "calibration", "rewrite"):
                        other = by_key["standard_hwa", r["case"], control]
                        for current in r["curve"]:
                            a = current["test"]
                            b = other["initial_test"] if control == "none" else other["curve"][current["epoch"]-1]["test"]
                            cross_source.append({**base, "control_source": "standard_hwa", "control": control,
                                                 "epoch": current["epoch"],
                                                 "kl_reduction": b["teacher_kl"] - a["teacher_kl"],
                                                 "accuracy_gain_pp": a["accuracy_percent"] - b["accuracy_percent"]})
                    for control in ("calibration", "rewrite"):
                        other = by_key[r["source"], r["case"], control]
                        for current, prior in zip(r["curve"], other["curve"]):
                            a, b = current["test"], prior["test"]
                            paired.append({**base, "control": control, "epoch": current["epoch"],
                                           "kl_reduction": b["teacher_kl"] - a["teacher_kl"],
                                           "accuracy_gain_pp": a["accuracy_percent"] - b["accuracy_percent"]})
    summary = summarize(flat, ("dataset", "backend", "source", "case", "method", "epoch"), ("teacher_kl", "accuracy_percent", "teacher_agreement_percent"))
    pairs = summarize(paired, ("dataset", "backend", "source", "case", "method", "control", "epoch"), ("kl_reduction", "accuracy_gain_pp"))
    cross_summary = summarize(cross_source, ("dataset", "backend", "source", "case", "method", "control_source", "control", "epoch"), ("kl_reduction", "accuracy_gain_pp"))
    csv_file(output / "per_array.csv", flat)
    csv_file(output / "summary.csv", summary)
    csv_file(output / "paired.csv", paired)
    csv_file(output / "paired_summary.csv", pairs)
    csv_file(output / "final_costs.csv", costs)
    csv_file(output / "cross_source.csv", cross_source)
    csv_file(output / "cross_source_summary.csv", cross_summary)
    studies = 2 * len(backends)
    value = dict(controls=120 * studies, learning_trajectories=96 * studies, post_recovery_test_endpoints=480 * studies,
                 backends=list(backends),
                 recovery_images=45000, epochs=5, summary=summary, paired=pairs,
                 provenance=provenance, hwa=hwa, costs=costs, cross_source=cross_summary,
                 clean_sources=clean_sources,
                 analysis_code_sha256=sha256_file(Path(__file__)))
    atomic_write_json(output / "results.json", value)
    text = ["# Five full recovery epochs: " + ", ".join(b.upper() for b in backends), "", "Exploratory results. Each epoch uses all 45,000 adaptation images; all 10,000 test images evaluated. Mean ± sample SD over three independent arrays. KL is KL(original digital teacher || student), nats at T=1. Fixed epoch 5 is the endpoint; no test selection.", "",
            "These runs use digital and standard device-specific HWA sources. Previous PCM noise-enhanced HWA/CDT arms are not repeated here. OM uses one limited 30-epoch HWA recipe. PCM recovery fully reprograms Gaussian endpoints; OM uses capped incremental pulse Adam. Their write costs and fault encodings differ. Five full epochs have 225,000 image presentations, versus 100,000 in the previous 20 passes over 5,000 images; this is not an isolated epoch-count comparison.", ""]
    for dataset in ("cifar10", "cifar100"):
        for backend in backends:
            clean = clean_sources[f"{dataset}/{backend}/standard_hwa"]
            text += [f"## {dataset.upper()} / {backend.upper()}", "",
                     f"Before programming, the selected HWA source has accuracy {clean['accuracy_percent']:.2f}% and teacher KL {clean['teacher_kl']:.5f}. The deployment rows below measure the additional effect of programming onto fresh arrays.", "",
                     "| Case | Source | Method | KL | Accuracy (%) |", "|---|---|---|---:|---:|"]
            for row in summary:
                if row["dataset"] != dataset or row["backend"] != backend or not ((row["epoch"] == 0 and row["method"] == "none") or row["epoch"] == 5):
                    continue
                text.append(f"| {case_label(row['case'], backend)} | {row['source']} | {LABELS[row['method']]} | {row['teacher_kl_mean']:.5f} ± {row['teacher_kl_sd']:.5f} | {row['accuracy_percent_mean']:.2f} ± {row['accuracy_percent_sd']:.2f} |")
            text += ["", f"![Learning curves]({dataset}_{backend}.png)", ""]
    (output / "report.md").write_text("\n".join(text) + "\n")
    plot(summary, output)
    return value


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(calibration="#555555", rewrite="#d89000", onchip_weights="#0072b2", onchip_calibration="#009e73")
    for dataset in ("cifar10", "cifar100"):
        for backend in sorted({r["backend"] for r in summary}):
            fig, axes = plt.subplots(2, 4, figsize=(15, 7))
            fig.subplots_adjust(left=.07, right=.99, top=.9, bottom=.19, wspace=.35, hspace=.3)
            for column, case in enumerate(CASES):
                for source in SOURCES:
                    for method, color in colors.items():
                        rows = sorted([r for r in summary if (r["dataset"], r["backend"], r["case"], r["source"], r["method"]) == (dataset, backend, case, source, method)], key=lambda r: r["epoch"])
                        for ax, metric in zip(axes[:, column], ("teacher_kl", "accuracy_percent")):
                            ax.errorbar([r["epoch"] for r in rows], [r[metric+"_mean"] for r in rows],
                                        yerr=[r[metric+"_sd"] for r in rows], color=color, linestyle="-" if source=="standard_hwa" else "--", linewidth=1.4,
                                        label=f"{source.replace('_', ' ')}: {LABELS[method]}", capsize=2)
                            ax.grid(alpha=.2)
                            ax.set_xticks(range(6))
                axes[0,column].set_title(case_label(case, backend))
                axes[0,column].set_yscale("log")
                axes[1,column].set_xlabel("Full adaptation epoch")
            axes[0,0].set_ylabel("Teacher KL (nats, log scale)")
            axes[1,0].set_ylabel("Test accuracy (%)")
            handles, labels = axes[0,0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01), ncols=4, fontsize=8)
            fig.suptitle(f"{dataset.upper()} / {backend.upper()} — 45,000 adaptation images, 3 arrays")
            for ext in ("png", "pdf"):
                fig.savefig(output / f"{dataset}_{backend}.{ext}", dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", choices=("pcm", "om"), nargs="+", default=("pcm", "om"))
    args = parser.parse_args()
    result = report(args.root, args.output, args.backends)
    print(json.dumps({k: result[k] for k in ("controls", "learning_trajectories", "post_recovery_test_endpoints")}))
