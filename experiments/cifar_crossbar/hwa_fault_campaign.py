"""Prepare and analyze matched HWA/CDT comparisons using the existing runner."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.analysis import records
from experiments.cifar_crossbar.fault_config import METHODS
from experiments.cifar_crossbar.hwa_fault_config import EXPERIMENT_ID, default_config


def prepare(root, dataset, *, device="cuda:0", disable_cudnn=False, smoke=False):
    root = Path(root).resolve()
    if (root / "campaign.json").exists():
        raise ValueError("Cannot overwrite a frozen HWA comparison.")
    common = dict(
        dataset=dataset, device=device, disable_cudnn=disable_cudnn, cpu_threads=4
    )
    if smoke:
        common.update(
            max_examples=256,
            hwa_epochs=1,
            extension_epochs=1,
            validation_every=1,
            hwa_learning_rates=(0.0001,),
            hwa_sgd_learning_rates=(0.05,),
            hwa_objectives=("teacher_kl",),
            hwa_noise_scales=(1.0,),
            cdt_rates=(0.01,),
            development_arrays=1,
            tuning_arrays=1,
            recovery_learning_rates=(0.0001,),
            calibration_learning_rates=(0.0003,),
        )
    nodes = []
    for stage in ("cache", "fit", "tune", "screen1", "screen2", "screen3"):
        actual = "screen" if stage.startswith("screen") else stage
        index = int(stage[-1]) if actual == "screen" else 1
        if smoke and actual == "screen" and index > 1:
            continue
        name = f"{dataset}_{stage}"
        spec = default_config(
            **common,
            stage=actual,
            array_seed=81000 + index,
            endpoint_seed=91000 + index,
        )
        path = root / "configs" / f"{name}.json"
        atomic_write_json(path, spec)
        inputs = {}
        if stage != "cache":
            inputs["device-data"] = {
                "node": f"{dataset}_cache",
                "artifact": "checkpoints/cache.pt",
            }
        if actual in ("tune", "screen"):
            inputs["weights"] = {
                "node": f"{dataset}_fit",
                "artifact": "checkpoints/sources.pt",
            }
        if actual == "screen":
            inputs["weights"] = {
                "node": f"{dataset}_tune",
                "artifact": "checkpoints/recovery.pt",
            }
        nodes.append(
            dict(
                id=name,
                dataset=dataset,
                config=str(path.relative_to(root)),
                sha256=sha256_file(path),
                depends=sorted({x["node"] for x in inputs.values()}),
                inputs=inputs,
            )
        )
    study_id = f"{dataset}-pcm-hwa-cdt-{'canary' if smoke else 'comparison'}-v1"
    plan = {
        "schema_version": 1,
        "study_id": study_id,
        "title": f"{dataset}: generic HWA/CDT and matched one-pass PCM recovery",
        "evidence_class": "exploratory",
        "hypothesis": "One recovery pass on a fresh fixed faulty array can reduce teacher KL beyond development-selected HWA/CDT, calibration and equally frequent unchanged-target rewriting.",
        "motivation": "Test whether the recovery seen from digital weights persists after generic fault-aware HWA inspired by Li et al. (2023).",
        "arms": [
            dict(
                arm_id=n["id"],
                configs=[n["config"]],
                description=n["id"],
                experiment_id=EXPERIMENT_ID,
                mode="train",
            )
            for n in nodes
        ],
        "completion_criteria": [
            "Complete every declared native stage and paired control.",
            "Full run: digital audit, 45000-image augmented HWA, development-only source/recovery selection, three fresh final arrays and all 10000 test images.",
            "Each final array has 42 source/case combinations and five controls (210 measurements).",
            "Record late HWA convergence flags rather than claiming converged robustness from an unresolved fit.",
        ],
        "analysis_plan": [
            "Primary teacher KL in nats at T=1; also accuracy, clean-source fidelity, paired recovery reductions and array spread.",
            "Report digital, standard HWA, noise-strength-selected HWA and per-failure CDT with exactly the same suffix/deployment law.",
            "CDT uses independently resampled physical-bank failures each minibatch; final masks never enter selection.",
            "Recover for one pass over 5000 images; calibration and weight learning rates selected only on independent development arrays.",
            "Exploratory inference conditional on one HWA/data-order seed. Preserve negative outcomes and report any unconverged selection; no superiority to the full paper's peripheral/retention model is claimed.",
        ],
    }
    atomic_write_json(root / "study-plan.json", plan)
    campaign = {
        "schema": "cifar_crossbar.campaign.v1",
        "study_id": study_id,
        "phase": "hwa_fault_comparison",
        "plan": "study-plan.json",
        "plan_sha256": sha256_file(root / "study-plan.json"),
        "selection_inputs": {},
        "nodes": nodes,
    }
    atomic_write_json(root / "campaign.json", campaign)
    return campaign


def write_csv(path, rows):
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cross_source_comparisons(rows):
    """Pair every recovery source with every applicable HWA control.

    All contrasts are retained; this is not a test-set model-selection rule.
    Pairing follows physical array identity, not the order of input records.
    """
    cases = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        case = tuple(row[k] for k in ("dataset", "fault_kind", "fault_rate_ppm"))
        key = (row["source"], row["method"])
        if row["array_seed"] in cases[case][key]:
            raise ValueError("Duplicate array in a cross-source contrast.")
        cases[case][key][row["array_seed"]] = row
    result = []
    for case, arms in sorted(cases.items()):
        for (source, method), candidate in sorted(arms.items()):
            if method not in ("onchip_weights", "onchip_calibration"):
                continue
            for (control_source, control_method), control in sorted(arms.items()):
                if control_source == "digital" or control_method not in (
                    "none",
                    "calibration",
                    "rewrite",
                ):
                    continue
                if set(candidate) != set(control) or len(candidate) != 3:
                    raise ValueError(
                        "Cross-source contrasts require three matched arrays."
                    )
                kl, accuracy = [], []
                for seed in sorted(candidate):
                    a, b = candidate[seed], control[seed]
                    if a["fault_identity_sha256"] != b["fault_identity_sha256"]:
                        raise ValueError(
                            "Cross-source contrast has unmatched physical faults."
                        )
                    kl.append(b["teacher_kl"] - a["teacher_kl"])
                    accuracy.append(a["accuracy_percent"] - b["accuracy_percent"])
                result.append(
                    {
                        **dict(zip(("dataset", "fault_kind", "fault_rate_ppm"), case)),
                        "source": source,
                        "method": method,
                        "control_source": control_source,
                        "control_method": control_method,
                        "arrays": 3,
                        "mean_kl_reduction": float(np.mean(kl)),
                        "std_kl_reduction": float(np.std(kl, ddof=1)),
                        "arrays_kl_improved": sum(x > 0 for x in kl),
                        "mean_accuracy_gain_pp": float(np.mean(accuracy)),
                        "std_accuracy_gain_pp": float(np.std(accuracy, ddof=1)),
                        "arrays_accuracy_improved": sum(x > 0 for x in accuracy),
                    }
                )
    return result


def report(executions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    flat = []
    fits = {}
    tuning = {}
    sources = []
    for execution in executions:
        campaign, runs = records(execution)
        if campaign["phase"] != "hwa_fault_comparison":
            raise ValueError("Expected an HWA/CDT comparison.")
        sources.append(
            {"path": str(Path(execution).resolve()), "sha256": sha256_file(execution)}
        )
        for run in runs:
            c, m = run["config"], run["metrics"]
            ds = c["dataset"]
            if m["smoke_only"]:
                raise ValueError("Cannot report a canary as full comparison evidence.")
            if c["stage"] == "fit":
                if ds in fits:
                    raise ValueError("Duplicate dataset fit stage.")
                fits[ds] = m
            if c["stage"] == "tune":
                tuning[ds] = m
            if c["stage"] != "screen":
                continue
            if (
                m["evaluation_images"] != 10000
                or m["recovery_images"] != 5000
                or m["completed_controls"] != 210
                or len(m["measurements"]) != 210
            ):
                raise ValueError("Incomplete source/case/control or data coverage.")
            for row in m["measurements"]:
                expected_images = 0 if row["method"] == "none" else 5000
                expected_calls = (
                    79
                    if row["method"]
                    in ("rewrite", "onchip_weights", "onchip_calibration")
                    else 0
                )
                if (
                    row["image_presentations"] != expected_images
                    or row["cost"]["array_reprogram_calls"] != expected_calls
                ):
                    raise ValueError(
                        "Recovery budget differs from the declared comparison."
                    )
                flat.append(
                    {
                        "dataset": ds,
                        "source": row["source"],
                        "fault_kind": row["fault_kind"],
                        "fault_rate_ppm": row["fault_rate_ppm"],
                        "array_seed": row["array_seed"],
                        "method": row["method"],
                        "teacher_kl": row["final"]["teacher_kl"],
                        "accuracy_percent": row["final"]["accuracy_percent"],
                        "initial_teacher_kl": row["initial"]["teacher_kl"],
                        "clean_teacher_kl": row["clean"]["teacher_kl"],
                        "clean_accuracy_percent": row["clean"]["accuracy_percent"],
                        "p0_sha256": row["p0_sha256"],
                        "initial_apparent_sha256": row["initial_apparent_sha256"],
                        "fault_identity_sha256": row["faults"]["identity_sha256"],
                        "run": str(run["run"]),
                    }
                )
    paired = defaultdict(dict)
    physical = defaultdict(set)
    for row in flat:
        key = tuple(
            row[k]
            for k in ("dataset", "source", "fault_kind", "fault_rate_ppm", "array_seed")
        )
        if row["method"] in paired[key]:
            raise ValueError("Duplicate paired method.")
        paired[key][row["method"]] = row
        physical[
            (
                row["dataset"],
                row["fault_kind"],
                row["fault_rate_ppm"],
                row["array_seed"],
            )
        ].add(row["fault_identity_sha256"])
    if any(len(x) != 1 for x in physical.values()):
        raise ValueError("Sources were deployed on different physical faults.")
    expected = set()
    for ds in fits:
        for source in (
            "digital",
            "standard_hwa",
            "noise_hwa",
            "cdt_open",
            "cdt_gmax",
            "cdt_random",
        ):
            kinds = (
                (source[4:],)
                if source.startswith("cdt_")
                else ("open", "gmax", "random")
            )
            for kind, ppm in [("none", 0)] + [
                (k, p) for k in kinds for p in (100, 1000, 10000)
            ]:
                for seed in (81001, 81002, 81003):
                    expected.add((ds, source, kind, ppm, seed))
    if set(paired) != expected or set(tuning) != set(fits) or not expected:
        raise ValueError(
            "Comparison does not have its exact declared source/case/array coverage."
        )
    differences = defaultdict(list)
    for key, methods in paired.items():
        if set(methods) != set(METHODS):
            raise ValueError("Missing control.")
        for field in ("p0_sha256", "initial_apparent_sha256", "initial_teacher_kl"):
            if len({r[field] for r in methods.values()}) != 1:
                raise ValueError("Controls do not restore identical P0.")
        for candidate in ("onchip_weights", "onchip_calibration"):
            for control in ("none", "calibration", "rewrite"):
                differences[(*key[:-1], candidate, control)].append(
                    methods[control]["teacher_kl"] - methods[candidate]["teacher_kl"]
                )
    grouped = defaultdict(list)
    for row in flat:
        grouped[
            tuple(
                row[k]
                for k in ("dataset", "source", "fault_kind", "fault_rate_ppm", "method")
            )
        ].append(row)
    summary = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != 3 or len({r["array_seed"] for r in rows}) != 3:
            raise ValueError("Expected three independent final arrays.")
        if any(
            not np.isfinite(r[k])
            for r in rows
            for k in (
                "teacher_kl",
                "accuracy_percent",
                "clean_teacher_kl",
                "clean_accuracy_percent",
            )
        ):
            raise ValueError("Nonfinite comparison metric.")
        if any(
            len({r[k] for r in rows}) != 1
            for k in ("clean_teacher_kl", "clean_accuracy_percent")
        ):
            raise ValueError("Clean source metrics differ across physical arrays.")
        summary.append(
            {
                **dict(
                    zip(
                        ("dataset", "source", "fault_kind", "fault_rate_ppm", "method"),
                        key,
                    )
                ),
                "arrays": 3,
                "teacher_kl_mean": float(np.mean([r["teacher_kl"] for r in rows])),
                "teacher_kl_std": float(
                    np.std([r["teacher_kl"] for r in rows], ddof=1)
                ),
                "accuracy_mean": float(np.mean([r["accuracy_percent"] for r in rows])),
                "accuracy_std": float(
                    np.std([r["accuracy_percent"] for r in rows], ddof=1)
                ),
                "clean_teacher_kl": rows[0]["clean_teacher_kl"],
                "clean_accuracy_percent": rows[0]["clean_accuracy_percent"],
            }
        )
    comparisons = []
    for key, values in sorted(differences.items()):
        if len(values) != 3:
            raise ValueError("Expected three paired reductions.")
        comparisons.append(
            {
                **dict(
                    zip(
                        (
                            "dataset",
                            "source",
                            "fault_kind",
                            "fault_rate_ppm",
                            "candidate",
                            "control",
                        ),
                        key,
                    )
                ),
                "mean_kl_reduction": float(np.mean(values)),
                "std_kl_reduction": float(np.std(values, ddof=1)),
                "arrays_improved": sum(v > 0 for v in values),
            }
        )
    write_csv(output / "per_array.csv", flat)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_comparisons.csv", comparisons)
    across_sources = cross_source_comparisons(flat)
    write_csv(output / "cross_source_comparisons.csv", across_sources)
    fit_columns = (
        "id",
        "optimizer",
        "objective",
        "learning_rate",
        "noise_scale",
        "fault_kind",
        "fault_rate",
        "selected_epoch",
        "epochs_run",
        "convergence_review_required",
    )
    write_csv(
        output / "fit_candidates.csv",
        [
            {
                "dataset": ds,
                "status": fit.get("status", "complete"),
                "rejection": fit.get("rejection"),
                **{k: fit[k] for k in fit_columns},
                "selected_development_kl": fit["selection"]["teacher_kl"],
            }
            for ds, value in sorted(fits.items())
            for fit in value["fits"]
        ],
    )
    write_csv(
        output / "fit_selection.csv",
        [
            {
                "dataset": ds,
                "source": source,
                **{k: fit[k] for k in fit_columns},
                "selected_development_kl": fit["selection"]["teacher_kl"],
            }
            for ds, value in sorted(fits.items())
            for source, fit in value["sources"].items()
            if fit is not None
        ],
    )
    atomic_write_json(
        output / "results.json",
        dict(
            sources=sources,
            analysis_code={
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            summary=summary,
            paired_comparisons=comparisons,
            cross_source_comparisons=across_sources,
            hwa_fits=fits,
            recovery_selection=tuning,
        ),
    )
    lines = [
        "# HWA/CDT versus fixed-array recovery",
        "",
        "Full CIFAR ResNet-32 with final four convolutions and classifier analog. Gaussian programming endpoints, immediate inference; no pulse loop, extra read noise or drift.",
        "HWA: 45000 training images, fresh crop/flip, development-only selection. Recovery: one pass over 5000 training images. Test: all 10000 images, three fresh arrays, one HWA initialization. Lower teacher KL is better.",
        "",
        "## Clean source fidelity before programming",
        "",
        "These measurements use the adapted model with exact digital weights. They expose any accuracy or teacher-fidelity cost of generic robustness training.",
        "",
        "| Dataset | Source | Teacher KL | Accuracy (%) |",
        "|---|---|---:|---:|",
    ]
    for row in summary:
        if row["fault_kind"] == "none" and row["method"] == "none":
            lines.append(
                f"| {row['dataset']} | {row['source']} | {row['clean_teacher_kl']:.5g} | {row['clean_accuracy_percent']:.2f} |"
            )
    lines += [
        "",
        "## Deployment and recovery KL",
        "",
        "| Dataset | Source | Fault | ppm | Deployed KL | Calibration KL | Rewrite KL | Weight recovery KL | Hybrid KL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_case = defaultdict(dict)
    for row in summary:
        by_case[
            tuple(row[k] for k in ("dataset", "source", "fault_kind", "fault_rate_ppm"))
        ][row["method"]] = row
    for key, methods in sorted(by_case.items()):
        values = " | ".join(
            f"{methods[m]['teacher_kl_mean']:.5g} ± {methods[m]['teacher_kl_std']:.2g}"
            for m in METHODS
        )
        lines.append("| " + " | ".join(map(str, key)) + " | " + values + " |")
    lines += [
        "",
        "## Deployment and recovery accuracy",
        "",
        "| Dataset | Source | Fault | ppm | Deployed (%) | Calibration (%) | Rewrite (%) | Weight recovery (%) | Hybrid (%) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, methods in sorted(by_case.items()):
        values = " | ".join(
            f"{methods[m]['accuracy_mean']:.2f} ± {methods[m]['accuracy_std']:.2f}"
            for m in METHODS
        )
        lines.append("| " + " | ".join(map(str, key)) + " | " + values + " |")
    lines += ["", "## HWA selection and convergence", ""]
    for ds, fit in fits.items():
        lines.append(
            f"{ds}: {fit['candidates']} attempted candidates; {fit.get('rejected_candidates', 0)} rejected as nonfinite and excluded from source selection."
        )
        for attempt in fit["fits"]:
            if attempt.get("rejection"):
                lines.append(f"- Rejected {attempt['id']}: {attempt['rejection']}.")
        for name, source in fit["sources"].items():
            if source is not None:
                lines.append(
                    f"- {name}: {source['id']}, selected epoch {source['selected_epoch']}/{source['epochs_run']}; convergence review required: {source['convergence_review_required']}."
                )
        lines.append(f"Recovery rates: {tuning[ds]['selected']}.")
    lines += [
        "",
        "CDT follows the paper's resampled-fault principle with independent physical-bank failures, matched to our deployment convention. It does not reproduce the paper's full-network/peripheral/retention benchmark.",
        "Paired reductions are control KL minus recovered KL. Arrays improved are descriptive counts, not significance tests. All negative outcomes are retained. See summary.csv for accuracy and per_array.csv for individual arrays.",
        "cross_source_comparisons.csv pairs each weight/hybrid recovery arm with every applicable standard-HWA, noise-HWA and CDT non-weight control. It retains every contrast rather than selecting a model using final test results. Accuracy gains are in percentage points.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    plot(summary, output)
    plot_fits(fits, output)
    return summary


def plot_fits(fits, output):
    """Expose every candidate, including epoch-zero and late selections."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = ("standard", "noise", "cdt_open", "cdt_random", "cdt_gmax")
    criteria = (
        "Nominal arrays",
        "All fault types and rates",
        "Open: all rates",
        "Random: all rates",
        "Gmax: all rates",
    )
    for dataset, data in sorted(fits.items()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, family, criterion in zip(axes.flat, families, criteria):
            for fit in data["fits"]:
                if not fit["id"].startswith(family + "_"):
                    continue
                history = fit["history"]
                x = [h["epoch"] for h in history]
                y = [h["teacher_kl"] for h in history]
                label = (
                    fit["id"]
                    .removeprefix(family + "_")
                    .replace("teacher_kl", "KL")
                    .replace("cross_entropy", "CE")
                )
                if fit.get("rejection"):
                    label += " (nonfinite; rejected)"
                (line,) = ax.plot(x, y, "o-", ms=2, lw=1, label=label)
                ax.plot(
                    fit["selected_epoch"],
                    fit["selection"]["teacher_kl"],
                    "*",
                    color=line.get_color(),
                    ms=10,
                )
            ax.set_yscale("log")
            ax.set(
                title=f"{family}: {criterion}",
                xlabel="Full training epochs",
                ylabel="Development teacher KL (nats)",
            )
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=6)
        ax = axes.flat[-1]
        ax.axis("off")
        notes = [
            "Stars mark selected checkpoints.",
            "Different panels use different case averages.",
            f"Selection images: {data['validation_images']:,} (development only).",
            "Final arrays are excluded from selection.",
            "",
            "Selected sources:",
        ]
        for name, fit in data["sources"].items():
            if fit is not None:
                flag = (
                    " [late; unresolved]" if fit["convergence_review_required"] else ""
                )
                notes += [
                    f"{name}: epoch {fit['selected_epoch']}/{fit['epochs_run']}{flag}",
                    f"  {fit['id']}",
                ]
        ax.text(0, 1, "\n".join(notes), va="top", fontsize=8)
        label = "CANARY" if data["smoke_only"] else "one training seed"
        fig.suptitle(f"{dataset}: HWA/CDT development histories ({label})")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for suffix in ("png", "pdf"):
            fig.savefig(
                output / f"{dataset}_hwa_development.{suffix}",
                dpi=180,
                bbox_inches="tight",
            )
        plt.close(fig)


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dataset in sorted({r["dataset"] for r in rows}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for ax, kind in zip(axes, ("open", "random", "gmax")):
            for source, method, label, color, style in (
                ("digital", "none", "Digital → array", "0.4", "--"),
                ("standard_hwa", "none", "Standard HWA", "#1f77b4", "-"),
                ("noise_hwa", "none", "Noise-selected HWA", "#17becf", "-"),
                (
                    "standard_hwa",
                    "calibration",
                    "Standard HWA + calibration",
                    "#1f77b4",
                    ":",
                ),
                ("noise_hwa", "calibration", "Noise HWA + calibration", "#17becf", ":"),
                (f"cdt_{kind}", "none", "HWA + CDT", "black", "-"),
                (f"cdt_{kind}", "calibration", "CDT + calibration", "#2ca02c", "-"),
                (
                    f"cdt_{kind}",
                    "rewrite",
                    "CDT + rewrite/calibration",
                    "#ff7f0e",
                    "--",
                ),
                (
                    f"cdt_{kind}",
                    "onchip_weights",
                    "CDT + weight recovery",
                    "#d62728",
                    "-",
                ),
                (
                    f"cdt_{kind}",
                    "onchip_calibration",
                    "CDT + hybrid recovery",
                    "#9467bd",
                    "-",
                ),
            ):
                selected = sorted(
                    (
                        r
                        for r in rows
                        if r["dataset"] == dataset
                        and r["source"] == source
                        and r["method"] == method
                        and r["fault_kind"] in ("none", kind)
                    ),
                    key=lambda r: r["fault_rate_ppm"],
                )
                x = [r["fault_rate_ppm"] for r in selected]
                y = np.array([r["teacher_kl_mean"] for r in selected])
                sd = np.array([r["teacher_kl_std"] for r in selected])
                ax.plot(x, y, "o" + style, label=label, color=color, ms=3)
                ax.fill_between(
                    x, np.maximum(y - sd, 1e-8), y + sd, color=color, alpha=0.07
                )
            ax.set_xscale("symlog", linthresh=100)
            ax.set_yscale("log")
            ax.set_xticks([0, 100, 1000, 10000], ["0", "100", "1,000", "10,000"])
            ax.set(title=kind, xlabel="Physical-device failure probability (ppm)")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Mean teacher KL (nats; lower is better)")
        axes[-1].legend(fontsize=7)
        fig.suptitle(
            f"{dataset}: HWA/CDT and one-pass recovery; mean ± SD over 3 fresh arrays"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for suffix in ("png", "pdf"):
            fig.savefig(
                output / f"{dataset}_hwa_cdt_kl.{suffix}", dpi=180, bbox_inches="tight"
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--disable-cudnn", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--execution", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(
            args.output,
            args.dataset,
            device=args.device,
            disable_cudnn=args.disable_cudnn,
            smoke=args.smoke,
        )
    else:
        report(args.execution, args.output)


if __name__ == "__main__":
    main()
