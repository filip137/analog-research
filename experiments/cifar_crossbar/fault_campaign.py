"""Prepare native PCM fault studies and summarize their measured KL curves.

Execution uses the existing subprocess-only cifar_crossbar.campaign runner.
One campaign per dataset permits independent launch receipts and CPU workers.
"""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.fault_config import (
    EXPERIMENT_ID,
    METHODS,
    default_config,
)


def prepare(
    root,
    dataset,
    *,
    device="cpu",
    threads=4,
    recovery_model="gaussian_endpoint_reprogramming",
):
    root = Path(root).resolve()
    if (root / "campaign.json").exists():
        raise ValueError(
            "Expected a new directory; frozen fault studies cannot be overwritten."
        )
    common = dict(
        dataset=dataset,
        device=device,
        cpu_threads=threads,
        disable_cudnn=device.startswith("cuda"),
        recovery_model=recovery_model,
    )
    nodes = []
    study_id = f"{dataset}-pcm-faults-one-epoch-v1"
    for index in range(4):
        cache = index == 0
        name = f"{dataset}_cache" if cache else f"{dataset}_array{index}"
        spec = default_config(
            **common,
            stage="cache" if cache else "screen",
            array_seed=51000 + max(index, 1),
            endpoint_seed=61000 + max(index, 1),
        )
        config = root / "configs" / f"{name}.json"
        atomic_write_json(config, spec)
        nodes.append(
            {
                "id": name,
                "dataset": dataset,
                "config": str(config.relative_to(root)),
                "sha256": sha256_file(config),
                "depends": [] if cache else [f"{dataset}_cache"],
                "inputs": (
                    {}
                    if cache
                    else {
                        "device-data": {
                            "node": f"{dataset}_cache",
                            "artifact": "checkpoints/cache.pt",
                        }
                    }
                ),
            }
        )
    plan = {
        "schema_version": 1,
        "study_id": study_id,
        "title": f"{dataset}: permanent PCM failures and one-epoch teacher-KL recovery",
        "evidence_class": "exploratory",
        "hypothesis": "One epoch adapting targets on a fixed failed array can lower teacher KL beyond calibration and equally frequent frozen-target rewrites.",
        "motivation": "Study open, Gmax and random permanent failures inspired by Li et al. (2023), doi:10.1063/5.0131797, using Gaussian programming endpoints.",
        "arms": [
            {
                "arm_id": n["id"],
                "configs": [n["config"]],
                "description": n["id"],
                "experiment_id": EXPERIMENT_ID,
                "mode": "train",
            }
            for n in nodes
        ],
        "completion_criteria": [
            "One full digital/cache audit and three completed independent array screens.",
            "Each array covers nominal plus all three failures at 100, 1000, 10000 ppm; five matched controls per case.",
            "Exactly one epoch on 5000 balanced training images; evaluate all 10000 official test images without selecting checkpoints.",
            "Permanent physical fault states and digital prefix remain unchanged; saved P0 restored for each control.",
        ],
        "analysis_plan": [
            "Primary metric: mean KL(p_teacher || p_student), natural logs, temperature 1; accuracy is secondary.",
            "Report raw deployment, calibration, equally frequent frozen-target rewriting, weight-only recovery and recovery plus calibration.",
            "Average three arrays and retain per-array outcomes; use paired reductions and mean/standard-deviation curves, without confirmatory p-values.",
            "Use original pretrained digital checkpoints, no HWA adaptation; this is not evidence of superiority to converged HWA.",
            "Immediate programming endpoints only; no drift, extra read noise, pulse/P&V loop, ADC/DAC or energy claim.",
        ],
    }
    atomic_write_json(root / "study-plan.json", plan)
    campaign = {
        "schema": "cifar_crossbar.campaign.v1",
        "study_id": study_id,
        "phase": "fault_screen",
        "plan": "study-plan.json",
        "plan_sha256": sha256_file(root / "study-plan.json"),
        "selection_inputs": {},
        "nodes": nodes,
    }
    atomic_write_json(root / "campaign.json", campaign)
    return campaign


def report(executions, output):
    from experiments.cifar_crossbar.analysis import records

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    flat, sources = [], []
    for execution in executions:
        campaign, runs = records(execution)
        if campaign["phase"] != "fault_screen":
            raise ValueError("Expected the declared PCM failure screen.")
        sources.append(
            {"path": str(Path(execution).resolve()), "sha256": sha256_file(execution)}
        )
        for run in runs:
            spec, result = run["config"], run["metrics"]
            if spec["stage"] != "screen":
                continue
            if result["smoke_only"] or result["evaluation_images"] != 10000:
                raise ValueError("Expected full test metrics, not the canary.")
            if result["recovery_images"] != 5000 or spec["epochs"] != 1:
                raise ValueError(
                    "Expected one pass over the 5000-image recovery cohort."
                )
            if result["completed_controls"] != 50 or result["cases"] != 10:
                raise ValueError("Expected all ten physical cases and five controls.")
            for row in result["measurements"]:
                expected_images = 0 if row["method"] == "none" else 5000
                expected_writes = (
                    79
                    if row["method"]
                    in ("rewrite", "onchip_weights", "onchip_calibration")
                    else 0
                )
                if (
                    row["image_presentations"] != expected_images
                    or row["cost"]["array_reprogram_calls"] != expected_writes
                ):
                    raise ValueError(
                        "Measured recovery budget differs from the declared screen."
                    )
                flat.append(
                    {
                        "dataset": spec["dataset"],
                        "fault_kind": row["fault_kind"],
                        "fault_rate_ppm": row["fault_rate_ppm"],
                        "array_seed": row["array_seed"],
                        "method": row["method"],
                        "initial_teacher_kl": row["initial"]["teacher_kl"],
                        "teacher_kl": row["final"]["teacher_kl"],
                        "kl_reduction": row["teacher_kl_reduction"],
                        "accuracy_percent": row["final"]["accuracy_percent"],
                        "failed_devices": row["faults"]["failed_devices"],
                        "affected_weights": row["faults"]["affected_weights"],
                        "realized_rate": row["faults"]["realized_rate"],
                        "reprogram_calls": row["cost"]["array_reprogram_calls"],
                        "p0_sha256": row["p0_sha256"],
                        "initial_apparent_sha256": row["initial_apparent_sha256"],
                        "run": str(run["run"]),
                    }
                )
    pairs = defaultdict(dict)
    for row in flat:
        key = tuple(
            row[k] for k in ("dataset", "fault_kind", "fault_rate_ppm", "array_seed")
        )
        if row["method"] in pairs[key]:
            raise ValueError("Duplicate paired control.")
        pairs[key][row["method"]] = row
    for methods in pairs.values():
        if set(methods) != set(METHODS):
            raise ValueError("Missing paired control.")
        for key in ("p0_sha256", "initial_apparent_sha256", "initial_teacher_kl"):
            if len({row[key] for row in methods.values()}) != 1:
                raise ValueError(f"Mismatched deployment: {key}")
    comparisons = paired_recovery_summary(pairs)
    grouped = defaultdict(list)
    for row in flat:
        grouped[
            tuple(row[k] for k in ("dataset", "fault_kind", "fault_rate_ppm", "method"))
        ].append(row)
    summary = []
    for (dataset, kind, ppm, method), rows in sorted(grouped.items()):
        if len(rows) != 3 or len({r["array_seed"] for r in rows}) != 3:
            raise ValueError(
                "Expected exactly three independent arrays per case/control."
            )
        summary.append(
            {
                "dataset": dataset,
                "fault_kind": kind,
                "fault_rate_ppm": ppm,
                "method": method,
                "arrays": 3,
                "teacher_kl_mean": float(np.mean([r["teacher_kl"] for r in rows])),
                "teacher_kl_std": float(
                    np.std([r["teacher_kl"] for r in rows], ddof=1)
                ),
                "accuracy_mean": float(np.mean([r["accuracy_percent"] for r in rows])),
                "accuracy_std": float(
                    np.std([r["accuracy_percent"] for r in rows], ddof=1)
                ),
                "kl_reduction_mean": float(np.mean([r["kl_reduction"] for r in rows])),
                "kl_reduction_std": float(
                    np.std([r["kl_reduction"] for r in rows], ddof=1)
                ),
            }
        )
    for name, rows in (
        ("per_array.csv", flat),
        ("summary.csv", summary),
        ("paired_comparisons.csv", comparisons),
    ):
        with (output / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_write_json(
        output / "results.json",
        {
            "sources": sources,
            "summary": summary,
            "paired_comparisons": comparisons,
            "kl_definition": "mean KL(p_teacher || p_student), nats at T=1",
            "evidence": "exploratory; three arrays; pretrained digital source; no HWA comparison",
        },
    )
    lines = [
        "# PCM permanent failures: one-epoch recovery",
        "",
        "Full CIFAR ResNet-32; final four convolutions and classifier on crossbars. Gaussian programming endpoints, fixed physical faults; no pulse loop.",
        "One pass over 5000 balanced training images, not an epoch over all 50000 CIFAR training images. All metrics use 10000 official test images; mean ± sample SD of three arrays. Lower teacher KL is better.",
        "Source: pretrained digital network without HWA adaptation. This is an exploratory fault-recovery screen, not a comparison against converged HWA.",
        "",
        "| Dataset | Failure | ppm | Deployed KL | Calibration KL | Rewrite KL | Weight-only recovery KL | Recovery + calibration KL |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_case = defaultdict(dict)
    for row in summary:
        by_case[(row["dataset"], row["fault_kind"], row["fault_rate_ppm"])][
            row["method"]
        ] = row
    for (dataset, kind, ppm), methods in sorted(by_case.items()):
        values = " | ".join(
            f"{methods[m]['teacher_kl_mean']:.5g} ± {methods[m]['teacher_kl_std']:.2g}"
            for m in METHODS
        )
        lines.append(f"| {dataset} | {kind} | {ppm} | {values} |")
    lines += [
        "",
        "## Test accuracy (%)",
        "",
        "| Dataset | Failure | ppm | Deployed | Calibration | Rewrite | Weight-only recovery | Recovery + calibration |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (dataset, kind, ppm), methods in sorted(by_case.items()):
        values = " | ".join(
            f"{methods[m]['accuracy_mean']:.2f} ± {methods[m]['accuracy_std']:.2f}"
            for m in METHODS
        )
        lines.append(f"| {dataset} | {kind} | {ppm} | {values} |")
    lines += [
        "",
        "The rewrite control reprograms frozen original targets once per minibatch and calibrates digital parameters. Weight-only recovery freezes all digital parameters. Hybrid recovery uses the same digital calibration parameters as the controls.",
        "Programming cost is counted as endpoint reprogramming requests, not physical pulses. Failed devices remain stuck during every update. The controller has no access to the fault map.",
        "See per_array.csv for all outcomes and summary.csv for standard deviations and accuracy. No checkpoint or hyperparameter was selected on test results.",
        "paired_comparisons.csv reports control KL minus recovery KL, paired by physical array. Positive reductions favor recovery; the array improvement counts are descriptive, not significance tests.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    plot(summary, output)
    return summary


def paired_recovery_summary(pairs):
    grouped = defaultdict(list)
    for (dataset, kind, ppm, _), methods in pairs.items():
        for candidate in ("onchip_weights", "onchip_calibration"):
            for control in ("none", "calibration", "rewrite"):
                grouped[(dataset, kind, ppm, candidate, control)].append(
                    methods[control]["teacher_kl"] - methods[candidate]["teacher_kl"]
                )
    output = []
    for (dataset, kind, ppm, candidate, control), values in sorted(grouped.items()):
        if len(values) != 3:
            raise ValueError("Expected three paired arrays for every KL comparison.")
        output.append(
            {
                "dataset": dataset,
                "fault_kind": kind,
                "fault_rate_ppm": ppm,
                "candidate": candidate,
                "control": control,
                "arrays": len(values),
                "kl_reduction_mean": float(np.mean(values)),
                "kl_reduction_std": float(np.std(values, ddof=1)),
                "arrays_improved": sum(value > 0 for value in values),
            }
        )
    return output


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("black", "#1f77b4", "#2ca02c", "#d62728", "#9467bd")
    labels = (
        "Deployed",
        "Calibration",
        "Frozen-target rewrite",
        "On-chip weights",
        "On-chip + calibration",
    )
    for dataset in sorted({r["dataset"] for r in rows}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, kind in zip(axes, ("open", "gmax", "random")):
            for method, color, label in zip(METHODS, colors, labels):
                selected = sorted(
                    (
                        r
                        for r in rows
                        if r["dataset"] == dataset
                        and r["method"] == method
                        and r["fault_kind"] in ("none", kind)
                    ),
                    key=lambda r: r["fault_rate_ppm"],
                )
                x = np.array([r["fault_rate_ppm"] for r in selected])
                y = np.array([r["teacher_kl_mean"] for r in selected])
                sd = np.array([r["teacher_kl_std"] for r in selected])
                ax.plot(x, y, "o-", color=color, label=label, ms=4)
                ax.fill_between(
                    x, np.maximum(y - sd, 1e-8), y + sd, color=color, alpha=0.1
                )
            ax.set_xscale("symlog", linthresh=100)
            ax.set_yscale("log")
            ax.set_xticks([0, 100, 1000, 10000], ["0", "100", "1,000", "10,000"])
            ax.set(
                title={
                    "open": "Open: fixed at 0 µS",
                    "gmax": "Stuck-high: fixed at 25 µS",
                    "random": "Random fixed conductance",
                }[kind],
                xlabel="Requested failure probability (ppm/device)",
            )
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Mean teacher KL (nats; lower is better)")
        axes[-1].legend(fontsize=8)
        fig.suptitle(
            f"{dataset.upper().replace('CIFAR', 'CIFAR-')}: Gaussian PCM, one pass / 5000 images; test mean ± SD over 3 arrays"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(
            output / f"{dataset}_fault_kl.png",
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.15,
        )
        fig.savefig(
            output / f"{dataset}_fault_kl.pdf", bbox_inches="tight", pad_inches=0.15
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    make.add_argument("--device", default="cpu")
    make.add_argument("--threads", type=int, default=4)
    make.add_argument(
        "--recovery-model",
        choices=("gaussian_endpoint_reprogramming", "ideal_update_upper_bound"),
        default="gaussian_endpoint_reprogramming",
    )
    analyze = sub.add_parser("report")
    analyze.add_argument("--execution", type=Path, nargs="+", required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(
            args.output,
            args.dataset,
            device=args.device,
            threads=args.threads,
            recovery_model=args.recovery_model,
        )
    else:
        report(args.execution, args.output)


if __name__ == "__main__":
    main()
