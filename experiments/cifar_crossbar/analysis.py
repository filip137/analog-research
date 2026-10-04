"""Development-only selection and paired, array-level confirmation analysis."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.campaign import cases, external, read_receipt


def relocated(path):
    """Read a collected snapshot without rewriting its original manifests."""
    path = Path(path)
    mappings = json.loads(os.environ.get("EBL_CIFAR_MIRROR", "{}"))
    for original, local in mappings.items():
        if path.is_relative_to(original):
            return Path(local) / path.relative_to(original)
    return path


def records(path, *, complete=True):
    execution = read_receipt(path, "cifar_crossbar.execution.v1")
    campaign_path = relocated(execution["campaign"])
    campaign = read_receipt(campaign_path, "cifar_crossbar.campaign.v1")
    if sha256_file(campaign_path) != execution["campaign_sha256"]:
        raise ValueError("Campaign changed after execution.")
    if complete and (
        execution["active"] is not None
        or set(execution["completed"]) != {node["id"] for node in campaign["nodes"]}
    ):
        raise ValueError("Expected exact terminal coverage of the declared campaign.")
    result = []
    for node in campaign["nodes"]:
        if node["id"] not in execution["completed"]:
            continue
        entry = execution["completed"][node["id"]]
        run = relocated(entry["run_dir"])
        if sha256_file(run / "result.json") != entry["result_sha256"]:
            raise ValueError(f"Result digest mismatch: {run}")
        config = json.loads((run / "config.resolved.json").read_text())
        source_config = campaign_path.parent / node["config"]
        if sha256_file(source_config) != node["sha256"] or config != json.loads(
            source_config.read_text()
        ):
            raise ValueError(f"Run settings differ from the frozen campaign: {run}")
        value = json.loads((run / "result.json").read_text())
        if value["status"] != "complete":
            raise ValueError(f"Incomplete native result: {run}")
        result.append(
            {
                "name": node["id"],
                "run": run,
                "config": config,
                "metrics": value["metrics"],
            }
        )
    return campaign, result


def group_key(config):
    return f"{config['dataset']}_{config['backend']}"


def backend_label(backend):
    return {
        "om": "OM ReRAM pulse model",
        "pcm": "PCM pulse/refresh model (legacy pilot)",
        "pcm_inference": "PCM Gaussian programming-endpoint model",
    }[backend]


def teacher_metric_rows(rows):
    """Expose already measured teacher KL, including HWA epoch-zero candidates."""
    output = []
    for row in rows:
        c, m = row["config"], row["metrics"]
        stage = c["stage"]
        if stage == "audit":
            measurements = [("digital_audit", m["metrics"], None)]
        elif stage == "hwa":
            measurements = [
                (
                    "population_validation",
                    {**entry, "accuracy_percent": entry["validation_accuracy_percent"]},
                    entry["epoch"],
                )
                for entry in m["history"]
            ]
        else:
            measurements = [
                (name, m[name], None)
                for name in {
                    "deploy": ("clean", "apparent"),
                    "recover": ("initial", "final"),
                    "pcm_reference": (
                        "clean",
                        "programmed",
                        "retained",
                        "normalization_calibrated",
                    ),
                }.get(stage, ())
                if name in m
            ]
        for name, metrics, epoch in measurements:
            output.append(
                {
                    "dataset": c["dataset"],
                    "backend": c["backend"],
                    "model": (
                        "digital" if stage == "audit" else backend_label(c["backend"])
                    ),
                    "stage": stage,
                    "measurement": name,
                    "method": c["method"] if stage == "recover" else "",
                    "epoch": epoch,
                    "selected_hwa_epoch": (
                        epoch == m["selected"]["epoch"] if stage == "hwa" else ""
                    ),
                    "evaluation": c["evaluation"],
                    "accuracy_percent": metrics["accuracy_percent"],
                    "teacher_kl_nats": metrics["teacher_kl"],
                    "examples": metrics.get(
                        "examples", c["max_examples"] or c["validation_images"]
                    ),
                    "run": str(row["run"]),
                }
            )
    return output


def scenario(config):
    for name, changes in cases(config["backend"]).items():
        expected = {
            "policy": "repaired",
            "noise_scale": 1.0,
            "variation_scale": 1.0,
            "fault_rate": 0.0,
            "fault_kind": "low",
            "read_noise": 0.0,
            **changes,
        }
        if all(config[k] == v for k, v in expected.items()):
            return name
    raise ValueError(
        "Unknown physical scenario; extend the declared case matrix before analysis."
    )


def development(path, phase):
    campaign, rows = records(path)
    if campaign["phase"] != phase:
        raise ValueError(f"Selection requires the {phase} development campaign.")
    selected = [r for r in rows if r["config"]["stage"] != "audit"]
    if any(
        r["config"]["evaluation"] != "development" or r["metrics"]["smoke_only"]
        for r in selected
    ):
        raise ValueError("Smoke/test results cannot select scientific settings.")
    suffixes = {r["config"]["suffix"] for r in rows}
    if len(suffixes) != 1:
        raise ValueError("Expected one common suffix.")
    return rows, suffixes.pop()


def select_hwa(path):
    rows, suffix = development(path, "hwa")
    grouped = defaultdict(list)
    for row in rows:
        if row["config"]["stage"] == "hwa":
            grouped[group_key(row["config"])].append(row)
    selections = {}
    for group, candidates in grouped.items():
        best = max(
            candidates,
            key=lambda r: (
                r["metrics"]["selected"]["validation_accuracy_percent"],
                -r["metrics"]["selected"]["teacher_kl"],
            ),
        )
        if best["metrics"]["convergence_review_required"]:
            raise ValueError(
                f"{group}: best HWA is still improving near the epoch limit; "
                "extend the HWA development budget in a new frozen campaign."
            )
        config = best["config"]
        selections[group] = {
            k: config[k] for k in ("learning_rate", "objective", "epochs")
        }
        selections[group].update(
            weights=external(best["run"] / "checkpoints/weights.pt"),
            selected=best["metrics"]["selected"],
            candidates=len(candidates),
        )
    return {
        "schema": "cifar_crossbar.hwa_selection.v1",
        "suffix": suffix,
        "source": external(path),
        "evaluation": "development",
        "groups": selections,
    }


def select_recovery(path):
    rows, suffix = development(path, "tune")
    candidates = defaultdict(lambda: defaultdict(list))
    for row in rows:
        c, m = row["config"], row["metrics"]
        if c["stage"] != "recover" or c["method"] == "none":
            continue
        key = (group_key(c), f"{c['method']}_{c['recovery_images']}")
        for entry in m["history"]:
            if entry["epoch"] in (5, 15, 30):
                params = (c["learning_rate"], c["calibration_lr"], entry["epoch"])
                candidates[key][params].append(
                    (
                        entry["metrics"]["accuracy_percent"],
                        entry["metrics"]["teacher_kl"],
                    )
                )
    selections = defaultdict(dict)
    for (group, method), values in candidates.items():
        counts = {len(v) for v in values.values()}
        if len(counts) != 1 or next(iter(counts)) < 4:
            raise ValueError(
                "Recovery settings require all two cases by two development arrays."
            )
        best = max(
            values,
            key=lambda p: (
                np.mean([v[0] for v in values[p]]),
                -np.mean([v[1] for v in values[p]]),
                -p[2],
                -p[0],
            ),
        )
        selections[group][method] = dict(
            zip(("learning_rate", "calibration_lr", "epochs"), best)
        )
    return {
        "schema": "cifar_crossbar.recovery_selection.v1",
        "suffix": suffix,
        "source": external(path),
        "evaluation": "development",
        "groups": dict(selections),
    }


def recovery_rows(rows):
    output = []
    for row in rows:
        c, m = row["config"], row["metrics"]
        if c["stage"] != "recover":
            continue
        output.append(
            {
                "group": group_key(c),
                "dataset": c["dataset"],
                "backend": c["backend"],
                "model": backend_label(c["backend"]),
                "case": scenario(c),
                "hwa_seed": c["seed"],
                "array": c["assignment_seed"],
                "endpoint": c["endpoint_seed"],
                "budget": c["recovery_images"],
                "method": c["method"],
                "accuracy_percent": m["final"]["accuracy_percent"],
                "teacher_kl": m["final"]["teacher_kl"],
                "initial_teacher_kl": m["initial"]["teacher_kl"],
                "teacher_kl_reduction_over_raw": (
                    m["initial"]["teacher_kl"] - m["final"]["teacher_kl"]
                ),
                "initial_accuracy_percent": m["initial"]["accuracy_percent"],
                "gain_over_raw_pp": m["gain_pp"],
                "physical_pulses": m["cost"]["physical_pulses"],
                "verify_reads": m["controller_verify_reads"]
                + (m["rewrite"] or {}).get("verify_reads", 0)
                + m["cost"]["refresh_reads"],
                "presentations": m["image_presentations"],
                "p0_sha256": m["p0_sha256"],
                "initial_apparent_sha256": m["initial_apparent_sha256"],
                "run": str(row["run"]),
            }
        )
    return output


def paired_rows(rows):
    grouped = defaultdict(dict)
    for row in recovery_rows(rows):
        key = tuple(
            row[k] for k in ("group", "case", "hwa_seed", "array", "endpoint", "budget")
        )
        if row["method"] in grouped[key]:
            raise ValueError(
                "Repeated method on one target; analyze tuning through select-recovery."
            )
        grouped[key][row["method"]] = row
    for methods in grouped.values():
        if set(methods) != {
            "none",
            "calibration",
            "rewrite",
            "open_loop",
            "closed_loop",
        }:
            raise ValueError("A matched recovery comparison is missing a control.")
        for key in (
            "p0_sha256",
            "initial_apparent_sha256",
            "initial_accuracy_percent",
            "initial_teacher_kl",
        ):
            if len({v[key] for v in methods.values()}) != 1:
                raise ValueError(
                    f"Paired arms did not share the same deployment: {key}"
                )
    return grouped


def select_screen(path):
    rows, suffix = development(path, "screen")
    pairs = paired_rows(rows)
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key, methods in pairs.items():
        group, case, _, _, _, budget = key
        if budget != 5000:
            continue
        for method, value in methods.items():
            values[group][case][method].append(value["accuracy_percent"])
    selections = {}
    for group, case_values in values.items():
        gains = {}
        for case, methods in case_values.items():
            means = {method: float(np.mean(v)) for method, v in methods.items()}
            gains[case] = max(means[m] for m in ("open_loop", "closed_loop")) - max(
                means[m] for m in ("calibration", "rewrite")
            )
        nonnominal = {
            k: v
            for k, v in gains.items()
            if k != "nominal" and not k.startswith("read")
        }
        best = max(nonnominal, key=nonnominal.get)
        selections[group] = {
            "cases": ["nominal", best, "read50"],
            "best_nonnominal_development_gain_pp": float(nonnominal[best]),
            "all_mean_gains_pp": gains,
        }
    return {
        "schema": "cifar_crossbar.screen_selection.v1",
        "suffix": suffix,
        "source": external(path),
        "evaluation": "development",
        "groups": selections,
        "rule": "nominal, maximum mean non-read-noise gain at 5000 images, fixed read-noise 0.05 control",
    }


def runtime_gate(path):
    import math

    campaign, rows = records(path)
    if campaign["phase"] != "pilot":
        raise ValueError(
            "Runtime sizing requires the predeclared implementation pilot."
        )
    groups = defaultdict(list)
    suffixes = {r["config"]["suffix"] for r in rows}
    if len(suffixes) != 1:
        raise ValueError("Expected one tested suffix.")
    for row in rows:
        c, m = row["config"], row["metrics"]
        if c["stage"] != "recover" or not m["history"]:
            continue
        batches = math.ceil(m["recovery_cohort"]["count"] / c["batch_size"])
        epoch_seconds = max(e["epoch_seconds"] for e in m["history"])
        forecast = (
            2 * epoch_seconds / batches * math.ceil(5000 / c["batch_size"]) * 30
            + m["elapsed_seconds"]
        )
        groups[group_key(c)].append(
            {
                "method": c["method"],
                "forecast_seconds": forecast,
                "peak_gpu_memory_bytes": m["peak_gpu_memory_bytes"],
            }
        )
    result = {}
    for group, values in groups.items():
        forecast = max(v["forecast_seconds"] for v in values)
        result[group] = {
            "passed": forecast <= 7200,
            "conservative_5000_image_seconds": forecast,
            "peak_gpu_memory_bytes": max(
                v["peak_gpu_memory_bytes"] or 0 for v in values
            ),
            "methods": values,
        }
    return {
        "schema": "cifar_crossbar.runtime_selection.v1",
        "source": external(path),
        "suffix": suffixes.pop(),
        "groups": result,
        "limit_seconds": 7200,
        "forecast": "double pilot epoch time per batch, 79 batches/epoch, 30 epochs, plus startup",
        "memory": "all declared arms completed; recheck memory when changing GPU type",
    }


def paired_statistics(values, strata, *, seed=927, draws=10000):
    """Endpoints are already averaged; resample arrays within fixed HWA strata."""
    values, strata = np.asarray(values, dtype=float), np.asarray(strata)
    rng = np.random.default_rng(seed)
    boot = np.zeros(draws)
    for label in np.unique(strata):
        x = values[strata == label]
        boot += rng.choice(x, size=(draws, len(x)), replace=True).sum(1) / len(values)
    mean = float(values.mean())
    if len(values) <= 18:
        numbers = np.arange(2 ** len(values), dtype=np.uint64)
        signs = (
            2
            * (
                (numbers[:, None] >> np.arange(len(values), dtype=np.uint64)) & 1
            ).astype(float)
            - 1
        )
        p = float(np.mean((signs @ values) / len(values) >= mean - 1e-12))
    else:
        signs = 2 * rng.integers(0, 2, size=(100000, len(values))) - 1
        p = float(
            (1 + np.sum((signs @ values) / len(values) >= mean - 1e-12))
            / (len(signs) + 1)
        )
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "gain_pp": mean,
        "ci95_pp": [float(lo), float(hi)],
        "p_one_sided": p,
        "independent_arrays": len(values),
        "hwa_seed_strata": len(np.unique(strata)),
    }


def holm(pvalues):
    order = np.argsort(pvalues)
    result = np.zeros(len(order))
    previous = 0.0
    for i, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - i) * pvalues[index]))
        result[index] = previous
    return result.tolist()


def report(path, output):
    campaign, rows = records(path)
    if campaign["phase"] not in ("pilot", "screen", "confirm"):
        raise ValueError("This report compares matched recovery phases only.")
    pairs = paired_rows(rows)
    effects = defaultdict(lambda: defaultdict(list))
    kl_effects = defaultdict(lambda: defaultdict(list))
    for key, methods in pairs.items():
        group, case, seed, array, _, budget = key
        for method in ("open_loop", "closed_loop"):
            for control in ("calibration", "rewrite"):
                comparison = (group, case, budget, method, control)
                effects[comparison][(seed, array)].append(
                    methods[method]["accuracy_percent"]
                    - methods[control]["accuracy_percent"]
                )
                kl_effects[comparison][(seed, array)].append(
                    methods[control]["teacher_kl"] - methods[method]["teacher_kl"]
                )
    comparisons = []
    for key, arrays in effects.items():
        group, case, budget, method, control = key
        result = {
            "group": group,
            "case": case,
            "budget": budget,
            "method": method,
            "control": control,
            "teacher_kl_reduction_nats": float(
                np.mean([np.mean(v) for v in kl_effects[key].values()])
            ),
        }
        if campaign["phase"] == "confirm":
            strata = [k[0] for k in arrays]
            if (
                len(set(strata)) != 3
                or any(strata.count(s) != 5 for s in set(strata))
                or any(len(v) != 2 for v in arrays.values())
            ):
                raise ValueError(
                    "Confirmation requires 3 HWA seeds x 5 independent arrays x 2 endpoints."
                )
            result.update(
                paired_statistics([np.mean(v) for v in arrays.values()], strata)
            )
        else:
            result.update(
                gain_pp=float(np.mean([np.mean(v) for v in arrays.values()])),
                independent_arrays=len(arrays),
                inferential_claim_supported=False,
            )
        comparisons.append(result)
    if campaign["phase"] == "confirm":
        adjusted = holm([r["p_one_sided"] for r in comparisons])
        for row, p in zip(comparisons, adjusted):
            row["p_holm"] = p
            row["reliable_positive_gain"] = p < 0.05 and row["ci95_pp"][0] > 0
    output.mkdir(parents=True, exist_ok=True)
    flat = recovery_rows(rows)
    with (output / "recovery.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    measured = teacher_metric_rows(rows)
    with (output / "teacher_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)
    payload = {
        "schema": "cifar_crossbar.analysis.v1",
        "phase": campaign["phase"],
        "source": external(path),
        "comparisons": comparisons,
        "teacher_kl": {
            "definition": "mean KL(p_teacher || p_student), natural logs, temperature 1",
            "reduction": "control KL minus recovery KL; positive is better",
            "inference": "descriptive secondary metric; original accuracy selection and tests retained",
        },
        "inference": "array bootstrap within fixed HWA seed strata; "
        "one-sided array sign-flip; Holm across all method/control/case/budget/dataset/device comparisons",
        "required_claim": "positive adjusted evidence against BOTH calibration and frozen-target rewrite",
        "limitations": [
            "Model-based presets; no measured hardware or energy conclusion.",
            "Bootstrap is conditional on the three sampled HWA training seeds.",
            "Pilot/screen outcomes are exploratory, regardless of observed accuracy gain.",
            "PCM pulse/refresh results are not the Gaussian PCM programming-endpoint benchmark.",
        ],
    }
    atomic_write_json(output / "comparisons.json", payload)
    lines = [
        f"# CIFAR suffix recovery — {campaign['phase']}",
        "",
        "All gains below are paired on the exact deployed state. Endpoint replicas are averaged within arrays.",
        "",
        "Teacher KL is mean KL(p_teacher || p_student) in nats at temperature 1. Lower absolute KL is better; positive KL reduction is better.",
        "PCM (`pcm`) means the legacy pulse/refresh model, not the Gaussian programming-endpoint model (`pcm_inference`).",
        "",
        "| Dataset/device | Case | Images | Method | Control | Accuracy gain (pp) | KL reduction (nats) |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['group']} | {row['case']} | {row['budget']} | {row['method']} | {row['control']} | {row['gain_pp']:.3f} | {row['teacher_kl_reduction_nats']:.6g} |"
        )
    lines += [
        "",
        "Absolute paired-control metrics (endpoint means within arrays, then array means):",
        "",
        "| Dataset/device | Case | Images | Method | Accuracy (%) | Teacher KL (nats) |",
        "|---|---|---:|---|---:|---:|",
    ]
    absolute = defaultdict(lambda: defaultdict(list))
    for row in flat:
        key = tuple(row[k] for k in ("group", "case", "budget", "method"))
        absolute[key][(row["hwa_seed"], row["array"])].append(
            (row["accuracy_percent"], row["teacher_kl"])
        )
    for (group, case, budget, method), arrays in sorted(absolute.items()):
        accuracy, kl = np.mean([np.mean(v, axis=0) for v in arrays.values()], axis=0)
        lines.append(
            f"| {group} | {case} | {budget} | {method} | {accuracy:.3f} | {kl:.6g} |"
        )
    lines += [
        "",
        "Only confirmation supports inference. Positive evidence must hold against both controls.",
        "KL reductions are descriptive; historical checkpoint selection and accuracy-based inference are unchanged.",
        "teacher_metrics.csv includes HWA epoch histories and deployment metrics, including unselected candidates.",
        "See comparisons.json for confidence intervals, adjusted tests and limitations.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    plot(comparisons, output)
    plot(comparisons, output, metric="teacher_kl")
    return payload


def plot(rows, output, *, metric="accuracy"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gain_key = "gain_pp" if metric == "accuracy" else "teacher_kl_reduction_nats"
    ylabel = (
        "Paired accuracy gain (percentage points)"
        if metric == "accuracy"
        else "Teacher KL reduction (nats; positive is better)"
    )
    for group in sorted({r["group"] for r in rows}):
        subset = [r for r in rows if r["group"] == group]
        budgets = sorted({r["budget"] for r in subset})
        fig, axes = plt.subplots(
            len(budgets),
            1,
            figsize=(max(8, len(subset) / 8), 4 * len(budgets)),
            squeeze=False,
        )
        for ax, budget in zip(axes[:, 0], budgets):
            selected = [r for r in subset if r["budget"] == budget]
            labels = sorted({(r["case"], r["method"]) for r in selected})
            for control, offset in (("calibration", -0.12), ("rewrite", 0.12)):
                data = [
                    next(
                        r
                        for r in selected
                        if (r["case"], r["method"]) == label and r["control"] == control
                    )
                    for label in labels
                ]
                x = np.arange(len(labels)) + offset
                y = np.array([r[gain_key] for r in data])
                errors = None
                if metric == "accuracy" and all("ci95_pp" in r for r in data):
                    ci = np.array([r["ci95_pp"] for r in data]).T
                    errors = np.maximum(0, np.stack((y - ci[0], ci[1] - y)))
                ax.errorbar(x, y, yerr=errors, fmt="o", capsize=3, label=control)
            ax.axhline(0, color="black", lw=0.8)
            ax.set(
                xticks=np.arange(len(labels)),
                xticklabels=[f"{a}\n{b}" for a, b in labels],
                ylabel=ylabel,
                title=f"{group}{' (legacy pulse/refresh)' if group.endswith('_pcm') else ''}, {budget} recovery images",
            )
            ax.tick_params(axis="x", labelrotation=45)
            ax.legend()
        fig.tight_layout()
        suffix = "" if metric == "accuracy" else "_teacher_kl"
        fig.savefig(output / f"{group}{suffix}.png", dpi=180)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=(
            "select-hwa",
            "select-recovery",
            "select-screen",
            "runtime-gate",
            "report",
        ),
    )
    p.add_argument("--execution", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.action == "report":
        report(args.execution, args.output)
    else:
        fn = {
            "select-hwa": select_hwa,
            "select-recovery": select_recovery,
            "select-screen": select_screen,
            "runtime-gate": runtime_gate,
        }[args.action]
        if args.output.exists():
            raise ValueError(
                "Selection receipts are immutable; choose a new output path."
            )
        atomic_write_json(args.output, fn(args.execution))


if __name__ == "__main__":
    main()
