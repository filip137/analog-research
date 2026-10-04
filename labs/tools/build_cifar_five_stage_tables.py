"""Recovery-stage comparison tables from the fixed-LR CIFAR fault sweep.

Keeps deployed HWA and digitally calibrated HWA as explicit alternative views.
No source, checkpoint, rate, or epoch is selected from the test measurements.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

STAGES = (
    ("clean", "Clean digital"),
    ("direct_deployment", "No HWA — direct deployment"),
    ("direct_calibration", "No HWA + digital calibration"),
    ("normal_hwa", "Normal HWA"),
    ("corruption_hwa", "Corruption-aware HWA"),
    ("normal_hwa_recovery", "Normal HWA + on-chip recovery"),
    ("corruption_hwa_recovery", "Corruption-aware HWA + on-chip recovery"),
)
KINDS = {"gmax": "stuck high", "open": "stuck low", "random": "random stuck"}


def stage_title(stage, title, policy):
    if policy == "calibrated" and stage in ("normal_hwa", "corruption_hwa"):
        return title + " + digital calibration"
    if stage.endswith("_recovery"):
        return title + " + digital calibration"
    return title


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def cell(row, stage):
    acc = row[stage + "_accuracy_percent_mean"]
    kl = row[stage + "_teacher_kl_mean"]
    if abs(kl) < 0.00005:
        kl = 0.0
    return f"{acc:.2f} / {kl:.4f}"


def pivot(rows, dataset, rate=5):
    selected = [r for r in rows if r["dataset"] == dataset and r["fault_kind"] == "gmax" and r["fault_rate_percent"] == rate]
    selected.sort(key=lambda r: (("pcm", "om").index(r["device"]), r["analog_convolutions"]))
    assert len(selected) == 4
    policy = selected[0]["hwa_only_policy"]
    lines = ["| Stage | PCM · 4 convs | PCM · 8 convs | OM · 4 convs | OM · 8 convs |", "|---|---:|---:|---:|---:|"]
    for stage, title in STAGES:
        lines.append("| " + stage_title(stage, title, policy) + " | " + " | ".join(cell(r, stage) for r in selected) + " |")
    return "\n".join(lines)


def direct_calibration_table(rows, dataset, depth):
    lookup = {(r["device"], r["fault_rate_percent"]): r for r in rows
              if r["dataset"] == dataset and r["analog_convolutions"] == depth
              and r["fault_kind"] == "gmax"}
    lines = ["| Stuck-high rate | PCM direct | PCM + digital calibration | OM direct | OM + digital calibration |",
             "|---|---:|---:|---:|---:|"]
    for rate in range(6):
        cells = []
        for device in ("pcm", "om"):
            row = lookup.get((device, rate))
            cells.extend(cell(row, stage) if row else "—" for stage in ("direct_deployment", "direct_calibration"))
        lines.append("| " + str(rate) + "% | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / "artifacts/cifar_recovery_five_stage_tables_20260920"
    out.mkdir(parents=True, exist_ok=True)
    summary_path = root / "results/cifar-sweep-analysis/summary.csv"
    clean_path = root / "results/cifar-sweep-analysis/clean_sources.csv"
    summary = read_csv(summary_path)
    clean = read_csv(clean_path)
    lookup = {}
    for r in summary:
        key = tuple(r[k] for k in ("dataset", "backend", "depth", "source", "case", "method", "epoch"))
        assert key not in lookup
        lookup[key] = r
    for policy, baseline_method, baseline_epoch in (("calibrated", "calibration", "5"), ("deployed", "none", "0")):
        rows = []
        for dataset in ("cifar10", "cifar100"):
            for device in ("pcm", "om"):
                for depth in (4, 8):
                    clean_rows = [r for r in clean if r["dataset"] == dataset and r["backend"] == device and r["depth"] == str(depth) and r["source"] == "digital" and r["case"] == "nominal"]
                    assert len(clean_rows) == 3
                    assert {r["array_seed"] for r in clean_rows} == {"251001", "251002", "251003"}
                    for kind in KINDS:
                        for rate in (0, 1, 2, 3, 5):
                            case = "nominal" if rate == 0 else f"{kind}_{rate * 10000}ppm"
                            row = {"dataset": dataset, "device": device, "analog_convolutions": depth,
                                   "fault_kind": kind, "fault_rate_percent": rate, "case": case,
                                   "corruption_hwa_source": "cdt_" + kind, "normal_hwa_source": "standard_hwa",
                                   "hwa_only_policy": policy, "recovery_method": "onchip_calibration",
                                   "recovery_weight_lr": 1e-4, "recovery_epochs": 5, "arrays": 3}
                            for metric in ("accuracy_percent", "teacher_kl"):
                                vals = [float(r[metric]) for r in clean_rows]
                                row["clean_" + metric + "_mean"] = statistics.mean(vals)
                                row["clean_" + metric + "_sd"] = statistics.stdev(vals)
                            terms = (("direct_deployment", "digital", "none", "0"),
                                     ("direct_calibration", "digital", "calibration", "5"),
                                     ("normal_hwa", "standard_hwa", baseline_method, baseline_epoch),
                                     ("corruption_hwa", "cdt_" + kind, baseline_method, baseline_epoch),
                                     ("normal_hwa_recovery", "standard_hwa", "onchip_calibration", "5"),
                                     ("corruption_hwa_recovery", "cdt_" + kind, "onchip_calibration", "5"))
                            for stage, source, method, epoch in terms:
                                item = lookup[(dataset, device, str(depth), source, case, method, epoch)]
                                for metric in ("accuracy_percent", "teacher_kl"):
                                    for statistic in ("mean", "sd"):
                                        name = metric + "_" + statistic
                                        row[stage + "_" + name] = float(item[name])
                            rows.append(row)
        assert len(rows) == 120
        with (out / (policy + "_all_conditions.csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        note = ("HWA-only rows include five epochs of digital calibration, with no physical weight training. "
                "Recovery rows include the same digital calibration plus physical weight updates."
                if policy == "calibrated" else
                "HWA-only rows are measured immediately after physical deployment, before calibration. "
                "Recovery rows include both digital calibration and physical weight updates; their improvement therefore cannot be attributed to weight training alone.")
        parts = ["# CIFAR recovery: deployment and adaptation comparison", "Headline condition: **5% stuck-high faults**. " + note,
                 "Entries are **test accuracy (%) / mean teacher KL**, nats at T=1, averaged across three physical arrays. "
                 "Clean is the original digital teacher before hardware effects. Direct deployment maps those original digital weights "
                 "to the specified faulty array, with no HWA, digital calibration or recovery. "
                 "The no-HWA plus digital-calibration row applies the same five-epoch calibration to those original weights, "
                 "so the effect of HWA can be compared at a matched calibration budget. "
                 "Normal HWA is the `standard_hwa` source, "
                 "not the separately trained noise-HWA source. Corruption-aware HWA is `cdt_gmax` in the headline tables. "
                 "Depth counts analog suffix convolutions; the classifier is also analog at both depths, and the preceding digital prefix remains frozen.",
                 "Both recovery columns use the same historical closed-loop/endpoint recovery campaign: five full 45,000-image epochs, "
                 "weight LR 1e-4, and joint weight/calibration updates. OM has the historical 640-pulse-per-cell cap. "
                 "PCM uses Gaussian endpoint reprogramming. The later tuned OM LR 1e-3 run is a separate study and is not mixed into these columns."]
        parts.append("Digital calibration trains per-matrix output gains, suffix batch-normalization scales and offsets, "
                     "and classifier bias against the teacher probabilities for five epochs over 45,000 adaptation images. "
                     "The physical weights, mapping scales, digital prefix and batch-normalization running statistics remain fixed. "
                     "This is post-deployment digital fine-tuning; it is additional learning beyond offline HWA.")
        for dataset in ("cifar10", "cifar100"):
            parts += ["## " + dataset.upper().replace("CIFAR", "CIFAR-"), pivot(rows, dataset)]
        parts += ["## Every fault condition", "The companion CSV retains unrounded means and sample SD for all 120 rows. "
                  "The 96 injected-fault rows cover both datasets, devices, depths and all 1/2/3/5% low/high/random cases. "
                  "The other 24 rows are nominal references for each of the three separately trained corruption-aware sources. "
                  "A nominal source is not chosen retrospectively from the test results."]
        for dataset in ("cifar10", "cifar100"):
            parts.append("### " + dataset.upper().replace("CIFAR", "CIFAR-"))
            parts.append("| Device | Convs | CDT trained for | Fault rate | " + " | ".join(stage_title(stage, name, policy) for stage, name in STAGES) + " |\n|---|---:|---|---:|" + "---:|" * len(STAGES))
            for r in rows:
                if r["dataset"] == dataset:
                    parts.append("| " + " | ".join([r["device"].upper(), str(r["analog_convolutions"]), KINDS[r["fault_kind"]], str(r["fault_rate_percent"]) + "%"] + [cell(r, stage) for stage, _ in STAGES]) + " |")
        # A single newline must join table rows; separate prose blocks elsewhere.
        headline = "\n\n".join(parts[:parts.index("## Every fault condition")]) + "\n"
        remaining = parts[parts.index("## Every fault condition"):]
        full = headline + "\n" + "\n".join((p if p.startswith("| ") else "\n" + p + "\n") for p in remaining) + "\n"
        (out / (policy + "_headline.md")).write_text(headline)
        (out / (policy + "_all_conditions.md")).write_text(full)
        stuck_rows = [r for r in rows if r["fault_kind"] == "gmax"]
        assert len(stuck_rows) == 40
        available_rates = sorted({r["fault_rate_percent"] for r in stuck_rows})
        missing_rates = sorted(set(range(1, 6)) - set(available_rates))
        assert available_rates == [0, 1, 2, 3, 5]
        coverage_note = ("Measured stuck-high rates: **1%, 2%, 3%, 5%**, with a 0% nominal reference. "
                         "**4% was not run**; a dash denotes an unavailable measurement. "
                         "All values are test accuracy (%) / teacher KL (nats, T=1), means over three modeled physical arrays.")
        focused = ["# CIFAR recovery: every measured stuck-high rate", coverage_note,
                   *parts[2:5], note]
        for dataset in ("cifar10", "cifar100"):
            focused.append("## " + dataset.upper().replace("CIFAR", "CIFAR-"))
            for rate in range(6):
                focused.append("### " + str(rate) + "% stuck high" if rate else "### 0%: nominal reference")
                focused.append(pivot(rows, dataset, rate) if rate in available_rates else "Not run; no measured result is available.")
        stuck_high = "\n\n".join(focused) + "\n"
        (out / (policy + "_stuck_high.md")).write_text(stuck_high)
        with (out / (policy + "_stuck_high.csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stuck_rows[0]))
            writer.writeheader(); writer.writerows(stuck_rows)
        import markdown
        css = "body{font:15px/1.5 system-ui,sans-serif;color:#203039;max-width:1250px;margin:35px auto;padding:0 20px}table{width:100%;border-collapse:collapse;margin:20px 0;font-size:13px}th,td{border-bottom:1px solid #d8e1e4;padding:10px;text-align:right}th{background:#eaf2f3}th:first-child,td:first-child{text-align:left}h1{font-size:28px}h2{margin-top:28px}@page{size:A4 landscape;margin:14mm}@media print{body{font-size:11px;margin:0}table{font-size:11px}h2{break-after:avoid}table,tr{break-inside:avoid}}"
        for title, content in (("headline", headline), ("all_conditions", full), ("stuck_high", stuck_high)):
            rendered = markdown.markdown(content, extensions=["tables"])
            (out / f"{policy}_{title}.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>CIFAR deployment and adaptation tables</title><style>{css}</style><body>{rendered}</body></html>')
        if policy == "calibrated":
            direct_parts = ["# CIFAR direct deployment and digital calibration: stuck-high faults", coverage_note,
                            "Every row uses the original digital source, with no HWA. Direct means the initial physical deployment. "
                            "Calibration means the fixed five-epoch digital adaptation, without physical weight updates. "
                            "The classifier is analog at both suffix depths. Clean digital references are "
                            "CIFAR-10: " + cell(stuck_rows[0], "clean") + "; CIFAR-100: " +
                            cell(next(r for r in stuck_rows if r["dataset"] == "cifar100"), "clean") + ".",
                            parts[4]]
            for depth in (8, 4):
                for dataset in ("cifar10", "cifar100"):
                    direct_parts += ["## " + dataset.upper().replace("CIFAR", "CIFAR-") + f" · {depth} analog convolutions plus classifier",
                                     direct_calibration_table(stuck_rows, dataset, depth)]
            direct_report = "\n\n".join(direct_parts) + "\n"
            (out / "stuck_high_direct_calibration.md").write_text(direct_report)
            rendered = markdown.markdown(direct_report, extensions=["tables"])
            (out / "stuck_high_direct_calibration.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>CIFAR stuck-high direct deployment and calibration</title><style>{css}</style><body>{rendered}</body></html>')
            fields = ["dataset", "device", "analog_convolutions", "fault_rate_percent", "case", "arrays"]
            fields += [k for k in stuck_rows[0] if k.startswith(("clean_", "direct_deployment_", "direct_calibration_"))]
            with (out / "stuck_high_direct_calibration.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader(); writer.writerows(stuck_rows)
    manifest = {"source_files": {}, "conditions_per_policy": 120,
                "headline_fault": "5% stuck high", "normal_hwa": "standard_hwa",
                "direct_deployment": "digital source, method none, epoch 0; no HWA, calibration or recovery",
                "direct_calibration": "digital source, method calibration, epoch 5; physical weights fixed",
                "recovery": "onchip_calibration, epoch 5, fixed weight LR 1e-4",
                "stuck_high": {"measured_rates_percent": available_rates, "missing_requested_rates_percent": missing_rates,
                               "measured_conditions_per_policy": 40},
                "clean_display": "Original digital baseline; KL below 0.00005 displays as zero; CSV preserves numerical residuals."}
    for path in (summary_path, clean_path):
        with path.open("rb") as f:
            manifest["source_files"][str(path)] = hashlib.file_digest(f, "sha256").hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
