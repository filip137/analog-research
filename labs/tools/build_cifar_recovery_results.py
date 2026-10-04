"""Package existing CIFAR recovery evidence; never launch or select experiments.

Run with the py312 environment (matplotlib and markdown are required).
The explorer contains test results only. Development and historical studies
remain in separately named original tables with their own contracts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import statistics
import zipfile

FAMILIES = {
    "fault_sweep": ("cifar-sweep-analysis", "summary.csv", "per_array.csv"),
    "open_loop": ("cifar-om-openloop-analysis", "means.csv", "curves.csv"),
    "lr_confirmation": ("cifar-om-closedloop-lr-analysis", "summary.csv", "curves.csv"),
}
KEY = ("campaign", "dataset", "device", "convolutions", "initialization", "source",
       "case", "method", "weight_lr", "epoch")
SOURCES = {"digital": "Digital", "standard_hwa": "Standard HWA", "noise_hwa": "Noise HWA",
           "cdt_gmax": "CDT high", "cdt_open": "CDT low", "cdt_random": "CDT random"}
METHODS = {"none": "Deployed (no recovery)", "calibration": "Calibration",
           "rewrite": "Rewrite + calibration", "onchip_weights": "Weights only",
           "onchip_calibration": "Joint weights + calibration"}
CASE_KINDS = {"nominal": "No injected faults", "gmax": "Stuck high",
              "open": "Stuck low", "random": "Random stuck"}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def key(row):
    return tuple(row[k] for k in KEY)


def normalize(row, family, original):
    if family == "lr_confirmation" and (row["phase"] != "confirmation" or row["split"] != "test"):
        return None
    kind = row["case"].split("_")[0]
    rate = 0.0 if kind == "nominal" else int(row["case"].split("_")[1][:-3]) / 10000
    out = dict(campaign=family, dataset=row["dataset"], device=row.get("backend", "om"),
               convolutions=int(row.get("depth", row.get("convolutions"))),
               initialization=row.get("initialization", "legacy_deployment" if family == "fault_sweep" else "saved_p0"),
               source=row["source"], case=row["case"], fault_kind=CASE_KINDS[kind],
               fault_rate_percent=rate, method=row["method"],
               weight_lr=float(row.get("learning_rate", "0.0001")) if row["method"].startswith("onchip") else "",
               epoch=int(row["epoch"]), split="test", original_table=original)
    for field in ("array_seed", "examples"):
        if field in row:
            out[field] = int(row[field])
    for field in ("accuracy_percent", "teacher_kl", "teacher_agreement_percent", "cross_entropy"):
        for suffix in ("", "_mean", "_sd"):
            name = field + suffix
            if name in row and row[name] != "":
                out[name] = float(row[name])
    if "accuracy_percent_mean" in out:
        out["arrays"] = int(row.get("arrays", 3))
    return out


def validate_means(means, individual):
    grouped = defaultdict(list)
    seen = set()
    for r in individual:
        identity = key(r) + (r["array_seed"],)
        assert identity not in seen, identity
        seen.add(identity)
        assert 0 <= r["accuracy_percent"] <= 100 and math.isfinite(r["teacher_kl"])
        assert r["teacher_kl"] >= -1e-8
        grouped[key(r)].append(r)
    assert len(means) == len(grouped)
    for r in means:
        group = grouped[key(r)]
        assert len(group) == r["arrays"] == 3
        assert {x["array_seed"] for x in group} == {251001, 251002, 251003}
        for metric in ("accuracy_percent", "teacher_kl"):
            vals = [x[metric] for x in group]
            for suffix, actual in (("mean", statistics.mean(vals)), ("sd", statistics.stdev(vals))):
                expected = r[metric + "_" + suffix]
                assert math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-10), (key(r), metric, suffix)
    return {"summary_rows_recomputed": len(means), "individual_rows_checked": len(individual),
            "arrays_per_test_condition": 3, "duplicate_individual_rows": 0,
            "accuracy_and_kl_means_and_sample_sd_match": True}


def load_coverage(root):
    rows = []
    for folder in ("cifar-sweep-collected/v2", "cifar-om-openloop-collected/v1", "cifar-om-closedloop-lr-collected/v1"):
        for p in sorted((root / "results" / folder / "results").iterdir()):
            if not p.is_dir() or "canary" in p.name:
                continue
            path = p / "analysis/summary.json"
            d = json.loads(path.read_text())
            assert d["ready_for_review"] and d["validation_mode"] == "full_artifact_hashes", p
            assert all(a["running"] == a["failed"] == a["invalid"] == 0 for a in d["arms"])
            rows.append({"study": d["study_id"], "ready_for_review": True,
                         "validation_mode": d["validation_mode"], "generated_at": d["generated_at"],
                         "runs": len(d["runs"]), "summary_path": str(path), "summary_sha256": digest(path),
                         "controls": sum(r["metrics"].get("completed_controls", 0) for r in d["runs"]),
                         "replays": sum(r["metrics"].get("exact_replays", 0) for r in d["runs"])})
    assert len(rows) == 24
    return rows


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def pair(row, spread=False):
    if spread:
        return (f"{row['accuracy_percent_mean']:.2f} ± {row['accuracy_percent_sd']:.2f} / "
                f"{row['teacher_kl_mean']:.4g} ± {row['teacher_kl_sd']:.3g}")
    return f"{row['accuracy_percent_mean']:.2f} / {row['teacher_kl_mean']:.4f}"


def find(rows, **terms):
    found = [r for r in rows if all(r[k] == v for k, v in terms.items())]
    assert len(found) == 1, (terms, len(found))
    return found[0]


def write_appendix(out, endpoints):
    sections = ["# All recovery endpoints", "Accuracy (%) ± sample SD / teacher KL ± sample SD; three arrays. "
                "KL is KL(original digital teacher || student), nats at T=1. Deployed values are epoch 0; "
                "all recovery values are epoch 5. A dash means the method was not part of that campaign. "
                "These are the complete 3,936 test endpoint means; no outcome was selected by test score."]
    grouping = defaultdict(list)
    for r in endpoints:
        grouping[(r["campaign"], r["dataset"], r["device"], r["convolutions"], r["initialization"], r["source"])].append(r)
    for group, rows in sorted(grouping.items()):
        family, dataset, device, depth, init, source = group
        sections.append(f"## {family} · {dataset} · {device.upper()} · {depth} convs · {init} · {SOURCES[source]}")
        if family == "lr_confirmation":
            heads = ["Fault", "LR 1e-4", "LR 1e-3"]
            body = []
            for case in sorted({r["case"] for r in rows}):
                body.append([case] + [pair(find(rows, case=case, weight_lr=lr), True) for lr in (1e-4, 1e-3)])
        else:
            methods = ["none", "calibration"] + (["rewrite"] if family == "fault_sweep" else []) + ["onchip_weights", "onchip_calibration"]
            heads = ["Fault"] + [METHODS[m] for m in methods]
            body = [[case] + [pair(find(rows, case=case, method=m), True) for m in methods]
                    for case in sorted({r["case"] for r in rows})]
        sections.append(table(heads, body))
    (out / "all_endpoints.md").write_text("\n\n".join(sections) + "\n")


def make_figures(out, endpoints):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160, "pdf.fonttype": 42})
    rows = [r for r in endpoints if r["campaign"] == "fault_sweep" and r["method"] != "none"]
    lookup = {(r["dataset"], r["device"], r["convolutions"], r["case"], r["source"], r["method"]): r for r in rows}
    with PdfPages(out / "recovery_figures.pdf") as pdf:
        for ds in ("cifar10", "cifar100"):
            for kind in ("gmax", "open", "random"):
                fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
                for i, (device, depth) in enumerate((("pcm", 4), ("pcm", 8), ("om", 4), ("om", 8))):
                    for source, color in (("standard_hwa", "#b45a3c"), ("noise_hwa", "#4872a5"), ("cdt_" + kind, "#168477")):
                        for method, style in (("calibration", "--"), ("onchip_calibration", "-")):
                            cases = ["nominal"] + [f"{kind}_{rate*10000}ppm" for rate in (1, 2, 3, 5)]
                            rr = [lookup[(ds, device, depth, c, source, method)] for c in cases]
                            label = SOURCES[source] + (" + calibration" if method == "calibration" else " + joint recovery")
                            for j, metric in enumerate(("accuracy_percent", "teacher_kl")):
                                y = [r[metric + "_mean"] for r in rr]
                                sd = [r[metric + "_sd"] for r in rr]
                                ax = axes[i, j]
                                ax.plot([0, 1, 2, 3, 5], y, style, color=color, marker="o", ms=3, label=label)
                                ax.fill_between([0, 1, 2, 3, 5], [max(a-b, 1e-8) for a,b in zip(y,sd)],
                                                [a+b for a,b in zip(y,sd)], color=color, alpha=.07)
                    for j in range(2):
                        ax = axes[i, j]
                        ax.set_title(f"{device.upper()} · {depth} analog convolutions + classifier")
                        ax.set_xlabel("Injected device fault rate (%)")
                        ax.set_xticks([0, 1, 2, 3, 5]); ax.grid(alpha=.18)
                    axes[i, 0].set_ylabel("Test accuracy (%)")
                    axes[i, 0].set_ylim(bottom=max(0, axes[i, 0].get_ylim()[0]), top=min(100, axes[i, 0].get_ylim()[1]))
                    axes[i, 1].set_ylabel("Teacher KL (nats; log scale)")
                    axes[i, 1].set_yscale("log")
                fig.suptitle(f"{ds.upper().replace('CIFAR', 'CIFAR-')} recovery · {CASE_KINDS[kind].lower()}\n"
                             "Five full epochs; three-array means ± sample SD; fixed weight LR 1e-4", fontsize=15)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
                stem = f"{ds}_{kind}"
                fig.savefig(out / (stem + ".png")); fig.savefig(out / (stem + ".svg")); pdf.savefig(fig)
                plt.close(fig)


def make_report(out, endpoints, coverage, validation, root):
    sweep = [r for r in endpoints if r["campaign"] == "fault_sweep"]
    ol = [r for r in endpoints if r["campaign"] == "open_loop"]
    lr = [r for r in endpoints if r["campaign"] == "lr_confirmation"]
    intro = """# CIFAR-10 and CIFAR-100: complete recovery results

Prepared from completed local artifacts on 20 September 2026. This package presents the numerical evidence; scientific review and managed-manifest finalization remain pending. No experiment was launched or selected for this report.

[Search all test results](explorer.html) · [Every endpoint with array spread](all_endpoints.md) · [Endpoint CSV](endpoints.csv) · [All test epochs](all_test_epochs.csv) · [Individual arrays](test_per_array.csv) · [Figures PDF](recovery_figures.pdf)

## Scope and how to read the tables

The complete pretrained ResNet-32 is retained. Its last **four or eight convolutions plus classifier** are analog; the preceding digital prefix is frozen. These depths are not four/eight devices per weight. Digital teacher accuracy is **93.53% on CIFAR-10** and **70.16% on CIFAR-100**. Tables show **accuracy (%) / KL(original digital teacher || student)** in nats at temperature 1. Higher accuracy and lower KL are favorable. All headline values are test-set means across **three physical arrays**, with 10,000 test images per array.

Recovery uses **five full epochs over 45,000 adaptation images**, 225,000 image presentations and 3,520 minibatch updates. Every saved epoch is included in the CSVs and explorer. Calibration trains digital gains, suffix BN affine parameters and classifier bias; joint recovery also changes physical weights. Weight-only recovery and frozen-target rewrite controls remain in the complete tables.

**Standard HWA** uses nominal device-aware training; **noise HWA** is a separately selected generic robustness source; **CDT** is HWA trained with the same failure type and mixed 2/3/5% rates. CDT sources cover nominal plus their own fault type. Missing other-type CDT combinations were not run. All fitting and LR choices use development evidence, never the test tables below.

| Campaign | Coverage | Key distinction |
|---|---|---|
| PCM/OM fault sweep V2 | 6,480 outcomes; both datasets, 4/8 convolutions, nominal and 1/2/3/5% low/high/random faults | Six source labels; five controls; fixed recovery weight LR 1e-4 |
| OM open-loop | 5,184 outcomes; same depth/fault/source grid, four controls | Exact saved deployment and fresh RESET are separate initialization arms; LR 1e-4; no cumulative pulse cap |
| OM closed-loop LR | 192 development + 144 confirmation trajectories | Digital/standard-HWA sources only; nominal/3%/5% high faults; development chooses among 1e-4/3e-4/1e-3; confirmation compares selected 1e-3 against fresh uncapped 1e-4 |

Outcomes include no-recovery controls. Overlapping saved deployments are comparisons, not independent repetitions. Native full-study coverage is ready for review in all 24 study roots; smoke studies are excluded from these scientific tables.

## Five-percent stuck-high faults: stronger HWA controls

This is the broad fault sweep with the historical OM closed-loop controller and fixed LR 1e-4. Each source has its own deployment. The within-CDT contrast is paired on the same initial physical state; comparison with noise HWA changes the source.
"""
    body = []
    for ds in ("cifar10", "cifar100"):
        for device in ("pcm", "om"):
            for depth in (4, 8):
                terms = dict(dataset=ds, device=device, convolutions=depth, case="gmax_50000ppm")
                noise = find(sweep, **terms, source="noise_hwa", method="calibration")
                cal = find(sweep, **terms, source="cdt_gmax", method="calibration")
                joint = find(sweep, **terms, source="cdt_gmax", method="onchip_calibration")
                body.append([ds, device.upper(), depth, pair(noise), pair(cal), pair(joint),
                             f"{joint['accuracy_percent_mean']-cal['accuracy_percent_mean']:+.2f}"])
    parts = [intro, table(["Dataset", "Device", "Convs", "Noise HWA + calibration", "CDT + calibration", "CDT + joint", "Within-CDT gain (pp)"], body),
             "CIFAR-100 gains are larger, but its OM CDT sources can be weak before recovery. At eight convolutions/5% high faults, generic noise HWA + rewrite/calibration reaches 42.18% / 1.5639, versus CDT + joint at 47.98% / 1.3690. The resulting 5.80-point gain is much smaller than the within-CDT 26.43-point gain. This is finite-budget evidence, not proof that recovery is necessary against fully optimized HWA."]
    parts.append("## All injected fault rates after CDT\n\nEach entry is calibration → joint recovery, with accuracy / KL on both sides. All 96 depth/device/dataset/type/rate cases are included below. Nominal and other sources are in the complete endpoint appendix.")
    for ds in ("cifar10", "cifar100"):
        parts.append("### " + ds.upper().replace("CIFAR", "CIFAR-"))
        body = []
        for device in ("pcm", "om"):
            for depth in (4, 8):
                for kind in ("open", "gmax", "random"):
                    for rate in (1, 2, 3, 5):
                        terms = dict(dataset=ds, device=device, convolutions=depth, source="cdt_"+kind, case=f"{kind}_{rate*10000}ppm")
                        cal = find(sweep, **terms, method="calibration")
                        joint = find(sweep, **terms, method="onchip_calibration")
                        body.append([device.upper(), depth, CASE_KINDS[kind], f"{rate}%", pair(cal), pair(joint)])
        parts.append(table(["Device", "Convs", "Fault type", "Rate", "Calibration", "Joint recovery"], body))
    parts.append("## OM closed-loop learning-rate confirmation\n\nThese are all 24 dataset/depth/source/fault conditions in the focused follow-up. Both closed-loop rates share the exact P0 and current loulou backend. Open-loop is the earlier saved-P0 run; small historical backend differences remain. All runs include digital calibration. Higher LR is not universally better on nominal arrays. These settings were not retuned for CDT or other fault kinds.")
    for source in ("standard_hwa", "digital"):
        body = []
        for ds in ("cifar10", "cifar100"):
            for depth in (4, 8):
                for case in ("nominal", "gmax_30000ppm", "gmax_50000ppm"):
                    terms = dict(dataset=ds, convolutions=depth, source=source, case=case)
                    body.append([ds, depth, case, pair(find(lr, **terms, weight_lr=1e-4)),
                                 pair(find(lr, **terms, weight_lr=1e-3)),
                                 pair(find(ol, **terms, initialization="saved_p0", method="onchip_calibration"))])
        parts.append("### " + SOURCES[source])
        parts.append(table(["Dataset", "Convs", "Fault", "Closed-loop 1e-4", "Closed-loop 1e-3", "Open-loop 1e-4"], body))
    selection = json.loads((root / "results/cifar-om-closedloop-lr-analysis/selection.json").read_text())
    parts.append("### Development-only LR selection\n\nEach score averages raw final development KL across two sources, three cases and two independent development arrays (12 endpoints). Severe cases can dominate. CIFAR-10/four-convolution 3e-4 and 1e-3 scores are nearly tied; the declared minimum rule selected 1e-3. These are not test scores.")
    parts.append(table(["Dataset/depth", "1e-4", "3e-4", "1e-3", "Selected"],
                       [[name]+[f"{group['mean_development_kl_by_rate'][str(v)]:.6f}" for v in (1e-4, 3e-4, 1e-3)]+["1e-3"]
                        for name, group in sorted(selection["groups"].items())]))
    parts.append("## OM open-loop: saved deployment versus fresh RESET\n\nStandard-HWA sources below, after five epochs of joint recovery. Fresh RESET programming changes initialization as well as the recovery controller; it is not a matched-P0 controller comparison. HWA was inherited, not refitted for that programmer.")
    body = []
    for ds in ("cifar10", "cifar100"):
        for depth in (4, 8):
            terms = dict(dataset=ds, convolutions=depth, source="standard_hwa", case="nominal")
            body.append([ds, depth, pair(find(ol, **terms, initialization="saved_p0", method="onchip_calibration")),
                         pair(find(ol, **terms, initialization="reset_open_loop", method="none")),
                         pair(find(ol, **terms, initialization="reset_open_loop", method="onchip_calibration"))])
    parts.append(table(["Dataset", "Convs", "Saved P0 + recovery", "Fresh RESET deployment", "Fresh RESET + recovery"], body))
    parts.append("## Write costs and uncertainty\n\nPCM recovery performs 3,520 full Gaussian endpoint reprograms in a weight-update arm; no extra pulse/P&V procedure is added. OM uses explicit pulse updates. The original sweep permits at most 640 recovery pulses per cell; the open-loop and LR follow-ups remove that cumulative cap while retaining one pulse per cell per update. Counts are not energy measurements; optimizers and gradients are digital. The open-loop recovery controller performs zero verification reads. Initial programming and historical closed-loop controllers have different read costs.\n\nFor eight convolutions/3% stuck-high faults with standard HWA, increasing closed-loop LR raises pulses per cell from 0.204 to 2.956 on CIFAR-10 and 0.619 to 7.189 on CIFAR-100. Selected-LR accuracy SD is 1.31 and 9.52 percentage points respectively. Every endpoint's sample SD appears in the explorer/CSV; per-array costs and comparisons are in the original tables.\n\nOne teacher/HWA training seed is used per source. Three physical arrays measure array variation, not training-seed variation or significance. Thirty-one of 40 selected HWA fits retain the late-convergence flag at the 60-epoch cap. Results model specified PCM endpoints and the OM preset, not all hardware nonidealities. Fault masks are matched where the contracts permit; PCM two-bank and OM active-cell semantics differ. Scientific interpretation remains unfinalized.")
    parts.append("## Recovery curves\n\nSolid lines: joint recovery; dashed: calibration. Shading is ± one sample SD across three arrays. KL uses a log axis so severe failures remain visible. Separate downloadable figures include all three fault types.")
    for ds in ("cifar10", "cifar100"):
        parts.append(f"![{ds} stuck-high curves]({ds}_gmax.png)")
    parts.append("## Earlier recovery studies\n\nThese historical studies use different cohorts, budgets, array draws or HWA recipes. Their complete reports, CSVs and figures are retained separately in the package; do not pool them with the latest sweep.")
    for name, label in (("cifar-pcm-fault-analysis", "Initial one-pass PCM fault screen"),
                        ("cifar-hwa-cdt-analysis", "Earlier HWA/CDT comparison"),
                        ("cifar-epochs-analysis", "Twenty passes over the 5,000-image PCM cohort"),
                        ("cifar-full5-analysis", "Initial five full epochs, four analog convolutions, PCM and OM")):
        parts.append(f"- [{label}](originals/results/{name}/report.md)")
    parts.append("## Verification and reproducibility\n\nAll 16,224 exported test means and sample SDs were recomputed from 48,672 individual-array records and checked against the source tables. All 3,936 endpoint means are present, including degraded and near-chance cases. Source files are copied byte-for-byte under `originals/`; `package_manifest.json` records their SHA-256 hashes and the artifact-verified study summaries. Raw physical checkpoints remain at their original collected-study paths and are not duplicated in this report archive. `coverage.json` includes the exact prepared study identities, summary paths and verification timestamps. Development-only LR curves and historical paired contrasts remain in their original CSVs, never mixed into test exports.")
    report = "\n\n".join(parts) + "\n"
    (out / "report.md").write_text(report)
    import markdown
    rendered = markdown.markdown(report, extensions=["tables"])
    css = """body{font:15px/1.55 system-ui,sans-serif;color:#203039;max-width:1180px;margin:40px auto;padding:0 24px}h1{font-size:34px}h2{margin-top:40px}a{color:#126e78}table{border-collapse:collapse;width:100%;font-size:12px;margin:20px 0}th,td{border-bottom:1px solid #d8e1e4;padding:7px;text-align:left}th{background:#eaf2f3}img{max-width:100%}p{max-width:1050px}@page{size:A4 landscape;margin:14mm}@media print{body{font-size:10px;margin:0;max-width:none}h1{font-size:24px}h2{font-size:18px}table{font-size:9px}tr{break-inside:avoid}thead{display:table-header-group}img{max-height:175mm;width:auto;display:block;margin:auto}a{color:inherit}}"""
    (out / "report.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>CIFAR recovery results</title><style>{css}</style><body>{rendered}</body></html>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / "artifacts/cifar_recovery_full_results_20260920"
    out.mkdir(parents=True, exist_ok=True)
    means, individual, input_files = [], [], []
    for family, (folder, summary_name, array_name) in FAMILIES.items():
        for filename, target in ((summary_name, means), (array_name, individual)):
            path = root / "results" / folder / filename
            rel = str(path.relative_to(root))
            input_files.append(path)
            target.extend(x for r in read_csv(path) if (x := normalize(r, family, rel)) is not None)
    validation = validate_means(means, individual)
    endpoints = [r for r in means if r["epoch"] == 5 or r["method"] == "none"]
    assert len(means) == 16224 and len(individual) == 48672 and len(endpoints) == 3936
    coverage = load_coverage(root)
    for filename, rows in (("endpoints.csv", endpoints), ("all_test_epochs.csv", means), ("test_per_array.csv", individual)):
        write_csv(out / filename, rows)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print("Verified", validation, flush=True)

    folders = [v[0] for v in FAMILIES.values()] + ["cifar-pcm-fault-analysis", "cifar-hwa-cdt-analysis", "cifar-epochs-analysis", "cifar-full5-analysis"]
    originals = []
    for folder in folders:
        for src in sorted((root / "results" / folder).iterdir()):
            if not src.is_file() or src.suffix not in (".csv", ".json", ".md", ".png", ".pdf", ".svg"):
                continue
            dst = out / "originals" / src.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
            originals.append({"original": str(src), "copy": str(dst.relative_to(out)), "sha256": digest(src), "bytes": src.stat().st_size})
            assert digest(dst) == originals[-1]["sha256"]
    for src in sorted((root / "docs").glob("cifar*.md")):
        dst = out / "originals/docs" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        originals.append({"original": str(src), "copy": str(dst.relative_to(out)), "sha256": digest(src), "bytes": src.stat().st_size})
    fits = json.loads((root / "results/cifar-sweep-analysis/results.json").read_text())["sources"]
    fit_rows = []
    for r in fits:
        fit_rows.append({**{k:v for k,v in r.items() if k != "fit"},
                         **{k:v for k,v in (r.get("fit") or {}).items() if k not in ("history", "rate_histogram")}})
    write_csv(out / "source_fit_selection.csv", fit_rows)
    write_appendix(out, endpoints)
    print("Writing figures", flush=True)
    make_figures(out, endpoints)
    make_report(out, endpoints, coverage, validation, root)
    template = Path(__file__).with_name("cifar_recovery_explorer.html").read_text()
    payload = json.dumps(means, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (out / "explorer.html").write_text(template.replace("__RESULT_DATA__", payload))
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "source_script": str(Path(__file__).resolve()),
                "source_script_sha256": digest(Path(__file__)), "scope": list(FAMILIES),
                "test_endpoint_means": len(endpoints), "test_epoch_means": len(means), "test_array_records": len(individual),
                "scientific_review": "pending", "validation": validation, "originals": originals}
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Report ready:", out, flush=True)


if __name__ == "__main__":
    main()
