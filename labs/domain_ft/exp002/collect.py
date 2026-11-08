"""exp-002 local collection: sync small files from Jean Zay and build summary tables.

python labs/domain_ft/exp002/collect.py [--no-sync]
-> <ART>/remote/ (runs without state.pt, deploy JSON, logs, slurm, queue), <ART>/summary/{screen.csv,
   selection.json, benchmark.csv}. Selection: best final dev (U = 1,000) per (target, arm); ties (0.2 pt acc_norm,
   0.5 % relative ppl) go to the lower LR. Selection reads only final.json (dev).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

ART = Path("/home/filip/analog-research/on-chip-learning/artifacts/domain-finetuning/exp-002")
JZR = "/lustre/fsn1/projects/rech/umg/ucy17uy/domain-ft-exp002-20261108"
ARMS = {"full": "B1", "full_hwa0.03": "B2", "mlp": "B3"}
TASKS = ("hellaswag",)


def sync():
    rem = ART / "remote"
    rem.mkdir(parents=True, exist_ok=True)
    for sub in ("runs", "deploy", "logs", "slurm", "queue", "drn"):
        subprocess.run(["rsync", "-a", "--exclude", "*.pt", "--exclude", "ckpt*", "-e", "ssh -o BatchMode=yes",
                        f"jean-zay:{JZR}/{sub}/", str(rem / sub) + "/"], timeout=900, check=False,
                       capture_output=True)


def screen_rows():
    rows = []
    for f in sorted((ART / "remote/runs").glob("*@std3_clean/*/final.json")):
        d = json.loads(f.read_text())
        target = f.parent.parent.name.split("@")[0]
        name = f.parent.name
        if "__u" in name:  # canary / short runs
            continue
        method = name.split("__lr")[0]
        rows.append({"target": target, "arm": ARMS.get(method, method), "method": method, "lr": d.get("lr"),
                     "status": d.get("status"), "dev_final": d.get("dev_final"), "dev_best": d.get("dev_best"),
                     "dev_best_update": d.get("dev_best_update"), "sec_per_update": d.get("sec_per_update"),
                     "train_tokens": d.get("train_tokens"), "params": d.get("params"), "wall_s": d.get("wall_s"),
                     "run": f"runs/{f.parent.parent.name}/{name}"})
    return rows


def select(rows):
    sel = {}
    groups = {}
    for r in rows:
        if r["status"] == "done" and r["dev_final"] is not None:
            groups.setdefault((r["target"], r["arm"]), []).append(r)
    for (t, arm), rs in sorted(groups.items()):
        task = t in TASKS
        rs = sorted(rs, key=lambda r: r["lr"])
        best = max(rs, key=lambda r: r["dev_final"]) if task else min(rs, key=lambda r: r["dev_final"])
        for r in rs:  # lowest LR within the tie band
            tie = (abs(r["dev_final"] - best["dev_final"]) <= 0.2) if task else \
                (abs(r["dev_final"] / best["dev_final"] - 1) <= 0.005)
            if tie:
                best = r
                break
        lrs = [r["lr"] for r in rs]
        sel[f"{t}/{arm}"] = {"lr": best["lr"], "dev_final": best["dev_final"], "run": best["run"],
                             "grid": lrs, "edge": best["lr"] in (min(lrs), max(lrs)) and len(lrs) > 1,
                             "complete": len(lrs) >= 3}
    return sel


def bench_rows():
    out = []
    for f in sorted((ART / "remote/deploy").glob("*/*/*.json")) + \
            sorted((ART / "remote/drn").glob("exp002/*/*/deploy/*.json")):
        d = json.loads(f.read_text())
        if str(d.get("arm", "")).startswith("canary"):
            continue
        arch = d.get("arch", "crossbar")
        c = (d.get("change_cost") or {}).get("total", {})
        rf = d.get("run_final") or {}
        prog = d.get("programming", {})
        wt = d.get("wikitext_test", {})
        out.append({
            "arch": arch, "target": d["dataset"], "arm": d["arm"], "lr": rf.get("lr"), "c_p": d["c_p"],
            "draw": d["draw"] if d["draw"] is not None else "",
            "metric": "acc_norm" if d["dataset"] in TASKS else "ppl",
            "dev_metric": d["dev"]["metric"], "test_metric": d["test"]["metric"],
            "test_kl_to_offline": d["test"].get("kl_to_offline", ""),
            "wikitext_ppl": wt.get("ppl"), "wikitext_teacher_kl": wt.get("teacher_kl"),
            "devices_written": prog.get("devices"), "channels_rescaled": prog.get("channels_rescaled"),
            "out_of_range_active": prog.get("out_of_range_active"),
            "out_of_range_reference": prog.get("out_of_range_reference"),
            "moved_any": c.get("moved_any"), "moved_gt_1sigma": c.get("moved_gt_1sigma"),
            "moved_gt_2sigma": c.get("moved_gt_2sigma"),
            "bytes_dense_fp16_changed_analog": c.get("bytes_dense_fp16_changed_analog"),
            "bytes_sparse_idx32_fp16_2sigma": c.get("bytes_sparse_idx32_fp16_2sigma"),
            "bytes_digital_changed_fp16": c.get("bytes_digital_changed_fp16"),
            "bytes_scales_changed_fp16": c.get("bytes_scales_changed_fp16"),
            "digital_changed": c.get("digital_changed"),
            "train_tokens": rf.get("train_tokens", 0) if rf else 0, "trainable_params": rf.get("params", 0) if rf else 0,
            "train_updates": rf.get("updates_done", 0) if rf else 0,
            "file": str(f.relative_to(ART))})
    key = {"crossbar": 0, "drn": 1}
    out.sort(key=lambda r: (key.get(r["arch"], 9), r["target"], r["arm"], r["c_p"], str(r["draw"])))
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    if not a.no_sync:
        sync()
    (ART / "summary").mkdir(parents=True, exist_ok=True)
    rows = screen_rows()
    write_csv(ART / "summary/screen.csv", rows)
    sel = select(rows)
    (ART / "summary/selection.json").write_text(json.dumps(sel, indent=2))
    b = bench_rows()
    write_csv(ART / "summary/benchmark.csv", b)
    print(f"screen {len(rows)} runs ({sum(r['status'] == 'done' for r in rows)} done); selections {len(sel)}; "
          f"benchmark rows {len(b)}")
    for k, v in sel.items():
        print(f"  {k}: lr {v['lr']:g} dev {v['dev_final']:.3f}{' EDGE' if v['edge'] else ''}"
              f"{'' if v['complete'] else ' (incomplete)'}")


if __name__ == "__main__":
    main()
