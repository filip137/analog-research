"""exp-005 local collection: sync small files from Jean Zay and build summary tables.

python labs/domain_ft/exp005/collect.py [--no-sync]
-> <ART>/remote/ (runs without *.pt, deploy JSON, logs, slurm, queue)
   <ART>/summary/screen.csv     every training run (dev from final.json)
   <ART>/summary/deploy.csv     every deployed evaluation
   <ART>/summary/selection.json one LR per (target, arm), chosen by dev at c_p = 10 (exp-005 design):
                                 L0, L1: mean deployed dev over draws at c_p = 10; L2: deployed dev at c_p = 10, d271002.
                                 Ties (0.2 pt acc_norm, 0.5 % relative ppl) go to the lower LR; edge flagged.
   <ART>/summary/compare.csv    per (target, c_p) at the selected LRs: L0, L1, L2 draw means and SDs, paired
                                 differences L2 - L1 and L1 - L0 with their SE over draws, and B0/B1 from exp-002.
Selection reads only dev numbers. Canary runs (__u<U>) are excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path

ART = Path("/home/filip/analog-research/on-chip-learning/artifacts/domain-finetuning/exp-005")
ART2 = Path("/home/filip/analog-research/on-chip-learning/artifacts/domain-finetuning/exp-002")
JZR = "/lustre/fsn1/projects/rech/umg/ucy17uy/domain-ft-exp005-20261111"
TASKS = ("hellaswag",)
SEL_CP = 10.0
SEL_DRAW = 271002


def sync():
    rem = ART / "remote"
    rem.mkdir(parents=True, exist_ok=True)
    for sub in ("runs", "deploy", "logs", "slurm", "queue"):
        subprocess.run(["rsync", "-a", "--exclude", "*.pt", "-e", "ssh -o BatchMode=yes",
                        f"jean-zay:{JZR}/{sub}/", str(rem / sub) + "/"], timeout=900, check=False, capture_output=True)


def parse_run(path: Path):
    """runs/<target>@<base>/lora_fc16[_hwacp<c>]__lr<lr> -> (target, arm, cp, draw, lr) or None for canaries."""
    target, base = path.parent.name.split("@")
    name = path.name
    if "__u" in name or "__s" in name:
        return None
    m = re.fullmatch(r"lora_fc16(?:_hwacp([0-9.e+-]+))?__lr([0-9.e+-]+)", name)
    if not m:
        return None
    lr = float(m.group(2))
    if base == "std3_clean":
        return (target, "L1", float(m.group(1)), None, lr) if m.group(1) else (target, "L0", None, None, lr)
    b = re.fullmatch(r"std3_cp([0-9.e+-]+)_d(\d+)", base)
    return (target, "L2", float(b.group(1)), int(b.group(2)), lr) if b else None


def screen_rows():
    rows = []
    for f in sorted((ART / "remote/runs").glob("*@*/*/final.json")):
        p = parse_run(f.parent)
        if p is None:
            continue
        d = json.loads(f.read_text())
        t, arm, cp, draw, lr = p
        rows.append({"target": t, "arm": arm, "train_cp": cp, "train_draw": draw, "lr": lr, "status": d.get("status"),
                     "dev_final": d.get("dev_final"), "dev_best": d.get("dev_best"),
                     "sec_per_update": d.get("sec_per_update"), "train_tokens": d.get("train_tokens"),
                     "params": d.get("params"), "wall_s": d.get("wall_s"),
                     "run": f"runs/{f.parent.parent.name}/{f.parent.name}"})
    return rows


def deploy_rows():
    rows = []
    for f in sorted((ART / "remote/deploy").glob("*/*/*.json")):
        arm_dir = f.parent.name
        if "__u" in arm_dir:
            continue
        m = re.fullmatch(r"(L0|L1cp[0-9.e+-]+|L2)__lr([0-9.e+-]+)", arm_dir)
        if not m:
            continue
        d = json.loads(f.read_text())
        arm = m.group(1)[:2]
        rf = d.get("run_final") or {}
        rows.append({"target": d["dataset"], "arm": arm, "train_cp": rf.get("hwa_cp") or None, "lr": float(m.group(2)),
                     "c_p": d["c_p"], "draw": d["draw"], "dev": d["dev"]["metric"], "test": d["test"]["metric"],
                     "wikitext_ppl": d["wikitext_test"]["ppl"], "wikitext_teacher_kl": d["wikitext_test"].get("teacher_kl"),
                     "run_base": d.get("run_base"), "eval_s": d.get("eval_s"), "run": d.get("run")})
    return rows


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return m, sd, len(xs)


def select(deps):
    groups = {}
    for r in deps:
        if r["c_p"] != SEL_CP:
            continue
        if r["arm"] == "L2" and r["draw"] != SEL_DRAW:
            continue
        if r["arm"] == "L1" and r["train_cp"] != SEL_CP:
            continue
        groups.setdefault((r["target"], r["arm"]), {}).setdefault(r["lr"], []).append(r["dev"])
    sel = {}
    for (t, arm), by_lr in sorted(groups.items()):
        task = t in TASKS
        cand = sorted((lr, sum(v) / len(v), len(v)) for lr, v in by_lr.items())
        best = (max if task else min)(cand, key=lambda c: c[1])
        for c in cand:  # lowest LR within the tie band
            if (abs(c[1] - best[1]) <= 0.2) if task else (abs(c[1] / best[1] - 1) <= 0.005):
                best = c
                break
        lrs = [c[0] for c in cand]
        sel[f"{t}/{arm}"] = {"lr": best[0], "dev_c10": best[1], "n_draws": best[2], "grid": lrs,
                             "grid_dev": {f"{c[0]:g}": c[1] for c in cand},
                             "edge": len(lrs) > 1 and best[0] in (min(lrs), max(lrs)), "complete": len(lrs) >= 3}
    return sel


def bench(t, arm, cp, policy="rescale"):
    f = ART2 / "remote/deploy" / t / (arm if policy == "rescale" else f"{arm}@{policy}")
    pat = "cp0.json" if cp == 0 else f"cp{cp:g}_d*.json"  # not cp{cp}*: "cp1*" also matches cp10
    xs = [json.loads(p.read_text())["test"]["metric"] for p in f.glob(pat)] if f.exists() else []
    return mean_sd(xs)[0]


def compare(deps, sel):
    out = []
    for t in sorted({r["target"] for r in deps}):
        task = t in TASKS
        for cp in sorted({r["c_p"] for r in deps if r["c_p"] > 0}):
            row = {"target": t, "c_p": cp, "metric": "acc_norm" if task else "ppl"}
            by = {}
            for arm in ("L0", "L1", "L2"):
                s = sel.get(f"{t}/{arm}")
                if not s:
                    continue
                rs = [r for r in deps if r["target"] == t and r["arm"] == arm and r["lr"] == s["lr"] and r["c_p"] == cp
                      and (arm != "L1" or r["train_cp"] == cp)]
                by[arm] = {r["draw"]: r for r in rs}
                m, sd, n = mean_sd([r["test"] for r in rs])
                row.update({f"{arm}_test": m, f"{arm}_sd": sd, f"{arm}_n": n, f"{arm}_lr": s["lr"],
                            f"{arm}_wt_ppl": mean_sd([r["wikitext_ppl"] for r in rs])[0]})
            for a, b in (("L2", "L1"), ("L1", "L0"), ("L2", "L0")):
                if a in by and b in by:
                    common = sorted(set(by[a]) & set(by[b]))
                    diffs = [by[a][d]["test"] - by[b][d]["test"] for d in common]
                    if not task:  # relative ppl change, negative = a better
                        diffs = [by[a][d]["test"] / by[b][d]["test"] - 1 for d in common]
                    m, sd, n = mean_sd(diffs)
                    row[f"d_{a}_{b}"] = m
                    row[f"se_{a}_{b}"] = sd / math.sqrt(n) if sd is not None else None
                    row[f"n_{a}_{b}"] = n
            row["B0_test"] = bench(t, "B0", cp)
            row["B1_clean"] = bench(t, "B1", 0)
            row["B1_test"] = bench(t, "B1", cp)
            row["B1clip_test"] = bench(t, "B1", cp, "clip")
            out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    if not a.no_sync:
        sync()
    (ART / "summary").mkdir(parents=True, exist_ok=True)
    scr, deps = screen_rows(), deploy_rows()
    write_csv(ART / "summary/screen.csv", scr)
    write_csv(ART / "summary/deploy.csv", deps)
    sel = select(deps)
    (ART / "summary/selection.json").write_text(json.dumps(sel, indent=2))
    cmp_rows = compare(deps, sel)
    write_csv(ART / "summary/compare.csv", cmp_rows)
    print(f"runs {len(scr)} ({sum(r['status'] == 'done' for r in scr)} done), deployed evals {len(deps)}")
    for k, v in sel.items():
        print(f"select {k}: lr {v['lr']:g} dev@c10 {v['dev_c10']:.3f} grid {v['grid_dev']}{' EDGE' if v['edge'] else ''}")
    for r in cmp_rows:
        f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"
        print(f"{r['target']:9s} c_p={r['c_p']:<4g} B0 {f(r['B0_test'])} L0 {f(r.get('L0_test'))} "
              f"L1 {f(r.get('L1_test'))} L2 {f(r.get('L2_test'))} | L2-L1 {f(r.get('d_L2_L1'), 4)}"
              f"±{f(r.get('se_L2_L1'), 4)} L1-L0 {f(r.get('d_L1_L0'), 4)}")


if __name__ == "__main__":
    main()
