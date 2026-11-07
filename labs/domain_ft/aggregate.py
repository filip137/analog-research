"""Build summary/ tables from synced runs (dev-only selection; test reported at the selected LR).

python labs/domain_ft/aggregate.py [artifact_root]
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan as P  # noqa: E402

ART = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/filip/analog-research/on-chip-learning/artifacts/domain-finetuning/prelim-20261108")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def majority() -> dict:
    out = {}
    for f in (ART / "data" / "receipts").glob("*-opt125m-dft1.json"):
        r = json.loads(f.read_text())
        if "majority_dev_percent" in r:
            out[r["task"]] = r["majority_dev_percent"]
    # hwa task files: dev label distribution from the files (cached)
    cache = ART / "summary" / "majority_hwa.json"
    if cache.exists():
        out.update(json.loads(cache.read_text()))
    return out


def main_test(d: dict):
    t = d.get("test") or {}
    if "test" not in t:
        return None, None, None
    x = t["test"]
    v = x.get("acc_norm", x.get("ppl"))
    v2 = t.get("test_mismatched", {}).get("acc_norm")
    return v, v2, x.get("n", x.get("blocks"))


def ret_cols(d: dict) -> dict:
    r = d.get("retention") or {}
    if not r:
        return {}
    return {"wikitext_ppl": r["wikitext_test"]["ppl"], "wikitext_kl": r["wikitext_test"].get("teacher_kl"),
            "piqa_accn": r["piqa_test"]["acc_norm"], "arc_easy_accn": r["arc_easy_test"]["acc_norm"],
            "lambada_acc": r["lambada_2000"]["acc"]}


def main():
    S = ART / "summary"
    S.mkdir(parents=True, exist_ok=True)
    runs = P.read_runs(ART / "runs")
    maj = majority()

    # screen.csv: every run
    screen = []
    for jid, d in sorted(runs.items()):
        ds, name = jid.split("/")
        parts = name.split("__")
        t1, t2, nt = main_test(d)
        row = {"id": jid, "dataset": ds.split("@")[0], "base": ds.split("@")[1] if "@" in ds else "opt",
               "method": parts[0], "lr": float(parts[1][2:]) if len(parts) > 1 else "",
               "updates": d.get("updates_done", d.get("updates_planned")), "status": d.get("status"),
               "params": d.get("params"), "metric": d.get("metric", ""), "dev_final": d.get("dev_final"),
               "dev_best": d.get("dev_best"), "dev_best_update": d.get("dev_best_update"),
               "test": t1, "test_mismatched": t2, "n_test": nt,
               "sec_per_update": d.get("sec_per_update"), "peak_mem_gb": d.get("peak_mem_gb"),
               "wall_s": d.get("wall_s"), **ret_cols(d)}
        dc = d.get("dev_curve")
        if dc:
            row["dev_curve"] = " ".join(f"{u}:{v:.4g}" for u, v in dc)
        screen.append(row)
    write_csv(S / "screen.csv", screen)

    # selection.json
    keys = sorted({(r["dataset"] + ("" if r["base"] == "opt" else "@" + r["base"]), r["method"], r["updates"])
                   for r in screen if r["method"] != "zs" and r["dataset"] != "gap" and r["status"] == "done"})
    selection = {}
    for ds, m, u in keys:
        u = int(u or 1000)
        sel = P.select(runs, ds, m, u)
        if sel is None:
            continue
        d = sel.pop("run")
        t1, t2, nt = main_test(d)
        grid = P.grid_of(m)
        sel.update({"test_selected": t1, "test_mismatched_selected": t2,
                    "best_over_checkpoints_at_selected": d.get("dev_best"),
                    "complete_grid": bool(grid) and all(any(abs(x - y) / y < 1e-6 for x in sel["lrs"]) for y in grid),
                    "retention_selected": ret_cols(d)})
        selection[f"{ds}|{m}|u{u}"] = sel
    (S / "selection.json").write_text(json.dumps(selection, indent=2))

    # gate.csv
    gate = []
    for ds in P.GATE_SETS:
        zs = runs.get(f"{ds}/zs", {})
        sel = selection.get(f"{ds}|full|u1000")
        lm = ds in P.DOMAINS
        row = {"dataset": ds, "zs_dev": zs.get("dev_final"), "zs_test": main_test(zs)[0] if zs else None}
        row["majority_or_chance"] = None if lm else maj.get(ds)
        if sel:
            row.update({"full_lr": sel["lr"], "full_edge": sel["edge"], "full_lrs_done": len(sel["lrs"]),
                        "full_dev": sel["dev_selected"], "full_test": sel["test_selected"],
                        "full_dev_by_lr": " ".join(f"{a:g}:{b:.4g}" for a, b in zip(sel["lrs"], sel["dev"]))})
            n_test = None
            for jid, d in runs.items():
                if jid.startswith(f"{ds}/full__lr") and main_test(d)[2]:
                    n_test = main_test(d)[2]
            row["n_test"] = n_test
            if lm:
                if row["zs_dev"]:
                    row["dev_gain"] = 100 * (1 - sel["dev_selected"] / row["zs_dev"])
                    row["dev_gain_unit"] = "% ppl reduction vs zs"
                row["pass"] = "n/a (LM domain)"
            else:
                ref = max(x for x in (row["zs_dev"], row["majority_or_chance"]) if x is not None)
                row["dev_gain"] = sel["dev_selected"] - ref
                row["dev_gain_unit"] = "pt vs max(zs, majority)"
                ok = row["dev_gain"] >= 10 and (n_test or 0) >= 3000
                row["pass"] = ("control: " if ds == "sst2" else "") + ("PASS" if ok else "fail")
            if ds == "mnli":
                row["full_test_mismatched"] = sel["test_mismatched_selected"]
                row["zs_test_mismatched"] = (zs.get("test") or {}).get("test_mismatched", {}).get("acc_norm")
        gate.append(row)
    write_csv(S / "gate.csv", gate)

    # capacity.csv
    cap = []
    for dom in P.DOMAINS:
        for key, sel in selection.items():
            ds, m, u = key.split("|")
            if ds != dom or u != "u1000":
                continue
            fam, size = (m.rstrip("0123456789"), m[len(m.rstrip("0123456789")):])
            for lr, dv in zip(sel["lrs"], sel["dev"]):
                d = runs.get(P.job_id(dom, m, lr))
                if not d:
                    continue
                cap.append({"domain": dom, "method": m, "family": fam, "size": size, "params": d.get("params"),
                            "lr": lr, "selected": lr == sel["lr"], "dev_ppl": dv, "test_ppl": main_test(d)[0],
                            **{k: v for k, v in ret_cols(d).items() if k.startswith("wikitext")}})
    write_csv(S / "capacity.csv", cap)

    # deltaw.csv
    dw = []
    for jid in sorted(runs):
        f = ART / "runs" / jid / "deltaw.json"
        if not f.exists():
            continue
        st = json.loads(f.read_text())
        ds, name = jid.split("/")
        agg = {}
        for mat, e in st.items():
            row = {"id": jid, "matrix": mat, "rel_norm": e["rel_norm"], "r50": e.get("r50"), "r90": e.get("r90"),
                   "r99": e.get("r99")}
            for grp in ("xbar", "drn"):
                for k, v in (e.get(grp) or {}).items():
                    row[f"{grp}_{k}"] = v
            dw.append(row)
            for k, v in row.items():
                if isinstance(v, (int, float)) and v is not None:
                    agg.setdefault(k, []).append(v)
        dw.append({"id": jid, "matrix": "MEAN", **{k: statistics.fmean(v) for k, v in agg.items()}})
    write_csv(S / "deltaw.csv", dw)

    # domain_gap.csv
    gap = {}
    for base in ("opt", "std3_clean", "std3_d1", "std3_d2", "std3_d3"):
        d = runs.get(f"gap/zs" if base == "opt" else f"gap@{base}/zs")
        if d and d.get("gap"):
            gap[base] = d["gap"]
    rows = []
    if gap:
        for corpus in ("synthtest", "wikitext", "pubmed", "code"):
            row = {"corpus": corpus}
            for base, g in gap.items():
                row[f"{base}_ppl"] = g[corpus]["ppl"]
                row[f"{base}_kl"] = g[corpus].get("teacher_kl")
            dep = [gap[b][corpus] for b in ("std3_d1", "std3_d2", "std3_d3") if b in gap]
            if dep and "std3_clean" in gap:
                c = gap["std3_clean"][corpus]
                row["deployed_mean_ppl"] = statistics.fmean(x["ppl"] for x in dep)
                row["deployed_minus_clean_ppl"] = row["deployed_mean_ppl"] - c["ppl"]
                row["deployed_minus_clean_kl"] = statistics.fmean(x["teacher_kl"] for x in dep) - c["teacher_kl"]
                row["deployed_draws"] = len(dep)
            if "std3_clean" in gap and "opt" in gap:
                row["clean_minus_opt_ppl"] = gap["std3_clean"][corpus]["ppl"] - gap["opt"][corpus]["ppl"]
                row["clean_minus_opt_kl"] = gap["std3_clean"][corpus]["teacher_kl"] - gap["opt"][corpus]["teacher_kl"]
            rows.append(row)
    write_csv(S / "domain_gap.csv", rows)

    # transfer.csv (Tier D2 vs Tier B on hellaswag)
    tr = []
    zs_b, zs_d, zs_c = (runs.get(x, {}).get("dev_final") for x in
                        ("hellaswag/zs", "hellaswag@std3_d1/zs", "hellaswag@std3_clean/zs"))
    for m in ("lora_fc16", "padapter64", "ln", "bitfit"):
        b = selection.get(f"hellaswag|{m}|u1000")
        dd = selection.get(f"hellaswag@std3_d1|{m}|u1000")
        if not (b or dd):
            continue
        row = {"method": m, "zs_dev_opt": zs_b, "zs_dev_deployed": zs_d, "zs_dev_clean": zs_c}
        if b:
            row.update({"tierB_lr": b["lr"], "tierB_edge": b["edge"], "tierB_dev": b["dev_selected"],
                        "tierB_dev_by_lr": " ".join(f"{x:g}:{y:.4g}" for x, y in zip(b["lrs"], b["dev"]))})
        if dd:
            row.update({"deployed_lr": dd["lr"], "deployed_edge": dd["edge"], "deployed_dev": dd["dev_selected"],
                        "deployed_dev_by_lr": " ".join(f"{x:g}:{y:.4g}" for x, y in zip(dd["lrs"], dd["dev"]))})
            if zs_d is not None:
                row["deployed_gain"] = dd["dev_selected"] - zs_d
        if b and zs_b is not None:
            row["tierB_gain"] = b["dev_selected"] - zs_b
        if b and dd and row.get("tierB_gain"):
            row["gain_retained_frac"] = row.get("deployed_gain", 0) / row["tierB_gain"]
            row["same_lr"] = b["lr"] == dd["lr"]
        if m == "lora_fc16":
            c = selection.get(f"hellaswag@std3_clean|{m}|u1000")
            if c:
                row["clean_lr"] = c["lr"]
                row["clean_dev"] = c["dev_selected"]
                if zs_c is not None:
                    row["clean_gain"] = c["dev_selected"] - zs_c
        tr.append(row)
    write_csv(S / "transfer.csv", tr)

    # timing.csv: per (dataset, method)
    tm = {}
    for r in screen:
        if r["status"] != "done" or not r["sec_per_update"]:
            continue
        k = (r["dataset"], r["base"], r["method"], r["updates"])
        tm.setdefault(k, []).append(r)
    rows = []
    for (ds, base, m, u), rs in sorted(tm.items()):
        rows.append({"dataset": ds, "base": base, "method": m, "updates": u, "runs": len(rs),
                     "params": rs[0]["params"],
                     "sec_per_update_mean": statistics.fmean(x["sec_per_update"] for x in rs),
                     "peak_mem_gb_max": max(x["peak_mem_gb"] or 0 for x in rs),
                     "wall_s_mean": statistics.fmean(x["wall_s"] for x in rs)})
    write_csv(S / "timing.csv", rows)
    print(json.dumps({"runs": len(runs), "done": sum(1 for d in runs.values() if d.get("status") == "done"),
                      "failed": [j for j, d in runs.items() if d.get("status") == "failed"],
                      "selections": len(selection)}))


if __name__ == "__main__":
    main()
