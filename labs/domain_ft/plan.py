"""exp-001 job plan, LR grids and the dev-only selection rule (shared by make_queue and aggregate)."""

from __future__ import annotations

import json
import math
from pathlib import Path

GRIDS = {
    "full": (1e-5, 3e-5, 1e-4), "mlp": (1e-5, 3e-5, 1e-4), "digital": (3e-5, 1e-4, 3e-4),
    "ln": (1e-3, 3e-3, 1e-2), "bitfit": (3e-4, 1e-3, 3e-3), "ia3": (1e-3, 3e-3, 1e-2),
    "lora_fc16": (3e-4, 1e-3, 3e-3), "padapter64": (1e-4, 3e-4, 1e-3), "lora_attn16": (3e-4, 1e-3, 3e-3),
    "calib": (1e-3, 3e-3, 1e-2), "calib_out": (1e-3, 3e-3, 1e-2),
}
TASKS = ("mnli", "rct", "hellaswag", "boolq", "casehold", "sst2")
DOMAINS = ("pubmed", "code")
GATE_SETS = ("mnli", "hellaswag", "rct", "boolq", "casehold", "sst2", "pubmed", "code")
B_METHODS = ("lora_fc16", "padapter64", "ln", "bitfit", "mlp", "digital", "ia3", "lora_attn16", "calib", "calib_out")
D2_METHODS = ("lora_fc16", "padapter64", "ln", "bitfit", "calib", "calib_out")
B_SETS = ("mnli", "hellaswag", "pubmed")
STEPS = [m * 10.0 ** e for e in range(-7, 0) for m in (1, 3)]


def tag(lr: float) -> str:
    return f"{lr:g}"


def step(lr: float, direction: int) -> float:
    i = min(range(len(STEPS)), key=lambda j: abs(math.log(STEPS[j] / lr)))
    return STEPS[i + direction]


def job_id(dataset, method, lr=None, base="opt", updates=1000, seed=1) -> str:
    ds = dataset if base == "opt" else f"{dataset}@{base}"
    name = method if method == "zs" else f"{method}__lr{tag(lr)}"
    if updates != 1000:
        name += f"__u{updates}"
    if seed != 1:
        name += f"__s{seed}"
    return f"{ds}/{name}"


def parse(name: str) -> tuple:
    """(method, lr, updates, seed) from a run directory name."""
    parts = name.split("__")
    lr = float(parts[1][2:]) if len(parts) > 1 and parts[1].startswith("lr") else None
    u, sd = 1000, 1
    for t in parts[2:]:
        if t.startswith("u"):
            u = int(t[1:])
        elif t.startswith("s"):
            sd = int(t[1:])
    return parts[0], lr, u, sd


def job(prio, dataset, method, lr=None, base="opt", updates=1000, extra="", seed=1) -> tuple:
    args = f"--dataset {dataset} --method {method}"
    if lr is not None:
        args += f" --lr {tag(lr)}"
    if base != "opt":
        args += f" --base {base}"
    if updates != 1000:
        args += f" --updates {updates}"
    if seed != 1:
        args += f" --seed {seed}"
    if extra:
        args += " " + extra
    return (prio, job_id(dataset, method, lr, base, updates, seed), args)


def read_runs(runs: Path) -> dict:
    """{job_id: final.json dict (+ test/retention when present)}."""
    out = {}
    for f in runs.glob("*/*/final.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        jid = f"{f.parent.parent.name}/{f.parent.name}"
        for extra in ("test", "retention", "gap"):
            p = f.parent / f"{extra}.json"
            if p.exists():
                try:
                    d[extra] = json.loads(p.read_text())
                except Exception:
                    pass
        out[jid] = d
    return out


def select(runs: dict, dataset_key: str, method: str, updates: int = 1000, lm: bool | None = None, seed: int = 1):
    """Dev-only selection over all done runs of (dataset_key, method, updates, seed)."""
    cands = []
    for jid, d in runs.items():
        ds, name = jid.split("/")
        if ds != dataset_key or d.get("status") != "done" or "dev_final" not in d:
            continue
        m, lr, u, sd = parse(name)
        if m != method or lr is None or u != updates or sd != seed:
            continue
        cands.append((lr, d))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    is_lm = (dataset_key.split("@")[0] in DOMAINS) if lm is None else lm
    vals = [d["dev_final"] for _, d in cands]
    if is_lm:
        best = min(vals)
        ok = [i for i, v in enumerate(vals) if v <= best * 1.005]
    else:
        best = max(vals)
        ok = [i for i, v in enumerate(vals) if v >= best - 0.2]
    i = ok[0]
    lrs = [lr for lr, _ in cands]
    return {"lr": lrs[i], "lrs": lrs, "dev": vals, "dev_selected": vals[i],
            "edge": "low" if (i == 0 and len(lrs) > 1) else ("high" if (i == len(lrs) - 1 and len(lrs) > 1) else ""),
            "n": len(lrs), "run": cands[i][1]}


def grid_of(method: str):
    if method in GRIDS:
        return GRIDS[method]
    return None


def plan(runs: dict, post_drain: bool = False) -> list[tuple]:
    """Full ordered job list given current results (dependent jobs appear once their inputs are done).

    Main's decisions 2026-11-07: pubmed rebuilt from PubMedQA (natural case), pubmed Tier B at the end of
    Tier B; calib/calib_out added to Tier B and D2; post-drain list a-c (seeds, H-002 budgets, D2 on mnli).
    """
    J = []
    # P1 Tier A
    for d in GATE_SETS:
        J.append(job(1000, d, "zs"))
    J.append(job(1001, "probes", "zs"))
    k = 1100
    for d in GATE_SETS:
        for lr in GRIDS["full"]:
            J.append(job(k, d, "full", lr))
            k += 1
    # P3 Tier D1 (eval only, cheap; inserted right after Tier A)
    for base in ("opt", "std3_clean", "std3_d1", "std3_d2", "std3_d3"):
        J.append(job(1500, "gap", "zs", base=base))
    # P2 Tier B: by method, interleaving datasets; pubmed at the end of Tier B (rebuild time)
    k = 2000
    for m in B_METHODS:
        for i in range(3):
            for d in B_SETS:
                prio = k if d != "pubmed" else 2800 + k - 2000
                J.append(job(prio, d, m, GRIDS[m][i]))
                k += 1
    # P4 Tier C stage 1 (code screens)
    k = 4000
    for m in ("lora_fc16", "padapter64", "mlp"):
        for lr in GRIDS[m]:
            J.append(job(k, "code", m, lr))
            k += 1
    # P4 Tier C stage 2 (capacity at the sibling's selected LR)
    k = 4100
    for d in DOMAINS:
        for sib, sizes, big in (("lora_fc16", (4, 64, 256), 256), ("padapter64", (16, 256, 1024), 1024)):
            sel = select(runs, d, sib)
            if sel is None or sel["n"] < 3:
                continue
            fam = sib.rstrip("0123456789")
            for sz in sizes:
                J.append(job(k, d, f"{fam}{sz}", sel["lr"]))
                k += 1
                if sz == big:
                    J.append(job(k, d, f"{fam}{sz}", step(sel["lr"], -1)))
                    k += 1
    # P5 Tier D2 (deployed replica, draw 1)
    k = 5000
    J.append(job(k, "hellaswag", "zs", base="std3_d1"))
    J.append(job(k, "hellaswag", "zs", base="std3_clean"))
    for m in D2_METHODS:
        for lr in GRIDS[m]:
            k += 1
            J.append(job(k, "hellaswag", m, lr, base="std3_d1"))
    sel = select(runs, "hellaswag@std3_d1", "lora_fc16")
    if sel is not None and sel["n"] >= 3:
        J.append(job(5100, "hellaswag", "lora_fc16", sel["lr"], base="std3_clean"))
    # P6 extensions: edge LRs first
    k = 6000
    keys = set()
    for jid in runs:
        ds, name = jid.split("/")
        m, lr, u, sd = parse(name)
        if m == "zs" or ds.startswith("gap") or lr is None or u != 1000 or sd != 1 or ds.endswith("_lc"):
            continue
        keys.add((ds, m))
    for (ds, m) in sorted(keys):
        g = grid_of(m)
        if g is None:
            continue
        sel = select(runs, ds, m)
        if sel is None or not sel["edge"]:
            continue
        if not all(any(abs(lr - x) / x < 1e-6 for lr in sel["lrs"]) for x in g) or sel["n"] > 3:
            continue
        base, dset = "opt", ds
        if "@" in ds:
            dset, base = ds.split("@")
        J.append(job(k, dset, m, step(sel["lr"], -1 if sel["edge"] == "low" else 1), base=base))
        k += 1
    k = 6100
    for m in B_METHODS[:8]:
        for lr in GRIDS[m]:
            J.append(job(k, "rct", m, lr))
            k += 1
    k = 6200
    for d in ("boolq", "casehold"):
        for m in ("lora_fc16", "padapter64"):
            for lr in GRIDS[m]:
                J.append(job(k, d, m, lr))
                k += 1
    sel = select(runs, "mnli", "lora_fc16")
    if sel is not None and sel["n"] >= 3:
        J.append(job(6300, "mnli", "lora_fc16", sel["lr"], updates=2000))
    if post_drain:
        # (a) seed replicates at the selected LR: every Tier B (dataset, method) + selected full per Tier B dataset
        k = 7000
        for d in B_SETS:
            for m in B_METHODS + ("full",):
                sel = select(runs, d, m)
                if sel is not None and sel["n"] >= 3:
                    J.append(job(k, d, m, sel["lr"], seed=2))
                    k += 1
        # (b) H-002 budget sensitivity on code and pubmed
        k = 7100
        for U in (2000, 4000):
            for d in DOMAINS:
                for m in ("full", "mlp", "lora_fc16", "lora_fc256", "padapter64"):
                    sel = select(runs, d, m)
                    if sel is not None and (sel["n"] >= 3 or m == "lora_fc256" and sel["n"] >= 2):
                        J.append(job(k, d, m, sel["lr"], updates=U))
                        k += 1
        # (c) D2 on mnli (deployed replica, draw 1)
        k = 7200
        J.append(job(k, "mnli", "zs", base="std3_d1"))
        for m in D2_METHODS:
            for lr in GRIDS[m]:
                k += 1
                J.append(job(k, "mnli", m, lr, base="std3_d1"))
    # de-duplicate keeping the first (highest priority) occurrence
    seen, out = set(), []
    for j in J:
        if j[1] not in seen:
            seen.add(j[1])
            out.append(j)
    return out
