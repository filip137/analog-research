"""exp-001 job plan, LR grids and the dev-only selection rule (shared by make_queue and aggregate)."""

from __future__ import annotations

import json
import math
from pathlib import Path

GRIDS = {
    "full": (1e-5, 3e-5, 1e-4), "mlp": (1e-5, 3e-5, 1e-4), "digital": (3e-5, 1e-4, 3e-4),
    "ln": (1e-3, 3e-3, 1e-2), "bitfit": (3e-4, 1e-3, 3e-3), "ia3": (1e-3, 3e-3, 1e-2),
    "lora_fc16": (3e-4, 1e-3, 3e-3), "padapter64": (1e-4, 3e-4, 1e-3), "lora_attn16": (3e-4, 1e-3, 3e-3),
}
TASKS = ("mnli", "rct", "hellaswag", "boolq", "casehold", "sst2")
DOMAINS = ("pubmed", "code")
GATE_SETS = ("mnli", "hellaswag", "rct", "boolq", "casehold", "sst2", "pubmed", "code")
B_METHODS = ("lora_fc16", "padapter64", "ln", "bitfit", "mlp", "digital", "ia3", "lora_attn16")
B_SETS = ("mnli", "hellaswag", "pubmed")
STEPS = [m * 10.0 ** e for e in range(-7, 0) for m in (1, 3)]


def tag(lr: float) -> str:
    return f"{lr:g}"


def step(lr: float, direction: int) -> float:
    i = min(range(len(STEPS)), key=lambda j: abs(math.log(STEPS[j] / lr)))
    return STEPS[i + direction]


def job_id(dataset, method, lr=None, base="opt", updates=1000) -> str:
    ds = dataset if base == "opt" else f"{dataset}@{base}"
    name = method if method == "zs" else f"{method}__lr{tag(lr)}"
    if updates != 1000:
        name += f"__u{updates}"
    return f"{ds}/{name}"


def job(prio, dataset, method, lr=None, base="opt", updates=1000, extra="") -> tuple:
    args = f"--dataset {dataset} --method {method}"
    if lr is not None:
        args += f" --lr {tag(lr)}"
    if base != "opt":
        args += f" --base {base}"
    if updates != 1000:
        args += f" --updates {updates}"
    if extra:
        args += " " + extra
    return (prio, job_id(dataset, method, lr, base, updates), args)


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


def select(runs: dict, dataset_key: str, method: str, updates: int = 1000, lm: bool | None = None):
    """Dev-only selection over all done runs of (dataset_key, method, updates)."""
    cands = []
    for jid, d in runs.items():
        ds, name = jid.split("/")
        if ds != dataset_key or d.get("status") != "done" or "dev_final" not in d:
            continue
        parts = name.split("__")
        if parts[0] != method or len(parts) < 2:
            continue
        u = int(parts[2][1:]) if len(parts) > 2 else 1000
        if u != updates:
            continue
        cands.append((float(parts[1][2:]), d))
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


def plan(runs: dict) -> list[tuple]:
    """Full ordered job list given current results (dependent jobs appear once their inputs are done)."""
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
    # P2 Tier B: by method, interleaving datasets
    k = 2000
    for m in B_METHODS:
        for i in range(3):
            for d in B_SETS:
                J.append(job(k, d, m, GRIDS[m][i]))
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
            for s in sizes:
                J.append(job(k, d, f"{fam}{s}", sel["lr"]))
                k += 1
                if s == big:
                    J.append(job(k, d, f"{fam}{s}", step(sel["lr"], -1)))
                    k += 1
    # P5 Tier D2 (deployed replica, draw 1)
    k = 5000
    J.append(job(k, "hellaswag", "zs", base="std3_d1"))
    J.append(job(k, "hellaswag", "zs", base="std3_clean"))
    for m in ("lora_fc16", "padapter64", "ln", "bitfit"):
        for lr in GRIDS[m]:
            k += 1
            J.append(job(k, "hellaswag", m, lr, base="std3_d1"))
    sel = select(runs, "hellaswag@std3_d1", "lora_fc16")
    if sel is not None and sel["n"] >= 3:
        J.append(job(5100, "hellaswag", "lora_fc16", sel["lr"], base="std3_clean"))
    # P6 extensions: edge LRs first
    k = 6000
    for jid, d in sorted(runs.items()):
        pass
    keys = {}
    for jid in runs:
        ds, name = jid.split("/")
        parts = name.split("__")
        if parts[0] == "zs" or ds == "gap":
            continue
        u = int(parts[2][1:]) if len(parts) > 2 else 1000
        keys[(ds, parts[0], u)] = 1
    for (ds, m, u) in sorted(keys):
        g = grid_of(m)
        if g is None or u != 1000:
            continue
        sel = select(runs, ds, m, u)
        if sel is None or not sel["edge"]:
            continue
        # extend only if all grid points are done and no extension exists yet
        if not all(any(abs(lr - x) / x < 1e-6 for lr in sel["lrs"]) for x in g) or sel["n"] > 3:
            continue
        base = "opt"
        dset = ds
        if "@" in ds:
            dset, base = ds.split("@")
        new = step(sel["lr"], -1 if sel["edge"] == "low" else 1)
        J.append(job(k, dset, m, new, base=base))
        k += 1
    k = 6100
    for m in B_METHODS:
        for lr in GRIDS[m]:
            J.append(job(k, "rct", m, lr))
            k += 1
    k = 6200
    for d in ("boolq", "casehold"):
        for m in ("lora_fc16", "padapter64"):
            for lr in GRIDS[m]:
                J.append(job(k, d, m, lr))
                k += 1
    for i, d in enumerate(("mnli", "pubmed")):
        sel = select(runs, d, "lora_fc16")
        if sel is not None and sel["n"] >= 3:
            J.append(job(6300 + i, d, "lora_fc16", sel["lr"], updates=2000))
    # de-duplicate keeping the first (highest priority) occurrence
    seen, out = set(), []
    for j in J:
        if j[1] not in seen:
            seen.add(j[1])
            out.append(j)
    return out
