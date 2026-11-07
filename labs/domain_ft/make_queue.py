"""Local orchestration: sync results from the hosts, extend and distribute the job queue.

python labs/domain_ft/make_queue.py [--sync-only] [--dry-run] [--hosts riri trex]
Each host has its own queue file; new jobs go to the host with the least pending
estimated work, and jobs already queued on a host never move (no double runs).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan as P  # noqa: E402

ART = Path("/home/filip/analog-research/on-chip-learning/artifacts/domain-finetuning/prelim-20261108")
R = "domain-ft-20261108"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def sh(host, cmd, inp=None, timeout=120):
    return subprocess.run(SSH + [host, cmd], input=inp, capture_output=True, text=True, timeout=timeout)


def sync(host):
    (ART / "runs").mkdir(parents=True, exist_ok=True)
    (ART / "logs" / host).mkdir(parents=True, exist_ok=True)
    (ART / "queue" / host).mkdir(parents=True, exist_ok=True)
    e = " ".join(SSH)
    subprocess.run(["rsync", "-a", "-e", e, f"{host}:{R}/runs/", str(ART / "runs") + "/"], timeout=600, check=False)
    subprocess.run(["rsync", "-a", "-e", e, f"{host}:{R}/logs/", str(ART / "logs" / host) + "/"], timeout=600,
                   check=False)
    subprocess.run(["rsync", "-a", "-e", e, f"{host}:{R}/queue/", str(ART / "queue" / host) + "/"], timeout=600,
                   check=False)


def cost(jid: str) -> float:
    ds, name = jid.split("/")
    m = name.split("__")[0]
    if m == "zs":
        return 0.5
    c = 2.5 if m in ("full",) else 1.6
    if ds.startswith("casehold") or ds.startswith("hellaswag"):
        c *= 1.5
    if "__u2000" in name:
        c *= 2
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", nargs="+", default=["riri", "trex"])
    ap.add_argument("--sync-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-prio", type=float, default=1e9)
    a = ap.parse_args()
    for h in a.hosts:
        sync(h)
    if a.sync_only:
        return
    runs = P.read_runs(ART / "runs")
    queued, pending = {}, {}
    for h in a.hosts:
        qf = ART / "queue" / h / "queue.txt"
        claims = {p.name.replace("~", "/") for p in (ART / "queue" / h / "claims").glob("*")} \
            if (ART / "queue" / h / "claims").exists() else set()
        lines = qf.read_text().splitlines() if qf.exists() else []
        ids = [ln.split("\t")[1] for ln in lines if ln.count("\t") == 2]
        for i in ids:
            queued[i] = h
        pending[h] = sum(cost(i) for i in ids if i not in claims)
    new = [j for j in P.plan(runs) if j[1] not in queued and j[0] <= a.max_prio]
    add = {h: [] for h in a.hosts}
    for prio, jid, args in new:
        h = min(a.hosts, key=lambda x: pending[x])
        add[h].append(f"{prio}\t{jid}\t{args}")
        pending[h] += cost(jid)
    for h in a.hosts:
        print(h, "add", len(add[h]), "pending_cost", round(pending[h], 1))
        for line in add[h][:200]:
            print("   ", line)
        if add[h] and not a.dry_run:
            r = sh(h, f"cat >> ~/{R}/queue/queue.txt", inp="\n".join(add[h]) + "\n")
            if r.returncode:
                print("ERROR appending", h, r.stderr)
    if not a.dry_run:
        for h in a.hosts:
            sync(h)


if __name__ == "__main__":
    main()
