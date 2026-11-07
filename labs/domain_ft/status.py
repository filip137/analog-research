"""Write status.json from synced queues/claims/runs: python labs/domain_ft/status.py"""

from __future__ import annotations

import json
import time
from pathlib import Path

ART = Path("/home/filip/analog-research/on-chip-learning/artifacts/domain-finetuning/prelim-20261108")
TIERS = [(1000, 1499, "A_gate"), (1500, 1999, "D1_gap"), (2000, 2999, "B_screens"), (4000, 4999, "C_capacity"),
         (5000, 5999, "D2_transfer"), (6000, 6099, "ext_edge"), (6100, 6199, "ext_rct"),
         (6200, 6299, "ext_boolq_casehold"), (6300, 6399, "ext_u2000")]


def tier(p):
    for a, b, n in TIERS:
        if a <= p <= b:
            return n
    return "other"


def main():
    st = {"last_check_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "hosts": {}, "tiers": {}}
    remaining_cost = {}
    for h in ("riri", "trex"):
        q = ART / "queue" / h
        lines = (q / "queue.txt").read_text().splitlines() if (q / "queue.txt").exists() else []
        hs = {"queued": 0, "running": [], "done": 0, "failed": 0, "pending": 0}
        for ln in lines:
            parts = ln.split("\t")
            if len(parts) != 3:
                continue
            p, jid = float(parts[0]), parts[1]
            key = jid.replace("/", "~")
            t = st["tiers"].setdefault(tier(p), {"total": 0, "done": 0, "failed": 0, "running": 0, "pending": 0,
                                                  "killed_deadline": 0})
            t["total"] += 1
            hs["queued"] += 1
            fin = ART / "runs" / jid / "final.json"
            status = None
            if fin.exists():
                try:
                    status = json.loads(fin.read_text()).get("status")
                except Exception:
                    status = None
            if (q / "claims" / key).exists():
                if (q / "claims" / key / "rc").exists():
                    s = status or "failed"
                    s = s if s in ("done", "failed", "killed_deadline") else "failed"
                    t[s] += 1
                    hs["done" if s == "done" else "failed"] += 1
                else:
                    t["running"] += 1
                    hs["running"].append(jid)
            else:
                t["pending"] += 1
                hs["pending"] += 1
        st["hosts"][h] = hs
    walls = []
    for f in (ART / "runs").glob("*/*/final.json"):
        try:
            d = json.loads(f.read_text())
            if d.get("status") == "done" and d.get("updates_done"):
                walls.append(d["wall_s"])
        except Exception:
            pass
    mean = sum(walls) / len(walls) if walls else 150.0
    pend = max(st["hosts"][h]["pending"] for h in st["hosts"])
    st["mean_train_run_wall_s"] = mean
    st["eta_queue_drain_min_approx"] = round(pend * mean / 60, 1)
    (ART / "status.json").write_text(json.dumps(st, indent=2))
    print(json.dumps({k: v for k, v in st.items() if k != "hosts"}, indent=1))
    print({h: {k: (v if k != "running" else v) for k, v in s.items()} for h, s in st["hosts"].items()})


if __name__ == "__main__":
    main()
