"""Queue worker for one GPU slot: python worker.py <run_root> <slot>

Reads <root>/queue/queue.txt (lines "<prio>\t<id>\t<train.py args>"), claims the
lowest-priority-number unclaimed job by mkdir <root>/queue/claims/<id-escaped>, runs it, and repeats. Host-clock limits
come from <root>/queue/limits.json {"launch_until": epoch, "kill_at": epoch}
(epochs on this host's clock). Stops when <root>/queue/STOP exists, or when the
queue is drained and <root>/queue/DRAIN exists.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PY = "/home/filip/miniconda3/envs/py312/bin/python"


def main(root: Path, slot: str):
    q = root / "queue"
    claims = q / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    log = open(root / "logs" / f"worker-{slot}.log", "a")

    def say(msg):
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{slot}] {msg}\n")
        log.flush()

    say("start")
    while True:
        if (q / "STOP").exists():
            say("STOP")
            return
        limits = json.loads((q / "limits.json").read_text())
        if time.time() > limits["launch_until"]:
            say("launch window closed")
            return
        job = None
        lines = (q / "queue.txt").read_text().splitlines() if (q / "queue.txt").exists() else []
        entries = []
        for n, line in enumerate(lines):
            parts = line.split("\t")
            if len(parts) != 3 or line.startswith("#"):
                continue
            entries.append((float(parts[0]), n, parts[1], parts[2]))
        for _, _, jid, args in sorted(entries):
            key = jid.replace("/", "~")
            if (claims / key).exists():
                continue
            try:
                os.mkdir(claims / key)
            except FileExistsError:
                continue
            (claims / key / "slot").write_text(f"{slot} {time.time()}\n")
            job = (jid, args, key)
            break
        if job is None:
            if (q / "DRAIN").exists():
                say("drained")
                return
            time.sleep(20)
            continue
        jid, args, key = job
        say(f"run {jid}")
        t0 = time.time()
        out = open(root / "logs" / f"{key}.log", "a")
        cmd = [PY, str(root / "code/labs/domain_ft/train.py"), "--root", str(root),
               "--deadline", str(limits["kill_at"])] + shlex.split(args)
        rc = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=str(root / "code"))
        (claims / key / "rc").write_text(f"{rc} {time.time() - t0:.1f}\n")
        say(f"end {jid} rc={rc} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser(), sys.argv[2])
