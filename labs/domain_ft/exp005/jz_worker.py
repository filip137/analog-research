"""exp-005 Jean Zay queue worker (one per Slurm job, one GPU): python jz_worker.py <root> <job_seconds>

As exp002/jz_worker.py, with a dependency column. <root>/queue/queue.txt lines:
"<prio>\t<id>\t<est_seconds>\t<needs>\t<script.py> <args>" (script relative to code/labs/domain_ft). <needs> is "-" or a
run dir relative to <root>; the line is claimable only once <root>/<needs>/final.json has status "done". A job is
claimed by mkdir <root>/queue/claims/<id with / -> ~> (atomic), only if its estimate fits the remaining wall time.
While a claimable-later line waits on a dependency that some other worker is still running, the worker sleeps and
retries; it exits when nothing is claimable and nothing it could wait for is running, or when <root>/queue/STOP exists.
"""
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

MARGIN = 240
POLL = 60


def status(root: Path, run: str):
    f = root / run / "final.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text()).get("status")
    except Exception:
        return None


def main(root: Path, job_seconds: float):
    t_end = time.time() + job_seconds - MARGIN
    q = root / "queue"
    claims = q / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    jid_env = os.environ.get("SLURM_JOB_ID", "nojob")
    log = open(root / "logs" / f"worker-{jid_env}.log", "a")

    def say(msg):
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{jid_env} {socket.gethostname()}] {msg}\n")
        log.flush()

    say(f"start, end at {time.strftime('%H:%M:%S', time.localtime(t_end))}")
    while not (q / "STOP").exists():
        entries = []
        for line in (q / "queue.txt").read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 5 and not line.startswith("#"):
                entries.append((float(parts[0]), parts[1], float(parts[2]), parts[3], parts[4]))
        job, waiting = None, False
        for prio, jid, est, needs, args in sorted(entries):
            key = jid.replace("/", "~")
            if (claims / key).exists() or est > t_end - time.time():
                continue
            if needs != "-":
                st = status(root, needs)
                if st != "done":
                    if st in (None, "running"):
                        waiting = True  # its training is queued or running somewhere
                    continue
            try:
                os.mkdir(claims / key)
            except FileExistsError:
                continue
            (claims / key / "slot").write_text(f"{jid_env} {socket.gethostname()} {time.time():.0f}\n")
            job = (jid, args, key)
            break
        if job is None:
            if waiting and t_end - time.time() > POLL + 300:
                time.sleep(POLL)
                continue
            say("nothing claimable fits; exit")
            return
        jid, args, key = job
        say(f"run {jid}")
        t0 = time.time()
        argv = shlex.split(args)
        cmd = [sys.executable, str(root / "code/labs/domain_ft" / argv[0]), "--root", str(root),
               "--deadline", f"{t_end:.0f}"] + argv[1:]
        with open(root / "logs" / f"{key}.log", "a") as out:
            out.write(f"# {jid_env} {socket.gethostname()} {' '.join(cmd)}\n")
            out.flush()
            rc = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=str(root / "code"))
        (claims / key / "rc").write_text(f"{rc} {time.time() - t0:.1f}\n")
        say(f"end {jid} rc={rc} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]))
