"""Print exp-002 crossbar queue lines (prio, id, est_seconds, command) for jz_worker.py.

python make_queue.py screen --est-pubmed-full S --est-hellaswag-full S ... > lines
python make_queue.py deploy --selection selection.json --est-pubmed S --est-hellaswag S
"""
import argparse
import json

JZR = "/lustre/fsn1/projects/rech/umg/ucy17uy/domain-ft-exp002-20261108"
S3 = f"{JZR}/inputs/crossbar-std3.pt"
GRIDS = {("pubmed", "full"): (1e-4, 3e-4, 1e-3), ("pubmed", "mlp"): (1e-4, 3e-4, 1e-3),
         ("hellaswag", "full"): (1e-5, 3e-5, 1e-4), ("hellaswag", "mlp"): (3e-5, 1e-4, 3e-4)}
ARMS = {"B1": ("full", ""), "B2": ("full", " --hwa-noise 0.03"), "B3": ("mlp", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("screen", "deploy"))
    ap.add_argument("--est", type=json.loads, default={}, help='{"pubmed/full": s, ...}')
    ap.add_argument("--selection", default=None)
    ap.add_argument("--prio", type=int, default=100)
    a = ap.parse_args()
    lines = []
    p = a.prio
    if a.mode == "screen":
        for target in ("hellaswag", "pubmed"):
            for arm, (method, extra) in ARMS.items():
                for lr in GRIDS[(target, method)]:
                    est = a.est[f"{target}/{method}"]
                    lines.append((p, f"screen/{target}/{arm}/{lr:g}", est,
                                  f"train.py --dataset {target} --method {method}{extra} --lr {lr:g} --base std3_clean "
                                  f"--std3 {S3} --save-state --no-retention --no-test"))
                    p += 1
    else:
        sel = json.load(open(a.selection))
        for target in ("hellaswag", "pubmed"):
            est = a.est[target]
            lines.append((p, f"deploy/{target}/B0", est, f"deploy.py --dataset {target} --arm B0 --std3 {S3}"))
            p += 1
            for arm in ("B1", "B2", "B3"):
                s = sel[f"{target}/{arm}"]
                for pol in ("rescale", "clip"):
                    lines.append((p, f"deploy/{target}/{arm}/{pol}", est,
                                  f"deploy.py --dataset {target} --arm {arm} --policy {pol} --run {s['run']} --std3 {S3}"))
                    p += 1
    for prio, jid, est, cmd in lines:
        print(f"{prio}\t{jid}\t{int(est)}\t{cmd}")


if __name__ == "__main__":
    main()
