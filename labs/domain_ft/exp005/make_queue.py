"""Print exp-005 queue lines (prio, id, est_seconds, needs, command) for exp005/jz_worker.py.

Arms (digital LoRA lora_fc16 on the std3 crossbar base, analog base frozen):
  L0  off-chip, clean training on std3_clean, deployed on every (c_p, draw)
  L1  off-chip HWA: clean base re-programmed every update with a fresh deployment-law draw at c_p (--hwa-cp),
      deployed at that c_p on every draw
  L2  on-chip: trained on the std3 base programmed at (c_p, draw) (--base std3_cp<c>_d<draw>), evaluated there only

python make_queue.py train  --arms L0 L1 L2 --targets pubmed hellaswag --lrs 1e-4 3e-4 1e-3 --cps 1 10 --draws 271002
python make_queue.py deploy --arms L0 L1 L2 ... (same selectors; each deploy line needs its run's final.json done)
"""
import argparse

JZR = "/lustre/fsn1/projects/rech/umg/ucy17uy/domain-ft-exp005-20261111"
S3 = f"{JZR}/inputs/crossbar-std3.pt"
METHOD = "lora_fc16"
EST_TRAIN = {"pubmed": 900, "hellaswag": 1500}       # s per 1,000-update run incl. 4 dev evals (V100), from canary
EST_EVAL = {"pubmed": 60, "hellaswag": 120}          # s per deployed (c_p, draw) evaluation incl. WikiText-2


def g(x: float) -> str:
    return f"{x:g}"


def run_path(arm, t, lr, cp=None, draw=None):
    if arm == "L0":
        return f"runs/{t}@std3_clean/{METHOD}__lr{g(lr)}"
    if arm == "L1":
        return f"runs/{t}@std3_clean/{METHOD}_hwacp{g(cp)}__lr{g(lr)}"
    return f"runs/{t}@std3_cp{g(cp)}_d{draw}/{METHOD}__lr{g(lr)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "deploy"))
    ap.add_argument("--arms", nargs="+", default=["L0", "L1", "L2"])
    ap.add_argument("--targets", nargs="+", default=["pubmed", "hellaswag"])
    ap.add_argument("--lrs", type=float, nargs="+", required=True)
    ap.add_argument("--cps", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--draws", type=int, nargs="+", default=[271002])
    ap.add_argument("--l0-cps", type=float, nargs="+", default=None, help="deploy L0 at these c_p (default --cps)")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--prio", type=int, default=100)
    a = ap.parse_args()
    p = a.prio
    utag = "" if a.updates == 1000 else f"__u{a.updates}"
    ufl = "" if a.updates == 1000 else f" --updates {a.updates}"
    frac = a.updates / 1000
    common = f"--method {METHOD} --std3 {S3}"
    lines = []
    for t in a.targets:
        for lr in a.lrs:
            for arm in a.arms:
                if a.mode == "train":
                    tr = f"train.py --dataset {t} {common} --lr {g(lr)} --no-test --no-retention{ufl}"
                    est = int(EST_TRAIN[t] * max(frac, 0.15))
                    if arm == "L0":
                        lines.append((p, f"train/{t}/L0/{g(lr)}{utag}", est, "-", f"{tr} --base std3_clean"))
                    for cp in a.cps:
                        if arm == "L1":
                            lines.append((p, f"train/{t}/L1/cp{g(cp)}/{g(lr)}{utag}", est, "-",
                                          f"{tr} --base std3_clean --hwa-cp {g(cp)}"))
                        if arm == "L2":
                            for d in a.draws:
                                lines.append((p, f"train/{t}/L2/cp{g(cp)}_d{d}/{g(lr)}{utag}", est, "-",
                                              f"{tr} --base std3_cp{g(cp)}_d{d}"))
                else:
                    ev = EST_EVAL[t]
                    draws = " ".join(map(str, a.draws))
                    if arm == "L0":
                        cps = a.l0_cps or a.cps
                        rp = run_path("L0", t, lr) + utag
                        lines.append((p + 1000, f"deploy/{t}/L0/{g(lr)}{utag}", ev * (1 + len(cps) * len(a.draws)), rp,
                                      f"deploy.py --dataset {t} {common} --arm L0__lr{g(lr)}{utag} --run {rp} "
                                      f"--cps 0 {' '.join(g(c) for c in cps)} --draws {draws}"))
                    for cp in a.cps:
                        if arm == "L1":
                            rp = run_path("L1", t, lr, cp) + utag
                            lines.append((p + 1000, f"deploy/{t}/L1/cp{g(cp)}/{g(lr)}{utag}", ev * (1 + len(a.draws)),
                                          rp, f"deploy.py --dataset {t} {common} --arm L1cp{g(cp)}__lr{g(lr)}{utag} "
                                              f"--run {rp} --cps 0 {g(cp)} --draws {draws}"))
                        if arm == "L2":
                            for d in a.draws:
                                rp = run_path("L2", t, lr, cp, d) + utag
                                lines.append((p + 1000, f"deploy/{t}/L2/cp{g(cp)}_d{d}/{g(lr)}{utag}", ev, rp,
                                              f"deploy.py --dataset {t} {common} --arm L2__lr{g(lr)}{utag} --run {rp} "
                                              f"--cps {g(cp)} --draws {d}"))
                p += 1
    for prio, jid, est, needs, cmd in lines:
        print(f"{prio}\t{jid}\t{int(est)}\t{needs}\t{cmd}")


if __name__ == "__main__":
    main()
