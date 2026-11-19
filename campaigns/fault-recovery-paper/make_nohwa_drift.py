"""Lifecycles of fault-recovery-paper series 003 exp-002 part (b), crossbar: HWA vs no-HWA under CMO x4 drift.

Owning note: on-chip-learning campaigns/fault-recovery-paper/series/003-mapping-deployment/experiments/
exp-002-nohwa-drn-and-drift.md (issue #33, user, 19 November 2026). Pilot note:
campaigns/pilots/20261119-nohwa-drift.md. Follow-up (b'): ``-m1`` cells at CMO x1 from the exp-001 x1 lifecycles.

Each cell copies the CMO drift-recovery exp-001 x4 lifecycle of its depth
(campaigns/pilots/20261118-cmo-drift-recovery/lifecycles/full/cmo-drift-recovery-20261118-<model>-m4.json:
law x4, clip 5-150 uS, draw seed 391103, ten ages, realization 271001, 0 % faults, naive P&V + 5 sigma
pair rewrite, dev 64, pool and targets, 2,400 updates) and changes only: the arms (A0, A0-twin, CAL
1e-3, as declared there), the mapping (base: the standard pair, no mapping key, as drift-regimes /
#66 and the drift token budget; two-sided: unchanged) and, for no-HWA, the single HWA arm
``pretrained`` with the pinned pretrained-teacher source of series 003 exp-001.

usage: python campaigns/fault-recovery-paper/make_nohwa_drift.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP001 = HERE.parent / "pilots/20261118-cmo-drift-recovery/lifecycles/full"
OUT = HERE / "lifecycles/nohwa-drift-20261119"
PRETRAINED = {"xb6": "3442d7d2b4137c4126b2501a54d27a594b1ccbddc171b39d7329e2e4fd73a1b4",
              "xb12": "480e1323aaed8a12036121d887f1aa38c3316b7cb39354ca14353cfc76c3abc8"}
ARMS = ("A0", "A0-twin", "CAL")
CELLS = (("hwa", "base"), ("nohwa", "base"), ("nohwa", "two"))
QUESTION = {
    "decision": "Source and mapping of the #33 rerun (series003 exp-002 part b); evaluation plus zero-write calibration.",
    "evidence_class": "exploratory_model_based",
    "primary_metric": "teacher_kl",
    "text": "Under CMO x4 drift, how do HWA and no-HWA crossbar sources compare (24-h mean KL, drift damage)?",
}


def cell(model, source, mapping, multiplier=4):
    doc = json.loads((EXP001 / f"cmo-drift-recovery-20261118-{model}-m{multiplier}.json").read_text())
    ft = doc["fault_tuning"]
    ft["arms"] = [a for a in ft["arms"] if a["label"] in ARMS]
    if mapping == "base":
        del ft["drift"]["mapping"]
    if source == "nohwa":
        doc["hwa"]["arms"] = [{"id": "pretrained", "method": "none"}]
        ft["source_sha256"] = PRETRAINED[model]
    doc["lifecycle_id"] = f"nohwa-drift-20261119-{model}-{source}-{mapping}" + ("" if multiplier == 4 else f"-m{multiplier}")
    doc["question"] = dict(QUESTION)
    return doc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for model in ("xb6", "xb12"):
        for source, mapping in CELLS:
            doc = cell(model, source, mapping)
            path = OUT / f"{doc['lifecycle_id']}.json"
            path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            print(path)
    # Follow-up (b'), user 20 November: no-HWA two-sided at CMO x1 (exp-001 x1 lifecycles; HWA x1
    # controls reused from exp-001 jobs 395739 / 395727).
    for model in ("xb6", "xb12"):
        doc = cell(model, "nohwa", "two", multiplier=1)
        path = OUT / f"{doc['lifecycle_id']}.json"
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        print(path)
    # Replay control (canary): the reused HWA two-sided cell, rerun on this pipeline with these arms;
    # its deployment hashes and A0 / A0-twin grids must equal CMO exp-001's (job 387978).
    doc = cell("xb6", "hwa", "two")
    doc["lifecycle_id"] += "-replay"
    path = OUT / f"{doc['lifecycle_id']}.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
