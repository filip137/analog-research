#!/usr/bin/env python3
"""Write lifecycle opt125m-cmo-l6-exp1e once Stage 1e-a has frozen the digital learning rate.

usage: python campaigns/opt-cmo-corruption/launch/make_exp1e_lifecycle.py <frozen digital learning rate> [output]

Arms (exp-005): pretrained; the reused analog-scope sources standard_hwa and cdt_random_3pct
(opt125m-cmo-l6-exp1c-decay) and standard_hwa_9pct (opt125m-cmo-l6-exp1c-decay-9pct); and the
whole-model standard_hwa_model_<tag> (exactly the Stage 1e-a arm, so its source is reused),
standard_hwa_9pct_model_<tag> and cdt_random_3pct_model_<tag>. Every other section is the
Stage 1e-a lifecycle's, so all reused sources pass provenance. Run `python -m ebl lifecycle check` after.
"""

import copy
import json
from pathlib import Path
import sys

LIFECYCLES = Path(__file__).resolve().parents[1] / "lifecycles"
NOTE = "campaigns/opt-cmo-corruption/series/001-hwa-robustness/experiments/exp-005-whole-model-hwa.md"


def tag(rate: float) -> str:
    """1e-05 -> dlr1em5, 3e-05 -> dlr3em5 (arm ids allow only [a-z0-9_])."""

    mantissa, exponent = f"{rate:.0e}".split("e")
    if int(exponent) >= 0:
        raise ValueError(f"Expected a learning rate below 1. Provided value: {rate}.")
    return f"dlr{mantissa}em{-int(exponent)}"


def main() -> int:
    rate = float(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else LIFECYCLES / "opt125m-cmo-l6-exp1e.json"
    lrdev = json.loads((LIFECYCLES / "opt125m-cmo-l6-exp1e-lrdev.json").read_text())
    decay = json.loads((LIFECYCLES / "opt125m-cmo-l6-exp1c-decay.json").read_text())["hwa"]["arms"]
    decay9 = json.loads((LIFECYCLES / "opt125m-cmo-l6-exp1c-decay-9pct.json").read_text())["hwa"]["arms"]
    b1 = json.loads((LIFECYCLES / "opt125m-cmo-l6-exp1b.json").read_text())["hwa"]["arms"]
    analog = {arm["id"]: arm for arm in decay + decay9}
    frozen = [arm for arm in lrdev["hwa"]["arms"] if arm["digital_learning_rate"] == rate]
    if len(frozen) != 1:
        raise SystemExit(f"Expected exactly one Stage 1e-a arm at {rate}; add the extension arm to the lrdev lifecycle first.")
    suffix = tag(rate)
    if frozen[0]["id"] != f"standard_hwa_model_{suffix}":
        raise SystemExit(f"Expected the Stage 1e-a arm id standard_hwa_model_{suffix}; found {frozen[0]['id']}.")
    arms = [next(a for a in b1 if a["id"] == "pretrained"), analog["standard_hwa"], analog["cdt_random_3pct"],
            analog["standard_hwa_9pct"], frozen[0]]
    for base in ("standard_hwa_9pct", "cdt_random_3pct"):
        arm = copy.deepcopy(analog[base])
        arm.update(id=f"{base}_model_{suffix}", scope="model", digital_learning_rate=rate)
        arms.append(arm)
    doc = copy.deepcopy(lrdev)
    doc["lifecycle_id"] = "opt125m-cmo-l6-exp1e"
    doc["question"]["text"] = (
        "Experiment 1e: on L6 CMO MLPs with HWA on 50 M generated tokens and learning-rate decay, how does whole-model "
        f"HWA (every digital parameter trained, digital learning rate {rate:g}) compare with analog-only HWA for "
        "standard 3 %, standard 9 % and 3 % corruption-aware sources at 0-5 % stuck devices, on the WikiText-2 test "
        "and on held-out generated text?"
    )
    doc["question"]["decision"] = f"The HWA scope and recipe of the on-chip recovery sources (see {NOTE})."
    doc["hwa"]["arms"] = arms
    output.write_text(json.dumps(doc, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
