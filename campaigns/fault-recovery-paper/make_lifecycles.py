"""Fault-recovery paper, series 001 (crossbar): strict lifecycles of the LR-tuning screen at 3 %.

usage: make_lifecycles.py [--pins pins.json]

Writes lifecycles/<id>.json. Units:
- ``deploy``: the pinned tuning deployment (naive P&V, fault-aware pair rewrite and the
  analog-LoRA tiles), holds only; run locally, its file becomes ``pins.deployment``.
- ``bscale``: the r16 digital LoRA at 3e-3 on that deployment (declared B-scale fallback).
- ``canary``: every method for 4 steps (readiness only, not evidence).
- ``probe``: full-length noisy, Child-Tuning_D and analog-LoRA arms at 3e-3 (timing/readiness only).
- ``j1``..``j3``: the Jean Zay screen units (one V100 dev job each, in-job V100 draw)
- ``a1``..``a3``: the same arms, loading the stored V100 draw (any GPU) (``SAMPLES_FILE``,
  written by the DRN half's V100 draw job only if its hash is ``df73ce66``)
- ``canaryfile``: the stored-draw path on the 3090 (readiness only)., all with the V100 sample draw ``df73ce66``.

Local units (3090) declare the 3090 draw of the same procedure (see the experiment note).

exp-002 (12 MLPs, local RTX 3090 ``nom-cool-1``; note ``EXP2_NOTE``): units ``l12-deploy``,
``l12-bscale``, ``l12-canary`` (every screen arm for 4 steps) and ``l12-s1``..``l12-s3`` (the
screen). Only the depth (``analog_decoder_layers`` 12), the 12-MLP std3 source and the extended
grids differ from exp-001; every l12 unit loads the stored V100 draw ``df73ce66`` (no 3090 draw).
Their pins are ``pins.json["l12"]``.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from ebl.schema.lifecycle import parse_lifecycle  # noqa: E402

BASE = ROOT / "campaigns/opt-cmo-corruption/archive/exp-021-022/opt125m-cmo-l6-exp22-pv-learning-noisy-random3.json"
NOTE = "campaigns/fault-recovery-paper/series/001-lr-tuning/experiments/exp-001-crossbar-lr-screen.md"
PREFIX = "opt125m-cmo-l6-fault-paper-tune"
SOURCE = "86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028"
WIKITEXT = ("wikitext-2-raw-v1", "33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01")
SAMPLES_V100 = "df73ce6615be3cdd3005d2ca0b97c03131f9236ba481bc7250edde5ff66fb01a"   # recovery-targets exp-001/002, V100
SAMPLES_3090 = "6bc43ffaa92134c39308b844aba562f97c396229976ab416cd2a452d9c338b4f"   # same procedure on the RTX 3090
SAMPLES_FILE = "teacher-samples8-c919329d-df73ce66"
SAMPLES_FILE_3090 = "teacher-samples8-c919329d-6bc43ffa-rtx3090"
GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
LORA_GRID = [1e-3, 3e-3, 1e-2]
# exp-002 (12 MLPs): the 12-MLP std3-recipe fit of drift-all-mlps; exp-001's grids plus their edge extensions.
EXP2_NOTE = "campaigns/fault-recovery-paper/series/001-lr-tuning/experiments/exp-002-crossbar-l12-lr-screen.md"
EXP2_PREFIX = "opt125m-cmo-l12-fault-paper-tune"
EXP2_SOURCE = "ed53b6cbe2a7315f364291c7142ee2826c6ca4a44c21a2d5d25190b8c90569b0"
EXP2_LORA_GRID = GRID
EXP2_ALORA_GRID = GRID + [3e-2]


def lr_tag(value):
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}em{-int(exponent)}"


def arms(method, grid, **extra):
    short = {"onchip_noisy": "noisy", "onchip_noiseless": "noiseless", "child_tuning_d": "childd",
             "digital_lora": f"dlora-r{extra.get('rank')}", "analog_lora": f"alora-r{extra.get('rank')}"}[method]
    return [{"label": f"{short}-{lr_tag(lr)}", "method": method, "learning_rate": lr, **extra} for lr in grid]


HOLDS = [{"label": "hold-naive", "method": "hold_naive"}, {"label": "hold-fault-aware", "method": "hold_fault_aware"}]
UNITS = {
    "j1": HOLDS + arms("onchip_noisy", GRID) + arms("onchip_noiseless", GRID),
    "j2": (arms("child_tuning_d", GRID, reserve_p=0.1) + arms("digital_lora", LORA_GRID, rank=16)
           + arms("digital_lora", LORA_GRID, rank=64)),
    "j3": arms("analog_lora", GRID, rank=16),
}


EXP2_UNITS = {
    "s1": HOLDS + arms("onchip_noisy", GRID) + arms("onchip_noiseless", GRID),
    "s2": (arms("child_tuning_d", GRID, reserve_p=0.1) + arms("digital_lora", EXP2_LORA_GRID, rank=16)
           + arms("digital_lora", EXP2_LORA_GRID, rank=64)),
    "s3": arms("analog_lora", EXP2_ALORA_GRID, rank=16),
}


def section_l12(pins, unit):
    """exp-002: exp-001's section with the 12-MLP source, the stored V100 draw and the extended grids."""
    pins = pins.get("l12", {})
    name = unit.removeprefix("l12-")
    if name not in ("deploy", "bscale", "canary", *EXP2_UNITS):
        raise SystemExit(f"Unknown exp-002 unit {unit}.")
    doc = section({"deployment": pins.get("deployment"), "b_scale": pins.get("b_scale")},
                  name if name in ("deploy", "bscale") else "a1")
    doc["source_sha256"] = EXP2_SOURCE
    doc["targets"] = {"kind": "teacher_samples", "count": 8, "seed": 20261104 + 104729, "draw_batch": 16,
                      "sha256": SAMPLES_V100, "file": SAMPLES_FILE}
    if name == "canary":
        doc["arms"] = [a for unit_arms in EXP2_UNITS.values() for a in unit_arms]
        doc["schedule"] = {**doc["schedule"], "steps": 4, "development_every_steps": 2}
    elif name in EXP2_UNITS:
        doc["arms"] = EXP2_UNITS[name]
    if any(a["method"] == "analog_lora" for a in doc["arms"]) and doc["analog_lora"]["b_scale"] is None:
        raise SystemExit(f"{unit}: pin the analog-LoRA B scale first (pins.json l12.b_scale).")
    return doc


def section(pins, unit):
    local = unit in ("deploy", "bscale", "canary", "probe", "canaryfile")
    analog = {"rank": 16, "b_scale": None if unit in ("deploy", "bscale") else pins.get("b_scale")}
    deployment = None if unit == "deploy" else pins.get("deployment")
    if unit == "deploy":
        unit_arms = HOLDS
    elif unit == "bscale":
        unit_arms = [{"label": "dlora-r16-3em3", "method": "digital_lora", "learning_rate": 3e-3, "rank": 16}]
    elif unit == "canary":
        unit_arms = HOLDS + [a[0] for a in (arms("onchip_noisy", [1e-3]), arms("onchip_noiseless", [1e-3]),
                                            arms("child_tuning_d", [1e-3], reserve_p=0.1),
                                            arms("digital_lora", [1e-3], rank=16), arms("analog_lora", [1e-3], rank=16))]
    elif unit == "canaryfile":
        unit_arms = HOLDS[:1] + arms("digital_lora", [1e-3], rank=16)
    elif unit == "probe":
        unit_arms = [arms("onchip_noisy", [3e-3])[0], arms("child_tuning_d", [3e-3], reserve_p=0.1)[0],
                     arms("analog_lora", [3e-3], rank=16)[0]]
    else:
        unit_arms = UNITS[unit.replace("a", "j")]
    if any(a["method"] == "analog_lora" for a in unit_arms) and analog["b_scale"] is None:
        raise SystemExit(f"{unit}: pin the analog-LoRA B scale first (pins.json b_scale).")
    if unit != "deploy" and deployment is None:
        raise SystemExit(f"{unit}: pin the deployment first (pins.json deployment).")
    targets = {"kind": "teacher_samples", "count": 8, "seed": 20261104 + 104729, "draw_batch": 16,
               "sha256": SAMPLES_3090 if local else SAMPLES_V100}
    if unit.startswith("a"):
        targets["file"] = SAMPLES_FILE
    elif unit == "canaryfile":
        targets["file"] = SAMPLES_FILE_3090
    return {
        "source_sha256": SOURCE,
        "assignment_seed": 271001,
        "fault_rewrite": {"detection": "verify", "roles": "pair", "seed": 241016, "threshold_sigma": 5.0},
        "recovery": {"corpus": WIKITEXT[0], "corpus_sha256": WIKITEXT[1], "split": "train", "start": 0,
                     "sequences": 1024, "seed": 20261104,
                     "cohort_sha256": "c919329dbe0d545d7614daf3d8f790881e944cd80ff53fff88fc19d9697f7762"},
        "development": {"corpus": WIKITEXT[0], "corpus_sha256": WIKITEXT[1], "split": "validation", "start": 0,
                        "sequences": 64, "seed": None,
                        "cohort_sha256": "7411cf50b625e4c62f65d8e12bac538fd83c6dad3e4c74b29a9f2a6f5ad83191"},
        "targets": targets,
        "schedule": {"optimizer": "adam", "betas": [0.9, 0.999], "eps": 1e-8,
                     "steps": 4 if unit.startswith("canary") else 300, "batch": 8, "order_seed": 20261104,
                     "lr_schedule": "cosine_to_zero", "warmup_steps": 0,
                     "development_every_steps": 2 if unit.startswith("canary") else 50},
        "write_law": {"threshold_sigma": 2.0, "accumulate_updates": True, "max_write_attempts_per_device": None},
        "fisher": {"estimator": "mean_squared_batch_gradient", "batch": 8},
        "lora_seed": 271001,
        "analog_lora": analog,
        "deployment": deployment,
        "arms": unit_arms,
    }


def lifecycle(pins, unit):
    doc = json.loads(BASE.read_text())
    doc["lifecycle_id"] = f"{PREFIX}-{unit}"
    doc["question"] = {
        "text": "Which learning rate does each crossbar recovery arm select at 3 % random stuck devices "
                "(tuning assignment 271001, final development teacher KL)?",
        "decision": "Fault-recovery paper series 001: per-arm learning rates for the paper seeds. " + NOTE,
        "evidence_class": "exploratory_model_based", "primary_metric": "teacher_kl"}
    doc["defects"]["cases"] = [{"array_concentration": None, "die_concentration": None, "kind": "random",
                                "rate_ppm": 30000}]
    doc["onchip"] = None
    doc.pop("transfer_tests", None)
    if unit.startswith("l12-"):
        doc["lifecycle_id"] = f"{EXP2_PREFIX}-{unit.removeprefix('l12-')}"
        doc["question"]["text"] = ("Which learning rate does each crossbar recovery arm select at 3 % random stuck "
                                   "devices with all 12 MLPs analog (tuning assignment 271001, final development "
                                   "teacher KL)?")
        doc["question"]["decision"] = ("Fault-recovery paper series 001, 12 MLPs: per-arm learning rates for the "
                                       "paper seeds. " + EXP2_NOTE)
        doc["network"]["analog_decoder_layers"] = 12
        doc["fault_tuning"] = section_l12(pins, unit)
    else:
        doc["fault_tuning"] = section(pins, unit)
    parse_lifecycle(copy.deepcopy(doc))
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, default=HERE / "pins.json")
    parser.add_argument("units", nargs="*", default=["deploy", "bscale", "canary", "probe", "canaryfile", "j1", "j2", "j3", "a1", "a2", "a3"])
    args = parser.parse_args()
    pins = json.loads(args.pins.read_text()) if args.pins.exists() else {}
    for unit in args.units:
        doc = lifecycle(pins, unit)
        path = HERE / "lifecycles" / f"{doc['lifecycle_id']}.json"
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
