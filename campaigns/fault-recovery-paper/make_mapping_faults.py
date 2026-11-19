"""Lifecycles of fault-recovery-paper series 003 exp-001 (crossbar half): mapping x faults x HWA, holds only.

Owning note: on-chip-learning campaigns/fault-recovery-paper/series/003-mapping-deployment/experiments/
exp-001-mapping-faults-hwa.md (issue #33, user, 19 November 2026). Pilot note: campaigns/pilots/
20261119-mapping-faults.md.

Each cell copies the paper-grid map-271002 lifecycle of its depth and rate unchanged (devices,
defects, deployment, P&V, rewrite seed 241016 at 5 sigma, development and held-out cohorts,
unused recovery/target/schedule pins) and changes only: the arms (the two holds), no analog-LoRA
section, the declared ``fault_tuning.mapping`` and, for the no-HWA source, the single HWA arm
``pretrained`` with the pinned pretrained-teacher source. ``--source-only`` writes the two
pretrained-source lifecycles (prepare, devices and ``hwa__pretrained`` are the only stages run).

usage: python campaigns/fault-recovery-paper/make_mapping_faults.py [PRETRAINED_XB6_SHA PRETRAINED_XB12_SHA]
"""
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
GRID = HERE / "lifecycles/paper-grid-20261116/paper"
OUT = HERE / "lifecycles/mapping-faults-20261119"
STAMP = "mapping-faults-20261119"
MAPPINGS = {"base": {"kind": "reference", "gain": {"rule": "unit"}, "margin_us": 0.0},
            "two": {"kind": "two_sided", "gain": {"rule": "unit"}, "margin_us": 0.0}}
RATES = ("00", "01", "03", "05")
MODELS = ("xb6", "xb12")
PRETRAINED = {"id": "pretrained", "method": "none"}
HOLDS = [{"label": "naive_hold", "method": "hold_naive"}, {"label": "fault_aware_hold", "method": "hold_fault_aware"}]
QUESTION = {
    "decision": "Mapping for the #33 rerun under drift-recovery conditions (series003 exp-001); evaluation only.",
    "evidence_class": "exploratory_model_based",
    "primary_metric": "teacher_kl",
    "text": "Which weight-to-conductance mapping gives the best deployed quality under permanent faults, "
            "with and without HWA?",
}


def template(model, rate):
    return json.loads((GRID / f"fault-paper-grid-20261116-{model}-r{rate}-271002.json").read_text())


def write(doc):
    path = OUT / f"{doc['lifecycle_id']}.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def source_lifecycle(model):
    doc = template(model, "00")
    del doc["fault_tuning"]
    doc["hwa"]["arms"] = [dict(PRETRAINED)]
    doc["lifecycle_id"] = f"{STAMP}-{model}-pretrained-source"
    doc["question"] = {**QUESTION, "decision": "Produces the pinned no-HWA (pretrained teacher) source only; "
                       "stages prepare, devices and hwa__pretrained."}
    return doc


def cell(model, source, mapping, rate, pretrained):
    doc = template(model, rate)
    ft = doc["fault_tuning"]
    ft.update(arms=copy.deepcopy(HOLDS), analog_lora=None, mapping=dict(MAPPINGS[mapping]))
    if source == "nohwa":
        doc["hwa"]["arms"] = [dict(PRETRAINED)]
        ft["source_sha256"] = pretrained[model]
    doc["lifecycle_id"] = f"{STAMP}-{model}-{source}-{mapping}-r{rate}"
    doc["question"] = dict(QUESTION)
    return doc


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        print(write(source_lifecycle(model)))
    if not argv:
        return
    pretrained = dict(zip(MODELS, argv))
    for model in MODELS:
        for source in ("hwa", "nohwa"):
            for mapping in MAPPINGS:
                for rate in RATES:
                    print(write(cell(model, source, mapping, rate, pretrained)))


if __name__ == "__main__":
    main(sys.argv[1:])
