"""Resolve native DRN configs for direct OPT-1.3B conversion and HWA."""
import argparse
import json
from pathlib import Path

from ebl.schema.opt_cmo_drn import parse_config

TEACHER = "cf7d5c970d6ddbd3b03009b397c0422e147edd5c8020d47a8d2fac0b11a3b08d"


def config(depth, phase, factor, corpus_digest, stage, assignment, rate):
    template = Path(__file__).resolve().parents[2] / "examples/opt_cmo_drn/l12/hwa_std3.json"
    doc = json.loads(template.read_text())
    steps = {"canary": 4, "screen": 512, "production": 12200}[phase]
    doc.update(stage=stage, model="facebook/opt-1.3b", analog_layers=depth,
               initialization="pretrained_ratio3_quad.v1", source_sha256=TEACHER, teacher_sha256=TEACHER)
    doc["data"].update(corpus="opt13b-synthetic-50m", sha256=corpus_digest, sequences=195200)
    if phase == "canary":
        doc["data"].update(corpus="opt13b-synthetic-canary", sequences=256)
    doc["deployment"].update(assignment_seed=assignment, fault_kind="random" if rate else "none",
                             fault_rate=rate)
    doc["training"].update(epochs=2, max_updates=0 if phase=="production" else steps,
                           conductance_lr=1e-4*factor, calibration_lr=1e-4*factor, digital_lr=3e-6*factor,
                           warmup_updates=max(1, round(0.008*steps)))
    if phase == "production" and stage == "deploy":
        doc["deployment"]["source_evaluation"] = "cmo_24h_x1_x4"
    parse_config(doc)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, choices=(12, 24), required=True)
    p.add_argument("--phase", choices=("canary", "screen", "production"), required=True)
    p.add_argument("--factor", type=float, choices=(0.01, 0.1, 1, 3, 9), required=True)
    p.add_argument("--stage", choices=("prepare", "hwa", "deploy", "hwa_precision_benchmark", "hwa_tf32_benchmark", "hwa_speed_benchmark"), required=True)
    p.add_argument("--assignment", type=int, choices=(271001,271002,271003,271004), default=271001)
    p.add_argument("--fault-rate", type=float, choices=(0,0.03), default=0)
    p.add_argument("--corpus-sha256", required=True, help="File digest from shared corpus generation")
    p.add_argument("--hwa-execution", type=Path, help="JSON update settings selected by the speed benchmark")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    doc = config(a.depth, a.phase, a.factor, a.corpus_sha256, a.stage, a.assignment, a.fault_rate)
    if a.hwa_execution:
        if a.stage != "hwa":
            p.error("--hwa-execution requires --stage hwa")
        doc["hwa_execution"] = json.loads(a.hwa_execution.read_text())
        parse_config(doc)
    with a.output.open("x") as out:
        out.write(json.dumps(doc, indent=2) + "\n")
