"""Resolve one native lifecycle against the shared corpus receipt."""
import argparse
import json
from pathlib import Path

from ebl.schema.lifecycle import parse_lifecycle


def lifecycle(depth, phase, factor, corpus_digest):
    path = Path(__file__).resolve().parents[1] / "pilots/20261118-opt1p3b-port" / f"opt1p3b-cmo-l{depth}-probe.json"
    doc = json.loads(path.read_text())
    steps = {"canary": 4, "screen": 512, "production": 12200}[phase]
    doc["lifecycle_id"] = f"opt13b-xb-l{depth}-{phase}-{factor:g}".replace(".", "p")
    doc["question"].update(text="How well does the mapped whole-model std3 source preserve teacher fidelity?",
                           decision="Select the peak learning rates independently per analog depth on development arrays.")
    doc["deployment"]["encoding"] = "two_sided"
    doc["data"].update(corpus={"name": "opt13b-synthetic-50m", "sha256": corpus_digest},
                       hwa_sequences=195200, development_sequences=256, recovery_sequences=1024)
    if phase == "canary":
        doc["data"].update(hwa_sequences=256, recovery_sequences=256)
        doc["data"]["corpus"]["name"] = "opt13b-synthetic-canary"
    if phase == "production":
        doc["data"].update(evaluation="test", test_sequences=384)
        doc["deployment"]["source_evaluation"] = "cmo_24h_x1_x4"
        doc["transfer_tests"] = [{"label": "wikitext", "split": "test", "sequences": 1024, "corpus": {
            "name": "wikitext-2-raw-v1-opt1p3b", "sha256": "15345cfac6d21630d320e2a5c98fed62da63e80ec547a719c94bdd0c13936aba"}}]
    hwa = doc["hwa"]
    hwa.update(epochs=steps//4, sequences_per_epoch=128, learning_rate=0.005*factor,
               numerics={"matmul": "tf32", "attention": "sdpa", "fused_adam": True},
               lr_schedule={"kind": "cosine_warmup_decay", "total_steps": steps,
                            "warmup_steps": max(1, round(0.008*steps)), "min_factor": 0.2})
    hwa["selection"]["every_epochs"] = 128
    hwa["arms"][0].update(id="std3", digital_learning_rate=1e-5*factor)
    if phase == "production":
        hwa["arms"].append({"id": "pretrained", "method": "none"})
        doc["devices"]["assignment_seeds"] = [271002, 271003, 271004]
    parse_lifecycle(doc)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, choices=(12, 24), required=True)
    p.add_argument("--phase", choices=("canary", "screen", "production"), required=True)
    p.add_argument("--factor", type=float, choices=(0.01, 0.1, 1, 3, 9), required=True)
    p.add_argument("--corpus-sha256", required=True, help="Digest printed by inputs.py corpus")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    doc = lifecycle(a.depth, a.phase, a.factor, a.corpus_sha256)
    with a.output.open("x") as out:
        out.write(json.dumps(doc, indent=2) + "\n")
