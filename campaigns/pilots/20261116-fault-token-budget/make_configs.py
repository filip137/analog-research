"""Bind the declared token-budget cases to verified, stored teacher-target files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BACKEND = 'drn'
BASELINE_SAMPLES = "df73ce6615be3cdd3005d2ca0b97c03131f9236ba481bc7250edde5ff66fb01a"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def build(case, sample, canary=False):
    case = dict(case)
    if canary:
        case.update(case_id=case["case_id"] + "-canary", steps=32, wall_minutes=20)
    if sample["sequences"] != case["sequences"] or sample["cohort_sha256"] != case["cohort_sha256"]:
        raise ValueError("Sample receipt does not match the declared recovery cohort")
    if sample["shape"] != [case["sequences"], 255, 8]:
        raise ValueError("Expected eight targets at each next-token position")
    if Path(sample["file"]).name != sample["file"] or not sample["file"].endswith(".pt"):
        raise ValueError("Expected a plain .pt target filename")
    model = case["model"]
    if BACKEND == "drn":
        base = "drn12-a-holds-noisy" if model == "drn12" else "drn6-matched-noiseless"
        doc = json.loads((ROOT / "examples/opt_cmo_drn/fault_paper_completion" / (base + ".json")).read_text())
        doc["cpu_threads"] = 10
        f = doc["fault_tuning"]
        f.update(recovery_sequences=case["sequences"], recovery_sha256=case["cohort_sha256"],
                 samples_file=sample["file"], samples_sha256=sample["tensor_sha256"], steps=case["steps"], development_every=100,
                 arms=[{"label": "noisy-fixed-lr", "method": "noisy", "learning_rate": case["learning_rate"]}])
        doc["training"]["max_wall_seconds"] = case["wall_minutes"] * 60 - 120
        from ebl.schema.opt_cmo_drn import parse_config
        parse_config(doc)
    else:
        base = f"fault-completion-20261115-{model}-matched-noiseless.json"
        doc = json.loads((ROOT / "campaigns/fault-recovery-paper/lifecycles" / base).read_text())
        doc["lifecycle_id"] = "fault-token-budget-20261116-" + case["case_id"]
        f = doc["fault_tuning"]
        f["recovery"].update(sequences=case["sequences"], cohort_sha256=case["cohort_sha256"])
        f["targets"].update(file=sample["file"][:-3], sha256=sample["tensor_sha256"])
        f["schedule"].update(steps=case["steps"], development_every_steps=100)
        f["arms"] = [{"label": "noisy-fixed-lr", "method": "onchip_noisy", "learning_rate": case["learning_rate"]}]
        doc["question"].update(text="How far does recovery KL decrease with a one fixed larger dataset and more updates at fixed selected LR?", decision="Compare full endpoints and KL-versus-token curves; do not retune LR or threshold.")
        from ebl.schema.lifecycle import parse_lifecycle
        parse_lifecycle(doc)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    ap.add_argument("samples_root", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--canaries", action="store_true")
    args = ap.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if receipt.get("schema") != "fault_token_targets.v1" or receipt.get("status") != "complete":
        raise ValueError("Expected complete target-preparation receipt")
    if receipt.get("baseline_prefix_sha256") != BASELINE_SAMPLES:
        raise ValueError("Original teacher samples were not preserved")
    if receipt.get("teacher_sha256") != "2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9" or receipt.get("samples_seed") != 20365833 or receipt.get("draw_batch") != 16 or receipt.get("count") != 8:
        raise ValueError("Unexpected teacher target preparation contract")
    samples = {s["sequences"]: s for s in receipt["samples"]}
    cases = json.loads(Path(__file__).with_name("cases.json").read_text())
    documents = []
    for case in cases:
        sample = samples[case["sequences"]]
        if digest(args.samples_root / sample["file"]) != sample["file_sha256"]:
            raise ValueError("Stored teacher target file differs from its receipt")
        documents.append((dict(case, case_id=case["case_id"] + ("-canary" if args.canaries else "")), build(case, sample, args.canaries)))
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for case, doc in documents:
        name = case["case_id"] if BACKEND == "drn" else doc["lifecycle_id"]
        path = args.output / (name + ".json")
        path.write_text(json.dumps(doc, indent=2) + "\n")
        records.append({"case_id": case["case_id"], "path": str(path), "sha256": digest(path)})
    (args.output / "config-receipt.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "cases": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
