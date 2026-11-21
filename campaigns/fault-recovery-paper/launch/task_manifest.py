"""Build a pinned native local-GPU fault-tuning unit from the matched XB lifecycle."""
import argparse
from pathlib import Path

from worker import digest, read, write


BASE = Path(__file__).resolve().parents[1] / "lifecycles/mapped-fault-recovery-20261120"
ROUTES = ["offline_lora", "onchip_lora", "offline_child01", "onchip_child01",
          "offline_child10", "onchip_child10", "offline_full"]
REFERENCE = {
    6: ("980be1f8b10daad1ce060b7d0f9a75d7a8eb6caffd38410be66df2f33f50831d", .05706044211983681),
    12: ("7c96f4b7f601705a6fad06954567c01427d833639d4880e3e867eae14d9d6944", .21380882339150298),
}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--depth", required=True, choices=(6, 12), type=int)
    cli.add_argument("--task", required=True, choices=("hellaswag", "boolq", "pubmed"))
    cli.add_argument("--phase", required=True, choices=("canary", "development", "evaluation"))
    cli.add_argument("--rate", required=True, choices=(0, 3), type=int)
    cli.add_argument("--root", required=True, type=Path)
    cli.add_argument("--selection", help="Pinned completed selection filename stem in inputs")
    cli.add_argument("--routes", nargs="+", choices=ROUTES, default=ROUTES)
    cli.add_argument("--tag", default="v1")
    args = cli.parse_args()
    root, source = args.root.resolve(), BASE.parents[3]
    inputs = root / "inputs"
    records = read(inputs / "receipt.json")
    unit = f"xb{args.depth}-{args.task}-r{args.rate:02d}-{args.phase}-{args.tag}"
    specdir = root / "manifests" / unit
    specdir.mkdir(parents=True, exist_ok=True)
    doc = read(BASE / f"xb{args.depth}-r{args.rate:02d}.json")
    old = doc["fault_tuning"]
    seed = {"canary": 271002, "development": 281101, "evaluation": 281201}[args.phase]
    canary = args.phase == "canary"
    pin = lambda name: {"name": name, "sha256": records[name]["sha256"]}
    selection = None
    if args.phase == "evaluation":
        if not args.selection:
            cli.error("--selection is required for evaluation")
        selection = {"name": args.selection, "sha256": digest(inputs / (args.selection + ".pt"))}
    t = {"task": pin(args.task), "retention": pin("wikitext"), "selection": selection, "phase": args.phase,
         "proxy_seed": 281102, "seed": 41, "routes": args.routes,
         "rates": {"lora": .001 if args.task == "pubmed" else .0003,
                   "child": .001, "full": .0003 if args.task == "pubmed" else .00003},
         "steps": 4 if canary else 128, "fisher_items": 32 if canary else 1024,
         "microbatch": 4, "eval_limit": 8 if canary else None}
    if canary and args.rate == 3:
        archive_digest = ("5a548e5a1a3d1244771897389787593ed1aeed22234a63a6ccb58c53119eb6a3" if args.depth == 6
                          else "e81cbfd2fae7320d3e7a79d3420f0a9cd70e9471e2d8bcf903286b12b3b3adad")
        t["historical"] = {"name": f"v100-xb{args.depth}-271002", "sha256": archive_digest}
    doc.update(lifecycle_id="s7-" + unit,
               question={"text": "Does adapting on the actual faulty device improve the matched task score?",
                         "decision": "Compare paired routes at 3% and their nominal twins on disjoint chips.",
                         "primary_metric": "perplexity" if args.task == "pubmed" else "task_accuracy_norm_percent",
                         "evidence_class": "exploratory_model_based"},
               fault_tuning={k: old[k] for k in ("source_sha256", "mapping", "fault_rewrite")})
    doc["fault_tuning"].update(assignment_seed=seed, task_adaptation=t)
    config = specdir / "lifecycle.json"
    write(config, doc)
    synth = doc["data"]["corpus"]
    manifest = {"schema": "fault_tuning.local_manifest.v1", "source": str(source),
                "source_receipt_sha256": digest(source / "source_snapshot.json"),
                "output": str(root / "runs"), "corpus_root": str(inputs),
                "teacher": str(inputs / "opt-125m-27dcfa74/pytorch_model.bin"),
                "teacher_sha256": "2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9",
                "weights": str(inputs / ("crossbar-std3.pt" if args.depth == 6 else "source-l12-ed53b6cb.pt")),
                "weights_sha256": old["source_sha256"],
                "pinned": {args.task: records[args.task]["sha256"], "wikitext": records["wikitext"]["sha256"],
                           synth["name"]: synth["sha256"]},
                "units": {unit: str(config)}, "config_sha256": digest(config)}
    if canary and args.rate == 3:
        apparent, score = REFERENCE[args.depth]
        manifest["historical"] = {"apparent": apparent, "teacher_kl": score, "atol": 1.e-6}
    write(specdir / "manifest.json", manifest)
    print(specdir / "manifest.json")


if __name__ == "__main__":
    main()
