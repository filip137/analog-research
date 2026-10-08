"""Lifecycle helper commands: describe, check, plan, collect and corpus.

None of these commands launches training. Execution goes through the public
``python -m ebl campaign run`` (or ``python -m ebl train`` per stage).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

from experiments.artifacts import sha256_file
from experiments.cifar_crossbar.runtime import save
from experiments.schema import ConfigError
from workflow.lifecycle import EXPERIMENT_ID, OPT_MODELS, load_lifecycle, parse_execution, stage_selections
from workflow.plan import build_plan, commands, default_python, write_plan

ROOT = Path(__file__).resolve().parents[1]
# Saved final on-chip state per analog weight and trajectory: upper values measured on
# the CIFAR suffix (OM ~68, PCM ~34 bytes) and a tiny OPT OM run (67-88 bytes).
STATE_BYTES_PER_WEIGHT = {"om": 90, "pcm": 35}


def summary(lifecycle) -> dict:
    devices, hwa, onchip = lifecycle.devices, lifecycle.hwa, lifecycle.onchip
    stages = stage_selections(lifecycle)
    trajectories = len(hwa.arms) * len(devices.assignment_seeds) * len(lifecycle.defects.cases) * len(onchip.arms)
    return {
        "lifecycle_id": lifecycle.lifecycle_id,
        "lifecycle_sha256": lifecycle.digest(),
        "question": lifecycle.question.text,
        "decision": lifecycle.question.decision,
        "network": {"family": lifecycle.network.family, **lifecycle.network.summary()},
        "network_description": lifecycle.network.describe(),
        "devices": {
            "technology": devices.technology,
            "model": devices.model,
            "assignment_seeds": list(devices.assignment_seeds),
            "selection_seeds": list(devices.selection_seeds),
        },
        "defect_cases": [case.label for case in lifecycle.defects.cases],
        "deployment": lifecycle.deployment.to_dict(),
        "hwa_arms": [arm.to_dict() for arm in hwa.arms],
        "program_verify": lifecycle.program_verify.to_dict(),
        "onchip": {"update_law": onchip.update_law, "arms": [arm.to_dict() for arm in onchip.arms]},
        "stage_runs": {kind: sum(s.kind == kind for s in stages) for kind in ("prepare", "devices", "hwa", "deploy", "onchip")},
        "trajectories": trajectories,
        "analog_weights": lifecycle.network.analog_weights,
        "onchip_state_gb_estimate": round(
            trajectories * lifecycle.network.analog_weights * STATE_BYTES_PER_WEIGHT[devices.technology] / 1e9, 2
        ),
        "smoke_only": lifecycle.data.smoke,
    }


def _describe(args) -> int:
    value = summary(load_lifecycle(args.lifecycle))
    if args.json:
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    print(f"{value['lifecycle_id']}  ({value['lifecycle_sha256'][:12]})")
    print(f"question: {value['question']}")
    print(f"decision: {value['decision']}")
    print(f"network: {value['network_description']}")
    print(
        f"devices: {value['devices']['technology']} ({value['devices']['model']}); assignments "
        f"{value['devices']['assignment_seeds']}; selection arrays {value['devices']['selection_seeds']}"
    )
    print(f"defects: {', '.join(value['defect_cases'])}")
    print("hwa arms: " + ", ".join(arm["id"] for arm in value["hwa_arms"]))
    print(f"program/verify: {json.dumps(value['program_verify'], sort_keys=True)}")
    print(f"on-chip: {value['onchip']['update_law']}; arms " + ", ".join(a["id"] for a in value["onchip"]["arms"]))
    print(f"stage runs: {value['stage_runs']}; trajectories: {value['trajectories']}")
    print(
        f"analog weights: {value['analog_weights']:,}; final on-chip states ~{value['onchip_state_gb_estimate']} GB "
        "(estimate; measure with a cost probe)"
    )
    if value["smoke_only"]:
        print("smoke: truncated cohorts; not scientific evidence")
    return 0


def _check(args) -> int:
    paths = args.paths or sorted(ROOT.glob("campaigns/**/lifecycles/*.json"))
    if not paths:
        print("No lifecycle files found.", file=sys.stderr)
        return 1
    failed = 0
    for path in paths:
        try:
            lifecycle = load_lifecycle(path)
        except ConfigError as error:
            failed += 1
            print(f"FAIL {path}: {error}", file=sys.stderr)
        else:
            if Path(path).stem != lifecycle.lifecycle_id:
                failed += 1
                print(f"FAIL {path}: file name must equal lifecycle_id {lifecycle.lifecycle_id!r}", file=sys.stderr)
            else:
                print(f"ok   {path}  {lifecycle.digest()[:12]}")
    return 1 if failed else 0


def _plan(args) -> int:
    lifecycle = load_lifecycle(args.lifecycle)
    execution = parse_execution(
        {"device": args.device, "cpu_threads": args.cpu_threads, "disable_cudnn": args.disable_cudnn}
    )
    files = build_plan(
        lifecycle,
        teacher_weights=args.teacher_weights,
        execution=execution,
        python=args.python or default_python(),
        source=args.lifecycle.resolve(),
    )
    status = write_plan(files, args.output)
    runs = (args.runs or args.output.resolve().parent / "runs").resolve()
    print(json.dumps({"plan": str(args.output.resolve()), "status": status, "stages": len(files) - 3,
                      "commands": commands(args.output, runs)}, indent=2))
    return 0


def _rows(run_dir: Path, result: dict) -> list[dict]:
    artifacts = {item["kind"]: item for item in result["artifacts"]}
    record = artifacts["measurements"]
    path = run_dir / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Expected unchanged measurements in {run_dir}.")
    rows = []
    for row in json.loads(path.read_text()):
        split = "test" if "test" in row["final"] else "development"
        # Every metric the family reports as a number, at clean, deployed and final state.
        metrics = sorted(k for k, v in row["final"][split].items() if isinstance(v, (int, float)))
        flat = {
            "lifecycle_id": row["lifecycle_id"],
            "technology": row["technology"],
            "hwa_arm": row["hwa_arm"],
            "array_seed": row["array_seed"],
            "case": row["case"],
            "onchip_arm": row["onchip_arm"],
            "weights": row["weights"],
            "calibration": row["calibration"],
            "evaluation": split,
            **{f"{state}_{k}": row[state][split][k] for state in ("clean", "initial", "final") for k in metrics},
            **{k: row[k] for k in ("image_presentations", "sequence_presentations") if k in row},
            "failed_devices": row["identity"]["failed_devices"],
            "hwa_convergence_review_required": row["hwa_convergence_review_required"],
            "run_dir": str(run_dir),
        }
        for key, value in sorted(row["cost"].items()):
            flat[f"cost_{key}"] = value
        rows.append(flat)
    return rows


def _collect(args) -> int:
    rows = []
    for root in args.roots:
        for result_path in sorted(Path(root).rglob("result.json")):
            result = json.loads(result_path.read_text())
            if (
                result.get("schema") != "ebl.run"
                or result.get("experiment_id") != EXPERIMENT_ID
                or result.get("status") != "complete"
                or result["metrics"].get("stage", {}).get("kind") != "onchip"
            ):
                continue
            rows.extend(_rows(result_path.parent, result))
    keys = [(r["lifecycle_id"], r["hwa_arm"], r["array_seed"], r["case"], r["onchip_arm"]) for r in rows]
    for row, key in zip(rows, keys):
        # Retries are not replicates: duplicated coverage is flagged, not pooled.
        row["duplicate_coverage"] = keys.count(key) > 1
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (args.output / "rows.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "duplicates": sum(r["duplicate_coverage"] for r in rows),
                      "output": str(args.output)}))
    return 0


def _corpus(args) -> int:
    from workflow.networks.opt_mlp import build_corpus

    root = args.root or (Path(os.environ["EBL_CORPUS_ROOT"]) if os.environ.get("EBL_CORPUS_ROOT") else None)
    if root is None:
        raise ValueError("Expected --root or EBL_CORPUS_ROOT for the corpus directory.")
    path = root.resolve() / f"{args.name}.pt"
    if path.exists():
        raise ValueError(f"Expected a new corpus name; {path} already exists.")
    corpus = build_corpus(
        model=args.model,
        model_dir=args.model_dir,
        texts={"train": args.train, "validation": args.validation, "test": args.test},
    )
    save(path, corpus)
    print(json.dumps({
        "path": str(path),
        "tokens": {split: len(stream) for split, stream in corpus["splits"].items()},
        "lifecycle_data_corpus": {"name": args.name, "sha256": sha256_file(path)},
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m workflow", description=__doc__)
    commands_ = parser.add_subparsers(dest="command", required=True)
    describe = commands_.add_parser("describe", help="validate and summarize one lifecycle")
    describe.add_argument("lifecycle", type=Path)
    describe.add_argument("--json", action="store_true")
    describe.set_defaults(handler=_describe)
    check = commands_.add_parser("check", help="validate lifecycle files (default: every campaign lifecycle)")
    check.add_argument("paths", type=Path, nargs="*")
    check.set_defaults(handler=_check)
    plan = commands_.add_parser("plan", help="write stage configs and an ebl campaign manifest")
    plan.add_argument("lifecycle", type=Path)
    plan.add_argument("--teacher-weights", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--runs", type=Path, help="runner output directory printed in the commands")
    plan.add_argument("--device", default="cpu")
    plan.add_argument("--cpu-threads", type=int, default=1)
    plan.add_argument("--disable-cudnn", action="store_true")
    plan.add_argument("--python", type=Path, help="interpreter recorded in the manifest target")
    plan.set_defaults(handler=_plan)
    collect = commands_.add_parser("collect", help="flatten on-chip measurements of completed runs")
    collect.add_argument("roots", type=Path, nargs="+")
    collect.add_argument("--output", type=Path, required=True)
    collect.set_defaults(handler=_collect)
    corpus = commands_.add_parser("corpus", help="tokenize train/validation/test text into a pinned OPT corpus")
    corpus.add_argument("--name", required=True, help="corpus name; written as <root>/<name>.pt")
    corpus.add_argument("--model", default="facebook/opt-125m", choices=sorted(OPT_MODELS))
    corpus.add_argument("--model-dir", type=Path, required=True, help="local snapshot with the tokenizer files")
    corpus.add_argument("--train", type=Path, required=True)
    corpus.add_argument("--validation", type=Path, required=True)
    corpus.add_argument("--test", type=Path, required=True)
    corpus.add_argument("--root", type=Path, help="corpus directory (default: EBL_CORPUS_ROOT)")
    corpus.set_defaults(handler=_corpus)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ConfigError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
