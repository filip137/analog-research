"""ebl.lifecycle.cli — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from core.artifacts import sha256_file
from core.contracts import ConfigError
from networks.config import OPT_MODELS
from ebl.schema.lifecycle import STAGE_KINDS, load_lifecycle, stage_selections
from core.execution import parse_execution
from ebl.lifecycle.plan import build_plan, commands, default_python, write_plan
from ebl.lifecycle.collect import _collect
from core.paths import REPO_ROOT as ROOT


# Saved final on-chip state per analog weight and trajectory: upper values measured on
# the CIFAR suffix (OM ~68, PCM ~34 bytes) and tiny OPT runs (OM 67-88, CMO 24-32.3 bytes).
STATE_BYTES_PER_WEIGHT = {"om": 90, "pcm": 35, "cmo": 35}


def summary(lifecycle) -> dict:
    devices, hwa, onchip = lifecycle.devices, lifecycle.hwa, lifecycle.onchip
    stages = stage_selections(lifecycle)
    deployments = len(hwa.arms) * len(devices.assignment_seeds) * len(lifecycle.defects.cases)
    if lifecycle.retention is not None or lifecycle.drift_recovery is not None or lifecycle.initial_programming is not None:
        deployments = 0
    trajectories = (0 if onchip is None else len(hwa.arms) * len(devices.assignment_seeds)
                    * len(lifecycle.onchip_cases) * len(onchip.arms))
    state_bytes_per_weight = STATE_BYTES_PER_WEIGHT[devices.technology]
    if onchip is not None and onchip.accumulate_updates:
        state_bytes_per_weight += 4  # Saved FP32 pending command per weight.
    if onchip is not None and onchip.max_write_attempts_per_device is not None:
        state_bytes_per_weight += 4  # Controller-owned int32 attempt count per weight.
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
        "onchip": (
            None if onchip is None else {"update_law": onchip.update_law, "arms": [arm.to_dict() for arm in onchip.arms]}
        ),
        "deployed_cases": deployments,
        "stage_runs": {kind: sum(s.kind == kind for s in stages) for kind in STAGE_KINDS
                       if (kind != "retention" or lifecycle.retention is not None)
                       and (kind != "drift_recovery" or lifecycle.drift_recovery is not None)
                       and (kind != "initial_programming" or lifecycle.initial_programming is not None)
                       and (kind != "compensate" or lifecycle.drift_compensation is not None)},
        **({"retention": lifecycle.retention.to_dict(),
            "retention_snapshots": len(hwa.arms) * len(lifecycle.retention.assignment_seeds)
            * len(lifecycle.retention.case_labels)
            * sum(1 + len(lifecycle.retention.snapshots(m)) for m in lifecycle.retention.models),
            "retention_state_gb_estimate": 0.0 if lifecycle.drift_compensation is not None else round(len(hwa.arms) * len(lifecycle.retention.assignment_seeds)
            * len(lifecycle.retention.case_labels) * lifecycle.network.analog_weights
            * sum(34 if m == "pcm" else 26 for m in lifecycle.retention.models) / 1e9, 2)}
           if lifecycle.retention is not None else {}),
        "trajectories": trajectories,
        **({'initial_programming': lifecycle.initial_programming.to_dict(),
            'initial_programming_trajectories': len(hwa.arms)*len(lifecycle.initial_programming.assignment_seeds),
            'initial_programming_snapshots': len(hwa.arms)*len(lifecycle.initial_programming.assignment_seeds)*len(lifecycle.initial_programming.evaluation_seconds),
            'initial_programming_state_gb_estimate': len(hwa.arms)*len(lifecycle.initial_programming.assignment_seeds)*lifecycle.network.analog_weights*9/1e9}
           if lifecycle.initial_programming is not None else {}),
        **({"drift_recovery": lifecycle.drift_recovery.to_dict(),
            "custom_drift_physical_devices": 2*lifecycle.network.analog_weights,
            "custom_drift_state_gb_per_trajectory": 2*lifecycle.network.analog_weights*26.4/1e9}
           if lifecycle.drift_recovery is not None else {}),
        **({"drift_compensation": lifecycle.drift_compensation.to_dict(),
            "compensated_snapshots": len(hwa.arms) * len(lifecycle.retention.assignment_seeds)
            * len(lifecycle.retention.case_labels) * len(lifecycle.drift_compensation.methods)
            * sum(len(lifecycle.drift_compensation.snapshots(lifecycle.retention, m)) for m in lifecycle.retention.models)}
           if lifecycle.drift_compensation is not None else {}),
        **({"task": lifecycle.task.to_dict()} if lifecycle.task is not None else {}),
        "analog_weights": lifecycle.network.analog_weights,
        "onchip_state_gb_estimate": round(
            trajectories * lifecycle.network.analog_weights * state_bytes_per_weight / 1e9, 2
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
    if "task" in value:
        print(f"task: {value['task']['name']} ({value['task']['decision']}); development/test examples "
              f"{value['task']['development_examples']}/{value['task']['evaluation_examples']}")
    print("hwa arms: " + ", ".join(arm["id"] for arm in value["hwa_arms"]))
    print(f"program/verify: {json.dumps(value['program_verify'], sort_keys=True)}")
    if value["onchip"] is None:
        print("on-chip: none (deployment-only loop)")
    else:
        print(f"on-chip: {value['onchip']['update_law']}; arms " + ", ".join(a["id"] for a in value["onchip"]["arms"]))
    print(
        f"stage runs: {value['stage_runs']}; deployed cases: {value['deployed_cases']}; "
        f"trajectories: {value['trajectories']}"
    )
    print(
        f"analog weights: {value['analog_weights']:,}; final on-chip states ~{value['onchip_state_gb_estimate']} GB "
        "(estimate; measure with a cost probe)"
    )
    if value["smoke_only"]:
        print("smoke: truncated cohorts; not scientific evidence")
    if "retention" in value:
        print(f"retention: {value['retention_snapshots']} snapshots; saved state ~{value['retention_state_gb_estimate']} GB")
    if "drift_recovery" in value:
        print(f"custom drift recovery: {value['custom_drift_physical_devices']:,} physical cells; "
              f"~{value['custom_drift_state_gb_per_trajectory']:.2f} GB per saved trajectory; "
              f"canary={value['drift_recovery']['canary']}")
    if "drift_compensation" in value:
        print(f"compensation: {value['compensated_snapshots']} method snapshots; original retention states reused in place")
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
    frozen_sources = {}
    for value in args.hwa_source:
        arm, separator, path = value.partition("=")
        if not separator or not arm or not path or arm in frozen_sources:
            raise ValueError("Expected --hwa-source ARM=PATH once per frozen source.")
        frozen_sources[arm] = Path(path)
    retention_sources = {}
    for value in args.retention_source:
        key, separator, path = value.partition("=")
        if not separator or not key or not path or key in retention_sources:
            raise ValueError("Expected --retention-source ARM:SEED=PATH once per retained array.")
        retention_sources[key] = Path(path)
    lora_bases = {}
    for value in args.lora_base:
        arm, separator, path = value.partition("=")
        if not separator or not arm or not path or arm in lora_bases:
            raise ValueError("Expected --lora-base ARM=PATH once per LoRA arm.")
        lora_bases[arm] = Path(path)
    files = build_plan(
        lifecycle,
        teacher_weights=args.teacher_weights,
        lora_bases=lora_bases,
        execution=execution,
        python=args.python or default_python(),
        source=args.lifecycle.resolve(),
        frozen_sources=frozen_sources,
        retention_sources=retention_sources,
    )
    status = write_plan(files, args.output)
    runs = (args.runs or args.output.resolve().parent / "runs").resolve()
    print(json.dumps({"plan": str(args.output.resolve()), "status": status, "stages": len(files) - 3,
                      "commands": commands(args.output, runs)}, indent=2))
    return 0




def _corpus(args) -> int:
    from core.tensor_io_native import save
    from networks.opt import build_corpus

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


def _task_data(args) -> int:
    """Build a pinned choice-task file and its gold-continuation token stream."""

    from core.tensor_io_native import save
    import pandas
    from transformers import AutoTokenizer

    import networks.tasks as tasks
    from networks.opt import CORPUS_SCHEMA

    root = args.root or (Path(os.environ["EBL_CORPUS_ROOT"]) if os.environ.get("EBL_CORPUS_ROOT") else None)
    if root is None:
        raise ValueError("Expected --root or EBL_CORPUS_ROOT for the task directory.")
    paths = {"task": root.resolve() / f"{args.name}.pt", "stream": root.resolve() / f"{args.name}-stream.pt"}
    if any(path.exists() for path in paths.values()):
        raise ValueError(f"Expected new task names; {paths} already exist.")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    tokenizer.model_max_length = 1 << 62
    files = ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")
    receipt = {
        "class": type(tokenizer).__name__,
        "special_tokens": "none from the tokenizer; BOS (2) prepended to every scored row and stream example",
        "files": {name: sha256_file(args.model_dir / name) for name in files if (args.model_dir / name).exists()},
    }
    records = {name: pandas.read_parquet(path).to_dict("records")
               for name, path in (("train", args.train), ("validation", args.validation))}
    document = tasks.build_task(
        task=args.task, model=args.model, train=records["train"], validation=records["validation"],
        encode=lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        development_examples=args.development_examples, split_seed=args.split_seed,
        sources={name: {"file": path.name, "sha256": sha256_file(path)}
                 for name, path in (("train", args.train), ("validation", args.validation))},
        tokenizer=receipt,
    )
    root.mkdir(parents=True, exist_ok=True)
    save(paths["task"], document)
    stream = {
        "schema": CORPUS_SCHEMA,
        "model": args.model,
        "tokenizer": {**receipt, "special_tokens": "BOS (2) before every example: context + gold continuation"},
        "sources": {"task_file": {"file": paths["task"].name, "sha256": sha256_file(paths["task"])},
                    "splits": {"train": "train", "validation": "development", "test": "test"}},
        "splits": {"train": tasks.gold_stream(document["splits"]["train"]),
                   "validation": tasks.gold_stream(document["splits"]["development"]),
                   "test": tasks.gold_stream(document["splits"]["test"])},
    }
    save(paths["stream"], stream)
    length = args.sequence_length
    print(json.dumps({
        "task_file": {"name": args.name, "sha256": sha256_file(paths["task"]), "path": str(paths["task"])},
        "stream_corpus": {"name": f"{args.name}-stream", "sha256": sha256_file(paths["stream"]),
                          "path": str(paths["stream"])},
        "examples": {name: len(split["labels"]) for name, split in document["splits"].items()},
        "split": document["split"],
        "overlap": document["overlap"],
        "stream_tokens": {name: len(tokens) for name, tokens in stream["splits"].items()},
        f"stream_blocks_{length}": {name: len(tokens) // length for name, tokens in stream["splits"].items()},
    }, indent=2))
    return 0


def build_parser(parser=None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(prog="python -m ebl lifecycle", description=__doc__)
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
    plan.add_argument("--retention-source", action="append", default=[], metavar="ARM:SEED=PATH",
                      help="reuse a completed native retention_states.json for digital compensation")
    plan.add_argument("--hwa-source", action="append", default=[], metavar="ARM=PATH",
                      help="reuse a frozen source artifact and omit its HWA training stage (repeatable)")
    plan.add_argument("--lora-base", action="append", default=[], metavar="ARM=PATH",
                      help="the frozen whole-model HWA base source of a LoRA arm (pinned by the arm)")
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
    task = commands_.add_parser("task-data", help="tokenize a labelled multiple-choice task and its gold stream")
    task.add_argument("--task", required=True, choices=("hellaswag", "sst2", "piqa"))
    task.add_argument("--name", required=True, help="task file name; writes <root>/<name>.pt and <name>-stream.pt")
    task.add_argument("--model", default="facebook/opt-125m", choices=sorted(OPT_MODELS))
    task.add_argument("--model-dir", type=Path, required=True, help="local snapshot with the tokenizer files")
    task.add_argument("--train", type=Path, required=True, help="official train split (parquet)")
    task.add_argument("--validation", type=Path, required=True, help="official validation split (parquet), used as test")
    task.add_argument("--development-examples", type=int, required=True)
    task.add_argument("--split-seed", type=int, required=True)
    task.add_argument("--sequence-length", type=int, default=256, help="only for the printed block counts")
    task.add_argument("--root", type=Path, help="output directory (default: EBL_CORPUS_ROOT)")
    task.set_defaults(handler=_task_data)
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
