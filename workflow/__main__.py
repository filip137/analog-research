"""Lifecycle helper commands: describe, check, plan, collect, corpus and task-data.

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
from workflow.lifecycle import EXPERIMENT_ID, OPT_MODELS, STAGE_KINDS, load_lifecycle, parse_execution, stage_selections
from workflow.plan import build_plan, commands, default_python, write_plan

ROOT = Path(__file__).resolve().parents[1]
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


def _rows(run_dir: Path, result: dict) -> list[dict]:
    artifacts = {item["kind"]: item for item in result["artifacts"]}
    record = artifacts["measurements"]
    path = run_dir / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Expected unchanged measurements in {run_dir}.")
    return _measurement_rows(json.loads(path.read_text()), run_dir)


def _measurement_rows(measurements: list[dict], run_dir: Path) -> list[dict]:
    """Flatten already verified native measurements, locally or after collection."""
    rows = []
    for row in measurements:
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
            # Transfer test cohorts (another pinned corpus), as extra columns.
            **{
                f"{state}_{name}_{k}": row[state][name][k]
                for name in sorted(n for n in row["final"] if n.startswith(("transfer_", "context")))
                for state in ("clean", "initial", "final")
                for k in metrics
            },
            # Labelled task cohorts report their own (multiple-choice) metrics.
            **_task_columns(row, ("clean", "initial", "final")),
            **{k: row[k] for k in ("image_presentations", "sequence_presentations", "input_token_presentations",
                                  "scored_token_presentations", "optimizer_updates") if k in row},
            "failed_devices": row["identity"]["failed_devices"],
            "hwa_convergence_review_required": row["hwa_convergence_review_required"],
            "run_dir": str(run_dir),
        }
        for key, value in sorted(row["cost"].items()):
            flat[f"cost_{key}"] = value
        for key, value in sorted(row.get("digital_adapter", {}).items()):
            flat[f"adapter_{key}"] = value
        rows.append(flat)
    return rows


def _task_columns(row: dict, states, names=None) -> dict:
    values = {state: row[state] for state in states} if names is None else dict(zip(names, states))
    final = next(reversed(values.values()))
    return {
        f"{label}_{name}_{k}": scores[name][k]
        for name in sorted(n for n in final if n.startswith("task_"))
        for label, scores in values.items()
        for k in sorted(k for k, v in final[name].items() if isinstance(v, (int, float)))
    }


def _deployment_rows(run_dir: Path, result: dict) -> list[dict]:
    """One row per assignment and defect case of a deploy stage: clean and deployed metrics."""

    metrics = result["metrics"]
    clean = metrics["clean"]
    split = "test" if "test" in clean else "development"
    rows = []
    for label, case in metrics["cases"].items():
        deployed = case["initial"][split]
        names = sorted(k for k, v in deployed.items() if isinstance(v, (int, float)))
        rows.append(
            {
                "lifecycle_id": metrics["lifecycle_id"],
                "technology": metrics["technology"],
                "hwa_arm": metrics["stage"]["hwa_arm"],
                "array_seed": metrics["array_seed"],
                "case": label,
                **{f"defect_{k}": v for k, v in case.get("case", {}).items()},
                "evaluation": split,
                **{f"clean_{k}": clean[split][k] for k in names},
                **{f"deployed_{k}": deployed[k] for k in names},
                # Transfer test cohorts (another pinned corpus), as extra columns.
                **{
                    f"{state}_{name}_{k}": values[name][k]
                    for name in sorted(n for n in clean if n.startswith("transfer_"))
                    for state, values in (("clean", clean), ("deployed", case["initial"]))
                    for k in names
                },
                **_task_columns(None, (clean, case["initial"]), ("clean", "deployed")),
                "failed_devices": case["identity"]["failed_devices"],
                "out_of_range_devices": case["programming"].get("out_of_range_devices"),
                "hwa_convergence_review_required": metrics["hwa_convergence_review_required"],
                "run_dir": str(run_dir),
            }
        )
    return rows


def _write_rows(rows: list[dict], key_fields: tuple[str, ...], output: Path, name: str) -> int:
    keys = [tuple(row[k] for k in key_fields) for row in rows]
    for row, key in zip(rows, keys):
        # Retries are not replicates: duplicated coverage is flagged, not pooled.
        row["duplicate_coverage"] = keys.count(key) > 1
    (output / f"{name}.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return sum(row["duplicate_coverage"] for row in rows)


def _collect(args) -> int:
    rows, deployments, retention, compensation, recovery, initial = [], [], [], [], [], []
    for root in args.roots:
        for result_path in sorted(Path(root).rglob("result.json")):
            result = json.loads(result_path.read_text())
            if (
                result.get("schema") != "ebl.run"
                or result.get("experiment_id") != EXPERIMENT_ID
                or result.get("status") != "complete"
            ):
                continue
            kind = result["metrics"].get("stage", {}).get("kind")
            if kind == "onchip":
                rows.extend(_rows(result_path.parent, result))
            elif kind == "deploy":
                deployments.extend(_deployment_rows(result_path.parent, result))
            elif kind == "retention":
                artifact = next(a for a in result["artifacts"] if a["kind"] == "retention_measurements")
                path = result_path.parent / artifact["path"]
                if sha256_file(path) != artifact["sha256"]:
                    raise ValueError(f"Changed retention measurements: {path}")
                from workflow.retention import flat_rows
                retention.extend(flat_rows(json.loads(path.read_text()), result_path.parent))
            elif kind == "compensate":
                artifact = next(a for a in result["artifacts"] if a["kind"] == "compensation_measurements")
                path = result_path.parent / artifact["path"]
                if sha256_file(path) != artifact["sha256"]:
                    raise ValueError(f"Changed compensation measurements: {path}")
                from workflow.compensation import flat_rows
                compensation.extend(flat_rows(json.loads(path.read_text()), result_path.parent))
            elif kind == 'initial_programming':
                artifact = next(a for a in result['artifacts'] if a['kind'] == 'initial_programming_measurements')
                path = result_path.parent / artifact['path']
                if sha256_file(path) != artifact['sha256']:
                    raise ValueError(f'Changed initial-programming measurements: {path}')
                from workflow.initial_programming import flat_rows
                initial.extend(flat_rows(json.loads(path.read_text()), result_path.parent))
            elif kind == "drift_recovery":
                artifact = next(a for a in result['artifacts'] if a['kind'] == 'drift_recovery_measurements')
                path = result_path.parent / artifact['path']
                if sha256_file(path) != artifact['sha256']:
                    raise ValueError(f'Changed recovery measurements: {path}')
                from training.drift_recovery_report import flat_rows
                recovery.extend(flat_rows(json.loads(path.read_text()), result_path.parent))
    args.output.mkdir(parents=True, exist_ok=True)
    duplicates = _write_rows(rows, ("lifecycle_id", "hwa_arm", "array_seed", "case", "onchip_arm"), args.output, "rows")
    deployment_duplicates = _write_rows(
        deployments, ("lifecycle_id", "hwa_arm", "array_seed", "case"), args.output, "deployments"
    )
    retention_duplicates = _write_rows(retention, ("lifecycle_id", "hwa_arm", "array_seed", "case", "model", "snapshot"),
                                       args.output, "retention") if retention else 0
    compensation_duplicates = _write_rows(compensation, ("lifecycle_id", "hwa_arm", "array_seed", "case", "model", "snapshot", "method"),
                                          args.output, "compensation") if compensation else 0
    recovery_duplicates = _write_rows(recovery, ('run_dir', 'trajectory', 'seconds', 'phase', 'split'),
                                      args.output, 'drift_recovery') if recovery else 0
    initial_duplicates = _write_rows(initial, ('source_sha256', 'model_sha256', 'cohort_sha256', 'policy', 'realization', 'seconds', 'canary'),
                                     args.output, 'initial_programming') if initial else 0
    print(json.dumps({"rows": len(rows), "duplicates": duplicates, "deployments": len(deployments),
                      "retention_snapshots": len(retention), "retention_duplicates": retention_duplicates,
                      "compensated_snapshots": len(compensation), "compensation_duplicates": compensation_duplicates,
                      "recovery_measurements": len(recovery), "recovery_duplicates": recovery_duplicates,
                      "initial_programming_snapshots": len(initial), "initial_programming_duplicates": initial_duplicates,
                      "deployment_duplicates": deployment_duplicates, "output": str(args.output)}))
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


def _task_data(args) -> int:
    """Build a pinned choice-task file and its gold-continuation token stream."""

    import pandas
    from transformers import AutoTokenizer

    from workflow import tasks
    from workflow.networks.opt_mlp import CORPUS_SCHEMA

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
