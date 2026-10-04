"""Native open-loop recovery from exact V2 P0 or fresh RESET programming."""
from dataclasses import asdict
from pathlib import Path
import json
import math
import os
import time
import gzip
import io
import torch
from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.cifar_crossbar.runtime import load, save, source_model, evaluate
from experiments.cifar_crossbar.model import CrossbarSuffix, tensor_hash
from experiments.cifar_crossbar.devices import make_plant, clone_cpu
from experiments.cifar_crossbar.fault_runtime import cache_context, limited_cache
from experiments.cifar_crossbar.full_epoch_runtime import trajectory, array_identity, restore_endpoint
from experiments.cifar_crossbar.sweep_config import SOURCES, cases
from experiments.cifar_crossbar.sweep_runtime import configure_runtime
from experiments.cifar_crossbar.sweep_devices import combined_om, CompactCheckpoints, expand_checkpoint
from experiments.cifar_crossbar.hwa_fault_runtime import case_name
from experiments.cifar_crossbar.om_open_loop_config import OmOpenLoopSpec
from experiments.cifar_crossbar.open_loop_updates import UncappedOpenLoopAdam, nominal_reset_counts, apply_pulse_counts

ROOT = Path(__file__).resolve().parents[2]


def save_compressed(path, value):
    """Lossless serialization reduces repeated integer/RNG checkpoint storage."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO(); torch.save(clone_cpu(value), stream)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(stream.getvalue(), compresslevel=1, mtime=0))
    temporary.replace(path)


def make_initial_array(student, target, spec, populations, inherited):
    kind, ppm = inherited["kind"], inherited["ppm"]
    population = combined_om(populations, student.layout, spec.dataset, spec.array_seed,
                             spec.endpoint_seed, kind, ppm / 1e6)
    if population["fingerprint"] != inherited["population_fingerprint"]:
        raise ValueError("V2 physical identity changed.")
    array = make_plant(population, spec.endpoint_seed, target.device)
    if spec.initialization == "saved_p0":
        array.engine.load_state_dict(inherited["engine"])
        if tensor_hash(array.read()) != inherited["apparent_sha256"] or tensor_hash(array.engine.persistent) != inherited["persistent_sha256"]:
            raise ValueError("Exact V2 deployment was not restored.")
        programming = {"rule": spec.programming_rule, "source_p0_sha256": inherited["p0_sha256"],
                       "new_programming_pulses": 0, "new_verify_reads": 0,
                       "inherited_initial_programming_pulse_cap": 128}
    else:
        counts = nominal_reset_counts(target, array.nominal_step)
        apply_pulse_counts(array.port(), torch.ones_like(target, dtype=torch.int8), counts)
        programming = {"rule": spec.programming_rule, "pulse_cap": None, "verify_reads": 0,
                       "nominal_step": array.nominal_step, "nominal_reset": -1., "nominal_reference": 0.,
                       "nominal_upper": 1., "requested_upper_asymptote_cells": int((target == 1).sum()),
                       "physical_pulses": int(counts.sum()), "max_cell_pulses": int(counts.max()),
                       "pulse_counts_sha256": tensor_hash(counts)}
    return array, programming


def verify_inherited_metrics(initial, expected):
    if set(initial) != set(expected):
        raise ValueError('Inherited P0 metric fields changed.')
    for metric, value in initial.items():
        equal = (math.isclose(value, expected[metric], abs_tol=2e-5, rel_tol=1e-6)
                 if isinstance(value, (int, float)) else value == expected[metric])
        if not equal:
            raise ValueError('Inherited P0 metric changed: ' + metric)


def screen(student, cache, spec, populations, incoming, store, context):
    rows, replays = [], []
    writer = CompactCheckpoints(store.run_dir, save_fn=save_compressed)
    for name in SOURCES:
        source = incoming["sources"][name]
        student.load_state_dict(source["model"])
        target = student.q.detach().clone()
        clean = evaluate(student, cache["test"], target.device)
        for kind, ppm in cases(spec, name):
            label = case_name(kind, ppm)
            inherited = incoming["p0"][name][label]
            if (inherited["kind"], inherited["ppm"]) != (kind, ppm):
                raise ValueError("Inherited source/fault case mismatch.")
            student.load_state_dict(source["model"])
            array, programming = make_initial_array(student, target, spec, populations, inherited)
            directory = store.run_dir / "checkpoints" / name / label
            p0 = {"context": context, "model": source["model"], "array": array.state_dict(),
                  "target": target, "identity": array_identity(array, spec), "programming": programming}
            writer(directory / "p0.pt", p0)
            initial = evaluate(student, cache["test"], target.device, q=array.read())
            if spec.initialization == "saved_p0" and not spec.max_examples:
                verify_inherited_metrics(initial, inherited["initial_test"])
            for method in spec.methods:
                student.load_state_dict(source["model"]); array.load_state_dict(p0["array"])
                base = {"source": name, "case": label, "array_seed": spec.array_seed, "backend": "om",
                        "suffix": spec.suffix, "initialization": spec.initialization,
                        "p0_sha256": sha256_file(directory / "p0.pt"), "identity": p0["identity"],
                        "inherited_p0_sha256": inherited["p0_sha256"], "clean": clean,
                        "initial_test": initial, "programming": programming}
                result = ({"method": method, "curve": [], "initial_apparent_sha256": tensor_hash(array.read())}
                          if method == "none" else trajectory(student, array, target, cache, spec, method,
                          store, directory, checkpoint_writer=writer, writer_factory=UncappedOpenLoopAdam))
                if method != "none":
                    endpoint = expand_checkpoint(load(directory / f"{method}_epoch5.pt"), store.run_dir)
                    restore_endpoint(student, array, p0, endpoint)
                    replay = evaluate(student, cache["test"], target.device, q=array.read())
                    if replay != result["curve"][-1]["test"]:
                        raise RuntimeError("Exact epoch-five physical-state replay failed.")
                    if result["curve"][-1]["cost"]["verify_reads"] != 0:
                        raise RuntimeError("Open-loop run performed verification reads.")
                    replays.append([name, label, method, 5])
                rows.append({**base, **result})
                atomic_write_json(store.run_dir / "measurements.json", rows)
    expected = sum(len(cases(spec, source)) for source in SOURCES) * len(spec.methods)
    if len(rows) != expected:
        raise RuntimeError("Incomplete open-loop screen.")
    atomic_write_json(store.run_dir / "replay.json", {"exact_replays": len(replays), "checks": replays})
    return {"measurements": rows, "completed_controls": len(rows), "exact_replays": len(replays),
            "source_fits": {name: source["fit"] for name, source in incoming["sources"].items()}}


def run_train(request):
    spec = request.spec
    if not isinstance(spec, OmOpenLoopSpec) or any(getattr(request, k, None) is None for k in
            ("teacher_weights", "weights", "device_model", "device_data")):
        raise ValueError("Expected explicit teacher, V2 source/P0 bundle, population, and feature cache.")
    if any(getattr(request, k, None) is not None for k in ("resume", "base_weights", "device_state")):
        raise ValueError("Unexpected initialization input.")
    inputs = [{"role": key, "path": str(getattr(request, key).resolve()),
               "sha256": sha256_file(getattr(request, key))} for key in
              ("teacher_weights", "weights", "device_model", "device_data")]
    receipt_path = os.environ.get("EBL_SOURCE_RECEIPT")
    if receipt_path:
        receipt = json.loads(Path(receipt_path).read_text())
        if any(sha256_file(ROOT / name) != digest for name, digest in receipt["files"].items()):
            raise ValueError("Frozen numerical source changed.")
        inputs.append({"role": "source_snapshot", "path": receipt_path,
                       "sha256": sha256_file(Path(receipt_path))})
    store = RunStore.create(output_root=request.output_dir, experiment_id=spec.experiment_id,
                            resolved_config=asdict(spec), command=request.command, repo_root=ROOT,
                            input_artifacts=inputs, resume_capability="unsupported")
    started = time.monotonic()
    try:
        runtime_math = configure_runtime(spec); device = torch.device(spec.device)
        teacher, digest = source_model(request.teacher_weights, spec, device)
        student = CrossbarSuffix(teacher, spec.suffix, spec.tile_size)
        expected = {**cache_context(spec, digest, student), "protocol": "cifar_crossbar_fault_sweep.v1", "max_examples": 0}
        incoming, populations, cache = load(request.weights), load(request.device_model), load(request.device_data)
        if incoming.get("schema") != "cifar_om_open_loop.inputs.v1" or incoming["source_context"] != expected:
            raise ValueError("Inherited source context mismatch.")
        if incoming["array_seed"] != spec.array_seed or set(incoming["sources"]) != set(SOURCES):
            raise ValueError("Wrong array or source coverage.")
        if cache["context"] != expected or populations["dataset"] != spec.dataset:
            raise ValueError("Feature cache or population mismatch.")
        for key, size in (("training", 45000), ("development", 1000), ("test", 10000)):
            if len(cache[key]["labels"]) != size:
                raise ValueError("Incomplete inherited cohort.")
            cache[key] = limited_cache(cache[key], spec.max_examples)
            for field in ("features", "teacher_logits"):
                cache[key][field] = cache[key][field].to(device)
        del cache["hwa_raw"]
        context = {**expected, "protocol": spec.experiment_id, "max_examples": spec.max_examples,
                   "initialization": spec.initialization, "update_law": spec.recovery_model}
        atomic_write_json(store.run_dir / "mapping.json", student.mapping_receipt())
        result = screen(student, cache, spec, populations, incoming, store, context)
        result.update(stage=spec.stage, dataset=spec.dataset, backend="om", suffix=spec.suffix,
                      initialization=spec.initialization, runtime_math=runtime_math, epochs=spec.epochs,
                      recovery_images=spec.max_examples or 45000, smoke_only=bool(spec.max_examples),
                      evidence_class="exploratory_model_based", elapsed_seconds=time.monotonic() - started,
                      recovery_pulse_cap=None, per_update_pulse_cap=1, verification_reads=0)
        store.append_metric({"terminal": True, "elapsed_seconds": result["elapsed_seconds"]})
        artifacts = [store.artifact_record(p, kind="checkpoint" if p.suffix == ".pt" else "analysis")
                     for p in store.run_dir.rglob("*") if p.is_file() and
                     (p.suffix == ".pt" or p.name in ("mapping.json", "measurements.json", "replay.json"))]
        store.complete(metrics=result, artifacts=artifacts)
        print(json.dumps({"run_dir": str(store.run_dir), "status": "complete"}), flush=True)
        return 0
    except BaseException as error:
        store.fail(error)
        raise
