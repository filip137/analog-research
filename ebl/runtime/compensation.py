"""Calibrate digital operators on replayed, immutable retention arrays."""

import json
import time
from pathlib import Path

import torch

from core.artifacts import atomic_write_json, content_hash, sha256_file
from core.tensor_identity import state_hash, tensor_hash
from core.tensor_io_native import load, save
import methods.calibration.output as output_calibration
import hardware.devices.cmo as cmo
from hardware.readout import Compensation
from methods.calibration.compensation import fit_readouts, apply_moments
from hardware.drift.inference import RetentionSeries
import networks.data as data_stage
import ebl.runtime.retention as retention
from ebl.schema.lifecycle import parse_stage_config
from networks.opt import evaluate_compensations


def upstream(stage):
    """Verify completed native parent, determining sections and declared files."""
    path = Path(stage.request.device_state)
    root = path.parent
    result = json.loads((root / "result.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    document = json.loads((root / "config.resolved.json").read_text())
    config = parse_stage_config(document)
    if (result.get("schema") != "ebl.run" or result.get("status") != "complete"
            or config.stage.kind != "retention" or config.stage.hwa_arm != stage.selection.hwa_arm
            or config.stage.array_seed != stage.selection.array_seed):
        raise ValueError("Expected a completed native retention stage for the same source and assignment.")
    if manifest["config"]["sha256"] != content_hash(document):
        raise ValueError("Changed parent retention configuration.")
    for name in config.lifecycle.present_sections():
        if name != "question" and config.lifecycle.section_digest(name) != stage.lifecycle.section_digest(name):
            raise ValueError(f"Compensation changed the upstream {name} section.")
    if any(result["metrics"][key] != value for key, value in stage.context.items()):
        raise ValueError("Retention parent differs in teacher or mapping.")
    artifacts = {a["kind"]: a for a in result["artifacts"]}

    def checked(kind):
        entry = artifacts[kind]
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root.resolve()) or sha256_file(target) != entry["sha256"]:
            raise ValueError(f"Changed or external retention artifact: {kind}")
        return target

    if checked("retention_states") != path.resolve():
        raise ValueError("Expected the declared retention state index.")
    measurements = json.loads(checked("retention_measurements").read_text())
    return checked, measurements, result


def fit(network, reference, observed, method, settings):
    return Compensation([fit_readouts(reference[m.offset:m.offset+m.size].reshape(m.shape),
                                     observed[m.offset:m.offset+m.size].reshape(m.shape), method,
                                     tile_size=network.tile_size, iterations=settings.row_column_iterations,
                                     bounds=settings.gain_bounds) for m in network.layout])


@torch.no_grad()
def moments(network, tokens, q, correction, batch_size=8):
    previous = network.drift_compensation
    correction.observe = True
    correction.statistics = [None] * len(network.layout)
    network.drift_compensation = correction
    try:
        for start in range(0, len(tokens), batch_size):
            network(tokens[start:start+batch_size], q=q)
        return correction.moments()
    finally:
        correction.observe = False
        network.drift_compensation = previous


def costs(network, method, count, length, *, identity):
    logical_reads = sum(m.shape[1] * ((m.shape[0] + network.tile_size - 1) // network.tile_size) for m in network.layout)
    tiles = sum(((m.shape[0] + network.tile_size - 1) // network.tile_size)
                * ((m.shape[1] + network.tile_size - 1) // network.tile_size) for m in network.layout)
    outputs, inputs = sum(m.shape[0] for m in network.layout), sum(m.shape[1] for m in network.layout)
    parameters = {"global": len(network.layout), "tile_global": tiles, "row": inputs, "column": outputs,
                  "row_column": inputs + outputs, "global_shift": len(network.layout) + outputs,
                  "global_moments": len(network.layout) + 2 * outputs}[method]
    uses_data = method in ("global_shift", "global_moments") and not identity
    reads = 0 if identity else logical_reads + (count * length * tiles if uses_data else 0)
    reference_values = (network.q.numel() if method == "row_column" else parameters)
    return {"device_writes": 0, "verify_reads": 0, "digital_parameters": parameters,
            "calibration_sequences": count if uses_data else 0,
            "calibration_tokens": count * length if uses_data else 0,
            "one_hot_logical_tile_mvm_per_age": 0 if identity else logical_reads,
            "age_logical_tile_mvm_reads": reads, "age_physical_bank_mvm_reads": 2 * reads,
            "deployment_reference_logical_tile_mvm_reads": logical_reads + (count * length * tiles if method in ("global_shift", "global_moments") else 0),
            "reference_storage_bytes": 4 * reference_values}


@torch.no_grad()
def run_power_law(stage):
    """Calibrate selected held scalar power-law snapshots against the parent's first read.

    The parent's saved first read and epsilon are replayed, never redrawn; every
    aged state and its uncompensated metrics must equal the parent measurement.
    """
    from methods.evaluation import evaluate
    from hardware.drift.power_law import snapshot
    from ebl.schema.source_fit import validate_source_fit

    lc, network, device = stage.lifecycle, stage.network, stage.device
    settings, spec = lc.drift_compensation, lc.retention.power_law
    cache = stage.cache()
    stage.device_bundle()
    source = stage.checked("weights", "hwa_source", hwa_arm=stage.selection.hwa_arm)
    validate_source_fit(lc, lc.hwa.arm(stage.selection.hwa_arm), source["fit"])
    network.load_state_dict(source["model"])
    del source
    network.eval().requires_grad_(False)
    source_sha = sha256_file(stage.request.weights)
    frozen_hash = state_hash(network.state_dict())
    checked, original, parent = upstream(stage)
    if source_sha != parent["metrics"]["source_sha256"]:
        raise ValueError("Expected the exact HWA checkpoint used for the retention parent.")
    saved = load(checked("retention_state.power_law"))
    if (saved["schema"] != "crossbar_lifecycle.power_law.v1" or saved["source_sha256"] != source_sha
            or saved["lifecycle_sha256"] != parent["metrics"]["lifecycle_sha256"]
            or saved["settings"] != lc.retention.to_dict()):
        raise ValueError("Changed power-law replay identity.")
    cohort = saved["cohort"]
    tokens = cache["development"]["tokens"][:cohort["sequences"]]
    identity = torch.load(checked("reference.identity"), map_location="cpu", weights_only=True)
    if (tensor_hash(tokens) != cohort["tokens_sha256"] or not torch.equal(identity["tokens"], tokens.cpu())
            or identity["identity"] != cohort):
        raise ValueError("Expected the parent's evaluation cohort.")
    refs = [torch.load(checked(f"reference.{i:04d}"), map_location="cpu", weights_only=True) for i in range(len(tokens))]
    series = [{**s, "programmed": s["programmed"].to(device), "epsilon": s["epsilon"].to(device)} for s in saved["series"]]
    del saved

    def held(seconds, sigma):
        pieces = []
        for s in series:
            g, _ = snapshot(s["programmed"], s["epsilon"], seconds, sigma, spec, cmo.G_MIN_US, cmo.G_MAX_US)
            pieces.append((g[0] - g[1]) / cmo.HALF_RANGE_US)
        return torch.cat(pieces)

    def native(q):
        return lambda batch: network.forward_features(network.features(batch.to(device)), q)

    index = {(r["seconds"], r["spread_us"]): r for r in original}
    reference = index[0., 0.]
    baseline = held(0., 0.)
    if tensor_hash(baseline) != reference["apparent_sha256"]:
        raise ValueError("First-read replay differs from the original measurement.")
    calibration_tokens = cache["training"]["tokens"][:settings.calibration_sequences]
    calibration = {"corpus_sha256": cache["corpus"]["sha256"], "split": "train",
                   "sequences": len(calibration_tokens), "tokens_sha256": tensor_hash(calibration_tokens),
                   "training_cohort": cache["training"]["cohort"]}
    stage.store.append_metric({"stage": "compensation_source_gate", "passed": True, "source_sha256": source_sha})
    layers, channels = len(network.mlp_norms), network.layout[1].shape[0]
    outputs = [m for m in settings.methods if m in output_calibration.METHODS]
    matrices = [m for m in settings.methods if m not in output_calibration.METHODS]
    artifacts, state_index, rows = [], [], []
    prefix = reference_statistics = None
    if outputs:
        prefix = [network.analog_input(b.to(device)) for b in calibration_tokens.split(8)]
        first = tokens[:1].to(device)
        decomposed = network.head(output_calibration.forward(network.analog_input(first), network.mlp_branch, layers, baseline))
        if not torch.equal(decomposed, native(baseline)(first)):
            raise RuntimeError("Decomposed MLP-output forward differs from the native forward.")
        reference_statistics = output_calibration.propagate(prefix, network.mlp_branch, layers, baseline)
        path = stage.run_dir / "checkpoints/output-reference-statistics.pt"
        save(path, {"schema": "drift.output_calibration.reference.v1", "apparent_sha256": reference["apparent_sha256"],
                    "calibration": calibration, "statistics": [s.to_dict() for s in reference_statistics]})
        artifacts.append(("output_calibration_reference", path))
    reference_moments = None
    if set(matrices) & {"global_shift", "global_moments"}:
        reference_moments = moments(network, calibration_tokens, baseline, fit(network, baseline, baseline, "global", settings))
    grid = settings.snapshots(lc.retention, "reram_cmo")
    expected = len(grid) * len(settings.methods)
    for seconds, sigma in grid:
        started = time.monotonic()
        prior = index[seconds, sigma]
        observed = held(seconds, sigma)
        apparent_hash = tensor_hash(observed)
        if apparent_hash != prior["apparent_sha256"]:
            raise ValueError("Aged replay differs from the original measurement.")
        control = evaluate(tokens, native(observed), refs, device)
        if control != prior["metrics"]["development"]:
            raise ValueError("Uncompensated control no longer exactly reproduces the original evaluation.")
        tag = f"t{seconds:g}s-s{sigma:g}uS"
        measured, paths, costs_by_method = {}, {}, {}
        current_moments = (moments(network, calibration_tokens, observed, fit(network, baseline, observed, "global", settings))
                           if reference_moments is not None else None)
        for method in matrices:
            correction = fit(network, baseline, observed, method, settings)
            if method in ("global_shift", "global_moments"):
                apply_moments(correction, reference_moments, current_moments,
                              shift_only=method == "global_shift", bounds=settings.gain_bounds)
            path = stage.run_dir / f"checkpoints/{tag}-{method}.pt"
            save(path, {"schema": "crossbar_lifecycle.drift_compensation.v1", "context": stage.context,
                        "source_sha256": source_sha, "apparent_sha256": apparent_hash, "method": method,
                        "reference_apparent_sha256": reference["apparent_sha256"], "calibration": calibration,
                        "settings": settings.to_dict(), "matrices": correction.state_dict()})
            restored = Compensation.from_state_dict(load(path)["matrices"], device)
            network.drift_compensation = restored
            try:
                measured[method] = evaluate(tokens, native(observed), refs, device)
            finally:
                network.drift_compensation = None
            paths[method] = path
            costs_by_method[method] = costs(network, method, len(calibration_tokens), calibration_tokens.shape[1], identity=False)
        for method in outputs:
            fitted = output_calibration.propagate(prefix, network.mlp_branch, layers, observed, method=method,
                                                  reference=reference_statistics, bounds=settings.gain_bounds)
            path = stage.run_dir / f"checkpoints/{tag}-{method}.pt"
            save(path, {"schema": "drift.output_calibration.v1", "context": stage.context, "source_sha256": source_sha,
                        "apparent_sha256": apparent_hash, "method": method,
                        "reference_apparent_sha256": reference["apparent_sha256"], "calibration": calibration,
                        "settings": settings.to_dict(), "corrections": [c.state_dict() for c in fitted]})
            restored = [output_calibration.OutputCorrection.from_state_dict(s, device) for s in load(path)["corrections"]]
            if any(not (torch.equal(a.gain, b.gain) and torch.equal(a.shift, b.shift)) for a, b in zip(fitted, restored)):
                raise RuntimeError("Calibration parameters did not replay exactly.")
            measured[method] = evaluate(tokens, lambda batch: network.head(output_calibration.forward(
                network.analog_input(batch.to(device)), network.mlp_branch, layers, observed, restored)), refs, device)
            paths[method] = path
            costs_by_method[method] = output_calibration.costs(method, len(calibration_tokens), calibration_tokens.shape[1], layers, channels)
        if tensor_hash(observed) != apparent_hash:
            raise RuntimeError("Digital compensation mutated the held apparent state.")
        elapsed = time.monotonic() - started
        for method in settings.methods:
            row = {"lifecycle_id": lc.lifecycle_id, "hwa_arm": stage.selection.hwa_arm, "source_sha256": source_sha,
                   "array_seed": stage.selection.array_seed, "case": "nominal", "model": "reram_cmo", "method": method,
                   "seconds": seconds, "spread_us": sigma, "mean_us": spec.mean_us, "mean_exponent": spec.mean_exponent,
                   "spread_exponent": spec.spread_exponent, "snapshot": prior["snapshot"], "baseline_seconds": 0,
                   "logical_weights": prior["logical_weights"], "physical_devices": prior["physical_devices"],
                   "failed_devices": 0, "cohort": cohort, "clean": prior["clean"], "baseline": reference["metrics"],
                   "uncompensated": prior["metrics"], "compensated": {"development": measured[method]},
                   "perplexity_ratio": measured[method]["perplexity"] / reference["metrics"]["development"]["perplexity"],
                   "uncompensated_perplexity_ratio": prior["perplexity_ratio"],
                   "apparent_sha256": apparent_hash, "state_unchanged": True, "replay_exact": True, "control_exact": True,
                   "identity_snapshot": False, "read_noise": False, "calibration": calibration,
                   "drift_statistics": prior["drift_statistics"], "cost": costs_by_method[method],
                   "shared_age_evaluation_seconds": elapsed}
            rows.append(row)
            kind = f"compensation_state.{tag}.{method}"
            artifacts.append((kind, paths[method]))
            state_index.append({"seconds": seconds, "spread_us": sigma, "method": method, "artifact_kind": kind})
            stage.store.append_metric({"stage": "compensate", "seconds": seconds, "spread_us": sigma, "method": method,
                                       "complete": len(rows), "expected": expected, "metrics": measured[method]})
        atomic_write_json(stage.run_dir / "compensation_measurements.json", rows)
        print(json.dumps({"snapshot": tag, "uncompensated": prior["perplexity_ratio"],
                          **{m: rows[-len(settings.methods) + i]["perplexity_ratio"] for i, m in enumerate(settings.methods)}}),
              flush=True)
        del observed
    if state_hash(network.state_dict()) != frozen_hash:
        raise RuntimeError("Digital compensation changed frozen source parameters.")
    atomic_write_json(stage.run_dir / "compensation_measurements.json", rows)
    atomic_write_json(stage.run_dir / "compensation_states.json", state_index)
    return {"snapshots": len(rows), "expected_snapshots": expected, "source_sha256": source_sha,
            "source_unchanged": True, "apparent_states_unchanged": True, "control_exact": True,
            "calibration": calibration, "cohort": cohort, "additional_writes": 0,
            "selection": "all predeclared methods; no selection"}, artifacts + [
                ("compensation_measurements", stage.run_dir / "compensation_measurements.json"),
                ("compensation_states", stage.run_dir / "compensation_states.json")]


@torch.no_grad()
def run(stage):
    from ebl.schema.source_fit import validate_source_fit

    if stage.lifecycle.retention.power_law is not None:
        return run_power_law(stage)
    lc, network, device = stage.lifecycle, stage.network, stage.device
    settings = lc.drift_compensation
    cache = stage.cache()
    stage.device_bundle()
    source = stage.checked("weights", "hwa_source", hwa_arm=stage.selection.hwa_arm)
    validate_source_fit(lc, lc.hwa.arm(stage.selection.hwa_arm), source["fit"])
    network.load_state_dict(source["model"])
    del source
    source_sha = sha256_file(stage.request.weights)
    frozen_hash = state_hash(network.state_dict())
    checked, original, parent = upstream(stage)
    if source_sha != parent["metrics"]["source_sha256"]:
        raise ValueError("Expected the exact HWA checkpoint used for the retention parent.")
    splits = list(original[0]["metrics"])
    expected_splits = ["development"] + (["test"] if data_stage.evaluation_split(lc) == "test" else []) + data_stage.transfer_splits(lc)
    if set(splits) != set(expected_splits):
        raise ValueError("Expected unchanged evaluation cohorts.")
    tokens = cache["training"]["tokens"][:settings.calibration_sequences]
    calibration = {"corpus_sha256": cache["corpus"]["sha256"], "split": "train",
                   "sequences": len(tokens), "tokens_sha256": tensor_hash(tokens),
                   "training_cohort": cache["training"]["cohort"]}
    stage.store.append_metric({"stage": "compensation_source_gate", "passed": True, "source_sha256": source_sha})
    index = {(r["case"], r["model"], r["seconds"], r.get("noise_scale", 1.0)): r for r in original}
    rows, artifacts, state_index = [], [], []
    expected = (len(lc.retention.case_labels) * len(settings.methods)
                * sum(len(lc.retention.snapshots(model)) for model in lc.retention.models))
    control_checked = False
    for case in lc.retention.case_labels:
        for model in lc.retention.models:
            raw_path = checked(f"retention_state.{case}.{model}")
            saved = load(raw_path)
            if (saved["source_sha256"] != source_sha or saved["context"] != stage.context
                    or saved["array_seed"] != stage.selection.array_seed or saved["model"] != model
                    or saved["lifecycle_sha256"] != parent["metrics"]["lifecycle_sha256"]):
                raise ValueError("Changed retention replay identity.")
            traces = [RetentionSeries.from_state_dict(s, device) for s in saved["series"]]
            del saved
            baseline = retention.apparent(traces, model, None)
            reference = index[case, model, None, 1.0]
            if tensor_hash(baseline) != reference["apparent_sha256"]:
                raise ValueError("Baseline replay differs from the original measurement.")
            if not control_checked:
                control = {name: stage.family.evaluate(network, cache[name], device, q=baseline) for name in splits}
                if control != reference["metrics"]:
                    raise ValueError("Uncompensated control no longer exactly reproduces the original evaluation.")
                control_checked = True
            needs_moments = bool(set(settings.methods) & {"global_shift", "global_moments"})
            reference_moments = moments(network, tokens, baseline, fit(network, baseline, baseline, "global", settings)) if needs_moments else None
            for seconds, scale in lc.retention.snapshots(model):
                observed = retention.apparent(traces, model, seconds, scale)
                prior = index[case, model, seconds, scale]
                tag = f"t{seconds:g}" + ("" if scale == 1 else f"-x{scale:g}")
                apparent_hash = tensor_hash(observed)
                if apparent_hash != prior["apparent_sha256"]:
                    raise ValueError("Aged replay differs from the original measurement.")
                identity = torch.equal(observed, baseline)
                current_moments = (moments(network, tokens, observed, fit(network, baseline, observed, "global", settings))
                                   if needs_moments and not identity else reference_moments)
                started = time.monotonic()
                corrections, paths = {}, {}
                for method in settings.methods:
                    correction = fit(network, baseline, observed, "global" if identity else method, settings)
                    if method in ("global_shift", "global_moments") and not identity:
                        apply_moments(correction, reference_moments, current_moments,
                                      shift_only=method == "global_shift", bounds=settings.gain_bounds)
                    path = stage.run_dir / f"checkpoints/{case}-{model}-{tag}-{method}.pt"
                    save(path, {"schema": "crossbar_lifecycle.drift_compensation.v1", "context": stage.context,
                                "source_sha256": source_sha, "apparent_sha256": apparent_hash, "method": method,
                                "reference_apparent_sha256": reference["apparent_sha256"], "calibration": calibration,
                                "settings": settings.to_dict(), "matrices": correction.state_dict()})
                    restored = Compensation.from_state_dict(load(path)["matrices"], device)
                    if state_hash({f"{i}/{k}": v for i, s in enumerate(correction.state_dict()) for k, v in s.items() if isinstance(v, torch.Tensor)}) != state_hash(
                            {f"{i}/{k}": v for i, s in enumerate(restored.state_dict()) for k, v in s.items() if isinstance(v, torch.Tensor)}):
                        raise RuntimeError("Calibration parameters did not replay exactly.")
                    corrections[method], paths[method] = restored, path
                measured = {} if identity else {name: evaluate_compensations(network, cache[name], device, q=observed,
                                                                             corrections=corrections) for name in splits}
                if tensor_hash(observed) != apparent_hash:
                    raise RuntimeError("Digital compensation mutated the held apparent state.")
                for method in settings.methods:
                    metrics = prior["metrics"] if identity else {name: measured[name][method] for name in splits}
                    row = {"lifecycle_id": lc.lifecycle_id, "hwa_arm": stage.selection.hwa_arm,
                           "source_sha256": source_sha, "array_seed": stage.selection.array_seed,
                           "case": case, "model": model, "method": method, "seconds": seconds, "noise_scale": scale,
                           "snapshot": prior["snapshot"], "baseline_seconds": prior["baseline_seconds"],
                           "logical_weights": prior["logical_weights"], "physical_devices": prior["physical_devices"],
                           "failed_devices": prior["failed_devices"], "fault_identity": prior["fault_identity"],
                           "baseline": reference["metrics"], "uncompensated": prior["metrics"], "compensated": metrics,
                           "apparent_sha256": apparent_hash, "state_unchanged": True, "replay_exact": True,
                           "identity_snapshot": identity, "read_noise": False, "calibration": calibration,
                           "cost": costs(network, method, len(tokens), tokens.shape[1], identity=identity),
                           "shared_age_evaluation_seconds": time.monotonic() - started}
                    rows.append(row)
                    kind = f"compensation_state.{case}.{model}.{tag}.{method}"
                    artifacts.append((kind, paths[method]))
                    state_index.append({"case": case, "model": model, "seconds": seconds, "noise_scale": scale,
                                        "method": method, "artifact_kind": kind})
                    stage.store.append_metric({"stage": "compensate", "case": case, "model": model, "seconds": seconds,
                                               "noise_scale": scale,
                                               "method": method, "complete": len(rows), "expected": expected,
                                               "metrics": metrics})
                del observed
            del traces, baseline
    if state_hash(network.state_dict()) != frozen_hash:
        raise RuntimeError("Digital compensation changed frozen source parameters.")
    atomic_write_json(stage.run_dir / "compensation_measurements.json", rows)
    atomic_write_json(stage.run_dir / "compensation_states.json", state_index)
    return {"snapshots": len(rows), "expected_snapshots": expected, "source_sha256": source_sha,
            "source_unchanged": True, "apparent_states_unchanged": True, "control_exact": control_checked,
            "calibration": calibration, "additional_writes": 0, "selection": "all predeclared methods; no selection"}, artifacts + [
                ("compensation_measurements", stage.run_dir / "compensation_measurements.json"),
                ("compensation_states", stage.run_dir / "compensation_states.json")]
