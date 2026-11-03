"""Native inference-only retention sweep of frozen HWA sources.

The source lifecycle remains unchanged. This stage maps its normalized digital
weights to each explicitly selected device range, programs once, and scores
held age snapshots. CMO hierarchical fault locations are shared across models;
random stuck conductance quantiles are mapped into each model's own range.
"""

from __future__ import annotations

import json
import time

import torch

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.model import state_hash, tensor_hash
from experiments.cifar_crossbar.runtime import load, save
from experiments.cifar_crossbar.sweep_devices import binding_seed
from training import reram_cmo as cmo
from training.inference_retention import RetentionSeries, wan_snapshot
from workflow import data as data_stage, deployment, devices

MODELS = {
    "pcm": {"class": "PCMLikeNoiseModel", "g_min": 0.0, "g_max": 25.0,
            "encoding": "differential_pair", "normalizer": 25.0},
    "reram_cmo": {"class": "ReRamCMONoiseModel", "g_min": cmo.G_MIN_US, "g_max": cmo.G_MAX_US,
                  "encoding": "active_reference_pair", "normalizer": cmo.HALF_RANGE_US},
    "reram_wan2022": {"class": "ReRamWan2022NoiseModel", "g_min": 0.0, "g_max": 40.0,
                      "encoding": "differential_pair", "normalizer": 40.0},
}


def targets(q, model):
    if not bool(torch.isfinite(q).all()) or bool((q.abs() > 1).any()):
        raise ValueError("Expected finite normalized frozen weights in [-1, 1].")
    if model == "reram_cmo":
        return torch.stack((cmo.to_conductance(q), torch.full_like(q, cmo.MID_US)))
    gmax = MODELS[model]["g_max"]
    return torch.stack((q.clamp_min(0), (-q).clamp_min(0))) * gmax


def mapped_faults(mask, stuck, case, model, device):
    """Keep device locations and uniform quantiles matched, not raw CMO uS."""
    mask, stuck = mask.to(device), stuck.to(device)
    spec = MODELS[model]
    if model == "reram_cmo":
        return mask, stuck
    if case.kind == "random":
        quantile = (stuck - cmo.G_MIN_US) / (cmo.G_MAX_US - cmo.G_MIN_US)
        mapped = spec["g_min"] + quantile * (spec["g_max"] - spec["g_min"])
    else:
        value = spec["g_max"] if case.kind == "gmax" else spec["g_min"]
        mapped = torch.full_like(stuck, value)
    return mask, torch.where(mask, mapped, 0.0)


@torch.no_grad()
def program(target, model, generator, *, acceptance, noise_scale):
    """One native programming draw; Wan's programming fit is its 1 s fit."""
    if model == "reram_wan2022":
        return wan_snapshot(target, generator, 1, g_max=40, noise_scale=noise_scale)
    noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
    if model == "reram_cmo":
        return target + noise_scale * cmo.programming_sigma(target, acceptance) * noise
    relative = target / 25.0
    sigma = 0.26348 + relative * 1.9650 + relative.square() * -1.1731
    return (target + noise_scale * sigma * noise).clamp_min(0)


def make_series(lifecycle, layout, q, case, fault_maps, model, assignment, device):
    if lifecycle.devices.noise_scale != 1:
        raise ValueError("Retention comparison requires the nominal programming-noise scale.")
    namespace = lifecycle.network.binding_namespace
    series, stream_seeds = [], {}
    for matrix in layout:
        target = targets(q[matrix.offset:matrix.offset + matrix.size], model)
        mask, stuck = mapped_faults(*fault_maps["matrices"][matrix.name], case, model, device)
        endpoint_seed = binding_seed(lifecycle.devices.endpoint_seed(assignment), namespace, matrix.name)
        age_seed = binding_seed(lifecycle.retention.seed, namespace, f"{assignment}/{matrix.name}")
        programmed = program(target, model, torch.Generator(device=device).manual_seed(endpoint_seed),
                             acceptance=lifecycle.program_verify.acceptance_percent, noise_scale=lifecycle.devices.noise_scale)
        programmed = torch.where(mask, stuck, programmed)
        trace = RetentionSeries(model, target, programmed, seed=age_seed,
                                read_noise_scale=int(lifecycle.retention.read_noise and model != "reram_wan2022"),
                                fault_mask=mask, stuck=stuck)
        series.append(trace)
        stream_seeds[matrix.name] = {"programming": endpoint_seed, "retention": age_seed}
    return series, stream_seeds


def label(seconds, scale=1.0):
    """Snapshot label; unscaled labels are unchanged from earlier bundles."""
    if seconds is None:
        return "programmed"
    return f"t{seconds:g}s" if scale == 1 else f"t{seconds:g}s-x{scale:g}"


def apparent(series, model, seconds, scale=1.0):
    pieces = []
    for trace in series:
        # Reuse the actual 1 s programming realization as Wan's first snapshot.
        g = (trace.baseline() if seconds is None or (model == "reram_wan2022" and seconds == 1)
             else trace.snapshot(seconds, scale))
        if not torch.equal(g[trace.mask], trace.stuck[trace.mask]):
            raise RuntimeError("Retention changed a permanent defect.")
        pieces.append((g[0] - g[1]) / MODELS[model]["normalizer"])
    return torch.cat(pieces)


@torch.no_grad()
def run(stage):
    if stage.lifecycle.retention.power_law is not None:
        return run_power_law(stage)
    from workflow.runtime import validate_source_fit

    lifecycle, network, device = stage.lifecycle, stage.network, stage.device
    settings, arm, assignment = lifecycle.retention, stage.selection.hwa_arm, stage.selection.array_seed
    cache = stage.cache()
    stage.device_bundle()
    source = stage.checked("weights", "hwa_source", hwa_arm=arm)
    validate_source_fit(lifecycle, lifecycle.hwa.arm(arm), source["fit"])
    if any(not bool(torch.isfinite(t).all()) for t in source["model"].values()):
        raise ValueError("Expected a finite frozen HWA source.")
    network.load_state_dict(source["model"])
    del source
    source_hash = sha256_file(stage.request.weights)
    frozen_hash = state_hash(network.state_dict())
    q = deployment.targets(network)
    splits = ["development"] + (["test"] if data_stage.evaluation_split(lifecycle) == "test" else [])
    splits += list(data_stage.transfer_splits(lifecycle))
    clean = {name: stage.family.evaluate(network, cache[name], device) for name in splits}
    stage.store.append_metric({"stage": "retention_source_gate", "source_sha256": source_hash, "passed": True})
    rows, states, artifacts = [], [], []
    cases = [case for case in lifecycle.defects.cases if case.label in settings.case_labels]
    expected = len(cases) * sum(1 + len(settings.snapshots(model)) for model in settings.models)
    for case in cases:
        if case.kind == "none":
            fault_maps = {"receipt": {"assignment_seed": assignment, "failed_devices": 0},
                          "matrices": {m.name: (torch.zeros((2, m.size), dtype=torch.bool),
                                                torch.zeros((2, m.size))) for m in network.layout}}
        else:
            fault_maps = devices.hierarchical_faults(network.layout, lifecycle.deployment.tile_size,
                                                    lifecycle.defects.arrays_per_die, case, assignment,
                                                    lifecycle.network.binding_namespace)
        for model in settings.models:
            series, seeds = make_series(lifecycle, network.layout, q, case, fault_maps, model, assignment, device)
            baseline = apparent(series, model, None)
            path = stage.run_dir / f"checkpoints/retention-{case.label}-{model}.pt"
            save(path, {"schema": "crossbar_lifecycle.retention.v1", "lifecycle_sha256": lifecycle.digest(),
                        "source_sha256": source_hash, "context": stage.context, "array_seed": assignment,
                        "case": case.to_dict(), "model": model, "mapping": MODELS[model],
                        "seeds": seeds, "torch_version": str(torch.__version__),
                        "matrices": [m.name for m in network.layout], "series": [s.state_dict() for s in series]})
            del series
            restored = load(path)
            series = [RetentionSeries.from_state_dict(s, device) for s in restored["series"]]
            del restored
            if not torch.equal(baseline, apparent(series, model, None)):
                raise RuntimeError("Saved retention baseline did not replay exactly.")
            state_id = f"retention_state.{case.label}.{model}"
            artifacts.append((state_id, path))
            states.append({"case": case.label, "model": model, "artifact_kind": state_id,
                           "programmed_apparent_sha256": tensor_hash(baseline)})
            baseline_metrics = None
            for seconds, scale in [(None, 1.0)] + settings.snapshots(model):
                started = time.monotonic()
                observed = baseline if seconds is None else apparent(series, model, seconds, scale)
                digest = tensor_hash(observed)
                if not torch.equal(observed, apparent(series, model, seconds, scale)):
                    raise RuntimeError("Retention snapshot did not replay exactly.")
                if model == "reram_wan2022" and seconds == 1:
                    metrics = baseline_metrics
                else:
                    metrics = {name: stage.family.evaluate(network, cache[name], device, q=observed) for name in splits}
                if seconds is None:
                    baseline_metrics = metrics
                row = {"lifecycle_id": lifecycle.lifecycle_id, "hwa_arm": arm, "source_sha256": source_hash,
                       "array_seed": assignment, "case": case.label, "defect": case.to_dict(), "model": model,
                       "model_class": MODELS[model]["class"], "mapping": MODELS[model],
                       "logical_weights": q.numel(), "physical_devices": 2 * q.numel(),
                       "failed_devices": fault_maps["receipt"]["failed_devices"],
                       "fault_identity": fault_maps["receipt"], "read_noise": bool(settings.read_noise and model != "reram_wan2022"),
                       "snapshot": label(seconds, scale), "seconds": seconds, "noise_scale": scale,
                       "baseline_seconds": 1 if model == "reram_wan2022" else 0,
                       "clean": clean, "baseline": baseline_metrics, "metrics": metrics,
                       "apparent_sha256": digest, "replay_exact": True,
                       "residual_rms": float((observed - q).square().mean().sqrt()),
                       "change_from_programmed_rms": float((observed - baseline).square().mean().sqrt()),
                       "cost": {"retention_device_writes": 0, "retention_verify_reads": 0,
                                "initial_device_programs": 2 * q.numel() - fault_maps["receipt"]["failed_devices"]},
                       "elapsed_seconds": time.monotonic() - started}
                rows.append(row)
                stage.store.append_metric({"stage": "retention", "case": case.label, "model": model,
                                           "snapshot": row["snapshot"], "snapshots_complete": len(rows),
                                           "snapshots_expected": expected, "metrics": metrics})
                print(json.dumps({"model": model, "case": case.label, "snapshot": row["snapshot"],
                                  "teacher_kl": metrics["development"]["teacher_kl"]}), flush=True)
            del series, baseline, observed
        del fault_maps
    if state_hash(network.state_dict()) != frozen_hash:
        raise RuntimeError("Retention evaluation changed the frozen source.")
    atomic_write_json(stage.run_dir / "retention_measurements.json", rows)
    atomic_write_json(stage.run_dir / "retention_states.json", states)
    result = {"source_sha256": source_hash, "source_model_sha256": frozen_hash, "source_unchanged": True,
              "array_seed": assignment, "snapshots": len(rows), "expected_snapshots": expected,
              "exact_replays": len(rows), "clean": clean, "retention": settings.to_dict(),
              "additional_writes": 0, "compensation": False, "selection": "frozen HWA source; no selection"}
    return result, artifacts + [("retention_measurements", stage.run_dir / "retention_measurements.json"),
                                ("retention_states", stage.run_dir / "retention_states.json")]


def flat_rows(rows, run_dir):
    result = []
    for row in rows:
        flat = {key: row[key] for key in ("lifecycle_id", "hwa_arm", "source_sha256", "array_seed", "case", "model",
                "snapshot", "seconds", "baseline_seconds", "read_noise", "logical_weights", "physical_devices",
                "failed_devices", "apparent_sha256", "replay_exact", "residual_rms", "change_from_programmed_rms")}
        flat["noise_scale"] = row.get("noise_scale", 1.0)
        for key in ("spread_us", "mean_us", "mean_exponent", "spread_exponent", "perplexity_ratio"):
            if key in row:
                flat[key] = row[key]
        for state, values in (("clean", row["clean"]), ("baseline", row["baseline"]), ("retained", row["metrics"])):
            flat.update({f"{state}_{split}_{name}": value for split, metrics in values.items()
                         for name, value in metrics.items() if isinstance(value, (int, float))})
        flat.update({f"cost_{k}": v for k, v in row["cost"].items()})
        flat["run_dir"] = str(run_dir)
        result.append(flat)
    return result


@torch.no_grad()
def run_power_law(stage):
    """Scalar drift on a bounded, explicitly saved first-read CMO state."""
    from training.power_law_drift import snapshot, aggregate_statistics
    from training.drift_evaluation import reference_cache, evaluate
    from workflow.runtime import validate_source_fit

    lc, network, device = stage.lifecycle, stage.network, stage.device
    settings, arm, assignment = lc.retention, stage.selection.hwa_arm, stage.selection.array_seed
    spec = settings.power_law
    cache = stage.cache()
    stage.device_bundle()
    source = stage.checked("weights", "hwa_source", hwa_arm=arm)
    validate_source_fit(lc, lc.hwa.arm(arm), source['fit'])
    network.load_state_dict(source['model'])
    del source
    network.eval().requires_grad_(False)
    frozen_hash = state_hash(network.state_dict())
    source_hash = sha256_file(stage.request.weights)
    all_tokens = cache['development']['tokens']
    if spec.development_sequences > len(all_tokens):
        raise ValueError('Drift cohort exceeds the pinned development cohort.')
    tokens = all_tokens[:spec.development_sequences]
    cohort = {'split': 'validation', 'first_sequence': 0, 'sequences': len(tokens),
              'tokens_sha256': tensor_hash(tokens), 'source_sha256': source_hash,
              'corpus': {'name': lc.data.corpus, 'sha256': lc.data.corpus_sha256}}
    def forward(batch, q=None):
        return network.forward_features(network.features(batch.to(device)), q)
    def teacher(batch):
        return network.teacher_logits(network.features(batch.to(device)))
    ref_dir = stage.run_dir / 'reference_logits'
    refs = reference_cache(tokens, forward, teacher, ref_dir, cohort)
    clean = evaluate(tokens, forward, refs, device)
    q = deployment.targets(network)
    series, raw_q, initial_counts = [], [], []
    for matrix in network.layout:
        target = targets(q[matrix.offset:matrix.offset + matrix.size], 'reram_cmo')
        endpoint_seed = binding_seed(lc.devices.endpoint_seed(assignment), lc.network.binding_namespace, matrix.name)
        drift_seed = binding_seed(settings.seed, lc.network.binding_namespace, f'{assignment}/{matrix.name}')
        raw = program(target, 'reram_cmo', torch.Generator(device=device).manual_seed(endpoint_seed),
                      acceptance=lc.program_verify.acceptance_percent, noise_scale=lc.devices.noise_scale)
        gp = raw.clamp(cmo.G_MIN_US, cmo.G_MAX_US)
        eps = torch.randn(gp.shape, device=device, generator=torch.Generator(device=device).manual_seed(drift_seed))
        series.append({'name': matrix.name, 'programmed': gp.cpu(), 'epsilon': eps.cpu(),
                       'programming_seed': endpoint_seed, 'drift_seed': drift_seed,
                       'raw_sha256': tensor_hash(raw)})
        raw_q.append((raw[0] - raw[1]) / cmo.HALF_RANGE_US)
        initial_counts.append(int((raw != gp).sum()))
    raw_metrics = evaluate(tokens, lambda batch: forward(batch, torch.cat(raw_q)), refs, device)
    del raw_q, raw, gp, eps, target
    state_path = stage.run_dir / 'checkpoints/power-law.pt'
    save(state_path, {'schema': 'crossbar_lifecycle.power_law.v1', 'source_sha256': source_hash,
                     'lifecycle_sha256': lc.digest(), 'cohort': cohort, 'settings': settings.to_dict(),
                     'first_read_clipped_devices': sum(initial_counts), 'series': series})
    del series
    restored = load(state_path)
    series = [{**s, 'programmed': s['programmed'].to(device), 'epsilon': s['epsilon'].to(device)}
              for s in restored['series']]
    del restored
    baseline = torch.cat([(s['programmed'][0] - s['programmed'][1]) / cmo.HALF_RANGE_US for s in series])
    rows, baseline_metrics = [], None
    for seconds, sigma in [(0., 0.)] + settings.snapshots('reram_cmo'):
        started = time.monotonic()
        pieces, stats = [], []
        for s in series:
            held, info = snapshot(s['programmed'], s['epsilon'], seconds, sigma, spec, cmo.G_MIN_US, cmo.G_MAX_US)
            replay, _ = snapshot(s['programmed'], s['epsilon'], seconds, sigma, spec, cmo.G_MIN_US, cmo.G_MAX_US)
            if not torch.equal(held, replay):
                raise RuntimeError('Drift snapshot did not replay exactly.')
            pieces.append((held[0] - held[1]) / cmo.HALF_RANGE_US)
            stats.append(info)
        observed = torch.cat(pieces)
        if seconds == 0 and not torch.equal(observed, baseline):
            raise RuntimeError('Zero-time drift changed the first-read state.')
        metrics = evaluate(tokens, lambda batch: forward(batch, observed), refs, device)
        if seconds == 0:
            baseline_metrics = metrics
        row = {'lifecycle_id': lc.lifecycle_id, 'hwa_arm': arm, 'source_sha256': source_hash,
               'array_seed': assignment, 'case': 'nominal', 'model': 'reram_cmo',
               'model_class': 'synthetic_scalar_power_law', 'mapping': MODELS['reram_cmo'],
               'snapshot': 'programmed' if seconds == 0 else f't{seconds:g}s-s{sigma:g}uS',
               'seconds': seconds, 'baseline_seconds': 0, 'spread_us': sigma, 'mean_us': spec.mean_us,
               'mean_exponent': spec.mean_exponent, 'spread_exponent': spec.spread_exponent,
               'read_noise': False, 'logical_weights': q.numel(), 'physical_devices': 2 * q.numel(),
               'failed_devices': 0, 'cohort': cohort, 'clean': {'development': clean},
               'baseline': {'development': baseline_metrics}, 'metrics': {'development': metrics},
               'raw_programmed': raw_metrics, 'first_read_clipped_devices': sum(initial_counts),
               'perplexity_ratio': metrics['perplexity'] / baseline_metrics['perplexity'],
               'apparent_sha256': tensor_hash(observed), 'replay_exact': True,
               'residual_rms': float((observed - q).square().mean().sqrt()),
               'change_from_programmed_rms': float((observed - baseline).square().mean().sqrt()),
               'drift_statistics': aggregate_statistics(stats),
               'cost': {'retention_device_writes': 0, 'retention_verify_reads': 0,
                        'initial_device_programs': 2 * q.numel()},
               'elapsed_seconds': time.monotonic() - started}
        rows.append(row)
        stage.store.append_metric({'stage': 'retention', 'snapshots_complete': len(rows),
                                   'snapshots_expected': 1 + len(settings.snapshots('reram_cmo')), **row})
        atomic_write_json(stage.run_dir / 'retention_measurements.json', rows)
        print(json.dumps({'snapshot': row['snapshot'], 'perplexity_ratio': row['perplexity_ratio']}), flush=True)
    if state_hash(network.state_dict()) != frozen_hash:
        raise RuntimeError('Drift changed the frozen model.')
    atomic_write_json(stage.run_dir / 'retention_states.json', [{
        'case': 'nominal', 'model': 'reram_cmo', 'artifact_kind': 'retention_state.power_law',
        'programmed_apparent_sha256': tensor_hash(baseline)}])
    artifacts = [('retention_state.power_law', state_path),
                 ('retention_measurements', stage.run_dir / 'retention_measurements.json'),
                 ('retention_states', stage.run_dir / 'retention_states.json')]
    artifacts += [(f'reference.{p.stem}', p) for p in sorted(ref_dir.glob('*.pt'))]
    return {'source_sha256': source_hash, 'source_unchanged': True, 'cohort': cohort,
            'snapshots': len(rows), 'expected_snapshots': 1 + len(settings.snapshots('reram_cmo')),
            'exact_replays': len(rows), 'additional_writes': 0, 'retention': settings.to_dict(),
            'clean': clean, 'raw_programmed': raw_metrics,
            'first_read_clipped_devices': sum(initial_counts)}, artifacts
