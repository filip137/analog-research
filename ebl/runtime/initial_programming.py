"""ebl.runtime.initial_programming — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core.artifacts import atomic_write_json, sha256_file
from core.tensor_io_native import save
from methods.maintenance.recovery import tensor_hash
from hardware.drift.flowers import FlowersLaw, FlowersState, named_seed
from methods.initial_programming import targets, prebias_table, select_pairs, observation_report, finish_observation_report, time_weights
from networks.opt_drift import CrossbarAdapter
import networks.opt as opt_mlp


@torch.no_grad()
def teacher_ppl(network, tokens, device):
    total, count = 0., 0
    for batch in tokens.split(1):
        batch = batch.to(device)
        output = network.teacher(batch)[:, :-1]
        if not torch.isfinite(output).all():
            raise FloatingPointError('Nonfinite teacher logits.')
        total += float(F.cross_entropy(output.flatten(0, 1), batch[:, 1:].flatten(), reduction='sum'))
        count += batch[:, 1:].numel()
    return {'cross_entropy': total/count, 'perplexity': math.exp(total/count), 'scored_tokens': count}


def physical_hash(physical):
    return hashlib.sha256(''.join(tensor_hash(g) for g in physical).encode()).hexdigest()


@torch.no_grad()
def run(stage):
    from ebl.schema.source_fit import validate_source_fit
    cfg = stage.lifecycle.initial_programming
    if sha256_file(stage.request.weights) != cfg.source_sha256:
        raise ValueError('Initial programming HWA source hash mismatch.')
    model_path = Path(cfg.model_path)
    if sha256_file(model_path) != cfg.model_sha256:
        raise ValueError('Initial programming Flowers model hash mismatch.')
    source = stage.checked('weights', 'hwa_source', hwa_arm=stage.selection.hwa_arm)
    validate_source_fit(stage.lifecycle, stage.lifecycle.hwa.arm(stage.selection.hwa_arm), source['fit'])
    stage.network.load_state_dict(source['model'])
    del source
    stage.cache()
    stage.device_bundle()
    corpus, _ = opt_mlp.load_corpus(stage.lifecycle.data, stage.lifecycle.network.model)
    length = stage.lifecycle.data.sequence_length
    tokens = corpus['splits']['validation'][:cfg.validation_sequences*length].reshape(cfg.validation_sequences, length).contiguous()
    if tensor_hash(tokens) != cfg.cohort_sha256:
        raise ValueError('Initial programming token-cohort hash mismatch.')
    del corpus
    adapter = CrossbarAdapter(stage.network, stage.device)
    original_hash = adapter.source_hash()
    parameters = json.loads(model_path.read_text())
    law = FlowersLaw(parameters, stage.device)
    model_copy = stage.run_dir/'flowers-model.json'
    model_copy.write_bytes(model_path.read_bytes())
    identity = {'source_sha256': cfg.source_sha256, 'model_sha256': cfg.model_sha256,
                'cohort_sha256': cfg.cohort_sha256, 'policy': cfg.policy,
                'realization': stage.selection.array_seed, 'canary': cfg.canary}
    stage.store.append_metric({'phase': 'initial_programming_start', **identity})
    lookup = {}
    if cfg.policy in ('P2', 'P4'):
        for anchor_time in (cfg.initial_anchor_seconds, cfg.rewrite_anchor_seconds):
            if cfg.policy == 'P2' and anchor_time == cfg.rewrite_anchor_seconds:
                continue
            q, a = prebias_table(law, cfg, anchor_time)
            lookup[anchor_time] = (law.tensor(q), law.tensor(a))
    matrices = stage.network.layout
    clean_q = [stage.network.q[m.offset:m.offset+m.size].detach() for m in matrices]
    states = [FlowersState(law, targets(q, cfg.policy, lookup.get(cfg.initial_anchor_seconds)),
                           cfg.initial_anchor_seconds, (cfg.seed, stage.selection.array_seed, m.name))
              for m, q in zip(matrices, clean_q)]
    if cfg.policy not in ('P2', 'P4'):
        if not torch.allclose(adapter.bind([s.anchor for s in states]), stage.network.q, atol=3e-7, rtol=1e-6):
            raise ValueError('Programming mapping failed to preserve the clean normalized weights.')
    ndevices = sum(s.anchor.numel() for s in states)
    masks = [torch.zeros_like(q, dtype=torch.bool) for q in clean_q]
    selected = 0
    references = {'clean_hwa': opt_mlp.evaluate(stage.network, {'tokens': tokens}, stage.device, batch_size=1),
                  'teacher': teacher_ppl(stage.network, tokens, stage.device)}
    atomic_write_json(stage.run_dir/'initial_programming_references.json', references)
    rows, events = [], []
    snapshot_times = {cfg.release+t: t for t in cfg.evaluation_seconds}
    retry = cfg.policy in ('P3', 'P4', 'P5')
    for clock in cfg.calendar():
        if clock > max(snapshot_times):
            break
        for state in states:
            state.advance(clock)
        if clock == cfg.verify_seconds and retry:
            reports = []
            for i, (state, q, matrix) in enumerate(zip(states, clean_q, matrices)):
                observed, raw = state.read(clock)
                if stage.network.scaling == 'per_output_channel_absmax':
                    scales = stage.network.channel_scales[stage.network.channel_offsets[i]:stage.network.channel_offsets[i+1]]
                else:
                    scales = stage.network.scales[i].expand(matrix.shape[0])
                seed = named_seed(cfg.selection_seed, stage.selection.array_seed, matrix.name) if cfg.policy == 'P5' else None
                masks[i] = select_pairs(observed, q, scales, matrix.shape, cfg.retry_fraction, random_seed=seed)
                reports.append(observation_report(law, state, raw))
            selected = sum(int(mask.sum()) for mask in masks)
            events.append({'event': 'verify', 'clock_seconds': clock, 'selected_pairs': selected,
                           'clipping': finish_observation_report(sum(reports), parameters['bin_edges'])})
            stage.store.append_metric({'phase': 'verify', **identity, 'selected_pairs': selected})
        if clock == cfg.rewrite_anchor_seconds and retry:
            for state, q, mask in zip(states, clean_q, masks):
                target = targets(q, cfg.policy, lookup.get(clock))
                state.rewrite(target, mask.expand_as(target), clock)
            events.append({'event': 'full_rewrite', 'clock_seconds': clock, 'device_writes': 2*selected})
        if clock not in snapshot_times:
            continue
        started = time.monotonic()
        physical, reports = [], []
        for state in states:
            observed, raw = state.read(clock)
            physical.append(observed)
            reports.append(observation_report(law, state, raw))
        q = adapter.bind(physical)
        metrics = opt_mlp.evaluate(stage.network, {'tokens': tokens}, stage.device, q=q, batch_size=1)
        if not all(math.isfinite(metrics[k]) for k in ('cross_entropy', 'perplexity', 'teacher_kl')):
            raise FloatingPointError('Nonfinite initial-programming metrics.')
        row = {**identity, 'seconds': snapshot_times[clock], 'clock_seconds': clock,
               'release_seconds': cfg.release, 'physical_devices': ndevices,
               'physical_sha256': physical_hash(physical), 'metrics': metrics,
               'q_rmse': float((q-stage.network.q).square().mean().sqrt()),
               'clipping': finish_observation_report(sum(reports), parameters['bin_edges']),
               'evaluation_wall_seconds': time.monotonic()-started}
        rows.append(row)
        atomic_write_json(stage.run_dir/'initial_programming_measurements.json', rows)
        stage.store.append_metric({'phase': 'snapshot', **identity, 'seconds': row['seconds'], **metrics})
    final_path = stage.run_dir/'checkpoints/initial_programming_final.pt'
    save(final_path, {'schema': 'initial_programming.final.v1', 'identity': identity,
                     'clock_seconds': rows[-1]['clock_seconds'], 'physical': [g.cpu() for g in physical],
                     'selected_pair_masks': [m.cpu() for m in masks], 'settings': cfg.to_dict()})
    saved = torch.load(final_path, map_location=stage.device, weights_only=False)
    replay_physical = saved['physical']
    if physical_hash(replay_physical) != rows[-1]['physical_sha256']:
        raise ValueError('Serialized final conductance state changed.')
    test_tokens = tokens[:1].to(stage.device)
    expected = stage.network(test_tokens, q=adapter.bind(physical))
    actual = stage.network(test_tokens, q=adapter.bind(replay_physical))
    if not torch.equal(expected, actual):
        raise ValueError('Final held-state replay is not exact.')
    if adapter.source_hash() != original_hash:
        raise ValueError('Frozen model changed during initial programming.')
    ce = np.asarray([r['metrics']['cross_entropy'] for r in rows])
    weights = time_weights(cfg.evaluation_seconds)
    cost = {'initial_programs': ndevices, 'extra_programs': 2*selected,
            'anchor_reads': ndevices+2*selected, 'delayed_verify_reads': ndevices if retry else 0,
            'release_reads': ndevices, 'diagnostic_snapshot_reads': ndevices*(len(rows)-1),
            'extra_writes_per_device': 2*selected/ndevices, 'preparation_seconds': cfg.release}
    report = {**identity, 'references': references, 'cost': cost, 'events': events,
              'initial_cross_entropy': ce[0], 'final_cross_entropy': ce[-1],
              'mean_log_ppl_growth': float(weights@(ce-ce[0])),
              'mean_cross_entropy': float(weights@ce), 'source_unchanged': True,
              'replay_exact': True, 'snapshots': len(rows), 'trajectories': 1}
    report_path = stage.run_dir/'initial_programming_summary.json'
    atomic_write_json(report_path, report)
    return report, [('initial_programming_measurements', stage.run_dir/'initial_programming_measurements.json'),
                    ('initial_programming_summary', report_path), ('initial_programming_state', final_path),
                    ('flowers_model', model_copy)]
