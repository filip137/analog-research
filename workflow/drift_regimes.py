"""Native drift-regimes stage: optimal-threshold P&V deployment, then every maintenance arm.

One trajectory per (HWA arm, realization) of one regime point. Every physical
device of the six analog MLPs (active and reference) is a ``ReramDriftCells``
device. Deployment programs every device (attempt 1) and verifies it by P&V;
release (t = 0) is the end of deployment P&V. Each maintenance arm
(``workflow.drift_regimes_arms``) restarts from the released state, so arms
are paired. Teacher KL is evaluated on a pinned cohort of the declared
evaluation corpus at each age after release, before any event at that age.
"""
from dataclasses import replace
import hashlib
import json
import math
import time

import numpy as np
import torch

from experiments.artifacts import atomic_write_json, sha256_file
from training.drift_family import ReramDriftCells, program_verify
from training.drift_recovery import tensor_hash
from training.drift_regimes_config import parse_target
from workflow.drift_recovery import CrossbarAdapter
from workflow.initial_programming import teacher_ppl
from workflow import drift_regimes_arms as arms
from workflow.networks import opt_mlp

SAMPLE_SEED_OFFSET = 104729   # teacher samples for sample-target arms (recovery-data-preparation exp-001)

ACTIVE, REFERENCE = 0, 1


def physical_hash(physical):
    return hashlib.sha256(''.join(tensor_hash(g) for g in physical).encode()).hexdigest()


def endpoint_weights(seconds):
    """Trapezoid weights for the linear time average over [0, seconds[-1]]."""
    t = np.asarray(seconds, dtype=float)
    w = np.zeros_like(t)
    w[:-1] += np.diff(t) / 2
    w[1:] += np.diff(t) / 2
    return w / (t[-1] - t[0])


def bound_report(cells, physical):
    law = cells[0].law
    report = {}
    for role, name in ((ACTIVE, 'active'), (REFERENCE, 'reference')):
        values = torch.cat([g[role].flatten() for g in physical])
        report[name] = {'devices': values.numel(),
                        'at_drift_min': int((values <= law.drift_min).sum()),
                        'at_drift_max': int((values >= law.drift_max).sum())}
    return report


def evaluation_cohort(lifecycle, cfg):
    data = replace(lifecycle.data, corpus=cfg.evaluation_corpus, corpus_sha256=cfg.evaluation_corpus_sha256)
    corpus, _ = opt_mlp.load_corpus(data, lifecycle.network.model)
    length = lifecycle.data.sequence_length
    stream = corpus['splits'][cfg.evaluation_split]
    first, last = cfg.evaluation_start * length, (cfg.evaluation_start + cfg.evaluation_sequences) * length
    if len(stream) < last:
        raise ValueError('Evaluation corpus is too small for the declared cohort.')
    tokens = stream[first:last].reshape(cfg.evaluation_sequences, length).contiguous()
    if tensor_hash(tokens) != cfg.cohort_sha256:
        raise ValueError('Drift-regimes token-cohort hash mismatch.')
    return tokens


@torch.no_grad()
def run(stage):
    from workflow.runtime import validate_source_fit
    lc, cfg = stage.lifecycle, stage.lifecycle.drift_regimes
    if sha256_file(stage.request.weights) != cfg.source_sha256:
        raise ValueError('Drift-regimes HWA source hash mismatch.')
    source = stage.checked('weights', 'hwa_source', hwa_arm=stage.selection.hwa_arm)
    validate_source_fit(lc, lc.hwa.arm(stage.selection.hwa_arm), source['fit'])
    stage.network.load_state_dict(source['model'])
    del source
    stage.cache()
    stage.device_bundle()
    tokens = evaluation_cohort(lc, cfg)
    adapter = CrossbarAdapter(stage.network, stage.device)
    original_hash = adapter.source_hash()
    law = cfg.family
    realization = stage.selection.array_seed
    identity = {'regime': cfg.regime, 'realization': realization,
                'hwa_arm': stage.selection.hwa_arm, 'source_sha256': cfg.source_sha256,
                'cohort_sha256': cfg.cohort_sha256, 'evaluation_corpus': cfg.evaluation_corpus,
                'law': dict(cfg.law), 'kappa': law.kappa, 'canary': cfg.canary}
    stage.store.append_metric({'phase': 'drift_regimes_start', **identity, 'arms': [a['label'] for a in cfg.arms]})

    targets, _, _, names = adapter.physical_layout()
    if not torch.allclose(adapter.bind(targets), stage.network.q, atol=3e-7, rtol=1e-6):
        raise ValueError('Physical mapping failed to preserve the clean normalized weights.')
    if any(bool(((t < law.program_min) | (t > law.program_max)).any()) for t in targets):
        raise ValueError('Mapped targets leave the programming window.')
    started = time.monotonic()
    cells = [ReramDriftCells(t.float().contiguous(), law=law, seed=f'{cfg.seed}/{realization}', name=name)
             for t, name in zip(targets, names)]
    del targets
    deployment = []
    for c in cells:
        n = c.target.numel()
        deployment.append(program_verify(c, torch.arange(n, device=c.target.device), c.target.flatten(),
                                         attempts=cfg.deployment_attempts,
                                         threshold=cfg.deployment_threshold,
                                         verify_seconds=cfg.verify_seconds, first_written=True))
    release = max(c.clock for c in cells)
    ndevices = sum(c.target.numel() for c in cells)
    histogram = [sum(d['accepted_at_attempt'][k] for d in deployment) for k in range(cfg.deployment_attempts)]
    cost = {'physical_devices': ndevices,
            'deployment_reset_attempts': sum(d['reset_attempts'] for d in deployment),
            'deployment_verify_reads': sum(d['verify_reads'] for d in deployment),
            'deployment_capped_devices': sum(d['capped'] for d in deployment),
            'accepted_at_attempt': histogram, 'release_seconds': release,
            'deployment_wall_seconds': time.monotonic() - started}
    cost['resets_per_device'] = cost['deployment_reset_attempts'] / ndevices
    stage.store.append_metric({'phase': 'deployed', **identity, **cost})
    released = [c.state_dict() for c in cells]
    horizon_live = cells[0].value(release + cfg.evaluation_seconds[-1])
    del cells

    references = {'clean_hwa': opt_mlp.evaluate(stage.network, {'tokens': tokens}, stage.device),
                  'teacher': teacher_ppl(stage.network, tokens, stage.device)}
    atomic_write_json(stage.run_dir / 'drift_regimes_references.json', references)
    hidden = probe = None
    if cfg.adaptation is not None:
        adaptation_tokens = arms.adaptation_cohort(lc, cfg.adaptation)
        hidden = arms.prefix_states(stage.network, adaptation_tokens, stage.device)
        counts = [n for kind, n in (parse_target(a.get('target', 'kl')) for a in cfg.arms) if kind == 'samples']
        if counts:   # one fixed draw shared by every sample-target arm (prefix columns for smaller counts)
            seed = cfg.adaptation['seed'] + SAMPLE_SEED_OFFSET
            hidden.samples = arms.teacher_samples(stage.network, hidden, max(counts), seed)
            stage.store.append_metric({'phase': 'teacher_samples', **identity, 'count': max(counts), 'seed': seed,
                                       'samples_sha256': tensor_hash(hidden.samples.cpu())})
        probe = hidden[:cfg.adaptation['probe_sequences']]
    for p in stage.network.parameters():
        p.requires_grad_(False)
    weights = endpoint_weights(cfg.evaluation_seconds)
    rows, summaries, events = [], {}, []
    for spec in cfg.arms:
        arm_started = time.monotonic()
        cells = [ReramDriftCells.restore(state, device=stage.device) for state in released]
        controller = arms.Arm(spec, stage.network, cfg.adaptation, hidden, probe, realization, cfg.verify_seconds,
                              cells, cfg.evaluation_seconds[-1])
        arm_id = {**identity, 'arm': spec['label'], 'method': spec['method'], 'twin': bool(spec.get('twin'))}
        arm_rows = []
        try:
            for age, evaluate, index in arms.timeline(cfg.evaluation_seconds, controller.schedule,
                                                      controller.evaluate_after_event):
                clock = release + age
                if not evaluate:
                    record = controller.event(index, age, clock, cells, adapter.bind)
                    events.append({**arm_id, **record})
                    stage.store.append_metric({'phase': 'event', **arm_id, **record})
                    continue
                started = time.monotonic()
                physical = controller.physical(cells, clock)
                q = adapter.bind(physical)
                controller.context(age)
                metrics = opt_mlp.evaluate(stage.network, {'tokens': tokens}, stage.device, q=q)
                stage.network.drift_compensation = None
                if not arms.finite(metrics):
                    raise FloatingPointError('Nonfinite drift-regimes metrics.')
                row = {**arm_id, 'seconds': age, 'clock_seconds': clock, 'release_seconds': release,
                       'physical_devices': ndevices, 'physical_sha256': physical_hash(physical),
                       'metrics': metrics, 'q_rmse': float((q - stage.network.q).square().mean().sqrt()),
                       'bounds': bound_report(cells, physical),
                       'evaluation_wall_seconds': time.monotonic() - started}
                arm_rows.append(row)
                atomic_write_json(stage.run_dir / 'drift_regimes_measurements.json', rows + arm_rows)
                stage.store.append_metric({'phase': 'snapshot', **arm_id, 'seconds': age, **metrics})
        finally:
            controller.close()
        rows.extend(arm_rows)
        kl = np.asarray([r['metrics']['teacher_kl'] for r in arm_rows])
        summaries[spec['label']] = {
            **arm_id, 'settings': {k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()},
            'cost': controller.cost, 'release_teacher_kl': float(kl[0]), 'final_teacher_kl': float(kl[-1]),
            'mean_teacher_kl': float(weights @ kl), 'final_physical_sha256': arm_rows[-1]['physical_sha256'],
            'arm_wall_seconds': time.monotonic() - arm_started}
        atomic_write_json(stage.run_dir / 'drift_regimes_events.json', events)
        del cells

    # Query-order independence of the released state: one jump equals the evaluation path.
    jump = ReramDriftCells.restore(released[0], device=stage.device)
    path = ReramDriftCells.restore(released[0], device=stage.device)
    for age in cfg.evaluation_seconds:
        path.advance(release + age)
    if not (torch.equal(jump.advance(release + cfg.evaluation_seconds[-1]), path.value())
            and torch.equal(path.value(), horizon_live)):
        raise ValueError('Drift-regimes replay from the release state is not exact.')
    if adapter.source_hash() != original_hash:
        raise ValueError('Frozen model changed during drift-regimes aging.')
    first = summaries[cfg.arms[0]['label']]
    report = {**identity, 'references': references, 'cost': cost, 'arms': summaries,
              'release_teacher_kl': first['release_teacher_kl'], 'final_teacher_kl': first['final_teacher_kl'],
              'mean_teacher_kl': first['mean_teacher_kl'], 'final_physical_sha256': first['final_physical_sha256'],
              'horizon_seconds': cfg.evaluation_seconds[-1],
              'source_unchanged': True, 'replay_exact': True, 'snapshots': len(rows),
              'arm_count': len(cfg.arms), 'trajectories': 1}
    report_path = stage.run_dir / 'drift_regimes_summary.json'
    atomic_write_json(report_path, report)
    atomic_write_json(stage.run_dir / 'drift_regimes_events.json', events)
    return report, [('drift_regimes_measurements', stage.run_dir / 'drift_regimes_measurements.json'),
                    ('drift_regimes_summary', report_path),
                    ('drift_regimes_references', stage.run_dir / 'drift_regimes_references.json'),
                    ('drift_regimes_events', stage.run_dir / 'drift_regimes_events.json')]


def flat_rows(rows, run_dir):
    return [{**{k: v for k, v in row.items() if k not in ('metrics', 'bounds', 'law')},
             **row['metrics'],
             **{f'{role}_{k}': v for role, b in row['bounds'].items() for k, v in b.items()},
             'run_dir': str(run_dir)} for row in rows]
