"""Native drift-regimes stage: optimal-threshold P&V deployment and untreated aging.

One trajectory per (HWA arm, realization) of one regime point. Every physical
device of the six analog MLPs (active and reference) is a ``ReramDriftCells``
device. Deployment programs every device (attempt 1) and verifies it by P&V;
release (t = 0) is the end of deployment P&V. Teacher KL is evaluated on a
pinned cohort of the declared evaluation corpus at each age after release.
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
from workflow.drift_recovery import CrossbarAdapter
from workflow.initial_programming import teacher_ppl
from workflow.networks import opt_mlp

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
    if len(stream) < cfg.evaluation_sequences * length:
        raise ValueError('Evaluation corpus is too small for the declared cohort.')
    tokens = stream[:cfg.evaluation_sequences * length].reshape(cfg.evaluation_sequences, length).contiguous()
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
    identity = {'regime': cfg.regime, 'arm': cfg.arm, 'realization': realization,
                'hwa_arm': stage.selection.hwa_arm, 'source_sha256': cfg.source_sha256,
                'cohort_sha256': cfg.cohort_sha256, 'evaluation_corpus': cfg.evaluation_corpus,
                'law': dict(cfg.law), 'kappa': law.kappa, 'canary': cfg.canary}
    stage.store.append_metric({'phase': 'drift_regimes_start', **identity})

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
    replay = cells[0].state_dict()

    references = {'clean_hwa': opt_mlp.evaluate(stage.network, {'tokens': tokens}, stage.device),
                  'teacher': teacher_ppl(stage.network, tokens, stage.device)}
    atomic_write_json(stage.run_dir / 'drift_regimes_references.json', references)
    rows = []
    for age in cfg.evaluation_seconds:
        started = time.monotonic()
        clock = release + age
        physical = [c.advance(clock) for c in cells]
        q = adapter.bind(physical)
        metrics = opt_mlp.evaluate(stage.network, {'tokens': tokens}, stage.device, q=q)
        if not all(math.isfinite(metrics[k]) for k in ('cross_entropy', 'perplexity', 'teacher_kl')):
            raise FloatingPointError('Nonfinite drift-regimes metrics.')
        row = {**identity, 'seconds': age, 'clock_seconds': clock, 'release_seconds': release,
               'physical_devices': ndevices, 'physical_sha256': physical_hash(physical),
               'metrics': metrics, 'q_rmse': float((q - stage.network.q).square().mean().sqrt()),
               'bounds': bound_report(cells, physical),
               'evaluation_wall_seconds': time.monotonic() - started}
        rows.append(row)
        atomic_write_json(stage.run_dir / 'drift_regimes_measurements.json', rows)
        stage.store.append_metric({'phase': 'snapshot', **identity, 'seconds': age, **metrics})

    restored = ReramDriftCells.restore(replay, device=stage.device)
    for age in cfg.evaluation_seconds:
        restored.advance(release + age)
    if not torch.equal(restored.value(), physical[0]):
        raise ValueError('Drift-regimes replay from the release state is not exact.')
    if adapter.source_hash() != original_hash:
        raise ValueError('Frozen model changed during drift-regimes aging.')
    kl = np.asarray([r['metrics']['teacher_kl'] for r in rows])
    weights = endpoint_weights(cfg.evaluation_seconds)
    report = {**identity, 'references': references, 'cost': cost,
              'release_teacher_kl': float(kl[0]), 'final_teacher_kl': float(kl[-1]),
              'mean_teacher_kl': float(weights @ kl),
              'horizon_seconds': cfg.evaluation_seconds[-1],
              'final_physical_sha256': rows[-1]['physical_sha256'],
              'source_unchanged': True, 'replay_exact': True, 'snapshots': len(rows), 'trajectories': 1}
    report_path = stage.run_dir / 'drift_regimes_summary.json'
    atomic_write_json(report_path, report)
    return report, [('drift_regimes_measurements', stage.run_dir / 'drift_regimes_measurements.json'),
                    ('drift_regimes_summary', report_path),
                    ('drift_regimes_references', stage.run_dir / 'drift_regimes_references.json')]


def flat_rows(rows, run_dir):
    return [{**{k: v for k, v in row.items() if k not in ('metrics', 'bounds', 'law')},
             **row['metrics'],
             **{f'{role}_{k}': v for role, b in row['bounds'].items() for k, v in b.items()},
             'run_dir': str(run_dir)} for row in rows]
