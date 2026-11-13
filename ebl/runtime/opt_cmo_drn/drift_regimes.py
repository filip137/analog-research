"""Native exp-003 drift-regimes stage on the frozen OPT CMO DRN.

One run is one (regime, realization) deployment followed by its declared arms. Every
physical device of the six DRN MLPs (four per logical weight) is a ``ReramDriftCells``
device. Deployment programs every device (attempt 1) and verifies it by P&V; release
(t = 0) is the end of deployment P&V. Each arm restarts from the released state, so arms
are paired. Teacher KL is evaluated on the pinned cohort at each age after release,
before any event at that age. An evaluation is reused across arms only when the held
physical state, the LayerNorm state, the output correction, the cohort and the forward
policy are identical; every logical row is still written, with its reuse provenance.

``stage: readiness`` runs the declared numerical checks and the bounded timing sample
instead of the arms.
"""
from dataclasses import asdict
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

import ebl.runtime.opt_cmo_drn.runtime as runtime
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.opt_cmo_drn_drift_regimes import EXPERIMENT_ID
from hardware.devices.cmo import G_MAX_US, HALF_RANGE_US, MID_US
from hardware.drift.family import ReramDriftCells, ReramFamily, program_verify
from hardware.faults import state_hash
from methods.drift_regimes import arms as arms_module
from methods.drift_regimes import forward
from networks.opt.data import blocks, load_corpus, pinned_load

READINESS_AGES = (0., 900., 86400.)
READINESS_SEQUENCES = 2
READINESS_KL_TOLERANCE = 1e-4
TIMING_ADAM_STEPS = 20


def plain_hash(value):
    """Plain-bytes SHA-256 (crossbar cohort convention)."""
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def physical_hash(physical):
    h = hashlib.sha256()
    for g in physical:
        h.update(g.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def norm_hash(model):
    h = hashlib.sha256()
    for norm in forward.layer_norms(model):
        for p in norm.parameters():
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def correction_hash(correction):
    if correction is None:
        return None
    h = hashlib.sha256()
    for a, b in correction:
        h.update(a.detach().cpu().numpy().tobytes())
        h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def endpoint_weights(seconds):
    """Trapezoid weights for the linear time average over [0, seconds[-1]]."""
    t = np.asarray(seconds, dtype=float)
    w = np.zeros_like(t)
    w[:-1] += np.diff(t) / 2
    w[1:] += np.diff(t) / 2
    return w / (t[-1] - t[0])


def physical_layout(model):
    """Targets (uS, ``u`` orientation), names and rail roles of the 12 DRN matrices.

    role = 4 j + 2 (input rail: 0 +x, 1 -x) + (fc1: hidden half 0 +, 1 -; fc2: output rail of the pair).
    """
    targets, names, roles = [], [], []
    for k, block in enumerate(model.blocks):
        for j, u in enumerate((block.u1, block.u2)):
            targets.append((MID_US + HALF_RANGE_US * u.detach()).float().contiguous())
            names.append(f'decoder.layers.{model.start + k}.fc{j + 1}')
            rows, cols = u.shape
            row = torch.arange(rows, device=u.device) // (rows // 2)
            col = torch.arange(cols, device=u.device)
            col = col % 2 if j else col // (cols // 2)
            roles.append((4 * j + 2 * row[:, None] + col[None, :]).to(torch.int8))
    return targets, names, roles


@torch.no_grad()
def remap(model, mapping):
    """Deployment remapping G' = min(offset + scale G, G_max) of every DRN device, in place.

    The bias current scales with ``scale``. With offset 0 and no saturation the DRN equilibrium is
    unchanged: drive and node conductance sums both scale by ``scale`` and the +x/-x rails cancel any
    offset in the drive. A nonzero offset changes node loading. Returns saturation statistics.
    """
    scale, offset = mapping['scale'], mapping['offset_us']
    saturated = total = 0
    for block in model.blocks:
        for u in (block.u1, block.u2):
            raw = offset + scale * (MID_US + HALF_RANGE_US * u.double())
            saturated += int((raw > G_MAX_US).sum())
            total += raw.numel()
            u.copy_(((raw.clamp(max=G_MAX_US) - MID_US) / HALF_RANGE_US).to(u.dtype))
        block.current_scale.mul_(scale)
    return {'scale': scale, 'offset_us': offset, 'saturated_devices': saturated, 'devices': total,
            'saturated_fraction': saturated / total}


def bound_report(law, physical, names, roles, targets):
    per_matrix, totals = {}, {}
    for name, values, role, target in zip(names, physical, roles, targets):
        r = role.flatten().long()
        v = values.flatten()
        devices = torch.bincount(r, minlength=8)
        low = torch.bincount(r[v <= law.drift_min], minlength=8)
        high = torch.bincount(r[v >= law.drift_max], minlength=8)
        present = [int(x) for x in torch.nonzero(devices).flatten()]
        entry = {str(x): {'devices': int(devices[x]), 'at_drift_min': int(low[x]), 'at_drift_max': int(high[x])}
                 for x in present}
        entry['error_rms_us'] = float((v - target.flatten()).double().square().mean().sqrt())
        per_matrix[name] = entry
        for x in present:
            t = totals.setdefault(str(x), {'devices': 0, 'at_drift_min': 0, 'at_drift_max': 0})
            for key in t:
                t[key] += entry[str(x)][key]
    return {'roles': totals, 'matrices': per_matrix}


def cohorts(settings):
    splits = load_corpus(settings.corpus, settings.corpus_sha256)
    length = settings.sequence_length
    e = settings.evaluation
    stream = splits[e.split]
    first, last = e.start * length, (e.start + e.sequences) * length
    if len(stream) < last:
        raise ValueError('Evaluation corpus is too small for the declared cohort.')
    evaluation = stream[first:last].reshape(e.sequences, length).contiguous()
    if plain_hash(evaluation) != e.sha256:
        raise ValueError('Drift-regimes evaluation cohort hash mismatch.')
    a = settings.adaptation
    available = blocks(splits[a.split], length, len(splits[a.split]) // length)
    if len(available) < a.sequences:
        raise ValueError('Adaptation corpus is too small for the declared cohort.')
    order = torch.randperm(len(available), generator=torch.Generator().manual_seed(a.seed))
    adaptation = available[order[:a.sequences]].contiguous()
    if plain_hash(adaptation) != a.sha256:
        raise ValueError('Drift-regimes adaptation cohort hash mismatch.')
    return evaluation, adaptation


def _store(request):
    identities = []
    snapshot = os.environ.get('OPT_DRN_SOURCE_SNAPSHOT')
    if snapshot:
        from pathlib import Path
        snapshot = Path(snapshot)
        receipt = json.loads(snapshot.read_text())
        for name, digest in receipt['files'].items():
            path = (snapshot.parent / name).resolve()
            if not path.is_relative_to(snapshot.parent.resolve()) or sha256_file(path) != digest:
                raise ValueError(f'Source snapshot mismatch: {name}.')
        identities.append({'role': 'source_snapshot', 'path': str(snapshot), 'sha256': sha256_file(snapshot)})
    for name in ('teacher_weights', 'weights'):
        path = getattr(request, name, None)
        if path:
            identities.append({'role': name, 'path': str(path.resolve()), 'sha256': sha256_file(path)})
    store = RunStore.create(output_root=request.output_dir, experiment_id=EXPERIMENT_ID,
                            resolved_config=asdict(request.spec), command=request.command, repo_root=runtime.ROOT,
                            input_artifacts=identities, run_id=os.environ.get('OPT_DRN_RUN_ID'))
    print(f'RUN_DIR={store.run_dir}', flush=True)
    return store


def configure(config):
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Stage:
    """Inputs, cohorts, deployment and evaluation shared by the screen and readiness."""

    def __init__(self, request):
        config, settings = request.spec, request.spec.drift_regimes
        if request.teacher_weights is None or request.weights is None:
            raise ValueError('The drift-regimes screen needs pinned --teacher-weights and --weights.')
        if any(getattr(request, k, None) is not None for k in
               ('resume', 'device_state', 'base_weights', 'device_data', 'device_model', 'deployment')):
            raise ValueError('Unexpected drift-regimes input; deployment replays from the source.')
        configure(config)
        self.config, self.settings, self.device = config, settings, config.device
        teacher = runtime.teacher_model(request.teacher_weights, config)
        source = pinned_load(request.weights, settings.source_sha256)
        self.model = runtime.from_mapped(teacher, config, source).to(self.device).eval().requires_grad_(False)
        del source
        self.mapping = None if settings.mapping is None else remap(self.model, settings.mapping)
        self.teacher = teacher.to(self.device).eval().requires_grad_(False)
        self.policy = forward.Policy(self.model, settings.forward_policy)
        self.policy.mode(False)
        self.evaluation_tokens, self.adaptation_tokens = cohorts(settings)
        self.law = ReramFamily(**settings.law)
        self.store = _store(request)
        self.identity = {'regime': settings.regime, 'realization': settings.realization, 'seed': settings.seed,
                         'source_sha256': settings.source_sha256, 'evaluation_sha256': settings.evaluation.sha256,
                         'adaptation_sha256': settings.adaptation.sha256, 'law': dict(settings.law),
                         'kappa': self.law.kappa, 'forward_policy': asdict(settings.forward_policy),
                         'canary': settings.canary}
        if self.mapping is not None:
            self.identity['mapping'] = self.mapping

    def prefixes(self):
        self.evaluation = forward.prefix_states(self.model, self.teacher, self.evaluation_tokens, self.device)
        self.hidden = forward.prefix_states(self.model, self.teacher, self.adaptation_tokens, self.device)
        self.probe = self.hidden[:self.settings.adaptation.probe_sequences]

    @torch.no_grad()
    def deploy(self):
        s, law = self.settings, self.law
        self.targets, self.names, self.roles = physical_layout(self.model)
        masters = [g for block in self.model.blocks for g in block.conductances(None)]
        if not all(torch.allclose(t / G_MAX_US, m, atol=3e-7, rtol=1e-6) for t, m in zip(self.targets, masters)):
            raise ValueError('Physical mapping failed to preserve the clean DRN conductances.')
        if any(bool(((t < law.program_min - 1e-4) | (t > law.program_max + 1e-4)).any()) for t in self.targets):
            raise ValueError('Mapped targets leave the programming window.')
        # FP32 rounding at the rails (e.g. 9 - 5e-7 uS); the cells clamp targets to the window identically.
        self.targets = [t.clamp(law.program_min, law.program_max) for t in self.targets]
        started = time.monotonic()
        cells = [ReramDriftCells(t, law=law, seed=f'{s.seed}/{s.realization}', name=name)
                 for t, name in zip(self.targets, self.names)]
        deployment = []
        for c in cells:
            n = c.target.numel()
            deployment.append(program_verify(c, torch.arange(n, device=c.target.device), c.target.flatten(),
                                             attempts=s.deployment_attempts, threshold=s.deployment_threshold,
                                             verify_seconds=s.verify_seconds, first_written=True))
        self.release = max(c.clock for c in cells)
        ndevices = sum(c.target.numel() for c in cells)
        histogram = [sum(d['accepted_at_attempt'][k] for d in deployment) for k in range(s.deployment_attempts)]
        self.cost = {'physical_devices': ndevices, 'logical_weights': ndevices // 4,
                     'deployment_reset_attempts': sum(d['reset_attempts'] for d in deployment),
                     'deployment_verify_reads': sum(d['verify_reads'] for d in deployment),
                     'deployment_capped_devices': sum(d['capped'] for d in deployment),
                     'accepted_at_attempt': histogram, 'release_seconds': self.release,
                     'deployment_wall_seconds': time.monotonic() - started}
        self.cost['resets_per_device'] = self.cost['deployment_reset_attempts'] / ndevices
        self.released = [c.state_dict() for c in cells]
        self.release_sha256 = physical_hash([c.value(self.release) for c in cells])
        self.horizon_live = cells[0].value(self.release + s.evaluation_seconds[-1]).cpu()
        self.store.append_metric({'phase': 'deployed', **self.identity, **self.cost,
                                  'release_physical_sha256': self.release_sha256})
        del cells

    def cells(self):
        return [ReramDriftCells.restore(state, device=self.device) for state in self.released]

    def evaluate(self, physical, correction=None, prefix=None, sweeps=None):
        view = forward.bind(physical)
        forward.check_view(self.model, view)
        prefix = self.evaluation if prefix is None else prefix
        return forward.evaluate(self.model, self.teacher, prefix, view, self.policy, correction=correction,
                                microbatch=self.settings.evaluation_microbatch, sweeps=sweeps)

    @torch.no_grad()
    def references(self):
        clean = forward.evaluate(self.model, self.teacher, self.evaluation, None, self.policy,
                                 microbatch=self.settings.evaluation_microbatch)
        ce = count = 0.
        for part in self.evaluation.split(self.settings.evaluation_microbatch):
            out = forward.teacher_logits(self.teacher, self.model.start, part.teacher).flatten(0, 1)
            ce += float(torch.nn.functional.cross_entropy(out, part.tokens[:, 1:].flatten(), reduction='sum'))
            count += out.shape[0]
        return {'clean_drn': clean, 'teacher': {'cross_entropy': ce / count, 'perplexity': math.exp(ce / count)}}


def run_screen(stage):
    s = stage.settings
    source_hash = state_hash(stage.model.state_dict())
    stage.store.append_metric({'phase': 'drift_regimes_start', **stage.identity, 'arms': [a['label'] for a in s.arms]})
    stage.prefixes()
    stage.deploy()
    references = stage.references()
    atomic_write_json(stage.store.run_dir / 'drift_regimes_references.json', references)
    weights = endpoint_weights(s.evaluation_seconds)
    cache = {}
    rows, summaries, events = [], {}, []
    measurements = stage.store.run_dir / 'drift_regimes_measurements.json'
    for spec in s.arms:
        arm_started = time.monotonic()
        cells = stage.cells()
        controller = arms_module.Arm(spec, stage.model, stage.teacher, s, stage.hidden, stage.probe,
                                     stage.policy, cells, s.evaluation_seconds[-1])
        arm_id = {**stage.identity, 'arm': spec['label'], 'method': spec['method'], 'twin': bool(spec.get('twin'))}
        arm_rows = []
        try:
            for age, evaluate, index in arms_module.timeline(s.evaluation_seconds, controller.schedule):
                clock = stage.release + age
                if not evaluate:
                    record = controller.event(index, age, clock, cells)
                    events.append({**arm_id, **record})
                    stage.store.append_metric({'phase': 'event', **arm_id, **record})
                    atomic_write_json(stage.store.run_dir / 'drift_regimes_events.json', events)
                    continue
                started = time.monotonic()
                physical = controller.physical(cells, clock)
                correction = controller.context()
                key = (physical_hash(physical), norm_hash(stage.model), correction_hash(correction))
                if key in cache:
                    metrics, diagnostics, source = cache[key]
                    reused = source
                else:
                    metrics = stage.evaluate(physical, correction)
                    diagnostics = forward.solver_diagnostics(stage.model, stage.evaluation, forward.bind(physical),
                                                             stage.policy)
                    cache[key] = (metrics, diagnostics, {'arm': spec['label'], 'seconds': age})
                    reused = None
                if not all(math.isfinite(metrics[k]) for k in ('cross_entropy', 'perplexity', 'teacher_kl')):
                    raise FloatingPointError('Nonfinite drift-regimes metrics.')
                row = {**arm_id, 'seconds': age, 'clock_seconds': clock, 'release_seconds': stage.release,
                       'physical_devices': stage.cost['physical_devices'], 'physical_sha256': key[0],
                       'layernorm_sha256': key[1], 'correction_sha256': key[2], 'reused_from': reused,
                       'metrics': metrics, 'numerical_diagnostics': diagnostics,
                       'bounds': bound_report(stage.law, physical, stage.names, stage.roles, stage.targets),
                       'evaluation_wall_seconds': time.monotonic() - started}
                arm_rows.append(row)
                atomic_write_json(measurements, rows + arm_rows)
                stage.store.append_metric({'phase': 'snapshot', **arm_id, 'seconds': age, 'reused': reused is not None,
                                           **metrics})
                print(json.dumps({'arm': spec['label'], 'seconds': age, 'teacher_kl': metrics['teacher_kl'],
                                  'reused': reused is not None, 'wall': row['evaluation_wall_seconds']}), flush=True)
        finally:
            controller.close()
        rows.extend(arm_rows)
        kl = np.asarray([r['metrics']['teacher_kl'] for r in arm_rows])
        summaries[spec['label']] = {
            **arm_id, 'settings': {k: list(v) if isinstance(v, tuple) else v for k, v in spec.items()},
            'cost': controller.cost, 'release_teacher_kl': float(kl[0]), 'final_teacher_kl': float(kl[-1]),
            'worst_teacher_kl': float(kl.max()), 'mean_teacher_kl': float(weights @ kl),
            'final_physical_sha256': arm_rows[-1]['physical_sha256'],
            'arm_wall_seconds': time.monotonic() - arm_started}
        atomic_write_json(stage.store.run_dir / 'drift_regimes_summary.partial.json', summaries)
        del cells
        if str(stage.device).startswith('cuda'):
            torch.cuda.empty_cache()

    # Query-order independence of the released state: one jump equals the evaluation path.
    jump = ReramDriftCells.restore(stage.released[0], device=stage.device)
    path = ReramDriftCells.restore(stage.released[0], device=stage.device)
    for age in s.evaluation_seconds:
        path.advance(stage.release + age)
    if not (torch.equal(jump.advance(stage.release + s.evaluation_seconds[-1]), path.value())
            and torch.equal(path.value().cpu(), stage.horizon_live)):
        raise ValueError('Drift-regimes replay from the release state is not exact.')
    if state_hash(stage.model.state_dict()) != source_hash:
        raise ValueError('Frozen model changed during drift-regimes aging.')
    report = {**stage.identity, 'references': references, 'cost': stage.cost, 'arms': summaries,
              'release_physical_sha256': stage.release_sha256, 'horizon_seconds': s.evaluation_seconds[-1],
              'source_unchanged': True, 'replay_exact': True, 'snapshots': len(rows),
              'evaluations_computed': len(cache), 'arm_count': len(s.arms),
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if str(stage.device).startswith('cuda') else 0}
    atomic_write_json(stage.store.run_dir / 'drift_regimes_summary.json', report)
    atomic_write_json(stage.store.run_dir / 'drift_regimes_events.json', events)
    return report, [('drift_regimes_measurements', measurements),
                    ('drift_regimes_summary', stage.store.run_dir / 'drift_regimes_summary.json'),
                    ('drift_regimes_references', stage.store.run_dir / 'drift_regimes_references.json'),
                    ('drift_regimes_events', stage.store.run_dir / 'drift_regimes_events.json')]


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def run_readiness(stage):
    """Declared numerical checks (two dev sequences, ages 0 / 15 min / 24 h) and a timing sample."""
    s = stage.settings
    stage.prefixes()
    stage.deploy()
    dev = stage.evaluation[:READINESS_SEQUENCES]
    cells = stage.cells()
    checks, failures = [], []
    for age in READINESS_AGES:
        clock = stage.release + age
        physical = [c.advance(clock) for c in cells]
        replay = [c.value(clock) for c in stage.cells()] if age else physical
        replay_exact = all(torch.equal(a, b) for a, b in zip(physical, replay))
        view = forward.bind(physical)
        forward.check_view(stage.model, view)
        eight = stage.evaluate(physical, prefix=dev)
        sixteen = stage.evaluate(physical, prefix=dev, sweeps=2 * s.forward_policy.evaluation_sweeps)
        # Training-mode forward (settle, then differentiate) against the no-graph evaluation forward.
        stage.policy.mode(False)
        with torch.no_grad():
            evaluation_logits = forward.suffix_logits(stage.model, dev.student, view)
        norms = [p for n in forward.layer_norms(stage.model) for p in n.parameters()]
        held = [p.detach().clone().requires_grad_(True) for p in physical]
        for p in norms:
            p.requires_grad_(True)
        stage.policy.mode(True)
        with torch.enable_grad():
            training_logits = forward.suffix_logits(stage.model, dev.student, forward.bind(held))
            reference = forward.teacher_logits(stage.teacher, stage.model.start, dev.teacher).log_softmax(-1).detach()
            loss = (reference.exp() * (reference - training_logits.log_softmax(-1))).sum(-1).mean()
            loss.backward()
        stage.policy.mode(False)
        finite_norm_grad = all(bool(torch.isfinite(p.grad).all()) for p in norms)
        finite_physical_grad = all(bool(torch.isfinite(p.grad).all()) for p in held)
        nonzero_norm_grad = any(bool((p.grad != 0).any()) for p in norms)
        for p in norms:
            p.grad = None
            p.requires_grad_(False)
        diagnostics = forward.solver_diagnostics(stage.model, dev, view, stage.policy, sequences=READINESS_SEQUENCES)
        row = {'seconds': age, 'replay_exact': replay_exact,
               'teacher_kl_8': eight['teacher_kl'], 'teacher_kl_16': sixteen['teacher_kl'],
               'kl_change_8_to_16': abs(eight['teacher_kl'] - sixteen['teacher_kl']),
               'train_eval_logit_max_abs': float((training_logits.detach() - evaluation_logits).abs().max()),
               'finite_layernorm_gradients': finite_norm_grad, 'nonzero_layernorm_gradients': nonzero_norm_grad,
               'finite_physical_gradients': finite_physical_grad, 'numerical_diagnostics': diagnostics,
               'bounds': bound_report(stage.law, physical, stage.names, stage.roles, stage.targets)['roles']}
        row['passed'] = bool(replay_exact and finite_norm_grad and finite_physical_grad and nonzero_norm_grad
                             and row['kl_change_8_to_16'] <= READINESS_KL_TOLERANCE
                             and all(math.isfinite(d['maximum_voltage']) and d['minimum_denominator'] > 0
                                     for d in diagnostics))
        if not row['passed']:
            failures.append(age)
        checks.append(row)
        stage.store.append_metric({'phase': 'readiness_check', **stage.identity, **{k: v for k, v in row.items()
                                                                                   if k not in ('numerical_diagnostics', 'bounds')}})
        print(json.dumps({k: v for k, v in row.items() if k not in ('numerical_diagnostics', 'bounds')}), flush=True)
        del held, training_logits, evaluation_logits
    del cells
    timing = timing_sample(stage)
    report = {**stage.identity, 'checks': checks, 'failed_ages': failures, 'passed': not failures,
              'tolerance_kl': READINESS_KL_TOLERANCE, 'cost': stage.cost, 'timing': timing,
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if str(stage.device).startswith('cuda') else 0}
    path = stage.store.run_dir / 'drift_regimes_readiness.json'
    atomic_write_json(path, report)
    return report, [('drift_regimes_readiness', path)]


def timing_sample(stage):
    """Bounded per-method timings on the released state (development seed, not evidence)."""
    s, device = stage.settings, stage.device
    timing = {'deployment_wall_seconds': stage.cost['deployment_wall_seconds']}

    def timed(name, fn):
        _sync(device)
        started = time.monotonic()
        out = fn()
        _sync(device)
        timing[name] = time.monotonic() - started
        stage.store.append_metric({'phase': 'timing', 'operation': name, 'seconds': timing[name]})
        print(json.dumps({'timing': name, 'seconds': timing[name]}), flush=True)
        return out

    cells = stage.cells()
    clock = stage.release + 900.
    physical = timed('advance_to_900s', lambda: [c.advance(clock) for c in cells])
    view = forward.bind(physical)
    timed('evaluation_full_cohort', lambda: stage.evaluate(physical))
    timed('calibration_fit', lambda: forward.fit_calibration(stage.model, view, stage.probe, stage.policy))
    a = s.adaptation
    norms = {id(n): {k: v.clone() for k, v in n.state_dict().items()} for n in forward.layer_norms(stage.model)}
    timed(f'layernorm_{TIMING_ADAM_STEPS}_adam_steps', lambda: forward.train_layernorm(
        stage.model, stage.teacher, stage.hidden, view, stage.policy, steps=TIMING_ADAM_STEPS, batch=a.batch,
        microbatch=a.microbatch, learning_rate=3e-3, seed=a.seed))
    for n in forward.layer_norms(stage.model):
        n.load_state_dict(norms[id(n)])
    timing['layernorm_seconds_per_adam_step'] = timing[f'layernorm_{TIMING_ADAM_STEPS}_adam_steps'] / TIMING_ADAM_STEPS
    law = stage.law
    moments = [forward.fresh_moments(law, t, 1350.) for t in stage.targets]
    pick = torch.arange(min(16, len(stage.hidden)), device=stage.hidden.student.device)
    timed('fisher_round_16_sequences', lambda: forward.fisher_scores(
        stage.model, stage.hidden, pick, physical, stage.targets, [m[0] for m in moments], moments[0][1],
        stage.policy, seed=a.seed))
    del physical, view
    for label, spec in (('rewrite_a9_event', {'fraction': .1, 'attempts': s.deployment_attempts,
                                              'threshold': s.deployment_threshold}),
                        ('rewrite_a10_event', {'fraction': .1, 'attempts': s.deployment_attempts,
                                               'threshold': s.deployment_threshold}),
                        ('rewrite_full_event', {'fraction': 1., 'attempts': s.deployment_attempts,
                                                'threshold': s.deployment_threshold})):
        cells = stage.cells()
        selection = 'precompensated' if 'a10' in label else 'observed'
        timed(label, lambda: arms_module.rewrite(cells, spec, clock, s.verify_seconds, selection=selection,
                                                 history=[]))
        del cells
    cells = stage.cells()
    timed('advance_to_86400s', lambda: [c.advance(stage.release + 86400.) for c in cells])
    del cells
    return timing


def run_train(request):
    stage = Stage(request)
    try:
        if stage.config.stage == 'readiness':
            result, artifacts = run_readiness(stage)
        else:
            result, artifacts = run_screen(stage)
        stage.store.complete(metrics=result, artifacts=[stage.store.artifact_record(p, kind=k) for k, p in artifacts])
        return 0
    except BaseException as error:
        stage.store.fail(error)
        raise
