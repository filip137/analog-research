"""Native drift-tuning stage: the fault-recovery learner under AIHWKit CMO log-time drift.

One run remaps the frozen DRN (``mapping_scale``), programs one fault-free deployment with
the native fault-aware programming (``program_rewritten``; at 0 % it rewrites programming
outliers only) and releases it at t = 0. Every device-facing forward reads the held state
through the drift transform at its age (``methods.drift_tuning``); writes shift the held
programmed value and never restart the drift clock.

Arms restart from the same released bank (and write RNG), so they are paired, and reads
at equal ages see identical drift draws:

* ``hold`` (A0): untreated drift, evaluated at every ``evaluation_seconds`` age.
* ``noisy`` (L1): the fault-tuning noisy physical learner (thresholded bounded Adam, cosine
  to zero over ``steps``) starting at release; update k reads at k * ``update_seconds``.
  Development KL every ``development_every`` updates, at every grid age inside the learning
  window and at the ``projection_steps`` (the current held state read at
  ``projection_seconds``, without mutating the run); then the post-window grid with no
  further writes, and a no-drift read of the final held state (diagnostic).
"""
from dataclasses import asdict
import time

import torch

import ebl.runtime.opt_cmo_drn.runtime as runtime
from ebl.runtime.opt_cmo_drn.fault_tuning import _store, cohorts, load_samples, plain_hash
from ebl.runtime.store import atomic_write_json, sha256_file
from methods import drift_tuning as dt
from methods import fault_tuning as ft
from methods.programming.cmo import program_rewritten
from networks.opt.physical import CmoDrnBank


class DriftPhysicalArm(ft.PhysicalArm):
    """``PhysicalArm`` whose gradient forward reads the drifted held state at the update's age."""

    def step_at(self, tokens, samples, microbatch, device, transform):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        loss = ft.backward_batch(self.model, self.bank.view(self.model, transform=transform), tokens, samples,
                                 microbatch, device)
        norm = 0.0
        for writer, u in zip(self.writers, self.masters):
            norm += float(u.grad.square().sum())
            writer.step(u.grad)
        self.bank.view(self.model, transform=transform)  # Reject invalid coupled denominators immediately.
        self.model.zero_grad(set_to_none=True)
        return {'loss': loss, 'gradient_norm': norm ** 0.5,
                'rate': self.writers[0].rate(self.writers[0].step_index)}


def run(request):
    c = request.spec
    s, d = c.fault_tuning, c.drift_tuning
    if request.teacher_weights is None or request.weights is None:
        raise ValueError('Drift tuning needs --teacher-weights and the HWA --weights.')
    if any(getattr(request, k, None) is not None for k in ('resume', 'base_weights', 'device_data', 'device_model',
                                                           'deployment', 'device_state')):
        raise ValueError('Drift tuning programs its own fault-free deployment; no state input is accepted.')
    runtime.configure_runtime(c)
    torch.backends.cudnn.allow_tf32 = False
    recovery, development = cohorts(s)
    samples, path = load_samples(s, recovery)
    source_sha = sha256_file(request.weights)
    teacher = runtime.teacher_model(request.teacher_weights, c)
    model = runtime.from_mapped(teacher, c, torch.load(request.weights, map_location='cpu', weights_only=True))
    model = model.to(c.device).eval().requires_grad_(False)
    teacher = teacher.to(c.device)
    device = c.device
    store = _store(request, [{'role': 'teacher_samples', 'path': str(path.resolve()), 'sha256': sha256_file(path),
                              'tensor_sha256': s.samples_sha256}])
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started >= c.training.max_wall_seconds:
            raise TimeoutError('Drift-tuning wall-time budget reached; completed arms are recorded.')

    def evaluate(bank, seconds, label, step=None, kind='grid'):
        check_time()
        before = time.monotonic()
        view = bank.view(model) if seconds is None else bank.view(model, transform=dt.transform(d, seconds))
        metrics = runtime.evaluate(model, teacher, development, device, view, s.evaluation_microbatch)
        metrics['seconds'] = time.monotonic() - before
        row = {'arm': label, 'kind': kind, 'age_seconds': seconds, 'step': step, 'teacher_kl': metrics['teacher_kl'],
               'development': metrics,
               'bounds': None if seconds is None else dt.bound_counts(bank, d, seconds)}
        store.append_metric({'event': 'drift_tuning_evaluation', **row})
        print(f"{label} {kind} t={seconds} step={step} KL={metrics['teacher_kl']:.6f}", flush=True)
        return row

    try:
        mapping = dt.remap(model, d.mapping_scale)
        before = time.monotonic()
        view, receipt, state = program_rewritten(model, c.deployment.assignment_seed, 'none', 0.0, capture_state=True)
        del view
        bank = CmoDrnBank(state, device)
        start_hashes = bank.held_hashes()
        deployment_seconds = time.monotonic() - before
        identity = {'source_weights_sha256': source_sha, 'deployment': asdict(c.deployment), 'mapping': mapping,
                    'start_held_hashes': start_hashes, 'deployment_seconds': deployment_seconds,
                    'recovery_sha256': s.recovery_sha256, 'development_sha256': s.development_sha256,
                    'samples_sha256': s.samples_sha256, 'drift_tuning': asdict(d),
                    'device_name': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'cpu'}
        store.append_metric({'event': 'drift_tuning_start', **identity, 'programming': receipt})
        del bank
        order = ft.batch_indices(s.steps, s.batch, len(recovery), s.order_seed)
        store.append_metric({'event': 'batch_order', 'sha256': plain_hash(torch.stack(order))})

        def fresh_bank():
            bank = CmoDrnBank(state, device)
            if bank.held_hashes() != start_hashes:
                raise ValueError('Released deployment failed held-state identity check.')
            return bank

        rows, hold = [], None
        horizon = d.age(s.steps)
        for arm in sorted(s.arms, key=lambda a: a.method != 'hold'):
            arm_started = time.monotonic()
            bank = fresh_bank()
            if arm.method == 'hold':
                grid = [evaluate(bank, t, arm.label) for t in d.evaluation_seconds]
                if bank.held_hashes() != start_hashes:
                    raise RuntimeError('Evaluation changed the held deployment.')
                row = {'label': arm.label, 'method': arm.method, 'grid': grid, 'costs': bank.costs()}
                hold = {r['age_seconds']: r['teacher_kl'] for r in grid}
            else:
                grid_steps = {d.step_at(t): t for t in d.evaluation_seconds if 0 < t <= horizon}
                development_steps = set(range(s.development_every, s.steps + 1, s.development_every)) \
                    if s.development_every else set()
                learner = DriftPhysicalArm(model, bank, 'noisy', arm.learning_rate, s.steps, None,
                                           c.recovery.threshold_sigma)
                grid = [evaluate(bank, 0.0, arm.label, 0)]
                curve, projections, losses = [], [], []
                for step, index in enumerate(order, start=1):
                    check_time()
                    tick = time.monotonic()
                    record = learner.step_at(recovery[index], samples[index], s.microbatch, device,
                                             dt.transform(d, d.age(step)))
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    losses.append(record['loss'])
                    store.append_metric({'event': 'drift_tuning_update', 'arm': arm.label, 'step': step,
                                         'age_seconds': d.age(step), **record,
                                         'update_seconds': time.monotonic() - tick,
                                         'elapsed_seconds': time.monotonic() - started,
                                         'peak_memory_bytes': torch.cuda.max_memory_allocated()
                                         if str(device).startswith('cuda') else 0, 'costs': bank.costs()})
                    if step in grid_steps:
                        point = evaluate(bank, grid_steps[step], arm.label, step)
                        grid.append(point)
                        curve.append(point)
                    elif step in development_steps:
                        curve.append(evaluate(bank, d.age(step), arm.label, step, 'development'))
                    if step in d.projection_steps:
                        projections.append(evaluate(bank, d.projection_seconds, arm.label, step, 'projection'))
                report = learner.write_report()
                learner.close()
                del learner
                grid += [evaluate(bank, t, arm.label, s.steps) for t in d.evaluation_seconds if t > horizon]
                twin = evaluate(bank, None, arm.label, s.steps, 'no_drift')
                row = {'label': arm.label, 'method': arm.method, 'learning_rate': arm.learning_rate,
                       'updates': s.steps, 'grid': grid, 'curve': curve, 'projections': projections,
                       'final_no_drift': twin, 'first_loss': sum(losses[:10]) / min(10, len(losses)),
                       'last_loss': sum(losses[-10:]) / min(10, len(losses)), **report}
                if report['costs']['write_attempts'] == 0 and hold is not None:
                    # No write: every post-release read must replay the hold arm exactly.
                    mismatched = [r['age_seconds'] for r in grid if r['teacher_kl'] != hold[r['age_seconds']]]
                    row['zero_write_replay'] = {'checked': True, 'mismatched_ages': mismatched}
                    if mismatched:
                        raise RuntimeError(f'A learner with no writes did not replay the hold arm: {mismatched}.')
                else:
                    row['zero_write_replay'] = {'checked': False}
            ages = [r['age_seconds'] for r in row['grid']]
            if ages != list(d.evaluation_seconds):
                raise RuntimeError(f'Arm {arm.label} grid coverage {ages} differs from the declared grid.')
            kl = [r['teacher_kl'] for r in row['grid']]
            row.update(mean_teacher_kl=dt.trapezoid_mean(ages, kl), final_held_hashes=bank.held_hashes(),
                       seconds=time.monotonic() - arm_started, primary_state='held_apparent_drifted')
            out = store.run_dir / f'artifacts/arm-{arm.label}.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out, row)
            store.append_metric({'event': 'drift_tuning_arm', 'label': arm.label,
                                 'mean_teacher_kl': row['mean_teacher_kl'], 'seconds': row['seconds']})
            rows.append({k: v for k, v in row.items() if k not in ('arrays', 'writers')})
            del bank
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
        summary_path = store.run_dir / 'artifacts/summary.json'
        atomic_write_json(summary_path, {'identity': identity, 'programming': receipt, 'arms': rows})
        artifacts = [store.artifact_record(store.run_dir / f'artifacts/arm-{r["label"]}.json', kind='arm_result')
                     for r in rows] + [store.artifact_record(summary_path, kind='summary')]
        store.complete(metrics={'arms': [{'label': r['label'], 'mean_teacher_kl': r['mean_teacher_kl']} for r in rows],
                                **{k: v for k, v in identity.items() if k != 'drift_tuning'}}, artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise
