"""Native drift-tuning stage: one-shot recovery arms under AIHWKit CMO log-time drift.

One run remaps the frozen DRN (``mapping_scale``), programs one fault-free deployment with
the native fault-aware programming (``program_rewritten``; at 0 % it rewrites programming
outliers only) and releases it at t = 0. Every device-facing forward reads the held state
through the drift transform at its age (``methods.drift_tuning``). Under ``write_clock:
continues`` writes shift the held programmed value and never restart the drift clock; under
``reset_absorb`` a write absorbs the drifted read and restarts the written device's clock.

Arms restart from the same released bank (and write RNG), so they are paired, and reads
at equal ages see identical drift draws:

* ``hold`` (A0): untreated drift, evaluated at every ``evaluation_seconds`` age. A hold arm
  listed in ``drift_off_arms`` is A0-twin: the released state read without drift.
* ``noisy`` (L1) and ``child_d`` (Child-Tuning_D): the fault-tuning noisy physical learner
  (thresholded bounded Adam, cosine to zero over ``steps``), dense or restricted to the fixed
  top-``child_fraction`` Fisher mask of each matrix, starting at release; update k reads at
  k * ``update_seconds``. The Fisher is the mean squared batch gradient over the recovery
  pool at the released (t = 0) state, computed once per run and shared by every child_d arm.
* ``calibration`` (per-channel MLP output gain/shift) and ``layernorm`` (affine of the
  LayerNorm in front of each DRN MLP): digital learners trained on the drifted held read on
  the same clock; they never write devices.
* An arm listed in ``drift_off_arms`` trains and is evaluated with drift switched off.

Learners get development KL every ``development_every`` updates, at every grid age inside
the learning window and (physical drifting learners) at the ``projection_steps`` (the
current held state read at ``projection_seconds``, without mutating the run); then the
post-window grid with no further writes, and a no-drift read of the final state.
"""
from dataclasses import asdict
import hashlib
import time

import torch

import ebl.runtime.opt_cmo_drn.runtime as runtime
from ebl.runtime.opt_cmo_drn.fault_tuning import _store, cohorts, load_samples, plain_hash
from ebl.runtime.store import atomic_write_json, sha256_file
from hardware.devices.cmo import HALF_RANGE_US
from hardware.devices.cmo_array import _pack_flags as _pack
from methods import drift_tuning as dt
from methods import fault_tuning as ft
from methods.programming.cmo import program_rewritten
from networks.opt.physical import CmoDrnBank

PHYSICAL = ('noisy', 'child_d')


class DriftPhysicalArm(ft.PhysicalArm):
    """``PhysicalArm`` whose gradient forward reads the drifted held state at the update's age.

    With a ``clock`` (``reset_absorb``) every write port absorbs the drifted read at the
    update's age before writing and restarts the written devices' clocks.
    """

    def __init__(self, model, bank, method, learning_rate, steps, masks=None, threshold_sigma=2.0, *,
                 gate='target_rho', law=None, clock=None):
        factory = None
        if clock is not None:
            def factory(k, array):
                return dt.AbsorbWritePort(array, 1.0, law=law, index=k, clock=clock)
        super().__init__(model, bank, method, learning_rate, steps, masks, threshold_sigma, gate=gate,
                         port_factory=factory)
        self.clock = clock

    def step_at(self, tokens, samples, microbatch, device, transform, seconds=None):
        if self.clock is not None:
            if seconds is None:
                raise ValueError('An absorbing drift learner needs the read time of each update.')
            for port in self.ports:
                port.seconds = seconds
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        loss = ft.backward_batch(self.model, self.bank.view(self.model, transform=transform), tokens, samples,
                                 microbatch, device)
        norm = 0.0
        for k, (writer, u) in enumerate(zip(self.writers, self.masters)):
            gradient = u.grad
            if self.masks is not None:
                gradient = torch.where(self.masks[k], gradient, torch.zeros_like(gradient))
            norm += float(gradient.square().sum())
            writer.step(gradient)
        self.bank.view(self.model, transform=transform)  # Reject invalid coupled denominators immediately.
        self.model.zero_grad(set_to_none=True)
        return {'loss': loss, 'gradient_norm': norm ** 0.5,
                'rate': self.writers[0].rate(self.writers[0].step_index)}


def model_digest(model):
    """Digest of every model parameter and buffer; no arm may leave the model changed."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


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
    cuda = str(device).startswith('cuda')
    campaign = isinstance(d, dt.CampaignDriftTuning)
    off = set(d.off_labels)
    absorbing = d.write_clock == 'reset_absorb'
    store = _store(request, [{'role': 'teacher_samples', 'path': str(path.resolve()), 'sha256': sha256_file(path),
                              'tensor_sha256': s.samples_sha256}])
    started = time.monotonic()

    def check_time():
        if time.monotonic() - started >= c.training.max_wall_seconds:
            raise TimeoutError('Drift-tuning wall-time budget reached; completed arms are recorded.')

    def evaluate(bank, seconds, label, step=None, kind='grid', clock=None, drift=True):
        check_time()
        before = time.monotonic()
        if bank is None:
            view = None
        elif seconds is None or not drift:
            view = bank.view(model)
        else:
            view = bank.view(model, transform=dt.transform(d, seconds, clock))
        metrics = runtime.evaluate(model, teacher, development, device, view, s.evaluation_microbatch)
        metrics['seconds'] = time.monotonic() - before
        row = {'arm': label, 'kind': kind, 'age_seconds': seconds, 'step': step, 'teacher_kl': metrics['teacher_kl'],
               'development': metrics,
               'bounds': None if seconds is None or bank is None or not drift else
               dt.bound_counts(bank, d, seconds, clock)}
        if not drift:
            row['drift'] = False
        store.append_metric({'event': 'drift_tuning_evaluation', **row})
        print(f"{label} {kind} t={seconds} step={step} KL={metrics['teacher_kl']:.6f}"
              f"{'' if drift else ' (drift off)'}", flush=True)
        return row

    try:
        clean = {}
        if campaign:
            # Per-model clean references (cmo-drift-recovery item 6): the source DRN with its ideal
            # weights, then the exactly programmed ratio-mapped targets (view None reads the targets).
            clean['source_ideal'] = evaluate(None, None, 'source_ideal', kind='clean_reference')
        mapping = dt.remap(model, d.mapping_scale)
        if campaign:
            clean['mapped_targets'] = evaluate(None, None, 'mapped_targets', kind='clean_reference')
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
                    'device_name': torch.cuda.get_device_name(device) if cuda else 'cpu'}
        if campaign:
            gate_q = 2.0 * float(dt.RecoveryWritePort(bank.arrays[0], 1.0).nominal_write_std_q())
            identity.update(write_clock=d.write_clock, write_gate=d.gate,
                            write_gate_us=None if d.gate == 'target_rho' else gate_q * HALF_RANGE_US,
                            drift_off_arms=sorted(off),
                            clean_reference={k: v['teacher_kl'] for k, v in clean.items()})
        store.append_metric({'event': 'drift_tuning_start', **identity, 'programming': receipt})
        del bank
        order = ft.batch_indices(s.steps, s.batch, len(recovery), s.order_seed)
        store.append_metric({'event': 'batch_order', 'sha256': plain_hash(torch.stack(order))})
        reference_model = model_digest(model) if campaign else None

        def fresh_bank():
            bank = CmoDrnBank(state, device)
            if bank.held_hashes() != start_hashes:
                raise ValueError('Released deployment failed held-state identity check.')
            return bank

        rows, hold, artifacts = [], None, []
        masks = fisher_record = None
        horizon = d.age(s.steps)
        for arm in sorted(s.arms, key=lambda a: (a.method != 'hold', a.label in off)):
            arm_started = time.monotonic()
            drift = arm.label not in off
            bank = fresh_bank()
            if arm.method == 'hold':
                grid = [evaluate(bank, t, arm.label, drift=drift) for t in d.evaluation_seconds]
                if bank.held_hashes() != start_hashes:
                    raise RuntimeError('Evaluation changed the held deployment.')
                row = {'label': arm.label, 'method': arm.method, 'grid': grid, 'costs': bank.costs()}
                if drift:
                    hold = {r['age_seconds']: r['teacher_kl'] for r in grid}
                else:
                    row['drift'] = False
            else:
                if arm.method == 'child_d' and masks is None:
                    fisher_started = time.monotonic()

                    def progress(done, total):
                        if done % 16 == 0 or done == total:
                            store.append_metric({'event': 'fisher_progress', 'batches': done, 'of': total,
                                                 'seconds': time.monotonic() - fisher_started})
                        check_time()
                    # At release (t = 0) the drifted read is the held state itself.
                    scores = ft.fisher_scores(model, bank, recovery, samples, s.fisher_batch, s.microbatch, device,
                                              progress)
                    if bank.held_hashes() != start_hashes:
                        raise RuntimeError('Fisher pass changed the held deployment.')
                    masks = ft.child_masks(scores, s.child_fraction)
                    fisher_record = {
                        'fraction': s.child_fraction, 'fisher_batch': s.fisher_batch, 'state': 'released_t0',
                        'batches': len(recovery) // s.fisher_batch, 'seconds': time.monotonic() - fisher_started,
                        'arrays': [{'name': a.name, 'devices': m.numel(), 'selected': int(m.sum()),
                                    'threshold': float(sc[m].min()), 'fisher_mean': float(sc.mean()),
                                    'fisher_selected_share': float(sc[m].sum() / sc.sum().clamp_min(1e-30))}
                                   for a, m, sc in zip(bank.arrays, masks, scores)],
                        'mask_sha256': plain_hash(torch.cat([m.flatten() for m in masks]))}
                    mask_path = store.run_dir / 'artifacts/child_d_masks.pt'
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    runtime.save(mask_path, {'schema': 'opt_cmo_drn.child_d_masks.v1', **fisher_record,
                                             'start_held_hashes': start_hashes,
                                             'masks': [_pack(m) for m in masks],
                                             'shapes': [list(m.shape) for m in masks]})
                    artifacts.append(store.artifact_record(mask_path, kind='child_d_masks'))
                    store.append_metric({'event': 'child_d_fisher', **fisher_record})
                    del scores
                physical = arm.method in PHYSICAL
                clock = dt.WriteClock(bank) if physical and drift and absorbing else None

                def reader(seconds):
                    return dt.transform(d, seconds, clock) if drift else None

                if physical:
                    learner = DriftPhysicalArm(model, bank, arm.method, arm.learning_rate, s.steps,
                                               masks if arm.method == 'child_d' else None, c.recovery.threshold_sigma,
                                               gate=d.gate, law=d, clock=clock)
                elif arm.method == 'calibration':
                    learner = ft.CalibrationArm(model, bank, arm.learning_rate, s.steps)
                elif arm.method == 'layernorm':
                    learner = ft.LayerNormArm(model, bank, arm.learning_rate, s.steps)
                else:
                    raise ValueError(f'Unsupported drift-tuning arm {arm.method}.')
                grid_steps = {d.step_at(t): t for t in d.evaluation_seconds if 0 < t <= horizon}
                development_steps = set(range(s.development_every, s.steps + 1, s.development_every)) \
                    if s.development_every else set()
                projecting = physical and drift
                grid = [evaluate(bank, 0.0, arm.label, 0, clock=clock, drift=drift)]
                curve, projections, losses, ticks = [], [], [], []
                for step, index in enumerate(order, start=1):
                    check_time()
                    tick = time.monotonic()
                    if physical:
                        record = learner.step_at(recovery[index], samples[index], s.microbatch, device,
                                                 reader(d.age(step)), d.age(step))
                    else:
                        record = learner.step(recovery[index], samples[index], s.microbatch, device,
                                              transform=reader(d.age(step)))
                    if cuda:
                        torch.cuda.synchronize()
                    losses.append(record['loss'])
                    ticks.append(time.monotonic() - tick)
                    store.append_metric({'event': 'drift_tuning_update', 'arm': arm.label, 'step': step,
                                         'age_seconds': d.age(step), **record, 'update_seconds': ticks[-1],
                                         'elapsed_seconds': time.monotonic() - started,
                                         'peak_memory_bytes': torch.cuda.max_memory_allocated() if cuda else 0,
                                         **({'costs': bank.costs()} if physical else {})})
                    if step in grid_steps:
                        point = evaluate(bank, grid_steps[step], arm.label, step, clock=clock, drift=drift)
                        grid.append(point)
                        curve.append(point)
                    elif step in development_steps:
                        curve.append(evaluate(bank, d.age(step), arm.label, step, 'development', clock, drift))
                    if projecting and step in d.projection_steps:
                        projections.append(evaluate(bank, d.projection_seconds, arm.label, step, 'projection', clock))
                report = learner.write_report() if physical else {'costs': bank.costs()}
                if not campaign:
                    learner.close()  # Historical order: physical learners close before the post-window grid.
                grid += [evaluate(bank, t, arm.label, s.steps, clock=clock, drift=drift)
                         for t in d.evaluation_seconds if t > horizon]
                twin = evaluate(bank, None, arm.label, s.steps, 'no_drift')
                row = {'label': arm.label, 'method': arm.method, 'learning_rate': arm.learning_rate,
                       'updates': s.steps, 'grid': grid, 'curve': curve, 'projections': projections,
                       'final_no_drift': twin, 'first_loss': sum(losses[:10]) / min(10, len(losses)),
                       'last_loss': sum(losses[-10:]) / min(10, len(losses)), **report}
                if campaign:
                    row['mean_update_seconds'] = sum(ticks) / len(ticks)
                    if not drift:
                        row['drift'] = False
                    if physical:
                        row['write_gate'] = d.gate
                        if arm.method == 'child_d':
                            attempts = [w.attempt_counts for w in learner.writers]
                            row['mask_sha256'] = fisher_record['mask_sha256']
                            row['writes_outside_mask'] = sum(int(a[~m].sum()) for a, m in zip(attempts, masks))
                            if row['writes_outside_mask']:
                                raise RuntimeError('Child-Tuning_D wrote a device outside its fixed mask.')
                        if clock is not None:
                            row['write_clock'] = clock.summary()
                            if row['write_clock']['reset_devices'] != report['touched_cells']:
                                raise RuntimeError('Write-clock resets differ from the written devices.')
                    else:
                        count = sum(p.numel() for p in learner.adapters.parameters())
                        row.update(trained_digital_parameters=count,
                                   added_digital_parameters=0 if arm.method == 'layernorm' else count)
                        adapter_path = store.run_dir / f'artifacts/{arm.method}-{arm.label}.pt'
                        adapter_path.parent.mkdir(parents=True, exist_ok=True)
                        runtime.save(adapter_path, {'schema': f'opt_cmo_drn.drift_{arm.method}.v1',
                                                    'state': learner.state()})
                        artifacts.append(store.artifact_record(adapter_path, kind=f'drift_{arm.method}'))
                        if bank.costs()['write_attempts'] != 0 or bank.held_hashes() != start_hashes:
                            raise RuntimeError('Digital correction changed the held physical state.')
                    learner.close()
                    if model_digest(model) != reference_model:
                        raise RuntimeError(f'Arm {arm.label} left the model changed after closing.')
                del learner
                if physical and drift and report['costs']['write_attempts'] == 0 and hold is not None:
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
            if campaign and row['grid'][0]['teacher_kl'] != hold[0.0]:
                # Every arm starts from the same released state with identity corrections (A0-twin included).
                raise RuntimeError(f'Arm {arm.label} does not start from the released state of A0.')
            kl = [r['teacher_kl'] for r in row['grid']]
            row.update(mean_teacher_kl=dt.trapezoid_mean(ages, kl), final_held_hashes=bank.held_hashes(),
                       seconds=time.monotonic() - arm_started, primary_state='held_apparent_drifted')
            if campaign:
                costs, last = row['costs'], row['grid'][-1]['bounds']
                row.update(writes_per_device=costs['attempts_per_device'],
                           fraction_written=row.get('touched_cells', 0) / costs['physical_devices'],
                           at_drift_min_fraction_final=None if last is None else last['at_drift_min_fraction'])
            out = store.run_dir / f'artifacts/arm-{arm.label}.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out, row)
            store.append_metric({'event': 'drift_tuning_arm', 'label': arm.label,
                                 'mean_teacher_kl': row['mean_teacher_kl'], 'seconds': row['seconds']})
            rows.append({k: v for k, v in row.items() if k not in ('arrays', 'writers')})
            del bank
            if cuda:
                torch.cuda.empty_cache()
        summary_path = store.run_dir / 'artifacts/summary.json'
        summary = {'identity': identity, 'programming': receipt, 'arms': rows}
        if campaign:
            summary.update(clean_reference=clean, child_d_fisher=fisher_record)
        atomic_write_json(summary_path, summary)
        artifacts = [store.artifact_record(store.run_dir / f'artifacts/arm-{r["label"]}.json', kind='arm_result')
                     for r in rows] + artifacts + [store.artifact_record(summary_path, kind='summary')]
        store.complete(metrics={'arms': [{'label': r['label'], 'mean_teacher_kl': r['mean_teacher_kl']} for r in rows],
                                **{k: v for k, v in identity.items() if k != 'drift_tuning'}}, artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise
