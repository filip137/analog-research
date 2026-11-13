"""Native fault-recovery paper LR screen on one saved, canonical CMO DRN deployment.

One run reads the fault-aware (partner-rewrite) deployment and runs its declared arms
in turn. Every arm restarts from the saved bank, including its device write RNG, so the
arms are paired. Learning arms take ``steps`` updates of ``batch`` recovery sequences in
one shared sampled order, with the fixed teacher samples as targets and a cosine decay
to zero. Development KL against the full FP32 teacher is evaluated at the start, every
``development_every`` updates and at the fixed final endpoint (no checkpoint selection).
Each completed arm writes its result row and artifact before the next arm starts.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch

import ebl.runtime.opt_cmo_drn.runtime as runtime
from hardware.devices.cmo_array import _pack_flags as _pack
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.opt_cmo_drn import EXPERIMENT_ID
from methods import fault_tuning as ft
from networks.opt.data import blocks, load_corpus
from networks.opt.physical import CmoDrnBank


def plain_hash(value):
    """Plain-bytes SHA-256 (crossbar and drift-regimes cohort convention)."""
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cohorts(settings):
    """Recovery cohort, development cohort and fixed teacher samples, each pinned by hash."""
    splits = load_corpus(settings.corpus, settings.corpus_sha256)
    length = settings.sequence_length
    stream = splits[settings.development_split]
    first = settings.development_start * length
    last = first + settings.development_sequences * length
    if len(stream) < last:
        raise ValueError('Development corpus is too small for the declared cohort.')
    development = stream[first:last].reshape(settings.development_sequences, length).contiguous()
    if plain_hash(development) != settings.development_sha256:
        raise ValueError('Development cohort hash mismatch.')
    available = blocks(splits[settings.recovery_split], length, len(splits[settings.recovery_split]) // length)
    if len(available) < settings.recovery_sequences:
        raise ValueError('Recovery corpus is too small for the declared cohort.')
    order = torch.randperm(len(available), generator=torch.Generator().manual_seed(settings.recovery_seed))
    recovery = available[order[:settings.recovery_sequences]].contiguous()
    if plain_hash(recovery) != settings.recovery_sha256:
        raise ValueError('Recovery cohort hash mismatch.')
    return recovery, development


def samples_path(settings):
    root = os.environ.get('EBL_CORPUS_ROOT')
    if not root:
        raise ValueError('Expected EBL_CORPUS_ROOT holding the pinned teacher-sample file.')
    return Path(root) / settings.samples_file


def load_samples(settings, recovery):
    path = samples_path(settings)
    samples = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(samples, torch.Tensor) or samples.dtype != torch.int64:
        raise ValueError('Expected an int64 teacher-sample tensor.')
    expected = (len(recovery), settings.sequence_length - 1, settings.samples_count)
    if tuple(samples.shape) != expected:
        raise ValueError(f'Expected teacher samples of shape {expected}; received {tuple(samples.shape)}.')
    if plain_hash(samples) != settings.samples_sha256:
        raise ValueError('Teacher-sample tensor hash mismatch; the declared draw is required, never a redraw.')
    return samples, path


def _store(request, extra):
    identities = []
    snapshot = os.environ.get('OPT_DRN_SOURCE_SNAPSHOT')
    if snapshot:
        snapshot = Path(snapshot)
        receipt = json.loads(snapshot.read_text())
        for name, digest in receipt['files'].items():
            path = (snapshot.parent / name).resolve()
            if not path.is_relative_to(snapshot.parent.resolve()) or sha256_file(path) != digest:
                raise ValueError(f'Source snapshot mismatch: {name}.')
        identities.append({'role': 'source_snapshot', 'path': str(snapshot), 'sha256': sha256_file(snapshot)})
    for name in ('teacher_weights', 'weights', 'device_state'):
        path = getattr(request, name, None)
        if path:
            identities.append({'role': name, 'path': str(path.resolve()), 'sha256': sha256_file(path)})
    identities.extend(extra)
    store = RunStore.create(output_root=request.output_dir, experiment_id=EXPERIMENT_ID,
                            resolved_config=asdict(request.spec), command=request.command, repo_root=runtime.ROOT,
                            input_artifacts=identities, resume_capability='unsupported',
                            run_id=os.environ.get('OPT_DRN_RUN_ID'))
    print(f'RUN_DIR={store.run_dir}', flush=True)
    return store


def run(request):
    c = request.spec
    s = c.fault_tuning
    if request.teacher_weights is None or request.weights is None or request.device_state is None:
        raise ValueError('Fault tuning needs --teacher-weights, the HWA --weights and the deployed --device-state.')
    if any(getattr(request, k, None) is not None for k in ('resume', 'base_weights', 'device_data', 'device_model',
                                                           'deployment')):
        raise ValueError('Unexpected fault-tuning input; arms restart from the saved deployment.')
    runtime.configure_runtime(c)
    torch.backends.cudnn.allow_tf32 = False
    recovery, development = cohorts(s)
    samples, path = load_samples(s, recovery)
    source_sha = sha256_file(request.weights)
    physical = torch.load(request.device_state, map_location='cpu', weights_only=True)
    if physical.get('schema') != 'opt_cmo_drn.deployment.v1':
        raise ValueError('Fault tuning restores only a canonical fault-aware deployment.')
    if physical['source_weights_sha256'] != source_sha or physical['deployment'] != asdict(c.deployment):
        raise ValueError('Deployment source weights or fault case differ from this run.')
    deployed_sha = sha256_file(request.device_state)
    teacher = runtime.teacher_model(request.teacher_weights, c)
    model = runtime.from_mapped(teacher, c, torch.load(request.weights, map_location='cpu', weights_only=True))
    model = model.to(c.device).eval().requires_grad_(False)
    teacher = teacher.to(c.device)
    store = _store(request, [{'role': 'teacher_samples', 'path': str(path.resolve()), 'sha256': sha256_file(path),
                              'tensor_sha256': s.samples_sha256}])
    started = time.monotonic()
    device = c.device

    def fresh_bank():
        bank = CmoDrnBank(physical['bank'], device)
        if bank.held_hashes() != physical['start_held_hashes']:
            raise ValueError('Canonical deployment failed held-state identity check.')
        return bank

    def evaluate(bank):
        before = time.monotonic()
        metrics = runtime.evaluate(model, teacher, development, device, bank.view(model), s.evaluation_microbatch)
        metrics['seconds'] = time.monotonic() - before
        return metrics

    def check_time():
        if time.monotonic() - started >= c.training.max_wall_seconds:
            raise TimeoutError('Fault-tuning wall-time budget reached; completed arms are recorded.')

    try:
        identity = {'deployed_sha256': deployed_sha, 'source_weights_sha256': source_sha,
                    'deployment': physical['deployment'], 'start_held_hashes': physical['start_held_hashes'],
                    'recovery_sha256': s.recovery_sha256, 'development_sha256': s.development_sha256,
                    'samples_sha256': s.samples_sha256, 'programming': physical['programming']}
        store.append_metric({'event': 'fault_tuning_start', **identity,
                             'device_name': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'cpu'})
        order = ft.batch_indices(s.steps, s.batch, len(recovery), s.order_seed)
        store.append_metric({'event': 'batch_order', 'sha256': plain_hash(torch.stack(order))})
        start_metrics = None
        masks = fisher_record = None
        rows, artifacts = [], []
        for arm in s.arms:
            check_time()
            arm_started = time.monotonic()
            bank = fresh_bank()
            if arm.method == 'naive_hold':
                from ebl.runtime.opt_cmo_drn.recovery import restore_initial_programming
                restore_initial_programming(bank)
                final = evaluate(bank)
                row = {'label': arm.label, 'method': arm.method, 'updates': 0, 'development': final,
                       'costs': bank.costs()}
            elif arm.method == 'hold':
                if start_metrics is None:
                    start_metrics = evaluate(bank)
                row = {'label': arm.label, 'method': arm.method, 'updates': 0, 'development': start_metrics,
                       'costs': bank.costs()}
            else:
                if start_metrics is None:
                    start_metrics = evaluate(bank)
                    store.append_metric({'event': 'deployed_development', 'development': start_metrics})
                if arm.method == 'child_d' and masks is None:
                    fisher_started = time.monotonic()
                    def progress(done, total):
                        if done % 16 == 0 or done == total:
                            store.append_metric({'event': 'fisher_progress', 'batches': done, 'of': total,
                                                 'seconds': time.monotonic() - fisher_started})
                        check_time()
                    scores = ft.fisher_scores(model, bank, recovery, samples, s.fisher_batch, s.microbatch, device,
                                              progress)
                    if bank.held_hashes() != physical['start_held_hashes']:
                        raise RuntimeError('Fisher pass changed the held deployment.')
                    masks = ft.child_masks(scores, s.child_fraction)
                    mask_path = store.run_dir / 'artifacts/child_d_masks.pt'
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    fisher_record = {
                        'fraction': s.child_fraction, 'fisher_batch': s.fisher_batch,
                        'batches': len(recovery) // s.fisher_batch, 'seconds': time.monotonic() - fisher_started,
                        'arrays': [{'name': a.name, 'devices': m.numel(), 'selected': int(m.sum()),
                                    'stuck_selected': int((m & a._mask).sum()),
                                    'threshold': float(sc[m].min()), 'fisher_mean': float(sc.mean()),
                                    'fisher_selected_share': float(sc[m].sum() / sc.sum().clamp_min(1e-30))}
                                   for a, m, sc in zip(bank.arrays, masks, scores)],
                        'mask_sha256': plain_hash(torch.cat([m.flatten() for m in masks]))}
                    runtime.save(mask_path, {'schema': 'opt_cmo_drn.child_d_masks.v1', **fisher_record,
                                             'deployed_sha256': deployed_sha,
                                             'masks': [_pack(m) for m in masks],
                                             'shapes': [list(m.shape) for m in masks]})
                    artifacts.append(store.artifact_record(mask_path, kind='child_d_masks'))
                    store.append_metric({'event': 'child_d_fisher', **fisher_record})
                    del scores
                if arm.method == 'lora':
                    learner = ft.LoraArm(model, bank, arm.learning_rate, s.steps, arm.rank, s.lora_seed)
                else:
                    learner = ft.PhysicalArm(model, bank, arm.method, arm.learning_rate, s.steps,
                                             masks if arm.method == 'child_d' else None, c.recovery.threshold_sigma)
                curve = [{'step': 0, 'teacher_kl': start_metrics['teacher_kl']}]
                losses = []
                for step, index in enumerate(order, start=1):
                    check_time()
                    before = time.monotonic()
                    record = learner.step(recovery[index], samples[index], s.microbatch, device)
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    losses.append(record['loss'])
                    store.append_metric({'event': 'fault_tuning_update', 'arm': arm.label, 'step': step, **record,
                                         'update_seconds': time.monotonic() - before,
                                         'elapsed_seconds': time.monotonic() - started,
                                         'peak_memory_bytes': torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
                                         **({} if arm.method == 'lora' else {'costs': bank.costs()})})
                    if step < s.steps and s.development_every and step % s.development_every == 0:
                        scored = evaluate(bank)
                        curve.append({'step': step, 'teacher_kl': scored['teacher_kl']})
                        store.append_metric({'event': 'fault_tuning_development', 'arm': arm.label, 'step': step,
                                             'development': scored})
                final = evaluate(bank)
                curve.append({'step': s.steps, 'teacher_kl': final['teacher_kl']})
                row = {'label': arm.label, 'method': arm.method, 'learning_rate': arm.learning_rate,
                       'updates': s.steps, 'development': final, 'curve': curve,
                       'first_loss': sum(losses[:10]) / min(10, len(losses)),
                       'last_loss': sum(losses[-10:]) / min(10, len(losses))}
                if arm.method == 'lora':
                    row.update(rank=arm.rank, alpha=arm.rank, costs=bank.costs(),
                               adapter_parameters=sum(p.numel() for p in learner.adapters.parameters()))
                    adapter_path = store.run_dir / f'artifacts/lora-{arm.label}.pt'
                    adapter_path.parent.mkdir(parents=True, exist_ok=True)
                    runtime.save(adapter_path, {'schema': 'opt_cmo_drn.digital_lora.v1', 'rank': arm.rank,
                                                'alpha': arm.rank, 'seed': s.lora_seed, 'state': learner.state()})
                    artifacts.append(store.artifact_record(adapter_path, kind='digital_lora'))
                    if bank.costs()['write_attempts'] != 0 or bank.held_hashes() != physical['start_held_hashes']:
                        raise RuntimeError('Digital LoRA changed the held physical state.')
                else:
                    row.update(learner.write_report())
                    if arm.method == 'child_d':
                        row['mask_sha256'] = fisher_record['mask_sha256']
                learner.close()
                del learner
            row.update(final_held_hashes=bank.held_hashes(), seconds=time.monotonic() - arm_started,
                       primary_state='held_apparent')
            path = store.run_dir / f'artifacts/arm-{arm.label}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, row)
            artifacts.append(store.artifact_record(path, kind='arm_result'))
            store.append_metric({'event': 'fault_tuning_arm', **{k: v for k, v in row.items() if k != 'arrays'}})
            rows.append({k: v for k, v in row.items() if k not in ('arrays', 'writers')})
            del bank
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
        summary = store.run_dir / 'artifacts/summary.json'
        atomic_write_json(summary, {'identity': identity, 'start_development': start_metrics,
                                    'child_d_fisher': fisher_record, 'arms': rows})
        artifacts.append(store.artifact_record(summary, kind='summary'))
        store.complete(metrics={'arms': rows, 'start_development': start_metrics, **identity}, artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise
