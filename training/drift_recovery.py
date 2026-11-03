"""Native-stage building blocks for custom drift recovery and maintenance.

Architectures supply observable block forwards and physical encodings. This
engine never consults CMO programming noise, internal solver voltages or test
data for selection. Mirrored verbatim in the DRN worktree.
"""
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import resource
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from training.writable_drift import WritableDrift, MaskedAdam, select_masks
from training.drift_recovery_cache import LogitCache, probe_cache, fit_calibration
from training.drift_recovery_report import durability
from training.drift_recovery_shards import screens, trajectories, read_artifact, select_learning_rate


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def token_cohorts(streams, length, seed, settings):
    def blocks(split):
        stream = streams[split]
        return stream[:len(stream)//length*length].reshape(-1, length)
    train = blocks('train')
    if len(train) < settings.training_sequences or len(blocks('validation')) < settings.validation_sequences:
        raise ValueError('Pinned corpus is too small for the declared cohorts.')
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))[:settings.training_sequences]
    return train[order], blocks('validation')[:settings.validation_sequences], {
        'training_indices': order.tolist(), 'training_tokens_sha256': tensor_hash(train[order]),
        'validation_tokens_sha256': tensor_hash(blocks('validation')[:settings.validation_sequences]),
        'sequence_length': length, 'split_seed': seed}


@torch.no_grad()
def evaluate(adapter, tokens, references, physical=None, correction=None):
    started = time.monotonic()
    ce = kl = 0.
    count = 0
    for i, batch in enumerate(tokens.split(1)):
        output = adapter.forward(batch, physical, correction).float()
        reference = references[i].to(adapter.device)
        if output.shape != reference.shape or not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Invalid recovery logits or reference shape.')
        flat = output.flatten(0, 1)
        labels = batch[:, 1:].to(adapter.device).flatten()
        ce += float(F.cross_entropy(flat, labels, reduction='sum'))
        kl += float(F.kl_div(flat.log_softmax(-1), reference.flatten(0, 1).softmax(-1), reduction='sum'))
        count += labels.numel()
    result = {'cross_entropy': ce/count, 'perplexity': math.exp(ce/count),
              'clean_hwa_kl': kl/count, 'scored_tokens': count, 'sequences': len(tokens)}
    if any(not math.isfinite(v) for v in result.values()): raise FloatingPointError('Nonfinite recovery metrics.')
    result['elapsed_seconds'] = time.monotonic()-started
    return result


def train_event(adapter, cells, mask, tokens, references, correction, settings, learning_rate, budget, progress=None):
    writer = MaskedAdam(cells, mask, learning_rate, budget)
    started = time.monotonic()
    updates = presentations = 0
    losses = []
    # Identical ordered presentations for all policies at a given allowance.
    for epoch in range(settings.passes):
        for start in range(0, len(tokens), settings.batch_size):
            if updates >= settings.max_updates or not writer.can_step(): break
            end = min(start+settings.batch_size, len(tokens))
            loss_sum = 0.
            for first in range(start, end, settings.microbatch_size):
                last = min(first+settings.microbatch_size, end)
                batch = tokens[first:last]
                target = torch.cat([references[i] for i in range(first, last)]).to(adapter.device)
                output = adapter.forward(batch, writer.view(), correction)
                n = output.shape[0]*output.shape[1]
                loss = F.kl_div(output.flatten(0, 1).log_softmax(-1),
                                target.flatten(0, 1).softmax(-1), reduction='sum')/n
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite recovery loss.')
                (loss*((last-first)/(end-start))).backward()
                loss_sum += float(loss.detach())*(last-first)/(end-start)
            writer.step()
            updates += 1
            presentations += end-start
            losses.append(loss_sum)
            if progress: progress({'updates': updates, 'writes': writer.spent, 'loss': loss_sum})
        if updates >= settings.max_updates or not writer.can_step(): break
    result = {'updates': updates, 'cell_writes': writer.spent, 'sequence_presentations': presentations,
              'input_token_presentations': presentations*tokens.shape[1], 'loss_curve': losses,
              'elapsed_seconds': time.monotonic()-started, 'budget': int(budget),
              'budget_unspent': int(budget)-writer.spent,
              'changed_cell_commands': writer.changed}
    del writer
    return result


def refresh(cells, groups, roles, fraction, method, seed):
    """Full pass or a single observed-error 10% pass; no repeated budget burning."""
    if method == 'full_refresh':
        masks = [torch.arange(c.target.numel(), device=c.target.device) for c in cells]
        reads = 0
    else:
        errors = torch.cat([(c.read()-c.target).abs().flatten() for c in cells])
        group = torch.cat([x.flatten().to(errors.device) for x in groups])
        role = torch.cat([x.flatten().to(errors.device) for x in roles])
        decile = torch.cat([((c.target.flatten()-c.law.g_min)/(c.law.g_max-c.law.g_min)*10)
                            .long().clamp(0, 9) for c in cells])
        selected = torch.zeros_like(errors, dtype=torch.bool)
        for g in group.unique():
            ids = torch.nonzero(group == g, as_tuple=False).flatten()
            count = max(1, int(len(ids)*fraction))
            selected[ids[torch.argsort(errors[ids], descending=True, stable=True)[:count]]] = True
        if method == 'random_refresh':
            strata = group.long()*1000 + role.long()*10 + decile
            random_mask = torch.zeros_like(selected)
            rng = torch.Generator(device=errors.device).manual_seed(seed)
            for s in strata[selected].unique(sorted=True):
                ids = torch.nonzero(strata == s, as_tuple=False).flatten()
                count = int((selected & (strata == s)).sum())
                random_mask[ids[torch.randperm(len(ids), generator=rng, device=errors.device)[:count]]] = True
            selected = random_mask
        masks = [torch.nonzero(x, as_tuple=False).flatten() for x in
                 selected.split([c.target.numel() for c in cells])]
        reads = len(errors)
    for cell, mask in zip(cells, masks): cell.write(mask, cell.target.flatten()[mask])
    return {'cell_writes': sum(len(m) for m in masks), 'selection_reads': reads, 'updates': 0}


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint(path, cells, correction, masks, metadata):
    """Persist at maintenance boundaries and continue from the saved state."""
    _save(path, {'schema': 'custom_drift.recovery_state.v1', 'metadata': metadata,
                 'cells': [c.state_dict() for c in cells],
                 'correction': None if correction is None else [(a.cpu(), b.cpu()) for a, b in correction],
                 'masks': {k: [idx.cpu() for idx in values] for k, values in masks.items()}})
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for i, state in enumerate(saved['cells']):
        restored = WritableDrift.restore(state, cells[i].target.device)
        if not torch.equal(cells[i].value(), restored.value()): raise RuntimeError('Held state failed exact replay.')
        future = cells[i].clock+3600
        if not torch.equal(cells[i].value(future), restored.value(future)):
            raise RuntimeError('Future aging failed exact replay.')
        cells[i] = restored
    return cells


def run(adapter, settings, training, validation, evaluations, identity, store, assignment):
    directory = store.run_dir
    started = time.monotonic()
    targets, groups, roles, names = adapter.physical_layout()
    physical_count = sum(t.numel() for t in targets)
    initial_source = adapter.source_hash()
    artifacts = []
    def progress(event, **values):
        store.append_metric({'event': event, **values})
        print(f'{event}: {values}', flush=True)
    def make_cells(rho, law=None):
        return [WritableDrift(t, law=law or settings.law, rho=rho,
                              seed=settings.seed+assignment, name=n) for t, n in zip(targets, names)]
    execution = settings.execution
    phase = execution['phase'] if execution else 'full'
    science_hash = content_hash(settings.science_dict())
    references = {}
    cohorts = {'training': training, 'validation': validation, **evaluations}
    curve_count = settings.curve_validation_sequences or len(validation)
    cohort_hashes = {split: tensor_hash(tokens) for split, tokens in cohorts.items()}
    dependency = None
    cache_start = time.monotonic()
    if phase in ('full', 'prepare'):
        cache_root = directory
        for split, tokens in cohorts.items():
            references[split] = LogitCache.create(directory/'reference_logits'/split, tokens, adapter.clean,
                                {**identity, 'split': split, 'tokens_sha256': cohort_hashes[split],
                                 'source_sha256': settings.source_sha256},
                                lambda n, total, split=split: progress('target_cache', split=split, complete=n, expected=total)
                                if n == total or n % 16 == 0 else None)
        initial = [t.clamp(settings.law.g_min, settings.law.g_max) for t in targets]
        probes = probe_cache(adapter, training[:settings.calibration_sequences], initial, directory/'calibration_probes')
        baseline = {split: evaluate(adapter, tokens[:curve_count] if split == 'validation' else tokens,
                                   references[split], initial)
                    for split, tokens in cohorts.items() if split != 'training'}
        progress('fresh_reference', physical_devices=physical_count, validation=baseline['validation'])
        # Selection always uses the full validation cohort, independent of curve size.
        selection_cells = make_cells(settings.persistent_fractions[0])
        physical = [c.advance(settings.first_intervention_seconds) for c in selection_cells]
        calibration_scores = {}
        for method in ('global', 'channel'):
            corr, _ = fit_calibration(adapter, physical, probes, method)
            calibration_scores[method] = evaluate(adapter, validation, references['validation'], physical, corr)
        calibration_method = min(calibration_scores, key=lambda k: calibration_scores[k]['perplexity'])
        masks, eligibility = select_masks(selection_cells, groups, roles, settings.selection_fraction, settings.selection_seed)
        mask_cpu = {k: [idx.cpu() for idx in v] for k, v in masks.items()}
        del selection_cells, physical, masks, initial
        progress('selection', calibration=calibration_method, eligibility=eligibility)
        _save(directory/'selection_masks.pt', mask_cpu)
        preparation = {'schema': 'custom_drift.preparation.v1', 'settings_sha256': science_hash,
            'architecture': adapter.architecture, 'assignment': assignment, 'identity': identity,
            'cohort_hashes': cohort_hashes, 'baseline': baseline, 'calibration_method': calibration_method,
            'calibration_validation': calibration_scores, 'eligibility': eligibility,
            'masks_sha256': sha256_file(directory/'selection_masks.pt'),
            'cache_manifests': {split: sha256_file(directory/'reference_logits'/split/'manifest.json') for split in cohorts}}
        atomic_write_json(directory/'preparation.json', preparation)
        artifacts += [('drift_preparation', directory/'preparation.json'), ('drift_selection_masks', directory/'selection_masks.pt')]
    else:
        cache_root = Path(execution['prepared_run'])
        preparation = read_artifact(cache_root, 'preparation.json')
        if (preparation['settings_sha256'] != science_hash or preparation['architecture'] != adapter.architecture
                or preparation['assignment'] != assignment or preparation['identity'] != identity
                or preparation['cohort_hashes'] != cohort_hashes):
            raise ValueError('Prepared recovery inputs differ from this shard.')
        dependency = sha256_file(cache_root/'preparation.json')
        for split in cohorts:
            path = cache_root/'reference_logits'/split
            if sha256_file(path/'manifest.json') != preparation['cache_manifests'][split]:
                raise ValueError('Changed reference-cache manifest.')
            references[split] = LogitCache(path)
        probes = cache_root/'calibration_probes'
        if sha256_file(cache_root/'selection_masks.pt') != preparation['masks_sha256']:
            raise ValueError('Changed characterization masks.')
        mask_cpu = torch.load(cache_root/'selection_masks.pt', map_location='cpu', weights_only=True)
        baseline = preparation['baseline']
        calibration_method = preparation['calibration_method']
        calibration_scores = preparation['calibration_validation']
        eligibility = preparation['eligibility']
        atomic_write_json(directory/'dependency.json', {'prepared_run': str(cache_root),
            'preparation_sha256': dependency, 'screen_runs': execution['screen_runs']})
        artifacts.append(('drift_dependency', directory/'dependency.json'))
    cache_seconds = time.monotonic()-cache_start

    # A separate development screen picks one LR for both masks and budgets.
    lr_rows = []
    lr = settings.learning_rate
    if phase == 'trajectory':
        lr, lr_rows = select_learning_rate(settings, dependency, execution['screen_runs'])
    candidates = screens(settings) if phase in ('full', 'screen') else []
    if phase == 'screen':
        candidates = [c for c in candidates if c['key'] == execution['key']]
        if len(candidates) != 1: raise ValueError('Unknown development-screen key.')
    for candidate in candidates:
        cells = make_cells(.9)
        view = [c.advance(settings.first_intervention_seconds) for c in cells]
        correction, _ = fit_calibration(adapter, view, probes, calibration_method)
        mask = [idx.to(adapter.device) for idx in mask_cpu[candidate['mask']]]
        training_row = train_event(adapter, cells, mask, training, references['training'], correction,
                                    settings, candidate['learning_rate'], int(candidate['allowance']*physical_count),
                                    lambda v: progress('screen_training', key=candidate['key'], **v))
        view = [c.value() for c in cells]
        correction, _ = fit_calibration(adapter, view, probes, calibration_method)
        score = evaluate(adapter, validation, references['validation'], view, correction)
        lr_rows.append({k: v for k, v in candidate.items() if k != 'key'} | score | {'training': training_row})
        progress('learning_rate_screen', **{k: v for k, v in lr_rows[-1].items() if k != 'training'})
        if phase == 'screen':
            atomic_write_json(directory/'screen.json', {'key': candidate['key'], 'row': lr_rows[-1],
                'settings_sha256': science_hash, 'preparation_sha256': dependency})
            artifacts.append(('drift_development_screen', directory/'screen.json'))
        del cells, view, correction
    if phase == 'full' and lr is None:
        lr = min(settings.learning_rates, key=lambda v: sum(r['clean_hwa_kl'] for r in lr_rows if r['learning_rate'] == v))
    atomic_write_json(directory/'selection.json', {'calibration_method': calibration_method,
        'calibration_validation': calibration_scores, 'learning_rate': lr, 'lr_development': lr_rows,
        'eligibility': eligibility, 'canary_lr_not_selected': settings.canary})
    artifacts.append(('drift_recovery_selection', directory/'selection.json'))

    rows, trajectory_rows = [], []
    work = trajectories(settings) if phase in ('full', 'trajectory') else []
    if phase == 'trajectory':
        work = [t for t in work if t['key'] == execution['key']]
        if len(work) != 1: raise ValueError('Unknown trajectory key.')
    for item in work:
        rho, no_drift, arm, allowance, protocol, label = (item[k] for k in
            ('rho', 'no_drift', 'arm', 'allowance', 'protocol', 'key'))
        law = replace(settings.law, mean_us=0., spread_us=0.) if no_drift else settings.law
        oracle_cells = make_cells(rho)
        for c in oracle_cells: c.advance(settings.first_intervention_seconds)
        selected, _ = select_masks(oracle_cells, groups, roles, settings.selection_fraction, settings.selection_seed)
        for key in ('stable', 'random'):
            if any(not torch.equal(a.cpu(), b) for a, b in zip(selected[key], mask_cpu[key])):
                raise RuntimeError('Initial observable selection changed across paired mixtures.')
        del oracle_cells
        cells = make_cells(rho, law)
        correction = None
        cumulative_probes = 0
        completed_events = 0
        event_rows = []
        path = directory/'checkpoints'/f'{label}.pt'
        trajectory_started = time.monotonic()
        def measure(seconds, phase):
            view = [c.value() for c in cells]
            metrics = {split: evaluate(adapter, tokens[:curve_count] if split == 'validation' else tokens,
                                           references[split], view, correction)
                       for split, tokens in cohorts.items() if split != 'training' and
                       (split == 'validation' or settings.heldout_evaluation == 'all' or seconds == settings.horizon_seconds)}
            bound_low = sum(int((g == law.g_min).sum()) for g in view)
            bound_high = sum(int((g == law.g_max).sum()) for g in view)
            writes = sum(c.write_commands for c in cells)
            row = {'trajectory': label, 'architecture': adapter.architecture, 'rho': rho,
                   'no_drift': no_drift, 'arm': arm, 'protocol': protocol,
                   'write_allowance': allowance, 'seconds': seconds, 'phase': phase,
                   'metrics': metrics, 'baseline': baseline,
                   'physical_devices': physical_count, 'cell_writes': writes,
                   'array_write_equivalents': writes/physical_count,
                   'distinct_written_cells': sum(int((c.cycles > 0).sum()) for c in cells),
                   'selection_reads': sum(c.selection_reads for c in cells)
                     + (eligibility['selection_reads'] if arm in ('stable_learning', 'random_learning') else 0),
                   'calibration_probe_vectors': cumulative_probes,
                   'at_lower_bound_fraction': bound_low/physical_count,
                   'at_upper_bound_fraction': bound_high/physical_count,
                   'long_time_extrapolation': seconds > 25*3600,
                   'canary': settings.canary}
            rows.append(row)
            atomic_write_json(directory/'drift_recovery_measurements.json', rows)
            progress('recovery_evaluation', trajectory=label, seconds=seconds, phase=phase,
                     writes=writes, validation_ppl=metrics['validation']['perplexity'])
        events = settings.intervention_times(protocol)
        for seconds in settings.times(protocol):
            for cell in cells: cell.advance(seconds)
            measure(seconds, 'before' if seconds in events else 'observation')
            if seconds not in events or arm == 'untreated': continue
            view = [c.value() for c in cells]
            method = ('global' if arm == 'global_calibration' else 'channel'
                      if arm == 'channel_calibration' else calibration_method)
            correction, reads = fit_calibration(adapter, view, probes, method)
            cumulative_probes += reads
            before = sum(c.write_commands for c in cells)
            event_started = time.monotonic()
            event = {'updates': 0, 'cell_writes': 0}
            if arm.endswith('_refresh'):
                event = refresh(cells, groups, roles, settings.selection_fraction, arm,
                                settings.selection_seed+completed_events)
            elif arm.endswith('_learning'):
                policy = arm.removesuffix('_learning')
                event = train_event(adapter, cells, selected[policy], training, references['training'],
                                    correction, settings, lr, int(allowance*physical_count),
                                    lambda v: progress('recovery_training', trajectory=label,
                                                       intervention=completed_events, **v))
            if arm.endswith(('_refresh', '_learning')):
                correction, reads = fit_calibration(adapter, [c.value() for c in cells], probes, method)
                cumulative_probes += reads
            event.update(seconds=seconds, elapsed_seconds=time.monotonic()-event_started)
            if sum(c.write_commands for c in cells)-before != event['cell_writes']:
                raise RuntimeError('Writer and physical counters disagree.')
            event_rows.append(event)
            completed_events += 1
            cells = checkpoint(path, cells, correction, selected,
                               {'settings_sha256': content_hash(settings.to_dict()),
                                'trajectory': label, 'events_completed': completed_events,
                                'events': event_rows, 'learning_rate': lr})
            measure(seconds, 'after')
        # Every final state is independently replayable, including holds.
        cells = checkpoint(path, cells, correction, selected,
                           {'settings_sha256': content_hash(settings.to_dict()), 'trajectory': label,
                            'events_completed': completed_events, 'events': event_rows,
                            'learning_rate': lr})
        artifacts.append((f'drift_recovery_state.{label}', path))
        trajectory_rows.append({'trajectory': label, 'events': event_rows, 'replay_exact': True,
            'elapsed_seconds': time.monotonic()-trajectory_started, 'checkpoint_bytes': path.stat().st_size,
            'final_cell_writes': sum(c.write_commands for c in cells)})
        atomic_write_json(directory/'drift_recovery_trajectories.json', trajectory_rows)
        del cells, correction
        gc.collect()
    if adapter.source_hash() != initial_source: raise RuntimeError('Recovery mutated frozen model parameters.')
    atomic_write_json(directory/'durability.json', durability(rows if settings.heldout_evaluation == 'all' else
        [{**r, 'metrics': {'validation': r['metrics']['validation']}} for r in rows]))
    artifacts.append(('drift_recovery_durability', directory/'durability.json'))
    result = {'architecture': adapter.architecture, 'canary': settings.canary,
              'source_sha256': settings.source_sha256, 'source_unchanged': True, 'execution_phase': phase,
              'settings_sha256': science_hash, 'preparation_sha256': dependency,
              'physical_devices': physical_count, 'initial_device_programs': physical_count,
              'identity': identity, 'settings': settings.to_dict(), 'baseline': baseline,
              'calibration_method': calibration_method, 'learning_rate': lr,
              'trajectories': len(trajectory_rows), 'measurements': len(rows),
              'exact_replays': len(trajectory_rows), 'eligibility': eligibility,
              'cache_bytes': sum(c.bytes for c in references.values()), 'cache_seconds': cache_seconds,
              'cache_load_seconds': sum(c.read_seconds for c in references.values()),
              'cache_bytes_read': sum(c.bytes_read for c in references.values()),
              'disk_bytes': sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()),
              'elapsed_seconds': time.monotonic()-started,
              'peak_cpu_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if str(adapter.device).startswith('cuda') else 0,
              'physical_model': 'custom additive drift; exact endpoint writes; no pulse/energy estimate'}
    atomic_write_json(directory/'drift_recovery_measurements.json', rows)
    atomic_write_json(directory/'drift_recovery_trajectories.json', trajectory_rows)
    artifacts += [('drift_recovery_measurements', directory/'drift_recovery_measurements.json'),
                  ('drift_recovery_trajectories', directory/'drift_recovery_trajectories.json')]
    # Native integrity checks cover each streamed target/probe; no in-memory logit list.
    artifacts += [(f"drift_reference.{p.relative_to(directory).as_posix().replace('/', '.')}", p)
                  for root in (directory/'reference_logits', directory/'calibration_probes')
                  for p in sorted(root.rglob('*')) if p.is_file()]
    return result, artifacts
