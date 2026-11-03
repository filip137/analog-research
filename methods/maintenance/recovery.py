"""methods.maintenance.recovery — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from hardware.drift.writable import WritableDrift
from methods.optimizers.masked import MaskedAdam


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
