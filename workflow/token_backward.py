"""Token-selective recovery for a held crossbar and digital LoRA.

Full forward context, with separate autograd/no-grad streams through every
suffix attention, normalization and MLP. Inspired by TokenTune; importance
selection uses TokenSeek's attention/gradient combination with explicit L1
gradient magnitude (the authors' released code instead uses a signed sum).
"""
import contextlib
import time

import torch
from torch.nn import functional as F


# Issue #27 selection controls. random_fixed: uniform positions drawn once per event
# and reused, like importance. *_sampled: positions drawn with replacement from a
# defensive mixture of the event score and uniform, redrawn each presentation, with
# weights making the loss an unbiased estimate of the mean over all scored positions.
SELECTION_MODES = ('random_fixed', 'importance_sampled', 'gradient_sampled')
SAMPLED_MODES = ('importance_sampled', 'gradient_sampled')
UNIFORM_MIX = .2


def parse_mode(value):
    if value == 'full':
        return 'full', None
    if not isinstance(value, str) or value.count(':') != 1:
        raise ValueError('Expected full or random/importance/loss_only/context:<count>.')
    mode, count = value.split(':')
    if mode not in ('random', 'importance', 'loss_only', 'context', *SELECTION_MODES):
        raise ValueError('Unknown token backward mode.')
    if not count.isdigit() or str(int(count)) != count or int(count) < 1:
        raise ValueError('Expected a positive canonical token count.')
    return mode, int(count)


def take(x, positions):
    return x.gather(1, positions[..., None].expand(-1, -1, x.shape[-1]))


def complement(positions, length):
    mask = torch.ones(positions.shape[0], length, dtype=torch.bool, device=positions.device)
    mask.scatter_(1, positions, False)
    return torch.arange(length, device=positions.device).expand_as(mask)[mask].reshape(positions.shape[0], -1)


def join(selected, other, positions, rest, length):
    out = other.new_zeros(other.shape[0], length, other.shape[-1])
    out.scatter_(1, rest[..., None].expand_as(other), other)
    return out.scatter(1, positions[..., None].expand_as(selected), selected)


def sparse_attention(layer, selected, other, positions, rest):
    """Only selected query outputs and selected input K/V carry gradients.

    All keys/values remain accessible, with the original causal positions.
    The no-grad query branch retains no attention backward graph.
    """
    attention = layer.self_attn
    b, k, d = selected.shape
    length = k + other.shape[1]
    normalized = layer.self_attn_layer_norm(selected)
    with torch.no_grad():
        normalized_other = layer.self_attn_layer_norm(other)
        key_other = attention.k_proj(normalized_other)
        value_other = attention.v_proj(normalized_other)
    keys = join(attention.k_proj(normalized), key_other, positions, rest, length)
    values = join(attention.v_proj(normalized), value_other, positions, rest, length)
    def heads(x):
        return x.reshape(b, -1, attention.heads, attention.head_dim).transpose(1, 2)
    keys, values = heads(keys), heads(values)
    def query(x, pos, key, value):
        q = heads(attention.q_proj(x) * attention.head_dim ** -0.5)
        scores = q @ key.transpose(-1, -2)
        mask = torch.arange(length, device=x.device)[None, None, None, :] > pos[:, None, :, None]
        weights = scores.masked_fill(mask, -torch.inf).softmax(-1)
        return attention.out_proj((weights @ value).transpose(1, 2).reshape(b, -1, d))
    attended = selected + query(normalized, positions, keys, values)
    with torch.no_grad():
        attended_other = other + query(normalized_other, rest, keys.detach(), values.detach())
    return attended, attended_other


def selected_logits(network, hidden, q, positions, return_hidden=False):
    """Forward matches the full suffix at selected positions, up to rounding."""
    digital = network.teacher if network.digital is None else network.digital
    length = hidden.shape[1]
    rest = complement(positions, length)
    selected, other = take(hidden, positions), take(hidden, rest).detach()
    for k, i in enumerate(range(network.start, len(network.teacher.layers))):
        selected, other = sparse_attention(digital.layers[i], selected, other, positions, rest)
        def mlp(x):
            y = network._matrix(network.mlp_norms[k](x), 2*k, q)
            return x + network._matrix(F.relu(y), 2*k+1, q)
        selected = mlp(selected)
        with torch.no_grad():
            other = mlp(other)
    if return_hidden:
        return join(selected, other, positions, rest, length)
    return digital.logits(selected)


def full_hidden(network, hidden, q, stop_before_last=False):
    digital = network.teacher if network.digital is None else network.digital
    end = len(network.teacher.layers) - int(stop_before_last)
    for k, i in enumerate(range(network.start, end)):
        hidden = digital.layers[i].attention(hidden)
        y = network._matrix(network.mlp_norms[k](hidden), 2*k, q)
        hidden = hidden + network._matrix(F.relu(y), 2*k+1, q)
    return hidden


def attention_importance(layer, hidden):
    attn = layer.self_attn
    x = layer.self_attn_layer_norm(hidden)
    b, s, _ = x.shape
    def heads(y):
        return y.reshape(b, s, attn.heads, attn.head_dim).transpose(1, 2)
    query = heads(attn.q_proj(x) * attn.head_dim**-0.5)
    key = heads(attn.k_proj(x))
    mask = torch.ones(s, s, dtype=torch.bool, device=x.device).triu(1)
    weights = (query @ key.transpose(-1, -2)).masked_fill(mask, -torch.inf).softmax(-1)
    # The final input token has no scored loss: exclude its query and selection.
    return weights[:, :, :-1, :-1].mean(1).sum(1)


def sampled_loss(logits, states, count, positions=None):
    targets = states.samples[..., :count]
    if positions is not None:
        targets = take(targets, positions)
    return -logits.log_softmax(-1).gather(-1, targets).mean()


def weighted_sampled_loss(logits, states, count, positions, weights):
    """Per-position sampled NLL, weighted and summed, averaged over the batch rows."""
    targets = take(states.samples[..., :count], positions)
    nll = -logits.log_softmax(-1).gather(-1, targets).mean(-1)
    return (nll * weights).sum() / nll.shape[0]


def importance_scores(network, q, hidden, samples, batch=1, attention=True):
    """Recompute scores once per recovery event, without changing any weight.

    Earlier suffix layers need only a forward pass: the score is the gradient
    with respect to the input of the final layer, not a full parameter gradient.
    Yields one score tensor per batch. attention=False keeps the gradient term only.
    """
    digital = network.teacher if network.digital is None else network.digital
    last = digital.layers[-1]
    k = len(network.mlp_norms)-1
    for states in hidden.split(batch):
        with torch.no_grad():
            z = full_hidden(network, states.student, q, stop_before_last=True)
            received = attention_importance(last, z) if attention else None
        z = z.detach().requires_grad_(True)
        with torch.enable_grad():
            x = last.attention(z)
            y = network._matrix(network.mlp_norms[k](x), 2*k, q)
            x = x + network._matrix(F.relu(y), 2*k+1, q)
            loss = sampled_loss(digital.logits(x[:, :-1]), states, samples)
            gradient = torch.autograd.grad(loss, z)[0][:, :-1].abs().sum(-1)
        low, high = gradient.amin(-1, keepdim=True), gradient.amax(-1, keepdim=True)
        gradient = (gradient-low) / (high-low).clamp_min(torch.finfo(gradient.dtype).eps)
        score = .7 * gradient + .3 * received.log1p() if attention else gradient
        if not bool(torch.isfinite(score).all()):
            raise FloatingPointError('Nonfinite importance scores.')
        yield score


def importance_indices(network, q, hidden, count, samples, batch=1):
    """exp-003's top-count positions of the combined score (selection unchanged)."""
    return torch.cat([score.topk(count, dim=-1).indices.sort(-1).values
                      for score in importance_scores(network, q, hidden, samples, batch)])


def sampling_probabilities(scores, mix=UNIFORM_MIX):
    """Defensive mixture of the per-sequence normalized score and uniform."""
    scores = scores.clamp_min(0)
    total = scores.sum(-1, keepdim=True)
    uniform = torch.full_like(scores, 1. / scores.shape[-1])
    shaped = torch.where(total > 0, scores / total.clamp_min(torch.finfo(scores.dtype).tiny), uniform)
    return (1 - mix) * shaped + mix * uniform


def draw_weighted(probabilities, count, generator):
    """Draw count positions with replacement from one row; return unique positions and weights.

    Position i drawn c_i times gets weight c_i / (count * L * p_i), so the weighted
    sum of per-position losses is an unbiased estimate of their mean over all L.
    """
    length = probabilities.shape[-1]
    draws = torch.multinomial(probabilities, count, replacement=True, generator=generator)
    positions, counts = draws.unique(sorted=True, return_counts=True)
    weights = counts.to(probabilities.dtype) / (count * length * probabilities[positions])
    return positions, weights


class SavedActivations:
    """Maximum live saved-tensor bytes, excluding persistent tensor storages.

    Views sharing storage are counted once. This is a saved-autograd-storage
    diagnostic, not allocator peak, total physical scratch RAM or offload.
    """
    def __init__(self, persistent=()):
        self.ignore = {self.key(x) for x in persistent if isinstance(x, torch.Tensor)}
        self.live = {}; self.bytes = 0; self.peak = 0
    @staticmethod
    def key(x):
        return (str(x.device), x.untyped_storage().data_ptr(), x.untyped_storage().nbytes())
    def pack(self, x):
        owner, key = self, self.key(x)
        class Saved:
            def __init__(self):
                self.tensor = x
                if key not in owner.ignore:
                    if key not in owner.live:
                        owner.bytes += key[-1]
                    owner.live[key] = owner.live.get(key, 0)+1
                    owner.peak = max(owner.peak, owner.bytes)
            def __del__(self):
                if key not in owner.ignore and key in owner.live:
                    owner.live[key] -= 1
                    if owner.live[key] == 0:
                        owner.bytes -= key[-1]; del owner.live[key]
        return Saved()
    def context(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, lambda saved: saved.tensor)


def train(network, parameters, q, hidden, *, steps, batch, learning_rate, seed,
          target, mode, microbatch=1):
    from workflow.drift_regimes_arms import suffix_logits, target_loss
    method, count = parse_mode(mode)
    if target[0] != 'samples':
        raise ValueError('The token-backward comparison requires fixed sampled teacher targets.')
    if method not in ('full', 'context') and count >= hidden.sequence_length:
        raise ValueError('Selected positions must be scored positions, excluding the final token.')
    if method in SAMPLED_MODES and microbatch != 1:
        raise ValueError('Sampled selections have per-sequence lengths; use microbatch 1.')
    for p in network.parameters():
        p.requires_grad_(False)
    q = q.detach()
    device = q.device
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    peak = lambda: torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    sync()
    prep_start = time.monotonic()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    indices = importance_indices(network, q, hidden, count, target[1], batch=microbatch) if method == 'importance' else None
    probabilities = None
    if method in SAMPLED_MODES:
        scores = torch.cat(list(importance_scores(network, q, hidden, target[1], batch=microbatch,
                                                  attention=method == 'importance_sampled')))
        probabilities = sampling_probabilities(scores.double().cpu())
        del scores
    sync()
    prep_seconds, prep_peak = time.monotonic()-prep_start, peak()
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    generator = torch.Generator().manual_seed(seed)
    selector = torch.Generator().manual_seed(seed+72913)
    if method == 'random_fixed':
        indices = torch.stack([torch.randperm(hidden.sequence_length-1, generator=selector)[:count]
                               for _ in range(len(hidden))]).sort(-1).values.to(device)
    seen = torch.zeros(len(hidden), hidden.sequence_length-1, dtype=torch.bool)
    presented = torch.zeros(len(hidden), dtype=torch.bool)
    backward_total, max_weight = 0, 0.
    losses = []
    network.train()
    saved_peak = 0
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    sync(); started = time.monotonic()
    with torch.enable_grad():
        for step in range(steps):
            index = torch.randint(0, len(hidden), (batch,), generator=generator).to(device)
            states = hidden[index]
            chosen = None
            if method in ('random', 'loss_only'):
                chosen = torch.stack([torch.randperm(hidden.sequence_length-1, generator=selector)[:count]
                                      for _ in range(batch)]).sort(-1).values.to(device)
            elif method in ('importance', 'random_fixed'):
                chosen = indices[index]
            elif method in SAMPLED_MODES:
                draws = [draw_weighted(probabilities[i], count, selector) for i in index.tolist()]
            elif method == 'context':
                n = hidden.sequence_length // count
                if hidden.sequence_length % count or not hasattr(hidden, 'short_context'):
                    raise ValueError('Missing independently recomputed short-context cache.')
                short_ids = (index[:, None]*n + torch.arange(n, device=device)).flatten()
                states = hidden.short_context[short_ids]
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for begin in range(0, len(states), microbatch):
                chunk = states[begin:begin+microbatch]
                positions = None if chosen is None else chosen[begin:begin+microbatch]
                weights = None
                if method in SAMPLED_MODES:
                    positions, weights = draws[begin]
                    max_weight = max(max_weight, float(weights.max()))
                    positions, weights = positions[None].to(device), weights[None].float().to(device)
                if method != 'context':
                    rows = index[begin:begin+microbatch].cpu()
                    presented[rows] = True
                    if positions is None:
                        seen[rows] = True
                    else:
                        seen[rows[:, None], positions.cpu()] = True
                    backward_total += (positions.numel() if method in ('random', 'importance', *SELECTION_MODES)
                                       else len(chunk)*hidden.sequence_length)
                audit = SavedActivations([*network.parameters(), *network.buffers(), q,
                                          chunk.student, chunk.samples])
                with audit.context() if step in (0, 1) else contextlib.nullcontext():
                    if method in ('random', 'importance', *SELECTION_MODES):
                        logits = selected_logits(network, chunk.student, q, positions)
                    else:
                        logits = suffix_logits(network, chunk.student, q)
                        if positions is not None: logits = take(logits, positions)
                    if weights is not None:
                        loss = weighted_sampled_loss(logits, chunk, target[1], positions, weights)
                    else:
                        loss = sampled_loss(logits, chunk, target[1], positions)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Nonfinite token-backward loss.')
                    weight = len(chunk)/len(states)
                    (loss*weight).backward()
                    total += float(loss.detach())*weight
                saved_peak = max(saved_peak, audit.peak)
                del logits, loss, chunk, audit
            optimizer.step(); schedule.step(); losses.append(total)
    sync()
    seconds, training_peak = time.monotonic()-started, peak()
    network.eval()
    for p in parameters: p.requires_grad_(False)
    scored = ((hidden.sequence_length//count)*(count-1) if method == 'context'
              else hidden.sequence_length-1 if method == 'full' else count)
    extra = {}
    if method != 'context':
        extra['distinct_trained_positions_per_presented_parent'] = float(seen[presented].sum(1).float().mean())
        extra['presented_parents'] = int(presented.sum())
    if method in SAMPLED_MODES:
        extra.update(sampling_uniform_mix=UNIFORM_MIX, max_importance_weight=max_weight,
                     mean_unique_positions=backward_total/(steps*batch))
    return {**extra, 'first_loss': sum(losses[:10])/min(10, len(losses)),
            'last_loss': sum(losses[-10:])/min(10, len(losses)),
            'token_backward': mode, 'microbatch': microbatch,
            'selection_seconds': prep_seconds, 'selection_peak_allocated_bytes': prep_peak,
            'selection_sequences': len(hidden) if indices is not None else 0,
            'selection_index_bytes': indices.numel()*indices.element_size() if indices is not None else 0,
            'training_seconds': seconds, 'training_peak_allocated_bytes': training_peak,
            'saved_activation_peak_bytes': saved_peak,
            'scored_positions': backward_total if method in SAMPLED_MODES else steps*batch*scored,
            'backward_positions': (backward_total if method in SAMPLED_MODES else
                                   steps*batch*(hidden.sequence_length if method in ('full','loss_only','context') else count))}
