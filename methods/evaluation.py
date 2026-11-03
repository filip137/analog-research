"""Frozen FP32 logit references and token-weighted drift diagnostics.

Mirrored in the DRN worktree. Reference logits are generated once, saved, and
held on CPU; no teacher or physical-state randomness is redrawn per condition.
"""

import math
from pathlib import Path
import torch
from torch.nn import functional as F


@torch.no_grad()
def reference_cache(tokens, clean_forward, teacher_forward, directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    chunks = []
    for i, batch in enumerate(tokens.split(1)):
        values = {'clean': clean_forward(batch).detach().float().cpu(),
                  'teacher': teacher_forward(batch).detach().float().cpu()}
        if any(not bool(torch.isfinite(v).all()) for v in values.values()):
            raise FloatingPointError('Nonfinite frozen reference logits.')
        torch.save(values, directory / f'{i:04d}.pt')
        chunks.append(values)
    torch.save({'schema': 'drift.reference.v1', 'identity': identity, 'tokens': tokens.cpu()},
               directory / 'identity.pt')
    return chunks


@torch.no_grad()
def evaluate(tokens, forward, references, device):
    if len(tokens) != len(references):
        raise ValueError('Reference cohort differs from evaluation cohort.')
    count = correct = 0
    ce = clean_kl = teacher_kl = 0.
    for batch, cache in zip(tokens.split(1), references):
        output = forward(batch).float()
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError('Nonfinite drifted logits.')
        flat = output.flatten(0, 1)
        logp = flat.log_softmax(-1)
        for key in ('clean', 'teacher'):
            ref = cache[key].to(device)
            if ref.shape != output.shape:
                raise ValueError('Reference logit shape differs from the evaluated model.')
            kl = float(F.kl_div(logp, ref.flatten(0, 1).softmax(-1), reduction='sum'))
            if key == 'clean': clean_kl += kl
            else: teacher_kl += kl
        labels = batch[:, 1:].to(device).flatten()
        ce += float(F.cross_entropy(flat, labels, reduction='sum'))
        correct += int((flat.argmax(-1) == labels).sum())
        count += labels.numel()
    result = {'clean_hwa_kl': clean_kl / count, 'teacher_kl': teacher_kl / count,
              'cross_entropy': ce / count, 'perplexity': math.exp(ce / count),
              'next_token_accuracy_percent': 100 * correct / count,
              'scored_tokens': count, 'sequences': len(tokens)}
    if any(not math.isfinite(v) for v in result.values()):
        raise FloatingPointError('Nonfinite drift metrics.')
    return result
