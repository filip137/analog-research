"""Task loss and scoring for pinned token or ragged-choice adaptation cohorts."""
import math

import torch
from torch.nn import functional as F


def count(split):
    return len(split["labels"] if "labels" in split else split["tokens"])


def choice_batch(split, indices, device):
    """Keep all alternatives for each example together; only score continuation tokens."""
    k = split["k"]
    rows = (indices[:, None] * k + torch.arange(k)).flatten()
    starts = split["offsets"][rows]
    lengths = split["offsets"][rows + 1] - starts
    position = torch.arange(int(lengths.max()))
    gather = starts[:, None] + position
    tokens = split["tokens"][gather.clamp_max(len(split["tokens"]) - 1)]
    tokens.masked_fill_(position >= lengths[:, None], split["pad"])
    mask = (position[1:] < lengths[:, None]) & (position[1:] >= split["context_lengths"][rows, None])
    if not mask.any(1).all():
        raise ValueError("Choice has an empty continuation.")
    return tokens.long().to(device), mask.to(device), split["chars"][rows].to(device), split["labels"][indices].long().to(device)


def choice_outputs(forward, split, indices, device):
    tokens, mask, chars, labels = choice_batch(split, indices, device)
    logits = forward(tokens)
    scores = -F.cross_entropy(logits[mask].float(), tokens[:, 1:][mask], reduction="none")
    row = torch.arange(len(tokens), device=device)[:, None].expand_as(mask)[mask]
    sums = torch.zeros(len(tokens), device=device, dtype=torch.float64).index_add_(0, row, scores.double())
    k = split["k"]
    loss = F.cross_entropy((sums / mask.sum(1)).view(-1, k), labels)
    prediction = (sums / chars).view(-1, k).argmax(1)
    return loss, prediction, labels


def task_loss(forward, split, indices, device):
    if "labels" in split:
        return choice_outputs(forward, split, indices, device)[0]
    tokens = split["tokens"][indices].long().to(device)
    logits = forward(tokens)
    return F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].reshape(-1))


def backward_batch(forward, split, indices, device, microbatch):
    """Accumulate the exact example-weighted loss of one logical training or Fisher batch."""
    total = torch.zeros((), device=device, dtype=torch.float64)
    for part in indices.split(microbatch):
        loss = task_loss(forward, split, part, device) * (len(part) / len(indices))
        loss.backward()
        total += loss.detach()
    if not torch.isfinite(total):
        raise FloatingPointError("Task loss is nonfinite.")
    return float(total)


@torch.no_grad()
def evaluate(forward, split, device, microbatch, teacher_fn=None, limit=None):
    n = count(split) if limit is None else min(count(split), limit)
    if n < 1:
        raise ValueError("Empty evaluation cohort.")
    nll = 0.
    total = 0
    kl = 0.
    predictions = []
    for ids in torch.arange(n).split(microbatch):
        if "labels" in split:
            loss, prediction, labels = choice_outputs(forward, split, ids, device)
            predictions.extend(prediction.tolist())
            total += int((prediction == labels).sum())
            nll += float(loss) * len(ids)
        else:
            tokens = split["tokens"][ids].long().to(device)
            logits = forward(tokens).float().flatten(0, 1)
            nll += float(F.cross_entropy(logits, tokens[:, 1:].reshape(-1), reduction="sum"))
            total += logits.shape[0]
            if teacher_fn:
                teacher = teacher_fn(tokens).float().flatten(0, 1).log_softmax(-1)
                kl += float(F.kl_div(logits.log_softmax(-1), teacher, reduction="sum", log_target=True))
    if "labels" in split:
        metrics = {"examples": n, "accuracy_percent": 100 * total / n, "choice_ce": nll / n,
                   "predictions": predictions}
    else:
        metrics = {"examples": n, "tokens": total, "nll": nll / total, "perplexity": math.exp(nll / total)}
        if teacher_fn:
            metrics["teacher_kl"] = kl / total
    if any(isinstance(v, float) and not math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite task endpoint.")
    return metrics
