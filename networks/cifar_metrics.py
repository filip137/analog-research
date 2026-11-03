"""networks.cifar_metrics — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from core.tensor_identity import tensor_hash


@torch.no_grad()
def evaluate(student, cache, device, *, q=None, batch_size=256, read_seed=50001):
    old_rng = student.read_generator.get_state()
    student.read_generator.manual_seed(read_seed)
    student.eval()
    total = correct = agreement = 0
    kl = ce = 0.0
    predictions = []
    try:
        for begin in range(0, len(cache["labels"]), batch_size):
            x = cache["features"][begin : begin + batch_size].to(device)
            t = cache["teacher_logits"][begin : begin + batch_size].to(device)
            y = cache["labels"][begin : begin + batch_size].to(device)
            out = student.forward_features(x, q)
            if not bool(torch.isfinite(out).all()):
                raise FloatingPointError("Nonfinite network scores.")
            p = out.argmax(1)
            predictions.append(p.cpu())
            correct += int((p == y).sum())
            agreement += int((p == t.argmax(1)).sum())
            total += len(y)
            kl += float(F.kl_div(out.log_softmax(1), t.softmax(1), reduction="sum"))
            ce += float(F.cross_entropy(out, y, reduction="sum"))
    finally:
        student.read_generator.set_state(old_rng)
    return {
        "examples": total,
        "accuracy_percent": 100 * correct / total,
        "teacher_agreement_percent": 100 * agreement / total,
        "teacher_kl": kl / total,
        "cross_entropy": ce / total,
        "prediction_sha256": tensor_hash(torch.cat(predictions)),
    }
