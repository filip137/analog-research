"""core.tensor_identity — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from hashlib import sha256
import torch


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    h = sha256(str((tuple(value.shape), value.dtype)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_hash(state: dict) -> str:
    h = sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()
