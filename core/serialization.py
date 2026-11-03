"""Restricted native tensor codecs and recursive CPU snapshots.

Preserve existing tensor containers, gzip detection and atomic rename semantics.
"""
from __future__ import annotations
from copy import deepcopy
import torch

def clone_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_cpu(v) for v in value)
    return deepcopy(value)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(clone_cpu(value), temporary)
    temporary.replace(path)


def load(path):
    import gzip
    import io
    with open(path, "rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rb") as handle:
            return torch.load(io.BytesIO(handle.read()), map_location="cpu", weights_only=True)
    return torch.load(path, map_location="cpu", weights_only=True)
