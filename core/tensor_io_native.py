"""core.tensor_io_native — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from core.tensor_io import clone_cpu


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
