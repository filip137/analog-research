"""core.seeds — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from hashlib import sha256


def binding_seed(seed, dataset, layer):
    return int.from_bytes(sha256(f'{seed}:{dataset}:{layer}'.encode()).digest()[:4], 'little') % 2000000000 + 1
