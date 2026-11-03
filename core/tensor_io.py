"""core.tensor_io — extracted native implementation; formats and numerical order preserved."""
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
