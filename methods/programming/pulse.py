"""methods.programming.pulse — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch


@torch.no_grad()
def program(port, target, *, tolerance, maximum_pulses):
    count = torch.zeros(port.size, dtype=torch.int64, device=target.device)
    reads = 0
    for _ in range(maximum_pulses):
        error = target - port.read()
        reads += port.size
        active = error.abs() > tolerance
        if not bool(active.any()):
            break
        port.pulse(error.sign().to(torch.int8) * active)
        count += active
    error = target - port.read()
    reads += port.size
    return {
        "accepted": int((error.abs() <= tolerance).sum()),
        "exhausted": int((error.abs() > tolerance).sum()),
        "verify_reads": reads,
        "pulses": int(count.sum()),
        "residual_rms": float(error.square().mean().sqrt()),
    }
