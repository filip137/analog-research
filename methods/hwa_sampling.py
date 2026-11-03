"""methods.hwa_sampling — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch


def sampled_weights(q, generator, noise_scale=1.0, kind="open", rate=0.0):
    """Resample the physical-bank population every minibatch; preserve master.

    Uses the paper's CDT principle with our physical-device probability
    convention consistently in training and deployment. The temporary mask
    never becomes a deployment fault map. Straight-through gradient follows
    the temporary effective weight, as in the existing HWA implementation.
    """
    with torch.no_grad():
        target = torch.stack((q.detach().clamp_min(0), (-q.detach()).clamp_min(0))) * 25
        r = target / 25
        sigma = 0.26348 + 1.965 * r - 1.1731 * r.square()
        g = (
            target
            + noise_scale
            * sigma
            * torch.randn(target.shape, generator=generator, device=q.device)
        ).clamp_min(0)
        mask = torch.rand(target.shape, generator=generator, device=q.device) < rate
        fixed = torch.rand(target.shape, generator=generator, device=q.device) * 25
        if kind == "open":
            fixed.zero_()
        elif kind == "gmax":
            fixed.fill_(25)
        elif kind != "random":
            raise ValueError("Expected open/gmax/random training corruption.")
        g = torch.where(mask, fixed, g)
        apparent = (g[0] - g[1]) / 25
    return q + (apparent - q).detach(), mask
