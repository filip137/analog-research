"""Held off-chip HWA views; the digital master remains network-owned."""
import torch
from hardware.devices.cmo import MID_US, HALF_RANGE_US, G_MAX_US

@torch.no_grad()
def hardware_view(model,noise_std,generator):
    views=[]
    projected=0
    for block in model.blocks:
        pair=[]
        for u in (block.u1,block.u2):
            # Consume matched streams in clean/std3 arms as well.
            noisy= MID_US + HALF_RANGE_US*(u + noise_std*torch.randn(u.shape,device=u.device,generator=generator))
            projected += int((noisy<0).sum())
            pair.append(noisy.clamp_min(0)/G_MAX_US)
        views.append(tuple(pair))
    return tuple(views),projected
