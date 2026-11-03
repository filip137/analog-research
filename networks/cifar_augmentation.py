"""networks.cifar_augmentation — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from networks.cifar_data import NORMALIZATION


def augment_images(raw, dataset, generator):
    """Vectorized random crop/flip before normalization, fresh each epoch."""
    x = raw.float() / 255
    padded = F.pad(x, (4, 4, 4, 4))
    n, channels = raw.shape[:2]
    offsets = torch.randint(9, (n, 2), generator=generator, device=raw.device)
    a = torch.arange(32, device=raw.device)
    ys = offsets[:, 0, None, None] + a[None, :, None]
    xs = offsets[:, 1, None, None] + a[None, None, :]
    x = padded[
        torch.arange(n, device=raw.device)[:, None, None, None],
        torch.arange(channels, device=raw.device)[None, :, None, None],
        ys[:, None, :, :],
        xs[:, None, :, :],
    ]
    flip = torch.rand(n, generator=generator, device=raw.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    mean, sd = NORMALIZATION[dataset]
    return (x - x.new_tensor(mean)[None, :, None, None]) / x.new_tensor(sd)[
        None, :, None, None
    ]
