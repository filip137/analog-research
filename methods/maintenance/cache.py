"""methods.maintenance.cache — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from pathlib import Path
import torch


@torch.no_grad()
def probe_cache(adapter, tokens, physical, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for i, batch in enumerate(tokens.split(1)):
        _, captures = adapter.forward(batch, physical, capture=True)
        torch.save([(x.detach().cpu(), y.detach().cpu()) for x, y in captures], directory/f'{i:05d}.pt')
    return directory


@torch.no_grad()
def fit_calibration(adapter, physical, directory, method):
    """Paired least squares from the same accessible block-input probes."""
    stats = None
    bound = adapter.bind(physical)
    probes = 0
    for path in sorted(Path(directory).glob('*.pt')):
        pairs = torch.load(path, map_location='cpu', weights_only=True)
        if stats is None: stats = [None]*len(pairs)
        for k, (x, reference) in enumerate(pairs):
            observed = adapter.block(k, x.to(adapter.device), bound).flatten(0, -2).double()
            expected = reference.to(adapter.device).flatten(0, -2).double()
            if method == 'global': observed, expected = observed.reshape(-1, 1), expected.reshape(-1, 1)
            row = torch.stack((observed.sum(0), expected.sum(0), observed.square().sum(0),
                               (observed*expected).sum(0), torch.full_like(observed.sum(0), len(observed))))
            stats[k] = row if stats[k] is None else stats[k] + row
            probes += x.numel()//x.shape[-1]
    if stats is None: raise ValueError('Calibration needs observable reference probes.')
    corrections = []
    for sx, sy, sxx, sxy, n in stats:
        var = (sxx-sx.square()/n).clamp_min(0)
        gain = torch.where(var > 1e-12, (sxy-sx*sy/n)/var.clamp_min(1e-12), torch.ones_like(var))
        offset = sy/n-gain*sx/n
        if not bool(torch.isfinite(gain).all() and torch.isfinite(offset).all()):
            raise FloatingPointError('Nonfinite output calibration.')
        corrections.append((gain.float(), offset.float()))
    return corrections, probes
