"""Streamed FP32 targets and observable block probes for custom recovery."""
import json
import time
from pathlib import Path
import torch
from experiments.artifacts import atomic_write_json, sha256_file


class LogitCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        self.read_seconds = 0.
        self.bytes_read = 0

    @classmethod
    @torch.no_grad()
    def create(cls, directory, tokens, forward, identity, progress=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        rows = []
        for i, batch in enumerate(tokens.split(1)):
            logits = forward(batch).detach().float().cpu()
            if not bool(torch.isfinite(logits).all()): raise FloatingPointError('Nonfinite clean targets.')
            path = directory/f'{i:05d}.pt'
            torch.save(logits, path)
            rows.append({'file': path.name, 'sha256': sha256_file(path), 'bytes': path.stat().st_size})
            if progress: progress(i+1, len(tokens))
        atomic_write_json(directory/'manifest.json', {'schema': 'custom_drift.targets.v1',
                          'identity': identity, 'precision': 'float32', 'rows': rows})
        return cls(directory)

    def __len__(self):
        return len(self.manifest['rows'])

    def __getitem__(self, i):
        start = time.monotonic()
        result = torch.load(self.directory/self.manifest['rows'][i]['file'], map_location='cpu', weights_only=True)
        self.read_seconds += time.monotonic()-start
        self.bytes_read += self.manifest['rows'][i]['bytes']
        return result

    @property
    def bytes(self):
        return sum(r['bytes'] for r in self.manifest['rows'])


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
