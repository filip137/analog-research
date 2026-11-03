"""core.logit_cache — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import json
import time
from pathlib import Path
import torch
from core.artifacts import atomic_write_json, sha256_file


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
