"""Reproduce the fixed 8-sample teacher draw ``df73ce66`` and save it, refusing any other draw.

The draw is recovery-data-preparation exp-001's (hwa ``exp/recovery-targets``,
``workflow/drift_regimes_arms.teacher_samples``): teacher prefix states at the first
analog layer in batches of 16, teacher suffix logits at the scored positions, softmax,
``torch.multinomial(.., count, replacement=True)`` with one CUDA generator seeded
``samples_seed``. It was only ever stored as a hash. The CUDA multinomial stream depends
on the GPU model: Jean Zay V100s with PyTorch 2.5.0 give ``df73ce66`` (21 logged runs);
an RTX 3090 gives a different draw. The file is written only if the plain-bytes tensor
hash equals the config's ``samples_sha256`` (``--allow-other`` labels a local canary draw).

Usage: python draw_samples.py CONFIG TEACHER OUT_DIR [--allow-other]
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ebl.runtime.opt_cmo_drn.fault_tuning import cohorts, plain_hash  # noqa: E402
from ebl.runtime.opt_cmo_drn.runtime import MODEL  # noqa: E402
from ebl.schema.opt_cmo_drn import parse_config  # noqa: E402
from methods.drift_regimes.forward import teacher_logits, teacher_prefix  # noqa: E402
from networks.opt.data import pinned_load  # noqa: E402
from networks.opt.decoder import from_hugging_face  # noqa: E402

START = 6  # first analog decoder layer of the six-MLP suffix
BATCH = 16


@torch.no_grad()
def draw(teacher, tokens, count, seed, device):
    prefix = torch.cat([teacher_prefix(teacher, tokens[i:i + BATCH].to(device), START)
                        for i in range(0, len(tokens), BATCH)])
    generator = torch.Generator(device=device).manual_seed(seed)
    out = []
    for part in prefix.split(BATCH):
        probs = teacher_logits(teacher, START, part).float().softmax(-1)
        shape = probs.shape[:-1]
        out.append(torch.multinomial(probs.flatten(0, 1), count, replacement=True, generator=generator)
                   .view(*shape, count))
    return torch.cat(out).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('teacher', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--allow-other', action='store_true')
    args = parser.parse_args()
    config = parse_config(json.loads(args.config.read_text()))
    s = config.fault_tuning
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recovery, _ = cohorts(s)
    teacher = from_hugging_face(pinned_load(args.teacher, config.teacher_sha256), MODEL).eval().requires_grad_(False)
    teacher = teacher.to(config.device)
    samples = draw(teacher, recovery, s.samples_count, s.samples_seed, config.device)
    digest = plain_hash(samples)
    receipt = {'device': torch.cuda.get_device_name(config.device), 'torch': torch.__version__,
               'shape': list(samples.shape), 'tensor_sha256': digest, 'expected': s.samples_sha256,
               'seed': s.samples_seed, 'recovery_sha256': s.recovery_sha256}
    print(json.dumps(receipt), flush=True)
    if digest != s.samples_sha256 and not args.allow_other:
        raise SystemExit('Teacher-sample draw differs from the declared df73ce66 tensor; not saved.')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / s.samples_file
    temporary = path.with_suffix('.tmp')
    torch.save(samples, temporary)
    os.replace(temporary, path)
    (args.out_dir / (s.samples_file + '.receipt.json')).write_text(json.dumps(receipt, indent=1) + '\n')


if __name__ == '__main__':
    main()
