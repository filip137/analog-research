"""Offline GPU training of chain A on MNIST (#63). Specification:
campaigns/pilots/20261117-mnist-attention.md."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import torch
from ebl.runtime.store import RunStore, atomic_write_json
from networks.energy_transformer.pseudo_mnist_data import load_raw
from networks.energy_transformer.mnist_chain_a import ChainA, inputs, mos_count

ROOT = Path(__file__).resolve().parents[2]


def labels(root):
    out = {}
    for s, p in (('train', 'train'), ('test', 't10k')):
        raw = (Path(root) / 'MNIST' / 'raw' / f'{p}-labels-idx1-ubyte').read_bytes()[8:]
        out[s] = torch.tensor(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
    return out


def run_train(request):
    c = request.spec
    store = RunStore.create(output_root=request.output_dir, experiment_id=c.experiment_id, resolved_config=asdict(c),
                            command=request.command, repo_root=ROOT, input_artifacts=[], resume_capability='unsupported')
    print(f'RUN_DIR={store.run_dir}', flush=True)
    try:
        t0 = time.monotonic()
        torch.manual_seed(c.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        dev = c.device
        imgs, split, paths = load_raw(c.data_root)
        lab = labels(c.data_root)
        X = {k: inputs(imgs[s][split[k]].to(dev)).double() for k, s in (('train', 'train'), ('validation', 'train'), ('test', 'test'))}
        Y = {k: lab[s][split[k]].to(dev) for k, s in (('train', 'train'), ('validation', 'train'), ('test', 'test'))}
        m = ChainA(c.heads, c.d_k, c.d_v, scale=c.scale, seed=c.seed).to(dev).double()
        opt = torch.optim.Adam(m.parameters(), lr=c.lr)
        steps = c.epochs * ((len(X['train']) + c.batch_size - 1) // c.batch_size)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if c.cosine else None
        acc = lambda k: float((m(X[k]).argmax(-1) == Y[k]).double().mean())
        for ep in range(c.epochs):
            if time.monotonic() - t0 > c.wall_seconds:
                raise TimeoutError('Declared allowance exceeded')
            perm = torch.randperm(len(X['train']), device=dev)
            tot = 0.
            for i in range(0, len(perm), c.batch_size):
                idx = perm[i:i + c.batch_size]
                loss = torch.nn.functional.cross_entropy(m(X['train'][idx]), Y['train'][idx])
                opt.zero_grad(); loss.backward(); opt.step()
                if sched: sched.step()
                tot += float(loss) * len(idx)
            with torch.no_grad():
                rec = {'epoch': ep + 1, 'train_loss': tot / len(perm), 'validation_accuracy': acc('validation')}
            store.append_metric(rec)
        with torch.no_grad():
            final = {'validation_accuracy': acc('validation'), 'test_accuracy': acc('test'),
                     'mos_count': mos_count(c.heads, c.d_k, c.d_v), 'seconds': time.monotonic() - t0}
        art = store.run_dir / 'artifacts'
        torch.save({'model': m.state_dict(), 'config': asdict(c)}, art / 'model.pt')
        atomic_write_json(art / 'export.json', m.export() | {'test_accuracy': final['test_accuracy'],
                                                               'validation_accuracy': final['validation_accuracy']})
        atomic_write_json(art / 'summary.json', final)
        artifacts = [store.artifact_record(p, kind='model') for p in sorted(art.rglob('*')) if p.is_file()]
        store.complete(metrics=final, artifacts=artifacts)
        print(json.dumps(final), flush=True)
        return 0
    except BaseException as err:
        store.fail(err)
        raise
