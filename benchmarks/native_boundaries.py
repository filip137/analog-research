"""Fixed CPU workload for architecture extraction and exact replay comparison.

Run as a module from each checkout, using the same interpreter. Setup/restoration
is outside timed regions. Outputs belong in the trunk's ignored results directory.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time

import torch

from networks.opt import OptConfig, OptDecoder, OptMlpSuffix
from methods.onchip import IdealizedWriteAdam
from hardware.devices.cmo import CmoMatrix
from hardware.drift.writable import WritableDrift
from hardware.drift.config import DriftLaw


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((tuple(x.shape), x.dtype)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for item in x: visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def storage_bytes(value):
    seen = set()
    def visit(x):
        if isinstance(x, torch.Tensor):
            storage = x.untyped_storage()
            key = (str(x.device), storage.data_ptr())
            if key in seen: return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(x, dict): return sum(visit(v) for v in x.values())
        if isinstance(x, (list, tuple)): return sum(visit(v) for v in x)
        return 0
    return visit(value)


def measure(setup, operation, repeats):
    elapsed = []
    for i in range(3 + repeats):
        setup()
        start = time.perf_counter_ns()
        operation()
        seconds = (time.perf_counter_ns() - start) / 1e9
        if i >= 3: elapsed.append(seconds)
    quartiles = statistics.quantiles(elapsed, n=4)
    return {'seconds': elapsed, 'median': statistics.median(elapsed),
            'iqr': quartiles[2] - quartiles[0], 'min': min(elapsed), 'max': max(elapsed)}


def run(workload, output, repeats=10, profile=False):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1729)
    torch.use_deterministic_algorithms(True)
    cfg = (OptConfig(64, 16, 3, 2, 32, 32) if workload == 'tiny'
           else OptConfig(64, 768, 1, 12, 3072, 32))
    net = OptMlpSuffix(OptDecoder(cfg).eval().requires_grad_(False),
                       2 if workload == 'tiny' else 1, 512)
    tokens = torch.randint(0, 64, (2, 9), generator=torch.Generator().manual_seed(2718))
    target = net.q.detach().clone()
    size = target.numel()
    mask = torch.zeros((2, size), dtype=torch.bool)
    mask[:, ::100] = True
    stuck = torch.full((2, size), 9.)
    array = CmoMatrix(size, 31415, 2., 1., 'cpu', (mask, stuck))
    array.program(target)
    initial = array.state_dict()
    cells = WritableDrift(array.g, law=DriftLaw(), rho=.9, seed=1618, name='benchmark')
    drift_initial = cells.state_dict()
    writer = IdealizedWriteAdam(array, learning_rate=.001)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'checkpoint.pt'

    def reset():
        nonlocal writer, cells
        array.load_state_dict(initial)
        cells = WritableDrift.restore(drift_initial)
        writer = IdealizedWriteAdam(array, learning_rate=.001)
        net.zero_grad(set_to_none=True)

    def backward():
        observed = array.read().detach().clone().requires_grad_(True)
        loss = net(tokens, observed).square().mean()
        loss.backward()
        return observed, loss

    def step():
        observed, loss = backward()
        writer.step(observed.grad)
        return loss

    operations = {
        'forward_backward': backward,
        'adaptation_step': step,
        'program': lambda: array.program(target),
        'drift_update': lambda: cells.advance(3600.),
        'checkpoint_save': lambda: torch.save({'array': array.state_dict(), 'drift': cells.state_dict(),
                                               'writer': writer.state_dict()}, checkpoint),
        'checkpoint_load': lambda: torch.load(checkpoint, weights_only=True),
    }
    timings = {name: measure(reset, operation, repeats) for name, operation in operations.items()}
    reset()
    losses = []
    for _ in range(3): losses.append(float(step().detach()))
    observed, loss = backward()
    replay = {'network': net.state_dict(), 'array': array.state_dict(), 'writer': writer.state_dict(),
              'losses': losses, 'logits': net(tokens, array.read()).detach(), 'gradient': observed.grad,
              'drift': cells.state_dict(), 'future_drift': cells.value(3600.), 'cost': array.cost()}
    trace_digest = digest(replay)
    if workload == 'tiny': torch.save(replay, output / 'replay.pt')
    del replay
    if profile:
        reset()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True,
                                    record_shapes=True) as prof:
            step(); array.program(target); cells.advance(3600.)
        (output / 'profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=30))
    result = {'workload': workload, 'python': platform.python_version(), 'torch': torch.__version__,
              'device': 'cpu', 'dtype': 'float32', 'threads': 1, 'seeds': [1729, 2718, 31415, 1618],
              'warmup': 3, 'repeats': repeats, 'logical_weights': size, 'trace_sha256': trace_digest,
              'timings': timings, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'persistent_bytes': storage_bytes([array.__dict__, cells.__dict__, writer.first, writer.second]),
              'checkpoint_bytes': checkpoint.stat().st_size}
    (output / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'timings'}), flush=True)
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', choices=('tiny', 'opt_mlp'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    run(args.workload, args.output, args.repeats, args.profile)
