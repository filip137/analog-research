"""Benchmark zeroth-order attention input errors in OPT DRN physical recovery.

Exploratory lab for on-chip-learning/campaigns/zo-attention. It restores one
saved deployment, keeps the DRN simulator (BPTT) gradients exact, and changes
only how errors cross the digital attention between analog MLPs:

Arms local_cached, local_onpolicy and local_stream replace the error crossing
attention by block-local targets (experiments/opt_cmo_drn/local_targets.py).

  epsilon    finite-difference directional derivatives against exact VJPs
  alignment  conductance-gradient quality on the first recovery batch
  recover    matched noise-free direct-Adam recovery from the same state

No RunStore bundle, checkpoint or queue entry is written; each command writes
one JSON report (and recover also a JSONL trace) under --output.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import torch

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.opt_cmo_drn import runtime
from experiments.opt_cmo_drn.config import parse_config
from experiments.opt_cmo_drn.data import cohorts
from experiments.opt_cmo_drn.local_targets import LocalRecoveryStep
from experiments.opt_cmo_drn.physical import CmoDrnBank
from experiments.opt_cmo_drn.recovery import RecoveryStep
from experiments.opt_cmo_drn.zo_attention import AttentionBackward, install, uninstall
from training.core.engine import train_epoch
from training.core.guards import FiniteGradientGuard


def arm_estimator(arm, args, device):
    if arm.startswith('local_'):
        return AttentionBackward('exact', device=device)
    if arm in ('exact', 'residual', 'stop', 'linearized'):
        return AttentionBackward(arm, device=device)
    control = 'linearized' if arm.startswith('linzo') else None
    directions = int(arm.removeprefix('lin').removeprefix('zo'))
    return AttentionBackward('zo', directions, args.epsilon, args.probe_seed, device, control=control)


def arm_core(arm, model, teacher, bank, config):
    if arm.startswith('local_'):
        return LocalRecoveryStep(model, teacher, bank, config, arm.removeprefix('local_'))
    return RecoveryStep(model, teacher, bank, config)


def setup(args):
    config = parse_config(json.loads(args.config.read_text()))
    recovery = replace(config.recovery, updates=args.updates, development_every=args.development_every)
    config = replace(config, device=args.device, recovery=recovery)
    if (recovery.arm, recovery.update_rule, recovery.write_noise_factor) != ('learning', 'direct_adam', 0.0):
        raise ValueError('The benchmark isolates gradients: expected noise-free direct-Adam learning.')
    runtime.configure_runtime(config)
    teacher = runtime.teacher_model(args.teacher, config)
    source = torch.load(args.weights, map_location='cpu', weights_only=True)
    model = runtime.from_mapped(teacher, config, source).to(config.device)
    teacher = teacher.to(config.device)
    train, dev, cohort = cohorts(config)
    physical = torch.load(args.device_state, map_location='cpu', weights_only=True)
    identity = {'weights_sha256': sha256_file(args.weights), 'device_state_sha256': sha256_file(args.device_state),
                'teacher_sha256': config.teacher_sha256, 'corpus_sha256': config.data.sha256, 'cohorts': cohort}
    if physical.get('schema') != 'opt_cmo_drn.deployment.v1' or physical['cohorts'] != cohort:
        raise ValueError('Expected the canonical deployment for these cohorts.')
    if physical['source_weights_sha256'] != identity['weights_sha256']:
        raise ValueError('Deployment was programmed from different source weights.')
    if physical['deployment'] != asdict(config.deployment):
        raise ValueError('Deployment assignment/fault case differs from the configuration.')
    bank = CmoDrnBank(physical['bank'], config.device)
    if bank.held_hashes() != physical['start_held_hashes']:
        raise ValueError('Canonical deployment failed held-state identity check.')
    del physical, source
    return config, model, teacher, bank, train, dev, identity


def batch_at(train, step, config):
    batches = math.ceil(len(train) / config.training.effective_batch)
    epoch, cursor = divmod(step, batches)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(config.data.seed + epoch))
    return train[order[cursor * config.training.effective_batch:(cursor + 1) * config.training.effective_batch]]


def provenance(args, identity):
    root = Path(__file__).resolve().parents[1]
    # Cluster compute nodes may lack git; per-file hashes below identify the source.
    git = lambda *a: (subprocess.run(['git', '-C', str(root), *a], capture_output=True, text=True).stdout.strip()
                      or None) if shutil.which('git') else None
    files = ['experiments/opt_cmo_drn/zo_attention.py', 'labs/opt_cmo_drn_zo_attention.py',
             'experiments/opt_cmo_drn/recovery.py', 'experiments/opt_cmo_drn/model.py',
             'model/resistive/differentiable.py', 'training/cmo_direct_adam.py']
    return {'command': args.command, 'arguments': {k: str(v) for k, v in vars(args).items()}, **identity,
            'config_sha256': sha256_file(args.config), 'git_commit': git('rev-parse', 'HEAD'),
            'source_sha256': {f: sha256_file(root / f) for f in files},
            'torch': str(torch.__version__), 'host': platform.node(),
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}


class Capture(AttentionBackward):
    """Exact VJP backward that keeps each block's input and incoming error."""

    def __init__(self, device):
        super().__init__('residual', device=device)
        self.captured = []

    def correction(self, branch, x, grad):
        with torch.enable_grad():
            leaf = x.detach().requires_grad_(True)
            exact, = torch.autograd.grad(branch(leaf), leaf, grad)
        self.captured.append((branch, x.detach(), grad.detach(), exact))
        return exact


def run_epsilon(args, config, model, teacher, bank, train, dev, report):
    core = RecoveryStep(model, teacher, bank, config)
    capture = Capture(config.device)
    install(model, capture)
    core.set_input(batch_at(train, 0, config)[:args.sequences])
    core.compute_gradient()
    generator = torch.Generator(device=config.device).manual_seed(args.probe_seed)
    rows = []
    with torch.no_grad():
        for sequence, (branch, x, grad, exact) in enumerate(capture.captured):
            u = torch.randn((args.probes, *x.shape), device=x.device, generator=generator)
            truth = (u * exact).flatten(1).sum(1).double()
            rms = float(x.square().mean().sqrt())
            for epsilon in args.epsilons:
                eps = epsilon * rms
                out = branch(torch.cat((x + eps * u, x - eps * u)).flatten(0, 1)).unflatten(0, (2, args.probes, len(x)))
                estimate = (((out[0] - out[1]) * grad).flatten(1).sum(1) / (2 * eps)).double()
                error = ((estimate - truth).abs() / truth.abs().clamp_min(1e-30)).median()
                rows.append({'layer': branch.layer, 'microbatch': sequence // 5, 'epsilon_relative': epsilon,
                             'input_rms': rms, 'median_relative_error': float(error)})
    uninstall(model)
    summary = {}
    for epsilon in args.epsilons:
        errors = torch.tensor([r['median_relative_error'] for r in rows if r['epsilon_relative'] == epsilon])
        summary[str(epsilon)] = {'median': float(errors.median()), 'max': float(errors.max())}
    report.update(rows=rows, summary=summary)


def master_gradients(core, batch):
    core.set_input(batch)
    return [g.detach().clone() for g in core.compute_gradient()]


def compare(estimated, exact):
    """Per DRN layer: cosine, relative error and first-Adam-step descent fraction."""
    rows = []
    pairs = list(zip(estimated, exact))
    for k in range(0, len(pairs), 2):
        a = torch.cat([e.flatten() for e, _ in pairs[k:k + 2]]).double()
        b = torch.cat([t.flatten() for _, t in pairs[k:k + 2]]).double()
        rows.append({'cosine': float(torch.nn.functional.cosine_similarity(a, b, 0)),
                     'relative_error': float((a - b).norm() / b.norm()),
                     # First direct-Adam command is -lr*sign(g); 1 = exact, 0 = random.
                     'sign_descent_fraction': float((b * a.sign()).sum() / b.abs().sum()),
                     'norm_ratio': float(a.norm() / b.norm()),
                     # Direct Adam's epsilon is 1e-8; gradients near it are damped.
                     'median_abs': float(a.abs().median())})
    return rows


def run_alignment(args, config, model, teacher, bank, train, dev, report):
    core = RecoveryStep(model, teacher, bank, config)
    batch = batch_at(train, 0, config)
    started = time.monotonic()
    exact = master_gradients(core, batch)
    report['exact_seconds'] = time.monotonic() - started
    report['first_analog_layer'] = model.start
    rows = []
    for arm in args.arms:
        for seed in range(args.repeats if 'zo' in arm else 1):
            args.probe_seed = 1000 + seed
            estimator = arm_estimator(arm, args, config.device)
            estimator.diagnose = True
            install(model, estimator)
            step = arm_core(arm, model, teacher, bank, config) if arm.startswith('local_') else core
            started = time.monotonic()
            estimated = master_gradients(step, batch)
            seconds = time.monotonic() - started
            losses = {k: v for k, v in step.last.items() if k in ('teacher_kl', 'local_losses')}
            del step
            local = {}
            for record in estimator.records:
                local.setdefault(record['layer'], []).append(record)
            row = {'arm': arm, 'repeat': seed, 'seconds_with_diagnostics': seconds, 'losses': losses,
                   'attention_forwards': estimator.attention_forwards,
                   'drn_layers': [{'layer': model.start + k, **r} for k, r in enumerate(compare(estimated, exact))],
                   'attention_blocks': [{'layer': layer, **{key: sum(r[key] for r in records) / len(records)
                                                            for key in records[0] if key != 'layer'}}
                                        for layer, records in sorted(local.items())]}
            rows.append(row)
            print(json.dumps({'arm': arm, 'repeat': seed,
                              'cosine': [round(r['cosine'], 4) for r in row['drn_layers']]}), flush=True)
    uninstall(model)
    report['rows'] = rows


def run_recover(args, config, model, teacher, bank, train, dev, report):
    estimator = arm_estimator(args.arm, args, config.device)
    install(model, estimator)
    core = arm_core(args.arm, model, teacher, bank, config)
    guard = FiniteGradientGuard(core.names)
    trace = args.output.with_suffix('.jsonl')
    trace.unlink(missing_ok=True)

    def log(row):
        with trace.open('a') as f:
            f.write(json.dumps(row) + '\n')

    def evaluate(step):
        before = time.monotonic()
        metrics = runtime.evaluate(model, teacher, dev, config.device, bank.view(model), config.training.microbatch)
        metrics.pop('numerical_diagnostics')
        log({'event': 'development', 'step': step, **metrics, 'seconds': time.monotonic() - before,
             'costs': bank.costs()})
        return metrics

    curve = [{'step': 0, **evaluate(0)}]
    started = time.monotonic()
    training_seconds = 0.0
    for step in range(config.recovery.updates):
        before = time.monotonic()
        forwards = estimator.attention_forwards
        batch = batch_at(train, step, config)
        train_epoch(core.components(), [(batch, batch)], start_global_step=step, event_handlers=(guard, core.observe))
        if str(config.device).startswith('cuda'):
            torch.cuda.synchronize()
        seconds = time.monotonic() - before
        training_seconds += seconds
        log({'event': 'update', 'step': step + 1, 'seconds': seconds,
             'probe_attention_forwards': estimator.attention_forwards - forwards, **core.last})
        if (step + 1) % config.recovery.development_every == 0:
            curve.append({'step': step + 1, **evaluate(step + 1)})
            print(json.dumps({'arm': args.arm, 'step': step + 1, 'teacher_kl': curve[-1]['teacher_kl'],
                              'seconds_per_update': training_seconds / (step + 1)}), flush=True)
    start, end = curve[0]['teacher_kl'], curve[-1]['teacher_kl']
    report.update(arm=args.arm, attention_backward={'mode': estimator.mode, 'directions': estimator.directions,
                                                    'control': estimator.control,
                                                    'epsilon_relative': estimator.epsilon,
                                                    'probe_seed': args.probe_seed},
                  updates=config.recovery.updates, development=curve,
                  kl_reduction=start - end, kl_reduction_fraction=(start - end) / start,
                  training_seconds=training_seconds, seconds_per_update=training_seconds / config.recovery.updates,
                  wall_seconds=time.monotonic() - started,
                  probe_attention_forwards=estimator.attention_forwards,
                  peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                  costs=bank.costs(), final_held_hashes=bank.held_hashes())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('epsilon', 'alignment', 'recover'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device-state', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--updates', type=int, default=128)
    parser.add_argument('--development-every', type=int, default=32)
    parser.add_argument('--arm', default='exact', help='exact, residual, stop, linearized, zo<m>, linzo<m> or local_<cached|onpolicy|stream>')
    parser.add_argument('--arms', nargs='+', default=['residual', 'zo1', 'zo4', 'zo16', 'zo64'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=1e-3, help='relative to the block-input RMS')
    parser.add_argument('--epsilons', type=float, nargs='+', default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument('--probes', type=int, default=16)
    parser.add_argument('--sequences', type=int, default=4)
    parser.add_argument('--probe-seed', type=int, default=20261028)
    args = parser.parse_args()
    if 'EBL_CORPUS_ROOT' not in os.environ:
        raise SystemExit('Set EBL_CORPUS_ROOT to the pinned token corpora.')
    started = time.monotonic()
    loaded = setup(args)
    report = {'schema': 'opt_cmo_drn.zo_attention_lab.v1', 'provenance': provenance(args, loaded[-1])}
    {'epsilon': run_epsilon, 'alignment': run_alignment, 'recover': run_recover}[args.command](
        args, *loaded[:-1], report)
    report['elapsed_seconds'] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, report)
    print('REPORT=' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
