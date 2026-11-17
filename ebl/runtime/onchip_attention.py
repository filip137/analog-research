"""SPICE-in-the-loop on-chip learning of chain A (#62).

Each training example is one transient: write, query, integrate, free settle, then +beta and -beta
loss nudges at the bank output node. Gradients use only measured signals (bank coordinates in
the nudged phases, value currents, softmax currents, held q/k voltages, summed score currents)
plus the crossbar formula and a cell derivative law; the update (Adam on bounded conductances,
the integration window and the readout) is digital. Specification:
campaigns/pilots/20261117-onchip-attention.md.
"""
from dataclasses import asdict, replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import math
import shutil
import time
import numpy as np
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.base import ConfigError
from networks.energy_transformer.spice import executable_identity
from networks.energy_transformer import end_to_end as e2e
from ebl.runtime.attention_energy import spec_for as _energy_spec

ROOT = Path(__file__).resolve().parents[2]


def energy_spec(c, ex, t_settle=None):
    return _energy_spec(c, ex, t_settle) | {'mirror_bias': c.mirror_bias}
WORKERS = 12
GMIN, GMAX = 9e-6, 88.2e-6
UT, IREF, H, H0 = e2e.UT, e2e.IREF, 1.0, e2e.H0


def task(rng, n):
    """keyed_retrieval distribution (N = 3, 8-slot circle, values in +-0.4), every slot used as query."""
    out = []
    for i in range(n):
        ph = rng.uniform(0, 2 * math.pi)
        ang = ph + rng.permutation(8)[:3] * 2 * math.pi / 8
        vals = rng.uniform(-.4, .4, 3)
        ctx = np.stack([vals, np.cos(ang), np.sin(ang)], 1)
        sel = i % 3
        out.append({'ctx': ctx, 'xq': np.array([0., ctx[sel, 1], ctx[sel, 2]]), 'target': float(vals[sel]), 'selected': sel})
    return out


sig = lambda x: 1 / (1 + np.exp(-x))
cond = lambda p: GMIN + (GMAX - GMIN) * sig(p)
logit = lambda G: np.log((G - GMIN) / (GMAX - G))
softplus = lambda x: np.log1p(np.exp(x))


class Params:
    def __init__(self, c):
        if c.init == 'random':
            r = np.random.default_rng(c.init_seed)
            self.v = {'Kp': r.standard_normal((2, 3)) * .5, 'Km': r.standard_normal((2, 3)) * .5,
                      'Vp': r.standard_normal((1, 3)) * .5, 'Vm': r.standard_normal((1, 3)) * .5,
                      'g': np.array(0.), 'alpha': np.array(1.), 'gamma': np.array(0.)}
        else:          # start from the configured design (weights in the config)
            self.v = {'Kp': logit(np.array(c.key_gp)), 'Km': logit(np.array(c.key_gm)),
                      'Vp': logit(np.array(c.value_gp)), 'Vm': logit(np.array(c.value_gm)),
                      'g': np.array(math.log(math.expm1(c.temperature_g / 5))),
                      'alpha': np.array(c.readout_alpha), 'gamma': np.array(c.readout_gamma)}
        self.m = {k: np.zeros_like(v) for k, v in self.v.items()}
        self.s = {k: np.zeros_like(v) for k, v in self.v.items()}
        self.t = 0

    def design(self, c):
        return replace(c, key_gp=tuple(map(tuple, cond(self.v['Kp']))), key_gm=tuple(map(tuple, cond(self.v['Km']))),
                       value_gp=tuple(map(tuple, cond(self.v['Vp']))), value_gm=tuple(map(tuple, cond(self.v['Vm']))),
                       temperature_g=float(softplus(self.v['g']) * 5), readout_alpha=float(self.v['alpha']),
                       readout_gamma=float(self.v['gamma']))

    def adam(self, g, lr, b1=.9, b2=.999, eps=1e-8):
        self.t += 1
        for k in self.v:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.s[k] = b2 * self.s[k] + (1 - b2) * g[k] ** 2
            self.v[k] = self.v[k] - lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.s[k] / (1 - b2 ** self.t)) + eps)


def cell_fp(d, S, kc):
    u = np.clip(kc * d / (2 * UT), -30, 30)
    c = 4 / S
    ch = np.cosh(u) ** 2
    return c * np.sinh(2 * u) * kc / (2 * UT) / (1 + c * ch) ** 2


def gradient(c, d, rows, exs):
    """Local rule from measured signals; returns parameter gradients (batch mean) and the train MSE."""
    al, ga, B = d.readout_alpha, d.readout_gamma, len(rows)
    G = {k: np.array(getattr(d, n)) for k, n in (('Kp', 'key_gp'), ('Km', 'key_gm'), ('Vp', 'value_gp'), ('Vm', 'value_gm'))}
    Sk, Sv = (G['Kp'] + G['Km']).sum(-1), (G['Vp'] + G['Vm']).sum(-1)
    gr = {k: np.zeros_like(v) for k, v in G.items()} | {'g': 0., 'alpha': 0., 'gamma': 0.}
    sq = []
    for r, ex in zip(rows, exs):
        ph = r['phases']
        xp, xm = np.array(ph['plus']['x']), np.array(ph['minus']['x'])
        xop, xom, xof = ph['plus']['xo'], ph['minus']['xo'], ph['free']['xo']
        y = al * (xof - 1) + ga
        e = y - ex['target']
        sq.append(e ** 2)
        sc = al ** 2 / (2 * c.beta * B)
        g_a = sc * .5 * ((xp - xop) ** 2 - (xm - xom) ** 2)
        g_z = -sc * H * (xp - xm)
        p = np.array(r['softmax'])
        g_s = c.amplitude * p * (g_a - (p * g_a).sum())
        fsum = np.array(r['isum_integrating']) / c.tail_current
        gr['g'] += float((g_s * (-fsum)).sum() * sig(d_g := math.log(math.expm1(d.temperature_g / 5))) * 5)
        q = np.array(r['query_held'])
        k = np.array([[r['kv'][f'k{j}{dd}']['held'] for dd in range(2)] for j in range(3)])
        g_d = g_s[:, None] * (-d.temperature_g) * cell_fp(q[None] - k, c.rule_cell_s, c.rule_cell_kappa)
        g_q, g_k = g_d.sum(0), -g_d
        lqp, lqm = c.v_cm + d.scale * ex['xq'], c.v_cm - d.scale * ex['xq']
        lcp, lcm = c.v_cm + d.scale * ex['ctx'], c.v_cm - d.scale * ex['ctx']
        wq = lambda gp, gm, S, lp, lm: (lp @ gp.T + lm @ gm.T) / S       # crossbar column voltages (formula)
        qf, kf = wq(G['Kp'], G['Km'], Sk, lqp, lqm), wq(G['Kp'], G['Km'], Sk, lcp, lcm)
        vf = wq(G['Vp'], G['Vm'], Sv, lcp, lcm)[:, 0]
        gr['Kp'] += (g_q[:, None] * (lqp[None] - qf[:, None]) + (g_k[:, :, None] * (lcp[:, None] - kf[:, :, None])).sum(0)) / Sk[:, None]
        gr['Km'] += (g_q[:, None] * (lqm[None] - qf[:, None]) + (g_k[:, :, None] * (lcm[:, None] - kf[:, :, None])).sum(0)) / Sk[:, None]
        z = np.array(r['z'])
        g_v = g_z * c.kappa_value / (2 * UT) * (1 - (z - 1) ** 2)
        gr['Vp'] += ((g_v[:, None] * (lcp - vf[:, None])).sum(0) / Sv)[None]
        gr['Vm'] += ((g_v[:, None] * (lcm - vf[:, None])).sum(0) / Sv)[None]
        gr['alpha'] += e * (xof - 1) / B
        gr['gamma'] += e / B
    out = {k: gr[k] * (G[k] - GMIN) * (GMAX - G[k]) / (GMAX - GMIN) for k in G}
    out |= {'g': np.array(gr['g']), 'alpha': np.array(gr['alpha']), 'gamma': np.array(gr['gamma'])}
    return out, float(np.mean(sq))


def _sim(args):
    spec, model, directory, exe = args
    try:
        r = e2e.simulate(spec, model, directory, exe)
        shutil.rmtree(directory, ignore_errors=True)               # traces of training steps are not retained
        return {'ok': True, 'r': r}
    except Exception as err:
        return {'ok': False, 'error': str(err)[-300:]}


def characterize(c, store, exe):
    started = time.monotonic()
    art = store.run_dir / 'artifacts'
    params = Params(c)
    test = task(np.random.default_rng(c.seed), c.n_test)
    rng = np.random.default_rng(c.train_seed)
    pool = ProcessPoolExecutor(WORKERS)
    history, evals, seconds = [], [], [0.]

    def evaluate(step):
        d = params.design(c)
        out = list(pool.map(_sim, [(energy_spec(d, ex), c.model, art / 'tmp' / f'eval{step}-{i:03}', exe) for i, ex in enumerate(test)]))
        ok = [(o['r'], ex) for o, ex in zip(out, test) if o['ok']]
        seconds[0] += sum(o['r']['seconds'] for o in out if o['ok'])
        y = np.array([d.readout_alpha * r['y_raw'][0] + d.readout_gamma for r, _ in ok])
        t = np.array([ex['target'] for _, ex in ok])
        ctx = np.array([ex['ctx'][:, 0].mean() for _, ex in ok])
        rec = {'step': step, 'test_mse': float(np.mean((y - t) ** 2)), 'context_average_mse': float(np.mean((ctx - t) ** 2)),
               'failed': len(out) - len(ok), 'temperature_g': d.temperature_g, 'alpha': d.readout_alpha, 'gamma': d.readout_gamma}
        evals.append(rec)
        store.append_metric({'case': f'eval-{step}'} | rec)
        print(f'eval step {step}: test mse {rec["test_mse"]:.3e} (ctx {rec["context_average_mse"]:.3e})', flush=True)

    evaluate(0)
    for step in range(1, c.steps + 1):
        if time.monotonic() - started > c.wall_seconds:
            raise TimeoutError('Declared allowance exceeded')
        d = params.design(c)
        exs = task(rng, c.batch)
        specs = []
        for ex in exs:
            xt = 1 + (ex['target'] - d.readout_gamma) / d.readout_alpha
            specs.append(energy_spec(d, ex, t_settle=c.t_settle_train) | {'nudge': {'beta': c.beta, 'x_target': xt, 't_phase': c.t_phase}})
        out = list(pool.map(_sim, [(sp, c.model, art / 'tmp' / f's{step}-{i:02}', exe) for i, sp in enumerate(specs)]))
        ok = [(o['r'], ex) for o, ex in zip(out, exs) if o['ok']]
        seconds[0] += sum(o['r']['seconds'] for o in out if o['ok'])
        if not ok:
            raise RuntimeError('All training simulations failed: ' + out[0]['error'])
        g, train_mse = gradient(c, d, [r for r, _ in ok], [ex for _, ex in ok])
        params.adam(g, c.lr)
        rec = {'step': step, 'train_mse_batch': train_mse, 'failed': len(out) - len(ok),
               'grad_norm': float(np.sqrt(sum((v ** 2).sum() for v in g.values())))}
        history.append(rec)
        store.append_metric({'case': f'step-{step}'} | rec)
        if step % c.eval_every == 0 or step == c.steps:
            evaluate(step)
    pool.shutdown()
    shutil.rmtree(art / 'tmp', ignore_errors=True)
    final = evals[-1]
    passed = final['test_mse'] <= 1e-3 and final['context_average_mse'] >= 5 * final['test_mse']
    d = params.design(c)
    summary = {'model': c.model, 'init': c.init, 'passed': passed,
               'checks': {'retrieval_after_training': passed},
               'metrics': {'initial_test_mse': evals[0]['test_mse'], 'final_test_mse': final['test_mse'],
                           'context_average_mse': final['context_average_mse'],
                           'context_over_retrieval': final['context_average_mse'] / final['test_mse'],
                           'training_simulations': c.steps * c.batch, 'failed_simulations': sum(h['failed'] for h in history),
                           'spice_seconds': seconds[0], 'wall_seconds': time.monotonic() - started},
               'evals': evals, 'history': history,
               'final_design': {'key_gp': d.key_gp, 'key_gm': d.key_gm, 'value_gp': d.value_gp, 'value_gm': d.value_gm,
                                'temperature_g': d.temperature_g, 'readout_alpha': d.readout_alpha, 'readout_gamma': d.readout_gamma},
               'claim_boundary': ('Generic BSIM3 chain A circuit. Gradient signals measured in SPICE: bank node coordinates '
                                  'under a behavioural loss nudge at the output node, value and softmax currents, held '
                                  'voltages, summed score currents; the backward products, the cell derivative law, the '
                                  'crossbar formula and the Adam update are digital, conductances bounded to 9-88.2 uS. '
                                  'No mismatch, noise, PDK or silicon claim.')}
    atomic_write_json(art / 'summary.json', summary)
    return summary


def run_characterize(request):
    if any(getattr(request, k, None) is not None for k in ['weights', 'base_weights', 'resume', 'device_data', 'device_model']):
        raise ConfigError('This standalone circuit pilot does not accept checkpoints or device datasets')
    c = request.spec
    exe, version = executable_identity(c.spice_executable)
    store = RunStore.create(output_root=request.output_dir, experiment_id=c.experiment_id, resolved_config=asdict(c),
                            command=request.command, repo_root=ROOT, input_artifacts=[{'path': str(exe), 'sha256': sha256_file(exe)}],
                            resume_capability='unsupported')
    print(f'RUN_DIR={store.run_dir}', flush=True)
    try:
        atomic_write_json(store.run_dir / 'artifacts' / 'environment.json', {'simulator': str(exe), 'version': version, 'numpy': np.__version__})
        res = characterize(c, store, exe)
        artifacts = [store.artifact_record(p, kind='circuit_training') for p in sorted((store.run_dir / 'artifacts').rglob('*')) if p.is_file()]
        store.complete(metrics={'passed': res['passed'], **res['metrics']}, artifacts=artifacts)
        print(res['checks'], f"test mse {res['metrics']['initial_test_mse']:.3e} -> {res['metrics']['final_test_mse']:.3e}", flush=True)
        return 0 if res['passed'] else 1
    except BaseException as err:
        store.fail(err)
        raise
