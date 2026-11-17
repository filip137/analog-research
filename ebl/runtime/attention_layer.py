"""End-to-end analog attention layer (#61): one transient per example from token voltages to
DRN class voltages; block-by-block comparison with the equations.
Specification: campaigns/pilots/20261117-attention-layer.md."""
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import math
import time
import numpy as np
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.base import ConfigError
from networks.energy_transformer.spice import executable_identity
from networks.energy_transformer import attention_layer as al

ROOT = Path(__file__).resolve().parents[2]
WORKERS = 8
CLASSES = 6


def examples(seed, n):
    """attention_layer.py task distribution (N = 3, 6 direction sectors), numpy generator."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ph = rng.uniform(0, 2 * math.pi)
        ang = ph + rng.permutation(8)[:3] * 2 * math.pi / 8
        r, th = rng.uniform(.2, 1, 3), rng.uniform(0, 2 * math.pi, 3)
        ctx = np.stack([r * np.cos(th), r * np.sin(th), np.cos(ang), np.sin(ang)], 1)
        sel = i % 3
        out.append({'ctx': ctx, 'xq': np.array([0., 0., *ctx[sel, 2:]]), 'target': int(th[sel] // (2 * math.pi / CLASSES)) % CLASSES,
                    'selected': sel})
    return out


def spec_for(c, ex):
    keys = ('v_cm', 'scale', 's_ratio', 'tail_current', 'amplitude', 'v_low', 'c_score', 'c_hold', 't_write', 't_reset', 't_warm',
            't_settle', 'norm_tail', 'eps', 'norm_width', 'norm_length', 'norm_pedestal', 'v_rect', 'rect_bulk', 'v_u', 'drn_ref',
            'g_s', 'abstol')
    sp = {k: getattr(c, k) for k in keys}
    sp |= {'key_gp': c.key_gp, 'key_gm': c.key_gm, 'value_gp': c.value_gp, 'value_gm': c.value_gm, 'drn_g1': c.drn_g1,
           'drn_g2': c.drn_g2, 'geometry': {k: list(v) for k, v in c.geometry}, 'ctx': ex['ctx'], 'xq': ex['xq'],
           't_integrate': c.temperature_g * c.c_score * al.UT / (c.kappa_design * c.tail_current),
           'temperature_g_trained': c.law_temperature_g}
    return sp


def _sim(args):
    spec, model, d, exe = args
    try:
        return {'ok': True, 'r': al.simulate(spec, model, d, exe)}
    except Exception as err:
        return {'ok': False, 'error': str(err)[-400:]}


def characterize(c, store, exe):
    art = store.run_dir / 'artifacts'
    exs = examples(c.seed, c.n_examples)
    specs = [spec_for(c, ex) for ex in exs]
    t0 = time.monotonic()
    with ProcessPoolExecutor(WORKERS) as pool:
        out = list(pool.map(_sim, [(sp, c.model, art / f'ex-{i:03}', exe) for i, sp in enumerate(specs)]))
    if time.monotonic() - t0 > c.wall_seconds:
        raise TimeoutError('Declared characterization allowance exceeded')
    for i, o in enumerate(out):
        store.append_metric({'case': f'ex-{i:03}', 'ok': o['ok'],
                             'spice_class': int(np.argmax(o['r']['out'])) if o['ok'] else None, 'target': exs[i]['target']})
    bad = [o['error'] for o in out if not o['ok']]
    if bad:
        raise RuntimeError(f'{len(bad)} simulations failed; first: {bad[0]}')
    rows = []
    for o, ex, sp in zip(out, exs, specs):
        r = o['r']
        o_l, p_l = al.law_attention(sp, c.law_cell_kappa, c.law_cell_s, c.law_kappa)
        n_l = al.law_norm(o_l, c.eps)
        _, out_l = al.law_drn(al.drn_inputs(n_l, c.v_u), c.drn_g1, c.drn_g2, c.g_s)
        n_from_o = al.law_norm(r['o'], c.eps)
        _, out_from_u = al.law_drn(r['u'], c.drn_g1, c.drn_g2, c.g_s)
        rows.append({'target': ex['target'], 'spice_class': int(np.argmax(r['out'])), 'law_class': int(np.argmax(out_l)),
                     'o_spice': r['o'].tolist(), 'o_law': o_l.tolist(), 'n_spice': r['n'].tolist(), 'n_of_o_spice': n_from_o.tolist(),
                     'u_spice': r['u'].tolist(), 'u_of_n_spice': al.drn_inputs(r['n'], c.v_u).tolist(),
                     'drn_out_spice': r['out'].tolist(), 'drn_out_law_of_u': out_from_u.tolist(), 'drn_out_law': out_l.tolist(),
                     'p_spice': r['softmax'].tolist(), 'p_law': p_l.tolist(), 'selected': ex['selected'],
                     'late_drift_out': r['late_drift_out'], 'energy': r['energy'], 'drn_power_W': r['drn_power_W'],
                     'seconds': r['seconds']})
    f = lambda key: np.array([r[key] for r in rows])
    acc_s = float(np.mean(f('spice_class') == f('target')))
    acc_l = float(np.mean(f('law_class') == f('target')))
    agree = float(np.mean(f('spice_class') == f('law_class')))
    m = {'accuracy_spice': acc_s, 'accuracy_law_same_examples': acc_l, 'class_agreement_spice_law': agree,
         'attention_o_rms_vs_law': float(np.sqrt(np.mean((f('o_spice') - f('o_law')) ** 2))),
         'norm_n_rms_vs_law_of_o_spice': float(np.sqrt(np.mean((f('n_spice') - f('n_of_o_spice')) ** 2))),
         'norm_n_max_abs_vs_law_of_o_spice': float(np.max(np.abs(f('n_spice') - f('n_of_o_spice')))),
         'interface_u_max_abs_V': float(np.max(np.abs(f('u_spice') - f('u_of_n_spice')))),
         'drn_out_max_abs_vs_law_of_u_V': float(np.max(np.abs(f('drn_out_spice') - f('drn_out_law_of_u')))),
         'softmax_on_selected': float(np.mean([r['p_spice'][r['selected']] for r in rows])),
         'late_drift_out_max_V': float(f('late_drift_out').max()),
         'energy_front_end_J': float(np.mean([r['energy']['total'] for r in rows])),
         'energy_core_J': float(np.mean([r['energy']['core'] for r in rows])),
         'energy_crossbar_J': float(np.mean([r['energy']['crossbar_tokens'] for r in rows])),
         'drn_power_W': float(f('drn_power_W').mean()),
         'drn_energy_per_2us_read_J': float(f('drn_power_W').mean() * 2e-6),
         'step_seconds': float(np.mean([r['energy']['step_seconds'] for r in rows])),
         'spice_seconds': float(f('seconds').sum())}
    net, _, counts = al.netlist(specs[0], c.model)
    m['counts'] = counts
    if c.model == 'ideal':      # wiring: every block against its own law
        checks = {'attention_wiring': m['attention_o_rms_vs_law'] <= 0.02, 'norm_wiring': m['norm_n_rms_vs_law_of_o_spice'] <= 0.03,
                  'drn_wiring': m['drn_out_max_abs_vs_law_of_u_V'] <= 3e-3}
    else:
        checks = {'accuracy': acc_s >= .85 and acc_s >= acc_l - .05, 'drn_wiring': m['drn_out_max_abs_vs_law_of_u_V'] <= 3e-3}
    summary = {'model': c.model, 'passed': all(checks.values()), 'checks': checks, 'metrics': m, 'rows': rows,
               'weights_source': c.weights_source, 'weights_source_sha256': c.weights_source_sha256,
               'claim_boundary': ('One attention-layer step per example: token crossbar, transmission-gate KV/query/reference '
                                  'holds, 1 um anti-bump scorer with tail gating and capacitor integration, PMOS softmax, NMOS '
                                  'diff-pair value weighting (2 features), subthreshold rectifier + translinear L1 normalizer, '
                                  'then the DRN (resistors in the CMO range, behavioural perfect diodes). Ideal: token, clock, '
                                  'bias and rail sources, the current-to-voltage buffers into the DRN, output probes; software '
                                  'argmax. Weights trained in equations. Generic BSIM3, no mismatch, noise, PDK or silicon claim.')}
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
        artifacts = [store.artifact_record(p, kind='circuit_characterization') for p in sorted((store.run_dir / 'artifacts').rglob('*')) if p.is_file()]
        m = res['metrics']
        store.complete(metrics={'passed': res['passed'], **{k: v for k, v in m.items() if k != 'counts'}}, artifacts=artifacts)
        print(res['checks'], f"acc {m['accuracy_spice']:.3f} law {m['accuracy_law_same_examples']:.3f}", flush=True)
        return 0 if res['passed'] else 1
    except BaseException as err:
        store.fail(err)
        raise
