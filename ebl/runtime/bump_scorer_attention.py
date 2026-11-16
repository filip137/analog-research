"""Declared selection, law, derivative and settling checks of the bump-chain scorer.

Specification: campaigns/pilots/20261116-bump-scorer-attention.md.
"""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.base import ConfigError
from networks.energy_transformer.spice import executable_identity
from networks.energy_transformer import pseudoconductance as circuit
from networks.energy_transformer import bump_scorer as bump

ROOT = Path(__file__).resolve().parents[2]
BASE_VALUES = np.array([[-.6, .4], [.2, -.5], [.7, .1]])
BASE_KEYS = np.array([[.85, .90], [1.20, .95], [1.00, 1.25]])
DERIVATIVE_OFFSET = np.array([.040, -.025])
FAR_QUERY = np.array([1.6, 1.6])
KEY_LOW, KEY_HIGH, KEY_SEPARATION = .8, 1.3, .25
FIT_FLOOR = 1e-3            # sweep currents below this are excluded from the kappa fit
EPSILON = 1e-3              # volts, central differences of query/key voltages
BETA = .001
ERROR = np.array([.7, -.4])
IDEAL_KAPPA = 2 / 3
TRANSIENT_ABSTOL = 1e-18     # revision 1: 1e-20 A aborted the 0.2 V BSIM3 start (fA branch currents)


def relative_error(actual, expected):
    actual, expected = np.ravel(actual), np.ravel(expected)
    absolute = float(np.linalg.norm(actual - expected))
    norm = float(np.linalg.norm(expected))
    return {'absolute_l2': absolute, 'reference_l2': norm,
            'relative_l2': absolute / norm if norm > 1e-6 else None,
            'passed': bool(absolute / norm < .05 if norm > 1e-6 else absolute < 1e-4)}


def fresh_keys(rng):
    while True:
        keys = rng.uniform(KEY_LOW, KEY_HIGH, (3, 2))
        sep = min(np.max(abs(keys[i] - keys[j])) for i in range(3) for j in range(i + 1, 3))
        if sep >= KEY_SEPARATION:
            return keys


def fit_kappa(queries, keys, measured, tail, iref):
    """Least squares of ln a against ln law over kappa and a free log-amplitude."""
    mask = measured >= FIT_FLOOR
    best = None
    for kappa in np.linspace(.4, 1.2, 8001):
        pred = np.array([bump.law(q, keys, tail, iref, kappa) for q in queries])
        r = np.log(measured[mask]) - np.log(pred[mask])
        offset = r.mean()
        sse = float(np.sum((r - offset) ** 2))
        if best is None or sse < best[0]:
            best = (sse, kappa, offset)
    sse, kappa, offset = best
    return {'kappa': float(kappa), 'log_amplitude_ratio': float(offset),
            'amplitude_ratio': float(np.exp(offset)),
            'rms_log_residual': float(np.sqrt(sse / mask.sum())), 'points': int(mask.sum())}


def characterize(c, store, exe):
    started = time.monotonic()
    artifacts = store.run_dir / 'artifacts'
    stats = []

    def run(name, values, query=None, keys=None, controls=None, **kw):
        if time.monotonic() - started > c.wall_seconds:
            raise TimeoutError('Declared characterization allowance exceeded')
        scorer = None if query is None else dict(query=query, keys=keys, tail_current=c.tail_current)
        result = circuit.simulate(values, controls, artifacts / name, exe, model=c.model, scorer=scorer, **kw)
        extra = {}
        if scorer is not None:
            result['a'] = bump.measured(result['port'], circuit.IREF)
            extra['min_tail_voltage'] = bump.min_tail_voltage(result['port'])
        stats.append(result['metrics'] | extra | {'seconds': result['seconds']})
        store.append_metric({'case': name, 'spice_calls': len(stats), 'output': result['y'].tolist(),
                             'spice_seconds': sum(s['seconds'] for s in stats)})
        if len(stats) % 12 == 0:
            print(f'{c.model}: cases={len(stats)}, last={name}', flush=True)
        return result

    rng = np.random.default_rng(c.seed)
    values = rng.uniform(-.75, .75, (36, 3, 2))
    keyset = np.array([fresh_keys(rng) for _ in range(36)])
    selected = np.arange(36) % 3
    sel_out, uni_out, op_out, a_sel = [], [], [], []
    for i in range(36):
        r = run(f'select-{i:02}', values[i], keyset[i, selected[i]], keyset[i])
        sel_out.append(r['y']); a_sel.append(r['a'])
        op_out.append(circuit.reference(values[i], r['a']))
        uni_out.append(run(f'uniform-{i:02}', values[i], controls=np.full(3, r['a'].mean()))['y'])
    sel_out, uni_out, op_out, a_sel = map(np.array, (sel_out, uni_out, op_out, a_sel))
    targets = values[np.arange(36), selected]

    sweep_q = [BASE_KEYS[0] + t * (BASE_KEYS[1] - BASE_KEYS[0]) for t in np.linspace(0, 1, 11)]
    sweep_a = np.array([run(f'sweep-{i:02}', BASE_VALUES, q, BASE_KEYS)['a'] for i, q in enumerate(sweep_q)])
    fit = fit_kappa(sweep_q, BASE_KEYS, sweep_a, c.tail_current, circuit.IREF)
    kappa = IDEAL_KAPPA if c.model == 'ideal' else fit['kappa']

    far = run('far-query', BASE_VALUES, FAR_QUERY, BASE_KEYS)

    q0 = BASE_KEYS[0] + DERIVATIVE_OFFSET
    free = run('derivative-free', BASE_VALUES, q0, BASE_KEYS)
    plus = run('nudge-plus', BASE_VALUES, q0, BASE_KEYS, nudge=BETA, error=ERROR)
    minus = run('nudge-minus', BASE_VALUES, q0, BASE_KEYS, nudge=-BETA, error=ERROR)
    lam = -(plus['x_read'] - minus['x_read']) / (2 * BETA)
    flux = np.array([[free['port'][f'edge_{j}_{d}'][-1] / circuit.IREF for d in range(2)] for j in range(3)])
    dlam = lam[:3] - lam[3]
    g = np.sum(flux * dlam, axis=1) / free['a']                    # measured-branch dL/da_j

    fd_aq = np.zeros((3, 2)); fd_ak = np.zeros((3, 2))
    fd_lq = np.zeros(2); fd_lk = np.zeros((3, 2))
    for d in range(2):
        step = np.zeros(2); step[d] = EPSILON
        p = run(f'query-fd-{d}-plus', BASE_VALUES, q0 + step, BASE_KEYS)
        m = run(f'query-fd-{d}-minus', BASE_VALUES, q0 - step, BASE_KEYS)
        fd_aq[:, d] = (p['a'] - m['a']) / (2 * EPSILON)
        fd_lq[d] = ERROR @ (p['y'] - m['y']) / (2 * EPSILON)
    for j, d in np.ndindex(3, 2):
        step = np.zeros((3, 2)); step[j, d] = EPSILON
        p = run(f'key-fd-{j}-{d}-plus', BASE_VALUES, q0, BASE_KEYS + step)
        m = run(f'key-fd-{j}-{d}-minus', BASE_VALUES, q0, BASE_KEYS - step)
        fd_ak[j, d] = (p['a'][j] - m['a'][j]) / (2 * EPSILON)
        fd_lk[j, d] = ERROR @ (p['y'] - m['y']) / (2 * EPSILON)
    d0 = q0[None, :] - BASE_KEYS
    slope = -(kappa / bump.UT) * np.tanh(kappa * d0 / (2 * bump.UT))
    law_aq = free['a'][:, None] * slope                           # measured a times law slope
    law_ak = -law_aq
    est_lq = g @ law_aq
    est_lk = g[:, None] * law_ak

    transient = [run(f'transient-{i}', BASE_VALUES, q0, BASE_KEYS, transient=True, initial=s,
                     transient_seconds=c.transient_seconds, transient_abstol=TRANSIENT_ABSTOL)
                 for i, s in enumerate([.2, .4])]

    sel_mse = float(np.mean((sel_out - targets) ** 2))
    uni_mse = float(np.mean((uni_out - targets) ** 2))
    scorer_grad = relative_error(np.concatenate([law_aq.ravel(), law_ak.ravel()]),
                                 np.concatenate([fd_aq.ravel(), fd_ak.ravel()]))
    scorer_grad_key0 = relative_error(np.concatenate([law_aq[0], law_ak[0]]),
                                      np.concatenate([fd_aq[0], fd_ak[0]]))
    e2e_q = relative_error(est_lq, fd_lq)
    e2e_k = relative_error(est_lk, fd_lk)
    settling = max(float(np.max(abs(t['y'] - free['y']))) for t in transient)
    ar = np.arange(36)
    metrics = {
        'selective_rmse': float(np.sqrt(sel_mse)), 'uniform_rmse': float(np.sqrt(uni_mse)),
        'uniform_over_selective_mse': uni_mse / sel_mse,
        'selective_vs_operator_rmse': float(np.sqrt(np.mean((sel_out - op_out) ** 2))),
        'selected_current_mean': float(a_sel[ar, selected].mean()),
        'selected_current_range': [float(a_sel[ar, selected].min()), float(a_sel[ar, selected].max())],
        'unselected_current_max': float(np.max(np.where(np.eye(3, dtype=bool)[selected], 0, a_sel))),
        'selected_over_law_peak': float(a_sel[ar, selected].mean() / (c.tail_current / circuit.IREF / 16)),
        'kappa_fit': fit, 'kappa_used': kappa,
        'far_query_output': far['y'].tolist(), 'far_query_currents': far['a'].tolist(),
        'scorer_gradient': scorer_grad, 'scorer_gradient_key0': scorer_grad_key0,
        'end_to_end_query_gradient': e2e_q, 'end_to_end_key_gradient': e2e_k,
        'transient_vs_dc_max_abs_error': settling,
        'min_node_voltage': min(s['min_node_voltage'] for s in stats),
        'max_node_voltage': max(s['max_node_voltage'] for s in stats),
        'min_anchor_vds': min(s['min_anchor_vds'] for s in stats),
        'min_scorer_tail_voltage': min(s['min_tail_voltage'] for s in stats if 'min_tail_voltage' in s),
        'max_read_mirror_ratio_error': max(s['mirror_ratio_error'] for s in stats),
        'max_supply_power_W': max(s['supply_power_W'] for s in stats),
        'spice_seconds': sum(s['seconds'] for s in stats), 'spice_calls': len(stats)}
    checks = {'selection': metrics['selective_rmse'] < .05 and uni_mse > 5 * sel_mse,
              'scorer_gradient': scorer_grad['passed'],
              'settling': settling < 1e-3,
              'voltage_range': metrics['min_node_voltage'] > 0 and metrics['max_node_voltage'] < circuit.VDD,
              'anchor_compliance': metrics['min_anchor_vds'] > 4 * circuit.UT}
    summary = {
        'model': c.model, 'tail_current': c.tail_current, 'transient_seconds': c.transient_seconds,
        'forward_passed': checks['selection'] and checks['settling'] and checks['voltage_range'] and checks['anchor_compliance'],
        'backward_passed': checks['scorer_gradient'],
        'passed': all(checks.values()), 'checks': checks, 'metrics': metrics,
        'counts': free['counts'], 'total_mos': sum(free['counts'].values()),
        'sweep': {'queries': np.array(sweep_q).tolist(), 'measured': sweep_a.tolist(),
                  'law_fitted': [(fit['amplitude_ratio'] * bump.law(q, BASE_KEYS, c.tail_current, circuit.IREF, fit['kappa'])).tolist() for q in sweep_q],
                  'law_nominal': [bump.law(q, BASE_KEYS, c.tail_current, circuit.IREF, kappa).tolist() for q in sweep_q]},
        'derivatives': {'query_point': q0.tolist(), 'currents': free['a'].tolist(),
                        'fd_da_dq': fd_aq.tolist(), 'law_da_dq': law_aq.tolist(),
                        'fd_da_dk': fd_ak.tolist(), 'law_da_dk': law_ak.tolist(),
                        'dL_da_branch': g.tolist(), 'fd_dL_dq': fd_lq.tolist(), 'est_dL_dq': est_lq.tolist(),
                        'fd_dL_dk': fd_lk.tolist(), 'est_dL_dk': est_lk.tolist()},
        'claim_boundary': ('One query, three 2-D keys, two value features. Explicit MOS bump-chain scorer '
                           '(differential pairs, PMOS loads, series correlators, tail mirrors) drives the '
                           'unchanged 51-MOS pseudoconductance bank. Ideal external query/key voltages, tail '
                           'reference, supply/reference/read-load voltages, nudge source and SPICE current '
                           'probes; software phase storage, statistics and law derivatives. Generic BSIM3 or '
                           'ideal laws; no PDK, mismatch, noise, projection circuit, analog scorer backward '
                           'pass, training, silicon or energy claim.')}
    np.savez_compressed(artifacts / 'selection.npz', values=values, keys=keyset, selected=selected,
                        targets=targets, selective=sel_out, uniform=uni_out, operator=op_out, currents=a_sel)
    atomic_write_json(artifacts / 'summary.json', summary)
    return summary


def run_characterize(request):
    if any(getattr(request, k, None) is not None for k in ['weights', 'base_weights', 'resume', 'device_data', 'device_model']):
        raise ConfigError('This standalone circuit pilot does not accept checkpoints or device datasets')
    c = request.spec
    exe, version = executable_identity(c.spice_executable)
    store = RunStore.create(output_root=request.output_dir, experiment_id=c.experiment_id,
                            resolved_config=asdict(c), command=request.command, repo_root=ROOT,
                            input_artifacts=[{'path': str(exe), 'sha256': sha256_file(exe)}],
                            resume_capability='unsupported')
    print(f'RUN_DIR={store.run_dir}', flush=True)
    try:
        atomic_write_json(store.run_dir / 'artifacts' / 'environment.json',
                          {'simulator': str(exe), 'version': version, 'numpy': np.__version__})
        result = characterize(c, store, exe)
        artifacts = [store.artifact_record(p, kind='circuit_characterization')
                     for p in sorted((store.run_dir / 'artifacts').rglob('*')) if p.is_file()]
        if sum(a.size_bytes for a in artifacts) > 250_000_000:
            raise RuntimeError('Per-model storage allowance exceeded')
        store.complete(metrics={'passed': result['passed'], **result['metrics']}, artifacts=artifacts)
        print(result['checks'], flush=True)
        return 0 if result['passed'] else 1
    except BaseException as err:
        store.fail(err)
        raise
