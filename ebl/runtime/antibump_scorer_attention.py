"""Declared retrieval, cell-law, D=8 summation and settling checks of the summed
anti-bump scorer (star or Elfadel-Wyatt softmax) driving the pseudoconductance bank.

Specification: campaigns/pilots/20261116-antibump-scorer.md.
"""
from dataclasses import asdict
from pathlib import Path
import math
import time
import numpy as np
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.base import ConfigError
from networks.energy_transformer.spice import executable_identity
from networks.energy_transformer import pseudoconductance as circuit
from networks.energy_transformer import antibump_scorer as ab
from ebl.runtime.crossbar_qk_attention import toy_examples, CASES

ROOT = Path(__file__).resolve().parents[2]
TRANSIENT_ABSTOL = 1e-16     # readiness: 1e-18 A aborted the star BSIM3 0.2 V start (fA coupler branch)
SWEEP = np.linspace(-.3, .3, 13)
BASE_KEYS = np.array([[.85, .90], [1.20, .95], [1.00, 1.25]])
BASE_VALUES = np.array([[-.6, .4], [.2, -.5], [.7, .1]])
SUM_DIMS, SUM_CASES = 8, 6


def fit_cell(deltas, measured, tail):
    """Grid fit of effective S and kappa to the summed (2-cell) current; dim 1 at delta 0."""
    best = None
    for s in np.geomspace(4, 400, 161):
        for kap in np.linspace(.4, 1.2, 161):
            c = 4 / s
            pred = tail * (c * np.cosh(kap * deltas / (2 * ab.UT)) ** 2 / (1 + c * np.cosh(kap * deltas / (2 * ab.UT)) ** 2)
                           + c / (1 + c))
            sse = float(np.sum((pred - measured) ** 2))
            if best is None or sse < best[0]:
                best = (sse, s, kap)
    sse, s, kap = best
    return {'s_effective': float(s), 'kappa': float(kap),
            'rms_residual_over_tail': float(math.sqrt(sse / len(deltas)) / tail)}


def characterize(c, store, exe):
    started = time.monotonic()
    artifacts = store.run_dir / 'artifacts'
    W = np.array(c.projection)
    vw = np.array(c.value_weights)
    R = c.temperature_g * ab.UT / (c.kappa_design * c.tail_current)
    stats = []

    def spec(q, k):
        return {'kind': 'antibump', 'variant': c.variant, 'query': q, 'keys': k, 's_ratio': c.s_ratio,
                'tail_current': c.tail_current, 'resistance': R, 'amplitude': c.amplitude,
                'iref': circuit.IREF, 'v_low': c.v_low, 'drives_gate': c.variant == 'star'}

    def guard():
        if time.monotonic() - started > c.wall_seconds:
            raise TimeoutError('Declared characterization allowance exceeded')

    def run(name, values, q=None, k=None, controls=None, **kw):
        guard()
        sc = None if q is None else spec(q, k)
        r = circuit.simulate(values, controls, artifacts / name, exe, model=c.model, scorer=sc, **kw)
        if sc is not None:
            r['isum'] = ab.measured_scores(r['port'])
        stats.append(r['metrics'] | {'seconds': r['seconds']})
        store.append_metric({'case': name, 'spice_calls': len(stats), 'output': r['y'].tolist(),
                             'spice_seconds': sum(s['seconds'] for s in stats)})
        if len(stats) % 12 == 0:
            print(f'{c.model}/{c.variant}: cases={len(stats)}, last={name}', flush=True)
        return r

    def attention_from_scores(isum):
        expo = -c.kappa_design * R * isum / ab.UT
        if c.variant == 'star':
            return c.amplitude * np.exp(expo)
        e = np.exp(expo - expo.max())
        return c.amplitude * e / e.sum()

    examples = toy_examples(np.random.default_rng(c.seed))
    spice, uniform, equation, targets, ctxavg, isum_meas, isum_law = [], [], [], [], [], [], []
    for i, ex in enumerate(examples):
        v = ex['ctx'] @ vw
        values = np.stack([v, -v], 1)
        q = c.v_cm + c.scale * ex['xq'] @ W.T
        k = c.v_cm + c.scale * ex['ctx'] @ W.T
        r = run(f'retrieve-{i:02}', values, q, k)
        a_eq = ab.law_attention(q, k, spec(q, k), c.kappa_design)
        spice.append(r['y']); equation.append(circuit.reference(values, a_eq))
        isum_meas.append(r['isum'])
        isum_law.append(ab.law_currents(q, k, c.s_ratio, c.kappa_design).sum(1) * c.tail_current)
        targets.append([ex['target'], -ex['target']])
        cv = ex['ctx'][:, 0].mean(); ctxavg.append([cv, -cv])
        a_est = attention_from_scores(r['isum'])
        uniform.append(run(f'uniform-{i:02}', values, controls=np.full(3, a_est.mean()))['y'])
    spice, uniform, equation, targets, ctxavg = map(np.array, (spice, uniform, equation, targets, ctxavg))
    isum_meas, isum_law = np.array(isum_meas), np.array(isum_law)

    # cell-law sweep: key 0 dimension 0 swept; dimension 1 matched (delta 0)
    sweep = []
    for i, dlt in enumerate(SWEEP):
        q = BASE_KEYS[0] + np.array([dlt, 0.])
        sweep.append(run(f'sweep-{i:02}', BASE_VALUES, q, BASE_KEYS)['isum'][0])
    sweep = np.array(sweep)
    fit = fit_cell(SWEEP, sweep, c.tail_current)

    # D=8 summation additivity (standalone netlists)
    rng = np.random.default_rng(c.seed + 1)
    add_err, sums, parts_total = [], [], []
    for n in range(SUM_CASES):
        deltas = rng.uniform(-.25, .25, SUM_DIMS)
        guard()
        total, vdiode, t = ab.run_sum_test(artifacts / f'sum8-{n}-all', exe, deltas=deltas, active=range(SUM_DIMS),
                                           model=c.model, s_ratio=c.s_ratio, tail_current=c.tail_current)
        stats.append({'seconds': t})
        parts = 0.
        for i in range(SUM_DIMS):
            p, _, t = ab.run_sum_test(artifacts / f'sum8-{n}-cell{i}', exe, deltas=deltas, active=[i],
                                      model=c.model, s_ratio=c.s_ratio, tail_current=c.tail_current)
            stats.append({'seconds': t}); parts += p
        sums.append(total); parts_total.append(parts)
        add_err.append(abs(total - parts) / parts)

    ex0 = examples[0]
    v0 = ex0['ctx'] @ vw
    values0 = np.stack([v0, -v0], 1)
    q0 = c.v_cm + c.scale * ex0['xq'] @ W.T
    k0 = c.v_cm + c.scale * ex0['ctx'] @ W.T
    transient = [run(f'transient-{i}', values0, q0, k0, transient=True, initial=s,
                     transient_seconds=c.transient_seconds, transient_abstol=TRANSIENT_ABSTOL)
                 for i, s in enumerate([.2, .4])]
    settling = max(float(np.max(abs(t['y'] - spice[0]))) for t in transient)

    mse = float(np.mean((spice - targets) ** 2))
    ctx_mse = float(np.mean((ctxavg - targets) ** 2))
    bank = [s for s in stats if 'min_node_voltage' in s]
    metrics = {
        'retrieval_mse': mse, 'context_average_mse': ctx_mse, 'context_over_retrieval': ctx_mse / mse,
        'uniform_mse': float(np.mean((uniform - targets) ** 2)),
        'equation_mse': float(np.mean((equation - targets) ** 2)),
        'spice_vs_equation_rmse': float(np.sqrt(np.mean((spice - equation) ** 2))),
        'summed_score_vs_law_rms_over_tail': float(np.sqrt(np.mean((isum_meas - isum_law) ** 2)) / c.tail_current),
        'resistance_ohm': R,
        'cell_fit': fit,
        'd8_additivity_max_rel_error': float(max(add_err)),
        'd8_summed_current_range_A': [float(min(sums)), float(max(sums))],
        'transient_vs_dc_max_abs_error': settling,
        'min_node_voltage': min(s['min_node_voltage'] for s in bank),
        'max_node_voltage': max(s['max_node_voltage'] for s in bank),
        'min_anchor_vds': min(s['min_anchor_vds'] for s in bank),
        'max_supply_power_W': max(s['supply_power_W'] for s in bank),
        'spice_seconds': sum(s['seconds'] for s in stats), 'spice_calls': len(stats)}
    checks = {'retrieval': mse <= 1e-3 and ctx_mse >= 5 * mse,
              'settling': settling < 1e-3,
              'voltage_range': metrics['min_node_voltage'] > 0 and metrics['max_node_voltage'] < circuit.VDD,
              'anchor_compliance': metrics['min_anchor_vds'] > 4 * circuit.UT}
    summary = {
        'model': c.model, 'variant': c.variant, 'passed': all(checks.values()), 'checks': checks,
        'metrics': metrics, 'counts': transient[0]['counts'], 'total_mos': sum(transient[0]['counts'].values()),
        'weights_source': c.weights_source, 'weights_source_sha256': c.weights_source_sha256,
        'sweep': {'delta_V': SWEEP.tolist(), 'summed_current_A': sweep.tolist()},
        'd8': {'summed_A': sums, 'sum_of_cells_A': parts_total},
        'claim_boundary': ('One query, three 2-D keys, two value features. Explicit MOS Delbrueck anti-bump cells, '
                           'summing PMOS mirrors and (softmax variant) PMOS Elfadel-Wyatt normalizer or (star variant) '
                           'NMOS sink mirrors drive the 51-MOS pseudoconductance bank. Ideal: query/key voltages, '
                           'reference currents, score resistor R, V_A buffer (VCVS), V_low, value currents, SPICE '
                           'probes; software decoding. Weights trained in ideal equations. Generic BSIM3 or ideal laws; '
                           'no mismatch, noise, PDK, on-chip training, silicon or energy-advantage claim.')}
    np.savez_compressed(artifacts / 'retrieval.npz', spice=spice, uniform=uniform, equation=equation,
                        targets=targets, context_average=ctxavg, isum=isum_meas, isum_law=isum_law)
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
