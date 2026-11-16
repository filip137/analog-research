"""Declared retrieval, crossbar-linearity and settling checks of the trained
crossbar query/key projection driving the bump scorer and pseudoconductance bank.

Specification: campaigns/pilots/20261116-crossbar-qk-attention.md.
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
from networks.energy_transformer import bump_scorer as bump
from networks.energy_transformer import crossbar_qk as xbar

ROOT = Path(__file__).resolve().parents[2]
CASES = 36
SLOTS = 8
VMAX = .4
IDEAL_KAPPA = 2 / 3
TRANSIENT_ABSTOL = 1e-18     # #55 revision 1: 1e-20 A aborts BSIM3 transients on fA branches


def toy_examples(rng):
    """Fresh keyed-retrieval examples (keyed_retrieval.make_data distribution, N=3)."""
    out = []
    for i in range(CASES):
        phi = rng.uniform(0, 2 * math.pi)
        slots = rng.permutation(SLOTS)[:3]
        ang = phi + slots * 2 * math.pi / SLOTS
        vals = rng.uniform(-VMAX, VMAX, 3)
        ctx = np.stack([vals, np.cos(ang), np.sin(ang)], 1)
        sel = i % 3
        out.append({'ctx': ctx, 'xq': np.array([0., ctx[sel, 1], ctx[sel, 2]]),
                    'target': vals[sel], 'selected': sel})
    return out


def characterize(c, store, exe):
    started = time.monotonic()
    artifacts = store.run_dir / 'artifacts'
    gp, gm = np.array(c.conductance_pos), np.array(c.conductance_neg)
    vw = np.array(c.value_weights)
    kappa = IDEAL_KAPPA if c.model == 'ideal' else c.reference_kappa
    amp = 1. if c.model == 'ideal' else c.reference_amplitude_ratio
    stats = []

    def run(name, values, ex=None, controls=None, **kw):
        if time.monotonic() - started > c.wall_seconds:
            raise TimeoutError('Declared characterization allowance exceeded')
        scorer = None
        if ex is not None:
            scorer = {'tail_current': c.tail_current,
                      'crossbar': {'q': ex['xq'], 'k': ex['ctx'], 'gp': gp, 'gm': gm,
                                   'v_cm': c.v_cm, 'scale': c.scale}}
        r = circuit.simulate(values, controls, artifacts / name, exe, model=c.model, scorer=scorer, **kw)
        extra = {}
        if scorer is not None:
            r['a'] = bump.measured(r['port'], circuit.IREF)
            r['rows'] = xbar.rows(r['port'])
            extra['min_tail_voltage'] = bump.min_tail_voltage(r['port'])
        stats.append(r['metrics'] | extra | {'seconds': r['seconds']})
        store.append_metric({'case': name, 'spice_calls': len(stats), 'output': r['y'].tolist(),
                             'spice_seconds': sum(s['seconds'] for s in stats)})
        if len(stats) % 12 == 0:
            print(f'{c.model}: cases={len(stats)}, last={name}', flush=True)
        return r

    examples = toy_examples(np.random.default_rng(c.seed))
    spice, uniform, equation, targets, ctxavg = [], [], [], [], []
    row_err, row_v, a_meas, a_eq, diss = [], [], [], [], []
    for i, ex in enumerate(examples):
        v = ex['ctx'] @ vw
        values = np.stack([v, -v], 1)
        r = run(f'retrieve-{i:02}', values, ex)
        q_f = xbar.formula(ex['xq'], gp, gm, c.v_cm, c.scale)
        k_f = xbar.formula(ex['ctx'], gp, gm, c.v_cm, c.scale)
        q_s, k_s = r['rows']
        row_err.append(max(np.max(abs(q_s - q_f)), np.max(abs(k_s - k_f))))
        row_v += list(q_s) + list(k_s.ravel())
        a_ref = amp * bump.law(q_f, k_f, c.tail_current, circuit.IREF, kappa)
        spice.append(r['y']); a_meas.append(r['a']); a_eq.append(a_ref)
        equation.append(circuit.reference(values, a_ref))
        targets.append([ex['target'], -ex['target']])
        cv = ex['ctx'][:, 0].mean()
        ctxavg.append([cv, -cv])
        diss.append(xbar.dissipation(ex['xq'], ex['ctx'], q_s, k_s, gp, gm, c.v_cm, c.scale))
        uniform.append(run(f'uniform-{i:02}', values, controls=np.full(3, r['a'].mean()))['y'])
    spice, uniform, equation, targets, ctxavg = map(np.array, (spice, uniform, equation, targets, ctxavg))
    a_meas, a_eq = np.array(a_meas), np.array(a_eq)

    ex0 = examples[0]
    v0 = ex0['ctx'] @ vw
    values0 = np.stack([v0, -v0], 1)
    transient = [run(f'transient-{i}', values0, ex0, transient=True, initial=s,
                     transient_seconds=c.transient_seconds, transient_abstol=TRANSIENT_ABSTOL)
                 for i, s in enumerate([.2, .4])]
    settling = max(float(np.max(abs(t['y'] - spice[0]))) for t in transient)

    mse = float(np.mean((spice - targets) ** 2))
    ctx_mse = float(np.mean((ctxavg - targets) ** 2))
    uni_mse = float(np.mean((uniform - targets) ** 2))
    eq_mse = float(np.mean((equation - targets) ** 2))
    ar = np.arange(CASES)
    sel = np.array([e['selected'] for e in examples])
    p_meas = (a_meas / (1 + a_meas)) / (circuit.H0 + (a_meas / (1 + a_meas)).sum(1, keepdims=True))
    row_v = np.array(row_v)
    metrics = {
        'retrieval_mse': mse, 'context_average_mse': ctx_mse, 'context_over_retrieval': ctx_mse / mse,
        'uniform_mse': uni_mse, 'equation_mse': eq_mse,
        'spice_vs_equation_rmse': float(np.sqrt(np.mean((spice - equation) ** 2))),
        'weight_on_target_measured_currents': float(p_meas[ar, sel].mean()),
        'selected_current_mean': float(a_meas[ar, sel].mean()),
        'selected_current_equation_mean': float(a_eq[ar, sel].mean()),
        'crossbar_row_max_abs_error_V': float(max(row_err)),
        'row_voltage_range_V': [float(row_v.min()), float(row_v.max())],
        'row_voltage_fraction_outside_0.8_1.3V': float(((row_v < .8) | (row_v > 1.3)).mean()),
        'crossbar_dissipation_W_mean': float(np.mean(diss)),
        'max_bank_scorer_supply_power_W': max(s['supply_power_W'] for s in stats),
        'transient_vs_dc_max_abs_error': settling,
        'min_node_voltage': min(s['min_node_voltage'] for s in stats),
        'max_node_voltage': max(s['max_node_voltage'] for s in stats),
        'min_anchor_vds': min(s['min_anchor_vds'] for s in stats),
        'min_scorer_tail_voltage': min(s['min_tail_voltage'] for s in stats if 'min_tail_voltage' in s),
        'max_read_mirror_ratio_error': max(s['mirror_ratio_error'] for s in stats),
        'spice_seconds': sum(s['seconds'] for s in stats), 'spice_calls': len(stats)}
    checks = {'retrieval': mse <= 1e-3 and ctx_mse >= 5 * mse,
              'crossbar_linearity': metrics['crossbar_row_max_abs_error_V'] < 1e-3,
              'settling': settling < 1e-3,
              'voltage_range': metrics['min_node_voltage'] > 0 and metrics['max_node_voltage'] < circuit.VDD,
              'anchor_compliance': metrics['min_anchor_vds'] > 4 * circuit.UT}
    summary = {
        'model': c.model, 'reference_kappa_used': kappa, 'reference_amplitude_used': amp,
        'weights_source': c.weights_source, 'weights_source_sha256': c.weights_source_sha256,
        'passed': all(checks.values()), 'checks': checks, 'metrics': metrics,
        'counts': transient[0]['counts'], 'total_mos': sum(transient[0]['counts'].values()),
        'resistors': 48,
        'claim_boundary': ('Trained (ideal-equation) shared crossbar projection programmed as ideal resistors in '
                           'the CMO conductance range, one copy per token, driving the explicit MOS bump scorer '
                           'and 51-MOS pseudoconductance bank for one query, three keys, two value features. '
                           'Ideal token-input and reference sources, ideal value currents from the learned value '
                           'projection, SPICE probes, software decoding. Generic BSIM3 or ideal laws; no device '
                           'programming error, drift, noise, mismatch, wire resistance, on-chip training, PDK, '
                           'silicon or energy-advantage claim.')}
    np.savez_compressed(artifacts / 'retrieval.npz', spice=spice, uniform=uniform, equation=equation,
                        targets=targets, context_average=ctxavg, currents=a_meas, currents_equation=a_eq,
                        selected=sel, ctx=np.array([e['ctx'] for e in examples]),
                        xq=np.array([e['xq'] for e in examples]))
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
