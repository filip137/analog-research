"""Declared end-to-end checks of one analog attention step (token crossbar -> KV hold ->
anti-bump scorer with capacitor integration -> Elfadel-Wyatt softmax -> bank or
translinear value weighting).

Specification: campaigns/pilots/20261117-end-to-end-attention.md.
"""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.base import ConfigError
from networks.energy_transformer.spice import executable_identity
from networks.energy_transformer import end_to_end as e2e
from ebl.runtime.crossbar_qk_attention import toy_examples

ROOT = Path(__file__).resolve().parents[2]


def spec_for(c, ex, t_settle=None):
    return dict(chain=c.chain, ctx=ex['ctx'], xq=ex['xq'], key_gp=c.key_gp, key_gm=c.key_gm,
                value_gp=c.value_gp, value_gm=c.value_gm, v_cm=c.v_cm, scale=c.scale, s_ratio=c.s_ratio,
                tail_current=c.tail_current, amplitude=c.amplitude, v_low=c.v_low, c_score=c.c_score,
                c_hold=c.c_hold, t_write=c.t_write, t_reset=c.t_reset,
                t_integrate=c.temperature_g * c.c_score * e2e.UT / (c.kappa_design * c.tail_current),
                t_settle=c.t_settle if t_settle is None else t_settle)


def characterize(c, store, exe):
    started = time.monotonic()
    artifacts = store.run_dir / 'artifacts'
    calls = []

    def run(name, spec):
        if time.monotonic() - started > c.wall_seconds:
            raise TimeoutError('Declared characterization allowance exceeded')
        r = e2e.simulate(spec, c.model, artifacts / name, exe)
        calls.append(r['seconds'])
        store.append_metric({'case': name, 'spice_calls': len(calls), 'y_raw': r['y_raw'].tolist(),
                             'spice_seconds': sum(calls)})
        if len(calls) % 12 == 0:
            print(f'{c.model}/{c.chain}: cases={len(calls)}, last={name}', flush=True)
        return r

    def dec(y):
        return c.readout_alpha * y + c.readout_gamma

    examples = toy_examples(np.random.default_rng(c.seed))
    rows = []
    for i, ex in enumerate(examples):
        spec = spec_for(c, ex)
        r = run(f'step-{i:02}', spec)
        law = e2e.law(spec, c.kappa_design)
        rows.append({'r': r, 'law': law, 'target': ex['target'], 'ctxavg': ex['ctx'][:, 0].mean(),
                     'selected': ex['selected']})
    y0 = np.array([dec(w['r']['y_raw'][0]) for w in rows])
    ydiff = np.array([dec((w['r']['y_raw'][0] - w['r']['y_raw'][1]) / 2) for w in rows])
    ylaw = np.array([dec(w['law']['y']) for w in rows])
    tgt = np.array([w['target'] for w in rows])
    ctx = np.array([w['ctxavg'] for w in rows])
    p_sp = np.array([w['r']['softmax'] for w in rows])
    p_law = np.array([w['law']['p'] for w in rows])
    sel = np.array([w['selected'] for w in rows])
    ar = np.arange(len(rows))
    isum_sp = np.array([w['r']['isum_integrating'] for w in rows])
    isum_law = np.array([w['law']['isum'] for w in rows])

    kv = {'key_pedestal_V': [], 'value_pedestal_V': [], 'key_droop_V_per_s': [], 'value_droop_V_per_s': [],
          'key_vs_formula_V': [], 'value_vs_formula_V': []}
    for w in rows:
        hs = w['r']['hold_seconds']
        for j in range(3):
            for d in range(2):
                h = w['r']['kv'][f'k{j}{d}']
                kv['key_pedestal_V'].append(h['held'] - h['column'])
                kv['key_droop_V_per_s'].append((h['end'] - h['held']) / hs)
                kv['key_vs_formula_V'].append(h['column'] - w['law']['k'][j, d])
            h = w['r']['kv'][f'v{j}']
            kv['value_pedestal_V'].append(h['held'] - h['column'])
            kv['value_droop_V_per_s'].append((h['end'] - h['held']) / hs)
            kv['value_vs_formula_V'].append(h['column'] - w['law']['v'][j])
    kv_stats = {k: {'mean': float(np.mean(v)), 'max_abs': float(np.max(np.abs(v)))} for k, v in kv.items()}

    sweep = []
    for h in c.hold_sweep:
        r = run(f'hold-{h:g}s', spec_for(c, examples[0], t_settle=h))
        sweep.append({'t_settle_s': h, 'hold_seconds': r['hold_seconds'], 'y_decoded': float(dec(r['y_raw'][0])),
                      'change_vs_base': float(dec(r['y_raw'][0]) - y0[0]),
                      'max_key_drift_V': max(abs(r['kv'][f'k{j}{d}']['end'] - r['kv'][f'k{j}{d}']['held'])
                                             for j in range(3) for d in range(2)),
                      'max_value_drift_V': max(abs(r['kv'][f'v{j}']['end'] - r['kv'][f'v{j}']['held']) for j in range(3))})

    mse = float(np.mean((y0 - tgt) ** 2))
    ctx_mse = float(np.mean((ctx - tgt) ** 2))
    metrics = {
        'retrieval_mse': mse, 'context_average_mse': ctx_mse, 'context_over_retrieval': ctx_mse / mse,
        'differential_readout_mse': float(np.mean((ydiff - tgt) ** 2)),
        'law_mse': float(np.mean((ylaw - tgt) ** 2)),
        'spice_vs_law_rmse_decoded': float(np.sqrt(np.mean((y0 - ylaw) ** 2))),
        'worst_abs_error': float(np.max(np.abs(y0 - tgt))),
        'softmax_on_target_spice': float(p_sp[ar, sel].mean()), 'softmax_on_target_law': float(p_law[ar, sel].mean()),
        'softmax_rms_vs_law': float(np.sqrt(np.mean((p_sp - p_law) ** 2))),
        'summed_score_vs_law_rms_over_tail': float(np.sqrt(np.mean((isum_sp - isum_law) ** 2)) / c.tail_current),
        'kv': kv_stats,
        't_integrate_s': spec_for(c, examples[0])['t_integrate'],
        'max_supply_power_W_end': max(w['r']['supply_power_W_end'] for w in rows),
        'spice_seconds': float(sum(calls)), 'spice_calls': len(calls)}
    # The ideal law has kappa = 2/3 while the weights were trained at kappa = 0.7911, so the
    # ideal model is a wiring check against its own law; BSIM3 is gated on retrieval.
    if c.model == 'ideal':
        checks = {'wiring': metrics['spice_vs_law_rmse_decoded'] <= 2e-3}
    else:
        checks = {'retrieval': mse <= 1e-3 and ctx_mse >= 5 * mse}
    if c.chain == 'A':
        metrics['late_drift_max'] = max(w['r']['late_drift'] for w in rows)
        metrics['min_node_voltage'] = min(w['r']['v_node_range'][0] for w in rows)
        metrics['max_node_voltage'] = max(w['r']['v_node_range'][1] for w in rows)
        metrics['min_anchor_vds'] = min(w['r']['min_anchor_vds'] for w in rows)
        checks['settling'] = metrics['late_drift_max'] < 1e-3
        checks['voltage_range'] = metrics['min_node_voltage'] > 0 and metrics['max_node_voltage'] < e2e.VDD
        checks['anchor_compliance'] = metrics['min_anchor_vds'] > 4 * e2e.UT
    counts = rows[0]['r']['counts']
    summary = {
        'model': c.model, 'chain': c.chain, 'passed': all(checks.values()), 'checks': checks, 'metrics': metrics,
        'counts': counts, 'total_mos': sum(v for k, v in counts.items() if k not in ('crossbar_resistors', 'hold_capacitors')),
        'hold_sweep': sweep, 'weights_source': c.weights_source, 'weights_source_sha256': c.weights_source_sha256,
        'claim_boundary': ('One attention step: three context tokens written in sequence through an explicit resistor '
                           'crossbar into MOS sample-and-hold capacitors, one query, three 2-D keys, a scalar value '
                           '(features +v and -v). Explicit MOS anti-bump cells, summing mirrors, switched-capacitor '
                           'integration, PMOS Elfadel-Wyatt softmax, and either the 51-MOS bank with PMOS diff-pair value '
                           'inputs (A) or NMOS diff-pair value weighting (B). Ideal: token sources, reference currents, '
                           'V_low, V_cm reference, clock waveforms, output compliance probes, ideal-model switches; '
                           'software readout affine (trained next-stage stand-in). Weights trained in ideal equations. '
                           'Generic BSIM3 or ideal laws; no mismatch, noise, junction leakage model, PDK, on-chip '
                           'training, silicon or energy claim.')}
    np.savez_compressed(artifacts / 'retrieval.npz', y_decoded=y0, y_differential=ydiff, y_law=ylaw, targets=tgt,
                        context_average=ctx, softmax=p_sp, softmax_law=p_law, isum=isum_sp, isum_law=isum_law,
                        selected=sel)
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
        if sum(a.size_bytes for a in artifacts) > 500_000_000:
            raise RuntimeError('Per-model storage allowance exceeded')
        store.complete(metrics={'passed': result['passed'], **{k: v for k, v in result['metrics'].items() if k != 'kv'}},
                       artifacts=artifacts)
        print(result['checks'], flush=True)
        return 0 if result['passed'] else 1
    except BaseException as err:
        store.fail(err)
        raise
