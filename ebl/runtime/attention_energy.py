"""Energy per attention step for declared design rungs (#59), with the #58 accuracy gates.

Specification: campaigns/pilots/20261117-attention-energy.md.
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
from ebl.runtime.end_to_end_attention import spec_for as base_spec

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = ('duty_cycle', 'query_hold', 'ref_hold', 'gate_scorer', 't_warm', 'c_state', 'abstol')


def spec_for(c, ex, t_settle=None):
    return base_spec(c, ex, t_settle) | {k: getattr(c, k) for k in OPTIONS} | {'power_split': True}


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
                             'energy_total_J': r['energy']['total'], 'spice_seconds': sum(calls)})
        return r

    def dec(y):
        return c.readout_alpha * y + c.readout_gamma

    examples = toy_examples(np.random.default_rng(c.seed))
    rows = [(run(f'step-{i:02}', spec_for(c, ex)), ex) for i, ex in enumerate(examples)]
    y = np.array([dec(r['y_raw'][0]) for r, _ in rows])
    tgt = np.array([ex['target'] for _, ex in rows])
    ctx = np.array([ex['ctx'][:, 0].mean() for _, ex in rows])
    ylaw = np.array([dec(e2e.law(spec_for(c, ex), c.kappa_design)['y']) for _, ex in rows])
    sel = np.array([ex['selected'] for _, ex in rows])
    p = np.array([r['softmax'] for r, _ in rows])
    keys = rows[0][0]['energy'].keys()
    energy = {k: float(np.mean([r['energy'][k] for r, _ in rows])) for k in keys}
    energy['total_max'] = float(max(r['energy']['total'] for r, _ in rows))
    energy['average_power_W'] = energy['total'] / energy['step_seconds']
    ped = {'key': [], 'value': []}
    samp = {'key': [], 'value': [], 'query': []}
    for r, ex in rows:
        law = e2e.law(spec_for(c, ex), c.kappa_design)
        for name, h in r['kv'].items():
            ped['value' if name.startswith('v') else 'key'].append(h['held'] - h['column'])
            if name.startswith('v'):          # value error relative to the matched reference, if sampled
                samp['value'].append(h['held'] - law['v'][int(name[1])] - r.get('ref_offset', 0.))
            else:
                samp['key'].append(h['held'] - law['k'][int(name[1]), int(name[2])])
        if 'query_held' in r:
            samp['query'] += [r['query_held'][d] - law['q'][d] for d in range(2)]

    mse = float(np.mean((y - tgt) ** 2))
    ctx_mse = float(np.mean((ctx - tgt) ** 2))
    metrics = {'retrieval_mse': mse, 'context_average_mse': ctx_mse, 'context_over_retrieval': ctx_mse / mse,
               'worst_abs_error': float(np.max(np.abs(y - tgt))),
               'spice_vs_drawn_law_rmse_decoded': float(np.sqrt(np.mean((y - ylaw) ** 2))),
               'softmax_on_target': float(p[np.arange(len(p)), sel].mean()),
               'key_pedestal_max_V': float(np.max(np.abs(ped['key']))),
               'value_pedestal_max_V': float(np.max(np.abs(ped['value']))),
               'key_sampling_error_max_V': float(np.max(np.abs(samp['key']))),
               'value_sampling_error_max_V': float(np.max(np.abs(samp['value']))),
               'query_sampling_error_max_V': float(np.max(np.abs(samp['query']))) if samp['query'] else None,
               'energy_J': energy, 'spice_seconds': 0., 'spice_calls': 0}
    if c.model == 'ideal':
        checks = {'wiring': metrics['spice_vs_drawn_law_rmse_decoded'] <= 2e-3}
    else:
        checks = {'retrieval': mse <= 1e-3 and ctx_mse >= 5 * mse,
                  # complete sampling, so the cached state does not depend on what the capacitor held before
                  'key_sampling': metrics['key_sampling_error_max_V'] <= 5e-3,
                  'value_sampling': metrics['value_sampling_error_max_V'] <= 1e-3}
    if c.chain == 'A':
        metrics['late_drift_max'] = max(r['late_drift'] for r, _ in rows)
        metrics['min_node_voltage'] = min(r['v_node_range'][0] for r, _ in rows)
        metrics['max_node_voltage'] = max(r['v_node_range'][1] for r, _ in rows)
        metrics['min_anchor_vds'] = min(r['min_anchor_vds'] for r, _ in rows)
        checks['settling'] = metrics['late_drift_max'] < 1e-3
        checks['voltage_range'] = metrics['min_node_voltage'] > 0 and metrics['max_node_voltage'] < e2e.VDD
        checks['anchor_compliance'] = metrics['min_anchor_vds'] > 4 * e2e.UT
    if c.settle_reference_s:                  # reported: output change if read after a long hold instead
        ref = [abs(dec(run(f'settle-ref-{i}', spec_for(c, examples[i], t_settle=c.settle_reference_s))['y_raw'][0]) - y[i])
               for i in range(3)]
        metrics['settle_vs_reference_max'] = float(max(ref))
    metrics['spice_seconds'], metrics['spice_calls'] = float(sum(calls)), len(calls)
    counts = rows[0][0]['counts']
    summary = {'model': c.model, 'chain': c.chain, 'rung': c.rung, 'passed': all(checks.values()), 'checks': checks,
               'metrics': metrics, 'counts': counts,
               'total_mos': sum(v for k, v in counts.items() if k not in ('crossbar_resistors', 'hold_capacitors')),
               'options': {k: getattr(c, k) for k in OPTIONS} | {'t_write': c.t_write, 't_reset': c.t_reset,
                                                                 't_settle': c.t_settle, 'c_hold': c.c_hold, 'scale': c.scale},
               'weights_source': c.weights_source, 'weights_source_sha256': c.weights_source_sha256,
               'claim_boundary': ('Energy of one N = 3 attention step in generic BSIM3 or ideal laws: block supplies, '
                                  'token drivers (charge sourced from a 1.8 V rail) and clock drivers (charge drawn '
                                  'while driving gates high), integrated over the step. Everything is assumed '
                                  'power-gated between steps; bias generators, DACs, the readout and the next layer '
                                  'are outside the count. No mismatch, noise, junction leakage, PDK or silicon claim.')}
    np.savez_compressed(artifacts / 'retrieval.npz', y=y, targets=tgt, context_average=ctx, y_law=ylaw, softmax=p,
                        energy_total=np.array([r['energy']['total'] for r, _ in rows]))
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
            raise RuntimeError('Per-configuration storage allowance exceeded')
        flat = {k: v for k, v in result['metrics'].items() if k != 'energy_J'}
        flat |= {f'energy_{k}': v for k, v in result['metrics']['energy_J'].items()}
        store.complete(metrics={'passed': result['passed'], **flat}, artifacts=artifacts)
        print(result['rung'], result['checks'], f"E={result['metrics']['energy_J']['total']:.3e} J", flush=True)
        return 0 if result['passed'] else 1
    except BaseException as err:
        store.fail(err)
        raise
