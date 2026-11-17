"""Size ladder (#60): nominal #59 gates plus mismatch/kT/C Monte Carlo yield, with gate area
and capacitance inventory. Specification: campaigns/pilots/20261117-attention-size.md."""
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.base import ConfigError
from networks.energy_transformer.spice import executable_identity
from networks.energy_transformer import end_to_end as e2e
from ebl.runtime.crossbar_qk_attention import toy_examples
from ebl.runtime.attention_energy import spec_for as energy_spec

ROOT = Path(__file__).resolve().parents[2]
WORKERS = 8


def spec_for(c, ex, t_settle=None):
    return energy_spec(c, ex, t_settle) | {'geometry': {k: list(v) for k, v in c.geometry}}


def _sim(args):
    spec, model, directory, exe = args
    try:
        r = e2e.simulate(spec, model, directory, exe)
        return {'ok': True, 'y0': float(r['y_raw'][0]), 'y1': float(r['y_raw'][1]), 'r': r, 'seconds': r['seconds']}
    except Exception as err:
        return {'ok': False, 'error': str(err)[-300:], 'seconds': 0.}


def characterize(c, store, exe):
    started = time.monotonic()
    art = store.run_dir / 'artifacts'
    test = toy_examples(np.random.default_rng(c.seed))
    cal = toy_examples(np.random.default_rng(c.seed + 99))[:c.cal_examples]
    dec = lambda y: c.readout_alpha * y + c.readout_gamma
    pool = ProcessPoolExecutor(WORKERS)
    seconds = [0.]

    def batch(tag, specs):
        if time.monotonic() - started > c.wall_seconds:
            raise TimeoutError('Declared characterization allowance exceeded')
        jobs = [(sp, c.model, art / tag / f'{i:03}', exe) for i, sp in enumerate(specs)]
        out = list(pool.map(_sim, jobs))
        seconds[0] += sum(o['seconds'] for o in out)
        for o in out:
            o.pop('r', None) if tag != 'nominal' else None
        store.append_metric({'case': tag, 'ok': sum(o['ok'] for o in out), 'n': len(out), 'spice_seconds': seconds[0]})
        return out

    # nominal circuit: #59 gates
    nom = batch('nominal', [spec_for(c, ex) for ex in test])
    if not all(o['ok'] for o in nom):
        raise RuntimeError('Nominal simulation failed: ' + next(o['error'] for o in nom if not o['ok']))
    y = np.array([dec(o['y0']) for o in nom]); tgt = np.array([ex['target'] for ex in test])
    ctx = np.array([ex['ctx'][:, 0].mean() for ex in test])
    mse, ctx_mse = float(np.mean((y - tgt) ** 2)), float(np.mean((ctx - tgt) ** 2))
    samp_k, samp_v, settle_k, ped_k = [], [], [], []
    for o, ex in zip(nom, test):
        r, law = o['r'], e2e.law(spec_for(c, ex), c.kappa_design)
        for name, h in r['kv'].items():
            if name.startswith('v'):
                samp_v.append(h['held'] - law['v'][int(name[1])] - r.get('ref_offset', 0.))
            else:
                samp_k.append(h['held'] - law['k'][int(name[1]), int(name[2])])
                settle_k.append(h['pre'] - h['column'])      # incomplete charging (history dependent)
                ped_k.append(h['held'] - h['pre'])           # switch pedestal (deterministic, shared with the held query)
    energy = {k: float(np.mean([o['r']['energy'][k] for o in nom])) for k in nom[0]['r']['energy']}
    net = e2e.netlist(spec_for(c, test[0]), c.model)[0]
    inv = e2e.inventory(net)
    metrics = {'retrieval_mse': mse, 'context_over_retrieval': ctx_mse / mse,
               'key_sampling_error_max_V': float(np.max(np.abs(samp_k))), 'value_sampling_error_max_V': float(np.max(np.abs(samp_v))),
               'key_settling_error_max_V': float(np.max(np.abs(settle_k))), 'key_pedestal_max_V': float(np.max(np.abs(ped_k))),
               'energy_J': energy, 'inventory': inv,
               'area_um2_at_2fF_per_um2': inv['gate_area_total_um2'] + inv['capacitance_F'] / 2e-15}
    checks = {'retrieval': mse <= 1e-3 and ctx_mse >= 5 * mse, 'key_settling': metrics['key_settling_error_max_V'] <= 5e-3,
              'value_sampling': metrics['value_sampling_error_max_V'] <= 1e-3}
    if c.chain == 'A':
        metrics['late_drift_max'] = max(o['r']['late_drift'] for o in nom)
        metrics['min_anchor_vds'] = min(o['r']['min_anchor_vds'] for o in nom)
        checks['settling'] = metrics['late_drift_max'] < 1e-3
        checks['anchor_compliance'] = metrics['min_anchor_vds'] > 4 * e2e.UT

    def monte_carlo(tag, avt, n):
        rows = []
        for k in range(n):
            mm = {'mismatch': {'avt': avt, 'abeta': c.abeta, 'seed': c.mc_seed + k}, 'ktc_seed': c.mc_seed + 50000 + k}
            out = batch(f'{tag}-{k:02}', [spec_for(c, ex) | mm for ex in test] + [spec_for(c, ex) | mm for ex in cal])
            t, cl = out[:len(test)], out[len(test):]
            ok_t = [i for i, o in enumerate(t) if o['ok']]
            ok_c = [i for i, o in enumerate(cl) if o['ok']]
            y0 = np.array([t[i]['y0'] for i in ok_t]); y1 = np.array([t[i]['y1'] for i in ok_t]); tt = tgt[ok_t]
            raw = float(np.mean((dec(y0) - tt) ** 2)); dif = float(np.mean((dec((y0 - y1) / 2) - tt) ** 2))
            A = np.stack([[cl[i]['y0'] for i in ok_c], np.ones(len(ok_c))], 1)
            a, g = np.linalg.lstsq(A, np.array([cal[i]['target'] for i in ok_c]), rcond=None)[0]
            calm = float(np.mean((a * y0 + g - tt) ** 2)); cctx = float(np.mean((ctx[ok_t] - tt) ** 2))
            rows.append({'instance': k, 'raw_mse': raw, 'differential_mse': dif, 'calibrated_mse': calm,
                         'calibration': [float(a), float(g)], 'failed_sims': len(t) + len(cl) - len(ok_t) - len(ok_c),
                         'passes_calibrated': calm <= 1e-3 and cctx >= 5 * calm, 'passes_raw': raw <= 1e-3 and cctx >= 5 * raw})
        f = lambda key: float(np.mean([r[key] for r in rows])) if rows else None
        return {'instances': rows, 'yield_calibrated': f('passes_calibrated'), 'yield_raw': f('passes_raw'),
                'calibrated_mse_median': float(np.median([r['calibrated_mse'] for r in rows])) if rows else None,
                'calibrated_mse_worst': float(np.max([r['calibrated_mse'] for r in rows])) if rows else None,
                'raw_mse_median': float(np.median([r['raw_mse'] for r in rows])) if rows else None,
                'failed_sims': int(sum(r['failed_sims'] for r in rows))}

    mc = monte_carlo('mc', c.avt, c.mc_instances)
    sens = monte_carlo('mc2x', c.avt_sensitivity, c.sensitivity_instances)
    pool.shutdown()
    checks['mismatch_yield'] = mc['yield_calibrated'] is not None and mc['yield_calibrated'] >= c.yield_target
    metrics |= {'monte_carlo': mc, 'monte_carlo_sensitivity': sens, 'spice_seconds': seconds[0]}
    summary = {'model': c.model, 'chain': c.chain, 'rung': c.rung, 'passed': all(checks.values()), 'checks': checks,
               'metrics': metrics, 'geometry': dict(c.geometry), 'weights_source': c.weights_source,
               'weights_source_sha256': c.weights_source_sha256,
               'claim_boundary': ('Generic BSIM3 with declared Pelgrom coefficients (A_VT, A_beta) applied as delvto/mulu0 on '
                                  'every MOS and kT/C offsets on every hold and score capacitor; per-chip two-parameter '
                                  'readout calibration (12 separate examples) stands for the trained next stage. No PDK '
                                  'mismatch data, layout parasitics, noise beyond kT/C, or silicon claim.')}
    atomic_write_json(art / 'summary.json', summary)
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
        res = characterize(c, store, exe)
        # keep the bundle small: traces of Monte Carlo instances are not retained
        import shutil
        for d in (store.run_dir / 'artifacts').glob('mc*'):
            shutil.rmtree(d)
        artifacts = [store.artifact_record(p, kind='circuit_characterization')
                     for p in sorted((store.run_dir / 'artifacts').rglob('*')) if p.is_file()]
        m = res['metrics']
        store.complete(metrics={'passed': res['passed'], 'retrieval_mse': m['retrieval_mse'],
                                'yield_calibrated': m['monte_carlo']['yield_calibrated'],
                                'yield_raw': m['monte_carlo']['yield_raw'],
                                'gate_area_um2': m['inventory']['gate_area_total_um2'],
                                'capacitance_F': m['inventory']['capacitance_F'],
                                'energy_total_J': m['energy_J']['total'], 'spice_seconds': m['spice_seconds']},
                       artifacts=artifacts)
        print(res['rung'], res['checks'], f"yield {m['monte_carlo']['yield_calibrated']}", flush=True)
        return 0 if res['passed'] else 1
    except BaseException as err:
        store.fail(err)
        raise
