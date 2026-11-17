"""SPICE inference of the trained MNIST chain A network on a few test images (#63).
Specification: campaigns/pilots/20261117-mnist-attention.md."""
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import time
import numpy as np
import torch
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.base import ConfigError
from networks.energy_transformer.spice import executable_identity
from networks.energy_transformer import mnist_spice as ms
from networks.energy_transformer.mnist_chain_a import ChainA, inputs, UT
from networks.energy_transformer.pseudo_mnist_data import load_raw
from ebl.runtime.analog_attention_mnist import labels

ROOT = Path(__file__).resolve().parents[2]


def _sim(args):
    spec, d, exe = args
    try:
        return {'ok': True, 'r': ms.simulate(spec, d, exe)}
    except Exception as err:
        return {'ok': False, 'error': str(err)[-400:]}


def characterize(c, store, exe):
    art = store.run_dir / 'artifacts'
    exp = Path(c.model_run) / 'artifacts' / 'export.json'
    if sha256_file(exp) != c.model_export_sha256:
        raise ConfigError('Model export hash mismatch')
    E = json.loads(exp.read_text())
    ck = torch.load(Path(c.model_run) / 'artifacts' / 'model.pt', map_location='cpu', weights_only=True)
    m = ChainA(E['heads'], E['d_k'], E['d_v'], s_eff=E['law']['s_eff'], kappa_cell=E['law']['kappa_cell']).double()
    m.load_state_dict(ck['model'])
    imgs, split, _ = load_raw(c.data_root)
    lab = labels(c.data_root)
    with torch.no_grad():       # digital calibration of the per-head integrator offsets on training images
        _, dc = m(inputs(imgs['train'][split['train'][:c.calibration_images]]).double(), detail=True)
        off = [1e-9 * (c.amp * float(torch.quantile(dc['f'][:, h].min(-1).values, c.offset_quantile)) + c.score_offset_extra)
               for h in range(E['heads'])]
    ids = list(range(c.first_image, c.first_image + c.n_images))
    specs, eqs = [], []
    for i in ids:
        x = inputs(imgs['test'][i:i + 1]).double()
        with torch.no_grad():
            lg, det = m(x, detail=True)
        eqs.append((int(lg.argmax()), det['p'][0].numpy(), det['xo'][0].numpy() - 1))
        specs.append(dict(tokens=x[0].numpy(), key_gp=E['key_gp_S'], key_gm=E['key_gm_S'], value_gp=E['value_gp_S'],
                          value_gm=E['value_gm_S'], query_V=E['query_V'],
                          t_integrate_heads=[g / c.amp * c.c_score * UT / (c.kappa_design * c.tail_current) for g in E['temperature_g']],
                          v_cm=c.v_cm, scale=c.scale, s_ratio=c.s_ratio, tail_current=c.tail_current, amplitude=c.amplitude,
                          v_low=c.v_low, c_score=c.c_score, c_hold=c.c_hold, t_write=c.t_write, t_reset=c.t_reset, t_warm=c.t_warm,
                          t_settle=c.t_settle, mirror_bias=c.mirror_bias, abstol=c.abstol, reltol=c.reltol, score_offset=off))
    t0 = time.monotonic()
    with ProcessPoolExecutor(c.workers) as pool:
        out = list(pool.map(_sim, [(sp, art / f'img-{i:05}', exe) for sp, i in zip(specs, ids)]))
    if time.monotonic() - t0 > c.wall_seconds:
        raise TimeoutError('Declared allowance exceeded')
    W, b = np.array(E['readout_W']), np.array(E['readout_b'])
    rows = []
    for i, o, (ceq, peq, oeq) in zip(ids, out, eqs):
        store.append_metric({'case': f'img-{i}', 'ok': o['ok']})
        if not o['ok']:
            rows.append({'image': i, 'label': int(lab['test'][i]), 'ok': False, 'error': o['error']})
            continue
        r = o['r']
        csp = int(np.argmax(W @ r['out'].ravel() + b))
        rows.append({'image': i, 'label': int(lab['test'][i]), 'ok': True, 'spice_class': csp, 'equation_class': ceq,
                     'p_max_abs_error': float(np.abs(r['p'] - peq).max()), 'out_rms_error': float(np.sqrt(np.mean((r['out'] - oeq) ** 2))),
                     'late_drift': r['late_drift'], 'supply_W_end': r['supply_W_end'], 'seconds': r['seconds']})
    ok = [r for r in rows if r['ok']]
    agree = float(np.mean([r['spice_class'] == r['equation_class'] for r in ok]))
    m_ = {'images': len(rows), 'simulated': len(ok), 'class_agreement': agree,
          'accuracy_spice': float(np.mean([r['spice_class'] == r['label'] for r in ok])),
          'accuracy_equations': float(np.mean([r['equation_class'] == r['label'] for r in ok])),
          'p_max_abs_error_median': float(np.median([r['p_max_abs_error'] for r in ok])),
          'out_rms_error_median': float(np.median([r['out_rms_error'] for r in ok])),
          'late_drift_max': float(max(r['late_drift'] for r in ok)), 'score_offsets_A': off,
          'model_test_accuracy_full': E.get('test_accuracy'), 'mos': sum(v for k, v in ms.netlist(specs[0])[2].items()
                                                                        if k not in ('crossbar_resistors', 'hold_capacitors')),
          'spice_seconds': float(sum(r['seconds'] for r in ok))}
    passed = len(ok) == len(rows) and agree >= .9
    summary = {'variant': c.variant, 'passed': passed, 'checks': {'class_agreement': passed}, 'metrics': m_, 'rows': rows,
               'claim_boundary': ('Generic BSIM3, one transient per image: 16 tokens through an explicit 65-input resistor crossbar '
                                  'into transmission-gate KV holds, 4 heads of anti-bump scorers, integrators with a programmed '
                                  'offset sink, Elfadel-Wyatt softmaxes and 16-feature pseudoconductance banks. Ideal: token, '
                                  'clock, query, bias, offset and rail sources; digital readout. Trained offline. No mismatch, '
                                  'noise, PDK or silicon claim.')}
    atomic_write_json(art / 'summary.json', summary)
    for d in art.glob('img-*'):                    # keep the netlists, logs and final values; drop nothing else needed
        pass
    return summary


def run_characterize(request):
    if any(getattr(request, k, None) is not None for k in ['weights', 'base_weights', 'resume', 'device_data', 'device_model']):
        raise ConfigError('Standalone circuit pilot')
    c = request.spec
    exe, version = executable_identity(c.spice_executable)
    store = RunStore.create(output_root=request.output_dir, experiment_id=c.experiment_id, resolved_config=asdict(c),
                            command=request.command, repo_root=ROOT, input_artifacts=[{'path': str(exe), 'sha256': sha256_file(exe)}],
                            resume_capability='unsupported')
    print(f'RUN_DIR={store.run_dir}', flush=True)
    try:
        atomic_write_json(store.run_dir / 'artifacts' / 'environment.json', {'simulator': str(exe), 'version': version})
        res = characterize(c, store, exe)
        artifacts = [store.artifact_record(p, kind='circuit_inference') for p in sorted((store.run_dir / 'artifacts').rglob('*')) if p.is_file()]
        store.complete(metrics={'passed': res['passed'], **{k: v for k, v in res['metrics'].items() if k != 'score_offsets_A'}}, artifacts=artifacts)
        print(res['checks'], json.dumps({k: res['metrics'][k] for k in ('class_agreement', 'accuracy_spice', 'accuracy_equations')}), flush=True)
        return 0 if res['passed'] else 1
    except BaseException as err:
        store.fail(err)
        raise
