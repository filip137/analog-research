"""Write the configs of the DRN fault-paper LR screens at 0, 1 and 5 % faults, plus the 3 % extensions.

Usage: python campaigns/pilots/20261114-fault-paper-rates/make_configs.py [--check]

Writes ``examples/opt_cmo_drn/fault_paper_rates/*.json``. Everything is the 20261113 screen
(same base deploy stage, settings, arms and selection) except the fault case and the grids:
- deploy-r00/r01/r05: tuning map 271001 at 0 % (fault-free), 1 % and 5 % random stuck devices;
- per rate, five packed jobs: holds + noisy, noiseless, Child-D, LoRA r16, LoRA r64;
- grids include the 3 % edge extensions: noiseless 3e-5 … 1e-2, LoRA 1e-4 … 1e-2;
- at 3 % (the existing deployment): noiseless 3e-5, LoRA r16/r64 at 1e-4 and 3e-4, and
  Child-D at p = 0.3 at its tuned rate 1e-3.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'examples' / 'opt_cmo_drn' / 'fault_paper_rates'
_spec = importlib.util.spec_from_file_location(
    'screen_3pct', ROOT / 'campaigns/pilots/20261113-fault-paper-lr-tuning/make_configs.py')
screen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(screen)
arm, SETTINGS = screen.arm, screen.SETTINGS

RATES = {'r00': ('none', 0.0), 'r01': ('random', 0.01), 'r05': ('random', 0.05)}
WRITE_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
NOISELESS_RATES = (3e-5,) + WRITE_RATES
LORA_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
JOBS = {
    'a-holds-noisy': [arm('naive_hold'), arm('hold')] + [arm('noisy', r) for r in WRITE_RATES],
    'b-noiseless': [arm('noiseless', r) for r in NOISELESS_RATES],
    'c-child-d': [arm('child_d', r) for r in WRITE_RATES],
    'd-lora16': [arm('lora', r, 16) for r in LORA_RATES],
    'e-lora64': [arm('lora', r, 64) for r in LORA_RATES],
}
EXTENSIONS_3PCT = {
    'x03-edges': [arm('noiseless', 3e-5)] + [arm('lora', r, k) for k in (16, 64) for r in (1e-4, 3e-4)],
    'x03-child30': [{**arm('child_d', 1e-3), 'label': 'child_d30-lr0.001'}],
}


def tuning_template(deploy):
    tuning = copy.deepcopy(deploy)
    tuning['stage'] = 'fault_tuning'
    tuning['recovery'].update(arm='learning', update_rule='bounded_thresholded_adam', write_noise_factor=1.0,
                              threshold_sigma=2.0, max_write_attempts=None, target_bounds='gmin_gmax',
                              conductance_floor='none', development_every=0)
    tuning['training']['max_wall_seconds'] = 6900
    tuning['cpu_threads'] = 10
    return tuning


def documents():
    base = json.loads(screen.BASE.read_text())
    out = {}
    for tag, (kind, rate) in RATES.items():
        deploy = copy.deepcopy(base)
        deploy['deployment'] = {'assignment_seed': 271001, 'fault_kind': kind, 'fault_rate': rate}
        deploy['device'] = 'cuda:0'
        out[f'deploy-{tag}.json'] = deploy
        tuning = tuning_template(deploy)
        for job, arms in JOBS.items():
            doc = copy.deepcopy(tuning)
            doc['fault_tuning'] = {**SETTINGS, 'arms': copy.deepcopy(arms)}
            out[f'{tag}-{job}.json'] = doc
    deploy3 = copy.deepcopy(base)
    deploy3['deployment'] = dict(screen.DEPLOYMENT)
    deploy3['device'] = 'cuda:0'
    tuning3 = tuning_template(deploy3)
    for job, arms in EXTENSIONS_3PCT.items():
        doc = copy.deepcopy(tuning3)
        doc['fault_tuning'] = {**SETTINGS, 'arms': copy.deepcopy(arms),
                               **({'child_fraction': 0.3} if job == 'x03-child30' else {})}
        out[f'{job}.json'] = doc
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, doc in documents().items():
        text = json.dumps(doc, indent=2) + '\n'
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
    if stale:
        raise SystemExit(f'Stale configs: {stale}')
    print(json.dumps({'configs': sorted(documents())}))


if __name__ == '__main__':
    main()
