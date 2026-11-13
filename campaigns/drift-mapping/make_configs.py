"""Generate the drn-drift-mapping configs from the exp-003 R0 screen configs.

Shared record: on-chip-learning/campaigns/drn-drift-mapping/. Every config is exp-003's
``screen-R0-<realization>-a.json`` with the arms reduced to A0, A1 and A0-twin and only
the declared law (and, from exp-002, mapping) fields changed. Usage:

    python campaigns/drift-mapping/make_configs.py exp-001 [--check]
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'drift-regimes' / 'configs' / 'exp-003'
SEEDS = (271001, 271002, 271003)
KEEP = ('A0', 'A1', 'A0-twin')

# exp-001: one law component removed per condition (regime label, law overrides).
EXP001 = {
    'R0-nofloor': {'drift_min': 0.0},
    'R0-mean': {'spread_us': 0.0},
    'R0-spread': {'mean_us': 0.0},
}


def offset_form(delta, k):
    """9 + delta + c (G - 9), c = k (79.2 - delta) / 79.2, as {'scale', 'offset_us'}."""
    c = k * (88.2 - 9.0 - delta) / (88.2 - 9.0)
    return {'scale': c, 'offset_us': 9.0 + delta - 9.0 * c}


# exp-002: deployment remappings G' = min(offset + scale G, G_max) under R0.
EXP002 = {
    'base': {'scale': 1.0, 'offset_us': 0.0},
    'margin11': offset_form(11.0, 1),
    'ratio2': {'scale': 2.0, 'offset_us': 0.0},
    'ratio3': {'scale': 3.0, 'offset_us': 0.0},
    'range4': offset_form(0.0, 4),
}
EARLY = [60.0, 900.0, 3600.0, 14400.0, 57600.0]
EXP002_ARMS = [{'label': 'A1e', 'method': 'calibration', 'events': EARLY},
               {'label': 'A1e-twin', 'method': 'calibration', 'events': EARLY, 'twin': True}]


def base(realization):
    config = json.loads((BASE / f'screen-R0-{realization}-a.json').read_text())
    settings = config['drift_regimes']
    settings['arms'] = [a for a in settings['arms'] if a['label'] in KEEP]
    if [a['label'] for a in settings['arms']] != list(KEEP):
        raise ValueError('exp-003 base config lacks A0, A1 or A0-twin.')
    return config


def exp001():
    configs = {}
    for realization in SEEDS:
        for regime, overrides in EXP001.items():
            config = base(realization)
            settings = config['drift_regimes']
            settings['regime'] = regime
            for key, value in overrides.items():
                if key not in settings['law']:
                    raise KeyError(key)
                settings['law'][key] = value
            configs[f'{regime}-{realization}'] = config
    configs['replay-R0-271001'] = base(271001)   # must reproduce exp-003 A0/A1/A0-twin
    return configs


def exp002():
    configs = {}
    for realization in SEEDS:
        for label, mapping in EXP002.items():
            config = base(realization)
            settings = config['drift_regimes']
            settings['regime'] = f'R0-map-{label}'
            settings['mapping'] = dict(mapping)
            settings['arms'] += copy.deepcopy(EXP002_ARMS)
            configs[f'{label}-{realization}'] = config
    return configs


# exp-003: the measured 28 October 25-h table (R0-G, drift-regimes/drift-family.md), as in the
# crossbar exp-004: (lower_us, mean_us, spread_us) at a 1-h reference.
MEASURED = [[0, .59, 2.35], [10, .02, 3.35], [20, -.43, 5.54], [30, -.29, 7.28], [40, -4.51, 9.93],
            [50, -3.04, 11.05], [65, -2.45, 8.49], [80, -2.52, 9.56], [100, -1.03, 6.66],
            [115, -.68, 6.28], [130, -.87, 4.83], [145, -1.01, 4.59], [160, -.88, 3.74]]
EXP003_MAPPINGS = ('base', 'ratio3', 'range4')


def exp003():
    configs = {}
    for realization in SEEDS:
        for label in EXP003_MAPPINGS:
            config = base(realization)
            settings = config['drift_regimes']
            settings['regime'] = f'R0-G-map-{label}'
            settings['law'].update(mean_us=None, spread_us=None, coefficient_bins=copy.deepcopy(MEASURED))
            settings['mapping'] = dict(EXP002[label])
            settings['arms'] += copy.deepcopy(EXP002_ARMS)
            configs[f'{label}-{realization}'] = config
    return configs


# exp-004: write-noise sensitivity, cmo_scale 10 under R0 and R0-G, base vs ratio3; A1 dropped.
EXP004_SCALE = 10.0


def exp004():
    configs = {}
    for realization in SEEDS:
        for law in ('R0', 'R0-G'):
            for label in ('base', 'ratio3'):
                config = base(realization)
                settings = config['drift_regimes']
                settings['regime'] = f'{law}-cp10-map-{label}'
                settings['law']['cmo_scale'] = EXP004_SCALE
                if law == 'R0-G':
                    settings['law'].update(mean_us=None, spread_us=None, coefficient_bins=copy.deepcopy(MEASURED))
                settings['mapping'] = dict(EXP002[label])
                settings['arms'] = [a for a in settings['arms'] if a['label'] != 'A1'] + copy.deepcopy(EXP002_ARMS)
                configs[f'{law}-{label}-{realization}'] = config
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['exp-001', 'exp-002', 'exp-003', 'exp-004'])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    configs = {'exp-001': exp001, 'exp-002': exp002, 'exp-003': exp003, 'exp-004': exp004}[args.experiment]()
    out = ROOT / 'configs' / args.experiment
    out.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, config in sorted(configs.items()):
        text = json.dumps(config, indent=1) + '\n'
        path = out / f'{name}.json'
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
    if stale:
        raise SystemExit(f'Stale configs: {stale}')
    print(f'{len(configs)} configs in {out}')


if __name__ == '__main__':
    main()
