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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['exp-001', 'exp-002'])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    configs = {'exp-001': exp001, 'exp-002': exp002}[args.experiment]()
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
