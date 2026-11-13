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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['exp-001'])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    configs = {'exp-001': exp001}[args.experiment]()
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
