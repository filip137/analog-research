"""Derive the drift token-budget configs from the verified fault token-budget DRN6/DRN12 configs.

Only the declared factors change: a fault-free deployment, the untreated hold arm, the
AIHWKit CMO x4 log-time drift with the clock continuing through writes, the ratio3
mapping, and the 24-h evaluation grid. Pool, targets, development cohort, learner, LR,
gate, batch order and schedule are inherited unchanged.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'examples/opt_cmo_drn/fault_token_budget'
TARGET = ROOT / 'examples/opt_cmo_drn/drift_token_budget'
GRID = [0.0, 60.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0, 57600.0, 86400.0]
DRIFT = {
    'law': 'aihwkit_cmo_log_independent.v1', 'multiplier': 4.0,
    'mean_us': -0.08900206, 'spread_slope_us': 0.04201137, 'spread_offset_us': 0.41183342,
    'drift_min_us': 5.0, 'drift_max_us': 150.0, 'write_clock': 'continues', 'draw_seed': 391103,
    'mapping_scale': 3.0, 'update_seconds': 1.5, 'evaluation_seconds': GRID,
    'projection_steps': [300, 600, 1200], 'projection_seconds': 86400.0,
}
ARMS = [{'label': 'A0', 'method': 'hold'}, {'label': 'L1', 'method': 'noisy', 'learning_rate': 3e-4}]


FULL_WALL_MINUTES = {'drn6': 110, 'drn12': 240}  # DRN6 on qos_gpu-dev; DRN12 (~2x per update) on qos_gpu-t3


def build(model, kind):
    doc = json.loads((SOURCE / kind / (model + '-n8192' + ('-canary' if kind == 'canary' else '') + '.json')).read_text())
    doc['deployment'] = {'assignment_seed': 271001, 'fault_kind': 'none', 'fault_rate': 0.0}
    doc['fault_tuning']['arms'] = ARMS
    drift = dict(DRIFT)
    if kind == 'canary':
        drift['projection_steps'] = [16]
        doc['training']['max_wall_seconds'] = 25 * 60 - 120
    else:
        doc['training']['max_wall_seconds'] = FULL_WALL_MINUTES[model] * 60 - 120
    doc['drift_tuning'] = drift
    from ebl.schema.opt_cmo_drn import parse_config
    parse_config(doc)
    return doc


def main():
    for model in FULL_WALL_MINUTES:
        for kind in ('full', 'canary'):
            out = TARGET / kind / (model + '-ratio3-m4' + ('-canary' if kind == 'canary' else '') + '.json')
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(build(model, kind), indent=2) + '\n')
            print(out)


if __name__ == '__main__':
    main()
