"""Derive the cmo-drift-recovery exp-001 DRN configs from the verified drift token-budget configs.

Only the declared factors change (on-chip-learning cmo-drift-recovery exp-001, user 18 Nov
2026): the drift multiplier M in {1, 4, 10}, writes that reset the drift clock and absorb
the drifted read (``reset_absorb``), the fixed 1.12 uS write gate (``fixed_mid``), the arm
set (A0, A0-twin, CAL, LN, CT, L1, and CT-nodrift in the x4 job) and the projection steps.
Pool, targets, development cohort, deployment, mapping (ratio3), law coefficients, clip
window, draw seed, batch, order seed, schedule, learner clock and grid are inherited.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'examples/opt_cmo_drn/drift_token_budget'
TARGET = ROOT / 'examples/opt_cmo_drn/cmo_drift_recovery'
MULTIPLIERS = (1.0, 4.0, 10.0)
# LRs reused untuned (exp-001 arms table): CAL 1e-2 and LN 1e-2 (DRN), CT 1e-3 (DRN6, DRN12), L1 3e-4.
ARMS = [
    {'label': 'A0', 'method': 'hold'},
    {'label': 'A0-twin', 'method': 'hold'},
    {'label': 'CAL', 'method': 'calibration', 'learning_rate': 1e-2},
    {'label': 'LN', 'method': 'layernorm', 'learning_rate': 1e-2},
    {'label': 'CT', 'method': 'child_d', 'learning_rate': 1e-3},
    {'label': 'L1', 'method': 'noisy', 'learning_rate': 3e-4},
]
NODRIFT = {'label': 'CT-nodrift', 'method': 'child_d', 'learning_rate': 1e-3}
# Wall caps (minutes) on one RTX 5090, about 2x the canary-measured estimate; see the pilot note.
FULL_WALL_MINUTES = {('drn6', 4.0): 240, ('drn6', 1.0): 200, ('drn6', 10.0): 200,
                     ('drn12', 4.0): 420, ('drn12', 1.0): 360, ('drn12', 10.0): 360}
CANARY_WALL_MINUTES = {'drn6': 40, 'drn12': 60}


def name(model, multiplier, kind):
    return f'{model}-ratio3-m{multiplier:g}' + ('-canary' if kind == 'canary' else '')


def build(model, multiplier, kind):
    doc = json.loads((SOURCE / kind / (name(model, 4.0, kind) + '.json')).read_text())
    arms = [dict(a) for a in ARMS] + ([dict(NODRIFT)] if multiplier == 4.0 else [])
    doc['fault_tuning']['arms'] = arms
    drift = doc['drift_tuning']
    drift.update(multiplier=multiplier, write_clock='reset_absorb', write_gate='fixed_mid',
                 drift_off_arms=['A0-twin'] + (['CT-nodrift'] if multiplier == 4.0 else []))
    if kind == 'canary':
        drift['projection_steps'] = [16, 32]
        doc['training']['max_wall_seconds'] = CANARY_WALL_MINUTES[model] * 60 - 120
    else:
        drift['projection_steps'] = [300, 600, 1200, 2400]
        doc['training']['max_wall_seconds'] = FULL_WALL_MINUTES[model, multiplier] * 60 - 120
    from ebl.schema.opt_cmo_drn import parse_config
    parse_config(doc)
    return doc


def cases():
    for model in ('drn6', 'drn12'):
        for multiplier in MULTIPLIERS:
            yield model, multiplier, 'full'
        yield model, 4.0, 'canary'


def main():
    for model, multiplier, kind in cases():
        out = TARGET / kind / (name(model, multiplier, kind) + '.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(build(model, multiplier, kind), indent=2) + '\n')
        print(out)


if __name__ == '__main__':
    main()
