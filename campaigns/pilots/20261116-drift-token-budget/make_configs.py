"""Derive the crossbar drift token-budget lifecycles from the verified fault token-budget XB6/XB12 ones.

Changed: no analog-LoRA tiles, the fault-free (0 %) case with in-job deployment (naive P&V + verify rewrite, as
the fault study), the untreated hold arm, and the AIHWKit CMO x4 log-time drift with the
clock continuing through writes. Pool, targets, development cohort, learner, LR, gate,
batch order and schedule are inherited unchanged.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'campaigns/fault-recovery-paper/lifecycles/fault-token-budget-20261116'
TARGET = Path(__file__).with_name('lifecycles')
GRID = [0.0, 60.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0, 57600.0, 86400.0]
DRIFT = {
    'law': 'aihwkit_cmo_log_independent.v1', 'multiplier': 4.0,
    'mean_us': -0.08900206, 'spread_slope_us': 0.04201137, 'spread_offset_us': 0.41183342,
    'drift_min_us': 5.0, 'drift_max_us': 150.0, 'write_clock': 'continues', 'draw_seed': 391103,
    'mapping_scale': 1.0, 'update_seconds': 1.5, 'evaluation_seconds': GRID,
    'projection_steps': [300, 600, 1200], 'projection_seconds': 86400.0,
}


def build(model, kind):
    name = f'fault-token-budget-20261116-{model}-n8192' + ('-canary' if kind == 'canary' else '')
    doc = json.loads((SOURCE / kind / (name + '.json')).read_text())
    doc['lifecycle_id'] = f'drift-token-budget-20261116-{model}-m4' + ('-canary' if kind == 'canary' else '')
    doc['defects']['cases'] = [{'array_concentration': None, 'die_concentration': None, 'kind': 'none', 'rate_ppm': 0}]
    f = doc['fault_tuning']
    f['deployment'] = None
    f['analog_lora'] = None  # no analog-LoRA arm: no adapter tiles and no pinned B scale
    f['arms'] = [{'label': 'A0', 'method': 'hold_fault_aware'},
                 {'label': 'L1', 'method': 'onchip_noisy', 'learning_rate': 1e-3}]
    drift = dict(DRIFT)
    if kind == 'canary':
        drift['projection_steps'] = [16]
    f['drift'] = drift
    doc['question'].update(
        text='One-shot drift recovery at deployment on the 2.1M-token pool under AIHWKit CMO x4 drift.',
        decision='Compare the learner with untreated drift on the 24-h grid and the 24-h projections; no retuning.')
    from ebl.schema.lifecycle import parse_lifecycle
    parse_lifecycle(doc)
    return doc


def main():
    for model in ('xb6', 'xb12'):
        for kind in ('full', 'canary'):
            doc = build(model, kind)
            out = TARGET / kind / (doc['lifecycle_id'] + '.json')
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(doc, indent=2) + '\n')
            print(out)


if __name__ == '__main__':
    main()
