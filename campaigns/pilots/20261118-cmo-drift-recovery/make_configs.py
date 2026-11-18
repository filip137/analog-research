"""Derive the crossbar CMO drift-recovery lifecycles (on-chip-learning cmo-drift-recovery exp-001).

Source: the drift token-budget (exp-005) XB6/XB12 lifecycles of the same model and kind, which
pin the pool, targets, development cohort, source, in-job deployment (naive P&V + 5 sigma
verify rewrite, assignment 271001), schedule (Adam, batch 8, order seed 20261104, 2,400
updates, cosine to zero), the 2 sigma mid-range gate and the drift draw seed.

Changed (user, 18 November 2026): write clock ``reset_absorb``, two-sided mapping (unit gain,
margin 0), multiplier M in {1, 4, 10}, and the arms below. One full lifecycle per model and
M bundles every arm on one in-job deployment; CT-nodrift (independent of M) runs in the x4
lifecycle only. One canary per model at x4 (32 updates, every arm).
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'campaigns/pilots/20261116-drift-token-budget/lifecycles'
TARGET = Path(__file__).with_name('lifecycles')
MULTIPLIERS = (1.0, 4.0, 10.0)
MAPPING = {'kind': 'two_sided', 'gain': {'rule': 'unit'}, 'margin_us': 0.0}
# Learning rates reused untuned (exp-001 arms table).
LR = {'xb6': {'CAL': 1e-3, 'LN': 3e-3, 'CT': 1e-3, 'L1': 1e-3, 'LoRA': 3e-4},
      'xb12': {'CAL': 1e-3, 'LN': 3e-3, 'CT': 3e-3, 'L1': 1e-3, 'LoRA': 1e-3}}


def arms(model, multiplier):
    lr = LR[model]
    out = [{'label': 'A0', 'method': 'hold_fault_aware'},
           {'label': 'A0-twin', 'method': 'hold_fault_aware', 'no_drift': True},
           {'label': 'CAL', 'method': 'calibration', 'learning_rate': lr['CAL']},
           {'label': 'LN', 'method': 'layernorm', 'learning_rate': lr['LN']},
           {'label': 'CT', 'method': 'child_tuning_d', 'learning_rate': lr['CT'], 'reserve_p': 0.1},
           {'label': 'L1', 'method': 'onchip_noisy', 'learning_rate': lr['L1']},
           {'label': 'LoRA', 'method': 'digital_lora', 'learning_rate': lr['LoRA'], 'rank': 16}]
    if multiplier == 4.0:
        out.append({'label': 'CT-nodrift', 'method': 'child_tuning_d', 'learning_rate': lr['CT'],
                    'reserve_p': 0.1, 'no_drift': True})
    return out


def name(model, multiplier, kind):
    return f'cmo-drift-recovery-20261118-{model}-m{multiplier:g}' + ('-canary' if kind == 'canary' else '')


def build(model, multiplier, kind):
    doc = json.loads((SOURCE / kind / (f'drift-token-budget-20261116-{model}-m4'
                                       + ('-canary' if kind == 'canary' else '') + '.json')).read_text())
    doc['lifecycle_id'] = name(model, multiplier, kind)
    f = doc['fault_tuning']
    f['arms'] = arms(model, multiplier)
    d = f['drift']
    d.update(multiplier=multiplier, write_clock='reset_absorb', mapping=dict(MAPPING))
    doc['question'].update(
        text=f'One-shot recovery methods at release under AIHWKit CMO x{multiplier:g} drift, two-sided mapping, '
             'reset-absorb writes (cmo-drift-recovery exp-001).',
        decision='Primary: 24-h trapezoid-mean dev teacher KL of each arm against A0 within this model x M cell; '
                 'untuned LRs, pre-declared LR fallback only.')
    from ebl.schema.lifecycle import parse_lifecycle
    parse_lifecycle(doc)
    return doc


def cases():
    for model in ('xb6', 'xb12'):
        for multiplier in MULTIPLIERS:
            yield model, multiplier, 'full'
        yield model, 4.0, 'canary'


def main():
    for model, multiplier, kind in cases():
        doc = build(model, multiplier, kind)
        out = TARGET / kind / (doc['lifecycle_id'] + '.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + '\n')
        print(out)


if __name__ == '__main__':
    main()
