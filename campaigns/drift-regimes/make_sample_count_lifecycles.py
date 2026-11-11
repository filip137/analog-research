"""Write the recovery-data-preparation exp-002 lifecycles: LoRA (A5) at R0-G with 1, 2 or 4 teacher samples.

Experiment note: on-chip-learning campaigns/recovery-data-preparation/series/001-teacher-targets/
experiments/exp-002-sample-count.md. Everything is exp-001 (make_target_lifecycles.py); only the
count n of the samples:n target changes. samples:n uses the first n columns of the same fixed
8-sample draw: every unit also carries exp-001's samples:8 arm at 1e-2, which makes the draw 8
columns (as in exp-001) and must reproduce exp-001 bit for bit.

Units:
- ``samples-tune-n<n>``: realization 271004, validation sequences 64-127, rates {1e-3, 3e-3, 1e-2};
- ``samples-eval``: realizations 271001-271003 with A0, exp-004's A5, samples:8 and the three counts
  at their tuned rates.

Usage: PYTHONPATH=. python campaigns/drift-regimes/make_sample_count_lifecycles.py [rates.json]
rates.json maps 'n1', 'n2', 'n4' to the tuned rates.
"""
import json
from pathlib import Path
import sys

from make_lifecycles import HERE
import make_measured_g_lifecycles as measured
import make_screen_lifecycles as screen

COUNTS = (1, 2, 4)
RATES = (1e-3, 3e-3, 1e-2)
DECISION = 'recovery-data-preparation exp-002: how few teacher samples per position are enough?'


def samples(n, lr):
    return {**screen.lora(f'T-samp{n}-lr{lr:g}', screen.EVENTS, lr), 'target': f'samples:{n}'}


GATE = samples(8, 1e-2)      # exp-001's arm, unchanged: forces the 8-column draw


def tuning():
    docs = []
    for n in COUNTS:
        doc = measured.unit(screen.TUNE_REALIZATION, f'samples-tune-n{n}', [GATE] + [samples(n, lr) for lr in RATES],
                            question=(f'Which LoRA learning rate minimizes the 24-h mean teacher KL at R0-G with '
                                      f'{n} fixed teacher sample(s) per position (development realization)?'),
                            evaluation=screen.TUNE_COHORT)
        doc['question']['decision'] = DECISION
        docs.append(doc)
    return docs


def evaluation(rates):
    if set(rates) != {f'n{n}' for n in COUNTS} or any(r not in RATES for r in rates.values()):
        raise ValueError('Expected one tuned grid rate for every count.')
    arms = [{'label': 'A0', 'method': 'none'}, screen.lora('A5-lr0.01', screen.EVENTS, 1e-2), GATE,
            *[samples(n, rates[f'n{n}']) for n in COUNTS]]
    docs = []
    for r in screen.REALIZATIONS:
        doc = measured.unit(r, 'samples-eval', arms, question=(
            f'How much of the full-KL LoRA recovery at R0-G survives 1, 2 or 4 fixed teacher samples per position '
            f'(realization {r})? 24-h mean teacher KL.'))
        doc['question']['decision'] = DECISION
        docs.append(doc)
    return docs


def main():
    docs = tuning()
    if len(sys.argv) > 1:
        docs += evaluation(json.loads(Path(sys.argv[1]).read_text()))
    out = HERE / 'lifecycles'
    for doc in docs:
        (out / f"{doc['lifecycle_id']}.json").write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'written': len(docs), 'ids': [d['lifecycle_id'] for d in docs]}))


if __name__ == '__main__':
    main()
