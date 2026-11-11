"""Write the recovery-data-preparation exp-001 lifecycles: LoRA (A5) under R0-G with cheaper training targets.

Experiment note: on-chip-learning campaigns/recovery-data-preparation/series/001-teacher-targets/
experiments/exp-001-lora-target-encoding.md. Everything is drift-regimes exp-004's R0-G A5 arm
(make_measured_g_lifecycles.py: same base, law, realizations, cohorts, events, rank 16, adaptation
set); only the training target of the LoRA arm changes:
- ``kl``: the full teacher distribution (exp-004's A5; the arm dict is exp-004's, without a target key);
- ``text``: the next token of the WikiText adaptation text (one-hot cross-entropy);
- ``topk:8`` / ``topk:32``: the teacher's top-k log-probabilities plus one tail bin, fp16;
- ``samples:8``: 8 tokens drawn once from the teacher at each position, fixed by a seed.

Units:
- ``targets-tune-<name>``: development realization 271004, validation sequences 64-127, the rates
  {3e-3, 1e-2, 3e-2}. The kl unit also holds A0; its 3e-3 and 1e-2 arms must reproduce exp-004's
  tuning (0.630 and 0.611; A0 4.0747).
- ``targets-eval``: realizations 271001-271003 with A0, exp-004's A5 at 1e-2 (reproduction), and every
  target at its tuned rate (the kl rate too, when it differs from 1e-2).

Usage: PYTHONPATH=. python campaigns/drift-regimes/make_target_lifecycles.py [rates.json]
Without rates only the tuning units are written; rates.json maps each target name to its rate.
"""
import json
from pathlib import Path
import sys

from make_lifecycles import HERE
import make_measured_g_lifecycles as measured
import make_screen_lifecycles as screen

TARGETS = {'kl': 'kl', 'text': 'text', 'top8': 'topk:8', 'top32': 'topk:32', 'samp8': 'samples:8'}
RATES = (3e-3, 1e-2, 3e-2)
EXP004_RATE = 1e-2


def arm(name, lr):
    """exp-004's A5 for kl (no target key, so it is that arm exactly); otherwise A5 with the target."""
    if name == 'kl':
        return screen.lora(f'A5-lr{lr:g}', screen.EVENTS, lr)
    return {**screen.lora(f'T-{name}-lr{lr:g}', screen.EVENTS, lr), 'target': TARGETS[name]}


def tuning():
    docs = []
    for name in TARGETS:
        arms = ([{'label': 'A0', 'method': 'none'}] if name == 'kl' else []) + [arm(name, lr) for lr in RATES]
        question = (f'Which LoRA learning rate minimizes the 24-h mean teacher KL at R0-G with the {TARGETS[name]} '
                    'training target (development realization, held-out validation sequences 64-127)?')
        doc = measured.unit(screen.TUNE_REALIZATION, f'targets-tune-{name}', arms, question=question,
                            evaluation=screen.TUNE_COHORT)
        doc['question']['decision'] = 'recovery-data-preparation exp-001: learning rate per training target.'
        docs.append(doc)
    return docs


def evaluation(rates):
    if set(rates) != set(TARGETS) or any(r not in RATES for r in rates.values()):
        raise ValueError('Expected one tuned grid rate for every target.')
    arms = [{'label': 'A0', 'method': 'none'}, arm('kl', EXP004_RATE)]
    if rates['kl'] != EXP004_RATE:
        arms.append(arm('kl', rates['kl']))
    arms += [arm(name, rates[name]) for name in TARGETS if name != 'kl']
    docs = []
    for r in screen.REALIZATIONS:
        doc = measured.unit(r, 'targets-eval', arms, question=(
            f'How much of the full-KL LoRA recovery at R0-G survives text-only, top-k and sampled training '
            f'targets (realization {r})? 24-h mean teacher KL.'))
        doc['question']['decision'] = ('recovery-data-preparation exp-001: does recovery need the full '
                                       'teacher distribution?')
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
