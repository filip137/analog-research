"""Frozen selection controls, recovery-data-preparation issue #27.

Does importance-32 lose to random-32 (exp-003) because it covers fewer positions
or because of its score? Same instrument, targets, parents, update budget, drift
ages and LR as exp-004; random-32 is the exact-reproduction control.
"""
import copy
import json
from pathlib import Path

import make_screen_lifecycles as screen
from make_sample_count_lifecycles import GATE
from make_token_backward_lifecycles import arm, unit

MODES = ('random:32', 'random_fixed:32', 'importance_sampled:32', 'gradient_sampled:32')
RATE = 3e-3


def documents():
    arms = [{'label': 'A0', 'method': 'none'}, copy.deepcopy(GATE), *[arm(m, RATE) for m in MODES]]
    specs = [(screen.TUNE_REALIZATION, 'select-canary', screen.TUNE_COHORT),
             (screen.TUNE_REALIZATION, 'select-dev', screen.TUNE_COHORT),
             *[(seed, 'select-eval', None) for seed in screen.REALIZATIONS]]
    docs = []
    for seed, part, cohort in specs:
        doc = unit(seed, part, copy.deepcopy(arms), cohort)
        doc['question']['text'] = ('Is importance-32 worse than random-32 because of coverage or because of its score? '
                                   'Fixed-per-event random, and score-sampled positions redrawn each presentation.')
        doc['question']['decision'] = 'recovery-data-preparation issue #27: selection controls at 32 backward positions'
        if part == 'select-canary':
            doc['drift_regimes']['canary'] = True
            doc['drift_regimes']['adaptation']['steps'] = 3
            for a in doc['drift_regimes']['arms']:
                if 'events' in a:
                    a['events'] = [900.]
        docs.append(doc)
    return docs


def main():
    root = Path(__file__).resolve().parent/'lifecycles'
    docs = documents()
    for doc in docs:
        (root/(doc['lifecycle_id']+'.json')).write_text(json.dumps(doc, indent=2)+'\n')
    print(json.dumps({'written': len(docs), 'ids': [d['lifecycle_id'] for d in docs]}))


if __name__ == '__main__':
    main()
