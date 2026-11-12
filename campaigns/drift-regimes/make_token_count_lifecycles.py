"""Frozen random-token count comparison, recovery-data-preparation exp-004."""
import copy
import json
from pathlib import Path

import make_screen_lifecycles as screen
from make_sample_count_lifecycles import GATE
from make_token_backward_lifecycles import arm, unit

COUNTS = (32, 64, 128)
RATE = 3e-3


def documents():
    arms = [{'label': 'A0', 'method': 'none'}, copy.deepcopy(GATE),
            arm('full', RATE), *[arm('random:'+str(k), RATE) for k in COUNTS]]
    specs = [(screen.TUNE_REALIZATION, 'count-canary', screen.TUNE_COHORT),
             (screen.TUNE_REALIZATION, 'count-dev', screen.TUNE_COHORT),
             *[(seed, 'count-eval', None) for seed in screen.REALIZATIONS]]
    docs = []
    for seed, part, cohort in specs:
        doc = unit(seed, part, copy.deepcopy(arms), cohort)
        doc['question']['text'] = 'How much recovery is retained with 32, 64 or 128 uniformly selected backward positions at context 256?'
        doc['question']['decision'] = 'recovery-data-preparation exp-004: paired recovery and saved activation cost at fixed learning rate'
        if part == 'count-canary':
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
