"""Write the drift-regimes exp-002 lifecycles (crossbar): the R0 arm cost probe so far.

Same base, deployment and realizations as the exp-001 qualification
(make_lifecycles.py). Arms follow the exp-002 note; the maintenance schedule is
15 min, 1 h, 4 h and 16 h after release (A7: x2 spacing, 7 events). Rewrite
arms use the regime's kappa*_N from kappa_family.json.

Digital arms adapt on a seeded 1,024-sequence cohort of the WikiText-2 training
split (300 Adam steps, batch 8); A1 fits on its first 16 sequences.

The probe lifecycle measures cost per arm class. Its LayerNorm learning rate
is a placeholder, not a tuned value: the probe is not a method result.

Usage: PYTHONPATH=. python campaigns/drift-regimes/make_screen_lifecycles.py
"""
import json
from pathlib import Path

from make_lifecycles import HERE, POINTS, lifecycle

EVENTS = [900., 3600., 14400., 57600.]
DENSE = [900., 1800., 3600., 7200., 14400., 28800., 57600.]
ADAPTATION = {'corpus': 'wikitext-2-raw-v1',
              'corpus_sha256': '33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01',
              'split': 'train', 'sequences': 1024,
              'cohort_sha256': 'c919329dbe0d545d7614daf3d8f790881e944cd80ff53fff88fc19d9697f7762',
              'seed': 20261104, 'probe_sequences': 16, 'steps': 300, 'batch': 8}
LORA_LR = 3e-3           # drift-mitigation exp-004
PROBE_LAYERNORM_LR = 1e-3  # placeholder for the cost probe only


def kappa(point, attempts):
    if attempts == 1:
        return None
    table = json.loads((HERE / 'kappa_family.json').read_text())['points']
    return round(table[point]['by_attempts'][str(attempts)]['kappa_star'], 4)


def arms(point, layernorm_lr, attempts=4):
    t = kappa(point, attempts)
    return [
        {'label': 'A0', 'method': 'none'},
        {'label': 'A1', 'method': 'calibration', 'events': EVENTS},
        {'label': 'A1o', 'method': 'calibration', 'events': EVENTS[:1]},
        {'label': 'A2', 'method': 'layernorm', 'events': EVENTS[:1], 'learning_rate': layernorm_lr},
        {'label': 'A3', 'method': 'layernorm', 'events': EVENTS, 'learning_rate': layernorm_lr},
        {'label': 'A4', 'method': 'lora', 'events': EVENTS[:1], 'rank': 16, 'learning_rate': LORA_LR},
        {'label': 'A5', 'method': 'lora', 'events': EVENTS, 'rank': 16, 'learning_rate': LORA_LR},
        {'label': 'A7', 'method': 'lora', 'events': DENSE, 'rank': 16, 'learning_rate': LORA_LR},
        {'label': f'A8-N{attempts}', 'method': 'rewrite', 'events': EVENTS, 'fraction': 1.,
         'attempts': attempts, 'threshold': t},
        {'label': f'A9-W1-N{attempts}', 'method': 'rewrite', 'events': EVENTS, 'fraction': .01,
         'attempts': attempts, 'threshold': t},
        {'label': f'A9-W2-N{attempts}', 'method': 'rewrite', 'events': EVENTS, 'fraction': .1,
         'attempts': attempts, 'threshold': t},
    ]


def probe():
    point = 'R0'
    doc = lifecycle(point, kappa(point, 4), canary=True)
    doc['lifecycle_id'] = 'opt125m-drift-regimes-screen-probe-r0'
    doc['question'] = {
        'text': 'What does each exp-002 arm class cost per trajectory on the six-MLP std3 crossbar at R0?',
        'decision': 'drift-regimes exp-001 cost probe that sizes exp-002; development only, no method comparison.',
        'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    doc['drift_regimes']['arms'] = arms(point, PROBE_LAYERNORM_LR)
    doc['drift_regimes']['adaptation'] = ADAPTATION
    del doc['drift_regimes']['arm']
    return doc


def main():
    assert set(POINTS) >= {'R0'}
    out = HERE / 'lifecycles'
    doc = probe()
    path = out / f"{doc['lifecycle_id']}.json"
    path.write_text(json.dumps(doc, indent=2) + '\n')
    print(path.relative_to(HERE.parent.parent))


if __name__ == '__main__':
    main()
