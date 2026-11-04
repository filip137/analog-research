"""Write the drift-regimes exp-002 crossbar lifecycles: R0 probe, LR tuning and the screen.

Same base, deployment and realizations as the exp-001 qualification
(make_lifecycles.py). Arms follow the exp-002 note, restricted to the arms
implemented so far (A6, A10, A11, O1, O2 come later; later arms stay paired
because deployment and drift draws are counter-keyed). Maintenance events are
at 15 min, 1 h, 4 h and 16 h after release (A7: x2 spacing, 7 events). Rewrite
arms use the regime's kappa*_N from kappa_family.json.

Digital arms adapt on a seeded 1,024-sequence cohort of the WikiText-2 training
split (300 Adam steps, batch 8); A1 fits on its first 16 sequences.

One lifecycle is one unit = (regime point, realization, part):
- part ``w``: every arm without a learning rate (A0, A1, A1o, A8, A9, N sweep), run 1;
- part ``d``: the LayerNorm and LoRA arms, with the tuned rates, run 2;
- ``tune``: R0 on development realization 271004, scored on WikiText-2
  validation sequences 64-127 (disjoint from the 64-sequence evaluation cohort), run 1.
Every unit redeploys its realization identically, so arms in different units are paired.

Usage: PYTHONPATH=. python campaigns/drift-regimes/make_screen_lifecycles.py [layernorm_lr lora_lr]
Without rates only the probe, tuning and part-w files are written.
"""
import json
import sys
from pathlib import Path

from make_lifecycles import HERE, POINTS, lifecycle

EVENTS = [900., 3600., 14400., 57600.]
DENSE = [900., 1800., 3600., 7200., 14400., 28800., 57600.]
ADAPTATION = {'corpus': 'wikitext-2-raw-v1',
              'corpus_sha256': '33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01',
              'split': 'train', 'sequences': 1024,
              'cohort_sha256': 'c919329dbe0d545d7614daf3d8f790881e944cd80ff53fff88fc19d9697f7762',
              'seed': 20261104, 'probe_sequences': 16, 'steps': 300, 'batch': 8}
LORA_LR = 3e-3             # drift-mitigation exp-004
PROBE_LAYERNORM_LR = 1e-3  # placeholder for the cost probe only
REALIZATIONS = (271001, 271002, 271003)
TUNE_REALIZATION = 271004
TUNE_COHORT = (64, '99c12a626280f6be079b24ae72cacb1900a07f6917ef530cc4c2d1bf1ad7014c')
TUNE_LAYERNORM = (3e-4, 1e-3, 3e-3, 1e-2)
TUNE_LORA = (1e-3, 3e-3, 1e-2)
N_SWEEP_POINTS = {'R0': ('A8', 'A9-W1', 'A9-W2'), 'rho1.0': ('A9-W1',)}


def kappa(point, attempts):
    if attempts == 1:
        return None
    table = json.loads((HERE / 'kappa_family.json').read_text())['points']
    return round(table[point]['by_attempts'][str(attempts)]['kappa_star'], 4)


def calibration(label, events):
    return {'label': label, 'method': 'calibration', 'events': events}


def rewrite(point, name, attempts):
    fraction = {'A8': 1., 'A9-W1': .01, 'A9-W2': .1}[name]
    return {'label': f'{name}-N{attempts}', 'method': 'rewrite', 'events': EVENTS, 'fraction': fraction,
            'attempts': attempts, 'threshold': kappa(point, attempts)}


def layernorm(label, events, lr):
    return {'label': label, 'method': 'layernorm', 'events': events, 'learning_rate': lr}


def lora(label, events, lr, rank=16):
    return {'label': label, 'method': 'lora', 'events': events, 'rank': rank, 'learning_rate': lr}


def group(point):
    if point == 'R0':
        return 'anchor'
    if point.startswith('rho'):
        return 'rho'
    if point.startswith('sigma'):
        return 'sigma'
    return 'f'                              # f0.3, f0.7, f1.0 and the corner


def write_arms(point):
    """Part w: arms that need no learning rate (exp-002 arm table, implemented arms)."""
    g = group(point)
    arms = [{'label': 'A0', 'method': 'none'}, calibration('A1', EVENTS)]
    if g in ('anchor', 'rho'):
        arms.append(calibration('A1o', EVENTS[:1]))
    names = {'anchor': ('A8', 'A9-W1', 'A9-W2'), 'rho': ('A9-W1',),
             'f': ('A8', 'A9-W1', 'A9-W2'), 'sigma': ('A8', 'A9-W1', 'A9-W2')}[g]
    for name in names:
        levels = (1, 2, 4, 8) if name in N_SWEEP_POINTS.get(point, ()) else (4,)
        arms.extend(rewrite(point, name, n) for n in levels)
    return arms


def digital_arms(point, ln, lr):
    g = group(point)
    if g == 'anchor':
        return [layernorm('A2', EVENTS[:1], ln), layernorm('A3', EVENTS, ln), lora('A4', EVENTS[:1], lr),
                lora('A5', EVENTS, lr), *[lora(f'A5-r{r}', EVENTS, lr, r) for r in (1, 4, 64)],
                lora('A7', DENSE, lr)]
    if g == 'rho':
        return [layernorm('A2', EVENTS[:1], ln), layernorm('A3', EVENTS, ln), lora('A4', EVENTS[:1], lr),
                lora('A5', EVENTS, lr), lora('A7', DENSE, lr)]
    if g == 'sigma':
        return [layernorm('A3', EVENTS, ln), lora('A5', EVENTS, lr)]
    return [lora('A5', EVENTS, lr)]


def slug(point):
    return point.replace('.', 'p').replace('_', '-').lower()


def unit(point, realization, part, arms, *, question):
    doc = lifecycle(point, kappa(point, 4))
    doc['lifecycle_id'] = f'opt125m-drift-regimes-screen-{slug(point)}-{realization}-{part}'
    doc['question'] = {'text': question,
                       'decision': 'drift-regimes exp-002 development regime screen (crossbar); development only.',
                       'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    spec = doc['drift_regimes']
    del spec['arm']
    spec['assignment_seeds'] = [realization]
    spec['arms'] = arms
    spec['adaptation'] = ADAPTATION if any(a['method'] != 'none' and a['method'] != 'rewrite' for a in arms) else None
    return doc


def tuning():
    point = 'R0'
    arms = [{'label': 'A0', 'method': 'none'},
            *[layernorm(f'A3-lr{lr:g}', EVENTS, lr) for lr in TUNE_LAYERNORM],
            *[lora(f'A5-lr{lr:g}', EVENTS, lr) for lr in TUNE_LORA]]
    docs = []
    for part, chosen in (('tune-ln', arms[:5]), ('tune-lora', arms[:1] + arms[5:])):
        doc = unit(point, TUNE_REALIZATION, part, chosen, question=(
            'Which LayerNorm and LoRA learning rates minimize the 24-h mean teacher KL at R0 '
            '(development realization, held-out validation sequences 64-127)?'))
        doc['drift_regimes']['evaluation_start'] = TUNE_COHORT[0]
        doc['drift_regimes']['cohort_sha256'] = TUNE_COHORT[1]
        docs.append(doc)
    return docs


def canaries():
    """Two merged units on 2 sequences: exercises the shared prepare/devices plan, every arm method,
    the N = 1 rewrite path and a non-anchor law. Gate for runs 1 and 2."""
    docs = []
    for point, arms in (('R0', [{'label': 'A0', 'method': 'none'}, calibration('A1', EVENTS),
                                rewrite('R0', 'A8', 1), rewrite('R0', 'A9-W1', 4),
                                layernorm('A3', EVENTS[:1], PROBE_LAYERNORM_LR),
                                lora('A5', EVENTS[:2], LORA_LR)]),
                        ('rho1.0', [{'label': 'A0', 'method': 'none'}, rewrite('rho1.0', 'A9-W1', 8)])):
        doc = lifecycle(point, kappa(point, 4), canary=True)
        doc['lifecycle_id'] = f'opt125m-drift-regimes-screen-canary-{slug(point)}'
        doc['question'] = {'text': f'Does the merged screen plan run every arm method at {point}?',
                           'decision': 'drift-regimes exp-002 launch gate; no method comparison.',
                           'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
        spec = doc['drift_regimes']
        del spec['arm']
        spec['arms'] = arms
        spec['adaptation'] = ADAPTATION if point == 'R0' else None
        docs.append(doc)
    return docs


def probe():
    point = 'R0'
    doc = lifecycle(point, kappa(point, 4), canary=True)
    doc['lifecycle_id'] = 'opt125m-drift-regimes-screen-probe-r0'
    doc['question'] = {
        'text': 'What does each exp-002 arm class cost per trajectory on the six-MLP std3 crossbar at R0?',
        'decision': 'drift-regimes exp-001 cost probe that sizes exp-002; development only, no method comparison.',
        'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    doc['drift_regimes']['arms'] = [{'label': 'A0', 'method': 'none'}, calibration('A1', EVENTS),
                                    calibration('A1o', EVENTS[:1]),
                                    layernorm('A2', EVENTS[:1], PROBE_LAYERNORM_LR),
                                    layernorm('A3', EVENTS, PROBE_LAYERNORM_LR), lora('A4', EVENTS[:1], LORA_LR),
                                    lora('A5', EVENTS, LORA_LR), lora('A7', DENSE, LORA_LR),
                                    rewrite(point, 'A8', 4), rewrite(point, 'A9-W1', 4), rewrite(point, 'A9-W2', 4)]
    doc['drift_regimes']['adaptation'] = ADAPTATION
    del doc['drift_regimes']['arm']
    return doc


def main():
    rates = [float(x) for x in sys.argv[1:3]]
    out = HERE / 'lifecycles'
    docs = [probe(), *canaries(), *tuning()]
    for point in POINTS:
        for r in REALIZATIONS:
            docs.append(unit(point, r, 'w', write_arms(point), question=(
                f'Untreated, calibration and rewrite arms at drift regime {point} (realization {r}): '
                '24-h mean teacher KL and write cost.')))
            if rates:
                docs.append(unit(point, r, 'd', digital_arms(point, *rates), question=(
                    f'LayerNorm and LoRA arms at drift regime {point} (realization {r}): 24-h mean teacher KL.')))
    for doc in docs:
        (out / f"{doc['lifecycle_id']}.json").write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'written': len(docs), 'arms': sum(len(d['drift_regimes']['arms']) for d in docs)}))


if __name__ == '__main__':
    main()
