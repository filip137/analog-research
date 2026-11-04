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

Usage: PYTHONPATH=. python campaigns/drift-regimes/make_screen_lifecycles.py [layernorm_lr lora_lr [extend]]
Without rates only the probe, canaries, tuning and part-w files are written; ``extend`` adds the
run-3 part-x units (A6, A10, A11 = exp-003 loop, O1, O2, no-drift controls).
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


# --- run 3: the remaining crossbar arms (A6, A10, A11, O1, O2, no-drift controls) ---------------
EVAL_AGES = [0., 60., 900., 1800., 3600., 7200., 14400., 28800., 57600., 86400.]
# The first grid (0.01-0.3) was cancelled: the run-3 canary showed step 0.1 raising KL to 1.5.
TUNE_A11_STEPS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
A11_BATCH = 16
TIER = {'W1': .01, 'W2': .1}


def rounds(attempts):
    """Closed-loop rounds per event: P&V of every round must end before the next evaluation (>= 900 s)."""
    return 4 if attempts <= 4 else 2


def learning(point, label, tier, attempts, step, *, ideal=False, twin=False):
    arm = {'label': label, 'method': 'learning', 'events': EVENTS, 'fraction': TIER[tier], 'attempts': attempts,
           'threshold': kappa(point, attempts), 'step': step, 'rounds': rounds(attempts),
           'batch_sequences': A11_BATCH, 'ideal': ideal}
    if twin:
        arm['twin'] = True
    return arm


def newton(point, label, tier, attempts, *, ideal=False, twin=False, score='fisher'):
    """A11: the exp-003 closed loop (reset of whole pairs to clean targets), with the unsigned
    fisher_net score h (e^2 - v). The signed newton_net score overshot in the run-3 canary
    (canary3, job 1957589) and stays as one R0 diagnostic. No learning rate."""
    arm = {'label': label, 'method': 'learning_newton', 'events': EVENTS, 'fraction': TIER[tier],
           'attempts': attempts, 'threshold': kappa(point, attempts), 'rounds': rounds(attempts),
           'batch_sequences': A11_BATCH, 'ideal': ideal}
    if score != 'newton':
        arm['score'] = score
    if twin:
        arm['twin'] = True
    return arm


def precompensated(point, tier, attempts):
    return {'label': f'A10-{tier}-N{attempts}', 'method': 'precompensated', 'events': EVENTS,
            'fraction': TIER[tier], 'attempts': attempts, 'threshold': kappa(point, attempts)}


def extension_arms(point, lr, step=None):
    """Part x: the remaining exp-002 arms for this point's group (N sweep at R0 and rho* 1.0)."""
    g = group(point)
    o1 = {'label': 'O1', 'method': 'lora_oracle', 'events': EVAL_AGES, 'rank': 16, 'learning_rate': lr}
    a6 = {'label': 'A6', 'method': 'lora_rescaled', 'events': EVENTS[:1], 'reads': EVENTS[1:], 'rank': 16,
          'learning_rate': lr}
    o2 = {'label': 'O2-W2-N4', 'method': 'rewrite_oracle', 'events': EVENTS, 'fraction': .1, 'attempts': 4,
          'threshold': kappa(point, 4)}
    if g == 'anchor':
        sweep = (1, 2, 4, 8)
        return [{'label': 'A0-twin', 'method': 'none', 'twin': True},
                {**lora('A5-twin', EVENTS, lr), 'twin': True},
                {'label': 'A4c', 'method': 'lora_clean', 'events': EVENTS[:1], 'rank': 16, 'learning_rate': lr},
                a6, o1, o2,
                *[precompensated(point, t, n) for t in ('W1', 'W2') for n in sweep],
                *[newton(point, f'A11-{t}-N{n}', t, n) for t in ('W1', 'W2') for n in sweep],
                newton(point, 'A11-W2-twin', 'W2', 1, twin=True),
                newton(point, 'A11-W2-ideal', 'W2', 1, ideal=True),
                newton(point, 'A11n-W2-N4', 'W2', 4, score='newton')]       # newton_net diagnostic
    if g == 'rho':
        levels = (1, 2, 4, 8) if point == 'rho1.0' else (4,)
        arms = [a6, o1, *[newton(point, f'A11-W1-N{n}', 'W1', n) for n in levels]]
        if point == 'rho0.5':   # off-anchor retuning check of the LoRA rate (exp-002 tuning rule)
            arms += [lora(f'A5-lr{x:g}', EVENTS, x) for x in (3e-3, 1e-2)]
        return arms
    if g == 'sigma':
        return [newton(point, f'A11-{t}-N4', t, 4) for t in ('W1', 'W2')]
    return [*[precompensated(point, t, 4) for t in ('W1', 'W2')],
            *[newton(point, f'A11-{t}-N4', t, 4) for t in ('W1', 'W2')], o2]


def canary3():
    """Run-3 gate for the exp-003 A11 loop on 2 sequences: N = 8 (2 rounds), twin and ideal variants."""
    point = 'R0'
    doc = lifecycle(point, kappa(point, 4), canary=True)
    doc['lifecycle_id'] = 'opt125m-drift-regimes-screen-canary3-r0'
    doc['question'] = {'text': 'Does the exp-003 A11 loop run at full scale and improve on A0?',
                       'decision': 'drift-regimes exp-002 run-3 launch gate; no method comparison.',
                       'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    spec = doc['drift_regimes']
    del spec['arm']
    spec['arms'] = [{'label': 'A0', 'method': 'none'}, newton(point, 'A11-W1-N4', 'W1', 4, score='newton'),
                    newton(point, 'A11-W2-N8', 'W2', 8, score='newton'),
                    newton(point, 'A11-W2-twin', 'W2', 1, twin=True, score='newton'),
                    newton(point, 'A11-W2-ideal', 'W2', 1, ideal=True, score='newton')]
    spec['adaptation'] = ADAPTATION
    return doc


def canary4():
    """Run-3 gate for the fisher_net A11 on 2 sequences."""
    point = 'R0'
    doc = lifecycle(point, kappa(point, 4), canary=True)
    doc['lifecycle_id'] = 'opt125m-drift-regimes-screen-canary4-r0'
    doc['question'] = {'text': 'Does the fisher_net A11 loop run at full scale without the newton_net overshoot?',
                       'decision': 'drift-regimes exp-002 run-3 launch gate; no method comparison.',
                       'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    spec = doc['drift_regimes']
    del spec['arm']
    spec['arms'] = [{'label': 'A0', 'method': 'none'}, newton(point, 'A11-W1-N4', 'W1', 4),
                    newton(point, 'A11-W2-N8', 'W2', 8), newton(point, 'A11-W2-twin', 'W2', 1, twin=True),
                    newton(point, 'A11-W2-ideal', 'W2', 1, ideal=True)]
    spec['adaptation'] = ADAPTATION
    return doc


def tuning_a11():
    point = 'R0'
    arms = [{'label': 'A0', 'method': 'none'},
            *[learning(point, f'A11-{t}-s{s:g}', t, 4, s) for t in ('W1', 'W2') for s in TUNE_A11_STEPS]]
    doc = unit(point, TUNE_REALIZATION, 'tune-a11', arms, question=(
        'Which analog-learning step minimizes the 24-h mean teacher KL at R0 '
        '(development realization, held-out validation sequences 64-127)?'))
    doc['drift_regimes']['evaluation_start'] = TUNE_COHORT[0]
    doc['drift_regimes']['cohort_sha256'] = TUNE_COHORT[1]
    return doc


def canaries_v2():
    """Run-3 gate on 2 sequences: one of every new method, plus run 1's A0, A8 and A9 canary arms,
    whose final conductance hashes must equal the run-1 canary's."""
    point = 'R0'
    doc = lifecycle(point, kappa(point, 4), canary=True)
    doc['lifecycle_id'] = 'opt125m-drift-regimes-screen-canary2-r0'
    doc['question'] = {'text': 'Does the run-3 plan run every new arm method, unchanged for the old ones?',
                       'decision': 'drift-regimes exp-002 run-3 launch gate; no method comparison.',
                       'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    spec = doc['drift_regimes']
    del spec['arm']
    spec['arms'] = [{'label': 'A0', 'method': 'none'}, rewrite(point, 'A8', 1), rewrite(point, 'A9-W1', 4),
                    {'label': 'A0-twin', 'method': 'none', 'twin': True},
                    {'label': 'O1', 'method': 'lora_oracle', 'events': [0., 900.], 'rank': 16,
                     'learning_rate': LORA_LR},
                    {'label': 'A6', 'method': 'lora_rescaled', 'events': EVENTS[:1], 'reads': EVENTS[1:2],
                     'rank': 16, 'learning_rate': LORA_LR},
                    precompensated(point, 'W1', 4),
                    {'label': 'O2', 'method': 'rewrite_oracle', 'events': EVENTS[:2], 'fraction': .01,
                     'attempts': 4, 'threshold': kappa(point, 4)},
                    {**learning(point, 'A11', 'W1', 8, .1), 'events': EVENTS[:2]},
                    {**learning(point, 'A11-twin', 'W1', 1, .1, twin=True), 'events': EVENTS[:1]},
                    {**learning(point, 'A11-ideal', 'W1', 1, .1, ideal=True), 'events': EVENTS[:1]}]
    spec['adaptation'] = ADAPTATION
    return doc


def main():
    rates = [float(x) for x in sys.argv[1:3]]
    extend = len(sys.argv) > 3 and sys.argv[3] == 'extend'
    out = HERE / 'lifecycles'
    docs = [probe(), *canaries(), *tuning(), canaries_v2(), tuning_a11(), canary3(), canary4()]
    for point in POINTS:
        for r in REALIZATIONS:
            docs.append(unit(point, r, 'w', write_arms(point), question=(
                f'Untreated, calibration and rewrite arms at drift regime {point} (realization {r}): '
                '24-h mean teacher KL and write cost.')))
            if rates:
                docs.append(unit(point, r, 'd', digital_arms(point, *rates), question=(
                    f'LayerNorm and LoRA arms at drift regime {point} (realization {r}): 24-h mean teacher KL.')))
            if rates and extend:
                arms = extension_arms(point, rates[1])
                # R0 is split so that each unit fits a 2-h dev job: part y holds the A11 arms.
                parts = ({'x': [a for a in arms if a['method'] != 'learning_newton'],
                          'y': [a for a in arms if a['method'] == 'learning_newton']}
                         if group(point) == 'anchor' else {'x': arms})
                for part, chosen in parts.items():
                    docs.append(unit(point, r, part, chosen, question=(
                        f'Rescaled LoRA, precompensated rewrite, analog learning, oracles and no-drift controls '
                        f'at drift regime {point} (realization {r}): 24-h mean teacher KL.')))
    for doc in docs:
        (out / f"{doc['lifecycle_id']}.json").write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'written': len(docs), 'arms': sum(len(d['drift_regimes']['arms']) for d in docs)}))


if __name__ == '__main__':
    main()
