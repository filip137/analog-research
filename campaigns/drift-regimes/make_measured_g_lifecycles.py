"""Write the drift-regimes exp-004 crossbar lifecycles: R0-G, R0 with the measured conductance bins.

exp-004 (on-chip-learning campaigns/drift-regimes/series/002-regime-map/experiments/
exp-004-measured-g-r0.md) reruns exp-002's full R0 arm set under the R0-G law: m and s
from the 28 October 25-h bin table, a step bin per programmed conductance. Everything
else is the exp-002 R0 screen: the same base, realizations, cohorts, events and arm
definitions (make_screen_lifecycles.py), and the same unit split:
- ``w``: arms without a learning rate (A0, A1, A1o, A8 and A9 with the N sweep);
- ``y``: the A11 arms (fisher_net closed loop, no learning rate);
- ``tune-ln``, ``tune-lora``: R0-G on realization 271004, validation sequences 64-127;
- ``d``: LayerNorm and LoRA arms at the R0-G tuned rates;
- ``x``: A6, A10, O1, O2, A4c and the no-drift twins (the LoRA ones at the tuned rate).
Deployment and rewrite thresholds are R0-G's kappa*_N (kappa_measured_g.json).

The canary (2 sequences) has two units: R0-G with one arm of every method, and scalar R0
with run 1's A0, A8-N1 and A9-W1-N4, whose final conductance hashes must equal the
exp-002 run-1 canary's (c730d79e, b2f72464, 11473859): the scalar law is unchanged.

Usage: PYTHONPATH=. python campaigns/drift-regimes/make_measured_g_lifecycles.py [layernorm_lr lora_lr]
Without rates only the canary, tuning, w and y units are written.
"""
from dataclasses import asdict
import json
import sys

from make_lifecycles import HERE, lifecycle
import make_screen_lifecycles as screen
from training.drift_family import ReramFamily

POINT = 'R0-G'
# (lower_us, mean_us, spread_us) at a 1-h reference: the 28 October table from the 25-h retention data.
MEASURED = ((0., .59, 2.35), (10., .02, 3.35), (20., -.43, 5.54), (30., -.29, 7.28), (40., -4.51, 9.93),
            (50., -3.04, 11.05), (65., -2.45, 8.49), (80., -2.52, 9.56), (100., -1.03, 6.66),
            (115., -.68, 6.28), (130., -.87, 4.83), (145., -1.01, 4.59), (160., -.88, 3.74))
LAW = asdict(ReramFamily(mean_us=None, spread_us=None, coefficient_bins=MEASURED))
DECISION = 'drift-regimes exp-004: R0 with measured conductance-dependent drift (crossbar); development only.'
PREFIX = 'opt125m-drift-regimes-screen-'


def base(point, *, canary=False):
    """exp-002's R0 base lifecycle; for R0-G the measured-bin law and R0-G's deployment threshold."""
    doc = lifecycle('R0', screen.kappa(point, 4), canary=canary)
    if point == POINT:
        spec = doc['drift_regimes']
        spec['regime'], spec['law'] = POINT, json.loads(json.dumps(LAW))
    return doc


def unit(realization, part, arms, *, question, evaluation=None):
    doc = base(POINT)
    doc['lifecycle_id'] = f'{PREFIX}{screen.slug(POINT)}-{realization}-{part}'
    doc['question'] = {'text': question, 'decision': DECISION, 'primary_metric': 'teacher_kl',
                       'evidence_class': 'exploratory_model_based'}
    spec = doc['drift_regimes']
    del spec['arm']
    spec['assignment_seeds'] = [realization]
    spec['arms'] = arms
    spec['adaptation'] = (screen.ADAPTATION if any(a['method'] not in ('none', 'rewrite') for a in arms)
                          else None)
    if evaluation is not None:
        spec['evaluation_start'], spec['cohort_sha256'] = evaluation
    return doc


def tuning():
    arms = [{'label': 'A0', 'method': 'none'},
            *[screen.layernorm(f'A3-lr{lr:g}', screen.EVENTS, lr) for lr in screen.TUNE_LAYERNORM],
            *[screen.lora(f'A5-lr{lr:g}', screen.EVENTS, lr) for lr in screen.TUNE_LORA]]
    question = ('Which LayerNorm and LoRA learning rates minimize the 24-h mean teacher KL at R0-G '
                '(development realization, held-out validation sequences 64-127)?')
    return [unit(screen.TUNE_REALIZATION, part, chosen, question=question, evaluation=screen.TUNE_COHORT)
            for part, chosen in (('tune-ln', arms[:5]), ('tune-lora', arms[:1] + arms[5:]))]


def canary():
    events2 = screen.EVENTS[:2]
    g = base(POINT, canary=True)
    g['lifecycle_id'] = f'{PREFIX}canary5-{screen.slug(POINT)}'
    g['question'] = {'text': 'Does every arm method run on the six-MLP crossbar under the R0-G law?',
                     'decision': 'drift-regimes exp-004 launch gate; no method comparison.',
                     'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    spec = g['drift_regimes']
    del spec['arm']
    lr = screen.LORA_LR
    spec['arms'] = [{'label': 'A0', 'method': 'none'}, screen.calibration('A1', events2),
                    screen.layernorm('A3', screen.EVENTS[:1], screen.PROBE_LAYERNORM_LR),
                    screen.lora('A5', events2, lr),
                    screen.rewrite(POINT, 'A8', 1), screen.rewrite(POINT, 'A9-W1', 4),
                    {**screen.precompensated(POINT, 'W1', 4), 'events': events2},
                    {'label': 'A6', 'method': 'lora_rescaled', 'events': screen.EVENTS[:1],
                     'reads': screen.EVENTS[1:2], 'rank': 16, 'learning_rate': lr},
                    {'label': 'O1', 'method': 'lora_oracle', 'events': [0., 900.], 'rank': 16, 'learning_rate': lr},
                    {'label': 'O2', 'method': 'rewrite_oracle', 'events': events2, 'fraction': .01,
                     'attempts': 4, 'threshold': screen.kappa(POINT, 4)},
                    {'label': 'A0-twin', 'method': 'none', 'twin': True},
                    {**screen.newton(POINT, 'A11-W2-N8', 'W2', 8), 'events': events2},
                    {**screen.newton(POINT, 'A11-W2-ideal', 'W2', 1, ideal=True), 'events': screen.EVENTS[:1]}]
    spec['adaptation'] = screen.ADAPTATION
    r = base('R0', canary=True)
    r['lifecycle_id'] = f'{PREFIX}canary5-r0'
    r['question'] = {'text': 'Do run 1 canary arms reproduce their final conductance hashes under the scalar R0 law?',
                     'decision': 'drift-regimes exp-004 launch gate: the scalar law is unchanged.',
                     'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    spec = r['drift_regimes']
    del spec['arm']
    spec['arms'] = [{'label': 'A0', 'method': 'none'}, screen.rewrite('R0', 'A8', 1), screen.rewrite('R0', 'A9-W1', 4)]
    spec['adaptation'] = None
    return [g, r]


def main():
    rates = [float(x) for x in sys.argv[1:3]]
    docs = [*canary(), *tuning()]
    extension = screen.extension_arms(POINT, rates[1] if rates else screen.LORA_LR)
    for r in screen.REALIZATIONS:
        docs.append(unit(r, 'w', screen.write_arms(POINT), question=(
            f'Untreated, calibration and rewrite arms at R0-G (realization {r}): 24-h mean teacher KL and write cost.')))
        docs.append(unit(r, 'y', [a for a in extension if a['method'] == 'learning_newton'], question=(
            f'Analog learning (A11) arms at R0-G (realization {r}): 24-h mean teacher KL and write cost.')))
        if rates:
            docs.append(unit(r, 'd', screen.digital_arms(POINT, *rates), question=(
                f'LayerNorm and LoRA arms at R0-G (realization {r}): 24-h mean teacher KL.')))
            docs.append(unit(r, 'x', [a for a in extension if a['method'] != 'learning_newton'], question=(
                f'Rescaled LoRA, precompensated rewrite, oracles and no-drift controls at R0-G '
                f'(realization {r}): 24-h mean teacher KL.')))
    out = HERE / 'lifecycles'
    for doc in docs:
        (out / f"{doc['lifecycle_id']}.json").write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'written': len(docs), 'arms': sum(len(d['drift_regimes']['arms']) for d in docs)}))


if __name__ == '__main__':
    main()
