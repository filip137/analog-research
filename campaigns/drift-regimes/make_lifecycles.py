"""Write the drift-regimes exp-001 crossbar qualification lifecycles.

The base is the drift-mitigation exp-002 six-MLP std3 lifecycle: the same
network, devices, HWA arm and deployment mapping, without its
initial_programming section. Each regime point gets one lifecycle (A0,
untreated). Its deployment threshold is the N = 4 optimum from the exp-001
step-2 table (kappa_family.json, copied from on-chip-learning).

Usage: python campaigns/drift-regimes/make_lifecycles.py
"""
from dataclasses import asdict
import json
from pathlib import Path

from training.drift_family import ReramFamily

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'drift-mitigation/lifecycles/opt125m-flowers-initial-p0.json'
SOURCE_SHA256 = '86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028'
CORPUS = ('wikitext-2-raw-v1', '33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01')
COHORTS = {64: '7411cf50b625e4c62f65d8e12bac538fd83c6dad3e4c74b29a9f2a6f5ad83191',
           2: '4e86fa42cb4af0c684b04fc03aadc913abdf8b62cfbbd03417d910df3b5270e3'}
ATTEMPTS = 4
POINTS = {
    'R0': {}, 'rho1.0': dict(rho_star=1.), 'rho0.7': dict(rho_star=.7), 'rho0.5': dict(rho_star=.5),
    'rho0.18': dict(rho_star=.18), 'f0.3': dict(persistent_share=.3), 'f0.7': dict(persistent_share=.7),
    'f1.0': dict(persistent_share=1.), 'sigma_a0.5': dict(amplitude_sigma=.5),
    'corner_rho1_f1': dict(rho_star=1., persistent_share=1.),
}


def lifecycle(point, threshold, *, canary=False):
    doc = json.loads(BASE.read_text())
    del doc['initial_programming']
    slug = point.replace('.', 'p').replace('_', '-').lower()
    doc['lifecycle_id'] = 'opt125m-drift-regimes-qual-canary' if canary else f'opt125m-drift-regimes-qual-{slug}'
    doc['question'] = {
        'text': ('How much untreated 24-h teacher-KL damage does drift regime '
                 f'{point} cause on the six-MLP std3 crossbar after optimal-threshold P&V deployment?'),
        'decision': ('drift-regimes exp-001 instrument qualification (damage guard and cost probe); '
                     'development only, no method comparison.'),
        'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    sequences = 2 if canary else 64
    doc['drift_regimes'] = {
        'source_sha256': SOURCE_SHA256, 'evaluation_corpus': CORPUS[0], 'evaluation_corpus_sha256': CORPUS[1],
        'cohort_sha256': COHORTS[sequences], 'regime': point,
        'law': asdict(ReramFamily(**POINTS[point])), 'arm': 'A0',
        'deployment_attempts': ATTEMPTS, 'deployment_threshold': threshold, 'verify_seconds': 60.,
        'assignment_seeds': [271001] if canary else [271001, 271002, 271003], 'seed': 391103,
        'evaluation_split': 'validation', 'evaluation_sequences': sequences,
        'evaluation_seconds': [0., 60., 900., 1800., 3600., 7200., 14400., 28800., 57600., 86400.],
        'canary': canary}
    return doc


def main():
    table = json.loads((HERE / 'kappa_family.json').read_text())['points']
    out = HERE / 'lifecycles'
    out.mkdir(exist_ok=True)
    written = []
    for point in POINTS:
        threshold = round(table[point]['by_attempts'][str(ATTEMPTS)]['kappa_star'], 4)
        for canary in ((True, False) if point == 'R0' else (False,)):
            doc = lifecycle(point, threshold, canary=canary)
            path = out / f"{doc['lifecycle_id']}.json"
            path.write_text(json.dumps(doc, indent=2) + '\n')
            written.append(str(path.relative_to(HERE.parent.parent)))
    print(json.dumps(written, indent=1))


if __name__ == '__main__':
    main()
