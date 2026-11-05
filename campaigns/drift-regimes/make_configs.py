"""Generate the strict exp-003 configs from the crossbar counterpart table.

Arm settings are copied verbatim from the counterpart table's ``crossbar_settings``, so
every DRN arm is the crossbar arm by construction, except for the declared Fisher
batch, which is unchanged (16). Usage:

    python campaigns/drift-regimes/make_configs.py COUNTERPARTS.csv [--check]

Writes ``campaigns/drift-regimes/configs/exp-003/*.json``: one screen config per
(regime, realization, part), one R0 tuning config on realization 271004 and the
two readiness configs (R0 and the persistent corner).
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'configs' / 'exp-003'
REGIMES = {  # rho_star, persistent share f, N=4 P&V threshold (crossbar launch rounding)
    'R0': (.335, 0., .7431), 'rho1.0': (1., 0., .7029), 'rho0.18': (.18, 0., .6953),
    'f1.0': (.335, 1., .9090), 'corner_rho1_f1': (1., 1., .8908),
}
SEEDS = (271001, 271002, 271003)
TUNING_SEED = 271004
TUNING_RATES = (3e-4, 1e-3, 3e-3, 1e-2)
EVENTS = [900., 3600., 14400., 57600.]
DRN_SOURCE = 'dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9'
CORPUS = ('wikitext-2-raw-v1', '33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01')
REPORTING = '7411cf50b625e4c62f65d8e12bac538fd83c6dad3e4c74b29a9f2a6f5ad83191'
TUNING = '99c12a626280f6be079b24ae72cacb1900a07f6917ef530cc4c2d1bf1ad7014c'
TRAINING = 'c919329dbe0d545d7614daf3d8f790881e944cd80ff53fff88fc19d9697f7762'
# Run parts: arms of one deployment split over jobs; arms within a part share evaluations.
# A3 is its own part ('ln'): it runs only after the R0 tuning run has frozen its rate.
PARTS = {'R0': {'a': ('A0', 'A1', 'A0-twin', 'A9-W2-N4', 'A10-W2-N4'),
                'b': ('A8-N4',),
                'c': ('A11-W2-N4', 'A11-W2-twin')}}


def parts(regime, labels):
    chosen = dict(PARTS.get(regime, {'a': tuple(l for l in labels if l != 'A3')}))
    if 'A3' in labels:
        chosen['ln'] = ('A3',)
    return chosen


def law(rho_star, f):
    return {'rho_star': rho_star, 'persistent_share': f, 'cmo_scale': 1.0, 'cmo_acceptance': 2.0,
            'amplitude_sigma': 0.0, 'mean_us': -2.0, 'spread_us': 3.0, 'mean_exponent': .23,
            'spread_exponent': .11, 'reference_seconds': 3600.0, 'program_min': 9.0, 'program_max': 88.2,
            'drift_min': 5.0, 'drift_max': 150.0, 'time_dtype': 'float64'}


def config(stage, regime, realization, arms, *, start=0, cohort=REPORTING):
    rho_star, f, threshold = REGIMES[regime]
    return {
        'experiment_id': 'opt_cmo_drn_drift_regimes.v1', 'schema_version': 1, 'stage': stage,
        'device': 'cuda:0', 'cpu_threads': 10,
        'drift_regimes': {
            'regime': regime, 'seed': 391103, 'realization': realization, 'source_sha256': DRN_SOURCE,
            'corpus': CORPUS[0], 'corpus_sha256': CORPUS[1], 'sequence_length': 256, 'law': law(rho_star, f),
            'deployment_attempts': 4, 'deployment_threshold': threshold, 'verify_seconds': 60.0,
            'evaluation_seconds': [0., 60., 900., 1800., 3600., 7200., 14400., 28800., 57600., 86400.],
            'evaluation': {'split': 'validation', 'start': start, 'sequences': 64, 'sha256': cohort},
            'evaluation_microbatch': 4,
            'adaptation': {'split': 'train', 'seed': 20261104, 'sequences': 1024, 'probe_sequences': 16,
                           'steps': 300, 'batch': 8, 'microbatch': 8, 'sha256': TRAINING},
            'forward_policy': {'name': 'settle4_grad4_eval8.v1', 'settle_sweeps': 4, 'gradient_sweeps': 4,
                               'evaluation_sweeps': 8},
            'arms': arms, 'canary': False}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('counterparts', type=Path)
    parser.add_argument('--check', action='store_true', help='fail if a generated file differs')
    parser.add_argument('--a3-rate', type=float, default=None,
                        help='LayerNorm rate frozen by the R0 tuning run; A3 configs are written only with it')
    args = parser.parse_args()
    cells = {}
    with args.counterparts.open() as handle:
        for row in csv.DictReader(handle):
            settings = json.loads(row['crossbar_settings'])
            key = (row['regime'], int(row['realization']))
            cells.setdefault(key, {})[settings['label']] = settings
    if sum(len(v) for v in cells.values()) != 75 or set(k[1] for k in cells) != set(SEEDS):
        raise SystemExit('Expected 75 counterparts over the three production realizations.')
    documents = {}
    for (regime, realization), arms in sorted(cells.items()):
        labels = list(arms)
        split = parts(regime, labels)
        if sorted(sum(map(list, split.values()), [])) != sorted(labels):
            raise SystemExit(f'Parts of {regime} do not cover its arms exactly.')
        for part, chosen in split.items():
            if part == 'ln':
                if args.a3_rate is None:
                    continue
                selected = [{**arms[label], 'learning_rate': args.a3_rate} for label in chosen]
            else:
                selected = [arms[label] for label in chosen]
            name = f'screen-{regime}-{realization}-{part}.json'
            documents[name] = config('drift_regimes', regime, realization, selected)
    tuning = [{'label': 'A0', 'method': 'none'}] + [
        {'events': EVENTS, 'label': f'A3-lr{rate:g}', 'learning_rate': rate, 'method': 'layernorm'}
        for rate in TUNING_RATES]
    # Two jobs; the shared deployment is deterministic, so splitting the arms changes no value.
    documents['tune-R0-271004-a.json'] = config('drift_regimes', 'R0', TUNING_SEED, tuning[:3], start=64, cohort=TUNING)
    documents['tune-R0-271004-b.json'] = config('drift_regimes', 'R0', TUNING_SEED, tuning[3:], start=64, cohort=TUNING)
    for regime in ('R0', 'corner_rho1_f1'):
        documents[f'readiness-{regime}-271004.json'] = config('readiness', regime, TUNING_SEED,
                                                              [{'label': 'A0', 'method': 'none'}],
                                                              start=64, cohort=TUNING)
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, document in documents.items():
        text = json.dumps(document, indent=2) + '\n'
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
    if stale:
        raise SystemExit(f'Stale configs: {stale}')
    trajectories = sum(len(d['drift_regimes']['arms']) for n, d in documents.items() if n.startswith('screen'))
    print(json.dumps({'configs': len(documents), 'screen_trajectories': trajectories,
                      'tuning_trajectories': len(tuning)}))


if __name__ == '__main__':
    main()
