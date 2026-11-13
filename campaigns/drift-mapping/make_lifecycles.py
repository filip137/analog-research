"""Write the crossbar lifecycles of the drift-mapping campaign: the DRN tests, head to head.

Shared record: on-chip-learning campaigns/drn-drift-mapping/ (series 002, crossbar). Every unit
is one realization of the six-MLP std3 crossbar with the exp-002 R0 base (make_lifecycles), the
same cohorts, realizations (271001-271003), N = 4 P&V deployment at threshold 0.7431 and arm
schedules as the DRN runs (DRN branch drn/drift-mapping, campaigns/drift-mapping/make_configs.py).
The crossbar keeps its own mapping (active device vs a 48.6 uS reference) and its own
calibration (a per-column affine readout of every analog matrix).

Units:
- match (DRN exp-002/003 base rows): R0 and R0-G; arms A0, A1, A0-twin, A1e, A1e-twin.
- attribution (DRN exp-001): R0 with no floor (drift_min 0), mean only (s 0), spread only (m 0);
  arms A0, A1, A0-twin.
- aihwkit (DRN exp-006): aihwkit 1.1.0 ReRamCMONoiseModel relaxation x M (log time, independent
  draws at every read time), M in {1, 4, 10}; arms A0, A0-twin, A1e, A1e-twin.

Usage: PYTHONPATH=. python campaigns/drift-mapping/make_lifecycles.py [--check]
"""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'drift-regimes'))
from make_lifecycles import lifecycle                      # noqa: E402
from make_measured_g_lifecycles import MEASURED            # noqa: E402
import make_screen_lifecycles as screen                    # noqa: E402
from training.drift_family import ReramFamily              # noqa: E402

PREFIX = 'opt125m-drift-mapping-xbar-'
THRESHOLD = 0.7431          # the DRN runs' deployment threshold (R0's kappa*_4), used for every law
REALIZATIONS = (271001, 271002, 271003)
EVENTS = [900., 3600., 14400., 57600.]
EARLY = [60.] + EVENTS
A0 = {'label': 'A0', 'method': 'none'}
A1 = {'label': 'A1', 'method': 'calibration', 'events': EVENTS}
TWIN = {'label': 'A0-twin', 'method': 'none', 'twin': True}
A1E = {'label': 'A1e', 'method': 'calibration', 'events': EARLY}
A1E_TWIN = {'label': 'A1e-twin', 'method': 'calibration', 'events': EARLY, 'twin': True}
AIHWKIT_CMO = {'mean': -0.08900206, 'slope': 0.04201137, 'offset': 0.41183342}

LAWS = {
    'R0': {},
    'R0-G': dict(mean_us=None, spread_us=None, coefficient_bins=MEASURED),
    'R0-nofloor': dict(drift_min=0.),
    'R0-mean': dict(spread_us=0.),
    'R0-spread': dict(mean_us=0.),
    **{f'CMO-M{m:g}': dict(rho_star=1., mean_us=AIHWKIT_CMO['mean'] * m, spread_us=AIHWKIT_CMO['slope'] * m,
                           time_law='log', log_offset_us=AIHWKIT_CMO['offset'] * m, temporal='independent')
       for m in (1., 4., 10.)},
}
ARMS = {
    'R0': [A0, A1, TWIN, A1E, A1E_TWIN], 'R0-G': [A0, A1, TWIN, A1E, A1E_TWIN],
    'R0-nofloor': [A0, A1, TWIN], 'R0-mean': [A0, A1, TWIN], 'R0-spread': [A0, A1, TWIN],
    'CMO-M1': [A0, TWIN, A1E, A1E_TWIN], 'CMO-M4': [A0, TWIN, A1E, A1E_TWIN], 'CMO-M10': [A0, TWIN, A1E, A1E_TWIN],
}


def unit(point, realization):
    doc = lifecycle('R0', THRESHOLD)
    doc['lifecycle_id'] = f"{PREFIX}{point.lower()}-{realization}"
    doc['question'] = {
        'text': (f'24-h mean teacher KL of the six-MLP std3 crossbar under drift law {point} '
                 f'(realization {realization}), untreated and with output calibration, head to head with the DRN.'),
        'decision': 'drift-mapping campaign, crossbar series; development only.',
        'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    spec = doc['drift_regimes']
    del spec['arm']
    spec['regime'] = point
    spec['law'] = json.loads(json.dumps(asdict(ReramFamily(**LAWS[point]))))
    spec['assignment_seeds'] = [realization]
    spec['arms'] = copy.deepcopy(ARMS[point])
    spec['adaptation'] = screen.ADAPTATION
    return doc


def main():
    check = '--check' in sys.argv[1:]
    out = HERE / 'lifecycles'
    out.mkdir(exist_ok=True)
    docs = [unit(p, r) for p in LAWS for r in REALIZATIONS]
    stale = []
    for doc in docs:
        path = out / f"{doc['lifecycle_id']}.json"
        text = json.dumps(doc, indent=2) + '\n'
        if check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.name)
        else:
            path.write_text(text)
    if stale:
        raise SystemExit(f'Stale lifecycles: {stale}')
    print(json.dumps({'units': len(docs), 'arms': sum(len(d['drift_regimes']['arms']) for d in docs)}))


if __name__ == '__main__':
    main()
