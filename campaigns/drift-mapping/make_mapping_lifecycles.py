"""Write the crossbar mapping-search lifecycles: series-002 CMO units with a deployment mapping.

Shared record: on-chip-learning campaigns/drn-drift-mapping/series/002-crossbar, exp-xb1
(crossbar mapping search, user request of 17 November 2026). Each unit is the series-002
CMO-M<m> unit of one realization (make_lifecycles.unit: same source, cohorts, seed 391103,
N = 4 P&V at 0.7431, verify 60 s, arms A0, A0-twin, A1e, A1e-twin) with one explicit
``drift_regimes.mapping`` (training.xbar_mapping). ``base`` is the series-002 mapping,
written explicitly so that the canary exercises the mapping code path.

The screen mappings were chosen from the frozen device-level predictions
(mapping_predictions.py) before any network run.

Usage: PYTHONPATH=. python campaigns/drift-mapping/make_mapping_lifecycles.py [--check]
"""
import importlib.util
import json
from pathlib import Path
import sys

from training.xbar_mapping import parse

HERE = Path(__file__).resolve().parent
# This directory's make_lifecycles (the series-002 units) imports drift-regimes' module of the same name.
_spec = importlib.util.spec_from_file_location('drift_mapping_make_lifecycles', HERE / 'make_lifecycles.py')
_series = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_series)
unit, REALIZATIONS = _series.unit, _series.REALIZATIONS

PREFIX = 'opt125m-xbar-mapping-'


def spec(kind='reference', rule='unit', margin=0., **gain):
    return parse({'kind': kind, 'gain': {'rule': rule, **gain}, 'margin_us': margin})


MAPPINGS = {
    'base': spec(),
    'push-pull': spec('push_pull'),
    'two-sided': spec('two_sided'),
    'two-sided-sat1pc': spec('two_sided', 'saturation', fraction=.01),
}
# (mapping, CMO multiplier) units; x1 and x10 are added for the winner only if budget allows.
UNITS = [(m, 4) for m in MAPPINGS]


def mapped(label, multiplier, realization):
    doc = unit(f'CMO-M{multiplier:g}', realization)
    doc['lifecycle_id'] = f'{PREFIX}{label}-cmo-m{multiplier:g}-{realization}'
    doc['question'] = {
        'text': (f'24-h mean teacher KL of the six-MLP std3 crossbar under drift law CMO-M{multiplier:g} '
                 f'(realization {realization}) with deployment mapping {label}, untreated and with output '
                 'calibration; crossbar mapping search matched to the DRN ratio3 search.'),
        'decision': 'drift-mapping campaign, crossbar series, exp-xb1 mapping search; development only.',
        'primary_metric': 'teacher_kl', 'evidence_class': 'exploratory_model_based'}
    doc['drift_regimes']['mapping'] = MAPPINGS[label]
    return doc


def main():
    check = '--check' in sys.argv[1:]
    out = HERE / 'lifecycles'
    docs = [mapped(label, m, r) for label, m in UNITS for r in REALIZATIONS]
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
    print(json.dumps({'units': len(docs), 'mappings': sorted({label for label, _ in UNITS})}))


if __name__ == '__main__':
    main()
