"""Derive the mapping-faults configs (on-chip-learning #33, series 003 exp-001) from the paper grid.

Only the declared factors change: ``deployment.mapping_scale`` in {1 (base), 2, 3} on both the
deploy and the fault_tuning config, and the arm list reduced to the two holds (naive and
fault-aware; nothing is trained). Fault map 271002, rates 0/1/3/5 %, cohorts (dev 64,
WikiText test 1,024, generated 384), source, teacher and solver are the paper-grid values.
One extra check case, ``drn6-ratio3-r00-271001``, replays the CMO exp-001 DRN6 ratio3
deployment (map 271001, 0 %) to reproduce its mapped-target and release KL.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'examples/opt_cmo_drn/fault_paper_grid/paper'
TARGET = ROOT / 'examples/opt_cmo_drn/mapping_faults'
MAPPINGS = {'base': 1.0, 'ratio2': 2.0, 'ratio3': 3.0}
RATES = ('00', '01', '03', '05')
ARMS = [{'label': 'naive_hold', 'method': 'naive_hold'}, {'label': 'fault_aware_hold', 'method': 'hold'}]


def build(model, mapping, rate, stage, assignment=271002):
    name = f'{model}-r{rate}-271002' + ('-deploy' if stage == 'deploy' else '')
    doc = json.loads((SOURCE / f'{name}.json').read_text())
    doc['deployment']['mapping_scale'] = MAPPINGS[mapping]
    doc['deployment']['assignment_seed'] = assignment
    if stage == 'tune':
        doc['fault_tuning']['arms'] = [dict(a) for a in ARMS]
    from ebl.schema.opt_cmo_drn import parse_config
    parse_config(doc)
    return doc


def cases():
    for model in ('drn6', 'drn12'):
        for mapping in MAPPINGS:
            for rate in RATES:
                yield f'{model}-{mapping}-r{rate}-271002', (model, mapping, rate, 271002)
    yield 'drn6-ratio3-r00-271001', ('drn6', 'ratio3', '00', 271001)


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for cell, (model, mapping, rate, assignment) in cases():
        for stage, suffix in (('deploy', '-deploy'), ('tune', '')):
            out = TARGET / f'{cell}{suffix}.json'
            out.write_text(json.dumps(build(model, mapping, rate, stage, assignment), indent=2, sort_keys=True) + '\n')
            print(out)


if __name__ == '__main__':
    main()
