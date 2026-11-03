"""exp-022 noiseless vs noisy on-chip learning after intelligent P&V: lifecycles and the per-case report.

usage:
  exp22_pv_learning.py generate          write the frozen lifecycles
  exp22_pv_learning.py jobs              print one line per job: <label> <lifecycle_id> <deployment>
  exp22_pv_learning.py report EVIDENCE   per-case table from EVIDENCE/<label>/result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from slice_campaign import write_frozen
from workflow.lifecycle import parse_lifecycle

PREFIX = 'opt125m-cmo-l6-exp22-pv-learning'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-022-pv-learning-noise.md'
LIFECYCLES = ROOT/'campaigns/opt-cmo-corruption/lifecycles'
# The meeting's deployments: exp-016 phase 3 (gmin/gmax 1-6%) and the exp-013 P0 (healthy, random 3%).
BASES = {'kinds': LIFECYCLES/'opt125m-cmo-l6-exp16-pair-rewrite-std3-kinds-learn.json',
         'random3': LIFECYCLES/'opt125m-cmo-l6-exp16-pair-rewrite-std3-random3.json'}
GROUPS = {
    'gmin-lo': ('kinds', ['gmin_10000ppm', 'gmin_20000ppm', 'gmin_30000ppm']),
    'gmin-hi': ('kinds', ['gmin_40000ppm', 'gmin_50000ppm', 'gmin_60000ppm']),
    'gmax-lo': ('kinds', ['gmax_10000ppm', 'gmax_20000ppm', 'gmax_30000ppm']),
    'gmax-hi': ('kinds', ['gmax_40000ppm', 'gmax_50000ppm', 'gmax_60000ppm']),
    'random3': ('random3', ['nominal', 'random_30000ppm']),
    # Amendment (21 October): the other random rates already in the exp-013 P0, for the DRN comparison figure.
    'random-lo': ('random3', ['random_10000ppm', 'random_20000ppm']),
    'random-hi': ('random3', ['random_40000ppm', 'random_50000ppm']),
    # Amendment (21 October): random 6% is not in the exp-013 P0; this group deploys it in its own deploy stage.
    'random6': ('random3', ['nominal', 'random_60000ppm']),
}
DEFECT_CASES = {'random6': [{'array_concentration': None, 'die_concentration': None, 'kind': kind, 'rate_ppm': rate}
                            for kind, rate in (('none', 0), ('random', 60000))]}
VERIFY = {'detection': 'verify', 'roles': 'pair', 'seed': 241016, 'threshold_sigma': 5.0}
ARMS = [{'id': 'none', 'weights': 'hold', 'calibration': False},
        {'id': 'pair_rewrite_verify', 'weights': 'hold', 'calibration': False, 'fault_rewrite': VERIFY},
        {'id': 'pair_rewrite_verify_recovery', 'weights': 'learn', 'calibration': False, 'fault_rewrite': VERIFY}]
# exp-021's valid exact-write oracle.
COSINE = {'kind': 'cosine_warmup_decay', 'total_steps': 256, 'warmup_steps': 13, 'min_factor': 0.0}


def labels():
    return [f'{law}-{group}' for law in ('noisy', 'exact') for group in GROUPS]


# Amendment (21 October): program-and-verify only at random 6%, without learning arms.
PV_ONLY = ['pv-random6']


def lifecycle(label):
    law, group = label.split('-', 1)
    base, cases = GROUPS[group]
    doc = json.loads(BASES[base].read_text())
    doc['lifecycle_id'] = f'{PREFIX}-{label}'
    if group in DEFECT_CASES:
        doc['defects']['cases'] = DEFECT_CASES[group]
    doc['question'] = {
        'text': 'After intelligent P&V, how much do noiseless and noisy on-chip learning add per fault kind and rate?',
        'decision': 'Descriptive per-case comparison for the October 22 meeting; fixed recipes chosen beforehand '
                    '(noisy: exp-016; noiseless: exp-021 oracle). No automatic follow-up. ' + NOTE,
        'evidence_class': 'exploratory_model_based', 'primary_metric': 'teacher_kl'}
    onchip = doc['onchip']
    onchip['arms'] = ARMS[:2] if law == 'pv' else ARMS
    onchip['case_labels'] = cases
    if law == 'exact':
        for key in ('threshold_sigma', 'accumulate_updates', 'max_write_attempts_per_device'):
            del onchip[key]
        onchip.update(update_law='cmo_idealized_write_adam', write_noise_scale=0.0, lr_schedule=COSINE)
    return parse_lifecycle(doc)


def generate():
    for label in labels() + PV_ONLY:
        lc = lifecycle(label)
        path = LIFECYCLES/f'{lc.lifecycle_id}.json'
        write_frozen(path, lc.to_dict())
        print(path.relative_to(ROOT), lc.digest())


def report(evidence):
    out = {}
    for label in labels():
        law, group = label.split('-', 1)
        path = evidence/label/'measurements.json'
        if not path.exists():
            print(f'{label}: missing', file=sys.stderr); continue
        for row in json.loads(path.read_text()):
            key = (row['case'], row['onchip_arm'])
            entry = out.setdefault(row['case'], {})
            name = {'none': 'initial', 'pair_rewrite_verify': 'rewrite'}.get(row['onchip_arm'], law)
            final = {d: {m: row['final'][d][m] for m in ('teacher_kl', 'perplexity')}
                     for d in ('development', 'transfer_generated', 'transfer_wikitext')} \
                if isinstance(row['final'].get('development'), dict) else row['final']
            value = {'final': final, 'cost': row.get('cost', {}), 'label': label}
            if name in ('initial', 'rewrite') and name in entry and entry[name]['final'] != final:
                raise ValueError(f'{key}: hold arms differ between laws')
            entry[name] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'jobs', 'report'))
    parser.add_argument('evidence', nargs='?', type=Path)
    args = parser.parse_args()
    if args.command == 'generate':
        generate()
    elif args.command == 'jobs':
        for label in labels():
            print(label, f'{PREFIX}-{label}', GROUPS[label.split('-', 1)[1]][0])
    else:
        print(json.dumps(report(args.evidence), indent=1))


if __name__ == '__main__':
    main()
