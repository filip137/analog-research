"""exp-021 on-chip LR/schedule development screen: lifecycles, Jean Zay job specs and the frozen report.

usage:
  exp21_lr_screen.py generate             write the 12 frozen lifecycles
  exp21_lr_screen.py jobs                 print one line per job: <label> <lifecycle_id>
  exp21_lr_screen.py report EVIDENCE      summarize collected metrics.jsonl files under EVIDENCE/<label>/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from slice_campaign import write_frozen
from ebl.schema.lifecycle import parse_lifecycle

PREFIX = 'opt125m-cmo-l6-exp21-lr-screen'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-021-onchip-lr-screen.md'
LIFECYCLES = ROOT/'campaigns/opt-cmo-corruption/lifecycles'
# Arm A of exp-015 (= exp-013 uncapped) and its exact-write dense arm C, unchanged except for the LR recipe.
BASES = {'noisy': LIFECYCLES/'opt125m-cmo-l6-exp7-recovery-regimes-std3-uncapped.json',
         'exact': LIFECYCLES/'opt125m-cmo-l6-exp15-noise-free-std3-dense.json'}
RATES = {'3em3': 3e-3, '1em3': 1e-3, '3em4': 3e-4}
SCHEDULES = ('const', 'cos')
CASES = ['nominal', 'random_30000ppm']
UPDATES = 256
WARMUP = 13
ARRAY = 271002
HWA_ARM = 'standard_hwa_model'


def labels():
    return [f'{law}-{schedule}-{rate}' for law in BASES for schedule in SCHEDULES for rate in RATES]


def lifecycle(label):
    law, schedule, rate = label.split('-')
    doc = json.loads(BASES[law].read_text())
    doc['lifecycle_id'] = f'{PREFIX}-{label}'
    doc['question'] = {
        'text': 'Which on-chip learning rate and schedule recover std3 array 271002 without degrading its healthy state?',
        'decision': 'Development-only LR/schedule screen with a frozen selection rule; held-out domains reported '
                    'after selection. No automatic follow-up. ' + NOTE,
        'evidence_class': 'exploratory_model_based', 'primary_metric': 'teacher_kl'}
    onchip = doc['onchip']
    onchip['learning_rate'] = RATES[rate]
    onchip['case_labels'] = CASES
    if schedule == 'cos':
        onchip['lr_schedule'] = {'kind': 'cosine_warmup_decay', 'total_steps': UPDATES,
                                 'warmup_steps': WARMUP, 'min_factor': 0.0}
    lc = parse_lifecycle(doc)
    if lc.onchip.epochs * -(-lc.data.cohort_size('training') // lc.onchip.batch_size) != UPDATES:
        raise ValueError('Unexpected on-chip budget')
    return lc


def generate():
    for label in labels():
        lc = lifecycle(label)
        path = LIFECYCLES/f'{lc.lifecycle_id}.json'
        write_frozen(path, lc.to_dict())
        print(path.relative_to(ROOT), lc.digest())


def records(path):
    trajectory, final = {}, {}
    for line in path.read_text().splitlines():
        x = json.loads(line)
        if x.get('arm') != 'onchip_weights' or 'case' not in x:
            continue
        if x['stage'] == 'onchip_development':
            trajectory.setdefault(x['case'], []).append((x['global_step'], x['development']['teacher_kl']))
        elif 'transfer_wikitext' in x:
            final[x['case']] = x
    return trajectory, final


def report(evidence):
    rows = {}
    for label in labels():
        files = sorted((evidence/label).glob('**/metrics.jsonl'))
        if len(files) != 1:
            print(f'{label}: missing ({len(files)} metrics files)'); continue
        trajectory, final = records(files[0])
        rows[label] = {case: {'trajectory': trajectory[case], 'development': final[case]['development']['teacher_kl'],
                              'held_out': {name: final[case][name]['teacher_kl']
                                           for name in ('transfer_generated', 'transfer_wikitext')},
                              'device_writes': final[case]['cost']['device_writes']}
                       for case in CASES if case in final}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'jobs', 'report'))
    parser.add_argument('evidence', nargs='?', type=Path)
    args = parser.parse_args()
    if args.command == 'generate':
        generate()
    elif args.command == 'jobs':
        for label in labels():
            print(label, f'{PREFIX}-{label}')
    else:
        print(json.dumps(report(args.evidence), indent=1))


if __name__ == '__main__':
    main()
