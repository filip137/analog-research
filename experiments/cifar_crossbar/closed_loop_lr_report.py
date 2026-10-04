"""Predeclared development selection and full learning-rate curve exports."""
import csv
import json
import math
from pathlib import Path
from statistics import mean
from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.closed_loop_lr_config import LRS, SOURCES, CASES, DEV_SEEDS, FINAL_SEEDS
from experiments.cifar_crossbar.hwa_fault_runtime import case_name


def validate(result, *, smoke=False):
    if result['smoke_only'] != smoke or result['recovery_pulse_cap'] is not None or result['per_update_pulse_cap'] != 1:
        raise ValueError('Pulse law or smoke label mismatch.')
    if result['verify_tolerance'] != .04745 or result['epochs'] != 5:
        raise ValueError('Frozen tolerance or epochs changed.')
    development = result['phase'] == 'development'
    if result['test_evaluated'] == development: raise ValueError('Test-set selection leakage.')
    if result['array_seed'] not in ((271001,) if smoke else DEV_SEEDS if development else FINAL_SEEDS):
        raise ValueError('Array partition changed.')
    rates = tuple(result['learning_rates'])
    if development and rates != LRS: raise ValueError('Incomplete LR grid.')
    expected = {(source, case_name(*case), 'onchip_calibration', rate)
                for source in SOURCES for case in CASES for rate in rates}
    if development:
        expected |= {(source, case_name(*case), 'calibration', None) for source in SOURCES for case in CASES}
    actual = [(r['source'], r['case'], r['method'], r['learning_rate']) for r in result['measurements']]
    if set(actual) != expected or len(actual) != len(expected): raise ValueError('Incomplete or duplicate controls.')
    if result['exact_replays'] != len(actual) or result['completed_controls'] != len(actual):
        raise ValueError('Incomplete replay/coverage.')
    p0 = {}
    for row in result['measurements']:
        key = row['source'], row['case']
        if p0.setdefault(key, row['p0_sha256']) != row['p0_sha256']: raise ValueError('Unpaired P0.')
        if len(row['curve']) != 5: raise ValueError('Incomplete epochs.')
        for index, epoch in enumerate(row['curve'], 1):
            if epoch['epoch'] != index or epoch['image_presentations'] != index * (128 if smoke else 45000):
                raise ValueError('Wrong scientific training budget.')
            if development and ('test' in epoch or 'test' in row['initial']): raise ValueError('Test metrics in development.')
            if not math.isfinite(epoch['development']['teacher_kl']): raise ValueError('Nonfinite development KL.')
            if epoch['cost']['max_recovery_cell_pulses'] > index * (2 if smoke else 704): raise ValueError('Per-update pulse limit.')
    return True


def choose(results):
    """Only non-smoke development endpoints can enter the selection function."""
    groups = {}
    for result in results:
        validate(result)
        if result['phase'] != 'development': raise ValueError('Confirmation cannot enter LR selection.')
        key = result['dataset'] + '_' + str(4 if result['suffix'] == 'last_two_blocks' else 8)
        groups.setdefault(key, []).append(result)
    if set(groups) != {ds + '_' + str(depth) for ds in ('cifar10', 'cifar100') for depth in (4, 8)}:
        raise ValueError('Incomplete dataset/depth selection groups.')
    selected = {}
    for key, runs in groups.items():
        if sorted(r['array_seed'] for r in runs) != list(DEV_SEEDS): raise ValueError('Incomplete development arrays.')
        hashes = {r['source_checkpoint_sha256'] for r in runs}
        if len(hashes) != 1: raise ValueError('Source changed within selection.')
        scores = {str(rate): mean(row['curve'][-1]['development']['teacher_kl'] for run in runs
                                 for row in run['measurements'] if row['learning_rate'] == rate)
                  for rate in LRS}
        selected[key] = {'selected_learning_rate': min(LRS, key=lambda rate: (scores[str(rate)], rate)),
                         'mean_development_kl_by_rate': scores, 'endpoints_per_rate': 12,
                         'source_checkpoint_sha256': next(iter(hashes))}
    return {'schema': 'cifar_om_closed_loop_lr.selection.v1', 'development_arrays': list(DEV_SEEDS),
            'criterion': 'mean_epoch5_development_teacher_kl',
            'objective_note': 'Raw KL mean over two sources, three fault cases and two arrays; severe cases may dominate.',
            'tie_break': 'smaller_learning_rate', 'groups': selected}


def select_from_ledger(base):
    state = json.loads((base / 'execution-v1/campaign-execution.json').read_text())
    entries = [(key, value) for key, value in state['completed'].items()
               if '-development-v1__' in key]
    receipt = choose([entry['metrics'] for _, entry in entries])
    receipt['evidence'] = {key: {'result_sha256': sha256_file(Path(entry['run']) / 'result.json'),
                                'run': entry['run']} for key, entry in entries}
    path = base / 'execution-v1/selection.json'
    if path.exists() and json.loads(path.read_text()) != receipt: raise ValueError('Frozen selection changed.')
    atomic_write_json(path, receipt)
    return path


def report(base, output):
    """Keep every LR and epoch, including clean and per-source outcomes."""
    state = json.loads((base / 'execution-v1/campaign-execution.json').read_text())
    flat = []
    for key, entry in state['completed'].items():
        result = entry['metrics']
        if result['smoke_only']: continue
        validate(result)
        for row in result['measurements']:
            for epoch in row['curve']:
                for split in ('development', 'test'):
                    if split not in epoch: continue
                    flat.append({'dataset': result['dataset'], 'convolutions': 4 if result['suffix'] == 'last_two_blocks' else 8,
                        'phase': result['phase'], 'array_seed': result['array_seed'], 'source': row['source'],
                        'case': row['case'], 'method': row['method'], 'learning_rate': row['learning_rate'],
                        'epoch': epoch['epoch'], 'split': split,
                        'accuracy_percent': epoch[split]['accuracy_percent'], 'teacher_kl': epoch[split]['teacher_kl'],
                        'accuracy_gain_pp': epoch[split]['accuracy_percent'] - row['initial'][split]['accuracy_percent'],
                        'kl_improvement': row['initial'][split]['teacher_kl'] - epoch[split]['teacher_kl'],
                        **epoch['cost']})
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / 'curves.json', flat)
    if flat:
        with (output / 'curves.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    receipt_path = base / 'execution-v1/selection.json'
    if receipt_path.exists(): atomic_write_json(output / 'selection.json', json.loads(receipt_path.read_text()))
    return {'rows': len(flat), 'completed_native_runs': len(state['completed']), 'active_native_runs': len(state['active'])}
