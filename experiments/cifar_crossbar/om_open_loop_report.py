"""Coverage checked open-loop comparisons, including inherited V2 controls."""
from collections import defaultdict
import json
import math
from pathlib import Path
from experiments.artifacts import atomic_write_json, sha256_file
from experiments.cifar_crossbar.om_open_loop_config import OmOpenLoopSpec
from experiments.cifar_crossbar.sweep_config import SOURCES, cases
from experiments.cifar_crossbar.hwa_fault_runtime import case_name
from experiments.cifar_crossbar.sweep_report import aggregate, csv_write


def validate(result, smoke=False):
    from dataclasses import replace
    spec = OmOpenLoopSpec()
    if smoke: spec = replace(spec, fault_rates_ppm=(0, 50000))
    wanted = {(s, case_name(k, p), m) for s in SOURCES for k, p in cases(spec, s) for m in spec.methods}
    rows = result['measurements']
    if len(rows) != len(wanted) or {(r['source'], r['case'], r['method']) for r in rows} != wanted:
        raise ValueError('Missing or duplicate open-loop controls.')
    if result['completed_controls'] != len(wanted) or result['exact_replays'] != len(wanted) * 3 // 4:
        raise ValueError('Missing controls or endpoint replay.')
    if result['epochs'] != 5 or result['recovery_images'] != (128 if smoke else 45000) or bool(result['smoke_only']) != smoke:
        raise ValueError('Wrong recovery budget.')
    if result['per_update_pulse_cap'] != 1 or result['recovery_pulse_cap'] is not None or result['verification_reads'] != 0:
        raise ValueError('Wrong pulse law.')
    groups = defaultdict(list)
    for row in rows: groups[(row['source'], row['case'])].append(row)
    for group in groups.values():
        for key in ('p0_sha256', 'initial_apparent_sha256', 'identity', 'initial_test', 'programming'):
            if len({json.dumps(r[key], sort_keys=True) for r in group}) != 1: raise ValueError('Unpaired ' + key)
        for row in group:
            expected_epochs = [] if row['method'] == 'none' else [1, 2, 3, 4, 5]
            if [p['epoch'] for p in row['curve']] != expected_epochs: raise ValueError('Missing epoch.')
            for value in [row['initial_test']] + [p['test'] for p in row['curve']]:
                if value['examples'] != (128 if smoke else 10000): raise ValueError('Wrong test cohort.')
                if any(not math.isfinite(value[k]) for k in ('teacher_kl', 'accuracy_percent', 'teacher_agreement_percent')):
                    raise ValueError('Nonfinite metric.')
            for point in row['curve']:
                cost = point['cost']; epoch = point['epoch']
                if point['image_presentations'] != epoch * (128 if smoke else 45000): raise ValueError('Wrong training cohort.')
                if cost['max_recovery_cell_pulses'] > epoch * (2 if smoke else 704) or cost['verify_reads'] or cost['capped_cells']:
                    raise ValueError('One-pulse open-loop contract violated.')
                if row['method'] == 'calibration' and cost['physical_pulses']: raise ValueError('Calibration wrote weights.')
    for case in {r['case'] for r in rows}:
        if len({json.dumps(r['identity'], sort_keys=True) for r in rows if r['case'] == case}) != 1:
            raise ValueError('Sources do not share physical identities.')
    return rows


def report(base, output, datasets=('cifar10', 'cifar100'), *, artifact_index=None):
    base, output = Path(base), Path(output); output.mkdir(parents=True, exist_ok=True)
    state = json.loads((base / 'execution-v1/campaign-execution.json').read_text())
    inputs = json.loads((base / 'inputs.json').read_text())
    flat, paired, costs, sources, provenance = [], [], [], [], []
    controls = 0
    def endpoint(row): return row['curve'][-1]['test'] if row['curve'] else row['initial_test']
    for dataset in datasets:
        for depth in (4, 8):
            for initialization, tag in (('saved_p0', 'paired'), ('reset_open_loop', 'reset')):
                for seed in (251001, 251002, 251003):
                    key = f'{dataset}-om-conv{depth}-openloop-{tag}-v1__screen{seed - 251000}'
                    entry = state['completed'][key]
                    path = (artifact_index.resolve(entry['result_sha256'], 'result.json')
                            if artifact_index else Path(entry['run']) / 'result.json')
                    if sha256_file(path) != entry['result_sha256']: raise ValueError('Collected result changed.')
                    result = json.loads(path.read_text())['metrics']; rows = validate(result); controls += len(rows)
                    provenance.append({'path': str(path), 'sha256': entry['result_sha256']})
                    baseline = inputs[f'{dataset}_{depth}']['arrays'][str(seed)]
                    oldpath = (artifact_index.resolve(baseline['baseline_result_sha256'], 'result.json')
                               if artifact_index else Path(baseline['baseline_run']) / 'result.json')
                    if sha256_file(oldpath) != baseline['baseline_result_sha256']: raise ValueError('V2 reference changed.')
                    old = {(r['source'], r['case'], r['method']): r for r in json.loads(oldpath.read_text())['metrics']['measurements']}
                    by = {(r['source'], r['case'], r['method']): r for r in rows}
                    for source, fit in result['source_fits'].items():
                        sources.append({'dataset': dataset, 'depth': depth, 'initialization': initialization,
                                        'array_seed': seed, 'source': source, 'fit': fit})
                    for row in rows:
                        common = {'dataset': dataset, 'depth': depth, 'initialization': initialization,
                                  'array_seed': seed, 'source': row['source'], 'case': row['case'], 'method': row['method']}
                        for point in row['curve'] or [{'epoch': 0, 'test': row['initial_test']}]:
                            flat.append({**common, 'epoch': point['epoch'], **point['test']})
                        if row['curve']: costs.append({**common, **row['curve'][-1]['cost']})
                        if not row['method'].startswith('onchip'): continue
                        current = endpoint(row)
                        references = [('openloop_calibration', by[(row['source'], row['case'], 'calibration')])]
                        if initialization == 'saved_p0':
                            for method in (row['method'], 'rewrite'):
                                ref = old[(row['source'], row['case'], method)]
                                if ref['p0_sha256'] != row['inherited_p0_sha256'] or ref['identity'] != row['identity']:
                                    raise ValueError('V2 comparison is not paired.')
                                references.append(('v2_' + method, ref))
                        for label, ref in references:
                            control = endpoint(ref)
                            paired.append({**common, 'reference': label,
                                           'accuracy_gain_pp': current['accuracy_percent'] - control['accuracy_percent'],
                                           'kl_improvement': control['teacher_kl'] - current['teacher_kl']})
    keys = ('dataset', 'depth', 'initialization', 'source', 'case', 'method')
    means = aggregate(flat, keys + ('epoch',), ('accuracy_percent', 'teacher_kl', 'teacher_agreement_percent'))
    comparisons = aggregate(paired, keys + ('reference',), ('accuracy_gain_pp', 'kl_improvement'))
    for name, rows in (('curves', flat), ('means', means), ('paired', paired), ('paired_means', comparisons), ('costs', costs)):
        csv_write(output / (name + '.csv'), rows)
    result = {'controls': controls, 'datasets': list(datasets), 'provenance': provenance,
              'source_fits': sources, 'means': means, 'paired_means': comparisons,
              'scientific_review': 'pending', 'teacher_kl': 'KL(teacher || student), nats, T=1'}
    atomic_write_json(output / 'summary.json', result)
    return result
