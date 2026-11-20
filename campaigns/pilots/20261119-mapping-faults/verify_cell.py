"""Check one mapping-faults cell (deploy bundle + hold-only fault_tuning bundle) against its configs.

Runs after ``ebl runs inspect --verify-artifacts --require-complete`` on both bundles:
source pin, recorded mapping scale and bias currents, the tune run restored exactly the
deploy run's deployed.pt, both holds present with finite dev KL, full held-out coverage of
every declared cohort, clean references, and the deployment report (denominators, devices).

    python verify_cell.py DEPLOY_CONFIG TUNE_CONFIG DEPLOY_RUN TUNE_RUN
"""
import hashlib
import json
import math
from pathlib import Path
import sys

SOURCE_PINS = {6: 'dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9',
               12: '6805b62950743838de89ed356cbaf5e92a98710cf87c3f7b11093be89c4a8a1a'}
RECOVERY_HOLD_KL = {(6, 3): 0.06456173050637339, (12, 2): 0.255741274356842,
                    (12, 3): 0.30084083502491316}


def fail(message):
    raise SystemExit('verify_cell: ' + message)


def finite(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f'Nonfinite {label}.')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(deploy_config, tune_config, deploy_run, tune_run):
    dc, tc = (json.loads(Path(p).read_text()) for p in (deploy_config, tune_config))
    deploy_run, tune_run = Path(deploy_run), Path(tune_run)
    if dc['deployment'] != tc['deployment'] or 'mapping_scale' not in dc['deployment']:
        fail('Deploy and tune configs must declare one explicit mapped deployment.')
    for run in (deploy_run, tune_run):
        if json.loads((run / 'result.json').read_text())['status'] != 'complete':
            fail(f'{run} is not complete.')
    scale = dc['deployment']['mapping_scale']
    report = json.loads((deploy_run / 'artifacts/deployment_report.json').read_text())
    if report['deployment'] != dc['deployment'] or report['mapping']['scale'] != scale:
        fail('Deployment report identity differs from the config.')
    r = report['report']
    rate = dc['deployment']['fault_rate']
    if (rate == 0 and r['receipt']['faults'] != 0) or abs(r['receipt']['faults'] / r['devices'] - rate) > 2e-3:
        fail('Stuck count differs from rate x devices.')
    for row in r['denominators']['layers']:
        for key in ('fault_aware_dh', 'fault_aware_do', 'naive_dh', 'naive_do'):
            finite(row[key]['p95'], key)
    s = json.loads((tune_run / 'artifacts/summary.json').read_text())
    i = s['identity']
    if i['deployed_sha256'] != digest(deploy_run / 'checkpoints/deployed.pt'):
        fail('Tune run did not restore this cell\'s deployed.pt.')
    if i['source_weights_sha256'] != SOURCE_PINS[tc['analog_layers']]:
        fail('Source DRN weights differ from the pinned HWA model.')
    if i['deployment'] != tc['deployment'] or i['mapping'] != report['mapping'] or i['current_scales'] != report['current_scales']:
        fail('Restored mapping or bias currents differ from the deployment.')
    if set(i['clean_reference']) != {'source_ideal', 'mapped_targets'}:
        fail('Missing clean references.')
    for k, v in i['clean_reference'].items():
        finite(v, k)
    if scale == 1 and i['clean_reference']['source_ideal'] != i['clean_reference']['mapped_targets']:
        fail('Scale 1 changed the mapped targets.')
    arms = s['arms']
    if [(a['label'], a['method']) for a in arms] != [(a['label'], a['method']) for a in tc['fault_tuning']['arms']]:
        fail('Arms differ from the config.')
    cohorts = tc['fault_tuning']['evaluation_cohorts']
    settings = tc['fault_tuning']
    matched_recovery = settings.get('write_gate') == 'fixed_mid'
    if matched_recovery:
        if i['write_gate'] != 'fixed_mid' or i['child_mask_scope'] != 'global':
            fail('Recovery did not use the declared gate and mask.')
        for key in ('recovery_sha256', 'development_sha256', 'samples_sha256'):
            if i[key] != settings[key]:
                fail(f'Recovery identity mismatch for {key}.')
        masks = s['child_d_fisher']
        selected = sum(x['selected'] for x in masks['arrays'])
        if selected != round(0.01 * (r['devices'] - r['receipt']['faults'])):
            fail('Incorrect global mask population.')
        if rate == 0.03 and tc['deployment']['assignment_seed'] == 271002:
            observed = arms[0]['held_out']['wikitext']['metrics']['teacher_kl']
            expected = RECOVERY_HOLD_KL[tc['analog_layers'], scale]
            if not math.isclose(observed, expected, abs_tol=1e-6, rel_tol=0):
                fail(f'3% mapped hold replay differs: {observed} vs {expected}.')
    for a in arms:
        finite(a['development']['teacher_kl'], 'dev KL')
        if a['method'] in ('naive_hold', 'hold', 'calibration', 'lora'):
            if a['costs']['write_attempts'] != 0:
                fail('A zero-write arm wrote devices.')
        if matched_recovery:
            if a['initial_held_hashes'] != i['start_held_hashes']:
                fail('Arm did not start from the shared deployment.')
            if a['method'] not in ('hold', 'naive_hold'):
                if a['updates'] != settings['steps'] or a['curve'][-1]['step'] != settings['steps']:
                    fail('Incomplete training arm.')
                for point in a['curve']:
                    finite(point['teacher_kl'], 'training curve KL')
            if a['method'] == 'child_d' and a['writes_outside_mask'] != 0:
                fail('Child-Tuning wrote outside its mask.')
        if set(a['held_out']) != {c['name'] for c in cohorts}:
            fail('Missing held-out cohorts.')
        for c in cohorts:
            score = a['held_out'][c['name']]
            if score['cohort'] != c or score['metrics']['scored_tokens'] != c['sequences'] * (tc['fault_tuning']['sequence_length'] - 1):
                fail('Held-out identity or coverage mismatch.')
            finite(score['metrics']['teacher_kl'], c['name'])
    return {'schema': 'mapping_faults.cell.v1', 'deploy_run': str(deploy_run.resolve()),
            'tune_run': str(tune_run.resolve()), 'status': 'verified'}


if __name__ == '__main__':
    result = verify(*sys.argv[1:5])
    (Path(sys.argv[4]) / 'cell-verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
