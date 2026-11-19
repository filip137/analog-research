"""Check one cmo-drift-recovery bundle against its config.

Arms, grid, projections, identities and finiteness (as the drift token-budget check), plus
the exp-001 semantics: the fixed 1.12 uS gate, reset-absorb write clocks (one reset per
written device), Child-Tuning_D writes only inside its shared mask, digital arms leave the
devices untouched, every arm starts at A0's release KL, A0-twin and drift-off arms are
flagged, zero-write drifting learners replay A0, and the per-model clean references.

    python verify_unit.py CONFIG RUN_DIR
"""
import json
import math
from pathlib import Path
import sys

# Exact trained DRN sources (cmo-drift-recovery exp-001 inputs).
SOURCE_PINS = {6: 'dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9',
               12: '6805b62950743838de89ed356cbaf5e92a98710cf87c3f7b11093be89c4a8a1a'}
# 2 sigma(MID_US) of the AIHWKit 2 % CMO fit, in uS (hardware.devices.cmo).
FIXED_GATE_US = 2 * (0.0112185391 + 0.01129027418 * (88.199997 + 9.0) / 2)
HALF_RANGE_US = (88.199997 - 9.0) / 2
PHYSICAL = ('noisy', 'child_d')
DIGITAL = ('calibration', 'layernorm')


def fail(message):
    raise SystemExit(message)


def finite(values, label):
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        fail(f'Nonfinite KL in {label}.')


def verify(config_path, run_dir, source_pins=SOURCE_PINS):
    c = json.loads(Path(config_path).read_text())
    f, d = c['fault_tuning'], c['drift_tuning']
    s = json.loads((Path(run_dir) / 'artifacts/summary.json').read_text())
    i = s['identity']
    for key, want in (('recovery_sha256', f['recovery_sha256']), ('development_sha256', f['development_sha256']),
                      ('samples_sha256', f['samples_sha256'])):
        if i[key] != want:
            fail(f'Identity mismatch: {key}')
    if source_pins is not None and i['source_weights_sha256'] != source_pins[c['analog_layers']]:
        fail('Source DRN weights differ from the pinned trained model.')
    if i['mapping']['scale'] != d['mapping_scale'] or i['drift_tuning']['multiplier'] != d['multiplier']:
        fail('Mapping or drift multiplier differs from the config.')
    if (i['write_clock'], i['write_gate'], i['drift_off_arms']) != (d['write_clock'], d['write_gate'],
                                                                    sorted(d['drift_off_arms'])):
        fail('Write clock, gate or drift-off arms differ from the config.')
    if d['write_gate'] == 'fixed_mid' and not math.isclose(i['write_gate_us'], FIXED_GATE_US, rel_tol=1e-6):
        fail(f"Write gate {i['write_gate_us']} uS is not 2 sigma(48.6 uS) = {FIXED_GATE_US} uS.")
    clean = i['clean_reference']
    if set(clean) != {'source_ideal', 'mapped_targets'}:
        fail('Missing clean references.')
    finite(list(clean.values()), 'clean references')
    off = set(d['drift_off_arms'])
    declared = {a['label']: a for a in f['arms']}
    arms = {a['label']: a for a in s['arms']}
    if sorted(arms) != sorted(declared) or any(arms[k]['method'] != declared[k]['method'] for k in arms):
        fail('Arm coverage differs from the config.')
    holds = [k for k, a in arms.items() if a['method'] == 'hold' and k not in off]
    if len(holds) != 1:
        fail('Expected one drifting hold arm.')
    a0 = arms[holds[0]]
    release = a0['grid'][0]['teacher_kl']
    a0_grid = {g['age_seconds']: g['teacher_kl'] for g in a0['grid']}
    fisher = s.get('child_d_fisher')
    if d.get('child_mask_scope') == 'global':
        if not fisher or fisher.get('scope') != 'global' or fisher['fraction'] != f['child_fraction']:
            fail('Global Child-Tuning_D mask configuration mismatch.')
        devices = sum(row['devices'] for row in fisher['arrays'])
        selected = sum(row['selected'] for row in fisher['arrays'])
        if selected != max(1, round(devices * f['child_fraction'])):
            fail('Global Child-Tuning_D mask count differs from its reserve fraction.')
    for label, a in arms.items():
        method, drift = a['method'], label not in off
        if [g['age_seconds'] for g in a['grid']] != d['evaluation_seconds']:
            fail(f'Grid coverage differs in {label}.')
        values = [g['teacher_kl'] for g in a['grid']] + [a['mean_teacher_kl']]
        if a['grid'][0]['teacher_kl'] != release:
            fail(f'{label} does not start at the A0 release KL.')
        if (a.get('drift', True) is False) == drift:
            fail(f'Drift flag of {label} differs from drift_off_arms.')
        for key in ('writes_per_device', 'fraction_written', 'at_drift_min_fraction_final'):
            if key not in a:
                fail(f'Missing {key} in {label}.')
        if (a['at_drift_min_fraction_final'] is None) == drift:
            fail(f'Floor fraction at the last age is inconsistent with the drift state of {label}.')
        if method == 'hold':
            if not drift and any(g['teacher_kl'] != release for g in a['grid']):
                fail(f'A0-twin {label} is not the constant released state.')
            if a['costs']['write_attempts']:
                fail(f'Hold arm {label} wrote devices.')
        else:
            if a['updates'] != f['steps']:
                fail(f'Updates differ in {label}.')
            values += [p['teacher_kl'] for p in a['projections'] + a['curve']] + [a['final_no_drift']['teacher_kl']]
            projections = [p['step'] for p in a['projections']]
            if method in PHYSICAL and drift:
                if projections != d['projection_steps']:
                    fail(f'Projections differ in {label}.')
            elif projections:
                fail(f'Unexpected projections in {label}.')
        if method in PHYSICAL:
            if a['write_gate'] != d['write_gate']:
                fail(f'Write gate differs in {label}.')
            writers = json.loads((Path(run_dir) / f'artifacts/arm-{label}.json').read_text())['writers']
            if d['write_gate'] == 'fixed_mid' and any(
                    w['threshold_reference'] != 'nominal_midrange_write_std'
                    or not math.isclose(w['threshold_q'] * HALF_RANGE_US, FIXED_GATE_US, rel_tol=1e-6)
                    for w in writers):
                fail(f'{label} did not use the fixed 2 sigma(48.6 uS) gate.')
            clocked = drift and d['write_clock'] == 'reset_absorb'
            if clocked != ('write_clock' in a):
                fail(f'Write-clock record of {label} is inconsistent with its drift state.')
            if clocked and a['write_clock']['reset_devices'] != a['touched_cells']:
                fail(f'{label}: clock resets differ from written devices.')
            if method == 'child_d' and (a['writes_outside_mask'] or fisher is None
                                        or a['mask_sha256'] != fisher['mask_sha256']):
                fail(f'{label}: Child-Tuning_D mask violated or not shared.')
            if drift and a['costs']['write_attempts'] == 0:
                replay = a['zero_write_replay']
                if not replay['checked'] or replay['mismatched_ages']:
                    fail(f'{label}: zero-write learner did not replay A0.')
                if any(g['teacher_kl'] != a0_grid[g['age_seconds']] for g in a['grid']):
                    fail(f'{label}: zero-write learner grid differs from A0.')
        if method in DIGITAL:
            if a['costs']['write_attempts'] != 0 or a['final_held_hashes'] != i['start_held_hashes']:
                fail(f'Digital arm {label} changed the devices.')
            if a['trained_digital_parameters'] <= 0:
                fail(f'Digital arm {label} trained nothing.')
        finite(values, label)
    return {'verified': True, 'release_kl': release, 'clean_reference': clean,
            'arms': {k: {'mean_teacher_kl': a['mean_teacher_kl'], 'final_kl': a['grid'][-1]['teacher_kl'],
                         'writes_per_device': a['writes_per_device'], 'fraction_written': a['fraction_written']}
                     for k, a in arms.items()}}


def main(config_path, run_dir):
    print(json.dumps(verify(config_path, run_dir)))


if __name__ == '__main__':
    main(*sys.argv[1:])
