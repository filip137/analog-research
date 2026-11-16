"""Check one drift-tuning bundle against its config: arms, grid, projections, identities, finiteness."""
import json
import math
from pathlib import Path
import sys


def main(config_path, run_dir):
    c = json.loads(Path(config_path).read_text())
    f, d = c['fault_tuning'], c['drift_tuning']
    s = json.loads((Path(run_dir) / 'artifacts/summary.json').read_text())
    i = s['identity']
    for key, want in (('recovery_sha256', f['recovery_sha256']), ('development_sha256', f['development_sha256']),
                      ('samples_sha256', f['samples_sha256'])):
        if i[key] != want:
            raise SystemExit(f'Identity mismatch: {key}')
    if i['mapping']['scale'] != d['mapping_scale'] or i['drift_tuning']['multiplier'] != d['multiplier']:
        raise SystemExit('Mapping or drift multiplier differs from the config.')
    arms = {a['label']: a for a in s['arms']}
    if sorted(arms) != sorted(a['label'] for a in f['arms']):
        raise SystemExit('Arm coverage differs from the config.')
    for a in arms.values():
        if [g['age_seconds'] for g in a['grid']] != d['evaluation_seconds']:
            raise SystemExit(f"Grid coverage differs in {a['label']}.")
        values = [g['teacher_kl'] for g in a['grid']] + [a['mean_teacher_kl']]
        if a['method'] == 'noisy':
            if a['updates'] != f['steps'] or [p['step'] for p in a['projections']] != d['projection_steps']:
                raise SystemExit(f"Updates or projections differ in {a['label']}.")
            values += [p['teacher_kl'] for p in a['projections'] + a['curve']] + [a['final_no_drift']['teacher_kl']]
        if not all(math.isfinite(v) for v in values):
            raise SystemExit(f"Nonfinite KL in {a['label']}.")
    print(json.dumps({'verified': True, 'mean_teacher_kl': {k: a['mean_teacher_kl'] for k, a in arms.items()}}))


if __name__ == '__main__':
    main(*sys.argv[1:])
