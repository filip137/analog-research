"""Tabulate exp-002 drift units (and the reused exp-001 HWA ratio3 controls).

    python collect_drift.py OUT.json RUN_DIR [RUN_DIR ...]
Per unit: release (A0 t=0), A0 / A0-twin / CAL 24-h trapezoid-mean KL, drift damage
(A0 mean - twin mean), CAL gain, A0 24-h KL, fractions at the 5 and 150 uS bounds at 24 h,
clean references, mapping saturation and start hashes.
"""
import json
from pathlib import Path
import sys


def unit(run):
    s = json.loads((Path(run) / 'artifacts/summary.json').read_text())
    i, arms = s['identity'], {a['label']: a for a in s['arms']}
    a0, twin, cal = arms['A0'], arms['A0-twin'], arms['CAL']
    last = a0['grid'][-1]
    bounds = last.get('bounds') or {}
    return {'run': str(run), 'mapping_scale': i['mapping']['scale'],
            'saturated_fraction': i['mapping']['saturated_fraction'],
            'source_weights_sha256': i['source_weights_sha256'], 'start_held_hashes': i['start_held_hashes'],
            'clean_reference': i['clean_reference'], 'release_kl': a0['grid'][0]['teacher_kl'],
            'a0_grid': [g['teacher_kl'] for g in a0['grid']],
            'a0_mean': a0['mean_teacher_kl'], 'twin_mean': twin['mean_teacher_kl'], 'cal_mean': cal['mean_teacher_kl'],
            'a0_24h': last['teacher_kl'], 'cal_24h': cal['grid'][-1]['teacher_kl'],
            'drift_damage': a0['mean_teacher_kl'] - twin['mean_teacher_kl'],
            'cal_gain': a0['mean_teacher_kl'] - cal['mean_teacher_kl'],
            'at_5us_24h': bounds.get('at_drift_min_fraction'), 'at_150us_24h': bounds.get('at_drift_max_fraction')}


def main(out, *runs):
    rows = {Path(r).parent.name: unit(r) for r in runs}
    Path(out).write_text(json.dumps(rows, indent=1) + '\n')
    for k, r in rows.items():
        print(f"{k:28s} rel {r['release_kl']:.4f} A0 {r['a0_mean']:.4f} twin {r['twin_mean']:.4f} CAL {r['cal_mean']:.4f} "
              f"dmg {r['drift_damage']:.4f} 5uS {r['at_5us_24h']} 150uS {r['at_150us_24h']}")


if __name__ == '__main__':
    main(*sys.argv[1:])
