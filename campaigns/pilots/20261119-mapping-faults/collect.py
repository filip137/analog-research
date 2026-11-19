"""Tabulate collected mapping-faults cells (compact copies of the native bundles).

    python collect.py COLLECTED_ROOT   # <root>/<cell>/{deploy,tune}/... -> cells.csv, cells.json
Each cell directory holds deploy/deployment_report.json and tune/summary.json (plus the
native result/manifest files and cell-verified.json), copied after the on-host checks.
"""
import csv
import json
from pathlib import Path
import sys


def row(cell_dir):
    report = json.loads((cell_dir / 'deploy/deployment_report.json').read_text())
    summary = json.loads((cell_dir / 'tune/summary.json').read_text())
    i, r = summary['identity'], report['report']
    model, mapping, rate, seed = cell_dir.name.split('-')
    out = {'cell': cell_dir.name, 'model': model, 'mapping': mapping, 'scale': i['mapping']['scale'],
           'rate_pct': int(rate[1:]), 'map': int(seed),
           'source_ideal_dev': i['clean_reference']['source_ideal'],
           'mapped_targets_dev': i['clean_reference']['mapped_targets'],
           'target_saturated_fraction': i['mapping']['saturated_fraction']}
    for arm in summary['arms']:
        key = 'naive' if arm['method'] == 'naive_hold' else 'fa'
        out[f'{key}_dev'] = arm['development']['teacher_kl']
        for name, score in arm['held_out'].items():
            out[f'{key}_{name}'] = score['metrics']['teacher_kl']
    out.update({k: r['receipt'][k] for k in ('faults', 'flagged', 'rewrite_attempts', 'stuck_rewrite_attempts',
                                               'clipped_targets', 'node_loading_change_l1_us')})
    out['devices'] = r['devices']
    for k in ('target_fraction_at_gmax', 'held_fraction_at_or_above_gmax', 'held_fraction_at_or_below_gmin',
              'raw_fraction_at_or_above_gmax', 'raw_fraction_at_or_below_gmin'):
        out[k] = r[k]
    s = r['denominators']['summary']
    for label in ('fault_aware', 'naive'):
        for d in ('dh', 'do'):
            out[f'{label}_{d}_median'] = s[f'{label}_{d}']['median_of_layer_medians']
            out[f'{label}_{d}_p95max'] = s[f'{label}_{d}']['max_layer_p95']
    out['layers'] = r['denominators']['layers']
    out['seconds_tune'] = sum(a['seconds'] for a in summary['arms'])
    return out


def main(root):
    root = Path(root)
    rows = [row(d) for d in sorted(root.iterdir()) if (d / 'tune/summary.json').is_file()]
    order = {'base': 0, 'ratio2': 1, 'ratio3': 2}
    rows.sort(key=lambda x: (x['model'] != 'drn6', x['map'], order[x['mapping']], x['rate_pct']))
    (root / 'cells.json').write_text(json.dumps(rows, indent=1) + '\n')
    flat = [{k: v for k, v in x.items() if k != 'layers'} for x in rows]
    with open(root / 'cells.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader(); w.writerows(flat)
    for x in flat:
        print(f"{x['cell']:26s} naive {x['naive_dev']:.4f}/{x['naive_wikitext']:.4f}/{x['naive_generated']:.4f} "
              f"FA {x['fa_dev']:.4f}/{x['fa_wikitext']:.4f}/{x['fa_generated']:.4f} "
              f"dh {x['fault_aware_dh_median']:.4f}/{x['fault_aware_dh_p95max']:.4f}")


if __name__ == '__main__':
    main(sys.argv[1])
