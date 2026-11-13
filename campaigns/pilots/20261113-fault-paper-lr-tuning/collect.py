"""Tabulate the DRN fault-paper LR screen and apply the frozen selection rule.

Usage: python collect.py <output root> [--json OUT]

Reads every ``<output root>/<config>/run/artifacts/arm-*.json``. Selection per arm family
(method, and rank for LoRA): the lowest final development KL; candidates within 1 % of it
go to the fewest write attempts (LoRA: no writes, so the lowest KL). Grid-edge picks are flagged.
"""
import argparse
import json
from pathlib import Path

GRIDS = {'noisy': (1e-4, 1e-2), 'noiseless': (1e-4, 1e-2), 'child_d': (1e-4, 1e-2), 'lora': (1e-3, 1e-2)}


def rows(root):
    out = []
    for path in sorted(Path(root).glob('*/run/artifacts/arm-*.json')):
        r = json.loads(path.read_text())
        costs = r.get('costs', {})
        out.append({'config': path.parts[-4], 'label': r['label'], 'method': r['method'],
                    'rank': r.get('rank'), 'learning_rate': r.get('learning_rate'),
                    'dev_kl': r['development']['teacher_kl'], 'perplexity': r['development']['perplexity'],
                    'curve': [round(p['teacher_kl'], 5) for p in r.get('curve', [])],
                    'write_attempts': costs.get('write_attempts', 0), 'healthy_writes': r.get('healthy_writes'),
                    'stuck_write_attempts': r.get('stuck_write_attempts'), 'touched_cells': r.get('touched_cells'),
                    'max_writes_per_cell': r.get('max_writes_per_cell'),
                    'out_of_range_devices': r.get('out_of_range_devices')})
    return out


def select(table):
    chosen = {}
    families = {}
    for r in table:
        if r['learning_rate'] is not None:
            families.setdefault(r['method'] + ('' if r['rank'] is None else str(r['rank'])), []).append(r)
    for family, members in sorted(families.items()):
        best = min(m['dev_kl'] for m in members)
        tied = [m for m in members if m['dev_kl'] <= 1.01 * best]
        pick = min(tied, key=lambda m: (m['write_attempts'], m['dev_kl']))
        low, high = GRIDS[pick['method']]
        chosen[family] = {'label': pick['label'], 'learning_rate': pick['learning_rate'], 'dev_kl': pick['dev_kl'],
                          'tied_within_1pct': [m['label'] for m in tied],
                          'grid_edge': pick['learning_rate'] in (low, high), 'rates_seen': len(members)}
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    table = rows(args.root)
    for r in table:
        print(f"{r['label']:<22} KL {r['dev_kl']:.5f}  curve {r['curve']}  attempts {r['write_attempts']}  "
              f"healthy {r['healthy_writes']}  stuck {r['stuck_write_attempts']}  touched {r['touched_cells']}  "
              f"max/cell {r['max_writes_per_cell']}  out-of-range {r['out_of_range_devices']}")
    chosen = select(table)
    print(json.dumps(chosen, indent=1))
    if args.json:
        args.json.write_text(json.dumps({'arms': table, 'selection': chosen}, indent=1) + '\n')


if __name__ == '__main__':
    main()
