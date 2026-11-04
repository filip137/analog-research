"""Apply the frozen exp-002 tuning rule to collected run-1 summaries.

Usage: python campaigns/drift-regimes/launch/select_rates.py <out.json> <summary.json> [...]
  (group summaries.json lists or per-unit drift_regimes_summary.json files)

Rule (exp-002 note): per method, the learning rate whose arm has the lowest
24-h mean teacher KL on the tuning unit (R0, realization 271004, validation
sequences 64-127). A3-lr* selects the LayerNorm rate, A5-lr* the LoRA rate.
A selected rate at the edge of its grid is flagged, not extended.
"""
import json
from pathlib import Path
import sys


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    arms = {}
    for path in paths:
        loaded = json.loads(Path(path).read_text())
        for summary in (loaded if isinstance(loaded, list) else [loaded]):
            if summary['realization'] == 271004:
                for label, arm in summary['arms'].items():
                    if label in arms and arms[label]['mean_teacher_kl'] != arm['mean_teacher_kl']:
                        raise ValueError(f'Conflicting tuning arm {label}.')
                    arms[label] = arm
    if 'A0' not in arms:
        raise ValueError('Expected the tuning A0 reference.')
    record = {'rule': 'lowest 24-h mean teacher KL per method; R0, realization 271004, validation 64-127',
              'reference_A0': arms['A0']['mean_teacher_kl'], 'methods': {}}
    for prefix, method in (('A3-lr', 'layernorm'), ('A5-lr', 'lora')):
        grid = sorted((arm['settings']['learning_rate'], arm['mean_teacher_kl'])
                      for label, arm in arms.items() if label.startswith(prefix))
        if not grid:
            raise ValueError(f'No {method} tuning arms.')
        best = min(grid, key=lambda x: (x[1], x[0]))
        record['methods'][method] = {'grid': grid, 'selected': best[0],
                                     'at_grid_edge': best[0] in (grid[0][0], grid[-1][0])}
    Path(out).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
