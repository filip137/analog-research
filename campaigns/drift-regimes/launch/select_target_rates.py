"""Select the LoRA rate per training target (recovery-data-preparation exp-001) from collected tuning summaries.

Usage: python campaigns/drift-regimes/launch/select_target_rates.py <out.json> <summary.json> [...]
  (group summaries.json lists or per-unit drift_regimes_summary.json files)

Rule (exp-004's, per target): the learning rate whose arm has the lowest 24-h mean teacher KL on
R0-G realization 271004, validation sequences 64-127. A5-lr* arms are the kl target (exp-004's arm),
T-<name>-lr* the others. A selected rate at the edge of the grid is flagged, not extended.
Gate: the kl arms at 3e-3 and 1e-2 and A0 must reproduce exp-004's tuning (0.630, 0.611, 4.0747
as recorded to three or four decimals); otherwise the script refuses.
"""
import json
from pathlib import Path
import sys

EXP004 = {'A0': (4.0747, 5e-5), 'A5-lr0.003': (0.630, 5e-4), 'A5-lr0.01': (0.611, 5e-4)}
NAMES = ('kl', 'text', 'top8', 'top32', 'samp8')


def target(label):
    if label.startswith('A5-lr'):
        return 'kl'
    if label.startswith('T-') and '-lr' in label:
        return label[2:label.rindex('-lr')]
    return None


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    arms = {}
    for path in paths:
        loaded = json.loads(Path(path).read_text())
        for summary in (loaded if isinstance(loaded, list) else [loaded]):
            if summary['realization'] != 271004:
                continue
            for label, arm in summary['arms'].items():
                if label in arms and arms[label]['mean_teacher_kl'] != arm['mean_teacher_kl']:
                    raise ValueError(f'Conflicting tuning arm {label}.')
                arms[label] = arm
    gate = {}
    for label, (value, tolerance) in EXP004.items():
        got = arms[label]['mean_teacher_kl']
        gate[label] = {'exp004': value, 'now': got, 'pass': abs(got - value) <= tolerance}
    if not all(g['pass'] for g in gate.values()):
        raise SystemExit(f'Reproduction gate failed: {json.dumps(gate)}')
    record = {'rule': 'lowest 24-h mean teacher KL per target; R0-G, realization 271004, validation 64-127',
              'reproduction_gate': gate, 'targets': {}, 'rates': {}}
    for name in NAMES:
        grid = sorted((arm['settings']['learning_rate'], arm['mean_teacher_kl'])
                      for label, arm in arms.items() if target(label) == name)
        if not grid:
            raise ValueError(f'No tuning arms for target {name}.')
        best = min(grid, key=lambda x: (x[1], x[0]))
        record['targets'][name] = {'grid': grid, 'selected': best[0],
                                   'at_grid_edge': best[0] in (grid[0][0], grid[-1][0])}
        record['rates'][name] = best[0]
    Path(out).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
