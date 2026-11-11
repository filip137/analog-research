"""Select the LoRA rate per sample count (recovery-data-preparation exp-002) from collected tuning summaries.

Usage: python campaigns/drift-regimes/launch/select_sample_rates.py <out.json> <summary.json> [...]

Rule (exp-001's): per count, the rate with the lowest 24-h mean teacher KL on R0-G realization 271004,
validation sequences 64-127; an edge rate is flagged, not extended. Gate: every unit's samples:8 arm
at 1e-2 equals exp-001's tuning value bit for bit (same fixed draw); otherwise the script refuses.
"""
import json
from pathlib import Path
import sys

EXP001_SAMP8 = 0.6914967786074333
COUNTS = (1, 2, 4)


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    arms, gate = {}, []
    for path in paths:
        loaded = json.loads(Path(path).read_text())
        for summary in (loaded if isinstance(loaded, list) else [loaded]):
            if summary['realization'] != 271004:
                continue
            for label, arm in summary['arms'].items():
                if label == 'T-samp8-lr0.01':
                    gate.append(arm['mean_teacher_kl'])
                    continue
                if label in arms and arms[label]['mean_teacher_kl'] != arm['mean_teacher_kl']:
                    raise ValueError(f'Conflicting tuning arm {label}.')
                arms[label] = arm
    if len(gate) != len(COUNTS) or any(v != EXP001_SAMP8 for v in gate):
        raise SystemExit(f'Reproduction gate failed: samples:8 = {gate}, exp-001 {EXP001_SAMP8}.')
    record = {'rule': 'lowest 24-h mean teacher KL per count; R0-G, realization 271004, validation 64-127',
              'reproduction_gate': {'exp001_samples8': EXP001_SAMP8, 'now': gate, 'pass': True},
              'counts': {}, 'rates': {}}
    for n in COUNTS:
        prefix = f'T-samp{n}-lr'
        grid = sorted((arm['settings']['learning_rate'], arm['mean_teacher_kl'])
                      for label, arm in arms.items() if label.startswith(prefix))
        if len(grid) != 3:
            raise ValueError(f'Expected three tuning rates for samples:{n}.')
        best = min(grid, key=lambda x: (x[1], x[0]))
        record['counts'][f'n{n}'] = {'grid': grid, 'selected': best[0],
                                     'at_grid_edge': best[0] in (grid[0][0], grid[-1][0])}
        record['rates'][f'n{n}'] = best[0]
    Path(out).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
