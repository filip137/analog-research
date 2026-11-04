"""Write a Jean Zay manifest for one exp-002 screen run, its units packed into GPU groups.

Usage: python campaigns/drift-regimes/launch/make_screen_manifest.py <run> <groups> <source dir> <output root> <out.json>
  run: canary | run1 (tuning + part w) | run2 (part d)

Units (lifecycles) are packed greedily, longest first, into <groups> groups by
an estimated V100 time; each group is one dev-queue job running its units in
turn. Estimates (seconds): 210 setup per unit, 40 evaluation per arm, 35 per
digital training event, 30 per full reset pass (times the expected resets per
device at N) and 6 per selective-rewrite event. Run-1 measurements refine the
run-2 estimates.
"""
import json
from pathlib import Path
import sys

from make_manifest import JZ, ROOT

RESETS = {1: 1.0, 2: 1.35, 4: 1.76, 8: 2.6}
SECONDS = {'setup': 210, 'evaluation': 40, 'training': 35, 'reset_pass': 30, 'selective': 6,  # training: V100 canary, 32.5 s
           'learning_round': 12}                                                             # learning: refined by the run-3 canary


def estimate(doc):
    spec, s = doc['drift_regimes'], SECONDS
    total = s['setup']
    for arm in spec['arms']:
        total += s['evaluation']
        events = len(arm.get('events', ()))
        if arm['method'] in ('layernorm', 'lora', 'lora_oracle', 'lora_rescaled', 'lora_clean'):
            total += events * s['training'] * (1.5 if arm.get('rank', 16) > 16 else 1)
        elif arm['method'] in ('rewrite', 'precompensated', 'rewrite_oracle'):
            total += events * (s['reset_pass'] * RESETS[arm['attempts']] if arm['fraction'] >= 1 else s['selective'])
        elif arm['method'] == 'learning':
            total += events * arm['rounds'] * s['learning_round']
    return total


def units(run):
    paths = sorted((ROOT / 'campaigns/drift-regimes/lifecycles').glob('opt125m-drift-regimes-screen-*.json'))
    pick = {'canary': lambda n: '-canary-' in n,
            'run1': lambda n: n.endswith('-w') or n.endswith('-tune-ln') or n.endswith('-tune-lora'),
            'run2': lambda n: n.endswith('-d'),
            'canary2': lambda n: '-canary2-' in n,
            'tune3': lambda n: n.endswith('-tune-a11'),
            'tune3b': lambda n: n.endswith('-tune-a11'),   # rerun with the lower step grid (tune3 cancelled)
            'run3': lambda n: n.endswith('-x')}[run]
    return [p for p in paths if pick(p.stem)]


def pack(paths, groups):
    jobs = sorted(((estimate(json.loads(p.read_text())), p) for p in paths), key=lambda x: (-x[0], x[1].name))
    bins = [[0., []] for _ in range(groups)]
    for seconds, path in jobs:
        target = min(bins, key=lambda b: b[0])
        target[0] += seconds
        target[1].append(path)
    return bins


def main():
    run, groups, source, output, out = sys.argv[1], int(sys.argv[2]), *sys.argv[3:6]
    paths = units(run)
    if not paths:
        raise ValueError(f'No lifecycles for {run}; generate them first.')
    bins = pack(paths, groups)
    configs, estimates = {}, {}
    for i, (seconds, members) in enumerate(bins):
        if members:
            name = f'{run}-g{i + 1}'
            configs[name] = [str(p.relative_to(ROOT)) for p in sorted(members)]
            estimates[name] = round(seconds / 60, 1)
    manifest = {
        'schema': 'drift_regimes.screen.launch.v1', 'run': run, 'campaign_prefix': 'opt125m-drift-regimes-screen-',
        'source': source, 'output': output,
        'teacher': f'{JZ}/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin',
        'teacher_sha256': '2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9',
        'weights': f'{JZ}/opt-drift-times-20261028/inputs/crossbar-std3.pt',
        'weights_sha256': '86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028',
        'corpus_root': f'{JZ}/opt-cmo-exp1/inputs/corpora',
        'corpora': {'opt125m-synthetic-50m': 'ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984',
                    'wikitext-2-raw-v1': '33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01'},
        'configs': configs, 'estimated_minutes': estimates,
        'resources': {'account': 'umg@v100', 'partition': 'gpu_p13', 'qos': 'qos_gpu-dev',
                      'constraint': 'v100-32g', 'gpus': 1, 'cpus': 10, 'memory': 'site-managed; no --mem flags'},
    }
    Path(out).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'run': run, 'units': len(paths), 'estimated_minutes': estimates}))


if __name__ == '__main__':
    main()
