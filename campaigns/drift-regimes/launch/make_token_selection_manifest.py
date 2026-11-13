"""Launch the issue #27 selection controls with the unchanged native worker."""
import argparse
import json
from pathlib import Path

from make_token_manifest import manifest as base_manifest, ROOT


def manifest(source, output):
    m = base_manifest(source, output)
    specs = [('ts-canary', 271004, 'canary'), ('ts-dev-271004', 271004, 'dev'),
             *[('ts-eval-'+str(seed), seed, 'eval') for seed in (271001, 271002, 271003)]]
    configs = {}
    for group, seed, part in specs:
        p = Path('campaigns/drift-regimes/lifecycles')/f'opt125m-drift-regimes-screen-r0-g-{seed}-tokens-select-{part}.json'
        if not (ROOT/p).is_file():
            raise ValueError('Missing immutable lifecycle: '+str(p))
        configs[group] = [str(p)]
    m.update(run='token-selection', campaign_prefix='opt125m-token-selection-',
             configs=configs, budget_gpu_seconds=9000)
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source'); p.add_argument('output'); p.add_argument('destination')
    a = p.parse_args()
    m = manifest(a.source, a.output)
    Path(a.destination).write_text(json.dumps(m, indent=2)+'\n')
    print(json.dumps({'groups': list(m['configs'])}))
