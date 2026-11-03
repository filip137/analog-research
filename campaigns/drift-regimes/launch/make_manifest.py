"""Write the Jean Zay launch manifest for the drift-regimes crossbar qualification.

Usage: python campaigns/drift-regimes/launch/make_manifest.py <source dir on JZ> <output root on JZ> <out.json>
One group per regime (its three realizations) plus the canary.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
JZ = '/lustre/fsn1/projects/rech/umg/ucy17uy'


def main():
    source, output, out = sys.argv[1:4]
    lifecycles = sorted((ROOT / 'campaigns/drift-regimes/lifecycles').glob('opt125m-drift-regimes-qual-*.json'))
    configs = {}
    for path in lifecycles:
        group = path.stem.removeprefix('opt125m-drift-regimes-qual-')
        configs[group] = [str(path.relative_to(ROOT))]
    launch = Path(__file__).resolve().parent
    manifest = {
        'schema': 'drift_regimes.qualification.launch.v1',
        'source': source, 'output': output,
        'teacher': f'{JZ}/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin',
        'teacher_sha256': '2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9',
        'weights': f'{JZ}/opt-drift-times-20261028/inputs/crossbar-std3.pt',
        'weights_sha256': '86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028',
        'corpus_root': f'{JZ}/opt-cmo-exp1/inputs/corpora',
        'corpora': {'opt125m-synthetic-50m': 'ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984',
                    'wikitext-2-raw-v1': '33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01'},
        'configs': configs,
        'resources': {'account': 'umg@v100', 'partition': 'gpu_p13', 'qos': 'qos_gpu-dev',
                      'constraint': 'v100-32g', 'gpus': 1, 'cpus': 10, 'memory': 'site-managed; no --mem flags'},
        'budget_gpu_seconds': 36000,
    }
    Path(out).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'groups': list(configs), 'out': out}))


if __name__ == '__main__':
    main()
