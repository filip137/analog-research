"""Pre-launch checks: staged source receipt, teacher, and the weights pinned for a config.

    python check_inputs.py CONFIG WEIGHTS TEACHER SOURCE_SNAPSHOT
For a prepare config the weights are the model-scope payload (sha256 = config.source_sha256);
otherwise they are the mapped/trained DRN state pinned in pins.json for (layers, source_sha256).
"""
import hashlib
import json
from pathlib import Path
import sys

TEACHER = '2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def weights_pin(config):
    c = json.loads(Path(config).read_text())
    if c['stage'] == 'prepare':
        return c['source_sha256']
    pins = json.loads((Path(__file__).parent / 'pins.json').read_text())
    return pins[f"{c['analog_layers']}:{c['source_sha256']}"]


def main(config, weights, teacher, snapshot):
    if digest(weights) != weights_pin(config):
        raise SystemExit('Wrong DRN source weights for ' + config)
    if digest(teacher) != TEACHER:
        raise SystemExit('Wrong teacher')
    receipt = json.loads(Path(snapshot).read_text())
    for name, sha in receipt['files'].items():
        if digest(Path(snapshot).parent / name) != sha:
            raise SystemExit('Staged source changed: ' + name)


if __name__ == '__main__':
    main(*sys.argv[1:5])
