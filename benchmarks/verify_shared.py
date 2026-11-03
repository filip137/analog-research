"""Verify the declared native modules in two independent source checkouts."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(left, right):
    manifest = json.loads((left / 'docs/shared_native_modules.json').read_text())
    if manifest != json.loads((right / 'docs/shared_native_modules.json').read_text()):
        raise ValueError('Shared native module manifests differ.')
    hashes = {}
    for relative in manifest['modules']:
        first = (left / relative).read_bytes()
        if first != (right / relative).read_bytes():
            raise ValueError(f'Shared native implementation differs: {relative}')
        hashes[relative] = hashlib.sha256(first).hexdigest()
    return hashes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.left.resolve(), args.right.resolve()), indent=2))
