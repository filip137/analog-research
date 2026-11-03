"""Expand a frozen CMO deployment into portable held tensors without new writes.

Run on its original GPU architecture. The original bundle remains unchanged;
the new artifact records its parent hash and preserves its provenance/metrics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from core.artifacts import atomic_write_json, sha256_file
from core.tensor_identity import tensor_hash
from core.tensor_io_native import load, save
import hardware.arrays as devices
import methods.programming.lifecycle as program_verify
from ebl.schema.lifecycle import parse_lifecycle, verify_provenance
from networks.opt import MatrixSpec


def materialize(run: Path, bundle: Path, output: Path, device: str):
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Refusing to overwrite materialized evidence')
    config = json.loads((run / 'config.resolved.json').read_text())
    lc = parse_lifecycle(config['lifecycle'])
    if lc.devices.technology != 'cmo' or config['stage']['kind'] != 'deploy':
        raise ValueError('Expected a native CMO deployment')
    result = json.loads((run / 'result.json').read_text())
    artifact = next(a for a in result['artifacts'] if a['kind'] == 'deployment')
    path = run / artifact['path']
    if sha256_file(path) != artifact['sha256']:
        raise ValueError('Parent deployment hash mismatch')
    p0, db = load(path), load(bundle)
    verify_provenance(p0['provenance'], lc, 'deployment', hwa_arm=config['stage']['hwa_arm'])
    verify_provenance(db['provenance'], lc, 'device_bundle')
    mapping = json.loads((run / 'mapping.json').read_text())
    layout = [MatrixSpec(**dict(m, shape=tuple(m['shape']))) for m in mapping['matrices']]
    observed = {}
    for case in lc.defects.cases:
        entry = p0['cases'][case.label]
        array = program_verify.fresh(lc, layout, p0['array_seed'], case, db['populations'], device)
        devices.load_array_state(array, entry['array'], 'cmo', p0['target'].to(device))
        before = tensor_hash(array.read())
        if before != entry['programming']['apparent_sha256']:
            raise ValueError('Materialization does not reproduce original held state')
        entry['array'] = devices.array_state(array, 'cmo', portable=True)
        observed[case.label] = before
    p0['materialized_from_sha256'] = artifact['sha256']
    save(output, p0)
    saved = load(output)
    for case in lc.defects.cases:
        array = program_verify.fresh(lc, layout, p0['array_seed'], case, db['populations'], device)
        devices.load_array_state(array, saved['cases'][case.label]['array'], 'cmo')
        if tensor_hash(array.read()) != observed[case.label]:
            raise ValueError('Portable state did not replay')
    receipt = {'schema': 'crossbar_lifecycle.portable_deployment.v1', 'parent': str(path),
        'parent_sha256': artifact['sha256'], 'device_bundle_sha256': sha256_file(bundle),
        'output': str(output), 'sha256': sha256_file(output), 'apparent_sha256': observed,
        'gpu': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else 'cpu',
        'torch': torch.__version__, 'physical_state_preserved': True, 'new_device_writes': 0}
    atomic_write_json(output.with_suffix('.json'), receipt)
    return receipt


def validate(output):
    r = json.loads(output.with_suffix('.json').read_text())
    if r['sha256'] != sha256_file(output) or not r['physical_state_preserved'] or r['new_device_writes']:
        raise ValueError('Portable deployment validation failed')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path)
    p.add_argument('--devices', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--validate', action='store_true')
    a = p.parse_args(); torch.set_num_threads(1)
    r = validate(a.output) if a.validate else materialize(a.run, a.devices, a.output, a.device)
    print(json.dumps(r, indent=2))

if __name__ == '__main__': main()
