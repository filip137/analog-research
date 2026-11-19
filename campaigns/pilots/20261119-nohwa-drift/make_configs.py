"""Derive the exp-002 DRN configs (on-chip-learning #33, series 003): no-HWA source and drift.

- prepare/: the native prepare stage of examples/opt_cmo_drn/{,l12/}prepare.json with only
  source_sha256 set to the no-HWA model-scope payload (build_source.py).
- faults/: the exp-001 mapping-faults configs with only source_sha256 changed (part a).
- drift/: CMO exp-001 drn{6,12}-ratio3-m4 with arms A0, A0-twin, CAL only (drift off: A0-twin),
  drift_tuning.mapping_scale 1 (base) or 3 (ratio3), and the source pin (part b). Wall caps
  are about twice the exp-001 CAL arm time. `-replay` configs: the HWA ratio3 control at the
  exp-001 canary length (32 updates), to check its deployment hashes and A0 grid.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
EX = ROOT / 'examples/opt_cmo_drn'
TARGET = EX / 'nohwa_drift'
HWA = {6: '86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028',
       12: 'ed53b6cbe2a7315f364291c7142ee2826c6ca4a44c21a2d5d25190b8c90569b0'}
NOHWA = {6: 'b1bb87051da8e40b68369457b6a9c43504c32486f00daa435d4020ea94f98ee9',
         12: '75347a8fb7886975c820b7707332fb39f9a444f67f43c17c610fe04f4b0954ea'}
ARMS = [{'label': 'A0', 'method': 'hold'}, {'label': 'A0-twin', 'method': 'hold'},
        {'label': 'CAL', 'method': 'calibration', 'learning_rate': 1e-2}]
WALL_MINUTES = {6: 40, 12: 60}
SCALES = {'base': 1.0, 'ratio3': 3.0}


def check(doc):
    from ebl.schema.opt_cmo_drn import parse_config
    parse_config(doc)
    return doc


def prepare(layers):
    doc = json.loads((EX / ('prepare.json' if layers == 6 else 'l12/prepare.json')).read_text())
    doc['source_sha256'] = NOHWA[layers]
    return check(doc)


def faults(path):
    doc = json.loads(path.read_text())
    doc['source_sha256'] = NOHWA[doc['analog_layers']]
    return check(doc)


def drift(layers, source, mapping, kind='full'):
    name = f'drn{layers}-ratio3-m4' + ('-canary' if kind == 'replay' else '')
    doc = json.loads((EX / 'cmo_drift_recovery' / ('canary' if kind == 'replay' else 'full') / f'{name}.json').read_text())
    doc['source_sha256'] = (HWA if source == 'hwa' else NOHWA)[layers]
    doc['fault_tuning']['arms'] = [dict(a) for a in ARMS]
    d = doc['drift_tuning']
    d.update(mapping_scale=SCALES[mapping], drift_off_arms=['A0-twin'])
    if kind == 'full':
        doc['training']['max_wall_seconds'] = WALL_MINUTES[layers] * 60 - 120
    return check(doc)


def main():
    out = []
    for layers in (6, 12):
        out.append((TARGET / f'prepare/drn{layers}-nohwa-prepare.json', prepare(layers)))
        for path in sorted((EX / 'mapping_faults').glob(f'drn{layers}-*-271002*.json')):
            out.append((TARGET / 'faults' / path.name.replace(f'drn{layers}-', f'drn{layers}-nohwa-', 1), faults(path)))
        for source, mapping in (('hwa', 'base'), ('nohwa', 'base'), ('nohwa', 'ratio3')):
            out.append((TARGET / f'drift/drn{layers}-{source}-{mapping}-m4.json', drift(layers, source, mapping)))
        out.append((TARGET / f'drift/drn{layers}-hwa-ratio3-m4-replay.json', drift(layers, 'hwa', 'ratio3', 'replay')))
    for path, doc in out:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(len(out), 'configs')


if __name__ == '__main__':
    main()
