"""Semantic checks after `runs inspect`: reuse the exp-001 verifiers with this pilot's pins.

    python verify.py cell DEPLOY_CONFIG TUNE_CONFIG DEPLOY_RUN TUNE_RUN
    python verify.py drift CONFIG RUN
    python verify.py prepare CONFIG RUN
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_inputs import weights_pin  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pins_for(config):
    c = json.loads(Path(config).read_text())
    return {c['analog_layers']: weights_pin(config)}


def main(mode, *args):
    if mode == 'cell':
        cell = load(HERE.parent / '20261119-mapping-faults/verify_cell.py', 'verify_cell')
        cell.SOURCE_PINS = pins_for(args[1])
        result = cell.verify(*args)
        (Path(args[3]) / 'cell-verified.json').write_text(json.dumps(result, indent=2) + '\n')
    elif mode == 'drift':
        unit = load(HERE.parent / '20261118-cmo-drift-recovery/verify_unit.py', 'verify_unit')
        result = unit.verify(args[0], args[1], source_pins=pins_for(args[0]))
        (Path(args[1]) / 'unit-verified.json').write_text(json.dumps(result, indent=2) + '\n')
    elif mode == 'prepare':
        run = Path(args[1])
        r = json.loads((run / 'result.json').read_text())
        if r['status'] != 'complete' or not (run / 'checkpoints/mapped.pt').is_file():
            raise SystemExit('Prepare run incomplete.')
        result = {'clean_development_kl': r['metrics']['clean_development']['teacher_kl'],
                  'readout_scales': [x['readout_scale'] for x in r['metrics']['mapping']],
                  'relative_squared_error': [x['relative_squared_error'] for x in r['metrics']['mapping']]}
    else:
        raise SystemExit('Unknown mode')
    print(json.dumps(result))


if __name__ == '__main__':
    main(*sys.argv[1:])
