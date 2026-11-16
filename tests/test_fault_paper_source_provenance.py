"""Paper deployment seeds must not change the frozen HWA-dependent devices section."""
import copy
import json
from pathlib import Path

import pytest

from ebl.schema.lifecycle import parse_lifecycle, provenance, stage_selections, verify_provenance

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'campaigns/fault-recovery-paper/lifecycles'
PAPER = BASE / 'paper-grid-20261116'
MODELS = ('xb6', 'xb12')
PATHS = sorted((PAPER / 'paper').glob('*xb*.json')) + [
    PAPER / f'canary/fault-paper-grid-20261116-{model}-paper-canary.json' for model in MODELS]
FROZEN_DEVICES = {
    'assignment_seeds': [271001, 271002, 271003, 271004, 271005],
    'endpoint_seed_offset': 10000, 'model': 'aihwkit_1.1.0_reram_cmo',
    'noise_scale': 1.0, 'selection_seeds': [211001, 211002], 'technology': 'cmo',
}


def original_source(model='xb12'):
    doc = json.loads((BASE / f'fault-completion-20261115-{model}-canary.json').read_text())
    assert doc['devices'] == FROZEN_DEVICES
    return provenance(parse_lifecycle(doc), 'hwa_source', hwa_arm='standard_hwa_model')


@pytest.mark.parametrize('path', PATHS, ids=lambda path: path.stem)
def test_paper_and_canary_reuse_frozen_hwa_devices_but_select_one_requested_map(path):
    doc = json.loads(path.read_text())
    assert doc['devices'] == FROZEN_DEVICES
    lifecycle = parse_lifecycle(doc)
    verify_provenance(original_source('xb6' if 'xb6' in path.stem else 'xb12'), lifecycle, 'hwa_source', hwa_arm='standard_hwa_model')
    chosen = [stage for stage in stage_selections(lifecycle) if stage.kind == 'fault_tuning']
    assert len(chosen) == 1
    assert chosen[0].array_seed == doc['fault_tuning']['assignment_seed']
    if 'paper-canary' in path.stem:
        assert chosen[0].array_seed == 271002
        assert lifecycle.defects.cases[0].rate_ppm == 30000
    else:
        tag, seed = path.stem.rsplit('-', 2)[-2:]
        rate = int(tag.removeprefix('r'))
        assert chosen[0].array_seed == int(seed)
        assert lifecycle.defects.cases[0].rate_ppm == rate * 10000
        assert lifecycle.defects.cases[0].kind == ('none' if rate == 0 else 'random')


@pytest.mark.parametrize('model', MODELS)
def test_paper_inventory_covers_all_requested_maps_and_zero_nonzero_faults(model):
    papers = [path for path in PATHS if path.parent.name == 'paper' and model in path.stem]
    assert len(papers) == 12
    assert {tuple(path.stem.rsplit('-', 2)[-2:]) for path in papers} == {
        (f'r{rate:02d}', str(seed)) for rate in (0, 1, 3, 5) for seed in (271002, 271003, 271004)}


@pytest.mark.parametrize('rate', [0, 30000])
def test_old_singleton_devices_change_is_rejected_even_though_it_selects_the_right_stage(rate):
    doc = json.loads(next(path for path in PATHS if 'xb12' in path.stem).read_text())
    doc['fault_tuning']['assignment_seed'] = 271002
    doc['defects']['cases'][0].update(kind='none' if rate == 0 else 'random', rate_ppm=rate)
    doc['devices']['assignment_seeds'] = [271002]
    lifecycle = parse_lifecycle(copy.deepcopy(doc))
    assert [s.array_seed for s in stage_selections(lifecycle) if s.kind == 'fault_tuning'] == [271002]
    with pytest.raises(ValueError, match="changed sections \\['devices'\\]"):
        verify_provenance(original_source(), lifecycle, 'hwa_source', hwa_arm='standard_hwa_model')


def test_xb6_paper_canary_preserves_full_science_and_only_reduces_updates():
    full = json.loads((PAPER / 'paper/fault-paper-grid-20261116-xb6-r03-271002.json').read_text())
    canary = json.loads((PAPER / 'canary/fault-paper-grid-20261116-xb6-paper-canary.json').read_text())
    assert full['fault_tuning']['analog_lora']['b_scale'] == {
        'name': 'fault-paper-bscale-r16-3em3-standard-df73ce66',
        'sha256': 'a9870ab948fd976735c1d389bfd51282229a8a499c716b3b6f9895b45289b740'}
    assert len(full['fault_tuning']['arms']) == 9
    assert [x['learning_rate'] for x in full['fault_tuning']['arms'] if x['label'] == 'analog_lora16'] == [.01]
    assert [x['sequences'] for x in full['fault_tuning']['evaluation_cohorts']] == [1024, 384]
    expected = copy.deepcopy(full)
    expected['lifecycle_id'] = canary['lifecycle_id']
    expected['fault_tuning']['schedule'].update(steps=4, development_every_steps=2)
    assert expected == canary
