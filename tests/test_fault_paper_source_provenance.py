"""Paper deployment seeds must not change the frozen HWA-dependent devices section."""
import copy
import json
from pathlib import Path

import pytest

from ebl.schema.lifecycle import parse_lifecycle, provenance, stage_selections, verify_provenance

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'campaigns/fault-recovery-paper/lifecycles'
PAPER = BASE / 'paper-grid-20261116'
TEMPLATE = BASE / 'fault-completion-20261115-xb12-canary.json'
PATHS = sorted((PAPER / 'paper').glob('*xb12*.json')) + [
    PAPER / 'canary/fault-paper-grid-20261116-xb12-paper-canary.json']
FROZEN_DEVICES = {
    'assignment_seeds': [271001, 271002, 271003, 271004, 271005],
    'endpoint_seed_offset': 10000, 'model': 'aihwkit_1.1.0_reram_cmo',
    'noise_scale': 1.0, 'selection_seeds': [211001, 211002], 'technology': 'cmo',
}


def original_source():
    doc = json.loads(TEMPLATE.read_text())
    assert doc['devices'] == FROZEN_DEVICES
    return provenance(parse_lifecycle(doc), 'hwa_source', hwa_arm='standard_hwa_model')


@pytest.mark.parametrize('path', PATHS, ids=lambda path: path.stem)
def test_paper_and_canary_reuse_frozen_hwa_devices_but_select_one_requested_map(path):
    doc = json.loads(path.read_text())
    assert doc['devices'] == FROZEN_DEVICES
    lifecycle = parse_lifecycle(doc)
    verify_provenance(original_source(), lifecycle, 'hwa_source', hwa_arm='standard_hwa_model')
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


def test_paper_inventory_covers_all_requested_maps_and_zero_nonzero_faults():
    papers = [path for path in PATHS if path.parent.name == 'paper']
    assert len(papers) == 12
    assert {tuple(path.stem.rsplit('-', 2)[-2:]) for path in papers} == {
        (f'r{rate:02d}', str(seed)) for rate in (0, 1, 3, 5) for seed in (271002, 271003, 271004)}


@pytest.mark.parametrize('rate', [0, 30000])
def test_old_singleton_devices_change_is_rejected_even_though_it_selects_the_right_stage(rate):
    doc = json.loads(PATHS[0].read_text())
    doc['fault_tuning']['assignment_seed'] = 271002
    doc['defects']['cases'][0].update(kind='none' if rate == 0 else 'random', rate_ppm=rate)
    doc['devices']['assignment_seeds'] = [271002]
    lifecycle = parse_lifecycle(copy.deepcopy(doc))
    assert [s.array_seed for s in stage_selections(lifecycle) if s.kind == 'fault_tuning'] == [271002]
    with pytest.raises(ValueError, match="changed sections \\['devices'\\]"):
        verify_provenance(original_source(), lifecycle, 'hwa_source', hwa_arm='standard_hwa_model')
