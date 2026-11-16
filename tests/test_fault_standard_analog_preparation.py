"""The rebuilt standard-target scale is identical in the readiness and full grids."""
import json
from pathlib import Path

from ebl.schema.lifecycle import parse_lifecycle, provenance, stage_selections, verify_provenance

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'campaigns/fault-recovery-paper/lifecycles'
GRID = BASE / 'paper-grid-20261116/prerequisites/fault-paper-grid-20261116-xb6-standard-analog-screen.json'
CANARY = BASE / 'paper-grid-20261116/canary/fault-paper-grid-20261116-xb6-standard-analog-screen-canary.json'
SCALE = {'name': 'fault-paper-bscale-r16-3em3-standard-df73ce66',
         'sha256': 'a9870ab948fd976735c1d389bfd51282229a8a499c716b3b6f9895b45289b740'}


def test_new_scale_and_full_grid_science_are_identical_in_the_canary():
    full = parse_lifecycle(json.loads(GRID.read_text()))
    ready = parse_lifecycle(json.loads(CANARY.read_text()))
    a, b = full.fault_tuning.to_dict(), ready.fault_tuning.to_dict()
    assert a['schedule']['steps'] == 300 and b['schedule']['steps'] == 2
    assert a['schedule']['development_every_steps'] == 50 and b['schedule']['development_every_steps'] == 1
    b['schedule'] = a['schedule']
    assert a == b
    assert a['analog_lora'] == {'rank': 16, 'b_scale': SCALE}
    assert [arm['learning_rate'] for arm in a['arms']] == [.0001, .0003, .001, .003, .01, .03]
    assert all(arm['method'] == 'analog_lora' and arm['rank'] == 16 for arm in a['arms'])
    assert not a.get('evaluation_cohorts')
    assert a['recovery']['sequences'] == 1024 and a['schedule']['batch'] == 8
    assert a['targets']['sha256'] == 'df73ce6615be3cdd3005d2ca0b97c03131f9236ba481bc7250edde5ff66fb01a'
    assert a['assignment_seed'] == 271001
    assert a['deployment']['sha256'] == '611a0aaba97d491af9bcbfebe81e2192603ceacf43192036ee0962609a0f1ffc'
    assert full.defects.cases[0].rate_ppm == ready.defects.cases[0].rate_ppm == 30000


def test_both_scale_configs_preserve_source_provenance_and_select_only_the_tuning_map():
    template = parse_lifecycle(json.loads((BASE / 'fault-completion-20261115-xb6-canary.json').read_text()))
    record = provenance(template, 'hwa_source', hwa_arm='standard_hwa_model')
    for path in (GRID, CANARY):
        lc = parse_lifecycle(json.loads(path.read_text()))
        assert path.stem == lc.lifecycle_id
        assert lc.devices == template.devices
        verify_provenance(record, lc, 'hwa_source', hwa_arm='standard_hwa_model')
        assert [s.array_seed for s in stage_selections(lc) if s.kind == 'fault_tuning'] == [271001]
