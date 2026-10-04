import copy
import json
from pathlib import Path
import pytest
import torch
from experiments.mnist_analog_relu.staged_config import parse_staged_crossbar_config
from experiments.mnist_analog_relu.staged_runtime import _adam_selection_rank
from training.ibm_om_standard_crossbar import PulseAdam, build_crossbar_layout

ROOT=Path(__file__).resolve().parents[1]

def test_explicit_schedule_bounds_and_kl_selection():
    cfg=json.loads((ROOT/'examples/mnist_analog_relu/long_kl_schedules_20260906/healthy-decay-1e-5.json').read_text())
    parsed=parse_staged_crossbar_config(cfg)
    assert parsed.stage.epochs==30
    assert parsed.stage.learning_rate_schedule['final_factor']==0.01
    bad=copy.deepcopy(cfg)
    bad['stage']['learning_rate_schedule']['final_factor']=0
    with pytest.raises(ValueError):
        parse_staged_crossbar_config(bad)
    high_accuracy=dict(student_accuracy=.98,objective_value=.06,epoch=1)
    low_kl=dict(student_accuracy=.97,objective_value=.03,epoch=3)
    assert _adam_selection_rank(**high_accuracy)<_adam_selection_rank(**low_kl)
    assert _adam_selection_rank(**low_kl,kl_first=True)<_adam_selection_rank(**high_accuracy,kl_first=True)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires the actual CUDA execution gate')
def test_decay_reduces_physical_writes_and_checkpoint_replays():
    layout=build_crossbar_layout((784,256,10),maximum_input_size=512)
    size=sum(tile.cells for tile in layout)
    class Port:
        def __init__(self):self.size=size
        def pulse(self,direction):self.last=direction.clone()
    def create():
        return PulseAdam(size=size,layout=layout,learning_rates=(.003,.003),betas=(.9,.999),epsilon=1e-8,layer_scope='all',nominal_dw_min=.0949,pulse_cap_per_cell=640,generator=torch.Generator(device='cuda').manual_seed(101),device='cuda')
    full,reduced=create(),create()
    reduced.learning_rate_scale=.1
    p1,p2=Port(),Port()
    gradient=torch.ones(size,device='cuda')
    a=full.step(gradient,p1)['applied_pulses']
    b=reduced.step(gradient,p2)['applied_pulses']
    assert 0.05*a < b < .2*a
    restored=create()
    restored.load_state_dict(reduced.state_dict())
    p3=Port()
    reduced.step(gradient,p2)
    restored.step(gradient,p3)
    assert restored.learning_rate_scale==.1
    assert torch.equal(p2.last,p3.last)
    assert torch.equal(reduced.first_moment,restored.first_moment)
    assert torch.equal(reduced.pulse_count,restored.pulse_count)
