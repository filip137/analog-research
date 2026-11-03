import json
from dataclasses import replace
from pathlib import Path
import torch
from core.artifacts import sha256_file
from methods.maintenance.config import DriftRecovery
from ebl.runtime.shared_recovery import run
from methods.maintenance.recovery import checkpoint
from hardware.drift.writable import WritableDrift
from networks.opt_drift import CrossbarAdapter
from networks.opt import OptMlpSuffix
from ebl.schema.lifecycle import parse_lifecycle, stage_selections, StageSelection
from test_workflow_opt import tiny_teacher, tokens
from test_workflow_cmo import cmo_inputs, small, case_doc
from workflow_helpers import Store, run_stage


def settings(digest='a'*64):
    return replace(DriftRecovery(digest), persistent_fractions=(.9,), assignment_seeds=(271001,),
                   protocols=('repeated',), arms=('full_refresh','stable_learning','random_learning'),
                   write_budgets=(1.,), horizon_seconds=9000., maintenance_interval_seconds=3600.,
                   followup_seconds=(60.,), evaluation_seconds=(0.,3600.,9000.),
                   training_sequences=4,validation_sequences=2,calibration_sequences=2,
                   batch_size=2,microbatch_size=1,max_updates=2,learning_rate=.001,
                   no_drift_control=False,canary=True)


def test_adapter_preserves_native_forward_and_uses_physical_cell_gradients():
    torch.set_num_threads(1)
    net=OptMlpSuffix(tiny_teacher(),2,8)
    adapter=CrossbarAdapter(net,'cpu')
    physical,_,_,_=adapter.physical_layout()
    x=tokens(2)
    torch.testing.assert_close(adapter.clean(x),adapter.forward(x,physical),atol=2e-5,rtol=2e-5)
    physical[0].requires_grad_(True)
    adapter.forward(x,physical).square().mean().backward()
    assert physical[0].grad is not None and float(physical[0].grad.abs().sum())>0
    assert all(p.grad is None for p in net.parameters())


def test_native_custom_stage_and_artifacts(cmo_inputs,tmp_path):
    teacher,digest=cmo_inputs
    lc=small(**{'onchip':None,'defects.cases':[case_doc()], 'data.corpus':{'name':'synthetic','sha256':digest},
                'data.sequence_length':9,'data.max_examples':4,'hwa.arms':[{'id':'digital','method':'none'}]})
    results={}
    for selection in stage_selections(lc):
        if selection.kind=='deploy': continue
        results[selection.stage_id]=run_stage(lc,selection,results,teacher,tmp_path/'parents')
    directory,result=results['hwa__digital']
    source=directory/next(a['path'] for a in result['artifacts'] if a['kind']=='hwa_source')
    doc=lc.to_dict();doc['drift_recovery']=settings(sha256_file(source)).to_dict()
    custom=parse_lifecycle(doc)
    selection=StageSelection('drift_recovery','digital',271001)
    directory,result=run_stage(custom,selection,results,teacher,tmp_path/'run')
    assert result['metrics']['source_unchanged']
    assert result['metrics']['trajectories']==3 and result['metrics']['exact_replays']==3
    rows=json.loads((directory/'drift_recovery_measurements.json').read_text())
    assert all(r['canary'] for r in rows)
    learn=[r for r in rows if r['arm']=='stable_learning' and r['phase']=='after']
    assert len(learn)==2 and learn[-1]['cell_writes']>learn[0]['cell_writes']>0
    assert all(r['distinct_written_cells']<=.1*r['physical_devices'] for r in learn)
