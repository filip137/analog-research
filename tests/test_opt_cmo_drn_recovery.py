"""Matched physical controls, controller accounting and continuation."""

import copy
from dataclasses import replace,asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ebl.schema.opt_cmo_drn import Config,Training,Data,Recovery,Deployment,Solver,parse_config
from methods.programming.cmo import program_rewritten
from networks.opt.physical import CmoDrnBank,ConductanceArray
from methods.cmo_thresholded_adam import ThresholdedWriteAdam
from tests.test_opt_cmo_drn import tiny_model,cpu_threads


class CommandPort:
    def __init__(self):self.commands=[]
    def read(self):return torch.zeros(2,3)
    def write(self,command):self.commands.append(command.clone())


def test_threshold_is_strict_and_attempt_budget_is_controller_owned():
    port=CommandPort()
    writer=ThresholdedWriteAdam(port,learning_rate=.1)
    writer.pending.fill_(writer.threshold_q)
    writer.step(torch.zeros(2,3))
    assert writer.attempt_counts.sum()==0
    for _ in range(12):writer.step(torch.ones(2,3))
    assert torch.equal(writer.attempt_counts,torch.full((2,3),4,dtype=torch.int32))
    assert sum(int((c!=0).sum()) for c in port.commands)==24
    assert torch.equal(writer.pending,torch.zeros_like(writer.pending))


def test_thresholded_controller_matches_pinned_upstream_golden():
    golden=torch.load(Path(__file__).parent/'golden/opt_cmo_drn/thresholded_writer.pt',weights_only=True)
    port=CommandPort()
    writer=ThresholdedWriteAdam(port,learning_rate=golden['learning_rate'])
    for gradient,expected in zip(golden['gradients'],golden['commands']):
        writer.step(gradient)
        assert torch.equal(port.commands[-1],expected)
    for key in ('first','second','pending','attempt_counts'):
        assert torch.equal(getattr(writer,key),golden['state'][key]),key


def test_uncapped_writer_keeps_writing_and_resumes_exactly():
    port=CommandPort()
    writer=ThresholdedWriteAdam(port,learning_rate=.1,cap=None)
    for _ in range(7):writer.step(torch.ones(2,3))
    assert torch.equal(writer.attempt_counts,torch.full((2,3),7,dtype=torch.int32))
    assert writer.diagnostics()['capped_devices']==0
    saved=writer.state_dict()
    clone_port=CommandPort()
    clone=ThresholdedWriteAdam(clone_port,learning_rate=.1,cap=None)
    clone.load_state_dict(saved)
    for gradient in (torch.full((2,3),-.3),torch.full((2,3),.7)):
        writer.step(gradient);clone.step(gradient)
        assert torch.equal(port.commands[-1],clone_port.commands[-1])
    for key in ('first','second','pending','attempt_counts'):
        assert torch.equal(getattr(writer,key),getattr(clone,key))
    capped=ThresholdedWriteAdam(CommandPort(),learning_rate=.1,cap=4)
    with pytest.raises(ValueError,match='contract differs'):
        capped.load_state_dict(saved)


@pytest.mark.parametrize('cap,expected_attempts',[(4,24),(None,72)])
def test_stuck_attempts_consume_budget_and_never_change_physical_state(cap,expected_attempts):
    x=torch.full((2,3),40.)
    state={'name':'test','noise_scale':1.,'held':x,'target':x,'mask':torch.ones_like(x,dtype=torch.bool),
           'stuck':x,'raw':x,'verify_flags':torch.ones_like(x,dtype=torch.bool),
           'rewrite_selected':torch.zeros_like(x,dtype=torch.bool),'rng':torch.Generator().manual_seed(7).get_state(),
           'write_attempts':0,'device_writes':0,'stuck_write_attempts':0}
    array=ConductanceArray(state,'cpu')
    writer=ThresholdedWriteAdam(array,learning_rate=.1,cap=cap)
    for _ in range(12):writer.step(torch.ones_like(x))
    assert torch.equal(array.held,x)
    assert array.counters=={'write_attempts':expected_attempts,'device_writes':0,'stuck_write_attempts':expected_attempts,
                            'gmin_clipped_writes':0}
    assert int(writer.attempt_counts.sum())==expected_attempts


@pytest.mark.parametrize('kind,rate',[('none',0.),('random',.03)])
def test_saved_bank_replays_held_view_and_write_rng_exactly(kind,rate):
    model,_=tiny_model()
    view,receipt,state=program_rewritten(model,271002,kind,rate,capture_state=True)
    bank=CmoDrnBank(state,'cpu')
    for expected,actual in zip(view,bank.view(model)):
        for x,y in zip(expected,actual):assert torch.equal(x,y)
    for array in bank.arrays:array.write(torch.full_like(array.held,.01))
    saved=bank.state_dict()
    assert saved['schema']=='opt_cmo_drn.bank.v2'
    for cell,array in zip(saved['cells'],bank.arrays):
        assert 'stuck' not in cell and cell['stuck_values'].numel()==int(array._mask.sum())
        assert cell['mask_bits'].numel()==(array.held.numel()+7)//8
    clone=CmoDrnBank(saved,'cpu')
    for a,b in zip(bank.arrays,clone.arrays):
        assert torch.equal(a._stuck,b._stuck)
        assert torch.equal(a._mask,b._mask)
        for key in a.evidence:assert torch.equal(a.evidence[key],b.evidence[key])
        delta=torch.full_like(a.held,-.001)
        a.write(delta);b.write(delta)
        assert torch.equal(a.held,b.held)
        assert torch.equal(a.generator.get_state(),b.generator.get_state())
    assert bank.costs()==clone.costs()


def test_physical_state_is_never_clipped_to_the_target_window():
    model,_=tiny_model()
    _,_,state=program_rewritten(model,271002,capture_state=True)
    cell=state['cells'][0]
    cell['noise_scale']=0.
    array=ConductanceArray(cell,'cpu')
    array.write(torch.full_like(array.held,3.))
    assert array.held.max()>88.2
    with pytest.raises(FloatingPointError):array.write(torch.full_like(array.held,-100.))


@pytest.mark.parametrize('cap,noise_factor,retain_optimizer,update_rule',[
    (4,1.,True,'thresholded_adam'),(None,1.,True,'thresholded_adam'),(None,0.,False,'thresholded_adam'),
    (None,0.,True,'direct_adam'),(None,1.,True,'direct_adam'),(None,1.,True,'bounded_thresholded_adam')])
def test_matched_controls_and_exact_recovery_resume(tmp_path,monkeypatch,cap,noise_factor,retain_optimizer,update_rule):
    import ebl.runtime.opt_cmo_drn.runtime as runtime
    import ebl.runtime.opt_cmo_drn.recovery as recovery
    from ebl.runtime.store import sha256_file
    from ebl.lifecycle.__main__ import collect
    model,teacher=tiny_model()
    teacher_path=tmp_path/'teacher.pt';torch.save({},teacher_path)
    config=replace(Config(),stage='deploy',teacher_sha256=sha256_file(teacher_path),
        solver=replace(Solver(),input_gain=5.,token_chunk_size=3,activation_checkpointing=False),
        training=replace(Training(),effective_batch=2,microbatch=1),
        data=replace(Data(),sequences=4,development_sequences=2,sequence_length=4),
        recovery=replace(Recovery(),updates=6,learning_rate=.03,max_write_attempts=cap,calibration_updates=2,checkpoint_every=1,
                         write_noise_factor=noise_factor,development_every=2,retain_final_optimizer=retain_optimizer,
                         conductance_floor='gmin' if noise_factor==0. and update_rule=='thresholded_adam' else 'none',
                         update_rule=update_rule,target_bounds='gmin_gmax' if update_rule!='thresholded_adam' else 'none'))
    source=tmp_path/'source.pt';runtime.save(source,runtime.weights_payload(model,config))
    tokens=torch.tensor([[1,2,3,4],[4,3,2,1],[3,4,1,2],[2,1,4,3]])
    monkeypatch.setattr(runtime,'teacher_model',lambda path,c:copy.deepcopy(teacher))
    monkeypatch.setattr(recovery,'cohorts',lambda c:(tokens,tokens[:2],{'training':'tiny','development':'tiny'}))
    monkeypatch.setattr(recovery,'TRANSFER_TESTS',(('test','tiny','pinned',2),))
    monkeypatch.setattr(recovery,'load_corpus',lambda name,digest:{'test':tokens.flatten()})
    original_save=runtime.save
    def save_intermediate(path,payload):
        original_save(path,payload)
        if path.name=='latest.pt' and payload.get('arm')=='learning' and payload['step']==2:
            original_save(tmp_path/'intermediate.pt',payload)
    monkeypatch.setattr(runtime,'save',save_intermediate)
    request=SimpleNamespace(spec=config,teacher_weights=teacher_path,weights=source,resume=None,device_state=None,
                            output_dir=tmp_path/'deploy',command=('test',))
    assert runtime.run_train(request)==0
    deployed=next(request.output_dir.iterdir())
    canonical=deployed/'checkpoints/deployed.pt'
    initial=torch.load(canonical,weights_only=True)
    outputs={};runs=[deployed]
    for arm in ('hold','calibration','learning','learning_calibration','rewrite'):
        request.spec=replace(config,stage='recover',recovery=replace(config.recovery,arm=arm))
        request.device_state=outputs['learning'] if arm=='learning_calibration' else canonical
        request.output_dir=tmp_path/arm
        assert runtime.run_train(request)==0
        run=next(request.output_dir.iterdir());runs.append(run)
        outputs[arm]=run/'checkpoints/final.pt'
        final=torch.load(outputs[arm],weights_only=True)
        assert final['initial_device_state_sha256']==sha256_file(canonical)
        assert final['start_held_hashes']==initial['start_held_hashes']
        if arm in ('hold','calibration'):
            assert final['costs']['write_attempts']==0
            for a,b in zip(final['bank']['cells'],initial['bank']['cells']):assert torch.equal(a['held'],b['held'])
        if arm=='learning':
            assert final['costs']['write_attempts']>0
            assert (final['engine'] is not None)==retain_optimizer
            assert final['write_noise_factor']==noise_factor
            assert final['conductance_floor']==config.recovery.conductance_floor
            if noise_factor==0. and update_rule=='thresholded_adam':assert final['costs']['gmin_clipped_writes']>0
            if update_rule!='thresholded_adam':
                assert final['update_rule']==update_rule and final['target_bounds']=='gmin_gmax'
                assert final['costs']['proposed_commands']==final['costs']['projected_noops']+final['costs']['write_attempts']
            capture=torch.load(run/'artifacts/first-write.pt',weights_only=True)
            assert capture['step']==1 and capture['write_noise_factor']==noise_factor
            assert all(x['commands_q'].numel()==x['indices'].numel()>0 for x in capture['cells'])
            import json
            scores=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()
                    if json.loads(line)['event']=='recovery_development']
            assert [x['step'] for x in scores]==[0,1,2,4,6]
            assert scores[0]['development']==initial['development']
            for k,v in final['weights']['model'].items():assert torch.equal(v,model.state_dict()[k]),k
        if arm=='learning_calibration':
            learning=torch.load(outputs['learning'],weights_only=True)
            assert final['costs']==learning['costs']
            for a,b in zip(final['bank']['cells'],learning['bank']['cells']):assert torch.equal(a['held'],b['held'])
        if arm=='rewrite':
            assert final['costs']['write_attempts']==6*final['costs']['physical_devices']
        validation=SimpleNamespace(spec=request.spec,teacher_weights=teacher_path,weights=outputs[arm],
                                   output_dir=tmp_path/(arm+'-validate'),command=('test',))
        assert runtime.run_validate(validation)==0
        assert collect([next(validation.output_dir.iterdir())])['complete']
    request.spec=replace(config,stage='recover',recovery=replace(config.recovery,arm='learning'))
    # Evaluate an interrupted endpoint without an optimizer step, a rewrite, or
    # reclassifying its checkpoint as completed training.
    intermediate=tmp_path/'intermediate.pt'
    before=sha256_file(intermediate)
    validation=SimpleNamespace(spec=request.spec,teacher_weights=teacher_path,weights=intermediate,
                               output_dir=tmp_path/'endpoint-validate',command=('test',))
    assert runtime.run_validate(validation)==0
    endpoint_run=next(validation.output_dir.iterdir())
    results=json.loads((endpoint_run/'artifacts/held_out.json').read_text())
    assert results['development']==next(s['development'] for s in scores if s['step']==2)
    assert results['checkpoint']['updates']==2
    assert not results['checkpoint']['training_complete']
    assert not results['checkpoint']['development_replay_verified']
    assert results['checkpoint']['costs']==torch.load(intermediate,weights_only=True)['costs']
    assert sha256_file(intermediate)==before
    assert collect([endpoint_run])['complete']
    validation.spec=replace(request.spec,recovery=replace(request.spec.recovery,updates=2))
    with pytest.raises(ValueError,match='unchanged training config'):
        runtime.run_validate(validation)
    validation.spec=request.spec
    with monkeypatch.context() as patch:
        patch.setattr(torch,'__version__','different')
        with pytest.raises(ValueError,match='saved PyTorch version'):
            runtime.run_validate(validation)
    request.weights=None;request.device_state=None;request.resume=tmp_path/'intermediate.pt'
    request.output_dir=tmp_path/'resumed'
    assert runtime.run_train(request)==0
    resumed=next(request.output_dir.iterdir())
    expected=torch.load(outputs['learning'],weights_only=True)
    actual=torch.load(resumed/'checkpoints/final.pt',weights_only=True)
    assert expected['development']==actual['development']
    assert expected['costs']==actual['costs']
    for a,b in zip(expected['bank']['cells'],actual['bank']['cells']):
        assert torch.equal(a['held'],b['held'])
        assert torch.equal(a['rng'],b['rng'])
    assert expected['writer_diagnostics']==actual['writer_diagnostics']
    if retain_optimizer:
        for a,b in zip(expected['engine']['writers'],actual['engine']['writers']):
            for key in ('first','second','pending','attempt_counts'):
                if key in a:assert torch.equal(a[key],b[key])
    assert torch.load(tmp_path/'intermediate.pt',weights_only=True)['engine'] is not None
    assert collect(runs)['complete']
    assert collect([resumed])['complete']


def test_strict_recovery_contract():
    assert parse_config({'stage':'recover'}).recovery.max_write_attempts==4
    assert parse_config({'recovery':{'max_write_attempts':None}}).recovery.max_write_attempts is None
    for invalid in (0,-1,5,True,4.0):
        with pytest.raises(ValueError):parse_config({'recovery':{'max_write_attempts':invalid}})
    with pytest.raises(ValueError):parse_config({'recovery':{'threshold_sigma':0}})
    with pytest.raises(ValueError):parse_config({'deployment':{'fault_kind':'none','fault_rate':.03}})
    assert parse_config({'recovery':{'write_noise_factor':0.}}).recovery.write_noise_factor==0.
    for factor in (-1.,.5,2.,True):
        with pytest.raises(ValueError):parse_config({'recovery':{'write_noise_factor':factor}})
    with pytest.raises(ValueError):parse_config({'recovery':{'development_every':-1}})
    assert parse_config({'recovery':{'conductance_floor':'gmin'}}).recovery.conductance_floor=='gmin'
    with pytest.raises(ValueError):parse_config({'recovery':{'conductance_floor':'zero'}})


@pytest.mark.parametrize('noise_factor',[0.,1.])
def test_gmin_clipping_only_changes_written_healthy_cells_and_resumes_exactly(noise_factor):
    from hardware.devices.cmo import G_MIN_US,HALF_RANGE_US
    x=torch.tensor([[8.,10.,8.],[0.,20.,25.]])
    mask=torch.zeros_like(x,dtype=torch.bool);mask[1,0]=True
    state={'name':'test','noise_scale':1.,'held':x,'target':x,'mask':mask,'stuck':x,
           'raw':x,'verify_flags':mask,'rewrite_selected':mask,
           'rng':torch.Generator().manual_seed(7).get_state(),
           'write_attempts':0,'device_writes':0,'stuck_write_attempts':0}
    array=ConductanceArray(state,'cpu')
    command=torch.tensor([[-1.,-2./HALF_RANGE_US,0.],[-1.,0.,3.]])
    array.write(command,noise_factor=noise_factor,conductance_floor='gmin')
    assert array.held[0,0]==array.held[0,1]==G_MIN_US
    assert array.held[0,2]==8. and array.held[1,0]==0. and array.held[1,1]==20.
    assert array.held[1,2]>88.2  # Only the requested lower bound is applied.
    assert array.counters=={'write_attempts':4,'device_writes':3,'stuck_write_attempts':1,'gmin_clipped_writes':2}
    clone=ConductanceArray(array.state_dict(),'cpu')
    for a in (array,clone):a.write(command,noise_factor=noise_factor,conductance_floor='gmin')
    assert torch.equal(array.held,clone.held)
    assert torch.equal(array.generator.get_state(),clone.generator.get_state())
    assert array.counters==clone.counters
    array.noise_scale=float('inf')
    with pytest.raises(FloatingPointError,match='Nonfinite post-write'):
        array.write(command,noise_factor=1.,conductance_floor='gmin')


def test_noiseless_writes_preserve_threshold_initial_state_faults_and_rng():
    from ebl.runtime.opt_cmo_drn.recovery import RecoveryWritePort,RecoveryStep
    from hardware.devices.cmo import HALF_RANGE_US
    model,teacher=tiny_model()
    _,_,state=program_rewritten(model,271002,'random',.1,capture_state=True)
    noisy=CmoDrnBank(state,'cpu');noiseless=CmoDrnBank(state,'cpu')
    for a,b in zip(noisy.arrays,noiseless.arrays):
        initial=b.held.clone()
        p=RecoveryWritePort(a,1.);q=RecoveryWritePort(b,0.)
        x=ThresholdedWriteAdam(p,learning_rate=.03,cap=None,noise_scale=a.noise_scale)
        y=ThresholdedWriteAdam(q,learning_rate=.03,cap=None,noise_scale=b.noise_scale)
        assert x.threshold_q==y.threshold_q>0
        assert torch.equal(a.held,b.held)
        gradient=torch.ones_like(a.held)
        x.step(gradient);y.step(gradient)
        commands=q.read()*0-.03/(1+1e-8)
        expected=torch.where(b._mask,initial,initial+HALF_RANGE_US*commands)
        assert torch.equal(b.held,expected)
        assert not torch.equal(a.held,b.held)
        assert torch.equal(a.generator.get_state(),b.generator.get_state())
        assert a.counters==b.counters
        assert torch.equal(b.held[b._mask],initial[b._mask])
    c=replace(Config(),recovery=replace(Recovery(),write_noise_factor=0.))
    engine=RecoveryStep(model,teacher,noiseless,c)
    wrong=engine.state_dict();wrong['write_noise_factor']=1.
    with pytest.raises(ValueError,match='write-noise'):engine.load_state_dict(wrong)


def test_recovery_campaign_preserves_the_learning_trajectory(tmp_path):
    from ebl.runtime.opt_cmo_drn.runtime import weights_payload,save
    from ebl.runtime.store import sha256_file
    from ebl.lifecycle.recovery import build_plan
    import sys
    model,_=tiny_model()
    teacher=tmp_path/'teacher.pt';torch.save({},teacher)
    config=replace(Config(),teacher_sha256=sha256_file(teacher))
    source=tmp_path/'selected.pt';save(source,weights_payload(model,config,selection={'step':512,'teacher_kl':.1}))
    plan=build_plan(asdict(config),teacher,source,tmp_path/'plan',sys.executable,Path.cwd())
    assert len(plan['stages'])==22
    stages={s['id']:s for s in plan['stages']}
    for kind in ('none','random'):
        assert stages[kind+'-learning_calibration']['inputs']['device_state']=={
            'stage':kind+'-learning','artifact_kind':'recovery_state'}
        for arm in ('hold','calibration','learning','rewrite'):
            assert stages[kind+'-'+arm]['inputs']['device_state']=={
                'stage':'deploy-'+kind,'artifact_kind':'device_state'}
