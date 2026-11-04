from methods.hwa.opt import hardware_view
"""Numerical and continuation contracts for the OPT DRN first study."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ebl.schema.opt_cmo_drn import Config, Training, parse_config
from networks.opt.decoder import OptConfig, OptDecoder
from networks.opt.drn import SourceDecoder, reconstruct_source, OptDrn, DrnMlp
from ebl.runtime.opt_cmo_drn.runtime import CoreStep, make_optimizer, checkpoint_payload, restore_checkpoint
from methods.programming.cmo import rewrite_plan, partner, program_rewritten
from networks.resistive.differentiable import DifferentiableDrn, DrnSettings, diode
from methods.core.engine import train_epoch
from methods.core.guards import FiniteGradientGuard


@pytest.fixture(autouse=True)
def cpu_threads():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_model(checkpointing=False):
    torch.manual_seed(7)
    teacher=OptDecoder(OptConfig(17,4,12,2,6,16)).eval().requires_grad_(False)
    source=SourceDecoder(teacher.config)
    source.load_state_dict(teacher.state_dict())
    source.start=6
    source.register_buffer('gains',torch.ones(12))
    settings=DrnSettings(input_gain=5,token_chunk_size=3,activation_checkpointing=checkpointing)
    return OptDrn(source,settings),teacher


def test_full_span_four_device_mapping_and_rail_layouts():
    from hardware.devices.cmo import G_MAX_US
    fc1=torch.nn.Linear(3,1)
    fc2=torch.nn.Linear(1,3)
    with torch.no_grad():
        fc1.weight.copy_(torch.tensor([[2.,-1.,0.]]))
        fc2.weight.copy_(torch.tensor([[2.],[-1.],[0.]]))
    block=DrnMlp(fc1,fc2,torch.ones(2),DrnSettings())
    g1,g2=(g*G_MAX_US for g in block.conductances())
    # Full positive, half negative, and zero weights; hidden rails are halves,
    # output rails adjacent pairs. Every weight has four physical devices.
    torch.testing.assert_close(g1,torch.tensor([
        [88.2,9.], [9.,48.6], [9.,9.],
        [9.,88.2], [48.6,9.], [9.,9.]]))
    torch.testing.assert_close(g2,torch.tensor([
        [88.2,9.,9.,48.6,9.,9.],
        [9.,88.2,48.6,9.,9.,9.]]))
    assert g1.numel()==4*fc1.weight.numel()
    assert g2.numel()==4*fc2.weight.numel()


def test_mapping_artifact_requires_full_span_initialization():
    from ebl.runtime.opt_cmo_drn.runtime import from_mapped,weights_payload
    model,teacher=tiny_model()
    config=Config()
    payload=weights_payload(model,config)
    assert payload['contract']['initialization']=='full_span_quad.v1'
    from_mapped(teacher,config,payload)
    del payload['contract']['initialization']
    with pytest.raises(ValueError,match='initialization'):
        from_mapped(teacher,config,payload)


def test_physics_matches_existing_resistive_interaction():
    from networks.resistive.interaction import DenseResistive
    from networks.resistive.layer import NonlinearResistiveLayer
    from networks.variable.layer import LinearLayer
    from networks.variable.parameter import DenseWeight
    torch.manual_seed(3)
    x=torch.randn(3,2,dtype=torch.double)
    g1=torch.rand(4,6,dtype=torch.double)+.3
    g2=torch.rand(6,4,dtype=torch.double)+.3
    current=torch.randn(6,dtype=torch.double)
    settings=DrnSettings(input_gain=2,activation_checkpointing=False)
    inp=LinearLayer((4,),batch_size=3,device='cpu')
    hidden=NonlinearResistiveLayer((6,),batch_size=3,device='cpu',non_linearity='perfect_diode')
    out=LinearLayer((4,),batch_size=3,device='cpu')
    inp.state=torch.cat((x,-x),1)*2
    hidden.state=torch.zeros(3,6,dtype=torch.double)
    out.state=torch.zeros(3,4,dtype=torch.double)
    w1=DenseWeight((4,),(6,),1,'cpu');w1.state=g1
    w2=DenseWeight((6,),(4,),1,'cpu');w2.state=g2
    e1=DenseResistive(inp,hidden,w1,5,.1,logical_pre_index=0,logical_post_index=1)
    e2=DenseResistive(hidden,out,w2,5,.1,logical_pre_index=1,logical_post_index=2)
    for _ in range(4):
        a=e1.a_coef_fn(hidden)()+e2.a_coef_fn(hidden)()
        b=e1.b_coef_fn(hidden)()+e2.b_coef_fn(hidden)()-current
        hidden.state=diode(-b/(2*a))
        out.state=-e2.b_coef_fn(out)()/(2*e2.a_coef_fn(out)())
    solver=DifferentiableDrn(settings).eval()
    actual=solver(x,g1,g2,current,x.new_tensor(1),x.new_tensor(1))
    torch.testing.assert_close(actual,out.state[:,0::2]-out.state[:,1::2],rtol=1e-12,atol=1e-12)


def test_input_and_conductance_gradcheck():
    torch.manual_seed(3)
    settings=DrnSettings(settling_iterations=0,gradient_iterations=4,activation_checkpointing=False,input_gain=2)
    solver=DifferentiableDrn(settings).train()
    args=(torch.randn(2,2,dtype=torch.double,requires_grad=True),
          (torch.rand(4,6,dtype=torch.double)+.2).requires_grad_(),
          (torch.rand(6,4,dtype=torch.double)+.2).requires_grad_(),
          torch.randn(6,dtype=torch.double,requires_grad=True),torch.tensor(1.,dtype=torch.double),torch.tensor(1.,dtype=torch.double))
    assert torch.autograd.gradcheck(solver,args,rtol=1e-3,atol=1e-5)


def test_all_six_blocks_and_prefix_receive_gradients():
    model,teacher=tiny_model()
    tokens=torch.tensor([[1,2,3,4],[4,3,2,1]])
    model(tokens).square().mean().backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert model.digital.embed_tokens.weight.grad.abs().sum()>0
    for b in model.blocks:
        assert b.u1.grad.abs().sum()>0
        assert b.u2.grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())


def test_checkpoint_recomputation_uses_identical_held_view():
    plain,_=tiny_model(False)
    checked,_=tiny_model(True)
    checked.load_state_dict(plain.state_dict())
    tokens=torch.tensor([[1,2,3,4]])
    view,_=hardware_view(plain,.03,torch.Generator().manual_seed(11))
    a=plain(tokens,view);b=checked(tokens,view)
    a.square().mean().backward();b.square().mean().backward()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for p,q in zip(plain.parameters(),checked.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_held_view_controls_forwards_even_when_master_changes():
    model,_=tiny_model()
    model.eval()
    tokens=torch.tensor([[1,2,3,4]])
    view,_=hardware_view(model,.03,torch.Generator().manual_seed(11))
    before=model(tokens,view)
    with torch.no_grad(): model.blocks[0].u1.add_(.1)
    torch.testing.assert_close(model(tokens,view),before,atol=1e-6,rtol=1e-6)


def test_rewrite_restores_feasible_pair_difference_and_changes_loading():
    target=torch.tensor([[30.,40.,25.,35.]])
    observed=target.clone();observed[0,0]=40
    selected,wanted,flags,clipped=rewrite_plan(observed,target)
    assert selected.tolist()==[[False,False,True,False]]
    assert clipped==0
    torch.testing.assert_close(wanted-partner(wanted),target-partner(target))
    assert wanted.sum()!=target.sum()


def test_faults_and_programming_are_exactly_replayed():
    model,_=tiny_model()
    a,ra=program_rewritten(model,211001,'random',.03)
    b,rb=program_rewritten(model,211001,'random',.03)
    assert ra==rb
    for aa,bb in zip(a,b):
        for x,y in zip(aa,bb): assert torch.equal(x,y)


def test_exact_optimizer_and_rng_continuation():
    model,teacher=tiny_model()
    config=replace(Config(),training=replace(Training(),effective_batch=2,microbatch=1,recipe='B'))
    optimizer=make_optimizer(model,config)
    generator=torch.Generator().manual_seed(9)
    core=CoreStep(model,teacher,optimizer,config,generator)
    tokens=torch.tensor([[1,2,3,4],[4,3,2,1]])
    def step(c): train_epoch(c.components(),[(tokens,tokens)],event_handlers=(FiniteGradientGuard(),))
    step(core)
    saved=copy.deepcopy(checkpoint_payload(model,optimizer,config,generator,1,0,1,{'teacher_kl':1.},{}))
    step(core)
    expected=copy.deepcopy(model.state_dict())
    restored,_=tiny_model()
    other=make_optimizer(restored,config)
    rng=torch.Generator().manual_seed(0)
    restore_checkpoint(saved,restored,other,config,rng,{})
    step(CoreStep(restored,teacher,other,config,rng))
    for k,v in restored.state_dict().items(): assert torch.equal(v,expected[k]),k
    assert torch.equal(rng.get_state(),generator.get_state())


def test_strict_assigned_contract():
    with pytest.raises(ValueError): parse_config({'solver':{'settling_iterations':8}})
    with pytest.raises(ValueError): parse_config({'training':{'unknown':1}})
    with pytest.raises(ValueError): parse_config({'training':{'microbatch':True}})
    with pytest.raises(ValueError): parse_config({'output_dir':'/tmp/surprise'})
    with pytest.raises(ValueError): parse_config({'initialization':'scale_search.v1'})
    assert parse_config({}).solver.gradient_iterations==4


def test_all_opt_examples_resolve_through_public_cli(tmp_path):
    import io
    from ebl.cli import main,CommandHandlers
    from ebl.registry import resolve_experiment_config
    from ebl.schema.base import RunMode
    examples=Path(__file__).resolve().parents[1]/'examples/opt_cmo_drn'
    for path in sorted(examples.glob('*.json')):
        definition,config=resolve_experiment_config(path,RunMode.TRAIN)
        assert definition.experiment_id==config.experiment_id=='opt_cmo_drn_hwa.v1'
        captured=[]
        error=io.StringIO()
        result=main(['train','--config',str(path),'--teacher-weights','teacher.pt',
                     '--weights','source.pt','--device-state','physical.pt','--output-dir',str(tmp_path)],
                    handlers=CommandHandlers(train=lambda request:captured.append(request)),stderr=error)
        assert result==0,error.getvalue()
        assert captured[0].spec==config
        assert captured[0].device_state==Path('physical.pt')


def test_pinned_device_goldens():
    from hardware.faults import hierarchical_faults,MatrixSpec,FaultCase
    from hardware.devices.cmo import CmoMatrix
    golden=torch.load(Path(__file__).parent/'golden/opt_cmo_drn/device.pt',weights_only=True)
    layout=[MatrixSpec('decoder.layers.6.fc1',(4,6),0,24),MatrixSpec('decoder.layers.6.fc2',(6,4),24,24)]
    faults=hierarchical_faults(layout,4,3,FaultCase('random',.25,4.,6.),77,'golden',domains={m.name:'decoder.layers.6' for m in layout})
    assert faults['receipt']==golden['faults']['receipt']
    for name in faults['matrices']:
        for a,b in zip(faults['matrices'][name],golden['faults']['matrices'][name]): assert torch.equal(a,b)
    plant=CmoMatrix(24,29,2.,1.,'cpu',faults['matrices'][layout[0].name])
    plant.program(golden['q']);assert torch.equal(plant.g,golden['endpoint'])
    plant.write(torch.full_like(golden['q'],.01));assert torch.equal(plant.g,golden['written'])


def source_payload(teacher):
    state={}
    qs=[];scales=[];channels=[]
    for k,v in teacher.state_dict().items():
        if not any(k.startswith(f'layers.{i}.{m}.') for i in range(6,12) for m in ('fc1','fc2','final_layer_norm')):
            state['digital.'+k]=v.clone()+.001
    for k,i in enumerate(range(6,12)):
        for j,name in enumerate(('fc1','fc2')):
            layer=getattr(teacher.layers[i],name)
            scale=layer.weight.abs().amax(1)
            qs.append((layer.weight/scale[:,None]).flatten());scales.append(layer.weight.abs().max());channels.append(scale)
            state[f'mlp_biases.{2*k+j}']=layer.bias.clone()
        for name in ('weight','bias'):state[f'mlp_norms.{k}.{name}']=getattr(teacher.layers[i].final_layer_norm,name).clone()
    state.update(q=torch.cat(qs),scales=torch.stack(scales),channel_scales=torch.cat(channels),output_gains=torch.linspace(.8,1.2,12))
    return {'model':state,'arm':{'scope':'model'}}


def test_reconstruction_preserves_digital_parameters_and_gains():
    _,teacher=tiny_model()
    payload=source_payload(teacher)
    source=reconstruct_source(teacher,payload)
    for i in range(6,12):
        torch.testing.assert_close(source.layers[i].fc1.weight,teacher.layers[i].fc1.weight)
    assert torch.equal(source.gains,payload['model']['output_gains'])
    assert torch.equal(source.embed_tokens.weight,payload['model']['digital.embed_tokens.weight'])
    tokens=torch.tensor([[1,2,3,4]])
    assert not torch.equal(source(tokens),teacher(tokens))
    del payload['model']['digital.embed_tokens.weight']
    with pytest.raises(ValueError):reconstruct_source(teacher,payload)


def test_tiny_native_lifecycle_and_collection(tmp_path,monkeypatch):
    import ebl.runtime.opt_cmo_drn.runtime as runtime
    from ebl.runtime.store import sha256_file
    from ebl.schema.opt_cmo_drn import Data,Selection
    from ebl.lifecycle.plan import collect
    _,teacher=tiny_model()
    teacher_path=tmp_path/'teacher.pt';torch.save({},teacher_path)
    payload=source_payload(teacher);payload['context']={'teacher_sha256':sha256_file(teacher_path)}
    source_path=tmp_path/'source.pt';torch.save(payload,source_path)
    tokens=torch.tensor([[1,2,3,4],[4,3,2,1],[3,4,1,2],[2,1,4,3]])
    config=replace(Config(),stage='prepare',source_sha256=sha256_file(source_path),teacher_sha256=sha256_file(teacher_path),
        data=replace(Data(),sequences=4,development_sequences=2,sequence_length=4),
        training=replace(Training(),epochs=1,max_updates=2,effective_batch=2,microbatch=1,warmup_updates=0,checkpoint_every=1),
        selection=replace(Selection(),every_updates=2))
    monkeypatch.setattr(runtime,'teacher_model',lambda path,c:copy.deepcopy(teacher))
    monkeypatch.setattr(runtime,'cohorts',lambda c:(tokens,tokens[:2],{'training':'tiny','development':'tiny'}))
    original_save=runtime.save
    def retain_intermediate(path,payload):
        original_save(path,payload)
        if path.name=='latest.pt' and payload['step']==1:
            original_save(tmp_path/'intermediate.pt',payload)
    monkeypatch.setattr(runtime,'save',retain_intermediate)
    request=SimpleNamespace(spec=config,teacher_weights=teacher_path,weights=source_path,resume=None,output_dir=tmp_path/'prepare',command=('test',))
    assert runtime.run_train(request)==0
    prepared=next((tmp_path/'prepare').glob('*'))
    mapped=torch.load(prepared/'checkpoints/mapped.pt',weights_only=True)
    assert all(row['fractions']==(1.,1.) for row in mapped['mapping'])
    assert all('candidates' not in row for row in mapped['mapping'])
    assert all(row['readout_scale']>0 for row in mapped['mapping'])
    runs=[prepared]
    for arm in ('clean','std3'):
        request.spec=replace(config,stage='hwa',training=replace(config.training,arm=arm))
        request.weights=prepared/'checkpoints/mapped.pt';request.output_dir=tmp_path/arm
        assert runtime.run_train(request)==0
        runs.append(next((tmp_path/arm).glob('*')))
    report=collect(runs)
    assert report['complete']
    assert all(r['inspection']['valid'] for r in report['runs'])
    request.weights=None;request.resume=tmp_path/'intermediate.pt';request.output_dir=tmp_path/'resume'
    assert runtime.run_train(request)==0
    resumed=next((tmp_path/'resume').glob('*'))
    actual=torch.load(resumed/'checkpoints/final.pt',weights_only=True)['model']
    expected=torch.load(runs[-1]/'checkpoints/final.pt',weights_only=True)['model']
    assert all(torch.equal(actual[k],v) for k,v in expected.items())
    assert collect([resumed])['complete']
    from ebl.lifecycle.plan import build_plan
    from dataclasses import asdict
    import sys
    plan=build_plan(asdict(config),teacher_path,source_path,tmp_path/'plan',sys.executable,Path.cwd(),prepared)
    assert len(plan['stages'])==1
    assert plan['stages'][0]['inputs']['weights']['path']==str(prepared/'checkpoints/mapped.pt')
    changed=replace(config,data=replace(config.data,seed=config.data.seed+1))
    with pytest.raises(ValueError):
        build_plan(asdict(changed),teacher_path,source_path,tmp_path/'bad-plan',sys.executable,Path.cwd(),prepared)
    assert not collect([*runs,tmp_path/'missing'])['complete']
    assert collect([runs[0],runs[0]])['runs'][0]['coverage']=='duplicate'
    (runs[-1]/'checkpoints/final.pt').write_bytes(b'corrupt')
    assert not collect([runs[-1]])['complete']


def test_hwa_projection_has_identity_coordinate_gradient():
    model,_=tiny_model()
    with torch.no_grad():
        for p in model.parameter_groups()[0]:p.fill_(-1)
    held,count=hardware_view(model,10.0,torch.Generator().manual_seed(31))
    assert count>0 and all((g>=0).all() for pair in held for g in pair)
    g=model.blocks[0].conductances(held[0])[0]
    g.sum().backward()
    from hardware.devices.cmo import HALF_RANGE_US,G_MAX_US
    torch.testing.assert_close(model.blocks[0].u1.grad,torch.full_like(model.blocks[0].u1,HALF_RANGE_US/G_MAX_US))


def test_tuning_retention_is_explicit():
    assert parse_config({'training':{'retention':'final'}}).training.retention=='final'
    with pytest.raises(ValueError):parse_config({'training':{'retention':'delete_everything'}})
