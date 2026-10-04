"""Scientific pairing, corruption sampler, and immutable checkpoint contracts."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from experiments.cifar_crossbar.sweep_config import SweepSpec, default_config, parse_config, SOURCES, cases
from experiments.cifar_crossbar.sweep_devices import (binding_seed, make_array, combined_om, MixedOmSampler,
    CompactCheckpoints, expand_checkpoint)
from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.runtime import save, load, evaluate
from experiments.cifar_crossbar.full_epoch_runtime import trajectory, restore_endpoint

@pytest.fixture(autouse=True)
def threads():
    prior=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def populations(student,dataset='cifar10',seed=251001):
    layers={}
    for m in student.layout:
        ones=torch.ones(1,m.size)
        layers[m.name]=dict(kind='om',size=m.size,seed=binding_seed(seed,dataset,m.name),
            policy='repaired',variation=1.,aihwkit_version='1.1.0',
            constants=dict(dw_min=.01,dw_min_std=.2,write_noise_std=.3),
            hidden=dict(min_bound=-ones,max_bound=ones,reference=ones*.1,dwmin_up=ones*.01,dwmin_down=ones*.01,
                        corrupt=torch.zeros_like(ones,dtype=torch.bool),published_corrupt=torch.zeros_like(ones,dtype=torch.bool)))
    return {'dataset':dataset,'layers':{str(seed):layers}}


def test_contract_and_counts():
    for dataset in ('cifar10','cifar100'):
        for backend in ('pcm','om'):
            for suffix in ('last_two_blocks','last_four_blocks'):
                s=parse_config(default_config(dataset=dataset,backend=backend,suffix=suffix,
                    calibration_lr=.0003 if dataset=='cifar10' else .001))
                assert sum(len(cases(s,k)) for k in SOURCES)==54
                assert sum(len(cases(s,k)) for k in SOURCES)*3*5*4==3240
                assert all(p!=10000 for k in SOURCES for _,p in cases(s,k,selection=True))
    for bad in ({'epochs':4},{'fault_rates_ppm':(0,20000)},{'suffix':'head'}, {'max_examples':500},
                {'array_seed':211001},{'candidate_learning_rate':.1},{'dataset':'cifar100'}):
        with pytest.raises(ValueError): default_config(**bad)
    from experiments.cifar_crossbar.full_epoch_config import default_config as old
    with pytest.raises(ValueError): old(suffix='last_four_blocks')

@pytest.mark.parametrize('classes',[10,100])
def test_eight_conv_digital_logits_gradient_and_prefix(classes):
    torch.manual_seed(18)
    teacher=CifarResNet32(classes).eval()
    student=CrossbarSuffix(teacher,'last_four_blocks')
    x=torch.randn(2,3,32,32); before=student.prefix_hash()
    torch.testing.assert_close(student(x),teacher(x),atol=2e-6,rtol=2e-5)
    assert student.q.numel()==294912+64*classes
    assert len(student.mapping_receipt()['logical_tiles'])==17
    student(x).square().sum().backward(); teacher(x).square().sum().backward()
    for m in student.layout:
        torch.testing.assert_close(student.q.grad[m.offset:m.offset+m.size],
            teacher.get_submodule(m.name).weight.grad.flatten()*m.scale,atol=2e-6,rtol=2e-4)
    assert student.prefix_hash()==before

@pytest.mark.parametrize('kind',['open','gmax','random'])
def test_pcm_nested_masks_and_common_layer_endpoints(kind):
    teacher=CifarResNet32(10).eval()
    a=CrossbarSuffix(teacher,'last_two_blocks'); b=CrossbarSuffix(teacher,'last_four_blocks')
    spec=SweepSpec()
    arrays=[make_array(a,a.q.detach(),spec,kind,p) for p in (10000,20000,30000,50000)]
    for low,high in zip(arrays,arrays[1:]):
        assert not (low.faults.mask & ~high.faults.mask).any()
        assert torch.equal(low.faults.conductance,high.faults.conductance)
    large=make_array(b,b.q.detach(),replace(spec,suffix='last_four_blocks'),kind,50000)
    for small,big in zip(arrays[-1].children,large.children[-5:]):
        assert torch.equal(small.faults.mask,big.faults.mask)
        assert torch.equal(small.read(),big.read())
    arrays[-1].program(a.q.detach()*.7); arrays[-1].verify_permanence()


def test_om_depth_bindings_and_nested_failures():
    teacher=CifarResNet32(10).eval(); a=CrossbarSuffix(teacher,'last_two_blocks'); b=CrossbarSuffix(teacher,'last_four_blocks')
    pop=populations(b)
    low=combined_om(pop,a.layout,'cifar10',251001,261001,'random',.02)
    high=combined_om(pop,b.layout,'cifar10',251001,261001,'random',.05)
    offset=sum(m.size for m in b.layout[:-5])
    for key in ('reference','published_corrupt'):
        assert torch.equal(low['hidden'][key],high['hidden'][key][:,offset:])
    m=low['hidden']['injected_faults']; mh=high['hidden']['injected_faults'][:,offset:]
    assert not (m & ~mh).any()
    assert torch.equal(low['hidden']['min_bound'][m],high['hidden']['min_bound'][:,offset:][m])
    assert low['trajectory_seed_blocks']==high['trajectory_seed_blocks'][-5:]


def test_om_inflation_does_not_move_stuck_endpoints():
    q=torch.linspace(-1,1,41).repeat_interleave(512).reshape(41,512)
    tables={'healthy':torch.ones_like(q)*.01,'open':-torch.ones_like(q)-q,
            'gmax':torch.ones_like(q)-q,'random':torch.zeros_like(q)-q}
    sampler=MixedOmSampler(tables,41,'cpu')
    target=torch.tensor([-.5,0.,.5],requires_grad=True)
    for kind,value in [('open',-1.),('gmax',1.),('random',0.)]:
        torch.testing.assert_close(sampler.perturb(target,kind,1.,5),torch.full_like(target,value),atol=1e-6,rtol=0)
    torch.testing.assert_close(sampler.perturb(target,'open',0,5),target+.05)

@pytest.mark.parametrize('backend',['pcm','om'])
@pytest.mark.parametrize('method',['calibration','rewrite','onchip_weights','onchip_calibration'])
def test_compact_five_epoch_replay(tmp_path,backend,method):
    teacher=CifarResNet32(10).eval(); student=CrossbarSuffix(teacher,'head')
    features=torch.randn(5,64)
    short={'features':features,'teacher_logits':teacher.fc(features).detach(),'labels':torch.arange(5)}
    cache={k:short for k in ('training','development','test')}
    spec=SweepSpec(backend=backend)
    array=make_array(student,student.q.detach(),spec,'gmax',50000,populations(student))
    directory=tmp_path/'checkpoints'; writer=CompactCheckpoints(tmp_path)
    p0={'model':clone_cpu(student.state_dict()),'array':array.state_dict(),'target':student.q.detach().clone()}
    writer(directory/'p0.pt',p0)
    store=SimpleNamespace(run_dir=tmp_path,append_metric=lambda r:None)
    result=trajectory(student,array,p0['target'],cache,spec,method,store,directory,checkpoint_writer=writer)
    for epoch in (1,5):
        endpoint=expand_checkpoint(load(directory/f'{method}_epoch{epoch}.pt'),tmp_path)
        restored=expand_checkpoint(load(directory/'p0.pt'),tmp_path)
        restore_endpoint(student,array,restored,endpoint)
        assert evaluate(student,short,'cpu',q=array.read())==result['curve'][epoch-1]['test']
    assert len(list((tmp_path/'immutable').glob('population*' if backend=='om' else 'faults*')))==1
    assert result['curve'][-1]['image_presentations']==25


def test_native_cache_artifacts(tmp_path,monkeypatch):
    import json
    from ebl.cli import main
    from experiments.study_workflow import prepare_study,summarize_study
    from experiments.cifar_crossbar import sweep_runtime as runtime
    from experiments.cifar_crossbar.sweep_campaign import prepare
    torch.manual_seed(3)
    teacher=CifarResNet32(10).eval().requires_grad_(False)
    monkeypatch.setenv('EBL_DEFER_CURRENT_SIMULATIONS','1')
    monkeypatch.setattr(runtime,'source_model',lambda *args:(teacher,'fixture'))
    def cache(teacher,student,spec,device):
        features=torch.randn(128,64,8,8)
        logits=student.forward_features(features).detach()
        short={'features':features,'teacher_logits':logits,'labels':logits.argmax(1),'cohort':{'examples':128}}
        return {**{k:short for k in ('training','development','test')},'hwa_raw':torch.zeros(128,3,32,32,dtype=torch.uint8)}
    monkeypatch.setattr(runtime,'build_cache',cache)
    campaign=prepare(tmp_path/'campaign','cifar10','pcm','last_two_blocks',True)
    path=tmp_path/'campaign/configs/cache.json';config=json.loads(path.read_text());config['device']='cpu';path.write_text(json.dumps(config))
    root=prepare_study(tmp_path/'campaign/study-plan.json',tmp_path/'results')
    teacher_path=tmp_path/'teacher.pt';save(teacher_path,{})
    assert main(['train','--config',str(path),'--teacher-weights',str(teacher_path),'--output-dir',str(root/'runs/cache')])==0
    summary=summarize_study(root,verify_artifacts=True)
    assert list((root/'runs/cache').glob('*/checkpoints/cache.pt'))
    text=json.dumps(summary)
    assert 'artifact digest mismatch' not in text.lower()


def test_failed_om_endpoints_do_not_leak_off_grid_target():
    grid=torch.linspace(-1,1,41)[:,None].expand(41,32)
    tables={'healthy':torch.zeros_like(grid),'open':-1-grid,'gmax':1-grid,'random':-grid}
    q=torch.tensor([-.531,.024,.497],requires_grad=True)
    sampler=MixedOmSampler(tables,22,'cpu')
    value=sampler.perturb(q,'gmax',1.,5.)
    torch.testing.assert_close(value,torch.ones_like(q),atol=1e-6,rtol=0)
    value.sum().backward();torch.testing.assert_close(q.grad,torch.ones_like(q))

@pytest.mark.parametrize('capability,disabled,expected',[(12,False,True),(8,False,False),(12,True,False)])
def test_compatible_fp32_execution_policy(monkeypatch,capability,disabled,expected):
    from experiments.cifar_crossbar.sweep_runtime import configure_runtime
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(capability,0))
    result=configure_runtime(SweepSpec(device='cuda:0',disable_cudnn=disabled))
    assert result['cudnn_enabled']==expected
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
