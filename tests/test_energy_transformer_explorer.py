"""Energy, local EP and bounded-input checks for explicit attention variants."""
from dataclasses import asdict,replace
import math
import pytest
import torch
from networks.energy_transformer.input_attention import InputAttentionNetwork,certificate,norms
from networks.energy_transformer.singular_projection import project
from networks.energy_transformer.finite_time import finite_ep,bptt_gradient,agreement
from networks.energy_transformer.gpu import check_domain

VARIANTS=[dict(score_gain=8.),dict(score_gain=16.),dict(gain=2.,feedback=.05),
          dict(gain=4.,feedback=.025),dict(key_mode='observed'),
          dict(query_context='observed_mean'),dict(score_gain=8.,gain=2.,feedback=.05)]

def fixture(settings):
    m=InputAttentionNetwork(tokens=3,width=8,patch_width=2,heads=2,dtype=torch.float64,**settings)
    project(m);gen=torch.Generator().manual_seed(81)
    x=torch.randn(2,3,4,generator=gen,dtype=torch.float64)*.2
    x[...,2:]=torch.tensor([[1.,1.],[0.,0.],[1.,1.]])
    x[:,1,:2]=0.
    y=torch.randn(2,3,2,generator=gen,dtype=torch.float64)*.2
    z=torch.randn(2,3,8,generator=gen,dtype=torch.float64)*.2
    return m,x,y,z

@pytest.mark.parametrize('settings',VARIANTS)
@pytest.mark.parametrize('attention',['normal','uniform','zero'])
def test_variant_force_and_all_weight_statistics_derive_from_one_energy(settings,attention):
    m,x,y,z=fixture(settings)
    for p in m.parameters():p.requires_grad_(True)
    z.requires_grad_(True);y.requires_grad_(True)
    mask=torch.ones_like(y);target=y.detach()*.5
    v=m.ports(z,y,attention,context=m.prepare(x));dz,dy=m.force(v,x,target,mask,-.005)
    derivatives=torch.autograd.grad(m.energy(v,x,target,mask,-.005,attention).sum(),(z,y,*m.parameters()),allow_unused=True,retain_graph=True)
    expected=-torch.autograd.grad((v['g']*dz.detach()).sum(),z)[0]
    torch.testing.assert_close(derivatives[0],expected,rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(derivatives[1],-dy,rtol=2e-11,atol=2e-12)
    for (name,p),actual in zip(m.named_parameters(),derivatives[2:]):
        torch.testing.assert_close(m.statistics(v,x)[name],torch.zeros_like(p) if actual is None else actual,rtol=2e-10,atol=2e-12)
    assert check_domain(m,v)['valid'].all()

@pytest.mark.parametrize('settings',VARIANTS)
def test_variant_ep_and_bptt_agree(settings):
    m,x,target,z=fixture(settings);mask=torch.ones_like(target)
    ep,_=finite_ep(m,x,target,mask,128);bp,_=bptt_gradient(m,x,target,mask,128)
    match=agreement(ep,bp)
    assert match['global']['cosine']>.9999
    assert all(v['cosine']>.9999 for v in match['groups'].values())


def test_context_queries_use_observed_inputs_and_preserve_the_norm_bound():
    m,x,y,z=fixture(dict(query_context='observed_mean'))
    context=m.prepare(x)
    torch.testing.assert_close(context['query_f'].norm(dim=-1),context['f'].norm(dim=-1),atol=1e-14,rtol=1e-14)
    assert context['query_f'].norm(dim=-1).max()<=math.sqrt(m.width)
    changed=x.clone();changed[:,0,:2]+=.2
    assert not torch.equal(context['q'][:,:,1],m.prepare(changed)['q'][:,:,1])
    local,_,_,_=fixture({})
    assert torch.equal(local.prepare(x)['q'][:,:,1],local.prepare(changed)['q'][:,:,1])


def test_hidden_key_branches_are_disconnected_in_normal_and_uniform_attention():
    m,x,y,z=fixture(dict(key_mode='observed'));context=m.prepare(x)
    for intervention in ('normal','uniform'):
        v=m.ports(z,y,intervention,context=context)
        assert torch.equal(v['p'][...,1],torch.zeros_like(v['p'][...,1]))
        torch.testing.assert_close(v['p'].sum(-1),torch.ones_like(v['p'].sum(-1)))
        assert check_domain(m,v)['valid'].all()
    x[...,2:]=0.
    with pytest.raises(ValueError,match='observed'):m.prepare(x)


def test_variants_keep_initial_parameters_and_contraction_certificate_identical():
    from ebl.runtime.energy_transformer_mnist import build_model,initialize_model
    from ebl.schema.energy_transformer_mnist import parse_config
    base=dict(case='more_tokens',energy_architecture='input_conditioned',initialization='input_attention_v1',
              recurrent_constraint='input_attention_singular_v1',phase_mode='fixed_steps',device_domain='physical',solver_dt=1.,max_steps=32)
    settings=[{},dict(attention_score_gain=16.),dict(attention_gain=4.,attention_feedback=.025),
              dict(attention_keys='observed'),dict(query_context='observed_mean')]
    models=[]
    for s in settings:
        c=parse_config({**base,**s});m=build_model(c,device='cpu');initialize_model(m,c);models.append(m)
    for m in models[1:]:
        for k,p in models[0].state_dict().items():assert torch.equal(p,m.state_dict()[k])
        assert certificate(m,norms(m))==certificate(models[0],norms(models[0]))


def test_legacy_checkpoints_receive_only_default_attention_options(tmp_path):
    from ebl.schema.energy_transformer_mnist import Config
    from ebl.runtime.energy_transformer_mnist import build_model,save_checkpoint,restore_checkpoint
    c=Config(arm='digital_bp');m=build_model(c,device='cpu');opt=torch.optim.Adam(m.parameters(),lr=c.learning_rate)
    a=torch.Generator().manual_seed(1);b=torch.Generator().manual_seed(2)
    p=tmp_path/'old.pt';save_checkpoint(p,m,opt,c,1,2.,a,b)
    payload=torch.load(p,weights_only=False)
    for key in ('attention_score_gain','attention_gain','attention_feedback','attention_keys','query_context'):del payload['config'][key]
    torch.save(payload,p)
    assert restore_checkpoint(p,m,opt,c,a,b)==(1,2.)


def test_screen_selection_uses_only_predeclared_validation_rule():
    from experiments.run_energy_transformer_explorer import select_confirmations
    screens=[dict(name='b',status='screen_complete',probe_mse=.09),dict(name='a',status='screen_complete',probe_mse=.09),
             dict(name='weak',status='screen_complete',probe_mse=.0999),dict(name='failed',status='gradient_rejected',probe_mse=.01)]
    assert select_confirmations(screens,.1)==['a','b']
    assert select_confirmations(screens,.08)==[]


def test_all_explorer_recipes_are_strict_public_configs():
    import json
    from pathlib import Path
    from ebl.registry import resolve_experiment_config
    from ebl.schema.base import RunMode
    root=Path(__file__).resolve().parents[1]/'examples/energy_transformer_mnist/attention_explorer'
    paths=list(root.rglob('*.json'))
    assert len(paths)==89
    for path in paths:
        mode=RunMode.VALIDATE if path.name.startswith('gradient-') else RunMode.TRAIN
        _,config=resolve_experiment_config(path,mode)
        if mode==RunMode.TRAIN:
            assert config.attention_gain*config.attention_feedback==.1
            assert config.precision=='fp32'


@pytest.mark.parametrize('processes,device,process_code,device_code,allowed',[
    ('','23000, 0',0,0,True),('1234\n','23000, 0',0,0,False),
    ('','23000, 0',1,0,False),('','',0,1,False),('','1000, 0',0,0,False)])
def test_admission_never_treats_busy_or_unknown_capacity_as_free(monkeypatch,processes,device,process_code,device_code,allowed):
    from types import SimpleNamespace
    import experiments.admit_energy_transformer_explorer as admission
    replies=iter([SimpleNamespace(returncode=process_code,stdout=processes),SimpleNamespace(returncode=device_code,stdout=device)])
    monkeypatch.setattr(admission.subprocess,'run',lambda *a,**k:next(replies))
    assert admission.gpu_available()[0] is allowed
