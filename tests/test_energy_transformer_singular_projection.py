"""Mathematical checks for the proposed, not-yet-launched projection control."""
from dataclasses import replace
import math
import torch
import pytest

from networks.energy_transformer.input_attention import InputAttentionNetwork, project as radial_project
from networks.energy_transformer.singular_projection import clip_singular, project

torch.set_num_threads(1)


def test_spectral_ball_projection_preserves_every_unsaturated_singular_value():
    gen=torch.Generator().manual_seed(33)
    u=torch.linalg.qr(torch.randn(6,3,generator=gen,dtype=torch.float64))[0]
    v=torch.linalg.qr(torch.randn(3,3,generator=gen,dtype=torch.float64))[0]
    singular=torch.tensor([4.,.7,.2],dtype=torch.float64)
    matrix=(u*singular)@v.T
    assert clip_singular(matrix,2.)==1
    torch.testing.assert_close(torch.linalg.svdvals(matrix),torch.tensor([2.,.7,.2],dtype=torch.float64))
    expected=(u*torch.tensor([2.,.7,.2],dtype=torch.float64))@v.T
    torch.testing.assert_close(matrix,expected)


def test_repeated_dominant_direction_updates_do_not_erase_other_directions():
    clipped=torch.diag(torch.tensor([.4,.1,.05],dtype=torch.float64));radial=clipped.clone()
    for _ in range(100):
        clipped[0,0]+=.01;radial[0,0]+=.01
        clip_singular(clipped,.4)
        radial*=.4/torch.linalg.matrix_norm(radial,ord=2)
    assert clipped[1,1]==pytest.approx(.1)
    assert clipped[2,2]==pytest.approx(.05)
    assert radial[1,1]<.009
    assert torch.linalg.matrix_norm(clipped,ord=2)<=.4+1e-12


def test_same_energy_bound_and_drive_parameters_are_preserved():
    m=InputAttentionNetwork(tokens=3,width=8,patch_width=2,heads=2,dtype=torch.float64)
    with torch.no_grad():
        for name in ('query','key','value','output','memory','decoder'):getattr(m,name).mul_(4)
    drives={name:getattr(m,name).clone() for name in ('encoder','position')}
    report=project(m)
    assert report['after']['normalized_total']<=.7
    assert report['after']['contraction_radius']<1
    for name,p in drives.items():assert torch.equal(p,getattr(m,name))
    for name in ('query','key','value','output'):
        assert torch.linalg.matrix_norm(getattr(m,name),ord=2).max()<=4/math.sqrt(m.width)+1e-12
    for name,portion in report['budgets'].items():
        assert report['after'][name]<=portion*math.sqrt(m.epsilon)


def test_programming_change_keeps_the_exact_paired_initialization():
    from ebl.schema.energy_transformer_mnist import parse_config
    from ebl.runtime.energy_transformer_mnist import build_model,initialize_model
    base=dict(case='more_tokens',energy_architecture='input_conditioned',
              phase_mode='fixed_steps',device_domain='physical',solver_dt=1.,max_steps=32,
              initialization='input_attention_v1',recurrent_constraint='input_attention_v1')
    a=parse_config(base);b=parse_config({**base,'recurrent_constraint':'input_attention_singular_v1'})
    models=[build_model(c,device='cpu') for c in (a,b)]
    for m,c in zip(models,(a,b)):initialize_model(m,c)
    for name,p in models[0].state_dict().items():assert torch.equal(p,models[1].state_dict()[name])


def test_ep_gradient_agreement_after_singular_projection():
    from networks.energy_transformer.finite_time import finite_ep,bptt_gradient,agreement
    m=InputAttentionNetwork(tokens=3,width=8,patch_width=2,heads=2,dtype=torch.float64)
    with torch.no_grad():m.memory.mul_(3);m.value.mul_(3);m.output.mul_(3)
    project(m);gen=torch.Generator().manual_seed(23)
    x=torch.randn(2,3,4,generator=gen,dtype=torch.float64)*.3
    target=torch.randn(2,3,2,generator=gen,dtype=torch.float64)*.2;mask=torch.ones_like(target)
    ep,_=finite_ep(m,x,target,mask,128);bp,_=bptt_gradient(m,x,target,mask,128)
    result=agreement(ep,bp)
    assert result['global']['cosine']>.99999
    assert all(v['cosine']>.99999 for v in result['groups'].values())



def test_followup_reuses_old_control_and_inherits_original_deadline(tmp_path):
    import json
    from experiments.run_energy_transformer_input_attention import full_cases,prior_comparison
    (tmp_path/'analysis').mkdir()
    launch={'deadline_unix':12345.,'source_commit':'verified-source'}
    result={'outcome':'comparison_collected','full_training_coverage':True,
            'final_gradient_passed':True,'deadline_unix':12345.,'seconds':1241.}
    (tmp_path/'launch.json').write_text(json.dumps(launch))
    (tmp_path/'result.json').write_text(json.dumps(result))
    (tmp_path/'analysis/review.json').write_text(json.dumps({'verdict':{'numerical_and_coverage_pass':True}}))
    prior=prior_comparison(tmp_path)
    assert prior['deadline_unix']==12345. and prior['seconds']==1241.
    assert [n for n,p in full_cases(tmp_path,32,'singular_clip')]==['input-energy_ep','input-energy_bp']
    assert len(full_cases(tmp_path,32,'radial'))==3
    result['final_gradient_passed']=False
    (tmp_path/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='verified'):prior_comparison(tmp_path)
