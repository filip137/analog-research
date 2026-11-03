"""Equation, coupling, clipping and cached-reference metric contracts."""
import math
import pytest
import torch
from training.power_law_drift import PowerLaw, snapshot, aggregate_statistics
from training.drift_evaluation import reference_cache, evaluate


def spec(**kwargs):
    return PowerLaw.parse(dict(mean_us=-2.,sigmas_us=[0.,1.,3.,12.],mean_exponent=.23,
        spread_exponent=.11,reference_seconds=3600.,development_sequences=2,
        first_read_policy='clamp_programmed',**kwargs))


def test_baseline_scaling_clipping_and_nonrecursive_replay(tmp_path):
    p=spec()
    gp=torch.tensor([9.,40.,60.,88.199997])
    eps=torch.tensor([-1.,-.5,.5,1.])
    original=gp.clone()
    assert torch.equal(snapshot(gp,eps,0,12,p,9.,88.199997)[0],gp)
    for age in (3600.,86400.,2592000.,31536000.):
        out,stats=snapshot(gp,eps,age,3,p,9.,88.199997)
        expected=(gp-2*(age/3600)**.23+3*(age/3600)**.11*eps).clamp(9.,88.199997)
        assert torch.equal(out,expected)
        assert stats['clipped_low']>0
    assert torch.equal(gp,original)
    path=tmp_path/'state.pt';torch.save({'gp':gp,'eps':eps},path)
    restored=torch.load(path,weights_only=True)
    assert torch.equal(snapshot(restored['gp'],restored['eps'],3600,3,p,9.,88.199997)[0],
                       snapshot(gp,eps,3600,3,p,9.,88.199997)[0])


def test_coupled_draw_spread_and_weighted_statistics():
    p=spec();gp=torch.full((10000,),40.)
    eps=torch.randn(gp.shape,generator=torch.Generator().manual_seed(9))
    a,sa=snapshot(gp,eps,3600,1,p,9,88.199997)
    b,sb=snapshot(gp,eps,3600,3,p,9,88.199997)
    torch.testing.assert_close(b-gp+2,3*(a-gp+2),atol=1e-5,rtol=1e-5)
    stats=aggregate_statistics([sa,sb]);delta=torch.cat((a-gp,b-gp)).double()
    assert stats['realized_mean_us']==pytest.approx(float(delta.mean()))
    assert stats['realized_std_us']==pytest.approx(float(delta.std(unbiased=False)))
    assert 30**p.spread_exponent==pytest.approx(1.453740,rel=1e-5)


@pytest.mark.parametrize('key,value',[('sigmas_us',[1,0]),('sigmas_us',[float('nan')]),
    ('mean_exponent',0),('spread_exponent',True),('reference_seconds',-1),
    ('development_sequences',0),('first_read_policy','silent')])
def test_strict_settings(key,value):
    raw=spec().to_dict();raw[key]=value
    with pytest.raises(ValueError):PowerLaw.parse(raw)


def test_cache_and_token_weighted_metric_use_same_frozen_targets(tmp_path):
    torch.set_num_threads(1)
    tokens=torch.tensor([[0,1,2],[2,1,0]])
    logits=torch.randn(3,3,generator=torch.Generator().manual_seed(5))
    def f(batch):return logits[batch[:,:-1]]
    cache=reference_cache(tokens,f,f,tmp_path/'refs',{'source':'test'})
    result=evaluate(tokens,f,cache,'cpu')
    assert abs(result['clean_hwa_kl'])<1e-6
    assert result['scored_tokens']==4
    expected=torch.nn.functional.cross_entropy(f(tokens).flatten(0,1),tokens[:,1:].flatten())
    assert result['perplexity']==pytest.approx(math.exp(float(expected)),rel=1e-6)
