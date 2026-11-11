from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from test_recovery_targets import _tiny
from test_workflow_drift_regimes import _run, _stage_inputs
from test_workflow_cmo import cmo_inputs
from training.drift_regimes_config import DriftRegimes
from workflow.drift_regimes_arms import prefix_states, suffix_logits, teacher_samples, train_digital
from workflow.token_backward import (SavedActivations, complement, full_hidden, importance_indices,
                                     parse_mode, sampled_loss, selected_logits, take, train)


def setup():
    network = _tiny()
    network.enable_lora(SimpleNamespace(rank=2, alpha=2.), seed=1)
    # Nonzero B makes tests exercise upstream gradient paths through both factors.
    with torch.no_grad():
        for adapter in network.lora:
            adapter['b'].normal_(0, .02)
    for p in network.parameters(): p.requires_grad_(False)
    for p in network.lora.parameters(): p.requires_grad_(True)
    tokens = torch.randint(0, 50, (4, 12), generator=torch.Generator().manual_seed(13))
    cache = prefix_states(network, tokens, 'cpu')
    cache.samples = teacher_samples(network, cache, 8, 17)
    return network, cache, network.q.detach()*.9


def dense_stopped(network, hidden, q, positions):
    mask = torch.zeros(*hidden.shape[:2], 1).scatter(1, positions[..., None], 1.)
    def stop(x): return mask*x + (1-mask)*x.detach()
    for k, i in enumerate(range(network.start, len(network.teacher.layers))):
        hidden = stop(hidden)
        hidden = stop(network.teacher.layers[i].attention(hidden))
        y = network._matrix(network.mlp_norms[k](hidden), 2*k, q)
        hidden = stop(hidden + network._matrix(F.relu(y), 2*k+1, q))
    return network.teacher.logits(take(hidden, positions))


def gradients(network, loss):
    return torch.cat([g.flatten() for g in torch.autograd.grad(loss, list(network.lora.parameters()))])


def test_sparse_forward_preserves_all_context_and_matches_dense_stopped_gradient():
    n, cache, q = setup()
    pos = torch.tensor([[1, 4, 9], [2, 5, 10], [0, 3, 7], [3, 8, 10]])
    actual = selected_logits(n, cache.student, q, pos)
    expected = take(suffix_logits(n, cache.student, q), pos)
    assert torch.allclose(actual, expected, atol=3e-6, rtol=2e-5)
    assert torch.allclose(selected_logits(n, cache.student, q, pos, True),
                          full_hidden(n, cache.student, q), atol=3e-6, rtol=2e-5)
    a = gradients(n, sampled_loss(actual, cache, 4, pos))
    b = gradients(n, sampled_loss(dense_stopped(n, cache.student, q, pos), cache, 4, pos))
    assert torch.allclose(a, b, atol=3e-7, rtol=3e-4)
    full_path = gradients(n, sampled_loss(take(suffix_logits(n, cache.student, q), pos), cache, 4, pos))
    assert not torch.allclose(a, full_path, atol=1e-5, rtol=1e-3)


def test_all_scored_positions_restore_full_gradient():
    n, cache, q = setup()
    pos = torch.arange(11).expand(4, -1)
    a = gradients(n, sampled_loss(selected_logits(n, cache.student, q, pos), cache, 4, pos))
    b = gradients(n, sampled_loss(suffix_logits(n, cache.student, q), cache, 4))
    assert torch.allclose(a, b, atol=3e-7, rtol=3e-4)


def test_no_future_leakage():
    n, cache, q = setup()
    pos = torch.tensor([[1, 2]]).expand(4, -1)
    before = selected_logits(n, cache.student, q, pos)
    changed = cache.student.clone(); changed[:, 3:] += 10
    after = selected_logits(n, changed, q, pos)
    assert torch.equal(before, after)


def test_saved_storage_reduces_and_importance_is_finite_and_weight_preserving():
    n, cache, q = setup()
    cache=cache[:1]
    pos=torch.tensor([[3, 9]])
    peaks=[]
    for sparse in (False, True):
        audit=SavedActivations([*n.parameters(), *n.buffers(), q, cache.student, cache.samples])
        with audit.context():
            logits=selected_logits(n, cache.student, q, pos) if sparse else suffix_logits(n, cache.student, q)
            gradients(n, sampled_loss(logits, cache, 4, pos if sparse else None))
        peaks.append(audit.peak)
    assert peaks[1] < peaks[0]
    old=[p.detach().clone() for p in n.parameters()]
    ids=importance_indices(n, q, cache, 2, 4)
    assert ids.shape==(1,2) and (ids>=0).all() and (ids<11).all() and ids[0,0]!=ids[0,1]
    assert all(torch.equal(a,b) for a,b in zip(old,n.parameters()))


def test_accumulation_matches_native_full_loop():
    n, cache, q = setup()
    old=[p.detach().clone() for p in n.lora.parameters()]
    args=dict(steps=3, batch=4, learning_rate=3e-3, seed=7, target=('samples',4))
    train_digital(n,list(n.lora.parameters()),q,cache,**args)
    expected=[p.detach().clone() for p in n.lora.parameters()]
    with torch.no_grad():
        for p,v in zip(n.lora.parameters(),old): p.copy_(v)
    train(n,list(n.lora.parameters()),q,cache,mode='full',microbatch=1,**args)
    assert all(torch.allclose(a,b,atol=2e-6,rtol=2e-4) for a,b in zip(expected,n.lora.parameters()))


@pytest.mark.parametrize('mode',['random:2','importance:2','loss_only:2','context:3','full'])
def test_native_stage_runs_selection_and_context_modes(cmo_inputs,tmp_path,mode):
    lc,results,teacher,base,adaptation=_stage_inputs(cmo_inputs,tmp_path)
    # Native tiny lifecycle has length 9, so context 3 divides it.
    adaptation={**adaptation,'steps':2}
    arm={'label':'selected','method':'lora','events':(60.,),'rank':2,'learning_rate':1e-3,
         'target':'samples:4','token_backward':mode,'microbatch':1}
    cfg=DriftRegimes.parse(replace(base,arms=({'label':'A0','method':'none'},arm),adaptation=adaptation).to_dict())
    directory,result=_run(lc,cfg,results,teacher,tmp_path/'run')
    assert result['source_unchanged'] and result['replay_exact']
    assert result['arms']['selected']['cost']['optimizer_steps']==2


@pytest.mark.parametrize('value',['random:0','unknown:32','context:01',None,3])
def test_invalid_modes(value):
    with pytest.raises(ValueError): parse_mode(value)
