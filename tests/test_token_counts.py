"""Exercise the actual 256-token path and paired sampler used by exp-004."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from workflow.drift_regimes_arms import prefix_states, suffix_logits, teacher_samples
from workflow.lifecycle import parse_lifecycle
from workflow.networks.opt_mlp import OptConfig, OptDecoder, OptMlpSuffix
import workflow.token_backward as tb
from test_token_backward import dense_stopped, gradients

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'campaigns/drift-regimes'))
sys.path.insert(0, str(ROOT/'campaigns/drift-regimes/launch'))
from make_token_count_lifecycles import documents
from make_token_count_manifest import manifest


def setup():
    torch.manual_seed(3)
    n = OptMlpSuffix(OptDecoder(OptConfig(50, 16, 3, 2, 32, 256)).eval(), 2).eval()
    n.enable_lora(SimpleNamespace(rank=2, alpha=2.), seed=1)
    with torch.no_grad():
        for adapter in n.lora:
            adapter['b'].normal_(0, .02)
    for p in n.parameters(): p.requires_grad_(False)
    for p in n.lora.parameters(): p.requires_grad_(True)
    tokens = torch.randint(0, 50, (3, 256), generator=torch.Generator().manual_seed(13))
    cache = prefix_states(n, tokens, 'cpu')
    cache.samples = teacher_samples(n, cache, 8, 17)
    return n, cache, n.q.detach()*.9


@pytest.mark.parametrize('count', [64, 128])
def test_full_context_and_stopped_gradient_at_requested_counts(count):
    n, cache, q = setup()
    cache = cache[:1]
    pos = torch.randperm(255, generator=torch.Generator().manual_seed(73))[:count].sort().values[None]
    actual = tb.selected_logits(n, cache.student, q, pos)
    assert torch.allclose(actual, tb.take(suffix_logits(n, cache.student, q), pos), atol=3e-6, rtol=2e-5)
    a = gradients(n, tb.sampled_loss(actual, cache, 4, pos))
    b = gradients(n, tb.sampled_loss(dense_stopped(n, cache.student, q, pos), cache, 4, pos))
    assert torch.allclose(a, b, atol=3e-7, rtol=3e-4)


def test_training_pairs_parents_and_nested_masks_and_redraws(monkeypatch):
    captured = {}
    original = tb.selected_logits
    for count in (32, 64, 128):
        n, cache, q = setup()
        calls = []
        def record(network, hidden, q, positions):
            calls.append((hidden.detach().clone(), positions.detach().clone()))
            return original(network, hidden, q, positions)
        monkeypatch.setattr(tb, 'selected_logits', record)
        result = tb.train(n, list(n.lora.parameters()), q, cache, steps=3, batch=2,
                          learning_rate=.003, seed=7, target=('samples', 4),
                          mode=f'random:{count}', microbatch=1)
        assert result['scored_positions'] == result['backward_positions'] == 6*count
        assert result['saved_activation_peak_bytes'] > 0
        assert len(calls) == 6
        assert len({tuple(pos.flatten().tolist()) for _, pos in calls}) == 6
        captured[count] = calls
    for i in range(6):
        assert torch.equal(captured[32][i][0], captured[64][i][0])
        assert torch.equal(captured[64][i][0], captured[128][i][0])
        masks = [set(captured[k][i][1].flatten().tolist()) for k in (32, 64, 128)]
        assert masks[0] < masks[1] < masks[2]
        assert max(masks[2]) < 255


def test_frozen_native_coverage_and_fixed_settings():
    docs = documents()
    assert len(docs) == 5
    for doc in docs:
        parse_lifecycle(doc)
        path = ROOT/'campaigns/drift-regimes/lifecycles'/(doc['lifecycle_id']+'.json')
        assert json.loads(path.read_text()) == doc
        cfg = doc['drift_regimes']
        assert doc['data']['sequence_length'] == 256
        assert cfg['adaptation']['sequences'] == 1024
        assert cfg['adaptation']['batch'] == 8
        assert cfg['adaptation']['steps'] == (3 if cfg['canary'] else 300)
        assert len(cfg['arms']) == 6
        selected = [a for a in cfg['arms'] if 'token_backward' in a]
        assert [a['token_backward'] for a in selected] == ['full', 'random:32', 'random:64', 'random:128']
        assert all(a['learning_rate'] == .003 and a['target'] == 'samples:4' and
                   a['rank'] == 16 and a['microbatch'] == 1 for a in selected)
    m = manifest('/source', '/out')
    assert set(m['configs']) == {'tc-canary', 'tc-dev-271004', 'tc-eval-271001', 'tc-eval-271002', 'tc-eval-271003'}
    assert {p for ps in m['configs'].values() for p in ps} == {
        'campaigns/drift-regimes/lifecycles/'+d['lifecycle_id']+'.json' for d in docs}
    assert m['budget_gpu_seconds'] == 10800
