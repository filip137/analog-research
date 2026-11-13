"""Issue #27 selection controls: random_fixed, importance_sampled, gradient_sampled."""
from dataclasses import replace

import pytest
import torch

import workflow.token_backward as tb
from test_token_backward import setup, dense_stopped, gradients
from test_workflow_drift_regimes import _run, _stage_inputs
from test_workflow_cmo import cmo_inputs  # noqa: F401  (fixture)
from training.drift_regimes_config import DriftRegimes


def test_modes_parse():
    for m in ('random_fixed', 'importance_sampled', 'gradient_sampled'):
        assert tb.parse_mode(m + ':32') == (m, 32)


def test_importance_indices_unchanged_by_score_refactor():
    """The exp-003 selection is the top-k of 0.7*normalized gradient + 0.3*log1p(attention)."""
    n, cache, q = setup()
    scores = torch.cat(list(tb.importance_scores(n, q, cache, 4)))
    assert torch.equal(tb.importance_indices(n, q, cache, 3, 4), scores.topk(3, -1).indices.sort(-1).values)
    gradient_only = torch.cat(list(tb.importance_scores(n, q, cache, 4, attention=False)))
    assert gradient_only.amin(-1).eq(0).all() and gradient_only.amax(-1).eq(1).all()
    assert not torch.equal(gradient_only, scores)


def test_sampling_probabilities_are_a_defensive_mixture():
    s = torch.tensor([[0., 1., 3.], [0., 0., 0.]], dtype=torch.float64)
    p = tb.sampling_probabilities(s, mix=.2)
    assert torch.allclose(p.sum(-1), torch.ones(2, dtype=torch.float64))
    assert (p >= .2 / 3 - 1e-12).all()
    assert torch.allclose(p[1], torch.full((3,), 1 / 3, dtype=torch.float64))


def test_weighted_draw_is_unbiased_for_the_mean_over_positions():
    g = torch.Generator().manual_seed(5)
    length, count = 20, 6
    p = tb.sampling_probabilities(torch.rand(length, generator=g, dtype=torch.float64) ** 3)
    x = torch.randn(length, generator=g, dtype=torch.float64)
    estimates = []
    for _ in range(20000):
        pos, w = tb.draw_weighted(p, count, g)
        assert torch.equal(pos, pos.unique(sorted=True)) and (w > 0).all()
        estimates.append(float((w * x[pos]).sum()))
    estimates = torch.tensor(estimates, dtype=torch.float64)
    standard_error = estimates.std() / len(estimates) ** .5
    assert abs(estimates.mean() - x.mean()) < 4 * standard_error


def test_weighted_loss_matches_manual_sum_and_dense_stopped_gradient():
    n, cache, q = setup()
    cache = cache[:1]
    pos = torch.tensor([[2, 5, 9]])
    w = torch.tensor([[.05, .2, .1]])
    logits = tb.selected_logits(n, cache.student, q, pos)
    loss = tb.weighted_sampled_loss(logits, cache, 4, pos, w)
    targets = tb.take(cache.samples[..., :4], pos)
    manual = (-(logits.log_softmax(-1).gather(-1, targets)).mean(-1) * w).sum()
    assert torch.allclose(loss, manual)
    a = gradients(n, loss)
    b = gradients(n, tb.weighted_sampled_loss(dense_stopped(n, cache.student, q, pos), cache, 4, pos, w))
    assert torch.allclose(a, b, atol=3e-7, rtol=3e-4)


def _capture(monkeypatch, mode, steps=4, batch=3, seed=7):
    n, cache, q = setup()
    calls = []
    original = tb.selected_logits

    def record(network, hidden, q_, positions):
        calls.append((hidden.detach().clone(), positions.detach().clone()))
        return original(network, hidden, q_, positions)
    monkeypatch.setattr(tb, 'selected_logits', record)
    result = tb.train(n, list(n.lora.parameters()), q, cache, steps=steps, batch=batch, learning_rate=1e-3,
                      seed=seed, target=('samples', 4), mode=mode, microbatch=1)
    return cache, calls, result


def test_random_fixed_reuses_one_draw_per_parent_and_pairs_parents_with_random(monkeypatch):
    cache, fixed, result = _capture(monkeypatch, 'random_fixed:3')
    _, redrawn, _ = _capture(monkeypatch, 'random:3')
    by_parent = {}
    for (h, pos), (h2, _) in zip(fixed, redrawn):
        assert torch.equal(h, h2)   # same parent presentations as random
        key = next(i for i in range(len(cache)) if torch.equal(cache.student[i], h[0]))
        by_parent.setdefault(key, set()).add(tuple(pos[0].tolist()))
    assert all(len(v) == 1 for v in by_parent.values())
    assert result['distinct_trained_positions_per_presented_parent'] == 3
    assert result['backward_positions'] == 4 * 3 * 3


@pytest.mark.parametrize('mode', ['importance_sampled:3', 'gradient_sampled:3'])
def test_sampled_modes_redraw_and_report_costs(monkeypatch, mode):
    _, calls, result = _capture(monkeypatch, mode, steps=6)
    sizes = [c[1].shape[1] for c in calls]
    assert all(1 <= s <= 3 for s in sizes)
    assert result['backward_positions'] == sum(sizes) == result['scored_positions']
    assert result['max_importance_weight'] > 0 and result['sampling_uniform_mix'] == tb.UNIFORM_MIX
    assert len({tuple(c[1][0].tolist()) for c in calls}) > 1
    assert all(torch.isfinite(torch.tensor([result['first_loss'], result['last_loss']])))


def test_sampled_modes_require_microbatch_one():
    n, cache, q = setup()
    with pytest.raises(ValueError):
        tb.train(n, list(n.lora.parameters()), q, cache, steps=1, batch=2, learning_rate=1e-3, seed=1,
                 target=('samples', 4), mode='importance_sampled:2', microbatch=2)


@pytest.mark.parametrize('mode', ['random_fixed:2', 'importance_sampled:2', 'gradient_sampled:2'])
def test_native_stage_runs_new_selection_modes(cmo_inputs, tmp_path, mode):
    lc, results, teacher, base, adaptation = _stage_inputs(cmo_inputs, tmp_path)
    adaptation = {**adaptation, 'steps': 2}
    arm = {'label': 'selected', 'method': 'lora', 'events': (60.,), 'rank': 2, 'learning_rate': 1e-3,
           'target': 'samples:4', 'token_backward': mode, 'microbatch': 1}
    cfg = DriftRegimes.parse(replace(base, arms=({'label': 'A0', 'method': 'none'}, arm),
                                     adaptation=adaptation).to_dict())
    directory, result = _run(lc, cfg, results, teacher, tmp_path / 'run')
    assert result['source_unchanged'] and result['replay_exact']
    assert result['arms']['selected']['cost']['optimizer_steps'] == 2


def test_frozen_selection_lifecycles_and_manifest():
    import json
    import sys
    from pathlib import Path
    from workflow.lifecycle import parse_lifecycle
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'campaigns/drift-regimes'))
    sys.path.insert(0, str(root/'campaigns/drift-regimes/launch'))
    from make_token_selection_lifecycles import documents
    from make_token_selection_manifest import manifest
    docs = documents()
    assert len(docs) == 5
    for doc in docs:
        parse_lifecycle(doc)
        path = root/'campaigns/drift-regimes/lifecycles'/(doc['lifecycle_id']+'.json')
        assert json.loads(path.read_text()) == doc
        cfg = doc['drift_regimes']
        assert doc['data']['sequence_length'] == 256 and cfg['adaptation']['sequences'] == 1024
        assert cfg['adaptation']['batch'] == 8 and cfg['adaptation']['steps'] == (3 if cfg['canary'] else 300)
        selected = [a for a in cfg['arms'] if 'token_backward' in a]
        assert [a['token_backward'] for a in selected] == ['random:32', 'random_fixed:32',
                                                          'importance_sampled:32', 'gradient_sampled:32']
        assert all(a['learning_rate'] == .003 and a['target'] == 'samples:4' and a['rank'] == 16
                   and a['microbatch'] == 1 for a in selected)
        assert [a['label'] for a in cfg['arms'][:2]] == ['A0', 'T-samp8-lr0.01']
    m = manifest('/source', '/out')
    assert set(m['configs']) == {'ts-canary', 'ts-dev-271004', 'ts-eval-271001', 'ts-eval-271002', 'ts-eval-271003'}
    assert {p for ps in m['configs'].values() for p in ps} == {
        'campaigns/drift-regimes/lifecycles/'+d['lifecycle_id']+'.json' for d in docs}
    assert m['budget_gpu_seconds'] == 9000
