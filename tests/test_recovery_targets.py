"""Training targets for drift-regimes LoRA arms (on-chip-learning recovery-data-preparation exp-001)."""
from dataclasses import replace
import json

import pytest
import torch
from torch.nn import functional as F

from training.drift_regimes_config import DriftRegimes, parse_target
from test_workflow_cmo import cmo_inputs
from test_workflow_drift_regimes import _run, _stage_inputs, config


def _tiny():
    from workflow.networks.opt_mlp import OptConfig, OptDecoder, OptMlpSuffix
    torch.manual_seed(3)
    teacher = OptDecoder(OptConfig(50, 16, 3, 2, 32, 16)).eval()
    return OptMlpSuffix(teacher, 2).eval()


@pytest.mark.parametrize('value,expected', [('kl', ('kl', None)), ('text', ('text', None)),
                                            ('topk:8', ('topk', 8)), ('samples:8', ('samples', 8))])
def test_targets_parse(value, expected):
    assert parse_target(value) == expected


@pytest.mark.parametrize('value', ['', 'KL', 'topk', 'topk:0', 'topk:08', 'samples:-1', 'text:3', 8, None])
def test_bad_targets_fail_closed(value):
    with pytest.raises(ValueError):
        parse_target(value)


def test_target_is_optional_and_only_on_lora_arms():
    adaptation = {'corpus': 'synthetic', 'corpus_sha256': 'd' * 64, 'split': 'train', 'sequences': 4,
                  'cohort_sha256': 'e' * 64, 'seed': 1, 'probe_sequences': 2, 'steps': 3, 'batch': 2}
    plain = {'label': 'A5', 'method': 'lora', 'events': (60.,), 'rank': 2, 'learning_rate': 1e-3}
    raw = replace(config(), arms=(plain,), adaptation=adaptation).to_dict()
    parsed = DriftRegimes.parse(raw)
    assert 'target' not in parsed.arms[0] and parsed.to_dict() == raw     # old files are unchanged
    raw['arms'] = [{**plain, 'target': 'topk:8'}]
    assert DriftRegimes.parse(raw).arms[0]['target'] == 'topk:8'
    raw['arms'] = [{**plain, 'target': 'nope'}]
    with pytest.raises(ValueError):
        DriftRegimes.parse(raw)
    raw['arms'] = [{'label': 'A3', 'method': 'layernorm', 'events': (60.,), 'learning_rate': 1e-3, 'target': 'text'}]
    with pytest.raises(ValueError):
        DriftRegimes.parse(raw)


def test_topk_with_the_whole_vocabulary_is_the_full_kl_and_smaller_k_is_a_lower_bound():
    from workflow.drift_regimes_arms import topk_tail_kl
    torch.manual_seed(0)
    teacher = (3 * torch.randn(20, 50)).log_softmax(-1)
    student = (3 * torch.randn(20, 50)).log_softmax(-1)
    full = (teacher.exp() * (teacher - student)).sum(-1).mean()
    assert torch.allclose(topk_tail_kl(teacher, student, 50, round_fp16=False), full, atol=1e-5)
    previous = 0.
    for k in (1, 4, 8, 32):         # coarsening never increases KL (data processing inequality)
        value = float(topk_tail_kl(teacher, student, k, round_fp16=False))
        assert -1e-6 <= value <= float(full) + 1e-6
        assert value >= previous - 1e-6
        previous = value
    assert torch.allclose(topk_tail_kl(teacher, student, 8), topk_tail_kl(teacher, student, 8, round_fp16=False),
                          atol=1e-2)
    assert float(topk_tail_kl(teacher, teacher, 8)) < 1e-3


def test_text_target_is_the_next_token_cross_entropy():
    from workflow.drift_regimes_arms import prefix_states, suffix_logits, target_loss
    network = _tiny()
    tokens = torch.randint(0, 50, (3, 8))
    cache = prefix_states(network, tokens, 'cpu', batch=2)
    q = network.q.detach()
    with torch.no_grad():
        logits = suffix_logits(network, cache.student, q)
        student = logits.flatten(0, 1).log_softmax(-1)
        expected = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
        assert torch.allclose(target_loss(('text', None), None, student, cache), expected, atol=1e-6)
        # Indexing a subset keeps tokens aligned with their states.
        part = cache[torch.tensor([2, 0])]
        assert torch.equal(part.tokens_ids, tokens[[2, 0]])


def test_teacher_samples_are_fixed_by_the_seed_and_follow_the_teacher():
    from workflow.drift_regimes_arms import prefix_states, teacher_samples
    network = _tiny()
    cache = prefix_states(network, torch.randint(0, 50, (4, 8)), 'cpu', batch=2)
    a = teacher_samples(network, cache, 8, seed=11, batch=3)
    b = teacher_samples(network, cache, 8, seed=11, batch=3)
    assert a.shape == (4, 7, 8) and torch.equal(a, b)
    assert not torch.equal(a, teacher_samples(network, cache, 8, seed=12, batch=3))
    with torch.no_grad():
        probs = network.teacher_logits(cache.teacher).softmax(-1)
    drawn = probs.gather(-1, a)
    assert float(drawn.mean()) > float(probs.mean())        # samples land on likely tokens


def test_default_target_is_the_original_full_kl_loop():
    """train_digital with the default target equals the pre-change loop, step for step."""
    from workflow.drift_regimes_arms import prefix_states, suffix_logits, train_digital
    from types import SimpleNamespace
    tokens = torch.randint(0, 50, (6, 8), generator=torch.Generator().manual_seed(5))

    def fresh():
        network = _tiny()
        network.enable_lora(SimpleNamespace(rank=2, alpha=2.), seed=1)
        return network

    network = fresh()
    cache = prefix_states(network, tokens, 'cpu')
    q = network.q.detach() * 0.9
    train_digital(network, list(network.lora.parameters()), q, cache, steps=4, batch=2, learning_rate=1e-2, seed=7)
    new = [p.detach().clone() for p in network.lora.parameters()]

    reference = fresh()
    parameters = list(reference.lora.parameters())
    for p in reference.parameters():
        p.requires_grad_(False)
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4)
    generator = torch.Generator().manual_seed(7)
    reference.train()
    for _ in range(4):
        states = cache[torch.randint(0, len(cache), (2,), generator=generator)]
        with torch.no_grad():
            teacher = reference.teacher_logits(states.teacher).flatten(0, 1).log_softmax(-1)
        student = suffix_logits(reference, states.student, q).flatten(0, 1).log_softmax(-1)
        loss = (teacher.exp() * (teacher - student)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
    for a, b in zip(new, parameters):
        assert torch.equal(a, b.detach())


def test_native_stage_runs_every_target(cmo_inputs, tmp_path):
    lc, results, teacher, base, adaptation = _stage_inputs(cmo_inputs, tmp_path)
    lora = {'method': 'lora', 'events': (60., 900.), 'rank': 2, 'learning_rate': 1e-3}
    arm_list = ({'label': 'A0', 'method': 'none'}, {'label': 'A5', **lora},
                {'label': 'T-full', **lora, 'target': 'kl'}, {'label': 'T-text', **lora, 'target': 'text'},
                {'label': 'T-top2', **lora, 'target': 'topk:2'}, {'label': 'T-samp3', **lora, 'target': 'samples:3'})
    cfg = DriftRegimes.parse(replace(base, arms=arm_list, adaptation=adaptation).to_dict())
    directory, metrics = _run(lc, cfg, results, teacher, tmp_path / 'targets')
    arms = metrics['arms']
    assert metrics['source_unchanged'] and metrics['replay_exact']
    # An explicit 'kl' target is the original arm, bit for bit.
    assert arms['T-full']['mean_teacher_kl'] == arms['A5']['mean_teacher_kl']
    kls = {label: arms[label]['mean_teacher_kl'] for label in ('T-text', 'T-top2', 'T-samp3')}
    assert len(set(kls.values()) | {arms['A5']['mean_teacher_kl']}) == 4
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    drawn = [r for r in rows if r.get('phase') == 'teacher_samples']
    assert len(drawn) == 1 and drawn[0]['count'] == 3
    _, again = _run(lc, cfg, results, teacher, tmp_path / 'again')
    assert {k: v['mean_teacher_kl'] for k, v in again['arms'].items()} == \
           {k: v['mean_teacher_kl'] for k, v in arms.items()}
