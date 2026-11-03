"""Zeroth-order attention input errors for the OPT DRN recovery benchmark."""

import copy

import pytest
import torch

from experiments.opt_cmo_drn.zo_attention import AttentionBackward, install
from tests.test_opt_cmo_drn import tiny_model


def master_gradients(model, tokens):
    model.zero_grad(set_to_none=True)
    model(tokens).square().mean().backward()
    return [p.grad.clone() for p in model.parameter_groups()[0]]


@pytest.fixture
def setup():
    model, _ = tiny_model()
    model.requires_grad_(False)
    for p in model.parameter_groups()[0]:
        p.requires_grad_(True)
    tokens = torch.randint(0, 17, (2, 5), generator=torch.Generator().manual_seed(3))
    return model, tokens


def test_exact_mode_leaves_the_model_untouched(setup):
    model, tokens = setup
    reference = master_gradients(model, tokens)
    install(model, AttentionBackward('exact'))
    assert all(torch.equal(a, b) for a, b in zip(reference, master_gradients(model, tokens)))


def test_forward_is_unchanged_and_last_block_gradient_is_exact(setup):
    model, tokens = setup
    with torch.no_grad():
        logits = model(tokens)
    exact = master_gradients(model, tokens)
    patched = install(copy.deepcopy(model), AttentionBackward('zo', 1, seed=1))
    with torch.no_grad():
        assert torch.equal(patched(tokens), logits)
    estimated = master_gradients(patched, tokens)
    # No attention follows the last analog MLP.
    assert all(torch.equal(a, b) for a, b in zip(exact[-2:], estimated[-2:]))
    assert not torch.allclose(exact[0], estimated[0])


def test_residual_mode_drops_only_the_attention_correction(setup):
    model, tokens = setup
    estimator = AttentionBackward('residual')
    estimator.diagnose = True
    install(model, estimator)
    master_gradients(model, tokens)
    assert len(estimator.records) == 5
    assert all(r['estimated_correction_norm'] == 0 for r in estimator.records)
    assert all(r['exact_correction_norm'] > 0 for r in estimator.records)


def test_stop_mode_trains_only_the_last_analog_mlp(setup):
    model, tokens = setup
    exact = master_gradients(model, tokens)
    install(model, AttentionBackward('stop'))
    stopped = master_gradients(model, tokens)
    assert all(torch.equal(a, b) for a, b in zip(exact[-2:], stopped[-2:]))
    assert all(not g.any() for g in stopped[:-2])


def test_many_directions_recover_the_exact_correction(setup):
    model, tokens = setup
    estimator = AttentionBackward('zo', 20000, epsilon=1e-3, seed=5, probe_chunk=4096)
    estimator.diagnose = True
    install(model, estimator)
    master_gradients(model, tokens)
    # 40 flattened input dimensions: Monte Carlo error ~ sqrt(40 / 20000).
    assert min(r['correction_cosine'] for r in estimator.records) > 0.99
    assert estimator.attention_forwards == 5 * 2 * 20000 * len(tokens)


def test_linearized_drops_only_the_softmax_path(setup):
    model, tokens = setup
    layer = model.digital.layers[model.start + 1]
    x = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(9))
    grad = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(10))
    from experiments.opt_cmo_drn.zo_attention import linearized_vjp
    # With uniform attention scores the softmax path vanishes: q = 0.
    frozen = copy.deepcopy(layer)
    frozen.self_attn.q_proj.weight.zero_()
    frozen.self_attn.q_proj.bias.zero_()
    leaf = x.clone().requires_grad_(True)
    exact, = torch.autograd.grad(frozen.self_attn(frozen.self_attn_layer_norm(leaf)), leaf, grad)
    assert torch.allclose(linearized_vjp(frozen, x, grad), exact, atol=1e-6)
    # In general it differs from the exact VJP.
    leaf = x.clone().requires_grad_(True)
    exact, = torch.autograd.grad(layer.self_attn(layer.self_attn_layer_norm(leaf)), leaf, grad)
    assert not torch.allclose(linearized_vjp(layer, x, grad), exact, atol=1e-6)


def test_linearized_control_variate_keeps_zo_unbiased(setup):
    model, tokens = setup
    estimator = AttentionBackward('zo', 20000, epsilon=1e-3, seed=5, probe_chunk=4096, control='linearized')
    estimator.diagnose = True
    install(model, estimator)
    master_gradients(model, tokens)
    assert min(r['correction_cosine'] for r in estimator.records) > 0.99


def test_invalid_declarations_are_rejected():
    with pytest.raises(ValueError):
        AttentionBackward('zo', 0)
    with pytest.raises(ValueError):
        AttentionBackward('residual', 4)
    with pytest.raises(ValueError):
        AttentionBackward('linearized', control='linearized')
