"""Block-local recovery targets for the OPT DRN attention-free benchmark."""

import pytest
import torch

from experiments.opt_cmo_drn.local_targets import MODES, local_forward
from tests.test_opt_cmo_drn import tiny_model


@pytest.fixture
def model():
    model, _ = tiny_model()
    model.requires_grad_(False)
    for p in model.parameter_groups()[0]:
        p.requires_grad_(True)
    return model


def held_view(model, perturbed=()):
    """Clean held state, with selected blocks' conductances perturbed."""
    generator = torch.Generator().manual_seed(4)
    view = []
    for k, block in enumerate(model.blocks):
        pair = tuple(g.detach().clone() for g in block.conductances())
        if k in perturbed:
            pair = tuple((g * (1 + 0.2 * torch.rand(g.shape, generator=generator))) for g in pair)
        view.append(pair)
    return tuple(view)


TOKENS = torch.randint(0, 17, (2, 5), generator=torch.Generator().manual_seed(3))


@pytest.mark.parametrize('mode', MODES)
def test_clean_state_has_zero_local_losses_and_clean_logits(model, mode):
    losses, logits = local_forward(model, TOKENS, held_view(model), mode)
    assert len(losses) == len(model.blocks) - 1
    assert all(float(loss) < 1e-10 for loss in losses)
    # Same T+K solver depth as a training forward on the clean masters.
    assert torch.allclose(logits, model(TOKENS), atol=1e-6)


@pytest.mark.parametrize('mode', MODES)
def test_each_loss_reaches_only_its_own_drn(model, mode):
    view = held_view(model, perturbed=(0, 2, 5))
    masters = model.parameter_groups()[0]
    losses, logits = local_forward(model, TOKENS, view, mode)
    for k, loss in enumerate([*losses, logits.square().mean()]):
        grads = torch.autograd.grad(loss, masters, retain_graph=True, allow_unused=True)
        touched = [g is not None and bool(g.any()) for g in grads]
        # Healthy blocks sit at a zero-gradient minimum of their own loss,
        # except in stream mode, where every block after a fault compensates.
        assert touched[2 * k] == (k in (0, 2, 5) or mode == 'stream')
        assert not any(t for j, t in enumerate(touched) if j // 2 != k)


def test_modes_agree_on_the_first_block_and_split_on_upstream_drift(model):
    # Only block 0 is faulty: every mode sees the same clean input there.
    view = held_view(model, perturbed=(0,))
    first = [float(local_forward(model, TOKENS, view, mode)[0][0]) for mode in MODES]
    assert first[0] > 0 and max(first) - min(first) < 1e-6 * first[0]
    # Block 1 is healthy but receives block 0's error on-policy.
    second = {mode: float(local_forward(model, TOKENS, view, mode)[0][1]) for mode in MODES}
    assert second['cached'] < 1e-10 and second['onpolicy'] < 1e-10
    assert second['stream'] > 1e-6
