from __future__ import annotations

import math

import pytest
import torch

from networks.resistive.interaction import DenseResistive, SignedDenseResistive
from networks.variable.layer import LinearLayer
from networks.variable.parameter import DenseWeight


DTYPE = torch.float64


def _weight(value: torch.Tensor) -> DenseWeight:
    result = DenseWeight(
        (value.shape[0],),
        (value.shape[1],),
        gain=0.0,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=1.0,
    )
    result.state = value.detach().clone()
    return result


def _layers() -> tuple[LinearLayer, LinearLayer]:
    pre = LinearLayer((2,), batch_size=3, device="cpu")
    post = LinearLayer((2,), batch_size=3, device="cpu")
    pre.state = torch.tensor(
        [[0.2, -0.4], [0.1, 0.3], [-0.5, 0.6]],
        dtype=DTYPE,
    )
    post.state = torch.tensor(
        [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.5]],
        dtype=DTYPE,
    )
    return pre, post


def _signed(
    voltage_amp: float,
    current_amp: float,
    logical_pre_index: int,
) -> tuple[SignedDenseResistive, LinearLayer, LinearLayer, DenseWeight, DenseWeight]:
    pre, post = _layers()
    plus = _weight(
        torch.tensor([[0.7, 0.2], [0.4, 0.9]], dtype=DTYPE)
    )
    minus = _weight(
        torch.tensor([[0.1, 0.3], [0.2, 0.5]], dtype=DTYPE)
    )
    interaction = SignedDenseResistive(
        pre,
        post,
        plus,
        minus,
        voltage_amp,
        current_amp,
        logical_pre_index=logical_pre_index,
        logical_post_index=logical_pre_index + 1,
    )
    return interaction, pre, post, plus, minus


@pytest.mark.parametrize(
    ("voltage_amp", "current_amp", "logical_pre_index"),
    [
        (1.0, 1.0, 0),
        (4.0, 0.25, 0),
        (4.0, 0.25, 1),
        (3.0, 2.0, 1),
        (0.5, 2.0, 2),
    ],
)
def test_state_coefficients_are_exact_derivatives_of_eval(
    voltage_amp: float,
    current_amp: float,
    logical_pre_index: int,
) -> None:
    interaction, pre, post, _, _ = _signed(
        voltage_amp,
        current_amp,
        logical_pre_index,
    )
    pre.state.requires_grad_(True)
    post.state.requires_grad_(True)
    expected_pre, expected_post = torch.autograd.grad(
        interaction.eval().sum(),
        (pre.state, post.state),
    )
    actual_pre = interaction.grad_layer_fn(pre)()
    actual_post = interaction.grad_layer_fn(post)()

    torch.testing.assert_close(actual_pre, expected_pre)
    torch.testing.assert_close(actual_post, expected_post)
    torch.testing.assert_close(
        actual_pre,
        2.0 * interaction.a_coef_fn(pre)() * pre.state
        + interaction.b_coef_fn(pre)(),
    )
    torch.testing.assert_close(
        actual_post,
        2.0 * interaction.a_coef_fn(post)() * post.state
        + interaction.b_coef_fn(post)(),
    )


@pytest.mark.parametrize(
    ("voltage_amp", "current_amp", "logical_pre_index"),
    [(4.0, 0.25, 0), (4.0, 0.25, 1), (3.0, 2.0, 1), (0.5, 2.0, 2)],
)
def test_conductance_gradients_are_exact_derivatives_of_eval(
    voltage_amp: float,
    current_amp: float,
    logical_pre_index: int,
) -> None:
    interaction, _, _, plus, minus = _signed(
        voltage_amp,
        current_amp,
        logical_pre_index,
    )
    plus.state.requires_grad_(True)
    minus.state.requires_grad_(True)
    expected_plus, expected_minus = torch.autograd.grad(
        interaction.eval().mean(),
        (plus.state, minus.state),
    )

    torch.testing.assert_close(
        interaction.grad_param_fn(plus)(),
        expected_plus,
    )
    torch.testing.assert_close(
        interaction.grad_param_fn(minus)(),
        expected_minus,
    )


@pytest.mark.parametrize(
    ("voltage_amp", "current_amp"),
    [(1.0, 1.0), (4.0, 0.25), (3.0, 2.0), (0.5, 2.0)],
)
def test_weighted_energy_gradient_recovers_raw_physical_kcl(
    voltage_amp: float,
    current_amp: float,
) -> None:
    interaction, pre, post, plus, minus = _signed(
        voltage_amp,
        current_amp,
        logical_pre_index=1,
    )
    total = plus.state + minus.state
    difference = plus.state - minus.state
    pre_metric = 1.0
    post_metric = current_amp / voltage_amp

    expected_pre_current = (
        voltage_amp
        * current_amp
        * pre.state
        * total.sum(dim=1).unsqueeze(0)
        - current_amp * (post.state @ difference.T)
    )
    expected_post_current = (
        post.state * total.sum(dim=0).unsqueeze(0)
        - voltage_amp * (pre.state @ difference)
    )

    torch.testing.assert_close(
        interaction.grad_layer_fn(pre)() / pre_metric,
        expected_pre_current,
    )
    torch.testing.assert_close(
        interaction.grad_layer_fn(post)() / post_metric,
        expected_post_current,
    )


@pytest.mark.parametrize(
    ("voltage_amp", "current_amp"),
    [(1.0, 1.0), (4.0, 0.25), (3.0, 2.0), (0.5, 2.0)],
)
def test_unity_input_edge_post_gradient_is_first_layer_kcl(
    voltage_amp: float,
    current_amp: float,
) -> None:
    interaction, pre, post, plus, minus = _signed(
        voltage_amp,
        current_amp,
        logical_pre_index=0,
    )
    total = plus.state + minus.state
    difference = plus.state - minus.state
    expected = (
        post.state * total.sum(dim=0).unsqueeze(0)
        - pre.state @ difference
    )
    torch.testing.assert_close(interaction.grad_layer_fn(post)(), expected)


def test_zero_negative_branch_preserves_coordinate_update_targets() -> None:
    pre, post = _layers()
    value = torch.tensor([[0.2, 0.5], [0.3, 0.1]], dtype=DTYPE)
    dense_weight = _weight(value)
    plus = _weight(value)
    minus = _weight(torch.zeros_like(value))
    dense = DenseResistive(
        pre,
        post,
        dense_weight,
        3.0,
        2.0,
        logical_pre_index=1,
        logical_post_index=2,
    )
    signed = SignedDenseResistive(
        pre,
        post,
        plus,
        minus,
        3.0,
        2.0,
        logical_pre_index=1,
        logical_post_index=2,
    )
    for layer in (pre, post):
        dense_target = -dense.b_coef_fn(layer)() / (
            2.0 * dense.a_coef_fn(layer)()
        )
        signed_target = -signed.b_coef_fn(layer)() / (
            2.0 * signed.a_coef_fn(layer)()
        )
        torch.testing.assert_close(signed_target, dense_target)


def test_equal_pair_cancels_cross_coupling_but_not_loading() -> None:
    pre, post = _layers()
    equal = torch.tensor([[0.4, 0.2], [0.3, 0.5]], dtype=DTYPE)
    interaction = SignedDenseResistive(
        pre,
        post,
        _weight(equal),
        _weight(equal),
        3.0,
        2.0,
        logical_pre_index=1,
        logical_post_index=2,
    )
    assert torch.count_nonzero(interaction.b_coef_fn(pre)()) == 0
    assert torch.count_nonzero(interaction.b_coef_fn(post)()) == 0
    assert torch.all(interaction.a_coef_fn(pre)() > 0.0)
    assert torch.all(interaction.a_coef_fn(post)() > 0.0)


def test_signed_branches_must_be_non_negative_clamped_conductances() -> None:
    pre, post = _layers()
    value = torch.full((2, 2), 0.2, dtype=DTYPE)
    unconstrained = DenseWeight(
        (2,),
        (2,),
        gain=0.0,
        device="cpu",
        clamp=False,
    )
    unconstrained.state = value.clone()
    with pytest.raises(ValueError, match="non-negative-clamped"):
        SignedDenseResistive(
            pre,
            post,
            unconstrained,
            _weight(value),
            3.0,
            2.0,
            logical_pre_index=0,
            logical_post_index=1,
        )


@pytest.mark.parametrize(
    ("clamp_min", "clamp_max"),
    [
        (0.0, -1.0),
        (0.5, 0.25),
        (0.0, math.inf),
        (0.0, math.nan),
        (0.0, True),
        (0.0, "1.0"),
    ],
)
def test_signed_branches_reject_invalid_upper_clamp_bounds(
    clamp_min: float,
    clamp_max: object,
) -> None:
    pre, post = _layers()
    value = torch.full((2, 2), 0.2, dtype=DTYPE)
    invalid = _weight(value)
    invalid.min_cond = clamp_min
    invalid.max_cond = clamp_max

    with pytest.raises(ValueError, match="non-negative-clamped"):
        SignedDenseResistive(
            pre,
            post,
            invalid,
            _weight(value),
            3.0,
            2.0,
            logical_pre_index=0,
            logical_post_index=1,
        )


@pytest.mark.parametrize(
    ("voltage_amp", "current_amp"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (math.inf, 1.0),
        (math.nan, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, math.inf),
        (1.0, math.nan),
        (True, 1.0),
        (1.0, True),
        ("3.0", 1.0),
        (1.0, "2.0"),
    ],
)
def test_amplifier_magnitudes_must_be_finite_and_positive(
    voltage_amp: float,
    current_amp: float,
) -> None:
    pre, post = _layers()
    value = torch.full((2, 2), 0.2, dtype=DTYPE)
    with pytest.raises(ValueError, match="finite positive"):
        SignedDenseResistive(
            pre,
            post,
            _weight(value),
            _weight(value),
            voltage_amp,
            current_amp,
            logical_pre_index=0,
            logical_post_index=1,
        )


@pytest.mark.parametrize(
    ("logical_pre_index", "logical_post_index"),
    [(0, 2), (-1, 0), (True, 1), (0.5, 1), (None, None)],
)
def test_logical_indices_must_be_adjacent_non_negative_integers(
    logical_pre_index: object,
    logical_post_index: object,
) -> None:
    pre, post = _layers()
    value = torch.full((2, 2), 0.2, dtype=DTYPE)
    with pytest.raises(ValueError, match="adjacent.*non-negative integers"):
        SignedDenseResistive(
            pre,
            post,
            _weight(value),
            _weight(value),
            3.0,
            2.0,
            logical_pre_index=logical_pre_index,
            logical_post_index=logical_post_index,
        )
