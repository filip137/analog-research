from __future__ import annotations

import pytest
import torch

from ebl.runtime.mnist_relu_drn.components import (
    collapse_clamped_dual_rail_input,
    signed_differential_lift,
    signed_dual_rail_lift,
)
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
        clamp_max=4.0,
    )
    result.state = value.detach().clone()
    return result


def test_inner_input_ports_are_not_new_independent_rails() -> None:
    logical = torch.tensor(
        [[0.4, -0.2, 0.7], [-0.1, 0.8, 0.3]],
        dtype=DTYPE,
    )
    outer_plus = logical
    outer_minus = -logical

    inner_plus_plus = outer_plus
    inner_plus_minus = -outer_plus
    inner_minus_plus = outer_minus
    inner_minus_minus = -outer_minus

    torch.testing.assert_close(inner_plus_minus, outer_minus)
    torch.testing.assert_close(inner_minus_plus, outer_minus)
    torch.testing.assert_close(inner_minus_minus, outer_plus)
    assert torch.equal(
        torch.cat((inner_plus_plus, inner_plus_minus), dim=1),
        torch.cat((outer_plus, outer_minus), dim=1),
    )


def test_eight_device_input_edge_collapses_to_four_post_facing_edges() -> None:
    generator = torch.Generator().manual_seed(20260819)
    logical_input = torch.randn(
        (5, 3),
        generator=generator,
        dtype=DTYPE,
    )
    physical_input = torch.cat((logical_input, -logical_input), dim=1)
    hidden = torch.randn(
        (5, 4),
        generator=generator,
        dtype=DTYPE,
    )
    plus_value = 0.1 + torch.rand(
        (6, 4),
        generator=generator,
        dtype=DTYPE,
    )
    minus_value = 0.1 + torch.rand(
        (6, 4),
        generator=generator,
        dtype=DTYPE,
    )
    collapsed_value = collapse_clamped_dual_rail_input(
        plus_value,
        minus_value,
    )

    pre = LinearLayer((6,), batch_size=5, device="cpu")
    post = LinearLayer((4,), batch_size=5, device="cpu")
    pre.state = physical_input
    post.state = hidden
    plus = _weight(plus_value)
    minus = _weight(minus_value)
    collapsed = _weight(collapsed_value)
    signed = SignedDenseResistive(
        pre,
        post,
        plus,
        minus,
        1.0,
        1.0,
        logical_pre_index=0,
        logical_post_index=1,
    )
    ordinary = DenseResistive(
        pre,
        post,
        collapsed,
        1.0,
        1.0,
        logical_pre_index=0,
        logical_post_index=1,
    )

    assert plus_value.numel() + minus_value.numel() == (
        2 * collapsed_value.numel()
    )
    torch.testing.assert_close(signed.eval(), ordinary.eval())
    torch.testing.assert_close(
        signed.a_coef_fn(post)(),
        ordinary.a_coef_fn(post)(),
    )
    torch.testing.assert_close(
        signed.b_coef_fn(post)(),
        ordinary.b_coef_fn(post)(),
    )
    torch.testing.assert_close(
        signed.grad_layer_fn(post)(),
        ordinary.grad_layer_fn(post)(),
    )

    signed_pre = signed.grad_layer_fn(pre)()
    ordinary_pre = ordinary.grad_layer_fn(pre)()
    torch.testing.assert_close(
        signed_pre[:, :3] - signed_pre[:, 3:],
        ordinary_pre[:, :3] - ordinary_pre[:, 3:],
    )
    assert not torch.allclose(signed_pre, ordinary_pre)


def test_collapse_is_not_valid_for_independent_dynamic_source_rails() -> None:
    plus_value = torch.tensor(
        [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]],
        dtype=DTYPE,
    )
    minus_value = torch.tensor(
        [[0.1, 0.5], [0.6, 0.2], [0.7, 0.3], [0.2, 0.8]],
        dtype=DTYPE,
    )
    independent_pre = torch.tensor(
        [[0.2, -0.4, 0.7, 0.1]],
        dtype=DTYPE,
    )
    post_state = torch.tensor([[0.3, -0.2]], dtype=DTYPE)

    pre = LinearLayer((4,), batch_size=1, device="cpu")
    post = LinearLayer((2,), batch_size=1, device="cpu")
    pre.state = independent_pre
    post.state = post_state
    signed = SignedDenseResistive(
        pre,
        post,
        _weight(plus_value),
        _weight(minus_value),
        1.0,
        1.0,
        logical_pre_index=0,
        logical_post_index=1,
    )
    ordinary = DenseResistive(
        pre,
        post,
        _weight(
            collapse_clamped_dual_rail_input(plus_value, minus_value)
        ),
        1.0,
        1.0,
        logical_pre_index=0,
        logical_post_index=1,
    )

    assert not torch.allclose(signed.eval(), ordinary.eval())
    assert not torch.allclose(
        signed.grad_layer_fn(post)(),
        ordinary.grad_layer_fn(post)(),
    )


@pytest.mark.parametrize("layout", ["halves", "paired"])
def test_teacher_mapped_pair_collapses_to_dual_rail_lift(
    layout: str,
) -> None:
    logical_weight = torch.tensor(
        [[0.8, -0.4], [-0.3, 1.0], [0.2, -0.7]],
        dtype=DTYPE,
    )
    baseline = 0.07
    scale = 0.4
    differential = signed_differential_lift(
        logical_weight,
        target_layout=layout,
    )
    plus = baseline + scale * differential.clamp_min(0.0)
    minus = baseline + scale * (-differential).clamp_min(0.0)

    collapsed = collapse_clamped_dual_rail_input(plus, minus)
    expected = (
        2.0 * baseline
        + scale
        * signed_dual_rail_lift(
            logical_weight,
            target_layout=layout,
        )
    )
    torch.testing.assert_close(collapsed, expected)


def test_collapse_rejects_non_physical_or_mismatched_branches() -> None:
    plus = torch.ones((4, 2), dtype=DTYPE)
    with pytest.raises(ValueError, match=r"Expected .*Provided value"):
        collapse_clamped_dual_rail_input(plus, -plus)
    with pytest.raises(ValueError, match=r"Expected .*Provided value"):
        collapse_clamped_dual_rail_input(plus[:3], plus[:3])
    with pytest.raises(ValueError, match=r"Expected .*Provided value"):
        collapse_clamped_dual_rail_input(plus, plus.to(torch.float32))
