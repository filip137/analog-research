from __future__ import annotations

import pytest
import torch

from methods.sign_sgd import SignSGD


def test_sign_sgd_uses_only_gradient_direction() -> None:
    parameter = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    optimizer = SignSGD([{"params": [parameter], "lr": 0.25}], lr=0.0)
    parameter.grad = torch.tensor([1000.0, -1e-12, 0.0])

    optimizer.step()

    torch.testing.assert_close(parameter, torch.tensor([0.75, -1.75, 3.0]))


def test_sign_sgd_preserves_parameter_group_rates_across_state_roundtrip() -> None:
    first = torch.tensor([1.0], requires_grad=True)
    second = torch.tensor([1.0], requires_grad=True)
    optimizer = SignSGD(
        [
            {"params": [first], "lr": 0.1},
            {"params": [second], "lr": 0.01},
        ],
        lr=0.0,
    )
    saved = optimizer.state_dict()
    restored = SignSGD(
        [
            {"params": [first], "lr": 1.0},
            {"params": [second], "lr": 1.0},
        ],
        lr=0.0,
    )
    restored.load_state_dict(saved)
    first.grad = torch.tensor([-4.0])
    second.grad = torch.tensor([7.0])

    restored.step()

    torch.testing.assert_close(first, torch.tensor([1.1]))
    torch.testing.assert_close(second, torch.tensor([0.99]))


@pytest.mark.parametrize(
    "learning_rate",
    [True, -1.0, float("inf"), float("nan")],
)
def test_sign_sgd_rejects_invalid_learning_rates(
    learning_rate: float | bool,
) -> None:
    parameter = torch.tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="finite and non-negative"):
        SignSGD([parameter], lr=learning_rate)
