from __future__ import annotations

import pytest
import torch

from ebl.runtime.mnist_relu_drn.ibm_om_raw_active_initialization import (
    transform_to_raw_active_coordinate,
)


def test_raw_active_initialization_uses_one_array_wide_affine_transform() -> None:
    lower = 0.1020408197973068
    upper = 1.0
    span = upper - lower
    weights = {
        "base.dense_weight.0": lower
        + span * torch.tensor([[0.0, 0.25], [0.5, 1.0]], dtype=torch.float32),
        "base.dense_weight.1": lower
        + span * torch.tensor([[1.0, 0.75], [0.5, 0.0]], dtype=torch.float32),
    }

    transformed, reports = transform_to_raw_active_coordinate(weights)

    torch.testing.assert_close(
        transformed["base.dense_weight.0"],
        torch.tensor([[0.0, 0.25], [0.5, 1.0]], dtype=torch.float32),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        transformed["base.dense_weight.1"],
        torch.tensor([[1.0, 0.75], [0.5, 0.0]], dtype=torch.float32),
        atol=1e-7,
        rtol=0.0,
    )
    assert [report["key"] for report in reports] == list(weights)
    assert all(report["maximum_clamp_correction"] == 0.0 for report in reports)


def test_raw_active_initialization_rejects_material_clipping() -> None:
    lower = 0.1020408197973068
    weights = {
        "base.dense_weight.0": torch.tensor([[lower - 0.01]], dtype=torch.float32),
        "base.dense_weight.1": torch.tensor([[1.0]], dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="without material clipping"):
        transform_to_raw_active_coordinate(weights)
