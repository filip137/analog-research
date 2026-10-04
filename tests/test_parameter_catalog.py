from types import SimpleNamespace

import pytest
import torch
from training.parameters import DenseWeight, ParameterBinding, ParameterCatalog


def test_explicit_catalog_supports_extension_groups_and_frozen_parameters():
    base = SimpleNamespace(state=torch.ones(2, 2))
    adapter = SimpleNamespace(state=torch.zeros(2, 1))
    frozen = SimpleNamespace(state=torch.full((1,), 0.5))
    catalog = ParameterCatalog(
        [
            ParameterBinding(
                "base.dense_weight.0",
                base,
                group="base",
                role="dense_weight",
                trainable=False,
            ),
            ParameterBinding(
                "adapter.factor.0",
                adapter,
                group="adapter",
                role="factor",
                trainable=True,
            ),
            ParameterBinding(
                "base.pool_weight.0",
                frozen,
                group="base",
                role="pool_weight",
                trainable=False,
                checkpointed=False,
            ),
        ]
    )
    assert [binding.key for binding in catalog.trainable] == [
        "adapter.factor.0"
    ]
    assert [binding.key for binding in catalog.checkpointed] == [
        "base.dense_weight.0",
        "adapter.factor.0",
    ]
    assert [
        binding.key
        for binding in catalog.for_group(
            "base", checkpointed_only=True
        )
    ] == ["base.dense_weight.0"]


def test_catalog_rejects_duplicate_keys_parameters_():
    first = SimpleNamespace(state=torch.ones(1))
    second = SimpleNamespace(state=torch.zeros(1))

    with pytest.raises(ValueError, match="duplicate keys"):
        ParameterCatalog(
            [
                ParameterBinding("base.value.0", first),
                ParameterBinding("base.value.0", second),
            ]
        )
    with pytest.raises(ValueError, match="exactly one binding"):
        ParameterCatalog(
            [
                ParameterBinding("base.value.0", first),
                ParameterBinding("base.value.1", first),
            ]
        )
