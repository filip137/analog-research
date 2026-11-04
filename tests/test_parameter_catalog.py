from types import SimpleNamespace

import pytest
import torch

from networks.resistive.builders import (
    ModelBundle,
    ParameterBinding,
    ParameterCatalog,
    build_deep_resistive_energy,
)
from networks.resistive.network import DeepResistiveEnergy


def _drn_kwargs():
    return {
        "layer_shapes": [(4,), (3,), (2,)],
        "weight_gains": [0.2, 0.3],
        "input_gain": 1.0,
        "non_linearity": "linear",
        "exponential_diode_param": {},
        "quadratic_diode_param": {},
        "hard_sigmoid_param": {},
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "weight_min": 0.0,
        "weight_max": 1.0,
    }


def test_inferred_catalog_has_stable_keys_and_explicit_views():
    bundle = build_deep_resistive_energy(**_drn_kwargs())

    assert isinstance(bundle.energy, DeepResistiveEnergy)
    assert [binding.key for binding in bundle.parameters.all] == [
        "base.dense_weight.0",
        "base.dense_weight.1",
        "base.bias.0",
    ]
    assert bundle.parameters.trainable == bundle.parameters.all
    assert bundle.parameters.checkpointed == bundle.parameters.all
    assert bundle.parameters.all_parameters == tuple(bundle.energy._all_params)
    assert bundle.parameters.trainable_parameters == tuple(
        bundle.energy.params()
    )
    assert bundle.parameters.checkpointed_parameters == tuple(
        bundle.energy._params
    )
    assert bundle.parameters.by_key["base.bias.0"].role == "bias"


def test_builder_wrap_does_not_change_initial_parameter_values():
    torch.manual_seed(17)
    direct = DeepResistiveEnergy(**_drn_kwargs())
    direct_states = [
        parameter.state.detach().clone() for parameter in direct._all_params
    ]

    torch.manual_seed(17)
    wrapped = build_deep_resistive_energy(**_drn_kwargs())

    for expected, actual in zip(
        direct_states,
        wrapped.parameters.all_parameters,
    ):
        torch.testing.assert_close(actual.state, expected, rtol=0.0, atol=0.0)


def test_passive_low_rank_catalog_has_stable_groups_roles_and_views():
    kwargs = _drn_kwargs()
    kwargs.update(
        {
            "layer_shapes": [(4,), (2,)],
            "weight_gains": [0.2],
            "passive_low_rank_adapter": {
                "rank": 3,
                "input_factor_gain": 0.1,
                "input_factor_min": 1e-7,
                "conductance_max": 1.0,
                "output_factor_init": "zero",
            },
        }
    )

    bundle = build_deep_resistive_energy(**kwargs)

    assert [binding.key for binding in bundle.catalog.all] == [
        "base.dense_weight.0",
        "adapter.input_factor.0",
        "adapter.output_factor.0",
    ]
    assert [
        (binding.group, binding.role)
        for binding in bundle.catalog.all
    ] == [
        ("base", "dense_weight"),
        ("adapter", "input_factor"),
        ("adapter", "output_factor"),
    ]
    assert [binding.key for binding in bundle.catalog.trainable] == [
        "adapter.input_factor.0",
        "adapter.output_factor.0",
    ]
    assert bundle.catalog.checkpointed == bundle.catalog.all
    assert bundle.catalog.all_parameters == tuple(bundle.energy._params)
    assert bundle.catalog.trainable_parameters == tuple(
        bundle.energy.adapter_params()
    )


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
    model = SimpleNamespace(
        _all_params=[base, adapter, frozen],
        _params=[base, adapter],
        params=lambda: [adapter],
    )

    bundle = ModelBundle.wrap(model, catalog=catalog)

    assert [binding.key for binding in bundle.parameters.trainable] == [
        "adapter.factor.0"
    ]
    assert [binding.key for binding in bundle.parameters.checkpointed] == [
        "base.dense_weight.0",
        "adapter.factor.0",
    ]
    assert [
        binding.key
        for binding in bundle.parameters.for_group(
            "base", checkpointed_only=True
        )
    ] == ["base.dense_weight.0"]


def test_catalog_rejects_duplicate_keys_parameters_and_incomplete_bundle():
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

    model = SimpleNamespace(
        _all_params=[first, second],
        _params=[first, second],
        params=lambda: [first, second],
    )
    incomplete = ParameterCatalog(
        [ParameterBinding("base.value.0", first)]
    )
    with pytest.raises(ValueError, match="every model parameter"):
        ModelBundle.wrap(model, catalog=incomplete)
